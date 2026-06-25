from __future__ import annotations

import json
import shutil
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MASS_DIR = ROOT / "selected_model_workbench" / "mass_router_meta_analysis"
OUT_DIR = ROOT / "selected_model_workbench" / "hard_case_adapter_experiment"


FEATURE_COLS = [
    "p_chex_frozen",
    "p_chex_head",
    "p_chex_lora1",
    "p_chex_lora2",
    "p_last1",
    "p_last2",
    "p_eva_min",
    "p_eva_max",
    "p_eva_mean",
    "p_chex_safe_min",
    "p_chex_safe_max",
    "p_chex_all_min",
    "p_chex_all_max",
    "p_all_core_min",
    "p_all_core_max",
    "p_all_with_lora2_max",
    "p_chex_head_last1_max",
    "p_chex_frozen_last1_max",
    "p_chex_lora1_last1_max",
    "p_chex_head_last1_mean",
    "p_chex_frozen_last1_mean",
    "uncertainty_core_max",
    "ood_score_chex",
    "ood_score_eva",
]


def bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    return series.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy(bool)


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(MASS_DIR / f"input_scores_{split}.csv", keep_default_na=False)
    df["critical_qa_bool"] = bool_array(df["critical_qa_bool"] if "critical_qa_bool" in df.columns else df["critical_qa"])
    return df


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def wilson_low(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return float((center - margin) / denom)


def metrics_for_mask(y: np.ndarray, mask: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    selected = int(mask.sum())
    tn = int(((y == 0) & mask).sum())
    fn = int(((y == 1) & mask).sum())
    return {
        "n": int(len(y)),
        "selected_count": selected,
        "auto_negative_coverage": float(selected / max(len(y), 1)),
        "TN_count": tn,
        "FN_count": fn,
        "NPV": float(tn / max(tn + fn, 1)),
        "NPV_ci95_low": wilson_low(tn, tn + fn),
        "FN_per_1000_selected": float(fn / max(selected, 1) * 1000.0),
        "auroc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
    }


def make_features(val: pd.DataFrame, final: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    cols = [c for c in FEATURE_COLS if c in val.columns and c in final.columns]
    Xv = val[cols].to_numpy(float)
    Xf = final[cols].to_numpy(float)
    keep = np.isfinite(Xv).all(axis=0) & np.isfinite(Xf).all(axis=0) & (np.nanstd(Xv, axis=0) > 1e-8)
    cols = [c for c, k in zip(cols, keep) if bool(k)]
    return cols, Xv[:, keep], Xf[:, keep]


def candidate_models() -> list[tuple[str, Any]]:
    return [
        (
            "logistic_l2_balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=0.2, class_weight="balanced", solver="liblinear", random_state=42),
            ),
        ),
        (
            "logistic_l1_sparse",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=0.05, penalty="l1", class_weight="balanced", solver="liblinear", random_state=42),
            ),
        ),
        (
            "gradient_boosting_shallow",
            GradientBoostingClassifier(n_estimators=80, learning_rate=0.035, max_depth=2, subsample=0.8, random_state=42),
        ),
        (
            "extra_trees_shallow",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=25,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest_shallow",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=25,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]


def router_grid_for_scores(df: pd.DataFrame, p: np.ndarray, y: np.ndarray, split: str) -> pd.DataFrame:
    rows = []
    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0
    thresholds = np.unique(np.quantile(p, np.linspace(0.002, 0.24, 160)))
    for t, t_ood_chex, t_ood_eva, t_quality, t_unc in product(
        thresholds,
        [1.10, 1.25, 1.50],
        [1.10, 1.25, 1.50],
        [0.25, 0.35],
        [0.50, 0.65, 0.80],
    ):
        mask = (
            (df["quality_score"].to_numpy(float) >= t_quality)
            & (~df["critical_qa_bool"].to_numpy(bool))
            & (df["ood_score_chex"].to_numpy(float) <= t_ood_chex)
            & (df["ood_score_eva"].to_numpy(float) <= t_ood_eva)
            & (uncertainty <= t_unc)
            & (p <= float(t))
        )
        row = metrics_for_mask(y, mask, p)
        row.update(
            {
                "split": split,
                "t_negative": float(t),
                "t_ood_chex": t_ood_chex,
                "t_ood_eva": t_ood_eva,
                "t_quality": t_quality,
                "t_uncertainty": t_unc,
                "safe_validation_candidate": bool(row["selected_count"] >= 10 and row["FN_count"] == 0 and row["NPV"] >= 0.99),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def apply_rule(df: pd.DataFrame, p: np.ndarray, y: np.ndarray, rule: pd.Series) -> dict[str, Any]:
    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0
    mask = (
        (df["quality_score"].to_numpy(float) >= float(rule["t_quality"]))
        & (~df["critical_qa_bool"].to_numpy(bool))
        & (df["ood_score_chex"].to_numpy(float) <= float(rule["t_ood_chex"]))
        & (df["ood_score_eva"].to_numpy(float) <= float(rule["t_ood_eva"]))
        & (uncertainty <= float(rule["t_uncertainty"]))
        & (p <= float(rule["t_negative"]))
    )
    out = metrics_for_mask(y, mask, p)
    for col in ["t_negative", "t_ood_chex", "t_ood_eva", "t_quality", "t_uncertainty"]:
        out[col] = float(rule[col])
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val = load_split("validation")
    final = load_split("final_test")
    yv = val["y_attention"].to_numpy(int)
    yf = final["y_attention"].to_numpy(int)
    feature_cols, Xv, Xf = make_features(val, final)

    results = []
    final_results = []
    for name, model in candidate_models():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xv, yv)
            pv = model.predict_proba(Xv)[:, 1]
            pf = model.predict_proba(Xf)[:, 1]
        val_scores = val[["study_id", "image_file", "split", "y_attention"]].copy()
        val_scores["p_adapter"] = pv
        final_scores = final[["study_id", "image_file", "split", "y_attention"]].copy()
        final_scores["p_adapter"] = pf
        val_scores.to_csv(OUT_DIR / f"{name}_scores_validation.csv", index=False)
        final_scores.to_csv(OUT_DIR / f"{name}_scores_final_test.csv", index=False)

        grid = router_grid_for_scores(val, pv, yv, "validation")
        grid["adapter_model"] = name
        grid.to_csv(OUT_DIR / f"{name}_validation_router_grid.csv", index=False)
        safe = grid[grid["safe_validation_candidate"]].copy().sort_values(["auto_negative_coverage", "NPV_ci95_low"], ascending=[False, False])
        if safe.empty:
            continue
        best = safe.iloc[0]
        final_eval = apply_rule(final, pf, yf, best)
        val_eval = best.to_dict()
        val_eval["adapter_model"] = name
        final_eval["adapter_model"] = name
        final_eval["validation_auto_negative_coverage"] = float(best["auto_negative_coverage"])
        final_eval["validation_selected_count"] = int(best["selected_count"])
        final_eval["validation_FN_count"] = int(best["FN_count"])
        final_eval["validation_NPV"] = float(best["NPV"])
        final_eval["final_safe"] = bool(final_eval["FN_count"] == 0 and final_eval["NPV"] >= 0.99)
        results.append(val_eval)
        final_results.append(final_eval)
        joblib.dump({"model": model, "feature_cols": feature_cols}, OUT_DIR / f"{name}.pkl")

    validation_df = pd.DataFrame(results).sort_values(["auto_negative_coverage", "NPV_ci95_low"], ascending=[False, False])
    final_df = pd.DataFrame(final_results).sort_values(["final_safe", "auto_negative_coverage", "NPV_ci95_low"], ascending=[False, False, False])
    validation_df.to_csv(OUT_DIR / "adapter_validation_selected_rules.csv", index=False)
    final_df.to_csv(OUT_DIR / "adapter_fixed_final_results.csv", index=False)

    report = [
        "# Hard-Case Adapter Experiment",
        "",
        "Research-only experiment: adapter/meta-head is trained on validation score features and checked on final. This is not a production candidate without a fresh clean split.",
        "",
        "## Feature Columns",
        "",
        ", ".join(feature_cols),
        "",
        "## Validation-Selected Adapter Rules",
        "",
        validation_df.head(10).to_markdown(index=False),
        "",
        "## Fixed Final Results",
        "",
        final_df.head(10).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "If adapter final-safe coverage is not higher than the rule-based ensemble, then more posthoc score mixing is unlikely to solve automation. We need label review and/or true model adaptation.",
    ]
    (OUT_DIR / "hard_case_adapter_experiment_report.md").write_text("\n".join(report), encoding="utf-8")

    compact = OUT_DIR / "export_compact"
    if compact.exists():
        shutil.rmtree(compact)
    compact.mkdir(parents=True)
    for name in [
        "adapter_validation_selected_rules.csv",
        "adapter_fixed_final_results.csv",
        "hard_case_adapter_experiment_report.md",
    ]:
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, compact / name)
    archive_base = OUT_DIR / "hard_case_adapter_experiment_export"
    if archive_base.with_suffix(".zip").exists():
        archive_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(archive_base), "zip", root_dir=compact)
    print("Saved:", OUT_DIR)
    print("Archive:", archive_base.with_suffix(".zip"))
    print(final_df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
