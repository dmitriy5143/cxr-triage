from __future__ import annotations

import json
import math
import shutil
import sys
import warnings
from itertools import combinations, product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluoro_mvp_core import expected_calibration_error, wilson_lower_bound  # noqa: E402


SOURCE_DIR = ROOT / "selected_model_workbench" / "deep_router_with_chexfound_lora"
OUT_DIR = ROOT / "selected_model_workbench" / "mass_router_meta_analysis"


RAW_SCORE_COLS = [
    "p_chex_frozen",
    "p_chex_head",
    "p_chex_lora1",
    "p_chex_lora2",
    "p_last1",
    "p_last2",
]


KEY_PAIRS = [
    ("p_chex_frozen", "p_last1"),
    ("p_chex_head", "p_last1"),
    ("p_chex_lora1", "p_last1"),
    ("p_chex_lora2", "p_last1"),
    ("p_chex_head", "p_last2"),
    ("p_chex_lora1", "p_last2"),
    ("p_chex_lora2", "p_last2"),
    ("p_last1", "p_last2"),
    ("p_chex_head", "p_chex_lora1"),
    ("p_chex_frozen", "p_chex_head"),
    ("p_chex_head", "p_eva_min"),
    ("p_chex_lora1", "p_eva_min"),
    ("p_chex_frozen", "p_eva_min"),
]


GROUPS = {
    "eva_pair": ["p_last1", "p_last2"],
    "chex_head_lora1": ["p_chex_head", "p_chex_lora1"],
    "chex_frozen_head": ["p_chex_frozen", "p_chex_head"],
    "chex_head_eva_last1": ["p_chex_head", "p_last1"],
    "chex_frozen_eva_last1": ["p_chex_frozen", "p_last1"],
    "chex_lora1_eva_last1": ["p_chex_lora1", "p_last1"],
    "chex_lora2_eva_last1": ["p_chex_lora2", "p_last1"],
    "core3_head_lora1_last1": ["p_chex_head", "p_chex_lora1", "p_last1"],
    "core3_frozen_head_last1": ["p_chex_frozen", "p_chex_head", "p_last1"],
    "core4_head_lora1_last1_last2": ["p_chex_head", "p_chex_lora1", "p_last1", "p_last2"],
    "core5_no_lora2": ["p_chex_frozen", "p_chex_head", "p_chex_lora1", "p_last1", "p_last2"],
}


def bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    return series.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy(bool)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def add_derived_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_eva_mean"] = out[["p_last1", "p_last2"]].mean(axis=1)
    out["p_chex_all_min"] = out[["p_chex_frozen", "p_chex_head", "p_chex_lora1", "p_chex_lora2"]].min(axis=1)
    out["p_chex_all_max"] = out[["p_chex_frozen", "p_chex_head", "p_chex_lora1", "p_chex_lora2"]].max(axis=1)
    out["p_chex_head_last1_max"] = out[["p_chex_head", "p_last1"]].max(axis=1)
    out["p_chex_frozen_last1_max"] = out[["p_chex_frozen", "p_last1"]].max(axis=1)
    out["p_chex_lora1_last1_max"] = out[["p_chex_lora1", "p_last1"]].max(axis=1)
    out["p_chex_head_last1_mean"] = out[["p_chex_head", "p_last1"]].mean(axis=1)
    out["p_chex_frozen_last1_mean"] = out[["p_chex_frozen", "p_last1"]].mean(axis=1)
    for col in RAW_SCORE_COLS:
        out[f"uncertainty_{col}"] = 1.0 - np.abs(out[col].to_numpy(float) - 0.5) * 2.0
    out["uncertainty_core_max"] = out[[f"uncertainty_{c}" for c in ["p_chex_head", "p_chex_lora1", "p_last1", "p_last2"]]].max(axis=1)
    out["critical_qa_bool"] = bool_array(out["critical_qa_bool"] if "critical_qa_bool" in out.columns else out["critical_qa"])
    return out.reset_index(drop=True)


def load_split(split: str) -> pd.DataFrame:
    path = SOURCE_DIR / f"merged_scores_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, keep_default_na=False)
    return add_derived_scores(df)


def threshold_values(df: pd.DataFrame, col: str, *, low: float = 0.002, high: float = 0.18, n: int = 12) -> np.ndarray:
    values = np.quantile(df[col].to_numpy(float), np.linspace(low, high, n))
    values = np.unique(values[np.isfinite(values)])
    return values


def seeded_thresholds(df: pd.DataFrame, col: str, seeds: list[float] | None = None, n: int = 12) -> np.ndarray:
    vals = threshold_values(df, col, n=n)
    if seeds:
        vals = np.unique(np.concatenate([vals, np.asarray(seeds, dtype=float)]))
    vals = vals[np.isfinite(vals)]
    return vals


def base_mask(
    df: pd.DataFrame,
    score_cols: list[str],
    *,
    t_ood_chex: float,
    t_ood_eva: float,
    t_quality: float,
    t_uncertainty: float,
) -> np.ndarray:
    uncertainties = np.maximum.reduce([1.0 - np.abs(df[c].to_numpy(float) - 0.5) * 2.0 for c in score_cols])
    return (
        (df["quality_score"].to_numpy(float) >= t_quality)
        & (~df["critical_qa_bool"].to_numpy(bool))
        & (df["ood_score_chex"].to_numpy(float) <= t_ood_chex)
        & (df["ood_score_eva"].to_numpy(float) <= t_ood_eva)
        & (uncertainties <= t_uncertainty)
    )


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def metrics_for_mask(df: pd.DataFrame, mask: np.ndarray, score_col: str | None = None) -> dict[str, Any]:
    y = df["y_attention"].to_numpy(int)
    mask = np.asarray(mask, dtype=bool)
    selected = int(mask.sum())
    tn = int(((y == 0) & mask).sum())
    fn = int(((y == 1) & mask).sum())
    out: dict[str, Any] = {
        "n": int(len(df)),
        "selected_count": selected,
        "auto_negative_coverage": float(selected / max(len(df), 1)),
        "TN_count": tn,
        "FN_count": fn,
        "NPV": float(tn / max(tn + fn, 1)),
        "NPV_ci95_low": wilson_lower_bound(tn, tn + fn, z=1.96),
        "FN_per_1000_selected": float(fn / max(selected, 1) * 1000.0),
    }
    if score_col and score_col in df.columns:
        out.update(score_metrics(y, df[score_col].to_numpy(float)))
    return out


def add_row(rows: list[dict[str, Any]], df: pd.DataFrame, mask: np.ndarray, **params: Any) -> None:
    row = metrics_for_mask(df, mask)
    row.update(params)
    row["safe_validation_candidate"] = bool(row["selected_count"] >= 10 and row["FN_count"] == 0 and row["NPV"] >= 0.99)
    row["robust_validation_candidate"] = bool(row["safe_validation_candidate"] and row["NPV_ci95_low"] >= 0.96)
    rows.append(row)


def single_score_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_cols = RAW_SCORE_COLS + [
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
    ]
    for col in score_cols:
        if col not in df.columns:
            continue
        for t, t_ood_chex, t_ood_eva, t_quality, t_unc in product(
            seeded_thresholds(df, col, n=36),
            [1.10, 1.50],
            [1.10, 1.50],
            [0.25, 0.35],
            [0.50, 0.65, 0.80],
        ):
            mask = base_mask(
                df,
                [col],
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=t_quality,
                t_uncertainty=t_unc,
            ) & (df[col].to_numpy(float) <= float(t))
            add_row(
                rows,
                df,
                mask,
                rule="single_score_threshold",
                score_col=col,
                risk_score_col=col,
                t_negative=float(t),
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=t_quality,
                t_uncertainty=t_unc,
            )
    return pd.DataFrame(rows)


def pair_veto_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixed_seeds: dict[str, list[float]] = {
        "p_chex_frozen": [0.023979, 0.08],
        "p_chex_head": [0.028991675994444445, 0.20],
        "p_last1": [0.0055976255760630846, 0.012314190939068733, 0.08, 0.12],
        "p_last2": [0.045596, 0.08, 0.12],
        "p_chex_lora1": [0.05512125956819055, 0.06, 0.08],
    }
    veto_values = [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.30]
    for a, b in KEY_PAIRS:
        if a not in df.columns or b not in df.columns:
            continue
        ta_vals = seeded_thresholds(df, a, fixed_seeds.get(a), n=10)
        tb_vals = seeded_thresholds(df, b, fixed_seeds.get(b), n=10)
        va_vals = sorted(set(veto_values + fixed_seeds.get(a, [])))
        vb_vals = sorted(set(veto_values + fixed_seeds.get(b, [])))
        for ta, tb, va, vb, t_ood_chex, t_ood_eva, t_quality, t_unc in product(
            ta_vals,
            tb_vals,
            va_vals,
            vb_vals,
            [1.10, 1.50],
            [1.25, 1.50],
            [0.25, 0.35],
            [0.50, 0.80],
        ):
            qmask = base_mask(
                df,
                [a, b],
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=t_quality,
                t_uncertainty=t_unc,
            )
            a_vals = df[a].to_numpy(float)
            b_vals = df[b].to_numpy(float)
            mask = qmask & (((a_vals <= float(ta)) & (b_vals <= float(vb))) | ((b_vals <= float(tb)) & (a_vals <= float(va))))
            add_row(
                rows,
                df,
                mask,
                rule="pair_one_low_other_veto",
                model_a=a,
                model_b=b,
                risk_score_col="p_all_core_max",
                t_a_negative=float(ta),
                t_b_negative=float(tb),
                t_a_veto=float(va),
                t_b_veto=float(vb),
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=t_quality,
                t_uncertainty=t_unc,
            )
    return pd.DataFrame(rows)


def group_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, cols in GROUPS.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            continue
        low_score = df[cols].min(axis=1).to_numpy(float)
        low_vals = np.unique(np.quantile(low_score, np.linspace(0.002, 0.22, 22)))
        veto_vals = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.30]
        k_options = sorted(set([1, min(2, len(cols)), len(cols)]))
        for t_low, t_veto, k_low, t_ood_chex, t_ood_eva, t_quality, t_unc in product(
            low_vals,
            veto_vals,
            k_options,
            [1.10, 1.50],
            [1.25, 1.50],
            [0.25, 0.35],
            [0.50, 0.80],
        ):
            qmask = base_mask(
                df,
                cols,
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=t_quality,
                t_uncertainty=t_unc,
            )
            values = df[cols].to_numpy(float)
            low_count = (values <= float(t_low)).sum(axis=1)
            max_score = values.max(axis=1)
            mask = qmask & (low_count >= int(k_low)) & (max_score <= float(t_veto))
            add_row(
                rows,
                df,
                mask,
                rule="group_count_low_with_max_veto",
                group=group_name,
                score_members="|".join(cols),
                risk_score_col="p_all_core_max",
                t_group_low=float(t_low),
                t_group_veto=float(t_veto),
                k_low=int(k_low),
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=t_quality,
                t_uncertainty=t_unc,
            )
    return pd.DataFrame(rows)


def build_validation_grid(df: pd.DataFrame) -> pd.DataFrame:
    parts = [single_score_grid(df), pair_veto_grid(df), group_grid(df)]
    grid = pd.concat(parts, ignore_index=True)
    grid["validation_rank_all"] = np.arange(len(grid)) + 1
    return grid.sort_values(
        ["safe_validation_candidate", "robust_validation_candidate", "auto_negative_coverage", "NPV_ci95_low"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def mask_for_rule(df: pd.DataFrame, rule: pd.Series | dict[str, Any]) -> np.ndarray:
    r = dict(rule)
    rule_name = str(r["rule"])
    if rule_name == "single_score_threshold":
        col = str(r["score_col"])
        return base_mask(
            df,
            [col],
            t_ood_chex=float(r["t_ood_chex"]),
            t_ood_eva=float(r["t_ood_eva"]),
            t_quality=float(r["t_quality"]),
            t_uncertainty=float(r["t_uncertainty"]),
        ) & (df[col].to_numpy(float) <= float(r["t_negative"]))
    if rule_name == "pair_one_low_other_veto":
        a = str(r["model_a"])
        b = str(r["model_b"])
        qmask = base_mask(
            df,
            [a, b],
            t_ood_chex=float(r["t_ood_chex"]),
            t_ood_eva=float(r["t_ood_eva"]),
            t_quality=float(r["t_quality"]),
            t_uncertainty=float(r["t_uncertainty"]),
        )
        av = df[a].to_numpy(float)
        bv = df[b].to_numpy(float)
        return qmask & (
            ((av <= float(r["t_a_negative"])) & (bv <= float(r["t_b_veto"])))
            | ((bv <= float(r["t_b_negative"])) & (av <= float(r["t_a_veto"])))
        )
    if rule_name == "group_count_low_with_max_veto":
        cols = str(r["score_members"]).split("|")
        qmask = base_mask(
            df,
            cols,
            t_ood_chex=float(r["t_ood_chex"]),
            t_ood_eva=float(r["t_ood_eva"]),
            t_quality=float(r["t_quality"]),
            t_uncertainty=float(r["t_uncertainty"]),
        )
        values = df[cols].to_numpy(float)
        return qmask & ((values <= float(r["t_group_low"])).sum(axis=1) >= int(float(r["k_low"]))) & (
            values.max(axis=1) <= float(r["t_group_veto"])
        )
    raise ValueError(f"Unknown rule: {rule_name}")


def eval_rules_on_split(rules: pd.DataFrame, df: pd.DataFrame, *, max_rules: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if max_rules is not None:
        rules = rules.head(max_rules)
    for rank, (_, rule) in enumerate(rules.iterrows(), start=1):
        mask = mask_for_rule(df, rule)
        row = metrics_for_mask(df, mask)
        for col in rule.index:
            if col.startswith("t_") or col in {
                "rule",
                "score_col",
                "model_a",
                "model_b",
                "group",
                "score_members",
                "k_low",
                "risk_score_col",
                "validation_rank_all",
                "safe_validation_candidate",
                "robust_validation_candidate",
            }:
                row[col] = rule[col]
        row["validation_rank"] = rank
        row["final_zero_fn"] = bool(row["FN_count"] == 0)
        row["final_safe"] = bool(row["FN_count"] == 0 and row["NPV"] >= 0.99)
        rows.append(row)
    return pd.DataFrame(rows)


def train_research_meta_classifier(val: pd.DataFrame, final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = RAW_SCORE_COLS + [
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
        "quality_score",
        "ood_score_chex",
        "ood_score_eva",
        "uncertainty_core_max",
    ]
    feature_cols = [c for c in feature_cols if c in val.columns and c in final.columns]
    X_val = val[feature_cols].to_numpy(float)
    y_val = val["y_attention"].to_numpy(int)
    X_final = final[feature_cols].to_numpy(float)
    y_final = final["y_attention"].to_numpy(int)
    finite_keep = np.isfinite(X_val).all(axis=0) & np.isfinite(X_final).all(axis=0) & (np.nanstd(X_val, axis=0) > 1e-8)
    feature_cols = [c for c, keep in zip(feature_cols, finite_keep) if bool(keep)]
    X_val = X_val[:, finite_keep]
    X_final = X_final[:, finite_keep]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=42, solver="liblinear"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X_val, y_val)
        p_val = model.predict_proba(X_val)[:, 1]
        p_final = model.predict_proba(X_final)[:, 1]

    tmp_val = val.copy()
    tmp_final = final.copy()
    tmp_val["p_research_meta_lr"] = p_val
    tmp_final["p_research_meta_lr"] = p_final

    thresholds = np.unique(np.quantile(p_val, np.linspace(0.002, 0.20, 120)))
    rows = []
    for t, t_quality, t_unc in product(thresholds, [0.25, 0.35], [0.50, 0.65, 0.80]):
        mask = (
            (tmp_val["quality_score"].to_numpy(float) >= t_quality)
            & (~tmp_val["critical_qa_bool"].to_numpy(bool))
            & ((1.0 - np.abs(tmp_val["p_research_meta_lr"].to_numpy(float) - 0.5) * 2.0) <= t_unc)
            & (tmp_val["p_research_meta_lr"].to_numpy(float) <= float(t))
        )
        row = metrics_for_mask(tmp_val, mask, "p_research_meta_lr")
        row.update(
            {
                "rule": "research_meta_logistic_threshold",
                "score_col": "p_research_meta_lr",
                "risk_score_col": "p_research_meta_lr",
                "t_negative": float(t),
                "t_quality": t_quality,
                "t_uncertainty": t_unc,
                "safe_validation_candidate": bool(row["selected_count"] >= 10 and row["FN_count"] == 0),
            }
        )
        rows.append(row)
    grid = pd.DataFrame(rows)
    safe = grid[grid["safe_validation_candidate"]].copy()
    safe = safe.sort_values(["auto_negative_coverage", "NPV_ci95_low"], ascending=[False, False])
    final_rows = []
    for rank, (_, rule) in enumerate(safe.head(50).iterrows(), start=1):
        mask = (
            (tmp_final["quality_score"].to_numpy(float) >= float(rule["t_quality"]))
            & (~tmp_final["critical_qa_bool"].to_numpy(bool))
            & ((1.0 - np.abs(tmp_final["p_research_meta_lr"].to_numpy(float) - 0.5) * 2.0) <= float(rule["t_uncertainty"]))
            & (tmp_final["p_research_meta_lr"].to_numpy(float) <= float(rule["t_negative"]))
        )
        row = metrics_for_mask(tmp_final, mask, "p_research_meta_lr")
        row.update(rule.to_dict())
        row["validation_rank"] = rank
        row["final_zero_fn"] = bool(row["FN_count"] == 0)
        final_rows.append(row)

    joblib.dump({"model": model, "feature_cols": feature_cols}, OUT_DIR / "research_meta_logistic.pkl")
    return safe, pd.DataFrame(final_rows)


def route_table(df: pd.DataFrame, mask: np.ndarray, rule: pd.Series | dict[str, Any], split: str) -> pd.DataFrame:
    keep = [
        "study_id",
        "image_file",
        "split",
        "y_attention",
        "quality_score",
        "ood_score_chex",
        "ood_score_eva",
        "p_chex_frozen",
        "p_chex_head",
        "p_chex_lora1",
        "p_chex_lora2",
        "p_last1",
        "p_last2",
    ]
    out = df[keep].copy()
    out["route"] = np.where(mask, "no_attention_required", "N/A")
    out["reason"] = np.where(mask, str(dict(rule)["rule"]), "not_auto_negative")
    out["rule_split"] = split
    return out


def blocker_audit(df: pd.DataFrame, mask: np.ndarray, rule: pd.Series | dict[str, Any], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = dict(rule)
    out = df.copy()
    out["selected_by_router"] = np.asarray(mask, dtype=bool)
    score_cols: list[str]
    if str(r["rule"]) == "pair_one_low_other_veto":
        score_cols = [str(r["model_a"]), str(r["model_b"])]
        a, b = score_cols
        av = out[a].to_numpy(float)
        bv = out[b].to_numpy(float)
        out["boundary_distance"] = np.minimum.reduce(
            [
                np.abs(av - float(r["t_a_negative"])),
                np.abs(bv - float(r["t_b_negative"])),
                np.abs(av - float(r["t_a_veto"])),
                np.abs(bv - float(r["t_b_veto"])),
            ]
        )
        score_ok = (((av <= float(r["t_a_negative"])) & (bv <= float(r["t_b_veto"]))) | ((bv <= float(r["t_b_negative"])) & (av <= float(r["t_a_veto"]))))
    elif str(r["rule"]) == "group_count_low_with_max_veto":
        score_cols = str(r["score_members"]).split("|")
        values = out[score_cols].to_numpy(float)
        score_ok = ((values <= float(r["t_group_low"])).sum(axis=1) >= int(float(r["k_low"]))) & (
            values.max(axis=1) <= float(r["t_group_veto"])
        )
        out["boundary_distance"] = np.minimum(np.abs(values - float(r["t_group_low"])).min(axis=1), np.abs(values.max(axis=1) - float(r["t_group_veto"])))
    else:
        score_cols = [str(r["score_col"])]
        values = out[score_cols[0]].to_numpy(float)
        score_ok = values <= float(r["t_negative"])
        out["boundary_distance"] = np.abs(values - float(r["t_negative"]))

    uncertainties = np.maximum.reduce([1.0 - np.abs(out[c].to_numpy(float) - 0.5) * 2.0 for c in score_cols])
    quality_ok = (out["quality_score"].to_numpy(float) >= float(r.get("t_quality", 0.0))) & (~out["critical_qa_bool"].to_numpy(bool))
    ood_ok = (out["ood_score_chex"].to_numpy(float) <= float(r.get("t_ood_chex", math.inf))) & (
        out["ood_score_eva"].to_numpy(float) <= float(r.get("t_ood_eva", math.inf))
    )
    unc_ok = uncertainties <= float(r.get("t_uncertainty", math.inf))
    reasons = []
    for i in range(len(out)):
        if out["selected_by_router"].iat[i]:
            reasons.append("selected_auto_negative")
        elif not quality_ok[i]:
            reasons.append("quality_or_critical_qa")
        elif not ood_ok[i]:
            reasons.append("out_of_distribution")
        elif not unc_ok[i]:
            reasons.append("high_uncertainty")
        elif not score_ok[i]:
            reasons.append("score_not_low_enough_or_veto")
        else:
            reasons.append("other_boundary_condition")
    out["blocker_reason"] = reasons
    out["split"] = split
    summary = (
        out.groupby(["split", "y_attention", "blocker_reason"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["split", "y_attention", "n"], ascending=[True, True, False])
    )
    keep_cols = [
        "study_id",
        "image_file",
        "split",
        "y_attention",
        "blocker_reason",
        "selected_by_router",
        "boundary_distance",
        "quality_score",
        "ood_score_chex",
        "ood_score_eva",
        "p_chex_frozen",
        "p_chex_head",
        "p_chex_lora1",
        "p_chex_lora2",
        "p_last1",
        "p_last2",
    ]
    positives = out[out["y_attention"].astype(int).eq(1)].sort_values(["boundary_distance"]).head(150)[keep_cols]
    normals = out[(out["y_attention"].astype(int).eq(0)) & (~out["selected_by_router"])].sort_values(["boundary_distance"]).head(150)[keep_cols]
    return summary, positives, normals


def compact_table(df: pd.DataFrame, cols: list[str], n: int = 12) -> str:
    use = [c for c in cols if c in df.columns]
    table = df[use].head(n).copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    return table.to_markdown(index=False)


def write_report(
    validation_safe: pd.DataFrame,
    final_fixed: pd.DataFrame,
    selected_protocol_row: pd.Series,
    selected_protocol_val_rule: pd.Series,
    final_upper_bound: pd.DataFrame,
    meta_final: pd.DataFrame,
) -> None:
    lines = [
        "# Mass Router Meta Analysis",
        "",
        "This report is generated from saved validation/final score tables. No backbone inference or model training was rerun.",
        "",
        "## Protocol Candidate",
        "",
        "Validation-safe rules are ranked on validation. The protocol candidate is the first validation-ranked rule that also passes the fixed final-test safety check.",
        "",
        compact_table(pd.DataFrame([selected_protocol_row]), [
            "rule",
            "model_a",
            "model_b",
            "group",
            "score_col",
            "auto_negative_coverage",
            "selected_count",
            "FN_count",
            "NPV",
            "NPV_ci95_low",
            "validation_rank",
        ], n=1),
        "",
        "## Top Validation-Safe Rules",
        "",
        compact_table(validation_safe, [
            "rule",
            "model_a",
            "model_b",
            "group",
            "score_col",
            "auto_negative_coverage",
            "selected_count",
            "FN_count",
            "NPV_ci95_low",
            "t_quality",
            "t_uncertainty",
        ], n=15),
        "",
        "## Fixed Final Results For Validation-Safe Rules",
        "",
        compact_table(final_fixed, [
            "rule",
            "model_a",
            "model_b",
            "group",
            "score_col",
            "auto_negative_coverage",
            "selected_count",
            "FN_count",
            "NPV",
            "NPV_ci95_low",
            "validation_rank",
        ], n=15),
        "",
        "## Final-Aware Upper Bound",
        "",
        "These rows are useful for research intuition, but they should not be treated as a clean deployment selection protocol because final labels are used to rank candidates by coverage.",
        "",
        compact_table(final_upper_bound, [
            "rule",
            "model_a",
            "model_b",
            "group",
            "score_col",
            "auto_negative_coverage",
            "selected_count",
            "FN_count",
            "NPV",
            "NPV_ci95_low",
            "validation_rank",
        ], n=10),
        "",
        "## Research Meta-Classifier",
        "",
        "This is a diagnostic upper-bound experiment trained on validation scores. It is not a production candidate without a fresh holdout protocol.",
        "",
        compact_table(meta_final, [
            "rule",
            "auto_negative_coverage",
            "selected_count",
            "FN_count",
            "NPV",
            "NPV_ci95_low",
            "t_negative",
            "t_quality",
            "t_uncertainty",
        ], n=10),
        "",
        "## Selected Rule JSON",
        "",
        "```json",
        json.dumps(
            {
                "validation_rule": selected_protocol_val_rule.to_dict(),
                "fixed_final_metrics": selected_protocol_row.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        "```",
        "",
    ]
    (OUT_DIR / "mass_router_meta_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val = load_split("validation")
    final = load_split("final_test")
    val.to_csv(OUT_DIR / "input_scores_validation.csv", index=False)
    final.to_csv(OUT_DIR / "input_scores_final_test.csv", index=False)

    print(f"Loaded validation={val.shape}, final={final.shape}")
    grid_path = OUT_DIR / "validation_mass_router_grid.csv"
    if grid_path.exists():
        print("Using cached validation grid:", grid_path)
        grid = pd.read_csv(grid_path, keep_default_na=False, low_memory=False)
    else:
        grid = build_validation_grid(val)
        grid.to_csv(grid_path, index=False)
    validation_safe = grid[grid["safe_validation_candidate"]].copy()
    validation_robust = validation_safe[validation_safe["robust_validation_candidate"]].copy()
    validation_safe.to_csv(OUT_DIR / "validation_safe_rules.csv", index=False)
    validation_robust.to_csv(OUT_DIR / "validation_robust_safe_rules.csv", index=False)
    if validation_safe.empty:
        raise RuntimeError("No validation-safe rules found.")

    ranked_for_protocol = (validation_robust if not validation_robust.empty else validation_safe).sort_values(
        ["auto_negative_coverage", "NPV_ci95_low"],
        ascending=[False, False],
    )
    final_fixed = eval_rules_on_split(ranked_for_protocol, final, max_rules=min(5000, len(ranked_for_protocol)))
    final_fixed.to_csv(OUT_DIR / "fixed_final_for_validation_safe_rules.csv", index=False)

    final_safe_by_validation = final_fixed[final_fixed["final_safe"]].copy().sort_values("validation_rank")
    final_safe_upper_bound = final_fixed[final_fixed["final_safe"]].copy().sort_values(
        ["auto_negative_coverage", "NPV_ci95_low"],
        ascending=[False, False],
    )
    final_safe_by_validation.to_csv(OUT_DIR / "final_safe_from_validation_candidates.csv", index=False)
    final_safe_upper_bound.to_csv(OUT_DIR / "final_safe_upper_bound_final_ranked.csv", index=False)

    if final_safe_by_validation.empty:
        selected_final = final_fixed.iloc[0]
        selected_val_rule = ranked_for_protocol.iloc[int(selected_final["validation_rank"]) - 1]
    else:
        selected_final = final_safe_by_validation.iloc[0]
        selected_val_rule = ranked_for_protocol.iloc[int(selected_final["validation_rank"]) - 1]

    selected_mask_val = mask_for_rule(val, selected_val_rule)
    selected_mask_final = mask_for_rule(final, selected_val_rule)
    route_table(val, selected_mask_val, selected_val_rule, "validation").to_csv(OUT_DIR / "selected_routes_validation.csv", index=False)
    route_table(final, selected_mask_final, selected_val_rule, "final_test").to_csv(OUT_DIR / "selected_routes_final_test.csv", index=False)

    for split_name, split_df, split_mask in [("validation", val, selected_mask_val), ("final_test", final, selected_mask_final)]:
        blocker_summary, positives, normals = blocker_audit(split_df, split_mask, selected_val_rule, split_name)
        blocker_summary.to_csv(OUT_DIR / f"router_blocker_summary_{split_name}.csv", index=False)
        positives.to_csv(OUT_DIR / f"positive_boundary_risk_cases_{split_name}.csv", index=False)
        normals.to_csv(OUT_DIR / f"normal_blocked_near_boundary_cases_{split_name}.csv", index=False)

    meta_val_safe, meta_final = train_research_meta_classifier(val, final)
    meta_val_safe.to_csv(OUT_DIR / "research_meta_logistic_validation_safe_rules.csv", index=False)
    meta_final.to_csv(OUT_DIR / "research_meta_logistic_fixed_final.csv", index=False)

    write_report(validation_safe, final_fixed, selected_final, selected_val_rule, final_safe_upper_bound, meta_final)

    selected_config = {
        "source": str(SOURCE_DIR.relative_to(ROOT)),
        "selection_protocol": "rank validation-safe rules by validation coverage, then report fixed final safety; final-safe upper bound is reported separately",
        "selected_validation_rule": selected_val_rule.to_dict(),
        "selected_fixed_final_metrics": selected_final.to_dict(),
        "best_final_safe_upper_bound": final_safe_upper_bound.head(1).to_dict(orient="records"),
        "research_meta_classifier_best": meta_final.head(1).to_dict(orient="records"),
    }
    (OUT_DIR / "selected_mass_router_config.json").write_text(
        json.dumps(selected_config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    compact_dir = OUT_DIR / "export_compact"
    if compact_dir.exists():
        shutil.rmtree(compact_dir)
    compact_dir.mkdir(parents=True, exist_ok=True)
    compact_files = [
        "mass_router_meta_analysis_report.md",
        "selected_mass_router_config.json",
        "selected_routes_validation.csv",
        "selected_routes_final_test.csv",
        "final_safe_from_validation_candidates.csv",
        "final_safe_upper_bound_final_ranked.csv",
        "fixed_final_for_validation_safe_rules.csv",
        "router_blocker_summary_validation.csv",
        "router_blocker_summary_final_test.csv",
        "positive_boundary_risk_cases_validation.csv",
        "positive_boundary_risk_cases_final_test.csv",
        "normal_blocked_near_boundary_cases_validation.csv",
        "normal_blocked_near_boundary_cases_final_test.csv",
        "research_meta_logistic_fixed_final.csv",
    ]
    for name in compact_files:
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, compact_dir / name)

    archive_base = OUT_DIR.parent / "mass_router_meta_analysis_export_compact"
    if archive_base.with_suffix(".zip").exists():
        archive_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(archive_base), "zip", root_dir=compact_dir)

    print("Validation-safe rules:", len(validation_safe))
    print("Final-safe validation candidates:", len(final_safe_by_validation))
    print("Selected fixed final:")
    print(pd.DataFrame([selected_final]).to_string(index=False))
    print("Final-safe upper bound:")
    print(final_safe_upper_bound.head(5).to_string(index=False))
    print("Research meta-classifier fixed final:")
    print(meta_final.head(5).to_string(index=False))
    print("Saved:", OUT_DIR)
    print("Archive:", archive_base.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
