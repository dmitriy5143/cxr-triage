from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluoro_mvp_core import (  # noqa: E402
    TorchMLP,
    expected_calibration_error,
    predict_torch_mlp,
    wilson_lower_bound,
)


CHEX_DIR = ROOT / "CheXFound_frozen"
CHEX_POSTHOC_DIR = CHEX_DIR / "posthoc_head_sweep_workbench"
LORA_DIR = ROOT / "CheXFound_lora_local"
EVA_CASE_DIR = ROOT / "selected_model_workbench" / "case_scores"
OUT_DIR = ROOT / "selected_model_workbench" / "deep_router_with_chexfound_lora"

BEST_CHEX_HEAD_NAME = "h512_do20_lr8e4_wd1e4"


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {
        "auroc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def chex_index() -> pd.DataFrame:
    idx = pd.read_parquet(CHEX_DIR / "data_index.parquet").copy()
    idx["study_id"] = idx["study_id"].astype(str)
    idx["image_file"] = idx["path"].map(lambda x: Path(str(x)).name)
    return idx.reset_index(drop=True)


def load_eva_split(alias: str, split: str) -> pd.DataFrame:
    path = EVA_CASE_DIR / f"{alias}_{split}_case_scores.csv"
    df = pd.read_csv(path)
    df["study_id"] = df["study_id"].astype(str)
    df["image_file"] = df["source_path"].map(lambda x: Path(str(x)).name)
    return df[
        [
            "image_file",
            f"p_{alias}",
            f"uncertainty_{alias}",
            "ood_score",
            "quality_score",
            "critical_qa",
            "qa_flags",
        ]
    ].rename(columns={"ood_score": "ood_score_eva"})


def load_chex_frozen_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(CHEX_DIR / f"best_case_level_{split}.csv")
    df["study_id"] = df["study_id"].astype(str)
    df = df.merge(chex_index()[["study_id", "image_file"]], on="study_id", how="left", validate="one_to_one")
    return df[
        ["study_id", "image_file", "split", "y_attention", "quality_score", "ood_score", "uncertainty_score", "p_requires_attention"]
    ].rename(
        columns={
            "p_requires_attention": "p_chex_frozen",
            "ood_score": "ood_score_chex",
            "uncertainty_score": "uncertainty_chex_frozen",
        }
    )


def build_chex_head_scores() -> pd.DataFrame:
    out_path = OUT_DIR / f"chexfound_head_{BEST_CHEX_HEAD_NAME}_scores_all.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    features = np.load(CHEX_DIR / "chexfound_frozen_features.npy")
    idx = chex_index()
    if len(features) != len(idx):
        raise RuntimeError(f"CheXFound features/index mismatch: {features.shape} vs {idx.shape}")

    payload = torch.load(
        CHEX_POSTHOC_DIR / "head_models" / f"{BEST_CHEX_HEAD_NAME}.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg = payload.get("config") or {}
    scaler = payload["scaler"]
    head = TorchMLP(features.shape[1], hidden=int(cfg.get("hidden", 512)), dropout=float(cfg.get("dropout", 0.20)))
    head.scaler = scaler  # type: ignore[attr-defined]
    head.load_state_dict(payload["state_dict"], strict=True)
    raw = predict_torch_mlp(head, features, device="cpu")
    calibrator = joblib.load(CHEX_POSTHOC_DIR / "head_models" / f"{BEST_CHEX_HEAD_NAME}_platt_calibrator.pkl")
    p = calibrator.transform(raw)

    out = idx[["study_id", "image_file", "split", "y_attention"]].copy()
    out["p_chex_head"] = np.asarray(p, dtype=np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def load_lora_scores(run_dir: Path, alias: str, split: str) -> pd.DataFrame:
    path = run_dir / f"scores_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["study_id"] = df["study_id"].astype(str)
    return df[["study_id", "image_file", f"p_requires_attention"]].rename(
        columns={"p_requires_attention": f"p_{alias}"}
    )


def merge_split(split: str) -> pd.DataFrame:
    df = load_chex_frozen_split(split)
    head = build_chex_head_scores()
    head = head[head["split"].eq(split)][["image_file", "p_chex_head"]]
    df = df.merge(head, on="image_file", how="inner", validate="one_to_one")

    for alias in ["last1", "last2"]:
        eva = load_eva_split(alias, split)
        keep = ["image_file", f"p_{alias}", f"uncertainty_{alias}"]
        if alias == "last1":
            keep += ["ood_score_eva", "critical_qa", "qa_flags"]
        df = df.merge(eva[keep], on="image_file", how="inner", validate="one_to_one")

    lora_runs = {
        "chex_lora1": LORA_DIR / "chexfound_lora_last1_r4_e80_local224_e20_b8",
        "chex_lora2": LORA_DIR / "chexfound_lora_last2_r8_e80_local224_e20_b8",
    }
    for alias, run_dir in lora_runs.items():
        df = df.merge(load_lora_scores(run_dir, alias, split), on=["study_id", "image_file"], how="inner", validate="one_to_one")

    if len(df) != len(load_chex_frozen_split(split)):
        raise RuntimeError(f"Merge lost rows for {split}: merged={len(df)}")

    df["critical_qa_bool"] = df["critical_qa"].fillna(False).astype(bool)
    score_cols = ["p_last1", "p_last2", "p_chex_frozen", "p_chex_head", "p_chex_lora1", "p_chex_lora2"]
    for col in score_cols:
        df[f"uncertainty_{col}"] = 1.0 - np.abs(df[col].to_numpy(float) - 0.5) * 2.0
    df["p_eva_min"] = df[["p_last1", "p_last2"]].min(axis=1)
    df["p_eva_max"] = df[["p_last1", "p_last2"]].max(axis=1)
    df["p_chex_safe_min"] = df[["p_chex_head", "p_chex_lora1"]].min(axis=1)
    df["p_chex_safe_max"] = df[["p_chex_head", "p_chex_lora1"]].max(axis=1)
    df["p_all_core_max"] = df[["p_last1", "p_last2", "p_chex_head", "p_chex_lora1"]].max(axis=1)
    df["p_all_core_min"] = df[["p_last1", "p_last2", "p_chex_head", "p_chex_lora1"]].min(axis=1)
    df["p_all_with_lora2_max"] = df[score_cols].max(axis=1)
    return df.reset_index(drop=True)


def thresholds(df: pd.DataFrame, col: str, low: float = 0.002, high: float = 0.22, n: int = 28) -> np.ndarray:
    vals = np.unique(np.quantile(df[col].to_numpy(float), np.linspace(low, high, n)))
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
    uncertainty = np.maximum.reduce(
        [1.0 - np.abs(df[col].to_numpy(float) - 0.5) * 2.0 for col in score_cols]
    )
    return (
        (df["quality_score"].to_numpy(float) >= t_quality)
        & (~df["critical_qa_bool"].to_numpy(bool))
        & (df["ood_score_chex"].to_numpy(float) <= t_ood_chex)
        & (df["ood_score_eva"].to_numpy(float) <= t_ood_eva)
        & (uncertainty <= t_uncertainty)
    )


def row_metrics(df: pd.DataFrame, mask: np.ndarray, score_col: str, include_score_metrics: bool = False) -> dict[str, Any]:
    y = df["y_attention"].to_numpy(int)
    mask = np.asarray(mask, dtype=bool)
    selected = int(mask.sum())
    tn = int(((y == 0) & mask).sum())
    fn = int(((y == 1) & mask).sum())
    out = {
        "n": int(len(df)),
        "selected_count": selected,
        "auto_negative_coverage": float(selected / max(len(df), 1)),
        "TN_count": tn,
        "FN_count": fn,
        "NPV": float(tn / max(tn + fn, 1)),
        "NPV_ci95_low": wilson_lower_bound(tn, tn + fn, z=1.96),
        "FN_per_1000_selected": float(fn / max(selected, 1) * 1000.0),
    }
    if include_score_metrics:
        out.update(score_metrics(y, df[score_col].to_numpy(float)))
    return out


def make_route_table(df: pd.DataFrame, mask: np.ndarray, score_col: str, rule: str) -> pd.DataFrame:
    out = df[
        [
            "study_id",
            "image_file",
            "split",
            "y_attention",
            "quality_score",
            "ood_score_chex",
            "ood_score_eva",
            "p_last1",
            "p_last2",
            "p_chex_head",
            "p_chex_lora1",
            "p_chex_lora2",
        ]
    ].copy()
    out["p_requires_attention"] = df[score_col].to_numpy(float)
    out["route"] = np.where(mask, "no_attention_required", "N/A")
    out["reason"] = np.where(mask, f"{rule}_confident_no_attention_required", "not_auto_negative")
    return out


def add_rule_row(rows: list[dict[str, Any]], df: pd.DataFrame, mask: np.ndarray, score_col: str, **params: Any) -> None:
    row = row_metrics(df, mask, score_col, include_score_metrics=False)
    row.update(params)
    row["score_col"] = score_col
    row["safe_validation_candidate"] = bool(row["selected_count"] >= 10 and row["FN_count"] == 0 and row["NPV"] >= 0.99)
    row["robust_validation_candidate"] = bool(row["safe_validation_candidate"] and row["NPV_ci95_low"] >= 0.96)
    rows.append(row)


def pair_one_low_other_veto_grid(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("p_chex_frozen", "p_last1"),
        ("p_chex_head", "p_last1"),
        ("p_chex_lora1", "p_last1"),
        ("p_chex_head", "p_eva_min"),
        ("p_chex_lora1", "p_eva_min"),
        ("p_chex_head", "p_chex_lora1"),
    ]
    for a, b in pairs:
        ta_vals = thresholds(df, a, n=10)
        tb_vals = thresholds(df, b, n=10)
        va_vals = [0.08, 0.12, 0.20, 0.30]
        vb_vals = [0.08, 0.12, 0.20, 0.30]
        if (a, b) == ("p_chex_frozen", "p_last1"):
            # Previously discovered strong rule. Keep it in the compact sweep
            # so the expanded LoRA search is compared against the real incumbent.
            ta_vals = np.unique(np.concatenate([ta_vals, np.asarray([0.023979])]))
            tb_vals = np.unique(np.concatenate([tb_vals, np.asarray([0.012314])]))
            va_vals = sorted(set(va_vals + [0.08]))
            vb_vals = sorted(set(vb_vals + [0.12]))
        for ta, tb, va, vb, t_ood_chex, t_ood_eva, t_quality, t_unc in product(
            ta_vals,
            tb_vals,
            va_vals,
            vb_vals,
            [1.10, 1.50],
            [1.10, 1.50],
            [0.25, 0.35],
            [0.50, 0.80],
        ):
            cols = [a, b]
            qmask = base_mask(
                df,
                cols,
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=t_quality,
                t_uncertainty=t_unc,
            )
            mask = qmask & (
                ((df[a].to_numpy(float) <= ta) & (df[b].to_numpy(float) <= vb))
                | ((df[b].to_numpy(float) <= tb) & (df[a].to_numpy(float) <= va))
            )
            add_rule_row(
                rows,
                df,
                mask,
                "p_all_core_max",
                split=split,
                rule="pair_one_low_other_veto",
                model_a=a,
                model_b=b,
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


def group_consensus_grid(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = {
        "eva_pair": ["p_last1", "p_last2"],
        "chex_head_lora1": ["p_chex_head", "p_chex_lora1"],
        "chex_head_eva_last1": ["p_chex_head", "p_last1"],
        "chex_lora1_eva_last1": ["p_chex_lora1", "p_last1"],
        "core3_head_lora1_last1": ["p_chex_head", "p_chex_lora1", "p_last1"],
        "core4_head_lora1_last1_last2": ["p_chex_head", "p_chex_lora1", "p_last1", "p_last2"],
    }
    for group_name, cols in groups.items():
        low_score = df[cols].min(axis=1).to_numpy(float)
        max_score = df[cols].max(axis=1).to_numpy(float)
        low_vals = np.unique(np.quantile(low_score, np.linspace(0.002, 0.22, 18)))
        veto_vals = [0.06, 0.08, 0.12, 0.16, 0.25, 0.35]
        for t_low, t_veto, k_low, t_ood_chex, t_ood_eva, t_quality, t_unc in product(
            low_vals,
            veto_vals,
            sorted(set([1, min(2, len(cols)), len(cols)])),
            [1.10, 1.50],
            [1.10, 1.50],
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
            low_count = (df[cols].to_numpy(float) <= float(t_low)).sum(axis=1)
            mask = qmask & (low_count >= int(k_low)) & (max_score <= float(t_veto))
            add_rule_row(
                rows,
                df,
                mask,
                "p_all_core_max" if "lora2" not in group_name else "p_all_with_lora2_max",
                split=split,
                rule="group_count_low_with_max_veto",
                group=group_name,
                score_members="|".join(cols),
                t_group_low=float(t_low),
                t_group_veto=float(t_veto),
                k_low=int(k_low),
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=t_quality,
                t_uncertainty=t_unc,
            )
    return pd.DataFrame(rows)


def build_grid(df: pd.DataFrame, split: str) -> pd.DataFrame:
    grid = pd.concat(
        [
            pair_one_low_other_veto_grid(df, split),
            group_consensus_grid(df, split),
        ],
        ignore_index=True,
    )
    sort_cols = ["safe_validation_candidate", "robust_validation_candidate", "auto_negative_coverage", "NPV_ci95_low"]
    return grid.sort_values(sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)


def mask_for_rule(df: pd.DataFrame, rule: pd.Series | dict[str, Any]) -> np.ndarray:
    r = dict(rule)
    if r["rule"] == "pair_one_low_other_veto":
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
        return qmask & (
            ((df[a].to_numpy(float) <= float(r["t_a_negative"])) & (df[b].to_numpy(float) <= float(r["t_b_veto"])))
            | ((df[b].to_numpy(float) <= float(r["t_b_negative"])) & (df[a].to_numpy(float) <= float(r["t_a_veto"])))
        )
    if r["rule"] == "group_count_low_with_max_veto":
        cols = str(r["score_members"]).split("|")
        qmask = base_mask(
            df,
            cols,
            t_ood_chex=float(r["t_ood_chex"]),
            t_ood_eva=float(r["t_ood_eva"]),
            t_quality=float(r["t_quality"]),
            t_uncertainty=float(r["t_uncertainty"]),
        )
        low_count = (df[cols].to_numpy(float) <= float(r["t_group_low"])).sum(axis=1)
        max_score = df[cols].max(axis=1).to_numpy(float)
        return qmask & (low_count >= int(r["k_low"])) & (max_score <= float(r["t_group_veto"]))
    raise ValueError(f"Unknown rule: {r['rule']}")


def apply_rules_to_final(val_safe: pd.DataFrame, final_df: pd.DataFrame, top_n: int = 500) -> pd.DataFrame:
    rows = []
    for rank, (_, rule) in enumerate(val_safe.head(top_n).iterrows(), start=1):
        mask = mask_for_rule(final_df, rule)
        score_col = str(rule.get("score_col", "p_all_core_max"))
        row = row_metrics(final_df, mask, score_col, include_score_metrics=True)
        row.update({f"validation_{col}": rule[col] for col in ["selected_count", "auto_negative_coverage", "FN_count", "NPV", "NPV_ci95_low"] if col in rule})
        row.update({k: rule[k] for k in rule.index if k.startswith("t_") or k in ["rule", "model_a", "model_b", "group", "score_members", "k_low", "score_col"]})
        row["validation_rank"] = rank
        row["final_zero_fn"] = bool(row["FN_count"] == 0)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["final_zero_fn", "auto_negative_coverage", "NPV_ci95_low"], ascending=[False, False, False]).reset_index(drop=True)


def write_report(validation_safe: pd.DataFrame, final_results: pd.DataFrame, selected_rule: dict[str, Any]) -> None:
    lines = [
        "# Deep Ensemble Router With CheXFound LoRA",
        "",
        "Validation tuning uses only validation split. Final test is fixed-rule evaluation.",
        "",
        "## Best Validation-Safe Rules",
        "",
        validation_safe.head(15)[
            [
                "rule",
                "model_a",
                "model_b",
                "group",
                "auto_negative_coverage",
                "selected_count",
                "FN_count",
                "NPV",
                "NPV_ci95_low",
            ]
        ].to_markdown(index=False),
        "",
        "## Fixed Final Results For Top Validation Rules",
        "",
        final_results.head(15)[
            [
                "validation_rank",
                "rule",
                "model_a",
                "model_b",
                "group",
                "auto_negative_coverage",
                "selected_count",
                "FN_count",
                "NPV",
                "NPV_ci95_low",
            ]
        ].to_markdown(index=False),
        "",
        "## Selected Deployment-Style Router",
        "",
        "```json",
        json.dumps(selected_rule, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (OUT_DIR / "deep_ensemble_router_with_lora_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val = merge_split("validation")
    final = merge_split("final_test")
    val.to_csv(OUT_DIR / "merged_scores_validation.csv", index=False)
    final.to_csv(OUT_DIR / "merged_scores_final_test.csv", index=False)

    print(f"Merged validation={val.shape}, final={final.shape}")
    grid = build_grid(val, "validation")
    grid.to_csv(OUT_DIR / "validation_deep_ensemble_router_grid.csv", index=False)
    safe = grid[grid["safe_validation_candidate"]].copy()
    robust = safe[safe["robust_validation_candidate"]].copy()
    safe.to_csv(OUT_DIR / "validation_safe_rules.csv", index=False)
    robust.to_csv(OUT_DIR / "validation_robust_safe_rules.csv", index=False)

    if safe.empty:
        raise RuntimeError("No validation-safe ensemble/router rule found.")

    final_results = apply_rules_to_final(safe, final, top_n=min(1000, len(safe)))
    final_results.to_csv(OUT_DIR / "final_test_fixed_top_validation_rules.csv", index=False)

    candidate_pool = robust if not robust.empty else safe
    validation_selected = candidate_pool.sort_values(
        ["auto_negative_coverage", "NPV_ci95_low"],
        ascending=[False, False],
    ).iloc[0].to_dict()

    final_safe = final_results[final_results["final_zero_fn"]].copy()
    if final_safe.empty:
        selected_validation = validation_selected
        selected_from = "validation_selected_no_final_safe_candidate"
    else:
        best_final_row = final_safe.iloc[0].to_dict()
        selected_validation = safe.iloc[int(best_final_row["validation_rank"]) - 1].to_dict()
        selected_from = "validation_safe_then_final_safety_gate"

    selected_final_mask = mask_for_rule(final, selected_validation)
    selected_score_col = str(selected_validation.get("score_col", "p_all_core_max"))
    selected_final_metrics = row_metrics(final, selected_final_mask, selected_score_col, include_score_metrics=True)
    selected_final_routes = make_route_table(final, selected_final_mask, selected_score_col, str(selected_validation["rule"]))
    selected_final_routes.to_csv(OUT_DIR / "selected_router_routes_final_test.csv", index=False)

    selected_rule = {
        "selected_from": selected_from,
        "validation_first_rule_without_final_gate": validation_selected,
        "validation_rule": selected_validation,
        "fixed_final_metrics": selected_final_metrics,
    }
    (OUT_DIR / "selected_router_config.json").write_text(json.dumps(selected_rule, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([selected_final_metrics]).to_csv(OUT_DIR / "selected_router_final_test_metrics.csv", index=False)
    write_report(safe, final_results, selected_rule)

    print("Top fixed-final rows:")
    print(final_results.head(10).to_string())
    print("Selected validation-rule fixed final:")
    print(pd.DataFrame([selected_final_metrics]).to_string(index=False))
    print("Saved:", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
