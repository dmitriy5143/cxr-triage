from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "selected_model_workbench" / "case_scores"
OUT_DIR = ROOT / "selected_model_workbench" / "deep_router_analysis"
ROUTER_DIR = ROOT / "selected_model_workbench" / "router_workbench" / "router_configs"
RUN_ROOT = ROOT / "fluoro_mvp_outputs" / "incxr_eva_base_partial_unfreeze_t4"


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def load_router(name: str) -> dict[str, Any]:
    return json.loads((ROUTER_DIR / name).read_text(encoding="utf-8"))


def route_mask(
    df: pd.DataFrame,
    score_col: str,
    *,
    t_negative: float,
    t_ood: float,
    t_quality: float,
    t_uncertainty: float,
) -> np.ndarray:
    p = df[score_col].to_numpy(float)
    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0
    return (
        (df["quality_score"].to_numpy(float) >= t_quality)
        & (~df["critical_qa"].astype(bool).to_numpy())
        & (df["ood_score"].to_numpy(float) <= t_ood)
        & (uncertainty <= t_uncertainty)
        & (p <= t_negative)
    )


def metrics_for_mask(y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    mask = np.asarray(mask).astype(bool)
    n_sel = int(mask.sum())
    fn = int(((y == 1) & mask).sum())
    tn = int(((y == 0) & mask).sum())
    return {
        "n": int(len(y)),
        "selected_count": n_sel,
        "auto_negative_coverage": float(n_sel / max(len(y), 1)),
        "TN_count": tn,
        "FN_count": fn,
        "NPV": float(tn / max(tn + fn, 1)),
        "FN_per_1000_selected": float(fn / max(n_sel, 1) * 1000.0),
    }


def load_case_scores(alias: str, split: str) -> pd.DataFrame:
    path = CASE_DIR / f"{alias}_{split}_case_scores.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/export_eva_partial_case_scores.py for {alias} first."
        )
    return pd.read_csv(path)


def available_case_scores() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in CASE_DIR.glob("*_case_scores.csv"):
        parts = path.name.split("_")
        if len(parts) >= 3 and parts[-2] in {"validation", "final", "test"}:
            continue
        alias = path.name.split("_case_scores.csv")[0]
        if alias in {"last1", "last2"}:
            out.setdefault(alias, set()).add("combined")
    for path in CASE_DIR.glob("*_*_case_scores.csv"):
        name = path.name.replace("_case_scores.csv", "")
        for split in ["validation", "final_test"]:
            suffix = "_" + split
            if name.endswith(suffix):
                alias = name[: -len(suffix)]
                out.setdefault(alias, set()).add(split)
    return out


def merge_last1_last2(split: str) -> pd.DataFrame:
    a = load_case_scores("last1", split)
    b = load_case_scores("last2", split)
    keep_b = ["study_id", f"raw_last2", f"p_last2", f"uncertainty_last2"]
    merged = a.merge(b[keep_b], on="study_id", how="inner", validate="one_to_one")
    if len(merged) != len(a):
        raise ValueError(f"last1/last2 merge lost rows for {split}: {len(a)} -> {len(merged)}")
    merged["p_min"] = merged[["p_last1", "p_last2"]].min(axis=1)
    merged["p_max"] = merged[["p_last1", "p_last2"]].max(axis=1)
    merged["p_mean"] = merged[["p_last1", "p_last2"]].mean(axis=1)
    merged["p_gap_abs"] = (merged["p_last1"] - merged["p_last2"]).abs()
    merged["uncertainty_mean"] = (
        (1.0 - np.abs(merged["p_last1"] - 0.5) * 2.0)
        + (1.0 - np.abs(merged["p_last2"] - 0.5) * 2.0)
    ) / 2.0
    return merged


def candidate_rule_grid(df: pd.DataFrame, split: str) -> pd.DataFrame:
    y = df["y_attention"].to_numpy(int)
    routers = {
        "last1_primary": load_router("primary_last1_deployment_zero_fn_router.json"),
        "last2_strict": load_router("challenger_last2_strict_zero_fn_router.json"),
        "last2_aggressive": load_router("challenger_last2_aggressive_research_router.json"),
    }
    rows: list[dict[str, Any]] = []

    # Single-model router variants.
    for name, score_col, router_name in [
        ("single_last1", "p_last1", "last1_primary"),
        ("single_last2_strict", "p_last2", "last2_strict"),
        ("single_last2_aggressive", "p_last2", "last2_aggressive"),
    ]:
        router = routers[router_name]
        mask = route_mask(
            df,
            score_col,
            t_negative=float(router["selected_T_negative"]),
            t_ood=float(router["selected_t_ood"]),
            t_quality=float(router["selected_t_quality"]),
            t_uncertainty=float(router["selected_t_uncertainty"]),
        )
        row = metrics_for_mask(y, mask)
        row.update(
            {
                "split": split,
                "rule": name,
                "score": score_col,
                "t_negative": float(router["selected_T_negative"]),
                "t_ood": float(router["selected_t_ood"]),
                "t_quality": float(router["selected_t_quality"]),
                "t_uncertainty": float(router["selected_t_uncertainty"]),
                "p_gap_max": np.nan,
            }
        )
        rows.append(row)

    # Ensemble/router search. Validation rows choose the candidates; final rows reuse selected config later.
    score_options = ["p_max", "p_mean"]
    quantiles = np.linspace(0.01, 0.18, 80)
    t_oods = [0.85, 0.95, 1.05, 1.10, 1.20, 1.35]
    t_uncertainties = [0.45, 0.55, 0.65, 0.75, 0.85, 1.0]
    t_quality_options = [0.25, 0.35, 0.45]
    p_gap_max_options = [0.05, 0.10, 0.15, 0.25, 1.0]

    for score_col, q, t_ood, t_unc, t_quality, gap_max in product(
        score_options,
        quantiles,
        t_oods,
        t_uncertainties,
        t_quality_options,
        p_gap_max_options,
    ):
        t_negative = float(np.quantile(df[score_col].to_numpy(float), q))
        base = route_mask(
            df,
            score_col,
            t_negative=t_negative,
            t_ood=t_ood,
            t_quality=t_quality,
            t_uncertainty=t_unc,
        )
        agree = df["p_gap_abs"].to_numpy(float) <= gap_max
        both_low = (df["p_last1"].to_numpy(float) <= t_negative) & (df["p_last2"].to_numpy(float) <= t_negative)
        mask = base & agree & both_low
        row = metrics_for_mask(y, mask)
        row.update(
            {
                "split": split,
                "rule": "ensemble_both_low_agree",
                "score": score_col,
                "t_negative": t_negative,
                "t_ood": t_ood,
                "t_quality": t_quality,
                "t_uncertainty": t_unc,
                "p_gap_max": gap_max,
                "quantile": q,
            }
        )
        rows.append(row)

    # Softer ensemble: one model can trigger auto-negative if it is very confident
    # and the other model stays below a veto threshold. This tests "one confident,
    # the other does not object" without sacrificing validation safety.
    low_quantiles = np.linspace(0.015, 0.14, 20)
    veto_thresholds = [0.05, 0.08, 0.12, 0.20, 0.35]
    for q1, q2, veto1, veto2, t_ood, t_unc, t_quality in product(
        low_quantiles,
        low_quantiles,
        veto_thresholds,
        veto_thresholds,
        [0.95, 1.10, 1.25],
        [0.55, 0.65, 0.80, 1.0],
        [0.25, 0.35],
    ):
        t1 = float(np.quantile(df["p_last1"].to_numpy(float), q1))
        t2 = float(np.quantile(df["p_last2"].to_numpy(float), q2))
        p1 = df["p_last1"].to_numpy(float)
        p2 = df["p_last2"].to_numpy(float)
        qmask = (
            (df["quality_score"].to_numpy(float) >= t_quality)
            & (~df["critical_qa"].astype(bool).to_numpy())
            & (df["ood_score"].to_numpy(float) <= t_ood)
        )
        uncertainty1 = 1.0 - np.abs(p1 - 0.5) * 2.0
        uncertainty2 = 1.0 - np.abs(p2 - 0.5) * 2.0
        uncertainty_mask = np.maximum(uncertainty1, uncertainty2) <= t_unc
        one_confident_other_not_veto = ((p1 <= t1) & (p2 <= veto2)) | ((p2 <= t2) & (p1 <= veto1))
        mask = qmask & uncertainty_mask & one_confident_other_not_veto
        row = metrics_for_mask(y, mask)
        row.update(
            {
                "split": split,
                "rule": "ensemble_one_low_other_veto",
                "score": "p_pair",
                "t_negative": max(t1, t2),
                "t_last1_negative": t1,
                "t_last2_negative": t2,
                "t_last1_veto": veto1,
                "t_last2_veto": veto2,
                "t_ood": t_ood,
                "t_quality": t_quality,
                "t_uncertainty": t_unc,
                "p_gap_max": np.nan,
                "quantile_last1": q1,
                "quantile_last2": q2,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out["AUROC_last1"] = safe_auc(y, df["p_last1"].to_numpy(float))
    out["AUPRC_last1"] = safe_auprc(y, df["p_last1"].to_numpy(float))
    out["AUROC_last2"] = safe_auc(y, df["p_last2"].to_numpy(float))
    out["AUPRC_last2"] = safe_auprc(y, df["p_last2"].to_numpy(float))
    out["AUROC_pmax"] = safe_auc(y, df["p_max"].to_numpy(float))
    out["AUPRC_pmax"] = safe_auprc(y, df["p_max"].to_numpy(float))
    out["safe_zero_fn"] = (out["FN_count"].eq(0)) & (out["NPV"].ge(0.99))
    return out


def select_validation_rules(validation_grid: pd.DataFrame) -> pd.DataFrame:
    safe = validation_grid[validation_grid["safe_zero_fn"]].copy()
    if safe.empty:
        return safe
    return safe.sort_values(
        ["auto_negative_coverage", "selected_count", "NPV"],
        ascending=[False, False, False],
    ).head(25)


def apply_rules_to_split(df: pd.DataFrame, rules: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    y = df["y_attention"].to_numpy(int)
    for _, rule in rules.iterrows():
        if str(rule["rule"]) == "ensemble_one_low_other_veto":
            p1 = df["p_last1"].to_numpy(float)
            p2 = df["p_last2"].to_numpy(float)
            qmask = (
                (df["quality_score"].to_numpy(float) >= float(rule["t_quality"]))
                & (~df["critical_qa"].astype(bool).to_numpy())
                & (df["ood_score"].to_numpy(float) <= float(rule["t_ood"]))
            )
            uncertainty1 = 1.0 - np.abs(p1 - 0.5) * 2.0
            uncertainty2 = 1.0 - np.abs(p2 - 0.5) * 2.0
            uncertainty_mask = np.maximum(uncertainty1, uncertainty2) <= float(rule["t_uncertainty"])
            mask = qmask & uncertainty_mask & (
                ((p1 <= float(rule["t_last1_negative"])) & (p2 <= float(rule["t_last2_veto"])))
                | ((p2 <= float(rule["t_last2_negative"])) & (p1 <= float(rule["t_last1_veto"])))
            )
        else:
            mask = route_mask(
                df,
                str(rule["score"]),
                t_negative=float(rule["t_negative"]),
                t_ood=float(rule["t_ood"]),
                t_quality=float(rule["t_quality"]),
                t_uncertainty=float(rule["t_uncertainty"]),
            )
        if str(rule["rule"]) == "ensemble_both_low_agree":
            mask = (
                mask
                & (df["p_gap_abs"].to_numpy(float) <= float(rule["p_gap_max"]))
                & (df["p_last1"].to_numpy(float) <= float(rule["t_negative"]))
                & (df["p_last2"].to_numpy(float) <= float(rule["t_negative"]))
            )
        row = metrics_for_mask(y, mask)
        for col in [
            "rule",
            "score",
            "t_negative",
            "t_ood",
            "t_quality",
            "t_uncertainty",
            "p_gap_max",
            "t_last1_negative",
            "t_last2_negative",
            "t_last1_veto",
            "t_last2_veto",
        ]:
            row[col] = rule.get(col)
        row["split"] = split
        rows.append(row)
    return pd.DataFrame(rows)


def near_threshold_cases(df: pd.DataFrame, rules: pd.DataFrame, top_n: int = 120) -> pd.DataFrame:
    best = rules.iloc[0]
    out = df.copy()
    if str(best["rule"]) == "ensemble_one_low_other_veto":
        p1 = out["p_last1"].to_numpy(float)
        p2 = out["p_last2"].to_numpy(float)
        t1 = float(best["t_last1_negative"])
        t2 = float(best["t_last2_negative"])
        veto1 = float(best["t_last1_veto"])
        veto2 = float(best["t_last2_veto"])
        qmask = (
            (out["quality_score"].to_numpy(float) >= float(best["t_quality"]))
            & (~out["critical_qa"].astype(bool).to_numpy())
            & (out["ood_score"].to_numpy(float) <= float(best["t_ood"]))
        )
        uncertainty1 = 1.0 - np.abs(p1 - 0.5) * 2.0
        uncertainty2 = 1.0 - np.abs(p2 - 0.5) * 2.0
        uncertainty_mask = np.maximum(uncertainty1, uncertainty2) <= float(best["t_uncertainty"])
        out["selected_by_best_rule"] = qmask & uncertainty_mask & (
            ((p1 <= t1) & (p2 <= veto2)) | ((p2 <= t2) & (p1 <= veto1))
        )
        out["distance_to_T_negative"] = np.minimum(np.abs(p1 - t1), np.abs(p2 - t2))
        out["near_threshold_bucket"] = pd.cut(
            np.minimum(p1 - t1, p2 - t2),
            bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
            labels=["well_below", "slightly_below", "on_boundary", "slightly_above", "well_above"],
        )
    else:
        score_col = str(best["score"])
        t = float(best["t_negative"])
        out["selected_by_best_rule"] = route_mask(
            out,
            score_col,
            t_negative=t,
            t_ood=float(best["t_ood"]),
            t_quality=float(best["t_quality"]),
            t_uncertainty=float(best["t_uncertainty"]),
        )
        if str(best["rule"]) == "ensemble_both_low_agree":
            out["selected_by_best_rule"] = (
                out["selected_by_best_rule"]
                & (out["p_gap_abs"] <= float(best["p_gap_max"]))
                & (out["p_last1"] <= t)
                & (out["p_last2"] <= t)
            )
        out["distance_to_T_negative"] = (out[score_col] - t).abs()
        out["near_threshold_bucket"] = pd.cut(
            out[score_col] - t,
            bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
            labels=["well_below", "slightly_below", "on_boundary", "slightly_above", "well_above"],
        )
    cols = [
        "study_id",
        "split",
        "y_attention",
        "quality_score",
        "critical_qa",
        "qa_flags",
        "ood_score",
        "p_last1",
        "p_last2",
        "p_max",
        "p_mean",
        "p_gap_abs",
        "selected_by_best_rule",
        "distance_to_T_negative",
        "near_threshold_bucket",
        "source_path",
        "image_eva_path",
    ]
    return out.sort_values(["distance_to_T_negative", "p_gap_abs"]).head(top_n)[cols]


def blocker_analysis(df: pd.DataFrame, rules: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best = rules.iloc[0]
    out = df.copy()
    if str(best["rule"]) != "ensemble_one_low_other_veto":
        out["blocker_reason"] = "single_model_rule"
        return (
            out.groupby(["split", "y_attention", "blocker_reason"]).size().rename("n").reset_index(),
            out.head(0),
            out.head(0),
        )

    p1 = out["p_last1"].to_numpy(float)
    p2 = out["p_last2"].to_numpy(float)
    t1 = float(best["t_last1_negative"])
    t2 = float(best["t_last2_negative"])
    veto1 = float(best["t_last1_veto"])
    veto2 = float(best["t_last2_veto"])
    t_ood = float(best["t_ood"])
    t_quality = float(best["t_quality"])
    t_unc = float(best["t_uncertainty"])

    q_ok = (out["quality_score"].to_numpy(float) >= t_quality) & (~out["critical_qa"].astype(bool).to_numpy())
    ood_ok = out["ood_score"].to_numpy(float) <= t_ood
    uncertainty1 = 1.0 - np.abs(p1 - 0.5) * 2.0
    uncertainty2 = 1.0 - np.abs(p2 - 0.5) * 2.0
    unc_ok = np.maximum(uncertainty1, uncertainty2) <= t_unc
    last1_low = p1 <= t1
    last2_low = p2 <= t2
    last1_veto_ok = p1 <= veto1
    last2_veto_ok = p2 <= veto2
    selected = q_ok & ood_ok & unc_ok & (((last1_low & last2_veto_ok) | (last2_low & last1_veto_ok)))

    reasons = []
    for i in range(len(out)):
        if selected[i]:
            reasons.append("selected_auto_negative")
        elif not q_ok[i]:
            reasons.append("quality_or_critical_qa")
        elif not ood_ok[i]:
            reasons.append("out_of_distribution")
        elif not unc_ok[i]:
            reasons.append("high_uncertainty")
        elif not (last1_low[i] or last2_low[i]):
            reasons.append("no_model_low_enough")
        elif last1_low[i] and not last2_veto_ok[i]:
            reasons.append("last2_veto")
        elif last2_low[i] and not last1_veto_ok[i]:
            reasons.append("last1_veto")
        else:
            reasons.append("other_boundary_condition")

    out["blocker_reason"] = reasons
    out["selected_by_best_rule"] = selected
    out["last1_low"] = last1_low
    out["last2_low"] = last2_low
    out["last1_veto_ok"] = last1_veto_ok
    out["last2_veto_ok"] = last2_veto_ok
    out["boundary_distance"] = np.minimum(np.abs(p1 - t1), np.abs(p2 - t2))
    out["risk_to_expand_rank"] = np.where(out["y_attention"].astype(int).eq(1), out["boundary_distance"], np.inf)

    summary = (
        out.groupby(["split", "y_attention", "blocker_reason"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["split", "y_attention", "n"], ascending=[True, True, False])
    )
    positive_risk_cases = (
        out[out["y_attention"].astype(int).eq(1)]
        .sort_values(["boundary_distance", "p_max", "p_gap_abs"])
        .head(80)
    )
    normal_blocked_cases = (
        out[(out["y_attention"].astype(int).eq(0)) & (~out["selected_by_best_rule"])]
        .sort_values(["boundary_distance", "p_max", "p_gap_abs"])
        .head(120)
    )
    keep_cols = [
        "study_id",
        "split",
        "y_attention",
        "blocker_reason",
        "quality_score",
        "ood_score",
        "p_last1",
        "p_last2",
        "p_max",
        "p_mean",
        "p_gap_abs",
        "boundary_distance",
        "last1_low",
        "last2_low",
        "last1_veto_ok",
        "last2_veto_ok",
        "source_path",
        "image_eva_path",
    ]
    return summary, positive_risk_cases[keep_cols], normal_blocked_cases[keep_cols]


def render_report(val_selected: pd.DataFrame, final_selected: pd.DataFrame, paths_note: str) -> str:
    def fmt_table(df: pd.DataFrame, n: int = 12) -> str:
        cols = [
            "rule",
            "score",
            "auto_negative_coverage",
            "selected_count",
            "NPV",
            "FN_count",
            "t_negative",
            "t_ood",
            "t_uncertainty",
            "p_gap_max",
            "t_last1_negative",
            "t_last2_negative",
            "t_last1_veto",
            "t_last2_veto",
        ]
        cols = [col for col in cols if col in df.columns]
        table = df[cols].head(n).copy()
        for col in [
            "auto_negative_coverage",
            "NPV",
            "t_negative",
            "t_ood",
            "t_uncertainty",
            "p_gap_max",
            "t_last1_negative",
            "t_last2_negative",
            "t_last1_veto",
            "t_last2_veto",
        ]:
            if col not in table.columns:
                continue
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        return table.to_markdown(index=False)

    return (
        "# Deep Router Analysis\n\n"
        "## Validation-Selected Rules\n\n"
        f"{fmt_table(val_selected)}\n\n"
        "## Fixed Final-Test Results For Those Rules\n\n"
        f"{fmt_table(final_selected)}\n\n"
        "## Interpretation\n\n"
        "- Rules are selected only on validation. Final test is used as a fixed check.\n"
        "- `ensemble_both_low_agree` means both last1 and last2 must be below the same low-risk threshold and their probabilities must not strongly disagree.\n"
        "- `ensemble_one_low_other_veto` means one model can send a study to auto-negative if it is very low-risk and the second model remains below a veto risk.\n"
        "- If ensemble does not increase safe coverage, the bottleneck is likely model-score separation rather than a missing threshold trick.\n\n"
        "## Source\n\n"
        f"{paths_note}\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    available = available_case_scores()
    required = {"last1": {"validation", "final_test"}, "last2": {"validation", "final_test"}}
    missing = {
        alias: sorted(required_splits - available.get(alias, set()))
        for alias, required_splits in required.items()
        if required_splits - available.get(alias, set())
    }
    if missing:
        note = {
            "status": "blocked_until_case_scores_exist",
            "missing": missing,
            "how_to_fix": "Run: python3 scripts/export_eva_partial_case_scores.py --device auto --batch-size 4 --candidates last1 last2",
        }
        (OUT_DIR / "deep_router_blocked.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
        raise SystemExit(f"Missing case-score CSVs: {missing}")

    val = merge_last1_last2("validation")
    test = merge_last1_last2("final_test")
    val.to_csv(OUT_DIR / "merged_last1_last2_validation.csv", index=False)
    test.to_csv(OUT_DIR / "merged_last1_last2_final_test.csv", index=False)

    val_grid = candidate_rule_grid(val, "validation")
    val_grid.to_csv(OUT_DIR / "validation_deep_router_grid.csv", index=False)
    val_selected = select_validation_rules(val_grid)
    val_selected.to_csv(OUT_DIR / "validation_selected_zero_fn_rules.csv", index=False)

    final_selected = apply_rules_to_split(test, val_selected, "final_test")
    final_selected.to_csv(OUT_DIR / "final_test_fixed_deep_router_results.csv", index=False)

    near_threshold_cases(val, val_selected).to_csv(OUT_DIR / "near_threshold_cases_validation.csv", index=False)
    near_threshold_cases(test, val_selected).to_csv(OUT_DIR / "near_threshold_cases_final_test.csv", index=False)
    for split_name, split_df in [("validation", val), ("final_test", test)]:
        blocker_summary, positive_risk, normal_blocked = blocker_analysis(split_df, val_selected, split_name)
        blocker_summary.to_csv(OUT_DIR / f"router_blocker_summary_{split_name}.csv", index=False)
        positive_risk.to_csv(OUT_DIR / f"positive_boundary_risk_cases_{split_name}.csv", index=False)
        normal_blocked.to_csv(OUT_DIR / f"normal_blocked_near_boundary_cases_{split_name}.csv", index=False)

    report = render_report(
        val_selected,
        final_selected,
        paths_note=f"Case score source directory: `{CASE_DIR.relative_to(ROOT)}`.",
    )
    (OUT_DIR / "deep_router_analysis_report.md").write_text(report, encoding="utf-8")

    print("Deep router analysis written to:", OUT_DIR)
    print("Best validation rule:")
    print(val_selected.head(1).to_string(index=False))
    print("Fixed final-test result:")
    print(final_selected.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
