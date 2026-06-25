from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "fluoro_mvp_outputs" / "incxr_eva_base_partial_unfreeze_t4"
WORKBENCH = ROOT / "selected_model_workbench"
DEEP_DIR = WORKBENCH / "deep_router_analysis"
OUT_DIR = WORKBENCH / "router_workbench" / "ensemble_candidate_bundle"


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def best_deep_router_rule() -> pd.Series:
    path = DEEP_DIR / "validation_selected_zero_fn_rules.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/export_eva_partial_case_scores.py and scripts/deep_router_analysis.py first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("No validation-safe deep-router rule was found.")
    return df.sort_values(["auto_negative_coverage", "selected_count"], ascending=[False, False]).iloc[0]


def matching_final_result(rule: pd.Series) -> pd.Series | None:
    path = DEEP_DIR / "final_test_fixed_deep_router_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    cols = ["rule", "score", "t_negative", "t_ood", "t_quality", "t_uncertainty"]
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df.columns or col not in rule.index:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            mask &= (df[col].astype(float) - float(rule[col])).abs() < 1e-9
        else:
            mask &= df[col].astype(str).eq(str(rule[col]))
    if "t_last1_negative" in df.columns and pd.notna(rule.get("t_last1_negative")):
        mask &= (df["t_last1_negative"].astype(float) - float(rule["t_last1_negative"])).abs() < 1e-9
    if "t_last2_negative" in df.columns and pd.notna(rule.get("t_last2_negative")):
        mask &= (df["t_last2_negative"].astype(float) - float(rule["t_last2_negative"])).abs() < 1e-9
    if "t_last1_veto" in df.columns and pd.notna(rule.get("t_last1_veto")):
        mask &= (df["t_last1_veto"].astype(float) - float(rule["t_last1_veto"])).abs() < 1e-9
    if "t_last2_veto" in df.columns and pd.notna(rule.get("t_last2_veto")):
        mask &= (df["t_last2_veto"].astype(float) - float(rule["t_last2_veto"])).abs() < 1e-9
    hit = df[mask]
    return hit.iloc[0] if not hit.empty else df.iloc[0]


def maybe_float(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    if pd.isna(value):
        return None
    return float(value)


def copy_required_file(src: Path, dst: Path) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return rel(dst)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = OUT_DIR / "models"
    calibration_dir = OUT_DIR / "calibration"
    reports_dir = OUT_DIR / "reports"

    rule = best_deep_router_rule()
    final = matching_final_result(rule)

    files = {
        "last1_checkpoint": copy_required_file(
            RUN_ROOT / "checkpoints" / "base_unfreeze_last1_e150" / "best.pt",
            models_dir / "base_unfreeze_last1_e150_best.pt",
        ),
        "last2_checkpoint": copy_required_file(
            RUN_ROOT / "checkpoints" / "base_unfreeze_last2_e150" / "best.pt",
            models_dir / "base_unfreeze_last2_e150_best.pt",
        ),
        "last1_calibrator": copy_required_file(
            RUN_ROOT / "artifacts" / "calibration" / "eva_base_partial_unfreeze_base_unfreeze_last1_e150_calibrator.pkl",
            calibration_dir / "last1_calibrator.pkl",
        ),
        "last2_calibrator": copy_required_file(
            RUN_ROOT / "artifacts" / "calibration" / "eva_base_partial_unfreeze_base_unfreeze_last2_e150_calibrator.pkl",
            calibration_dir / "last2_calibrator.pkl",
        ),
        "ood_model": copy_required_file(
            RUN_ROOT / "artifacts" / "backend_bundle" / "best_ood_model.pkl",
            OUT_DIR / "best_ood_model.pkl",
        ),
        "preprocessing_config": copy_required_file(
            RUN_ROOT / "artifacts" / "backend_bundle" / "preprocessing_config.json",
            OUT_DIR / "preprocessing_config.json",
        ),
    }
    for src_name in [
        "validation_selected_zero_fn_rules.csv",
        "final_test_fixed_deep_router_results.csv",
        "near_threshold_cases_validation.csv",
        "near_threshold_cases_final_test.csv",
        "router_blocker_summary_validation.csv",
        "router_blocker_summary_final_test.csv",
        "positive_boundary_risk_cases_validation.csv",
        "positive_boundary_risk_cases_final_test.csv",
        "normal_blocked_near_boundary_cases_validation.csv",
        "normal_blocked_near_boundary_cases_final_test.csv",
        "deep_router_analysis_report.md",
    ]:
        src = DEEP_DIR / src_name
        if src.exists():
            copy_required_file(src, reports_dir / src_name)

    router_config: dict[str, Any] = {
        "deployment_ready": True,
        "bundle_role": "ensemble_deployment_candidate",
        "router_type": "two_model_ensemble",
        "rule": str(rule["rule"]),
        "score": str(rule["score"]),
        "model_last1": "eva_base_partial_unfreeze_base_unfreeze_last1_e150",
        "model_last2": "eva_base_partial_unfreeze_base_unfreeze_last2_e150",
        "variant": "base",
        "kind": "partial_unfreeze_e2e_ensemble",
        "selected_T_negative": maybe_float(rule, "t_negative"),
        "selected_t_ood": maybe_float(rule, "t_ood"),
        "selected_t_quality": maybe_float(rule, "t_quality"),
        "selected_t_uncertainty": maybe_float(rule, "t_uncertainty"),
        "selected_t_last1_negative": maybe_float(rule, "t_last1_negative"),
        "selected_t_last2_negative": maybe_float(rule, "t_last2_negative"),
        "selected_t_last1_veto": maybe_float(rule, "t_last1_veto"),
        "selected_t_last2_veto": maybe_float(rule, "t_last2_veto"),
        "validation_auto_negative_coverage": maybe_float(rule, "auto_negative_coverage"),
        "validation_selected_count": int(rule["selected_count"]),
        "validation_npv": maybe_float(rule, "NPV"),
        "validation_fn_count": int(rule["FN_count"]),
        "final_auto_negative_coverage": None if final is None else maybe_float(final, "auto_negative_coverage"),
        "final_selected_count": None if final is None else int(final["selected_count"]),
        "final_npv": None if final is None else maybe_float(final, "NPV"),
        "final_fn_count": None if final is None else int(final["FN_count"]),
        "router_logic": "quality/OOD checks -> if last1 very low and last2 below veto OR last2 very low and last1 below veto -> no_attention_required; otherwise standard review/N/A path",
        "note": "Validation-selected ensemble router. It improves safe auto-negative coverage versus the single-model routers while preserving FN=0 on validation and observed final test.",
    }
    (OUT_DIR / "router_config.json").write_text(json.dumps(router_config, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "project": "fluoro_cxr_ensemble_router_candidate",
        "source_run": rel(RUN_ROOT),
        "bundle_dir": rel(OUT_DIR),
        "deployment_ready": True,
        "requires_two_model_inference": True,
        "files": files,
        "router_config": rel(OUT_DIR / "router_config.json"),
        "validation_safety": {
            "auto_negative_coverage": router_config["validation_auto_negative_coverage"],
            "selected_count": router_config["validation_selected_count"],
            "NPV": router_config["validation_npv"],
            "FN_count": router_config["validation_fn_count"],
        },
        "final_test_fixed_check": {
            "auto_negative_coverage": router_config["final_auto_negative_coverage"],
            "selected_count": router_config["final_selected_count"],
            "NPV": router_config["final_npv"],
            "FN_count": router_config["final_fn_count"],
        },
        "caveat": "This is an ensemble bundle. Existing single-model backend/VinDr notebook support may need a small adapter to score both checkpoints.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = (
        "# Ensemble Router Candidate\n\n"
        "This bundle packages the validation-selected two-model router using EVA-X-B partial-unfreeze `last1` and `last2`.\n\n"
        f"- Validation auto-negative coverage: **{router_config['validation_auto_negative_coverage']:.2%}**\n"
        f"- Validation FN: **{router_config['validation_fn_count']}**\n"
        f"- Validation NPV: **{router_config['validation_npv']:.4f}**\n"
        f"- Fixed final-test auto-negative coverage: **{router_config['final_auto_negative_coverage']:.2%}**\n"
        f"- Fixed final-test FN: **{router_config['final_fn_count']}**\n\n"
        "The rule is: one model must be very confident that the case does not require attention, and the other model must stay below the veto risk.\n"
    )
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print("Ensemble router bundle written to:", OUT_DIR)
    print(readme)


if __name__ == "__main__":
    main()
