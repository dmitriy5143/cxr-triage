from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "fluoro_mvp_outputs" / "incxr_eva_base_partial_unfreeze_t4"
REPORTS_DIR = RUN_ROOT / "reports"
ARTIFACTS_DIR = RUN_ROOT / "artifacts"
CHECKPOINTS_DIR = RUN_ROOT / "checkpoints"
OUT_DIR = ROOT / "selected_model_workbench" / "router_workbench"
ROUTER_CONFIG_DIR = OUT_DIR / "router_configs"
PRIMARY_BUNDLE_DIR = OUT_DIR / "primary_deployment_bundle"


MODEL_LAST1 = "eva_base_partial_unfreeze_base_unfreeze_last1_e150"
MODEL_LAST2 = "eva_base_partial_unfreeze_base_unfreeze_last2_e150"


def load_policy_summary(model_name: str) -> pd.DataFrame:
    path = REPORTS_DIR / f"{model_name}_router_policy_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def choose_policy(df: pd.DataFrame, policy: str) -> pd.Series:
    selected = df[df["threshold_policy"].eq(policy)].copy()
    if selected.empty:
        raise ValueError(f"Policy {policy!r} was not found for {df['model'].iloc[0]}.")
    selected["auto_negative_coverage"] = pd.to_numeric(selected["auto_negative_coverage"], errors="coerce")
    return selected.sort_values("auto_negative_coverage", ascending=False).iloc[0]


def row_to_router_config(row: pd.Series, *, deployment_ready: bool, bundle_role: str, note: str) -> dict[str, Any]:
    def maybe_float(key: str) -> float | None:
        value = row.get(key)
        if pd.isna(value):
            return None
        return float(value)

    def maybe_int(key: str) -> int | None:
        value = row.get(key)
        if pd.isna(value):
            return None
        return int(round(float(value)))

    return {
        "deployment_ready": deployment_ready,
        "bundle_role": bundle_role,
        "model": row.get("model"),
        "variant": row.get("variant"),
        "kind": row.get("kind"),
        "calibration_method": row.get("calibration_method"),
        "selected_T_negative": maybe_float("selected_T_negative"),
        "selected_t_ood": maybe_float("selected_t_ood"),
        "selected_t_positive": maybe_float("selected_t_positive"),
        "selected_t_quality": maybe_float("selected_t_quality"),
        "selected_t_uncertainty": maybe_float("selected_t_uncertainty"),
        "threshold_policy": row.get("threshold_policy"),
        "validation_auto_negative_coverage": maybe_float("auto_negative_coverage"),
        "validation_npv": maybe_float("threshold_validation_NPV"),
        "validation_fn_count": maybe_int("threshold_validation_FN_count"),
        "npv_ci95_low": maybe_float("NPV_ci95_low"),
        "validation_selected_count": maybe_int("threshold_selected_count"),
        "validation_tn_count": maybe_int("threshold_validation_TN_count"),
        "meets_policy_constraints": bool(row.get("meets_policy_constraints")),
        "router_logic": "quality/OOD checks -> T_negative no_attention_required -> T_positive requires_attention -> gray-zone N/A",
        "note": note,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_primary_bundle() -> None:
    source = ARTIFACTS_DIR / "backend_bundle"
    if not source.exists():
        raise FileNotFoundError(source)
    if PRIMARY_BUNDLE_DIR.exists():
        shutil.rmtree(PRIMARY_BUNDLE_DIR)
    shutil.copytree(source, PRIMARY_BUNDLE_DIR)
    strict_router = ROUTER_CONFIG_DIR / "primary_last1_deployment_zero_fn_router.json"
    shutil.copy2(strict_router, PRIMARY_BUNDLE_DIR / "router_config.json")
    manifest_path = PRIMARY_BUNDLE_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["backend_bundle"] = str(PRIMARY_BUNDLE_DIR.relative_to(ROOT))
        manifest["workbench_source_bundle"] = str(source.relative_to(ROOT))
        manifest["router_config"] = str((PRIMARY_BUNDLE_DIR / "router_config.json").relative_to(ROOT))
        manifest["workbench_note"] = (
            "Copied from the original backend bundle and rewritten by router_workbench "
            "to use the explicit primary strict zero-FN router."
        )
        write_json(manifest_path, manifest)


def make_research_last2_manifest(strict_config: dict[str, Any], aggressive_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_role": "research_challenger",
        "deployment_ready": False,
        "reason": "last2 has a strict zero-FN router candidate, but the notebook-selected aggressive policy had 1 unsafe FN on validation. Promote only after fixed validation-safe export.",
        "model": MODEL_LAST2,
        "checkpoint_best": str((CHECKPOINTS_DIR / "base_unfreeze_last2_e150" / "best.pt").relative_to(ROOT)),
        "checkpoint_final": str((CHECKPOINTS_DIR / "base_unfreeze_last2_e150_final.pt").relative_to(ROOT)),
        "calibrator": str((ARTIFACTS_DIR / "calibration" / f"{MODEL_LAST2}_calibrator.pkl").relative_to(ROOT)),
        "strict_zero_fn_router": strict_config,
        "aggressive_research_router": aggressive_config,
    }


def render_report(configs: list[dict[str, Any]]) -> str:
    rows = []
    for cfg in configs:
        rows.append(
            {
                "model": cfg["model"],
                "role": cfg["bundle_role"],
                "deployment_ready": cfg["deployment_ready"],
                "policy": cfg["threshold_policy"],
                "coverage": cfg["validation_auto_negative_coverage"],
                "NPV": cfg["validation_npv"],
                "FN": cfg["validation_fn_count"],
                "T_negative": cfg["selected_T_negative"],
                "OOD": cfg["selected_t_ood"],
            }
        )
    table = pd.DataFrame(rows)
    for col in ["coverage", "NPV", "T_negative", "OOD"]:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")

    return (
        "# Router Workbench\n\n"
        "## Decision\n\n"
        "- `last1` remains the main deployment bundle because it is validation-safe and strongest by final ranking quality.\n"
        "- `last2` is the coverage challenger. Its strict zero-FN router is useful for follow-up, while its aggressive router stays research-only.\n"
        "- No final-test threshold retuning is done here; this workbench only packages validation-selected routing policies.\n\n"
        "## Router Configs\n\n"
        f"{table.to_markdown(index=False)}\n\n"
        "## Files\n\n"
        "- `router_configs/primary_last1_deployment_zero_fn_router.json`\n"
        "- `router_configs/challenger_last2_strict_zero_fn_router.json`\n"
        "- `router_configs/challenger_last2_aggressive_research_router.json`\n"
        "- `primary_deployment_bundle/` copied from the backend bundle with the explicit primary router config.\n"
        "- `challenger_last2_research_manifest.json` points to the last2 checkpoint and calibrator without declaring it deployment-ready.\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROUTER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    last1 = load_policy_summary(MODEL_LAST1)
    last2 = load_policy_summary(MODEL_LAST2)

    primary_row = choose_policy(last1, "zero_fn_cap_08pct_ood110")
    last2_strict_row = choose_policy(last2, "zero_fn_cap_08pct_ood110")
    last2_aggressive_row = choose_policy(last2, "target_npv_max_coverage")

    primary_cfg = row_to_router_config(
        primary_row,
        deployment_ready=True,
        bundle_role="primary_deployment",
        note="Main MVP candidate: strict zero-FN validation routing with strong final-test ranking.",
    )
    last2_strict_cfg = row_to_router_config(
        last2_strict_row,
        deployment_ready=False,
        bundle_role="coverage_challenger_strict",
        note="Coverage challenger: validation FN=0, but requires a separate backend export before deployment.",
    )
    last2_aggressive_cfg = row_to_router_config(
        last2_aggressive_row,
        deployment_ready=False,
        bundle_role="coverage_challenger_research_only",
        note="Research-only aggressive policy: higher coverage but 1 unsafe FN on validation.",
    )

    configs = [
        ("primary_last1_deployment_zero_fn_router.json", primary_cfg),
        ("challenger_last2_strict_zero_fn_router.json", last2_strict_cfg),
        ("challenger_last2_aggressive_research_router.json", last2_aggressive_cfg),
    ]
    for file_name, payload in configs:
        write_json(ROUTER_CONFIG_DIR / file_name, payload)

    write_json(
        OUT_DIR / "challenger_last2_research_manifest.json",
        make_research_last2_manifest(last2_strict_cfg, last2_aggressive_cfg),
    )
    copy_primary_bundle()
    (OUT_DIR / "router_workbench_report.md").write_text(
        render_report([payload for _, payload in configs]),
        encoding="utf-8",
    )

    print("Router workbench written to:", OUT_DIR)
    print("Primary deployment router:", ROUTER_CONFIG_DIR / configs[0][0])
    print("Last2 strict challenger router:", ROUTER_CONFIG_DIR / configs[1][0])


if __name__ == "__main__":
    main()
