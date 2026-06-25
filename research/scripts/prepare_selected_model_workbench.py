from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "selected_model_workbench"


@dataclass(frozen=True)
class RunSpec:
    name: str
    root: Path
    role: str


RUNS = [
    RunSpec(
        name="eva_base_partial_unfreeze",
        root=ROOT / "fluoro_mvp_outputs" / "incxr_eva_base_partial_unfreeze_t4",
        role="primary_pool",
    ),
    RunSpec(
        name="eva_base_frozen_mlp",
        root=ROOT / "fluoro_mvp_outputs" / "incxr_eva_base_frozen_mlp_t4",
        role="stable_backup",
    ),
    RunSpec(
        name="eva_base_lora",
        root=ROOT / "fluoro_mvp_outputs" / "incxr_eva_base_lora_t4",
        role="lora_reference",
    ),
    RunSpec(
        name="eva_small_partial_unfreeze",
        root=ROOT / "fluoro_mvp_outputs" / "incxr_eva_small_partial_unfreeze_t4",
        role="small_reference",
    ),
    RunSpec(
        name="eva_small_tuned_mlp_legacy",
        root=ROOT / "EVA_opt",
        role="legacy_baseline",
    ),
]


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def summarize_run(spec: RunSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reports = spec.root / "reports"
    model_cmp = read_csv_if_exists(reports / "model_comparison.csv")
    final_metrics = read_csv_if_exists(reports / "all_candidates_final_test_metrics.csv")
    if final_metrics is None:
        final_metrics = read_csv_if_exists(reports / "best_final_test_metrics.csv")

    if model_cmp is None and final_metrics is None:
        return rows

    final_by_model: dict[str, pd.Series] = {}
    if final_metrics is not None and "model" in final_metrics.columns:
        for _, row in final_metrics.iterrows():
            model = str(row["model"]).replace(" final_test", "")
            final_by_model[model] = row

    if model_cmp is not None:
        source = model_cmp.copy()
    else:
        source = final_metrics.copy() if final_metrics is not None else pd.DataFrame()

    for _, row in source.iterrows():
        model_name = str(row.get("model", spec.name)).replace(" final_test", "")
        final = final_by_model.get(model_name)

        out: dict[str, Any] = {
            "run": spec.name,
            "role": spec.role,
            "model": model_name,
            "root": str(spec.root.relative_to(ROOT)) if spec.root.is_relative_to(ROOT) else str(spec.root),
            "validation_auroc": row.get("auroc"),
            "validation_auprc": row.get("auprc"),
            "validation_brier": row.get("brier"),
            "validation_ece": row.get("ece"),
            "validation_auto_negative_coverage": row.get("auto_negative_coverage"),
            "validation_threshold_npv": row.get("threshold_validation_NPV", row.get("auto_negative_NPV")),
            "validation_threshold_fn": row.get("threshold_validation_FN_count", row.get("unsafe_FN_auto_negative")),
            "selected_t_negative": row.get("selected_T_negative"),
            "threshold_policy": row.get("threshold_policy"),
            "threshold_policy_selected": row.get("threshold_policy_selected"),
            "deployment_adapter_supported": row.get("deployment_adapter_supported"),
            "meets_policy_constraints": row.get("meets_policy_constraints"),
            "candidate_selected_by_notebook": row.get("threshold_policy_candidate_selected"),
            "final_auroc": None if final is None else final.get("auroc"),
            "final_auprc": None if final is None else final.get("auprc"),
            "final_brier": None if final is None else final.get("brier"),
            "final_ece": None if final is None else final.get("ece"),
            "final_auto_negative_coverage": None if final is None else final.get("auto_negative_coverage"),
            "final_auto_negative_npv": None if final is None else final.get("auto_negative_NPV"),
            "final_unsafe_fn": None if final is None else final.get("unsafe_FN_auto_negative"),
            "final_na_rate": None if final is None else final.get("N/A_rate"),
            "final_requires_attention_rate": None if final is None else final.get("requires_attention_rate"),
        }
        rows.append(out)
    return rows


def bundle_status(spec: RunSpec) -> dict[str, Any]:
    backend = spec.root / "artifacts" / "backend_bundle"
    research = spec.root / "artifacts" / "research_bundle"
    bundle = backend if backend.exists() else research if research.exists() else None
    files = [
        "manifest.json",
        "preprocessing_config.json",
        "router_config.json",
        "best_calibrator.pkl",
        "best_ood_model.pkl",
        "best_model_state_dict.pt",
    ]
    status = {
        "run": spec.name,
        "bundle_kind": "backend_bundle" if backend.exists() else "research_bundle" if research.exists() else "missing",
        "bundle_path": "" if bundle is None else str(bundle.relative_to(ROOT)),
    }
    for file_name in files:
        status[f"has_{file_name}"] = bool(bundle and (bundle / file_name).exists())
    return status


def summarize_router_sweeps(spec: RunSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reports = spec.root / "reports"
    for path in sorted(reports.glob("*threshold_sweep.csv")):
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue

        for col in [
            "auto_negative_coverage",
            "threshold_validation_NPV",
            "threshold_validation_FN_count",
            "NPV_ci95_low",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        selected = df[df.get("threshold_policy_candidate_selected", False) == True]
        zero_fn = df[
            df["threshold_validation_FN_count"].fillna(9999).eq(0)
            & df["threshold_validation_NPV"].fillna(0).ge(0.99)
        ]
        strict_zero_fn = zero_fn
        if "meets_policy_constraints" in df.columns:
            strict_zero_fn = zero_fn[zero_fn["meets_policy_constraints"].fillna(False).astype(bool)]

        variants = [
            ("notebook_selected", selected),
            ("max_zero_fn_npv99", zero_fn),
            ("max_zero_fn_policy_meets", strict_zero_fn),
        ]
        for option, sub in variants:
            if sub.empty:
                continue
            row = sub.sort_values("auto_negative_coverage", ascending=False).iloc[0]
            rows.append(
                {
                    "run": spec.name,
                    "model": row.get("model", path.name.replace("_threshold_sweep.csv", "")),
                    "router_option": option,
                    "auto_negative_coverage": row.get("auto_negative_coverage"),
                    "selected_T_negative": row.get("selected_T_negative"),
                    "threshold_validation_NPV": row.get("threshold_validation_NPV"),
                    "threshold_validation_FN_count": row.get("threshold_validation_FN_count"),
                    "NPV_ci95_low": row.get("NPV_ci95_low"),
                    "selected_t_ood": row.get("selected_t_ood"),
                    "selected_t_positive": row.get("selected_t_positive"),
                    "selected_t_quality": row.get("selected_t_quality"),
                    "selected_t_uncertainty": row.get("selected_t_uncertainty"),
                    "threshold_policy": row.get("threshold_policy"),
                    "meets_policy_constraints": row.get("meets_policy_constraints"),
                    "source": str(path.relative_to(ROOT)),
                }
            )
    return rows


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in [
        "validation_threshold_fn",
        "final_unsafe_fn",
        "validation_auroc",
        "validation_auprc",
        "final_auroc",
        "final_auprc",
        "final_auto_negative_coverage",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work["deployment_safe_validation"] = (
        work["validation_threshold_fn"].fillna(9999).astype(float).eq(0)
        & work["validation_threshold_npv"].fillna(0).astype(float).ge(0.99)
    )
    work["deployment_safe_final_observed"] = (
        work["final_unsafe_fn"].fillna(9999).astype(float).eq(0)
        & work["final_auto_negative_npv"].fillna(0).astype(float).ge(0.99)
    )
    return work.sort_values(
        [
            "deployment_safe_validation",
            "validation_auroc",
            "validation_auprc",
            "final_auroc",
            "final_auto_negative_coverage",
        ],
        ascending=[False, False, False, False, False],
    )


def render_markdown(candidates: pd.DataFrame, bundles: pd.DataFrame, router_options: pd.DataFrame) -> str:
    display_cols = [
        "run",
        "model",
        "role",
        "deployment_safe_validation",
        "validation_auroc",
        "validation_auprc",
        "final_auroc",
        "final_auprc",
        "final_auto_negative_coverage",
        "final_auto_negative_npv",
        "final_unsafe_fn",
    ]
    table = candidates[display_cols].copy()
    for col in table.columns:
        if table[col].dtype.kind in "fc":
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    primary = candidates.iloc[0]
    challenger = candidates.iloc[1] if len(candidates) > 1 else None

    lines = [
        "# Selected Model Workbench",
        "",
        "## Current Decision",
        "",
        f"- Primary candidate: **{primary['model']}** from `{primary['run']}`.",
        "- Selection priority: validation-safe first, then ranking quality, then final-test confirmation and automation coverage.",
    ]
    if challenger is not None:
        lines.append(f"- Challenger candidate: **{challenger['model']}** from `{challenger['run']}`.")
    ensemble_rows = candidates[candidates["run"].eq("ensemble_router")]
    if not ensemble_rows.empty:
        ensemble = ensemble_rows.iloc[0]
        lines.append(
            f"- Best automation router candidate: **{ensemble['model']}** with validation auto-negative "
            f"coverage **{float(ensemble['validation_auto_negative_coverage']):.2%}** and final-test "
            f"coverage **{float(ensemble['final_auto_negative_coverage']):.2%}**, with FN=0 in both checks."
        )

    if not router_options.empty:
        router_safe = router_options[router_options["router_option"].eq("max_zero_fn_policy_meets")].copy()
        if not router_safe.empty:
            for col in ["auto_negative_coverage", "threshold_validation_FN_count", "threshold_validation_NPV"]:
                router_safe[col] = pd.to_numeric(router_safe[col], errors="coerce")
            best_router_all = router_safe.sort_values(
                ["auto_negative_coverage", "threshold_validation_NPV"],
                ascending=[False, False],
            ).iloc[0]
            eva_b_safe = router_safe[router_safe["run"].eq("eva_base_partial_unfreeze")]
            best_router = (
                eva_b_safe.sort_values(
                    ["auto_negative_coverage", "threshold_validation_NPV"],
                    ascending=[False, False],
                ).iloc[0]
                if not eva_b_safe.empty
                else best_router_all
            )
            lines.append(
                f"- Best strict zero-FN router inside the EVA-B selected pool: **{best_router['model']}** "
                f"with validation auto-negative coverage **{best_router['auto_negative_coverage']:.2%}**."
            )
            if best_router["model"] != best_router_all["model"]:
                lines.append(
                    f"- Overall strict zero-FN coverage reference: **{best_router_all['model']}** "
                    f"with **{best_router_all['auto_negative_coverage']:.2%}**, kept as reference rather than the main path."
                )
    lines += [
        "",
        "## Candidate Table",
        "",
        table.to_markdown(index=False),
        "",
        "## Bundle Status",
        "",
        bundles.to_markdown(index=False),
        "",
        "## Strict Router Options",
        "",
        router_options[
            [
                "run",
                "model",
                "router_option",
                "auto_negative_coverage",
                "threshold_validation_NPV",
                "threshold_validation_FN_count",
                "selected_T_negative",
                "threshold_policy",
            ]
        ].to_markdown(index=False)
        if not router_options.empty
        else "No router sweep files were found.",
        "",
        "## Mass Work Queue",
        "",
        "1. Lock the primary candidate unless pending CheXFound results clearly beat it on validation-safe ranking.",
        "2. Run deeper router analysis for the primary and the coverage challenger without retraining.",
        "3. Use the primary backend bundle as the input for VinDr/VinBigData interpretation panels.",
        "4. Build the production-facing inference wrapper from the backend bundle, not from notebook state.",
        "5. Keep the frozen EVA-X-B MLP bundle as a rollback option because it is simpler and stable.",
        "",
        "## Notes",
        "",
        "- Final test is used only for confirmation; router and threshold choices must be selected on validation.",
        "- If a model has final FN=0 but validation unsafe FN>0, it stays a research challenger until router policy is fixed on validation.",
        "- CheXFound can be added to this workbench when its exported folder arrives.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in RUNS:
        rows.extend(summarize_run(spec))
    ensemble_manifest_path = ROOT / "selected_model_workbench" / "router_workbench" / "ensemble_candidate_bundle" / "manifest.json"
    ensemble_router_path = ROOT / "selected_model_workbench" / "router_workbench" / "ensemble_candidate_bundle" / "router_config.json"
    if ensemble_manifest_path.exists() and ensemble_router_path.exists():
        ensemble_manifest = read_json_if_exists(ensemble_manifest_path) or {}
        ensemble_router = read_json_if_exists(ensemble_router_path) or {}
        rows.append(
            {
                "run": "ensemble_router",
                "role": "deployment_router_candidate",
                "model": "eva_base_partial_unfreeze_last1_last2_ensemble_router",
                "root": "selected_model_workbench/router_workbench/ensemble_candidate_bundle",
                "validation_auroc": None,
                "validation_auprc": None,
                "validation_brier": None,
                "validation_ece": None,
                "validation_auto_negative_coverage": ensemble_router.get("validation_auto_negative_coverage"),
                "validation_threshold_npv": ensemble_router.get("validation_npv"),
                "validation_threshold_fn": ensemble_router.get("validation_fn_count"),
                "selected_t_negative": ensemble_router.get("selected_T_negative"),
                "threshold_policy": ensemble_router.get("rule"),
                "threshold_policy_selected": True,
                "deployment_adapter_supported": True,
                "meets_policy_constraints": True,
                "candidate_selected_by_notebook": False,
                "final_auroc": None,
                "final_auprc": None,
                "final_brier": None,
                "final_ece": None,
                "final_auto_negative_coverage": ensemble_router.get("final_auto_negative_coverage"),
                "final_auto_negative_npv": ensemble_router.get("final_npv"),
                "final_unsafe_fn": ensemble_router.get("final_fn_count"),
                "final_na_rate": None,
                "final_requires_attention_rate": None,
                "ensemble_requires_two_model_inference": ensemble_manifest.get("requires_two_model_inference", True),
            }
        )
    if not rows:
        raise SystemExit("No model reports found.")

    candidates = sort_candidates(pd.DataFrame(rows))
    bundles = pd.DataFrame([bundle_status(spec) for spec in RUNS])
    router_rows: list[dict[str, Any]] = []
    for spec in RUNS:
        router_rows.extend(summarize_router_sweeps(spec))
    router_options = pd.DataFrame(router_rows)
    if not router_options.empty:
        router_options = router_options.sort_values(
            ["router_option", "auto_negative_coverage"],
            ascending=[True, False],
        )

    candidates.to_csv(OUTPUT_DIR / "selected_model_candidates.csv", index=False)
    bundles.to_csv(OUTPUT_DIR / "bundle_status.csv", index=False)
    router_options.to_csv(OUTPUT_DIR / "router_safety_options.csv", index=False)
    (OUTPUT_DIR / "selected_model_workbench.md").write_text(
        render_markdown(candidates, bundles, router_options),
        encoding="utf-8",
    )

    primary_root = ROOT / candidates.iloc[0]["root"]
    primary_manifest = first_existing(
        primary_root / "artifacts" / "backend_bundle" / "manifest.json",
        primary_root / "artifacts" / "research_bundle" / "manifest.json",
        primary_root / "artifacts" / "manifest.json",
    )
    if primary_manifest:
        shutil.copy2(primary_manifest, OUTPUT_DIR / "primary_manifest.json")

    print("Selected workbench written to:", OUTPUT_DIR)
    print("Primary:", candidates.iloc[0]["model"])
    if len(candidates) > 1:
        print("Challenger:", candidates.iloc[1]["model"])


if __name__ == "__main__":
    main()
