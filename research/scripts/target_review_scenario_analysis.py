from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRONTIER_DIR = ROOT / "selected_model_workbench" / "automation_frontier_analysis"
AUDIT_DIR = ROOT / "selected_model_workbench" / "data_label_audit"
OUT_DIR = ROOT / "selected_model_workbench" / "target_review_scenario_analysis"


def load_target_blockers(target: int) -> pd.DataFrame:
    path = FRONTIER_DIR / f"positive_blockers_target_{target}pct.csv"
    df = pd.read_csv(path, keep_default_na=False)
    return df.sort_values(["frontier_rank", "p_chex_head", "p_last1"]).reset_index(drop=True)


def review_template() -> pd.DataFrame:
    frames = []
    for target in [20, 30]:
        df = load_target_blockers(target)
        df["review_target_coverage"] = target
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(["study_id", "review_target_coverage"])
    template = all_df[
        [
            "review_target_coverage",
            "study_id",
            "image_file",
            "frontier_rank",
            "frontier_score_col",
            "p_chex_head",
            "p_last1",
            "p_last2",
            "p_chex_lora1",
            "p_chex_frozen",
            "ood_score_chex",
            "ood_score_eva",
        ]
    ].copy()
    template["current_label"] = "requires_attention"
    template["review_decision"] = ""
    template["reviewer_comment"] = ""
    template["recommended_question"] = "Does this case truly require attention in the product screening target?"
    return template


def scenario_table() -> pd.DataFrame:
    rows = []
    target_info = {
        20: {"selected": 252, "initial_fn": 4},
        30: {"selected": 377, "initial_fn": 13},
    }
    for target, info in target_info.items():
        selected = info["selected"]
        initial_fn = info["initial_fn"]
        for reviewed_as_not_attention in range(0, initial_fn + 1):
            remaining_fn = initial_fn - reviewed_as_not_attention
            tn_effective = selected - remaining_fn
            npv = tn_effective / selected
            rows.append(
                {
                    "target_coverage": target / 100,
                    "selected_count": selected,
                    "initial_positive_blockers": initial_fn,
                    "positive_blockers_reclassified_as_not_attention": reviewed_as_not_attention,
                    "remaining_FN_under_original_label": remaining_fn,
                    "effective_NPV_after_target_review": npv,
                    "deployment_safe_if_review_accepted": bool(remaining_fn == 0),
                }
            )
    return pd.DataFrame(rows)


def write_report(template: pd.DataFrame, scenarios: pd.DataFrame) -> None:
    lines = [
        "# Target Review Scenario Analysis",
        "",
        "Этот анализ не меняет разметку автоматически. Он показывает, сколько near-threshold positive cases нужно врачебно пересмотреть, чтобы честно заявлять 20-30% auto-negative.",
        "",
        "## Required Review Volume",
        "",
        "- Для 20% auto-negative текущие score-модели выбирают 4 positive blockers.",
        "- Для 30% auto-negative текущие score-модели выбирают 13 positive blockers.",
        "- Если все эти blockers клинически действительно требуют внимания, 20-30% нельзя честно обеспечить без улучшения image-level модели.",
        "- Если часть blockers по продуктовому таргету не требует внимания, claim можно уточнить только после документированного review.",
        "",
        "## Scenario Table",
        "",
        scenarios.to_markdown(index=False),
        "",
        "## Review Template Preview",
        "",
        template.head(20).to_markdown(index=False),
        "",
        "## How To Use",
        "",
        "1. Врач/эксперт заполняет `review_decision`: `requires_attention`, `not_attention`, `uncertain`.",
        "2. Мы пересчитываем target definition и frontier только после review.",
        "3. Без такого review безопасный публичный claim остается около 10%, а 20-30% является исследовательской целью.",
    ]
    (OUT_DIR / "target_review_scenario_report_ru.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    template = review_template()
    scenarios = scenario_table()
    template.to_csv(OUT_DIR / "target_review_template.csv", index=False)
    scenarios.to_csv(OUT_DIR / "target_review_scenarios.csv", index=False)
    write_report(template, scenarios)

    # Include already rendered panels in the compact packet so a reviewer can open one folder.
    compact = OUT_DIR / "export_compact"
    if compact.exists():
        shutil.rmtree(compact)
    compact.mkdir(parents=True)
    for name in ["target_review_template.csv", "target_review_scenarios.csv", "target_review_scenario_report_ru.md"]:
        shutil.copy2(OUT_DIR / name, compact / name)
    for name in [
        "positive_blockers_target_20pct.png",
        "positive_blockers_target_30pct.png",
        "automation_frontier_fn_curve.png",
    ]:
        src = FRONTIER_DIR / name
        if src.exists():
            shutil.copy2(src, compact / name)
    for name in ["aggressive_lora_fn_cases.png"]:
        src = AUDIT_DIR / "panels" / name
        if src.exists():
            shutil.copy2(src, compact / name)
    archive_base = OUT_DIR / "target_review_scenario_export"
    if archive_base.with_suffix(".zip").exists():
        archive_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(archive_base), "zip", root_dir=compact)
    print("Review rows:", len(template))
    print(scenarios.to_string(index=False))
    print("Saved:", OUT_DIR)
    print("Archive:", archive_base.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
