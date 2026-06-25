from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MASS_DIR = ROOT / "selected_model_workbench" / "mass_router_meta_analysis"
CASE_DIR = ROOT / "selected_model_workbench" / "case_scores"
OUT_DIR = ROOT / "selected_model_workbench" / "automation_frontier_analysis"


BASE_RISK_SCORES = [
    "p_chex_head",
    "p_chex_frozen",
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
]

TARGET_COVERAGES = [0.10, 0.15, 0.20, 0.25, 0.30]


def bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    return series.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy(bool)


def load_scores(split: str) -> pd.DataFrame:
    df = pd.read_csv(MASS_DIR / f"input_scores_{split}.csv", keep_default_na=False)
    df["critical_qa_bool"] = bool_array(df["critical_qa_bool"] if "critical_qa_bool" in df.columns else df["critical_qa"])
    df["p_pair_selected_router_soft"] = np.maximum(df["p_chex_head"].astype(float), df["p_last1"].astype(float))
    df["p_pair_selected_router_mean"] = df[["p_chex_head", "p_last1"]].astype(float).mean(axis=1)
    return df


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def score_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in BASE_RISK_SCORES + ["p_pair_selected_router_soft", "p_pair_selected_router_mean"] if c in df.columns]
    return list(dict.fromkeys(cols))


def frontier_for_score(df: pd.DataFrame, split: str, score_col: str, eligible: np.ndarray | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    y = df["y_attention"].to_numpy(int)
    p = df[score_col].to_numpy(float)
    if eligible is None:
        eligible = np.ones(len(df), dtype=bool)
    eligible_idx = np.flatnonzero(eligible)
    ordered = eligible_idx[np.argsort(p[eligible_idx], kind="mergesort")]

    zero_fn_count = 0
    first_positive_score = math.nan
    first_positive_study = ""
    for idx in ordered:
        if y[idx] == 1:
            first_positive_score = float(p[idx])
            first_positive_study = str(df.iloc[idx]["study_id"])
            break
        zero_fn_count += 1

    summary: dict[str, Any] = {
        "split": split,
        "score_col": score_col,
        "eligible_count": int(len(eligible_idx)),
        "auroc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "zero_fn_selected_count": int(zero_fn_count),
        "zero_fn_coverage_total": float(zero_fn_count / max(len(df), 1)),
        "zero_fn_coverage_eligible": float(zero_fn_count / max(len(eligible_idx), 1)),
        "first_positive_score": first_positive_score,
        "first_positive_study_id": first_positive_study,
    }

    rows = []
    for target in TARGET_COVERAGES:
        n_total = int(math.ceil(target * len(df)))
        n = min(n_total, len(ordered))
        selected = ordered[:n]
        fn = int(y[selected].sum())
        tn = int(((y[selected] == 0)).sum())
        rows.append(
            {
                "split": split,
                "score_col": score_col,
                "target_coverage": target,
                "selected_count": int(n),
                "selected_coverage": float(n / max(len(df), 1)),
                "TN_count": tn,
                "FN_count": fn,
                "NPV": float(tn / max(tn + fn, 1)),
                "threshold_at_target": float(p[selected[-1]]) if len(selected) else math.nan,
            }
        )
    return summary, pd.DataFrame(rows)


def current_selected_router_metrics() -> dict[str, Any]:
    cfg = json.loads((MASS_DIR / "selected_mass_router_config.json").read_text(encoding="utf-8"))
    return cfg["selected_fixed_final_metrics"]


def eligible_mask(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "all":
        return np.ones(len(df), dtype=bool)
    if mode == "current_quality_ood":
        return (
            (df["quality_score"].to_numpy(float) >= 0.25)
            & (~df["critical_qa_bool"].to_numpy(bool))
            & (df["ood_score_chex"].to_numpy(float) <= 1.10)
            & (df["ood_score_eva"].to_numpy(float) <= 1.25)
        )
    raise ValueError(mode)


def selected_positives_at_target(df: pd.DataFrame, score_col: str, target: float, mode: str) -> pd.DataFrame:
    p = df[score_col].to_numpy(float)
    y = df["y_attention"].to_numpy(int)
    eligible = eligible_mask(df, mode)
    idx = np.flatnonzero(eligible)
    ordered = idx[np.argsort(p[idx], kind="mergesort")]
    n = min(int(math.ceil(target * len(df))), len(ordered))
    selected = ordered[:n]
    out = df.iloc[selected].copy()
    out["frontier_score_col"] = score_col
    out["frontier_target_coverage"] = target
    out["frontier_mode"] = mode
    out["frontier_rank"] = np.arange(1, len(out) + 1)
    return out[out["y_attention"].astype(int).eq(1)].sort_values(score_col)


def load_source_map() -> pd.DataFrame:
    frames = []
    for alias in ["last1", "last2"]:
        for split in ["validation", "final_test"]:
            path = CASE_DIR / f"{alias}_{split}_case_scores.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, keep_default_na=False)
            df["image_file"] = df["source_path"].map(lambda x: Path(str(x)).name)
            frames.append(df[["image_file", "source_path"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates("image_file")


def read_gray(path: str | Path) -> Image.Image | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return ImageOps.autocontrast(Image.open(p).convert("L"))
    except Exception:
        return None


def render_positive_blockers(df: pd.DataFrame, out_path: Path, title: str, n: int = 12) -> None:
    source = load_source_map()
    panel = df.head(n).merge(source, on="image_file", how="left")
    rows = int(math.ceil(max(len(panel), 1) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, max(4, rows * 4.2)))
    axes = np.atleast_1d(axes).reshape(rows, 3)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (_, row) in enumerate(panel.iterrows()):
        ax = axes.ravel()[i]
        img = read_gray(row.get("source_path", ""))
        if img is not None:
            ax.imshow(img, cmap="gray")
        else:
            ax.text(0.5, 0.5, "image missing", ha="center", va="center")
        text = "\n".join(
            [
                f"target={float(row['frontier_target_coverage']):.0%}",
                f"rank={int(row['frontier_rank'])}",
                f"score={row['frontier_score_col']}",
                f"chex_head={float(row['p_chex_head']):.3f}",
                f"last1={float(row['p_last1']):.3f}",
                f"last2={float(row['p_last2']):.3f}",
                f"lora1={float(row['p_chex_lora1']):.3f}",
            ]
        )
        ax.text(
            0.01,
            0.01,
            text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.62, "pad": 3},
        )
        ax.set_title(str(row["image_file"])[:46], fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_frontier_plot(target_table: pd.DataFrame) -> None:
    plot_df = target_table[target_table["split"].eq("final_test") & target_table["mode"].eq("all")].copy()
    best_cols = (
        plot_df.groupby("score_col")["FN_count"].min().sort_values().head(8).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in best_cols:
        part = plot_df[plot_df["score_col"].eq(col)].sort_values("target_coverage")
        ax.plot(part["target_coverage"] * 100, part["FN_count"], marker="o", label=col)
    ax.axvline(20, color="gray", linestyle="--", linewidth=1)
    ax.axvline(30, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Target auto-negative coverage, %")
    ax.set_ylabel("FN count if selecting lowest-risk cases")
    ax.set_title("Current-score automation frontier on final test")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "automation_frontier_fn_curve.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    targets = []
    for split in ["validation", "final_test"]:
        df = load_scores(split)
        for mode in ["all", "current_quality_ood"]:
            eligible = eligible_mask(df, mode)
            for col in score_cols(df):
                summary, target_df = frontier_for_score(df, split, col, eligible=eligible)
                summary["mode"] = mode
                target_df["mode"] = mode
                summaries.append(summary)
                targets.append(target_df)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["split", "mode", "zero_fn_coverage_total", "auroc"],
        ascending=[True, True, False, False],
    )
    target_df = pd.concat(targets, ignore_index=True)
    target_best = (
        target_df[target_df["split"].eq("final_test")]
        .sort_values(["target_coverage", "FN_count", "NPV", "selected_count"], ascending=[True, True, False, False])
        .groupby(["mode", "target_coverage"], as_index=False)
        .head(5)
    )
    summary_df.to_csv(OUT_DIR / "score_zero_fn_frontier.csv", index=False)
    target_df.to_csv(OUT_DIR / "score_target_coverage_frontier.csv", index=False)
    target_best.to_csv(OUT_DIR / "best_scores_by_target_coverage_final_test.csv", index=False)

    final = load_scores("final_test")
    blockers = []
    for target in [0.20, 0.30]:
        # Use the empirically strongest low-risk score at each target.
        best = (
            target_df[
                target_df["split"].eq("final_test")
                & target_df["mode"].eq("all")
                & target_df["target_coverage"].eq(target)
            ]
            .sort_values(["FN_count", "NPV"], ascending=[True, False])
            .iloc[0]
        )
        selected_pos = selected_positives_at_target(final, str(best["score_col"]), target, "all")
        blockers.append(selected_pos)
        selected_pos.to_csv(OUT_DIR / f"positive_blockers_target_{int(target*100)}pct.csv", index=False)
        render_positive_blockers(
            selected_pos,
            OUT_DIR / f"positive_blockers_target_{int(target*100)}pct.png",
            f"Positive cases selected at {int(target*100)}% target with {best['score_col']}",
            n=12,
        )
    blocker_df = pd.concat(blockers, ignore_index=True) if blockers else pd.DataFrame()
    blocker_df.to_csv(OUT_DIR / "positive_blockers_20_30pct_combined.csv", index=False)
    write_frontier_plot(target_df)

    current = current_selected_router_metrics()
    current_table = pd.DataFrame([{
        "model": "current_deployment_router",
        "selected_count": current["selected_count"],
        "auto_negative_coverage": current["auto_negative_coverage"],
        "FN_count": current["FN_count"],
        "NPV": current["NPV"],
        "NPV_ci95_low": current["NPV_ci95_low"],
    }])
    current_table.to_csv(OUT_DIR / "current_router_reference.csv", index=False)

    final_best_20_30 = target_best[
        target_best["mode"].eq("all") & target_best["target_coverage"].isin([0.20, 0.30])
    ].copy()
    lines = [
        "# Automation Frontier Analysis",
        "",
        "Цель: понять, можно ли честно обещать 20-30% auto-negative на текущих score-моделях, просто меняя score/routing.",
        "",
        "## Current Router Reference",
        "",
        current_table.to_markdown(index=False),
        "",
        "## Best Zero-FN Frontiers By Score",
        "",
        summary_df[summary_df["split"].eq("final_test") & summary_df["mode"].eq("all")][
            ["score_col", "zero_fn_selected_count", "zero_fn_coverage_total", "auroc", "auprc", "first_positive_score", "first_positive_study_id"]
        ].head(12).to_markdown(index=False),
        "",
        "## Best Current Scores At 20-30% Target Coverage",
        "",
        final_best_20_30[
            ["target_coverage", "score_col", "selected_count", "FN_count", "NPV", "threshold_at_target"]
        ].head(20).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Текущий deployment-router дает 9.95% при FN=0.",
        "- На текущих score-ах 20% auto-negative почти неизбежно выбирает несколько positive cases даже в лучших одномерных ранжированиях.",
        "- 30% auto-negative на текущих score-ах выбирает уже двузначное число positive cases.",
        "- Значит рекламные 20-30% нельзя честно обеспечить простым порогом или score-mixer без нового подтверждения.",
        "- Два реалистичных пути: label/target review near-threshold positive cases или реальное улучшение image-level модели/adapter, которое поднимет эти positive cases выше low-risk зоны.",
        "",
        "## Files",
        "",
        "- `score_zero_fn_frontier.csv`",
        "- `score_target_coverage_frontier.csv`",
        "- `best_scores_by_target_coverage_final_test.csv`",
        "- `positive_blockers_target_20pct.csv`",
        "- `positive_blockers_target_30pct.csv`",
        "- `automation_frontier_fn_curve.png`",
    ]
    (OUT_DIR / "automation_frontier_analysis_report_ru.md").write_text("\n".join(lines), encoding="utf-8")

    compact = OUT_DIR / "export_compact"
    if compact.exists():
        shutil.rmtree(compact)
    compact.mkdir(parents=True)
    for name in [
        "automation_frontier_analysis_report_ru.md",
        "score_zero_fn_frontier.csv",
        "score_target_coverage_frontier.csv",
        "best_scores_by_target_coverage_final_test.csv",
        "positive_blockers_target_20pct.csv",
        "positive_blockers_target_30pct.csv",
        "positive_blockers_20_30pct_combined.csv",
        "current_router_reference.csv",
        "automation_frontier_fn_curve.png",
        "positive_blockers_target_20pct.png",
        "positive_blockers_target_30pct.png",
    ]:
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, compact / name)
    archive_base = OUT_DIR / "automation_frontier_analysis_export"
    if archive_base.with_suffix(".zip").exists():
        archive_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(archive_base), "zip", root_dir=compact)

    print("Current:", current_table.to_string(index=False))
    print("Best 20/30% target rows:")
    print(final_best_20_30[["target_coverage", "score_col", "selected_count", "FN_count", "NPV", "threshold_at_target"]].head(20).to_string(index=False))
    print("Saved:", OUT_DIR)
    print("Archive:", archive_base.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
