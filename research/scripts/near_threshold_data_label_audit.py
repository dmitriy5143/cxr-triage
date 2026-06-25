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


ROOT = Path(__file__).resolve().parents[1]
MASS_DIR = ROOT / "selected_model_workbench" / "mass_router_meta_analysis"
CASE_DIR = ROOT / "selected_model_workbench" / "case_scores"
OUT_DIR = ROOT / "selected_model_workbench" / "data_label_audit"


SCORE_COLS = ["p_chex_head", "p_last1", "p_last2", "p_chex_frozen", "p_chex_lora1", "p_chex_lora2"]


def bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    return series.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy(bool)


def load_scores(split: str) -> pd.DataFrame:
    df = pd.read_csv(MASS_DIR / f"input_scores_{split}.csv", keep_default_na=False)
    df["critical_qa_bool"] = bool_array(df["critical_qa_bool"] if "critical_qa_bool" in df.columns else df["critical_qa"])
    return df


def load_source_map() -> pd.DataFrame:
    frames = []
    for alias in ["last1", "last2"]:
        for split in ["validation", "final_test"]:
            path = CASE_DIR / f"{alias}_{split}_case_scores.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, keep_default_na=False)
            df["image_file"] = df["source_path"].map(lambda x: Path(str(x)).name)
            keep = ["image_file", "source_path", "image_eva_path"]
            frames.append(df[keep])
    if not frames:
        raise FileNotFoundError("No EVA case score files found for source_path mapping.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("image_file")


def read_gray(path: str | Path) -> Image.Image | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        img = Image.open(p).convert("L")
        return ImageOps.autocontrast(img)
    except Exception:
        return None


def row_text(row: pd.Series) -> str:
    parts = [
        f"label={int(row.get('y_attention', -1))}",
        f"block={row.get('blocker_reason', '')}",
        f"q={float(row.get('quality_score', np.nan)):.2f}",
        f"oodC={float(row.get('ood_score_chex', np.nan)):.2f}",
        f"oodE={float(row.get('ood_score_eva', np.nan)):.2f}",
    ]
    for col in ["p_chex_head", "p_last1", "p_last2", "p_chex_lora1"]:
        if col in row:
            parts.append(f"{col.replace('p_', '')}={float(row[col]):.3f}")
    return "\n".join(parts)


def render_panel(df: pd.DataFrame, title: str, out_path: Path, source_map: pd.DataFrame, n: int = 12) -> None:
    panel = df.head(n).copy()
    panel = panel.merge(source_map, on="image_file", how="left")
    rows = int(math.ceil(len(panel) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, max(4, rows * 4.2)))
    axes_arr = np.atleast_1d(axes).reshape(rows, 3)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for i, (_, row) in enumerate(panel.iterrows()):
        ax = axes_arr.ravel()[i]
        img = read_gray(row.get("source_path", ""))
        if img is None:
            ax.text(0.5, 0.55, "image not found", ha="center", va="center", fontsize=11)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(str(row["image_file"])[:46], fontsize=9)
        ax.text(
            0.01,
            0.01,
            row_text(row),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.62, "pad": 3},
        )
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def selected_router_mask(df: pd.DataFrame) -> np.ndarray:
    cfg = json.loads((MASS_DIR / "selected_mass_router_config.json").read_text(encoding="utf-8"))
    r = cfg["selected_validation_rule"]
    a = r["model_a"]
    b = r["model_b"]
    av = df[a].to_numpy(float)
    bv = df[b].to_numpy(float)
    uncertainty = np.maximum(1.0 - np.abs(av - 0.5) * 2.0, 1.0 - np.abs(bv - 0.5) * 2.0)
    return (
        (df["quality_score"].to_numpy(float) >= float(r["t_quality"]))
        & (~df["critical_qa_bool"].to_numpy(bool))
        & (df["ood_score_chex"].to_numpy(float) <= float(r["t_ood_chex"]))
        & (df["ood_score_eva"].to_numpy(float) <= float(r["t_ood_eva"]))
        & (uncertainty <= float(r["t_uncertainty"]))
        & (
            ((av <= float(r["t_a_negative"])) & (bv <= float(r["t_b_veto"])))
            | ((bv <= float(r["t_b_negative"])) & (av <= float(r["t_a_veto"])))
        )
    )


def aggressive_lora_fn_cases(final: pd.DataFrame) -> pd.DataFrame:
    fixed = pd.read_csv(MASS_DIR / "fixed_final_for_validation_safe_rules.csv", keep_default_na=False)
    aggressive = fixed.iloc[0].to_dict()
    a = str(aggressive["model_a"])
    b = str(aggressive["model_b"])
    av = final[a].to_numpy(float)
    bv = final[b].to_numpy(float)
    uncertainty = np.maximum(1.0 - np.abs(av - 0.5) * 2.0, 1.0 - np.abs(bv - 0.5) * 2.0)
    mask = (
        (final["quality_score"].to_numpy(float) >= float(aggressive["t_quality"]))
        & (~final["critical_qa_bool"].to_numpy(bool))
        & (final["ood_score_chex"].to_numpy(float) <= float(aggressive["t_ood_chex"]))
        & (final["ood_score_eva"].to_numpy(float) <= float(aggressive["t_ood_eva"]))
        & (uncertainty <= float(aggressive["t_uncertainty"]))
        & (
            ((av <= float(aggressive["t_a_negative"])) & (bv <= float(aggressive["t_b_veto"])))
            | ((bv <= float(aggressive["t_b_negative"])) & (av <= float(aggressive["t_a_veto"])))
        )
    )
    out = final[mask & final["y_attention"].astype(int).eq(1)].copy()
    out["blocker_reason"] = "aggressive_lora_false_negative"
    return out


def split_leakage_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx_path = ROOT / "CheXFound_frozen" / "data_index.parquet"
    idx = pd.read_parquet(idx_path)
    idx["image_file"] = idx["path"].map(lambda x: Path(str(x)).name)
    split_counts = idx.groupby(["split", "y_attention"]).size().rename("n").reset_index()
    leak_rows = []
    for col in ["content_sha256", "patient_id_hash", "image_file"]:
        grouped = idx.groupby(col)["split"].agg(lambda s: "|".join(sorted(set(map(str, s)))))
        leaked = grouped[grouped.str.contains("\\|", regex=True)]
        leak_rows.append({"key": col, "cross_split_duplicate_keys": int(len(leaked))})
    leakage = pd.DataFrame(leak_rows)
    return split_counts, leakage


def score_distribution_table(df: pd.DataFrame, split: str, selected: np.ndarray) -> pd.DataFrame:
    tmp = df.copy()
    tmp["selected_auto_negative"] = selected
    rows = []
    for group_name, group_df in tmp.groupby(["y_attention", "selected_auto_negative"], dropna=False):
        y, sel = group_name
        row: dict[str, Any] = {"split": split, "y_attention": int(y), "selected_auto_negative": bool(sel), "n": int(len(group_df))}
        for col in SCORE_COLS:
            row[f"{col}_median"] = float(group_df[col].median())
            row[f"{col}_p10"] = float(group_df[col].quantile(0.10))
            row[f"{col}_p90"] = float(group_df[col].quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    split_counts: pd.DataFrame,
    leakage: pd.DataFrame,
    final_route_metrics: dict[str, Any],
    blocker_summary: pd.DataFrame,
    aggressive_fns: pd.DataFrame,
) -> None:
    report = [
        "# Near-Threshold Data/Label Audit",
        "",
        "Аудит сфокусирован на зоне, которая ограничивает рост `auto-negative`: positive near-threshold, normal near-threshold, rejected aggressive LoRA cases и split integrity.",
        "",
        "## Dataset Contract",
        "",
        split_counts.to_markdown(index=False),
        "",
        "## Split Leakage Checks",
        "",
        leakage.to_markdown(index=False),
        "",
        "Ноль по `content_sha256`, `patient_id_hash` и `image_file` между split означает, что явной утечки по этим ключам не найдено.",
        "",
        "## Selected Router Final-Test Check",
        "",
        pd.DataFrame([final_route_metrics]).to_markdown(index=False),
        "",
        "## Final-Test Blockers",
        "",
        blocker_summary.to_markdown(index=False),
        "",
        "## Aggressive LoRA Rejected Cases",
        "",
        "Эти positive cases были бы ошибочно отправлены в `no_attention_required` агрессивным validation-first LoRA router, поэтому он отвергнут.",
        "",
        aggressive_fns[["study_id", "image_file", "p_chex_lora1", "p_chex_head", "p_last1", "p_last2", "ood_score_chex", "ood_score_eva"]].to_markdown(index=False) if not aggressive_fns.empty else "No aggressive LoRA FN cases found.",
        "",
        "## Interpretation",
        "",
        "- Главный ограничитель сейчас не качество/OOD, а `score_not_low_enough_or_veto`: есть реальные positive cases очень близко к порогам.",
        "- Простое ослабление порогов быстро возвращает FN, что подтверждает массовый router sweep.",
        "- Следующий полезный шаг: ручной label review этих near-threshold positive cases и hard-case adapter, обученный с отдельным clean protocol.",
        "",
        "## Visual Panels",
        "",
        "- `panels/positive_boundary_risk_final_test.png`",
        "- `panels/normal_blocked_near_boundary_final_test.png`",
        "- `panels/selected_auto_negative_final_test.png`",
        "- `panels/aggressive_lora_fn_cases.png`",
        "",
    ]
    (OUT_DIR / "near_threshold_data_label_audit_report_ru.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "panels").mkdir(exist_ok=True)
    source_map = load_source_map()
    val = load_scores("validation")
    final = load_scores("final_test")
    selected_final = selected_router_mask(final)
    selected_val = selected_router_mask(val)

    final_metrics = {
        "split": "final_test",
        "n": int(len(final)),
        "selected": int(selected_final.sum()),
        "coverage": float(selected_final.mean()),
        "TN": int(((final["y_attention"].astype(int) == 0) & selected_final).sum()),
        "FN": int(((final["y_attention"].astype(int) == 1) & selected_final).sum()),
    }
    final_metrics["NPV"] = final_metrics["TN"] / max(final_metrics["TN"] + final_metrics["FN"], 1)

    positive_boundary = pd.read_csv(MASS_DIR / "positive_boundary_risk_cases_final_test.csv", keep_default_na=False)
    normal_blocked = pd.read_csv(MASS_DIR / "normal_blocked_near_boundary_cases_final_test.csv", keep_default_na=False)
    selected_normals = final[selected_final & final["y_attention"].astype(int).eq(0)].copy().sort_values(["p_chex_head", "p_last1"]).head(24)
    selected_normals["blocker_reason"] = "selected_auto_negative"
    aggressive_fns = aggressive_lora_fn_cases(final)

    positive_boundary.to_csv(OUT_DIR / "positive_boundary_risk_cases_final_test.csv", index=False)
    normal_blocked.to_csv(OUT_DIR / "normal_blocked_near_boundary_cases_final_test.csv", index=False)
    selected_normals.to_csv(OUT_DIR / "selected_auto_negative_examples_final_test.csv", index=False)
    aggressive_fns.to_csv(OUT_DIR / "aggressive_lora_fn_cases_final_test.csv", index=False)

    render_panel(
        positive_boundary,
        "Positive near-threshold cases: do not relax blindly",
        OUT_DIR / "panels" / "positive_boundary_risk_final_test.png",
        source_map,
        n=12,
    )
    render_panel(
        normal_blocked,
        "Normal near-threshold cases: automation opportunity",
        OUT_DIR / "panels" / "normal_blocked_near_boundary_final_test.png",
        source_map,
        n=12,
    )
    render_panel(
        selected_normals,
        "Selected auto-negative normals",
        OUT_DIR / "panels" / "selected_auto_negative_final_test.png",
        source_map,
        n=12,
    )
    render_panel(
        aggressive_fns,
        "Rejected aggressive LoRA false negatives",
        OUT_DIR / "panels" / "aggressive_lora_fn_cases.png",
        source_map,
        n=max(1, len(aggressive_fns)),
    )

    split_counts, leakage = split_leakage_report()
    split_counts.to_csv(OUT_DIR / "dataset_split_counts.csv", index=False)
    leakage.to_csv(OUT_DIR / "split_leakage_checks.csv", index=False)
    score_distribution_table(val, "validation", selected_val).to_csv(OUT_DIR / "score_distribution_validation.csv", index=False)
    score_distribution_table(final, "final_test", selected_final).to_csv(OUT_DIR / "score_distribution_final_test.csv", index=False)

    blocker_summary = pd.read_csv(MASS_DIR / "router_blocker_summary_final_test.csv", keep_default_na=False)
    write_report(split_counts, leakage, final_metrics, blocker_summary, aggressive_fns)

    compact = OUT_DIR / "export_compact"
    if compact.exists():
        shutil.rmtree(compact)
    shutil.copytree(OUT_DIR, compact, ignore=shutil.ignore_patterns("export_compact", "*.zip"))
    archive_base = OUT_DIR / "near_threshold_data_label_audit_export"
    if archive_base.with_suffix(".zip").exists():
        archive_base.with_suffix(".zip").unlink()
    shutil.make_archive(str(archive_base), "zip", root_dir=compact)
    print("Audit metrics:", final_metrics)
    print("Aggressive LoRA FN cases:", len(aggressive_fns))
    print("Saved:", OUT_DIR)
    print("Archive:", archive_base.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
