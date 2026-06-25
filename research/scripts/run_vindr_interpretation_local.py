from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluoro_mvp_core import (  # noqa: E402
    NotebookConfig,
    ProbabilityCalibrator,
    bbox_mask_for_result,
    discover_vindr_dataset,
    ensure_dirs,
    get_result_image_eva,
    get_result_raw_preview,
    heatmap_localization_metrics,
    load_eva_end_to_end_checkpoint,
    make_splits,
    preprocess_dataframe,
    safe_auprc,
    safe_auc,
)
from scripts.run_chexfound_lora_local import (  # noqa: E402
    CheXFoundLoRAClassifier,
    LoraRunConfig,
    image_tensor as chexfound_image_tensor_from_path,
    make_lora_model,
)


os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")


VINDR_ROOT = ROOT / "data" / "vindr_cxr"
OUT_DIR = ROOT / "vindr_interpretation_outputs"
EVA_BUNDLE = ROOT / "selected_model_workbench" / "router_workbench" / "ensemble_candidate_bundle"
CHEX_LORA_DIR = ROOT / "CheXFound_lora_local" / "chexfound_lora_last1_r4_e80_local224_e20_b8"


def log(message: str) -> None:
    print(message, flush=True)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def tensor_from_pil_for_chexfound(img: Image.Image, image_size: int) -> torch.Tensor:
    img = img.convert("L").resize((image_size, image_size), Image.BILINEAR).convert("RGB")
    x = np.asarray(img).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    lo = x.reshape(3, -1).min(axis=1).reshape(3, 1, 1)
    hi = x.reshape(3, -1).max(axis=1).reshape(3, 1, 1)
    x = (x - lo) / np.maximum(hi - lo, 1e-6)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return torch.tensor((x - mean) / std, dtype=torch.float32)


def load_pickle(path: Path) -> Any:
    # Several notebook-produced pickles reference __main__.ProbabilityCalibrator.
    globals()["ProbabilityCalibrator"] = ProbabilityCalibrator
    return joblib.load(path)


def predict_eva_calibrated(
    model: nn.Module,
    calibrator: Any,
    images: list[Image.Image],
    *,
    image_size: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    from fluoro_mvp_core import predict_eva_end_to_end_images

    raw = predict_eva_end_to_end_images(
        model,
        images,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
    )
    if calibrator is None:
        return raw.astype(np.float32)
    return np.asarray(calibrator.transform(raw), dtype=np.float32)


def predict_chexfound_lora_calibrated(
    model: CheXFoundLoRAClassifier,
    calibrator: Any,
    images: list[Image.Image],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval().to(device)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            xb = torch.stack([tensor_from_pil_for_chexfound(img, model.image_size) for img in batch]).to(device)
            logits = model(xb)
            preds.append(torch.sigmoid(logits).detach().float().cpu().numpy())
            del xb, logits
    raw = np.concatenate(preds).astype(np.float32)
    if calibrator is None:
        return raw
    return np.asarray(calibrator.transform(raw), dtype=np.float32)


def occluded_images(img: Image.Image, grid: int, fill_value: int = 0) -> list[Image.Image]:
    img = img.convert("L")
    target = img.size[0]
    patch = max(1, target // grid)
    out = []
    for gy in range(grid):
        for gx in range(grid):
            arr = np.asarray(img).copy()
            y0, y1 = gy * patch, target if gy == grid - 1 else (gy + 1) * patch
            x0, x1 = gx * patch, target if gx == grid - 1 else (gx + 1) * patch
            arr[y0:y1, x0:x1] = fill_value
            out.append(Image.fromarray(arr.astype(np.uint8), mode="L"))
    return out


def calibrated_occlusion_heatmap(
    predict_fn,
    img: Image.Image,
    *,
    grid: int,
    fill_value: int = 0,
) -> tuple[np.ndarray, float]:
    img = img.convert("L")
    target = img.size[0]
    occ = occluded_images(img, grid=grid, fill_value=fill_value)
    base_score = float(predict_fn([img])[0])
    occ_scores = np.asarray(predict_fn(occ), dtype=np.float32)
    drops = np.clip(base_score - occ_scores, 0, None).reshape(grid, grid)
    if drops.max() > 0:
        drops = drops / drops.max()
    from scipy import ndimage

    heatmap = ndimage.zoom(drops, (target / grid, target / grid), order=1)
    return heatmap[:target, :target].astype(np.float32), base_score


def load_eva_bundle(device: str):
    cfg = json.loads((EVA_BUNDLE / "router_config.json").read_text(encoding="utf-8"))
    log("Loading EVA-X-B last1 checkpoint; this may download the public EVA-X-B weights once.")
    last1, info1 = load_eva_end_to_end_checkpoint(
        str(OUT_DIR / "runtime"),
        EVA_BUNDLE / "models" / "base_unfreeze_last1_e150_best.pt",
        device=device,
        image_size=224,
    )
    log("Loading EVA-X-B last2 checkpoint")
    last2, info2 = load_eva_end_to_end_checkpoint(
        str(OUT_DIR / "runtime"),
        EVA_BUNDLE / "models" / "base_unfreeze_last2_e150_best.pt",
        device=device,
        image_size=224,
    )
    cal1 = load_pickle(EVA_BUNDLE / "calibration" / "last1_calibrator.pkl")
    cal2 = load_pickle(EVA_BUNDLE / "calibration" / "last2_calibrator.pkl")
    return {
        "router_config": cfg,
        "last1": last1,
        "last2": last2,
        "last1_calibrator": cal1,
        "last2_calibrator": cal2,
        "last1_info": info1,
        "last2_info": info2,
    }


def load_chexfound_lora(device: str):
    manifest = json.loads((CHEX_LORA_DIR / "manifest.json").read_text(encoding="utf-8"))
    run_cfg = dict(manifest["run_config"])
    # Keep the exact training config so LoRA module names and image size match the checkpoint.
    cfg = LoraRunConfig(**{k: run_cfg[k] for k in LoraRunConfig.__dataclass_fields__ if k in run_cfg})
    model = make_lora_model(cfg, device)
    payload = torch.load(CHEX_LORA_DIR / "best.pt", map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        log(f"CheXFound LoRA load_state_dict non-strict: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    calibrator = load_pickle(CHEX_LORA_DIR / "calibrator.pkl")
    router = json.loads((CHEX_LORA_DIR / "router_config.json").read_text(encoding="utf-8"))
    return {"model": model, "calibrator": calibrator, "router_config": router, "manifest": manifest}


def classify_ensemble_route(row: pd.Series, cfg: dict[str, Any]) -> tuple[str, str]:
    quality = float(row["quality_score"])
    uncertainty = float(row["ensemble_uncertainty"])
    p1 = float(row["p_eva_last1"])
    p2 = float(row["p_eva_last2"])
    if quality < float(cfg["selected_t_quality"]):
        return "N/A", "low_quality"
    if uncertainty > float(cfg["selected_t_uncertainty"]):
        return "N/A", "high_uncertainty"
    auto = (
        (p1 <= float(cfg["selected_t_last1_negative"]) and p2 <= float(cfg["selected_t_last2_veto"]))
        or (p2 <= float(cfg["selected_t_last2_negative"]) and p1 <= float(cfg["selected_t_last1_veto"]))
    )
    if auto:
        return "no_attention_required", "ensemble_consensus_low_risk"
    if max(p1, p2) >= 0.80:
        return "requires_attention", "high_requires_attention_score"
    return "N/A", "gray_zone"


def classify_single_route(p: float, quality: float, cfg: dict[str, Any]) -> tuple[str, str]:
    uncertainty = 1.0 - abs(float(p) - 0.5) * 2.0
    if quality < float(cfg["t_quality"]):
        return "N/A", "low_quality"
    if uncertainty > float(cfg["t_uncertainty"]):
        return "N/A", "high_uncertainty"
    if p <= float(cfg["t_negative"]):
        return "no_attention_required", "single_low_risk"
    if p >= 0.80:
        return "requires_attention", "high_requires_attention_score"
    return "N/A", "gray_zone"


def render_panel(
    *,
    result,
    bbox_mask: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    scores: dict[str, float],
    routes: dict[str, str],
    out: Path,
) -> None:
    panels = [
        ("Original normalized", get_result_raw_preview(result), "gray", None),
        ("Preprocessed input", np.asarray(get_result_image_eva(result)), "gray", None),
        ("Radiologist bbox", np.asarray(get_result_image_eva(result)), "gray", bbox_mask),
    ]
    for name, hm in heatmaps.items():
        panels.append((name, hm, "magma", bbox_mask))
    n = len(panels)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes_flat = axes.ravel()
    for ax in axes_flat:
        ax.axis("off")
    for ax, (title, image, cmap, contour) in zip(axes_flat, panels):
        if title in heatmaps:
            ax.imshow(np.asarray(get_result_image_eva(result)), cmap="gray")
            ax.imshow(image, cmap=cmap, alpha=0.58)
        else:
            ax.imshow(image, cmap=cmap)
        if contour is not None and np.nansum(contour) > 0:
            ax.contour(contour, levels=[0.5], colors="cyan", linewidths=2.0)
        extra = ""
        if title in metrics:
            m = metrics[title]
            extra = (
                f"\nEnergy={m['energy_inside_bbox']:.2f} | "
                f"Pointing={m['pointing_game_hit']:.0f} | "
                f"IoU={m['bbox_iou_at_top20pct']:.2f}"
            )
        ax.set_title(title + extra, fontsize=11)
    text_lines = [
        f"study_id: {result.study_id}",
        "cyan contour: radiologist bbox",
        "heatmap: score drop after local occlusion",
        "",
    ]
    for key, value in scores.items():
        text_lines.append(f"{key}: p={value:.4f}; route={routes.get(key, 'n/a')}")
    axes_flat[-1].text(0.02, 0.95, "\n".join(text_lines), va="top", ha="left", fontsize=12)
    fig.suptitle(f"VinDr bbox interpretation: {result.study_id}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize_external_scores(case_scores: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = case_scores["y_attention"].to_numpy(dtype=int)
    for col in score_cols:
        p = case_scores[col].to_numpy(dtype=float)
        rows.append(
            {
                "score": col,
                "n": len(case_scores),
                "prevalence": float(np.mean(y)),
                "auroc": safe_auc(y, p),
                "auprc": safe_auprc(y, p),
                "mean_score_normal": float(np.mean(p[y == 0])) if np.any(y == 0) else np.nan,
                "mean_score_abnormal": float(np.mean(p[y == 1])) if np.any(y == 1) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vindr-root", type=Path, default=VINDR_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-studies", type=int, default=5000)
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--chex-cases", type=int, default=12)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-chexfound", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    if args.smoke:
        args.max_studies = min(args.max_studies, 160)
        args.cases = min(args.cases, 2)
        args.chex_cases = min(args.chex_cases, 1)
        args.grid = min(args.grid, 4)
        args.out_dir = args.out_dir / "smoke"

    device = choose_device()
    log(f"VinDr interpretation | device={device} | out={args.out_dir}")
    cfg = NotebookConfig(
        project_dir=str(args.out_dir),
        vindr_root=str(args.vindr_root),
        max_vindr_studies=int(args.max_studies),
        eva_image_size=224,
        batch_size=int(args.batch_size),
        cache_preprocessed_to_disk=True,
        preprocessed_cache_dir=str(args.out_dir / "preprocessed_cache"),
        preprocess_progress_every=250,
    )
    ensure_dirs(cfg)
    reports_dir = Path(cfg.reports_dir)
    artifacts_dir = Path(cfg.artifacts_dir)
    panels_dir = artifacts_dir / "vindr_bbox_panels"
    reports_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    log("Loading VinDr index and bbox annotations")
    manifest_path = Path(args.vindr_root) / "vindr_subset_manifest.csv"
    discover_max = None if manifest_path.exists() else args.max_studies
    vindr_df, bboxes = discover_vindr_dataset(args.vindr_root, max_studies=discover_max, cfg=cfg)
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if {"image_id", "y_attention"}.issubset(manifest.columns):
            manifest = manifest.head(int(args.max_studies)).copy()
            manifest["study_id"] = manifest["image_id"].astype(str)
            selected = set(manifest["study_id"])
            label_by_id = manifest.set_index("study_id")["y_attention"].astype(int).to_dict()
            vindr_df = vindr_df[vindr_df["study_id"].astype(str).isin(selected)].copy()
            vindr_df["y_attention"] = vindr_df["study_id"].astype(str).map(label_by_id).astype(int)
            vindr_df["label_text"] = np.where(vindr_df["y_attention"].eq(1), "abnormal", "normal")
            bboxes = bboxes[bboxes["study_id"].astype(str).isin(selected)].reset_index(drop=True)
            log(
                "Using prepared VinDr subset manifest: "
                f"rows={len(vindr_df)} labels={vindr_df['y_attention'].value_counts().to_dict()}"
            )
        else:
            log(f"VinDr manifest found but ignored because required columns are missing: {manifest_path}")
    vindr_df = make_splits(vindr_df, seed=42)
    log(f"VinDr rows={len(vindr_df)} labels={vindr_df['y_attention'].value_counts().to_dict()} splits={vindr_df['split'].value_counts().to_dict()}")
    log(f"VinDr bbox rows={len(bboxes)}")
    vindr_df.to_csv(artifacts_dir / "vindr_data_index.csv", index=False)
    bboxes.to_csv(artifacts_dir / "vindr_bboxes.csv", index=False)

    log("Preprocessing VinDr images")
    results, meta = preprocess_dataframe(vindr_df, cfg)
    meta.to_csv(artifacts_dir / "vindr_preprocessing_report.csv", index=False)
    y = meta["y_attention"].to_numpy(dtype=int)
    val_idx = meta.index[meta["split"].eq("validation")].to_numpy(dtype=int)
    if len(val_idx) == 0:
        raise RuntimeError("No validation split after VinDr split creation.")

    log("Loading EVA-X-B partial-unfreeze ensemble")
    eva = load_eva_bundle(device)
    eva_batch = max(1, int(args.batch_size))
    val_images = [get_result_image_eva(results[int(i)]) for i in val_idx]
    p_last1 = predict_eva_calibrated(
        eva["last1"],
        eva["last1_calibrator"],
        val_images,
        image_size=224,
        batch_size=eva_batch,
        device=device,
    )
    p_last2 = predict_eva_calibrated(
        eva["last2"],
        eva["last2_calibrator"],
        val_images,
        image_size=224,
        batch_size=eva_batch,
        device=device,
    )
    case_scores = meta.iloc[val_idx][["study_id", "y_attention", "split", "quality_score", "critical_qa"]].reset_index(drop=True)
    case_scores["p_eva_last1"] = p_last1
    case_scores["p_eva_last2"] = p_last2
    case_scores["p_eva_pair_max"] = np.maximum(p_last1, p_last2)
    case_scores["p_eva_pair_mean"] = 0.5 * (p_last1 + p_last2)
    case_scores["ensemble_uncertainty"] = 1.0 - np.abs(case_scores["p_eva_pair_max"].to_numpy(float) - 0.5) * 2.0
    route_rows = [classify_ensemble_route(row, eva["router_config"]) for _, row in case_scores.iterrows()]
    case_scores["eva_ensemble_route"] = [x[0] for x in route_rows]
    case_scores["eva_ensemble_reason"] = [x[1] for x in route_rows]

    chex = None
    if args.include_chexfound:
        log("Loading CheXFound LoRA last1")
        chex = load_chexfound_lora(device)
        p_chex = predict_chexfound_lora_calibrated(
            chex["model"],
            chex["calibrator"],
            val_images,
            batch_size=max(1, min(4, eva_batch)),
            device=device,
        )
        case_scores["p_chexfound_lora_last1"] = p_chex
        chex_routes = [
            classify_single_route(float(p), float(q), chex["router_config"])
            for p, q in zip(case_scores["p_chexfound_lora_last1"], case_scores["quality_score"])
        ]
        case_scores["chexfound_lora_route"] = [x[0] for x in chex_routes]
        case_scores["chexfound_lora_reason"] = [x[1] for x in chex_routes]

    score_cols = ["p_eva_last1", "p_eva_last2", "p_eva_pair_max", "p_eva_pair_mean"]
    if args.include_chexfound:
        score_cols.append("p_chexfound_lora_last1")
    score_summary = summarize_external_scores(case_scores, score_cols)
    case_scores.to_csv(reports_dir / "vindr_validation_case_scores.csv", index=False)
    score_summary.to_csv(reports_dir / "vindr_external_score_summary.csv", index=False)

    bbox_ids = set(
        bboxes[
            bboxes[["x_min", "y_min", "x_max", "y_max"]].notna().all(axis=1)
            & bboxes["class_name"].fillna("").astype(str).str.lower().ne("no finding")
        ]["study_id"].astype(str)
    )
    candidate_idx = [
        int(i)
        for i in val_idx
        if int(y[int(i)]) == 1 and str(results[int(i)].study_id) in bbox_ids
    ]
    candidate_idx = candidate_idx[: int(args.cases)]
    if not candidate_idx:
        raise RuntimeError("No abnormal validation cases with bbox were found.")
    log(f"Rendering heatmaps for {len(candidate_idx)} bbox-positive validation cases")

    metric_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(candidate_idx, start=1):
        result = results[idx]
        img = get_result_image_eva(result)
        bbox_mask = bbox_mask_for_result(result, bboxes, target=224)
        result_scores = case_scores[case_scores["study_id"].astype(str).eq(str(result.study_id))]
        score_row = result_scores.iloc[0].to_dict() if not result_scores.empty else {}

        pred_last1 = lambda images: predict_eva_calibrated(  # noqa: E731
            eva["last1"], eva["last1_calibrator"], images, image_size=224, batch_size=eva_batch, device=device
        )
        pred_last2 = lambda images: predict_eva_calibrated(  # noqa: E731
            eva["last2"], eva["last2_calibrator"], images, image_size=224, batch_size=eva_batch, device=device
        )
        hm_last1, score_last1 = calibrated_occlusion_heatmap(pred_last1, img, grid=int(args.grid))
        hm_last2, score_last2 = calibrated_occlusion_heatmap(pred_last2, img, grid=int(args.grid))
        hm_ensemble = np.maximum(hm_last1, hm_last2)
        if hm_ensemble.max() > 0:
            hm_ensemble = hm_ensemble / hm_ensemble.max()

        heatmaps = {
            "EVA last1 heatmap + bbox": hm_last1,
            "EVA last2 heatmap + bbox": hm_last2,
            "EVA ensemble max heatmap + bbox": hm_ensemble,
        }
        scores = {
            "EVA last1": float(score_row.get("p_eva_last1", score_last1)),
            "EVA last2": float(score_row.get("p_eva_last2", score_last2)),
            "EVA ensemble max": float(score_row.get("p_eva_pair_max", max(score_last1, score_last2))),
        }
        routes = {
            "EVA ensemble max": str(score_row.get("eva_ensemble_route", "")),
            "EVA last1": "",
            "EVA last2": "",
        }

        if args.include_chexfound and chex is not None and rank <= int(args.chex_cases):
            pred_chex = lambda images: predict_chexfound_lora_calibrated(  # noqa: E731
                chex["model"],
                chex["calibrator"],
                images,
                batch_size=max(1, min(4, eva_batch)),
                device=device,
            )
            hm_chex, score_chex = calibrated_occlusion_heatmap(pred_chex, img, grid=int(args.grid))
            heatmaps["CheXFound LoRA heatmap + bbox"] = hm_chex
            scores["CheXFound LoRA"] = float(score_row.get("p_chexfound_lora_last1", score_chex))
            routes["CheXFound LoRA"] = str(score_row.get("chexfound_lora_route", ""))

        local_metrics = {}
        for name, hm in heatmaps.items():
            loc = heatmap_localization_metrics(hm, bbox_mask)
            local_metrics[name] = loc
            metric_rows.append(
                {
                    "rank": rank,
                    "study_id": result.study_id,
                    "model_heatmap": name,
                    "y_attention": int(result.y_attention),
                    "p_requires_attention": scores.get(name.replace(" heatmap + bbox", ""), np.nan),
                    **loc,
                }
            )

        panel_path = panels_dir / f"vindr_case_{rank:03d}_{result.study_id}.png"
        render_panel(
            result=result,
            bbox_mask=bbox_mask,
            heatmaps=heatmaps,
            metrics=local_metrics,
            scores=scores,
            routes=routes,
            out=panel_path,
        )
        panel_rows.append({"rank": rank, "study_id": result.study_id, "panel_path": str(panel_path)})
        log(f"Rendered {rank}/{len(candidate_idx)}: {panel_path.name}")

    heatmap_metrics = pd.DataFrame(metric_rows)
    heatmap_metrics.to_csv(reports_dir / "vindr_heatmap_localization_metrics.csv", index=False)
    pd.DataFrame(panel_rows).to_csv(reports_dir / "vindr_panel_manifest.csv", index=False)

    summary_rows = []
    for name, part in heatmap_metrics.groupby("model_heatmap"):
        summary_rows.append(
            {
                "model_heatmap": name,
                "n_cases": int(len(part)),
                "pointing_game_hit_rate": float(part["pointing_game_hit"].mean()),
                "mean_energy_inside_bbox": float(part["energy_inside_bbox"].mean()),
                "median_energy_inside_bbox": float(part["energy_inside_bbox"].median()),
                "mean_top20_iou": float(part["bbox_iou_at_top20pct"].mean()),
                "median_top20_iou": float(part["bbox_iou_at_top20pct"].median()),
            }
        )
    localization_summary = pd.DataFrame(summary_rows).sort_values("model_heatmap")
    localization_summary.to_csv(reports_dir / "vindr_localization_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for model_name, part in heatmap_metrics.groupby("model_heatmap"):
        axes[0].hist(part["energy_inside_bbox"].dropna(), bins=10, alpha=0.45, label=model_name)
        axes[1].hist(part["bbox_iou_at_top20pct"].dropna(), bins=10, alpha=0.45, label=model_name)
    hit_rates = localization_summary.set_index("model_heatmap")["pointing_game_hit_rate"]
    axes[2].barh(hit_rates.index, hit_rates.values)
    axes[0].set_title("Energy inside bbox")
    axes[1].set_title("Top-20% heatmap IoU")
    axes[2].set_title("Pointing game hit rate")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    summary_png = reports_dir / "vindr_interpretation_summary.png"
    fig.savefig(summary_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# VinDr bbox interpretation report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"VinDr root: `{args.vindr_root}`",
        f"Output dir: `{args.out_dir}`",
        f"Device: `{device}`",
        "",
        "## What was tested",
        "",
        "- EVA-X-B partial-unfreeze ensemble: last1 + last2 checkpoints from the selected ensemble bundle.",
        "- CheXFound LoRA last1 when `--include-chexfound` is enabled.",
        "- Heatmaps are occlusion maps: a local patch is hidden and we measure how much the calibrated `requires_attention` score drops.",
        "- Cyan contour is the VinDr radiologist bounding box.",
        "",
        "## External score sanity on VinDr validation split",
        "",
        score_summary.to_markdown(index=False),
        "",
        "## BBox localization summary",
        "",
        localization_summary.to_markdown(index=False),
        "",
        "## Files",
        "",
        f"- Case scores: `{reports_dir / 'vindr_validation_case_scores.csv'}`",
        f"- Heatmap metrics: `{reports_dir / 'vindr_heatmap_localization_metrics.csv'}`",
        f"- Localization summary: `{reports_dir / 'vindr_localization_summary.csv'}`",
        f"- Summary plot: `{summary_png}`",
        f"- Panels: `{panels_dir}`",
        "",
        "## Interpretation notes",
        "",
        "- `Energy inside bbox` shows what share of heatmap mass falls inside the radiologist box.",
        "- `Pointing game` is 1 if the hottest heatmap point lands inside a radiologist box.",
        "- `Top-20% IoU` compares the hottest 20% of the heatmap with the bbox mask.",
        "- VinDr labels and IN-CXR training labels are not identical tasks; this is a localization sanity check, not a direct production validation replacement.",
    ]
    report_path = reports_dir / "vindr_interpretation_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "vindr_root": str(args.vindr_root),
        "out_dir": str(args.out_dir),
        "max_studies": int(args.max_studies),
        "validation_cases": int(len(val_idx)),
        "heatmap_cases": int(len(candidate_idx)),
        "grid": int(args.grid),
        "include_chexfound": bool(args.include_chexfound),
        "runtime_sec": float(time.time() - started),
        "files": {
            "report": str(report_path),
            "case_scores": str(reports_dir / "vindr_validation_case_scores.csv"),
            "heatmap_metrics": str(reports_dir / "vindr_heatmap_localization_metrics.csv"),
            "localization_summary": str(reports_dir / "vindr_localization_summary.csv"),
            "panel_manifest": str(reports_dir / "vindr_panel_manifest.csv"),
            "summary_png": str(summary_png),
            "panels_dir": str(panels_dir),
        },
    }
    (args.out_dir / "vindr_interpretation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(json.dumps(manifest, indent=2, ensure_ascii=False))

    del eva
    if chex is not None:
        del chex
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
