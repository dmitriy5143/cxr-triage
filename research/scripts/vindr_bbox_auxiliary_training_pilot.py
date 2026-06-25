from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fluoro_mvp_core import (  # noqa: E402
    heatmap_localization_metrics,
    image_to_eva_tensor,
    load_real_eva_x,
    transform_bbox_to_padded_square,
)


VINDR_DIR = ROOT / "vindr_interpretation_outputs"
OUT_DIR = ROOT / "selected_model_workbench" / "vindr_bbox_auxiliary_training_pilot"
MODEL_CACHE = ROOT / ".model_cache" / "eva_hard_case"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_device(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def make_bbox_mask_for_row(row: pd.Series, bboxes: pd.DataFrame, target: int) -> np.ndarray:
    mask = np.zeros((target, target), dtype=np.float32)
    part = bboxes[bboxes["study_id"].astype(str) == str(row["study_id"])]
    if part.empty:
        return mask
    for _, bbox in part.iterrows():
        if str(bbox.get("class_name", "")).lower() == "no finding":
            continue
        if not pd.notna(bbox.get("x_min")):
            continue
        rows = int(float(bbox.get("bbox_original_rows") or row.get("rows") or target))
        cols = int(float(bbox.get("bbox_original_columns") or row.get("columns") or target))
        x0, y0, x1, y1 = transform_bbox_to_padded_square(bbox, rows, cols, target)
        x0i, y0i, x1i, y1i = map(lambda v: int(round(v)), [x0, y0, x1, y1])
        if x1i > x0i and y1i > y0i:
            mask[y0i : y1i + 1, x0i : x1i + 1] = 1.0
    return mask


def downsample_mask(mask: np.ndarray, grid: int) -> np.ndarray:
    img = Image.fromarray((np.asarray(mask) * 255).astype(np.uint8), mode="L")
    small = img.resize((grid, grid), Image.Resampling.BOX)
    return np.asarray(small).astype(np.float32) / 255.0


class VinDrBboxDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        masks_by_study: dict[str, np.ndarray],
        *,
        image_size: int,
        grid: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.masks_by_study = masks_by_study
        self.image_size = int(image_size)
        self.grid = int(grid)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img = Image.open(str(row["source_path"])).convert("L")
        x = image_to_eva_tensor(img, image_size=self.image_size)
        y = torch.tensor(float(row["y_attention"]), dtype=torch.float32)
        mask_224 = self.masks_by_study.get(str(row["study_id"]))
        if mask_224 is None:
            mask_224 = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        mask = downsample_mask(mask_224, self.grid)
        has_bbox = float(np.sum(mask_224) > 0)
        return x, y, torch.tensor(mask, dtype=torch.float32), torch.tensor(has_bbox, dtype=torch.float32)


class EVABboxAuxModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, grid: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.grid = int(grid)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )
        self.loc_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1),
        )

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            z = self.encoder.forward_features(x)
        else:
            z = self.encoder(x)
        if z.ndim != 3:
            raise RuntimeError(f"Expected token features [B,N,C], got {tuple(z.shape)}")
        if z.shape[1] == self.grid * self.grid + 1:
            z = z[:, 1:, :]
        elif z.shape[1] > self.grid * self.grid:
            z = z[:, -self.grid * self.grid :, :]
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encode_tokens(x)
        pooled = tokens.mean(dim=1)
        cls_logit = self.cls_head(pooled).squeeze(-1)
        loc_logits = self.loc_head(tokens).squeeze(-1)
        loc_logits = loc_logits[:, : self.grid * self.grid].reshape(-1, self.grid, self.grid)
        return cls_logit, loc_logits


def infer_feature_dim(encoder: nn.Module, *, image_size: int, device: str, grid: int) -> int:
    encoder.eval()
    with torch.no_grad():
        xb = torch.zeros(1, 3, image_size, image_size, device=device)
        if hasattr(encoder, "forward_features"):
            z = encoder.forward_features(xb)
        else:
            z = encoder(xb)
        if z.ndim != 3:
            raise RuntimeError(f"Expected token features [B,N,C], got {tuple(z.shape)}")
        if z.shape[1] < grid * grid:
            raise RuntimeError(f"Expected at least {grid * grid} patch tokens, got {z.shape[1]}")
        return int(z.shape[-1])


def stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(frame) <= n:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    parts = []
    for label in [0, 1]:
        part = frame[frame["y_attention"].astype(int) == label]
        take = min(len(part), n // 2)
        if take:
            parts.append(part.sample(n=take, random_state=seed + label))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_aux(
    model: EVABboxAuxModel,
    frame: pd.DataFrame,
    masks_by_study: dict[str, np.ndarray],
    *,
    image_size: int,
    grid: int,
    batch_size: int,
    device: str,
    epochs: int,
    max_steps: int,
    lr: float,
    lambda_loc: float,
) -> pd.DataFrame:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    dataset = VinDrBboxDataset(frame, masks_by_study, image_size=image_size, grid=grid)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    rows = []
    step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        cls_losses, loc_losses = [], []
        started = time.time()
        for xb, yb, mask, has_bbox in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)
            has_bbox = has_bbox.to(device)
            cls_logit, loc_logits = model(xb)
            cls_loss = F.binary_cross_entropy_with_logits(cls_logit, yb)
            loc_loss_all = F.binary_cross_entropy_with_logits(loc_logits, mask, reduction="none").mean(dim=(1, 2))
            # Normal images also contribute with an all-zero mask, but bbox-positive cases get stronger signal.
            loc_weights = torch.where(has_bbox > 0, torch.tensor(1.0, device=device), torch.tensor(0.35, device=device))
            loc_loss = (loc_loss_all * loc_weights).mean()
            loss = cls_loss + float(lambda_loc) * loc_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            step += 1
            cls_losses.append(float(cls_loss.detach().cpu()))
            loc_losses.append(float(loc_loss.detach().cpu()))
            if step == 1 or step % 20 == 0:
                print(
                    f"[bbox-aux] epoch={epoch} step={step} cls={cls_losses[-1]:.4f} loc={loc_losses[-1]:.4f}",
                    flush=True,
                )
            del xb, yb, mask, has_bbox, cls_logit, loc_logits, cls_loss, loc_loss, loss
            clear_device(device)
            if max_steps > 0 and step >= max_steps:
                break
        rows.append(
            {
                "epoch": epoch,
                "steps_seen": step,
                "cls_loss": float(np.mean(cls_losses)) if cls_losses else float("nan"),
                "loc_loss": float(np.mean(loc_losses)) if loc_losses else float("nan"),
                "seconds": float(time.time() - started),
            }
        )
        if max_steps > 0 and step >= max_steps:
            break
    model.eval()
    return pd.DataFrame(rows)


def predict_aux(
    model: EVABboxAuxModel,
    frame: pd.DataFrame,
    masks_by_study: dict[str, np.ndarray],
    *,
    image_size: int,
    grid: int,
    batch_size: int,
    device: str,
    label: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    dataset = VinDrBboxDataset(frame, masks_by_study, image_size=image_size, grid=grid)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    heatmaps: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i, (xb, _, _, _) in enumerate(loader):
            xb = xb.to(device)
            cls_logit, loc_logits = model(xb)
            probs.append(torch.sigmoid(cls_logit).detach().float().cpu().numpy())
            hm = torch.sigmoid(loc_logits).detach().float().cpu().numpy()
            heatmaps.extend([x for x in hm])
            if i == 0 or (i + 1) % 20 == 0:
                done = min((i + 1) * batch_size, len(frame))
                print(f"[predict:{label}] {done}/{len(frame)}", flush=True)
            del xb, cls_logit, loc_logits
            clear_device(device)
    return np.concatenate(probs).astype(np.float32), heatmaps


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    return {
        "n": float(len(y)),
        "prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def localization_frame(frame: pd.DataFrame, heatmaps: list[np.ndarray], masks_by_study: dict[str, np.ndarray], image_size: int) -> pd.DataFrame:
    rows = []
    for row, heatmap in zip(frame.to_dict("records"), heatmaps):
        mask_224 = masks_by_study.get(str(row["study_id"]), np.zeros((image_size, image_size), dtype=np.float32))
        metrics = heatmap_localization_metrics(np.asarray(heatmap), mask_224)
        metrics.update(
            {
                "study_id": row["study_id"],
                "y_attention": int(row["y_attention"]),
                "split": row["split"],
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_masks(meta: pd.DataFrame, bboxes: pd.DataFrame, image_size: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    masks: dict[str, np.ndarray] = {}
    rows = []
    for _, row in meta.iterrows():
        mask = make_bbox_mask_for_row(row, bboxes, image_size)
        study_id = str(row["study_id"])
        masks[study_id] = mask
        rows.append(
            {
                "study_id": study_id,
                "split": row["split"],
                "y_attention": int(row["y_attention"]),
                "has_bbox_mask": bool(mask.sum() > 0),
                "bbox_mask_area_fraction": float(mask.mean()),
            }
        )
    return masks, pd.DataFrame(rows)


def save_panel(frame: pd.DataFrame, heatmaps: list[np.ndarray], masks_by_study: dict[str, np.ndarray], path: Path, n: int = 8) -> None:
    import matplotlib.pyplot as plt

    sample = frame[frame["y_attention"].astype(int) == 1].head(n).reset_index(drop=True)
    if sample.empty:
        return
    fig, axes = plt.subplots(len(sample), 3, figsize=(9, 3 * len(sample)))
    if len(sample) == 1:
        axes = np.asarray([axes])
    study_to_heat = {str(row["study_id"]): heatmaps[i] for i, row in enumerate(frame.to_dict("records"))}
    for i, row in sample.iterrows():
        img = Image.open(str(row["source_path"])).convert("L").resize((224, 224))
        mask = masks_by_study[str(row["study_id"])]
        heat = study_to_heat[str(row["study_id"])]
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("VinDr image")
        axes[i, 1].imshow(img, cmap="gray")
        axes[i, 1].imshow(mask, cmap="Reds", alpha=0.35)
        axes[i, 1].set_title("Radiologist bbox mask")
        axes[i, 2].imshow(img, cmap="gray")
        axes[i, 2].imshow(heat, cmap="magma", alpha=0.45)
        axes[i, 2].set_title("Aux heatmap")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, default=VINDR_DIR / "artifacts" / "vindr_preprocessing_report.csv")
    parser.add_argument("--bboxes", type=Path, default=VINDR_DIR / "artifacts" / "vindr_bboxes.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grid", type=int, default=14)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-loc", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.device = choose_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reports").mkdir(exist_ok=True)
    (OUT_DIR / "checkpoints").mkdir(exist_ok=True)
    (OUT_DIR / "panels").mkdir(exist_ok=True)

    meta = pd.read_csv(args.meta)
    bboxes = pd.read_csv(args.bboxes)
    meta = meta[meta["source_path"].astype(str).map(lambda p: Path(p).exists())].reset_index(drop=True)
    masks, mask_stats = build_masks(meta, bboxes, args.image_size)
    mask_stats.to_csv(OUT_DIR / "reports" / "bbox_mask_stats.csv", index=False)

    train = stratified_sample(meta[meta["split"] == "train"], args.train_samples, args.seed)
    validation = stratified_sample(meta[meta["split"] == "validation"], args.eval_samples, args.seed + 10)
    final_test = stratified_sample(meta[meta["split"] == "final_test"], args.eval_samples, args.seed + 20)

    print(
        f"Device={args.device} | train={len(train)} validation={len(validation)} final_test={len(final_test)}",
        flush=True,
    )
    print("BBox mask rows:", mask_stats["has_bbox_mask"].value_counts().to_dict(), flush=True)

    encoder = load_real_eva_x(str(MODEL_CACHE), variant=args.variant, device=args.device)
    feature_dim = infer_feature_dim(encoder, image_size=args.image_size, device=args.device, grid=args.grid)
    model = EVABboxAuxModel(encoder, feature_dim=feature_dim, grid=args.grid).to(args.device)

    history = train_aux(
        model,
        train,
        masks,
        image_size=args.image_size,
        grid=args.grid,
        batch_size=args.batch_size,
        device=args.device,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        lambda_loc=args.lambda_loc,
    )
    history.to_csv(OUT_DIR / "reports" / "bbox_aux_training_history.csv", index=False)

    val_p, val_heat = predict_aux(
        model,
        validation,
        masks,
        image_size=args.image_size,
        grid=args.grid,
        batch_size=args.batch_size,
        device=args.device,
        label="validation",
    )
    test_p, test_heat = predict_aux(
        model,
        final_test,
        masks,
        image_size=args.image_size,
        grid=args.grid,
        batch_size=args.batch_size,
        device=args.device,
        label="final_test",
    )

    val_scores = validation[["study_id", "split", "source_path", "y_attention"]].copy()
    val_scores["p_bbox_aux"] = val_p
    test_scores = final_test[["study_id", "split", "source_path", "y_attention"]].copy()
    test_scores["p_bbox_aux"] = test_p
    val_scores.to_csv(OUT_DIR / "reports" / "bbox_aux_scores_validation.csv", index=False)
    test_scores.to_csv(OUT_DIR / "reports" / "bbox_aux_scores_final_test.csv", index=False)

    val_loc = localization_frame(validation, val_heat, masks, args.image_size)
    test_loc = localization_frame(final_test, test_heat, masks, args.image_size)
    val_loc.to_csv(OUT_DIR / "reports" / "bbox_aux_localization_validation.csv", index=False)
    test_loc.to_csv(OUT_DIR / "reports" / "bbox_aux_localization_final_test.csv", index=False)
    save_panel(final_test, test_heat, masks, OUT_DIR / "panels" / "bbox_aux_sample_panel.png", n=8)

    val_metrics = score_metrics(validation["y_attention"].to_numpy(), val_p)
    test_metrics = score_metrics(final_test["y_attention"].to_numpy(), test_p)
    loc_summary = pd.concat(
        [
            val_loc.assign(eval_split="validation"),
            test_loc.assign(eval_split="final_test"),
        ],
        ignore_index=True,
    ).groupby(["eval_split", "y_attention"], as_index=False)[
        ["energy_inside_bbox", "pointing_game_hit", "bbox_iou_at_top20pct"]
    ].mean()
    loc_summary.to_csv(OUT_DIR / "reports" / "bbox_aux_localization_summary.csv", index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "feature_dim": feature_dim,
            "grid": args.grid,
            "note": "Short VinDr bbox auxiliary pilot. Not a deployment model.",
        },
        OUT_DIR / "checkpoints" / "vindr_bbox_aux_pilot.pt",
    )

    manifest = {
        "kind": "vindr_bbox_auxiliary_training_pilot",
        "args": vars(args),
        "feature_dim": feature_dim,
        "validation": val_metrics,
        "final_test": test_metrics,
        "mask_stats": {
            "rows": int(len(mask_stats)),
            "has_bbox": int(mask_stats["has_bbox_mask"].sum()),
            "mean_mask_area_fraction": float(mask_stats["bbox_mask_area_fraction"].mean()),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    report = f"""# VinDr Bbox-Aware Auxiliary Training Pilot

Это короткий технический pilot для идеи localization-aware adapter. Он не заменяет
IN-CXR модель и не смешивает VinDr с primary обучением. Цель: проверить, что bbox
можно корректно превратить в patch-level target и совместить classification loss
с localization loss.

## Setup

- EVA variant: `{args.variant}`
- Train samples: `{len(train)}`
- Validation samples: `{len(validation)}`
- Final test samples: `{len(final_test)}`
- Epochs / max steps: `{args.epochs}` / `{args.max_steps}`
- Lambda localization: `{args.lambda_loc}`
- Patch grid: `{args.grid}x{args.grid}`
- Device: `{args.device}`

## Classification Metrics

| split | AUROC | AUPRC | Brier |
|---|---:|---:|---:|
| validation | {val_metrics['auroc']:.4f} | {val_metrics['auprc']:.4f} | {val_metrics['brier']:.4f} |
| final_test | {test_metrics['auroc']:.4f} | {test_metrics['auprc']:.4f} | {test_metrics['brier']:.4f} |

## Localization Summary

```text
{loc_summary.to_string(index=False)}
```

## What This Means

Если даже короткий pilot учит heatmap попадать в bbox лучше случайного уровня, есть
смысл делать полноценный auxiliary adapter: frozen/partially-unfrozen EVA backbone,
classification head для IN-CXR-like таргета и localization head на VinDr как
регуляризатор внимания. Это отдельная research-ветка: ее нельзя напрямую считать
улучшением FLG MVP, пока она не проверена на IN-CXR final protocol.
"""
    (OUT_DIR / "vindr_bbox_auxiliary_training_report_ru.md").write_text(report, encoding="utf-8")
    archive = OUT_DIR.parent / "vindr_bbox_auxiliary_training_pilot_export.zip"
    if archive.exists():
        archive.unlink()
    archive = Path(shutil.make_archive(str(archive.with_suffix("")), "zip", OUT_DIR))

    print("Validation metrics:", val_metrics)
    print("Final metrics:", test_metrics)
    print(loc_summary.to_string(index=False))
    print("Saved:", OUT_DIR)
    print("Archive:", archive)


if __name__ == "__main__":
    main()
