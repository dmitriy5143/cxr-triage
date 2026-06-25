from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
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

from fluoro_mvp_core import (  # noqa: E402
    LoRALinear,
    ProbabilityCalibrator,
    TorchMLP,
    count_parameters,
    expected_calibration_error,
    inject_lora_last_blocks,
    load_image_pixels,
    metrics_summary,
    resize_pad_array,
    robust_normalize,
    wilson_lower_bound,
)


os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

CHEX_DIR = ROOT / "CheXFound_frozen"
LOCAL_DATA_ROOT = ROOT / "data" / "incxr_png"
OUT_DIR = ROOT / "CheXFound_lora_local"
EXTERNAL_DIR = ROOT / "fluoro_mvp_outputs" / "external"
HF_CACHE_DIR = ROOT / "fluoro_mvp_outputs" / "hf_cache"
FROZEN_HEAD_PATH = CHEX_DIR / "posthoc_head_sweep_workbench" / "head_models" / "h512_do20_lr8e4_wd1e4.pt"


@dataclass
class LoraRunConfig:
    name: str
    n_last_blocks: int
    rank: int
    alpha: float
    dropout: float
    lora_lr: float
    head_lr: float
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 10
    batch_size: int = 1
    grad_accum_steps: int = 4
    image_size: int = 512
    early_eval_max: int = 512
    seed: int = 42


DEFAULT_RUNS = [
    LoraRunConfig(
        name="chexfound_lora_last1_r4_e80",
        n_last_blocks=1,
        rank=4,
        alpha=8.0,
        dropout=0.05,
        lora_lr=5e-5,
        head_lr=2e-4,
        epochs=80,
        patience=10,
    ),
    LoraRunConfig(
        name="chexfound_lora_last2_r8_e80",
        n_last_blocks=2,
        rank=8,
        alpha=16.0,
        dropout=0.05,
        lora_lr=3e-5,
        head_lr=1.5e-4,
        epochs=80,
        patience=12,
    ),
]


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def ensure_deps() -> None:
    missing = []
    for dep in ["huggingface_hub", "safetensors", "fvcore", "iopath", "omegaconf"]:
        try:
            __import__(dep)
        except Exception:
            missing.append(dep)
    if missing:
        log(f"Installing missing CheXFound deps: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def ensure_chexfound_repo() -> Path:
    repo = EXTERNAL_DIR / "CheXFound"
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        log(f"Cloning CheXFound repo to {repo}")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/RPIDIAL/CheXFound.git", str(repo)], check=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def load_chexfound_backbone(device: str) -> nn.Module:
    ensure_deps()
    ensure_chexfound_repo()
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from chexfound.models.vision_transformer import vit_large

    weights_path = hf_hub_download(
        repo_id=os.environ.get("CHEXFOUND_HF_REPO", "DIAL-RPI/CheXFound"),
        filename=os.environ.get("CHEXFOUND_HF_FILENAME", "model.safetensors"),
        cache_dir=str(HF_CACHE_DIR),
    )
    log(f"Loading CheXFound weights: {weights_path}")
    state = load_file(weights_path, device="cpu")
    mapped = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        key = key.replace(".ls1.weight", ".ls1.gamma").replace(".ls2.weight", ".ls2.gamma")
        mapped[key] = value
    del state

    model = vit_large(
        img_size=512,
        patch_size=16,
        num_register_tokens=4,
        init_values=1e-5,
        ffn_layer="swiglufused",
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        block_chunks=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"CheXFound load mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    model.to(device)
    model.eval()
    return model


def build_index() -> pd.DataFrame:
    index_path = CHEX_DIR / "data_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    df = pd.read_parquet(index_path).copy()
    df["image_file"] = df["path"].map(lambda x: Path(str(x)).name)
    local_paths = {p.name: p for p in LOCAL_DATA_ROOT.rglob("*.png")}
    missing = [name for name in df["image_file"] if name not in local_paths]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} local IN-CXR images. First missing: {missing[:5]}")
    df["local_path"] = df["image_file"].map(lambda name: str(local_paths[name]))
    df["y_attention"] = df["y_attention"].astype(int)
    return df.reset_index(drop=True)


def image_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    raw, *_ = load_image_pixels(str(path))
    arr = resize_pad_array(robust_normalize(raw), image_size)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    x = np.asarray(img).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    lo = x.reshape(3, -1).min(axis=1).reshape(3, 1, 1)
    hi = x.reshape(3, -1).max(axis=1).reshape(3, 1, 1)
    x = (x - lo) / np.maximum(hi - lo, 1e-6)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return torch.tensor((x - mean) / std, dtype=torch.float32)


def chexfound_features_from_outputs(outputs: Any) -> torch.Tensor:
    cls_tokens = []
    patch_tokens = None
    for item in outputs:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            patch_tokens, cls_token = item[0], item[1]
        elif isinstance(item, dict):
            cls_token = item.get("x_norm_clstoken")
            patch_tokens = item.get("x_norm_patchtokens")
        else:
            patch_tokens = item
            cls_token = item[:, 0] if item.ndim == 3 else item
            if item.ndim == 3 and item.shape[1] > 1:
                patch_tokens = item[:, 1:, :]
        cls_tokens.append(cls_token)
    if patch_tokens is not None and patch_tokens.ndim == 3:
        cls_tokens.append(patch_tokens.mean(dim=1))
    return torch.cat(cls_tokens, dim=1)


class CheXFoundLoRAClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: TorchMLP,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        *,
        image_size: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.image_size = int(image_size)
        self.register_buffer("scaler_mean", torch.tensor(scaler_mean.astype(np.float32)))
        self.register_buffer("scaler_scale", torch.tensor(np.maximum(scaler_scale.astype(np.float32), 1e-6)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
        return chexfound_features_from_outputs(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z = (z - self.scaler_mean.to(z.device)) / self.scaler_scale.to(z.device)
        return self.head(z)


def load_frozen_head(device: str) -> tuple[TorchMLP, np.ndarray, np.ndarray, dict[str, Any]]:
    payload = torch.load(FROZEN_HEAD_PATH, map_location="cpu", weights_only=False)
    config = dict(payload.get("config") or {})
    hidden = int(config.get("hidden", 512))
    dropout = float(config.get("dropout", 0.20))
    scaler = payload["scaler"]
    head = TorchMLP(int(scaler.mean_.shape[0]), hidden=hidden, dropout=dropout)
    head.load_state_dict(payload["state_dict"], strict=True)
    head.to(device)
    head.train()
    return head, scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32), config


def make_lora_model(cfg: LoraRunConfig, device: str) -> CheXFoundLoRAClassifier:
    backbone = load_chexfound_backbone(device)
    replaced = inject_lora_last_blocks(
        backbone,
        n_last_blocks=int(cfg.n_last_blocks),
        r=int(cfg.rank),
        alpha=float(cfg.alpha),
        dropout=float(cfg.dropout),
        target_names=("qkv", "proj", "fc1", "fc2", "w12", "w3"),
    )
    if replaced <= 0:
        raise RuntimeError("No CheXFound Linear modules matched LoRA targets.")
    head, mean, scale, head_config = load_frozen_head(device)
    model = CheXFoundLoRAClassifier(backbone, head, mean, scale, image_size=cfg.image_size).to(device)
    log(f"{cfg.name}: LoRA Linear modules replaced={replaced}; frozen head={head_config}")
    log(f"{cfg.name}: params={count_parameters(model)}")
    return model


def split_indices(df: pd.DataFrame, split: str) -> np.ndarray:
    return df.index[df["split"].eq(split)].to_numpy(dtype=int)


def stratified_subset(indices: np.ndarray, labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= n:
        return indices
    rng = np.random.default_rng(seed)
    y = labels[indices]
    out = []
    for cls in [0, 1]:
        cls_idx = indices[y == cls]
        take = min(len(cls_idx), n // 2)
        out.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    if len(out) < n:
        rest = np.setdiff1d(indices, np.asarray(out, dtype=int), assume_unique=False)
        out.extend(rng.choice(rest, size=min(len(rest), n - len(out)), replace=False).tolist())
    return np.asarray(out, dtype=int)


def internal_train_es_split(train_idx: np.ndarray, labels: np.ndarray, seed: int, early_eval_max: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    y_train = labels[train_idx]
    tr, es = train_test_split(train_idx, test_size=0.15, random_state=seed, stratify=y_train)
    es = stratified_subset(es, labels, early_eval_max, seed=seed + 101)
    return np.asarray(tr, dtype=int), np.asarray(es, dtype=int)


def make_batch(df: pd.DataFrame, indices: np.ndarray, image_size: int, device: str) -> torch.Tensor:
    tensors = [image_tensor(df.loc[int(i), "local_path"], image_size=image_size) for i in indices]
    return torch.stack(tensors).to(device)


def predict_raw(model: CheXFoundLoRAClassifier, df: pd.DataFrame, indices: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            xb = make_batch(df, idx, model.image_size, device)
            logits = model(xb)
            preds.append(torch.sigmoid(logits).detach().float().cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def load_checkpoint_if_available(model: nn.Module, candidate_dir: Path, device: str) -> tuple[int, float, int, list[dict[str, Any]]]:
    last_path = candidate_dir / "last.pt"
    if not last_path.exists():
        return 0, -np.inf, 0, []
    state = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    history = list(state.get("history", []))
    if history:
        best_row = max(history, key=lambda row: row.get("early_stop_auroc_raw", float("-inf")))
        return int(state.get("epoch", len(history))), float(best_row.get("early_stop_auroc_raw", -np.inf)), int(best_row.get("epoch", 0)), history
    return int(state.get("epoch", 0)), -np.inf, 0, history


def train_one(cfg: LoraRunConfig, df: pd.DataFrame, device: str, *, force: bool = False, max_epochs: int | None = None) -> Path:
    set_seed(cfg.seed)
    labels = df["y_attention"].to_numpy(dtype=np.int64)
    train_idx = split_indices(df, "train")
    train_idx, early_idx = internal_train_es_split(train_idx, labels, cfg.seed, cfg.early_eval_max)
    early_y = labels[early_idx]
    candidate_dir = OUT_DIR / cfg.name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    complete_flag = candidate_dir / "complete.flag"
    if complete_flag.exists() and not force:
        log(f"{cfg.name}: already complete, skipping. Use --force to rerun.")
        return candidate_dir

    log(f"{cfg.name}: building model on {device}")
    model = make_lora_model(cfg, device)
    start_epoch, best_score, best_epoch, history = load_checkpoint_if_available(model, candidate_dir, device)
    if start_epoch:
        log(f"{cfg.name}: resumed from epoch={start_epoch}, best_epoch={best_epoch}, best_auc={best_score:.5f}")

    head_params = [p for n, p in model.named_parameters() if n.startswith("head.") and p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg.lora_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    pos = float((labels[train_idx] == 1).sum())
    neg = float((labels[train_idx] == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(labels.astype(np.float32), dtype=torch.float32, device=device)

    max_epochs = int(max_epochs or cfg.epochs)
    last_path = candidate_dir / "last.pt"
    best_path = candidate_dir / "best.pt"
    epoch_times = []
    for epoch in range(start_epoch, max_epochs):
        epoch_t0 = time.time()
        model.train()
        order = np.random.default_rng(cfg.seed + epoch).permutation(train_idx)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        step_count = 0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb = make_batch(df, idx, cfg.image_size, device)
            yb = y_tensor[idx]
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight) / cfg.grad_accum_steps
            loss.backward()
            step_count += 1
            if step_count % cfg.grad_accum_steps == 0 or start + cfg.batch_size >= len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * cfg.grad_accum_steps)
            if step_count == 3 and epoch == start_epoch:
                log(f"{cfg.name}: first train steps OK; current_loss={losses[-1]:.4f}")

        raw_early = predict_raw(model, df, early_idx, device=device, batch_size=cfg.batch_size)
        early_auc = safe_auc(early_y, raw_early)
        early_auprc = safe_auprc(early_y, raw_early)
        monitor = early_auc if np.isfinite(early_auc) else early_auprc
        epoch_time = time.time() - epoch_t0
        epoch_times.append(epoch_time)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "early_stop_auroc_raw": early_auc,
            "early_stop_auprc_raw": early_auprc,
            "epoch_sec": epoch_time,
            **asdict(cfg),
        }
        history.append(row)
        payload = {"epoch": epoch + 1, "state_dict": model.state_dict(), "history": history, "lora_cfg": asdict(cfg)}
        torch.save(payload, last_path)
        if np.isfinite(monitor) and monitor > best_score + 1e-4:
            best_score = float(monitor)
            best_epoch = epoch + 1
            torch.save(payload, best_path)
        pd.DataFrame(history).to_csv(candidate_dir / "training_history.csv", index=False)
        log(
            f"{cfg.name} epoch {epoch + 1}/{max_epochs}: "
            f"loss={row['train_loss']:.4f}; early_auc={early_auc:.4f}; "
            f"early_auprc={early_auprc:.4f}; best_epoch={best_epoch}; "
            f"epoch_sec={epoch_time:.1f}"
        )
        if epoch + 1 - best_epoch >= cfg.patience:
            log(f"{cfg.name}: early stopping at epoch={epoch + 1}; best_epoch={best_epoch}")
            break

    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"], strict=False)
    evaluate_candidate(cfg, model, df, candidate_dir, device)
    complete_flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return candidate_dir


def choose_calibrator(raw_calib: np.ndarray, y_calib: np.ndarray, raw_val: np.ndarray, y_val: np.ndarray) -> tuple[ProbabilityCalibrator, np.ndarray, str, pd.DataFrame]:
    rows = []
    fitted: dict[str, tuple[ProbabilityCalibrator, np.ndarray]] = {}
    for method in ["platt", "isotonic", "raw"]:
        try:
            calibrator = ProbabilityCalibrator(method=method).fit(raw_calib, y_calib)
            p_val = calibrator.transform(raw_val)
            row = metrics_summary(y_val, p_val)
            row["calibration_method"] = method
            row["calibration_error"] = ""
            rows.append(row)
            fitted[method] = (calibrator, p_val)
        except Exception as exc:
            rows.append({"calibration_method": method, "calibration_error": repr(exc)})
    table = pd.DataFrame(rows)
    valid = table[table["calibration_error"].fillna("").eq("")].copy()
    valid = valid.sort_values(["brier", "ece", "auroc", "auprc"], ascending=[True, True, False, False])
    method = str(valid.iloc[0]["calibration_method"])
    calibrator, p_val = fitted[method]
    return calibrator, p_val.astype(np.float32), method, table


def load_route_meta(split: str, df_split: pd.DataFrame | None = None) -> pd.DataFrame:
    path = CHEX_DIR / f"best_case_level_{split}.csv"
    df = pd.read_csv(path)
    df["study_id"] = df["study_id"].astype(str)
    out = df[["study_id", "quality_score", "ood_score", "y_attention", "split"]].copy()
    if df_split is None:
        return out.reset_index(drop=True)
    order = df_split[["study_id"]].copy()
    order["study_id"] = order["study_id"].astype(str)
    aligned = order.merge(out, on="study_id", how="left", validate="one_to_one")
    if aligned["quality_score"].isna().any():
        missing = aligned.loc[aligned["quality_score"].isna(), "study_id"].head(5).tolist()
        raise RuntimeError(f"Could not align route metadata for {split}; examples={missing}")
    return aligned[["study_id", "quality_score", "ood_score", "y_attention", "split"]].reset_index(drop=True)


def route_mask(p: np.ndarray, route_meta: pd.DataFrame, t_negative: float, t_ood: float, t_quality: float, t_uncertainty: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0
    return (
        (p <= float(t_negative))
        & (route_meta["quality_score"].to_numpy(float) >= float(t_quality))
        & (route_meta["ood_score"].to_numpy(float) <= float(t_ood))
        & (uncertainty <= float(t_uncertainty))
    )


def router_metrics(y: np.ndarray, p: np.ndarray, route_meta: pd.DataFrame, params: dict[str, float], include_score_metrics: bool = False) -> dict[str, Any]:
    mask = route_mask(
        p,
        route_meta,
        params["t_negative"],
        params["t_ood"],
        params["t_quality"],
        params["t_uncertainty"],
    )
    y = np.asarray(y, dtype=int)
    n_sel = int(mask.sum())
    tn = int(((y == 0) & mask).sum())
    fn = int(((y == 1) & mask).sum())
    out: dict[str, Any] = {
        **params,
        "n": int(len(y)),
        "selected_count": n_sel,
        "auto_negative_coverage": float(n_sel / max(len(y), 1)),
        "TN_count": tn,
        "FN_count": fn,
        "NPV": float(tn / max(tn + fn, 1)),
        "NPV_ci95_low": wilson_lower_bound(tn, tn + fn, z=1.96),
        "FN_per_1000_selected": float(fn / max(n_sel, 1) * 1000.0),
    }
    if include_score_metrics:
        out.update(metrics_summary(y, p))
    return out


def sweep_router(y_val: np.ndarray, p_val: np.ndarray, route_meta: pd.DataFrame) -> pd.DataFrame:
    thresholds = np.unique(np.quantile(p_val, np.linspace(0.0, 0.22, 220)))
    rows = []
    for t_negative in thresholds:
        for t_ood in [0.90, 0.95, 1.05, 1.10, 1.25, 1.50, 2.00]:
            for t_quality in [0.25, 0.35, 0.45]:
                for t_uncertainty in [0.50, 0.65, 0.80, 1.00]:
                    params = {
                        "t_negative": float(t_negative),
                        "t_ood": float(t_ood),
                        "t_quality": float(t_quality),
                        "t_uncertainty": float(t_uncertainty),
                    }
                    row = router_metrics(y_val, p_val, route_meta, params)
                    row["safe_validation_candidate"] = bool(
                        row["selected_count"] >= 10
                        and row["FN_count"] == 0
                        and row["NPV"] >= 0.99
                    )
                    rows.append(row)
    sweep = pd.DataFrame(rows)
    return sweep.sort_values(
        ["safe_validation_candidate", "auto_negative_coverage", "NPV_ci95_low", "t_negative"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def route_table(df_split: pd.DataFrame, p: np.ndarray, route_meta: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    mask = route_mask(p, route_meta, params["t_negative"], params["t_ood"], params["t_quality"], params["t_uncertainty"])
    out = df_split[["study_id", "image_file", "split", "y_attention", "local_path"]].copy()
    out["p_requires_attention"] = p
    out["quality_score"] = route_meta["quality_score"].to_numpy(float)
    out["ood_score"] = route_meta["ood_score"].to_numpy(float)
    out["uncertainty_score"] = 1.0 - np.abs(np.asarray(p) - 0.5) * 2.0
    out["route"] = np.where(mask, "no_attention_required", "N/A")
    out["reason"] = np.where(mask, "confident_no_attention_required", "not_auto_negative")
    return out


def evaluate_candidate(cfg: LoraRunConfig, model: CheXFoundLoRAClassifier, df: pd.DataFrame, candidate_dir: Path, device: str) -> None:
    split_arrays = {split: split_indices(df, split) for split in ["calibration", "validation", "final_test"]}
    raw = {
        split: predict_raw(model, df, idx, device=device, batch_size=cfg.batch_size)
        for split, idx in split_arrays.items()
    }
    y = {split: df.loc[idx, "y_attention"].to_numpy(int) for split, idx in split_arrays.items()}
    calibrator, p_val, method, calib_table = choose_calibrator(raw["calibration"], y["calibration"], raw["validation"], y["validation"])
    p = {split: calibrator.transform(raw_scores).astype(np.float32) for split, raw_scores in raw.items()}
    calib_table.to_csv(candidate_dir / "calibration_tuning.csv", index=False)
    joblib.dump(calibrator, candidate_dir / "calibrator.pkl")

    for split, idx in split_arrays.items():
        out = df.loc[idx, ["study_id", "image_file", "split", "y_attention", "local_path"]].copy()
        out["raw_probability"] = raw[split]
        out["p_requires_attention"] = p[split]
        out.to_csv(candidate_dir / f"scores_{split}.csv", index=False)

    val_df = df.loc[split_arrays["validation"]].reset_index(drop=True)
    test_df = df.loc[split_arrays["final_test"]].reset_index(drop=True)
    val_meta = load_route_meta("validation", val_df)
    test_meta = load_route_meta("final_test", test_df)
    val_sweep = sweep_router(y["validation"], p["validation"], val_meta)
    val_sweep.insert(0, "model", cfg.name)
    val_sweep.insert(1, "calibration_method", method)
    val_sweep.to_csv(candidate_dir / "validation_router_sweep.csv", index=False)
    safe = val_sweep[val_sweep["safe_validation_candidate"]].copy()
    if safe.empty:
        selected = val_sweep.iloc[0].to_dict()
        log(f"{cfg.name}: WARNING no validation-safe router candidate; using top diagnostic row.")
    else:
        selected = safe.iloc[0].to_dict()
    params = {
        "t_negative": float(selected["t_negative"]),
        "t_ood": float(selected["t_ood"]),
        "t_quality": float(selected["t_quality"]),
        "t_uncertainty": float(selected["t_uncertainty"]),
    }
    (candidate_dir / "router_config.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    val_metrics = router_metrics(y["validation"], p["validation"], val_meta, params, include_score_metrics=True)
    final_metrics = router_metrics(y["final_test"], p["final_test"], test_meta, params, include_score_metrics=True)
    val_metrics.update({"model": cfg.name, "split": "validation", "calibration_method": method})
    final_metrics.update({"model": cfg.name, "split": "final_test", "calibration_method": method})
    pd.DataFrame([val_metrics]).to_csv(candidate_dir / "validation_metrics.csv", index=False)
    pd.DataFrame([final_metrics]).to_csv(candidate_dir / "final_test_metrics.csv", index=False)
    route_table(val_df, p["validation"], val_meta, params).to_csv(candidate_dir / "routes_validation.csv", index=False)
    route_table(test_df, p["final_test"], test_meta, params).to_csv(candidate_dir / "routes_final_test.csv", index=False)

    manifest = {
        "run_config": asdict(cfg),
        "device": device,
        "frozen_head_path": str(FROZEN_HEAD_PATH),
        "calibration_method": method,
        "router_config": params,
        "validation_metrics": val_metrics,
        "final_test_metrics": final_metrics,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (candidate_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(
        f"{cfg.name}: final AUROC={final_metrics['auroc']:.4f}; "
        f"AUPRC={final_metrics['auprc']:.4f}; "
        f"coverage={final_metrics['auto_negative_coverage']:.4f}; "
        f"FN={final_metrics['FN_count']}; NPV={final_metrics['NPV']:.4f}"
    )


def aggregate_results() -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("*/final_test_metrics.csv")):
        row = pd.read_csv(path).iloc[0].to_dict()
        row["candidate_dir"] = str(path.parent)
        rows.append(row)
    if not rows:
        return
    table = pd.DataFrame(rows).sort_values(
        ["FN_count", "auto_negative_coverage", "auroc", "auprc"],
        ascending=[True, False, False, False],
    )
    table.to_csv(OUT_DIR / "chexfound_lora_aggregate_final_test.csv", index=False)
    md = [
        "# CheXFound LoRA Local Runs",
        "",
        "## Final Test Summary",
        "",
        table[
            [
                "model",
                "auroc",
                "auprc",
                "brier",
                "ece",
                "auto_negative_coverage",
                "FN_count",
                "NPV",
                "candidate_dir",
            ]
        ].to_markdown(index=False),
        "",
    ]
    (OUT_DIR / "chexfound_lora_local_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[cfg.name for cfg in DEFAULT_RUNS], default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--early-eval-max", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--smoke", action="store_true", help="Run only two epochs on a tiny train subset for code verification.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    log(f"CheXFound LoRA local runner | device={device} | out={OUT_DIR}")
    df = build_index()
    log(f"Dataset: n={len(df)} splits={df['split'].value_counts().to_dict()} labels={df['y_attention'].value_counts().to_dict()}")

    runs = []
    for cfg in DEFAULT_RUNS:
        if args.only not in {None, cfg.name}:
            continue
        cfg = LoraRunConfig(**asdict(cfg))
        if args.image_size is not None:
            cfg.image_size = int(args.image_size)
        if args.early_eval_max is not None:
            cfg.early_eval_max = int(args.early_eval_max)
        if args.batch_size is not None:
            cfg.batch_size = int(args.batch_size)
        if args.grad_accum_steps is not None:
            cfg.grad_accum_steps = int(args.grad_accum_steps)
        if args.run_suffix:
            cfg.name = f"{cfg.name}_{args.run_suffix}"
        elif args.image_size is not None and int(args.image_size) != 512:
            cfg.name = f"{cfg.name}_img{int(args.image_size)}"
        runs.append(cfg)
    if args.smoke:
        # Keep the same code path but make it cheap.
        smoke_runs = []
        for cfg in runs:
            cfg = LoraRunConfig(**asdict(cfg))
            cfg.name = f"{cfg.name}_smoke"
            cfg.epochs = 2
            cfg.patience = 2
            cfg.early_eval_max = 64
            smoke_runs.append(cfg)
        runs = smoke_runs
        keep = []
        for split in ["train", "calibration", "validation", "final_test"]:
            part = df[df["split"].eq(split)]
            keep.append(part.groupby("y_attention", group_keys=False).head(16 if split == "train" else 8))
        df = pd.concat(keep, ignore_index=True)
        log(f"Smoke subset: n={len(df)} splits={df['split'].value_counts().to_dict()}")

    started = time.time()
    for cfg in runs:
        train_one(cfg, df, device, force=args.force, max_epochs=args.max_epochs)
        aggregate_results()
    aggregate_results()
    log(f"All requested CheXFound LoRA runs done in {(time.time() - started) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
