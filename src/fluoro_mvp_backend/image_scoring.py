from __future__ import annotations

import contextlib
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .calibration import load_research_artifact


@dataclass(frozen=True)
class ImageScoreResult:
    scores: dict[str, Any]
    preprocessing: dict[str, Any]


def score_image_with_bundle(
    image_path: str | Path,
    bundle_dir: str | Path,
    *,
    device: str | None = None,
    study_id: str | None = None,
) -> ImageScoreResult:
    scorer = EnsembleImageScorer(bundle_dir, device=device)
    return scorer.score_image(image_path, study_id=study_id)


class EnsembleImageScorer:
    """Full image-to-score adapter for the selected MVP ensemble."""

    def __init__(self, bundle_dir: str | Path, *, device: str | None = None) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.device = device or _default_device()
        self._eva: EVAEndToEndScorer | None = None
        self._chex: CheXFoundFrozenHeadScorer | None = None

    def score_image(self, image_path: str | Path, *, study_id: str | None = None) -> ImageScoreResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image was not found: {image_path}")

        raw, metadata, source_type = load_image_pixels(image_path)
        arr01 = robust_normalize(raw)
        quality_score, qa_flags, critical_qa = quality_checks(arr01, metadata)

        chex = self.chex.score(arr01)
        eva = self.eva.score(arr01)
        scores = {
            "study_id": study_id or image_path.stem,
            "image_file": str(image_path),
            "p_chex_head": float(chex["p_calibrated"]),
            "p_last1": float(eva["p_calibrated"]),
            "ood_score_chex": float(chex["ood_score"]),
            "ood_score_eva": float(eva["ood_score"]),
            "quality_score": float(quality_score),
            "critical_qa": bool(critical_qa),
            "critical_qa_bool": bool(critical_qa),
            "qa_flags": "|".join(qa_flags),
            "source_type": source_type,
        }
        preprocessing = {
            "original_shape": list(map(int, raw.shape)),
            "source_type": source_type,
            "quality_score": float(quality_score),
            "critical_qa": bool(critical_qa),
            "qa_flags": qa_flags,
            "metadata": metadata,
            "eva_image_size": int(self.eva.image_size),
            "chexfound_image_size": int(self.chex.image_size),
        }
        return ImageScoreResult(scores=scores, preprocessing=preprocessing)

    @property
    def eva(self) -> "EVAEndToEndScorer":
        if self._eva is None:
            self._eva = EVAEndToEndScorer(self.bundle_dir, device=self.device)
        return self._eva

    @property
    def chex(self) -> "CheXFoundFrozenHeadScorer":
        if self._chex is None:
            self._chex = CheXFoundFrozenHeadScorer(self.bundle_dir, device=self.device)
        return self._chex


class EVAEndToEndScorer:
    def __init__(self, bundle_dir: Path, *, device: str) -> None:
        torch, nn, _ = _torch_modules()
        self.bundle_dir = bundle_dir
        self.device = device
        self.image_size = int(_preprocessing_config(bundle_dir).get("eva_image_size", 224))
        checkpoint_path = bundle_dir / "models" / "eva_base_partial_unfreeze_last1_best.pt"
        calibrator_path = bundle_dir / "calibration" / "eva_last1_calibrator.pkl"
        ood_path = bundle_dir / "calibration" / "eva_ood_model.pkl"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EVA-X checkpoint not found: {checkpoint_path}")
        self.model = load_eva_end_to_end_model(bundle_dir, checkpoint_path, device=device).eval()
        self.ood_encoder = load_eva_frozen_encoder(bundle_dir, device=device).eval()
        self.calibrator = load_research_artifact(calibrator_path)
        self.ood_model = load_research_artifact(ood_path)
        self.torch = torch
        self.nn = nn

    def score(self, arr01: np.ndarray) -> dict[str, float]:
        torch = self.torch
        x = image_to_eva_tensor(arr01, self.image_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with _autocast_context(torch, self.device):
                logits = self.model(x)
                features = encode_eva_features(self.ood_encoder, x)
            raw = torch.sigmoid(logits).detach().float().cpu().numpy()
            feature_np = features.detach().float().cpu().numpy().astype(np.float32)
        p = _calibrate(self.calibrator, raw)
        return {
            "p_raw": float(raw[0]),
            "p_calibrated": float(p[0]),
            "ood_score": float(ood_score(self.ood_model, feature_np)[0]),
        }


class CheXFoundFrozenHeadScorer:
    def __init__(self, bundle_dir: Path, *, device: str) -> None:
        torch, nn, _ = _torch_modules()
        self.bundle_dir = bundle_dir
        self.device = device
        self.image_size = 512
        self.backbone = load_chexfound_backbone(bundle_dir, device=device).eval()
        self.head, self.scaler_mean, self.scaler_scale = load_chexfound_head(bundle_dir, device=device)
        self.calibrator = load_research_artifact(bundle_dir / "calibration" / "chexfound_head_platt_calibrator.pkl")
        self.ood_model = load_research_artifact(bundle_dir / "calibration" / "chexfound_ood_model.pkl")
        self.torch = torch
        self.nn = nn

    def score(self, arr01: np.ndarray) -> dict[str, float]:
        torch = self.torch
        x = image_to_chexfound_tensor(arr01, self.image_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with _autocast_context(torch, self.device):
                outputs = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
                features = chexfound_features_from_outputs(outputs)
                z = (features - self.scaler_mean.to(features.device)) / self.scaler_scale.to(features.device)
                logits = self.head(z)
            raw = torch.sigmoid(logits).detach().float().cpu().numpy()
            feature_np = features.detach().float().cpu().numpy().astype(np.float32)
        p = _calibrate(self.calibrator, raw)
        return {
            "p_raw": float(raw[0]),
            "p_calibrated": float(p[0]),
            "ood_score": float(ood_score(self.ood_model, feature_np)[0]),
        }


def load_eva_end_to_end_model(bundle_dir: Path, checkpoint_path: Path, *, device: str):
    torch, nn, _ = _torch_modules()
    encoder = instantiate_eva_x_base(bundle_dir)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"]
    head_weight = state_dict.get("head.1.weight")
    if head_weight is None:
        raise KeyError("EVA-X checkpoint is missing head.1.weight.")
    hidden, feature_dim = map(int, head_weight.shape)
    model = EVAEndToEndClassifier(encoder, feature_dim=feature_dim, hidden=hidden, dropout=0.20)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"EVA-X checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device)


def load_eva_frozen_encoder(bundle_dir: Path, *, device: str):
    torch, _, _ = _torch_modules()
    _add_external_path(bundle_dir / "external" / "EVA-X")
    eva_x = importlib.import_module("eva_x")
    weights_path = bundle_dir / "models" / "eva_x" / "eva_x_base_patch16_merged520k_mim.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"EVA-X-B frozen OOD weights not found: {weights_path}")
    encoder = instantiate_eva_x_base(bundle_dir)
    raw_state = torch.load(weights_path, map_location="cpu", weights_only=False)
    filtered = eva_x.checkpoint_filter_fn(raw_state, encoder)
    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading EVA-X-B frozen weights: {unexpected[:10]}")
    # Missing head/norm keys are expected for MIM pretraining checkpoints.
    return encoder.to(device)


def instantiate_eva_x_base(bundle_dir: Path):
    _add_external_path(bundle_dir / "external" / "EVA-X")
    eva_x = importlib.import_module("eva_x")
    encoder = eva_x.EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),
    )
    patch_eva_forward_features_compat(encoder)
    return encoder


def encode_eva_features(encoder: Any, x: Any):
    z = encoder.forward_features(x) if hasattr(encoder, "forward_features") else encoder(x)
    if z.ndim == 3:
        z = z[:, 1:, :].mean(dim=1) if z.shape[1] > 1 else z.mean(dim=1)
    return z


def patch_eva_forward_features_compat(model: Any) -> None:
    if hasattr(model, "_pos_embed"):
        return

    def forward_features_compat(self, x):
        x = self.patch_embed(x)
        if getattr(self, "cls_token", None) is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = getattr(self, "pos_embed", None)
        if pos_embed is not None:
            x = x + pos_embed
        pos_drop = getattr(self, "pos_drop", None)
        if pos_drop is not None:
            x = pos_drop(x)
        rope = getattr(self, "rope", None)
        rot_pos_embed = rope.get_embed() if rope is not None else None
        patch_drop = getattr(self, "patch_drop", None)
        if patch_drop is not None:
            dropped = patch_drop(x)
            x = dropped[0] if isinstance(dropped, tuple) else dropped
        for block in self.blocks:
            x = block(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    torch, _, _ = _torch_modules()
    model.forward_features = forward_features_compat.__get__(model, model.__class__)


class EVAEndToEndClassifier:
    def __init__(self, encoder: Any, feature_dim: int, hidden: int = 128, dropout: float = 0.20):
        torch, nn, _ = _torch_modules()

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.head = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )

            def encode(self, x):
                z = self.encoder.forward_features(x) if hasattr(self.encoder, "forward_features") else self.encoder(x)
                if z.ndim == 3:
                    z = z[:, 1:, :].mean(dim=1) if z.shape[1] > 1 else z.mean(dim=1)
                return z

            def forward(self, x):
                return self.head(self.encode(x)).squeeze(-1)

        self._model = _Model()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def load_chexfound_backbone(bundle_dir: Path, *, device: str):
    _add_external_path(bundle_dir / "external" / "CheXFound")
    torch, _, _ = _torch_modules()
    from safetensors.torch import load_file
    from chexfound.models.vision_transformer import vit_large

    weights_path = bundle_dir / "external" / "chexfound_hf" / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"CheXFound safetensors file not found: {weights_path}")
    state = load_file(str(weights_path), device="cpu")
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
    return model.to(device)


def load_chexfound_head(bundle_dir: Path, *, device: str):
    torch, nn, _ = _torch_modules()
    payload = torch.load(
        bundle_dir / "models" / "chexfound_frozen_head_h512_do20_lr8e4_wd1e4.pt",
        map_location="cpu",
        weights_only=False,
    )
    scaler = payload["scaler"]
    config = dict(payload.get("config") or {})
    head = TorchMLP(int(scaler.mean_.shape[0]), hidden=int(config.get("hidden", 512)), dropout=float(config.get("dropout", 0.20)))
    head.load_state_dict(payload["state_dict"], strict=True)
    mean = torch.tensor(scaler.mean_.astype(np.float32), dtype=torch.float32)
    scale = torch.tensor(np.maximum(scaler.scale_.astype(np.float32), 1e-6), dtype=torch.float32)
    return head.to(device).eval(), mean.to(device), scale.to(device)


class TorchMLP:
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.20):
        _, nn, _ = _torch_modules()

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, max(32, hidden // 4)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(32, hidden // 4), 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        self._model = _Model()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def chexfound_features_from_outputs(outputs: Any):
    torch, _, _ = _torch_modules()
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


def load_image_pixels(path: str | Path) -> tuple[np.ndarray, dict[str, Any], str]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        try:
            import pydicom
        except Exception as exc:
            raise RuntimeError("pydicom is required for DICOM image inference.") from exc
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr
        metadata = {
            "rows": int(getattr(ds, "Rows", arr.shape[0])),
            "columns": int(getattr(ds, "Columns", arr.shape[1])),
            "modality": str(getattr(ds, "Modality", "")),
            "view_position": str(getattr(ds, "ViewPosition", "")),
            "photometric_interpretation": photometric,
        }
        return arr, metadata, "dicom"

    with Image.open(path) as image:
        img = ImageOps.exif_transpose(image).convert("L")
        arr = np.asarray(img, dtype=np.float32)
    return arr, {"rows": int(arr.shape[0]), "columns": int(arr.shape[1])}, "png_jpeg"


def robust_normalize(arr: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def quality_checks(arr01: np.ndarray, metadata: dict[str, Any]) -> tuple[float, list[str], bool]:
    flags: list[str] = []
    h, w = arr01.shape
    if h < 256 or w < 256:
        flags.append("low_resolution")
    contrast = float(np.percentile(arr01, 99) - np.percentile(arr01, 1))
    if contrast < 0.08:
        flags.append("low_contrast")
    if float(np.mean(arr01 < 0.01)) > 0.55:
        flags.append("large_black_border_or_empty_area")
    if float(np.mean(arr01 > 0.99)) > 0.25:
        flags.append("large_white_saturation")
    modality = str(metadata.get("modality", "")).upper()
    if modality and modality not in {"DX", "CR", "DR", "OT"}:
        flags.append("wrong_or_unknown_modality")
    view = str(metadata.get("view_position", "")).upper()
    if view and view not in {"PA", "AP"}:
        flags.append("non_frontal_or_unknown_projection")
    score = 1.0
    penalties = {
        "low_resolution": 0.25,
        "low_contrast": 0.30,
        "large_black_border_or_empty_area": 0.25,
        "large_white_saturation": 0.25,
        "wrong_or_unknown_modality": 0.30,
        "non_frontal_or_unknown_projection": 0.30,
    }
    for flag in flags:
        score -= penalties.get(flag, 0.15)
    score = float(np.clip(score, 0.0, 1.0))
    critical = any(flag in flags for flag in ["wrong_or_unknown_modality", "non_frontal_or_unknown_projection"]) or score < 0.35
    return score, flags, critical


def resize_pad_array(arr01: np.ndarray, target: int) -> np.ndarray:
    arr01 = np.clip(arr01, 0, 1)
    img = Image.fromarray((arr01 * 255).astype(np.uint8), mode="L")
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (target, target), 0)
    canvas.paste(img, ((target - new_w) // 2, (target - new_h) // 2))
    return np.asarray(canvas).astype(np.float32) / 255.0


def image_to_eva_tensor(arr01: np.ndarray, image_size: int):
    torch, _, _ = _torch_modules()
    arr = resize_pad_array(arr01, image_size)
    rgb = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32)
    return torch.tensor((rgb - 0.5) / 0.5, dtype=torch.float32)


def image_to_chexfound_tensor(arr01: np.ndarray, image_size: int):
    torch, _, _ = _torch_modules()
    arr = resize_pad_array(arr01, image_size)
    x = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32)
    lo = x.reshape(3, -1).min(axis=1).reshape(3, 1, 1)
    hi = x.reshape(3, -1).max(axis=1).reshape(3, 1, 1)
    x = (x - lo) / np.maximum(hi - lo, 1e-6)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return torch.tensor((x - mean) / std, dtype=torch.float32)


def ood_score(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    required = {"scaler", "nn", "ref95", "iso", "iso_p5", "iso_p95"}
    missing = sorted(required.difference(model))
    if missing:
        raise ValueError(f"OOD model is incomplete; missing fields: {missing}")
    Xs = model["scaler"].transform(X)
    dists, _ = model["nn"].kneighbors(Xs)
    knn = dists[:, -1] / max(float(model["ref95"]), 1e-6)
    iso_raw = -model["iso"].score_samples(Xs)
    iso = (iso_raw - float(model["iso_p5"])) / max(float(model["iso_p95"]) - float(model["iso_p5"]), 1e-6)
    return np.clip(0.5 * knn + 0.5 * iso, 0, 2)


def _calibrate(calibrator: Any, p: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(p, dtype=np.float32)
    if hasattr(calibrator, "transform"):
        return np.asarray(calibrator.transform(p), dtype=np.float32)
    return np.asarray(p, dtype=np.float32)


def _preprocessing_config(bundle_dir: Path) -> dict[str, Any]:
    import json

    path = bundle_dir / "preprocessing_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _add_external_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"External model code is missing: {path}")
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _default_device() -> str:
    requested = os.environ.get("FLUORO_IMAGE_DEVICE")
    if requested:
        return requested
    torch, _, _ = _torch_modules()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("Install the image inference extras to use /predict-image: torch, timm, safetensors.") from exc
    return torch, nn, F


@contextlib.contextmanager
def _autocast_context(torch: Any, device: str):
    if device == "cuda":
        with torch.cuda.amp.autocast():
            yield
    else:
        yield
