from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from safetensors import safe_open

from .router import load_router_config, route_record
from .schemas import ScorePayload


def load_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    path = Path(bundle_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_from_scores(scores: dict[str, Any] | ScorePayload, bundle_dir: str | Path) -> dict[str, Any]:
    """Run backend routing from already computed model scores."""

    payload = scores if isinstance(scores, ScorePayload) else ScorePayload.from_mapping(dict(scores))
    config = load_router_config(bundle_dir)
    decision = route_record(payload.to_record(), config)
    out = {
        "study_id": payload.study_id,
        "route": decision.route,
        "reason": decision.reason,
        "p_requires_attention": decision.p_requires_attention,
        "selected_by_rule": decision.selected_by_rule,
        "thresholds": decision.thresholds,
    }
    return out


class PrecomputedScoreProvider:
    """Score provider for validation, tests, and offline demos."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.copy()
        if "study_id" in self.frame:
            self.frame["study_id"] = self.frame["study_id"].astype(str)

    @classmethod
    def from_csv(cls, path: str | Path) -> "PrecomputedScoreProvider":
        return cls(pd.read_csv(path))

    def get(self, study_id: str) -> dict[str, Any]:
        if "study_id" not in self.frame.columns:
            raise KeyError("Score CSV does not contain study_id.")
        rows = self.frame[self.frame["study_id"].astype(str).eq(str(study_id))]
        if rows.empty:
            raise KeyError(f"No precomputed score row for study_id={study_id!r}")
        return rows.iloc[0].to_dict()


class ImageModelScoreProvider:
    """Boundary for full image scorers.

    Heavy EVA/CheXFound encoders are imported lazily so router tests and
    score-only API flows stay fast and deterministic.
    """

    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir)
        self.manifest = load_manifest(self.bundle_dir)
        self._image_scorer: Any | None = None

    def artifact_status(self) -> dict[str, Any]:
        chex_weights = self.bundle_dir / "external" / "chexfound_hf" / "model.safetensors"
        chex_config = self.bundle_dir / "external" / "chexfound_hf" / "config.json"
        chex_head = self.bundle_dir / "models" / "chexfound_frozen_head_h512_do20_lr8e4_wd1e4.pt"
        eva_checkpoint = self.bundle_dir / "models" / "eva_base_partial_unfreeze_last1_best.pt"
        eva_frozen_weights = self.bundle_dir / "models" / "eva_x" / "eva_x_base_patch16_merged520k_mim.pt"
        eva_code = self.bundle_dir / "external" / "EVA-X" / "eva_x.py"
        chex_code = self.bundle_dir / "external" / "CheXFound" / "chexfound" / "models" / "vision_transformer.py"
        status = {
            "chexfound_hf_model_safetensors": _file_status(chex_weights),
            "chexfound_hf_config": _file_status(chex_config),
            "chexfound_external_code": _file_status(chex_code),
            "chexfound_tuned_head": _file_status(chex_head),
            "eva_x_external_code": _file_status(eva_code),
            "eva_x_base_last1_checkpoint": _file_status(eva_checkpoint),
            "eva_x_base_frozen_ood_weights": _file_status(eva_frozen_weights),
            "router_config": _file_status(self.bundle_dir / "reports" / "selected_mass_router_config.json"),
        }
        status["full_image_adapter_wired"] = all(item["exists"] for item in status.values() if isinstance(item, dict))
        status["ready_for_score_router"] = status["router_config"]["exists"]
        return status

    def validate_chexfound_safetensors_header(self) -> dict[str, Any]:
        weights = self.bundle_dir / "external" / "chexfound_hf" / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"CheXFound safetensors file not found: {weights}")
        with safe_open(weights, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            sample = keys[:5]
        return {"path": str(weights), "tensor_count": len(keys), "sample_keys": sample}

    def score_image(self, image_path: str | Path) -> dict[str, Any]:
        return self._get_image_scorer().score_image(image_path).scores

    def score_image_with_metadata(self, image_path: str | Path) -> dict[str, Any]:
        result = self._get_image_scorer().score_image(image_path)
        return {"scores": result.scores, "preprocessing": result.preprocessing}

    def _get_image_scorer(self) -> Any:
        if self._image_scorer is None:
            from .image_scoring import EnsembleImageScorer

            self._image_scorer = EnsembleImageScorer(self.bundle_dir)
        return self._image_scorer


def _file_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}
