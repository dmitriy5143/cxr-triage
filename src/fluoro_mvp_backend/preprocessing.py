from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreprocessOutput:
    image: np.ndarray
    quality_score: float
    critical_qa: bool
    qa_flags: list[str]
    original_size: tuple[int, int]
    target_size: int

    def metadata(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("image", None)
        return out


def load_preprocessing_config(bundle_dir: str | Path) -> dict[str, Any]:
    import json

    path = Path(bundle_dir) / "preprocessing_config.json"
    if not path.exists():
        return {"image_size": 224}
    return json.loads(path.read_text(encoding="utf-8"))


def preprocess_image(image_path: str | Path, image_size: int = 224) -> PreprocessOutput:
    """Run the shared production image loading, QA, and resize contract."""

    from .image_scoring import load_image_pixels, quality_checks, resize_pad_array, robust_normalize

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image was not found: {path}")

    raw, metadata, _ = load_image_pixels(path)
    normalized = robust_normalize(raw)
    quality_score, flags, critical_qa = quality_checks(normalized, metadata)
    arr = resize_pad_array(normalized, image_size)
    original_size = (int(raw.shape[1]), int(raw.shape[0]))
    return PreprocessOutput(
        image=arr,
        quality_score=quality_score,
        critical_qa=critical_qa,
        qa_flags=flags,
        original_size=original_size,
        target_size=image_size,
    )
