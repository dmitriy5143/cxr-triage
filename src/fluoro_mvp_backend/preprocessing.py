from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


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
    """Load and normalize one radiograph-like image for model adapters.

    This lightweight backend preprocessing intentionally mirrors the MVP
    inference contract: grayscale conversion, aspect-preserving resize, center
    crop/pad, and stable 0..1 normalization. Research notebooks remain the
    source of truth for dataset-wide QA and training-time preprocessing.
    """

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image was not found: {path}")

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("L")
    original_size = img.size
    flags: list[str] = []

    if min(original_size) < 128:
        flags.append("low_resolution")
    if max(original_size) / max(1, min(original_size)) > 3.0:
        flags.append("extreme_aspect_ratio")

    img = ImageOps.contain(img, (image_size, image_size), method=Image.Resampling.BICUBIC)
    canvas = Image.new("L", (image_size, image_size), color=0)
    offset = ((image_size - img.width) // 2, (image_size - img.height) // 2)
    canvas.paste(img, offset)

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    quality_score = 0.75
    if flags:
        quality_score = 0.5 if "low_resolution" in flags else 0.65
    critical_qa = False
    return PreprocessOutput(
        image=arr,
        quality_score=quality_score,
        critical_qa=critical_qa,
        qa_flags=flags,
        original_size=original_size,
        target_size=image_size,
    )
