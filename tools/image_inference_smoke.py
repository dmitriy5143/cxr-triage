from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluoro_mvp_backend.inference import ImageModelScoreProvider, predict_from_scores  # noqa: E402


def make_demo_image(path: Path) -> None:
    y = np.linspace(0, 1, 768, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, 768, dtype=np.float32)[None, :]
    arr = (0.35 + 0.45 * (1 - ((x - 0.5) ** 2 + (y - 0.48) ** 2) * 3)).clip(0, 1)
    Image.fromarray((arr * 255).astype("uint8"), mode="L").save(path)


def main() -> int:
    bundle = ROOT / "model_bundle"
    image_path = ROOT / "runtime" / "image_smoke_demo.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    make_demo_image(image_path)
    provider = ImageModelScoreProvider(bundle)
    scored = provider.score_image_with_metadata(image_path)
    decision = predict_from_scores(scored["scores"], bundle)
    output = {
        "scores": scored["scores"],
        "preprocessing": scored["preprocessing"],
        "decision": decision,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    assert 0.0 <= float(scored["scores"]["p_chex_head"]) <= 1.0
    assert 0.0 <= float(scored["scores"]["p_last1"]) <= 1.0
    assert decision["route"] in {"no_attention_required", "N/A", "requires_attention"}
    image_path.unlink(missing_ok=True)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
