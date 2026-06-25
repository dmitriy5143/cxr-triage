from pathlib import Path

import numpy as np
from PIL import Image

from fluoro_mvp_backend.preprocessing import preprocess_image


def test_preprocess_image_is_deterministic(tmp_path: Path):
    arr = np.linspace(0, 255, 256 * 384, dtype=np.uint8).reshape(256, 384)
    path = tmp_path / "demo.png"
    Image.fromarray(arr, mode="L").save(path)

    out1 = preprocess_image(path, image_size=224)
    out2 = preprocess_image(path, image_size=224)

    assert out1.image.shape == (224, 224)
    assert out1.image.dtype == np.float32
    assert 0.0 <= float(out1.image.min()) <= float(out1.image.max()) <= 1.0
    assert np.array_equal(out1.image, out2.image)
    assert out1.critical_qa is False
