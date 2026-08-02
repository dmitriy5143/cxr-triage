import json

import numpy as np
import pytest

from fluoro_mvp_backend.calibration import (
    PortableCalibrator,
    calibrate_probabilities,
)


def test_portable_platt_roundtrip_matches_explicit_formula(tmp_path):
    raw = np.asarray([0.02, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 0.98], dtype=np.float32)
    portable = PortableCalibrator(
        schema_version=1,
        artifact_type="probability_calibrator",
        method="platt",
        ready=True,
        coef=(0.75,),
        intercept=-0.1,
        source_runtime={"scikit_learn": "test"},
    )
    path = portable.save(tmp_path / "calibrator.json")
    loaded = PortableCalibrator.load(path)

    logits = np.log(raw / (1 - raw))
    expected = 1.0 / (1.0 + np.exp(-(0.75 * logits - 0.1)))
    actual = calibrate_probabilities(loaded, raw)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert json.loads(path.read_text())["artifact_type"] == "probability_calibrator"


def test_production_rejects_nonportable_calibrator_objects():
    with pytest.raises(TypeError, match="PortableCalibrator"):
        calibrate_probabilities(object(), np.asarray([0.2, 0.8]))
