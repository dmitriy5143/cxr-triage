import json

import numpy as np

from fluoro_mvp_backend.calibration import (
    PortableCalibrator,
    ProbabilityCalibrator,
    calibrate_probabilities,
    portable_calibrator_from_legacy,
)


def test_portable_platt_matches_fitted_sklearn_model(tmp_path):
    raw = np.asarray([0.02, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 0.98], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    legacy = ProbabilityCalibrator("platt").fit(raw, labels)
    portable = portable_calibrator_from_legacy(legacy, source_runtime={"scikit_learn": "test"})
    path = portable.save(tmp_path / "calibrator.json")
    loaded = PortableCalibrator.load(path)

    expected = calibrate_probabilities(legacy, raw)
    actual = calibrate_probabilities(loaded, raw)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert json.loads(path.read_text())["artifact_type"] == "probability_calibrator"


def test_legacy_calibration_never_calls_serialized_transform():
    class InnerModel:
        def predict_proba(self, values):
            p = np.full(len(values), 0.25, dtype=np.float32)
            return np.column_stack([1 - p, p])

    class DangerousLegacyCalibrator:
        ready = True
        method = "platt"
        model = InnerModel()

        def transform(self, values):
            raise AssertionError("serialized transform must never be called")

    actual = calibrate_probabilities(DangerousLegacyCalibrator(), np.asarray([0.2, 0.8]))
    np.testing.assert_array_equal(actual, np.asarray([0.25, 0.25], dtype=np.float32))
