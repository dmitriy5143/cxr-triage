import json

import numpy as np
import pytest
import sklearn

from fluoro_mvp_backend.ood import fit_ood_model, load_ood_model, save_ood_model, score_ood_model


def test_ood_artifact_roundtrip_and_runtime_lock(tmp_path):
    rng = np.random.default_rng(42)
    train = rng.normal(size=(64, 8)).astype(np.float32)
    model = fit_ood_model(train)
    path = save_ood_model(model, tmp_path / "eva_ood_model.pkl")
    metadata = {
        "schema_version": 1,
        "runtime": {"scikit_learn": sklearn.__version__},
    }
    metadata_path = tmp_path / "ood_artifact_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = load_ood_model(path)
    np.testing.assert_allclose(score_ood_model(loaded, train[:5]), score_ood_model(model, train[:5]))

    metadata["runtime"]["scikit_learn"] = "0.0.invalid"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires scikit-learn"):
        load_ood_model(path)
