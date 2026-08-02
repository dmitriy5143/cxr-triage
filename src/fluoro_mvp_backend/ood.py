from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


REQUIRED_OOD_FIELDS = {"scaler", "nn", "ref95", "iso", "iso_p5", "iso_p95"}


def resolve_ood_artifact_dir(bundle_dir: str | Path) -> Path:
    configured = os.environ.get("FLUORO_SITE_OOD_DIR")
    return Path(configured) if configured else Path(bundle_dir) / "calibration"


def ood_profile_status(bundle_dir: str | Path) -> dict[str, Any]:
    selected_dir = resolve_ood_artifact_dir(bundle_dir)
    site_profile_path = selected_dir / "site_profile.json"
    reference_policy_path = Path(bundle_dir) / "calibration" / "site_ood_policy.json"
    is_site_profile = bool(os.environ.get("FLUORO_SITE_OOD_DIR"))
    if site_profile_path.exists():
        profile = json.loads(site_profile_path.read_text(encoding="utf-8"))
    elif reference_policy_path.exists():
        profile = json.loads(reference_policy_path.read_text(encoding="utf-8"))
    else:
        profile = {"status": "missing"}
    approved = is_site_profile and profile.get("status") == "approved_for_clinical_validation"
    return {
        "selected_dir": str(selected_dir),
        "profile_id": profile.get("profile_id", "unknown"),
        "status": profile.get("status", "missing"),
        "is_target_site_profile": is_site_profile,
        "target_site_validated": bool(approved),
        "clinical_auto_negative_ready": bool(approved),
        "policy_file": str(site_profile_path if site_profile_path.exists() else reference_policy_path),
    }


def fit_ood_model(X_train: np.ndarray, *, random_state: int = 42) -> dict[str, Any]:
    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.ndim != 2 or len(X_train) < 5:
        raise ValueError("OOD fitting requires a two-dimensional feature matrix with at least five rows.")
    scaler = StandardScaler().fit(X_train)
    scaled = scaler.transform(X_train)
    nn_model = NearestNeighbors(n_neighbors=min(5, len(scaled))).fit(scaled)
    distances, _ = nn_model.kneighbors(scaled)
    isolation = IsolationForest(random_state=random_state, contamination="auto").fit(scaled)
    isolation_raw = -isolation.score_samples(scaled)
    return {
        "scaler": scaler,
        "nn": nn_model,
        "ref95": float(np.percentile(distances[:, -1], 95)),
        "iso": isolation,
        "iso_p5": float(np.percentile(isolation_raw, 5)),
        "iso_p95": float(np.percentile(isolation_raw, 95)),
    }


def score_ood_model(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    missing = sorted(REQUIRED_OOD_FIELDS.difference(model))
    if missing:
        raise ValueError(f"OOD model is incomplete; missing fields: {missing}")
    values = np.asarray(X, dtype=np.float32)
    scaled = model["scaler"].transform(values)
    distances, _ = model["nn"].kneighbors(scaled)
    knn = distances[:, -1] / max(float(model["ref95"]), 1e-6)
    isolation_raw = -model["iso"].score_samples(scaled)
    isolation = (isolation_raw - float(model["iso_p5"])) / max(
        float(model["iso_p95"]) - float(model["iso_p5"]),
        1e-6,
    )
    return np.clip(0.5 * knn + 0.5 * isolation, 0, 2)


def load_ood_model(path: str | Path, *, metadata_path: str | Path | None = None) -> dict[str, Any]:
    artifact = Path(path)
    metadata_file = Path(metadata_path) if metadata_path else artifact.parent / "ood_artifact_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"OOD runtime metadata not found: {metadata_file}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    expected = str(metadata.get("runtime", {}).get("scikit_learn", ""))
    if expected != sklearn.__version__:
        raise RuntimeError(
            f"OOD artifact requires scikit-learn=={expected}, but runtime has {sklearn.__version__}. "
            "Install the locked production environment before inference."
        )
    model = joblib.load(artifact)
    missing = sorted(REQUIRED_OOD_FIELDS.difference(model))
    if missing:
        raise ValueError(f"OOD model is incomplete; missing fields: {missing}")
    return model


def save_ood_model(model: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target, compress=0)
    return target
