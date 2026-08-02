"""Migrate legacy calibration/OOD artifacts to the locked production runtime.

This is a release-engineering tool. It requires the preserved research feature
matrices and split indexes, but it never needs to run either image backbone.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluoro_mvp_backend.calibration import (  # noqa: E402
    calibrate_probabilities,
    load_research_artifact,
    portable_calibrator_from_legacy,
)
from fluoro_mvp_backend.ood import fit_ood_model, save_ood_model, score_ood_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    workspace = ROOT.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "model_bundle")
    parser.add_argument(
        "--chex-features",
        type=Path,
        default=workspace / "CheXFound_frozen" / "chexfound_frozen_features.npy",
    )
    parser.add_argument(
        "--chex-index",
        type=Path,
        default=workspace / "CheXFound_frozen" / "data_index.parquet",
    )
    parser.add_argument(
        "--eva-features",
        type=Path,
        default=(
            workspace
            / "fluoro_mvp_outputs"
            / "incxr_eva_base_partial_unfreeze_t4"
            / "artifacts"
            / "embeddings"
            / "eva_x_base_features.npy"
        ),
    )
    parser.add_argument(
        "--eva-index",
        type=Path,
        default=(
            workspace
            / "fluoro_mvp_outputs"
            / "incxr_eva_base_partial_unfreeze_t4"
            / "artifacts"
            / "data_index.parquet"
        ),
    )
    parser.add_argument("--remove-legacy-calibrators", action="store_true")
    return parser


def migrate_calibrator(source: Path, target: Path) -> dict[str, object]:
    legacy = load_research_artifact(source)
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    portable = portable_calibrator_from_legacy(legacy, source_runtime=runtime)
    grid = np.linspace(1e-4, 1 - 1e-4, 2001, dtype=np.float32)
    expected = calibrate_probabilities(legacy, grid)
    actual = portable.transform(grid)
    max_delta = float(np.max(np.abs(expected - actual)))
    if max_delta > 2e-7:
        raise AssertionError(f"Calibration migration drift for {source.name}: {max_delta}")
    portable.save(target)
    return {"source": source.name, "target": target.name, "max_abs_delta": max_delta}


def migrate_ood(
    source: Path,
    target: Path,
    features_path: Path,
    index_path: Path,
    *,
    gate: float,
) -> dict[str, object]:
    features = np.load(features_path, mmap_mode="r")
    index = pd.read_parquet(index_path)
    if len(features) != len(index):
        raise ValueError(f"Feature/index length mismatch: {features_path} vs {index_path}")
    train_mask = index["split"].astype(str).eq("train").to_numpy()
    validation_mask = index["split"].astype(str).eq("validation").to_numpy()
    X_train = np.asarray(features[train_mask], dtype=np.float32)
    X_validation = np.asarray(features[validation_mask], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = joblib.load(source)
    expected = score_ood_model(legacy, X_validation)
    migrated = fit_ood_model(X_train, random_state=42)
    actual = score_ood_model(migrated, X_validation)
    max_delta = float(np.max(np.abs(expected - actual)))
    mean_delta = float(np.mean(np.abs(expected - actual)))
    gate_changes = int(np.sum((expected <= gate) != (actual <= gate)))
    if max_delta > 1e-6 or gate_changes:
        raise AssertionError(
            f"OOD migration drift for {source.name}: max_delta={max_delta}, gate_changes={gate_changes}"
        )
    save_ood_model(migrated, target)
    return {
        "source": source.name,
        "target": target.name,
        "feature_dim": int(features.shape[1]),
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "gate": gate,
        "max_abs_delta": max_delta,
        "mean_abs_delta": mean_delta,
        "gate_decision_changes": gate_changes,
    }


def main() -> int:
    args = build_parser().parse_args()
    calibration_dir = args.bundle / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    required = [args.chex_features, args.chex_index, args.eva_features, args.eva_index]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing migration inputs:\n" + "\n".join(missing))

    calibration_results = [
        migrate_calibrator(
            calibration_dir / "chexfound_head_platt_calibrator.pkl",
            calibration_dir / "chexfound_head_platt_calibrator.json",
        ),
        migrate_calibrator(
            calibration_dir / "eva_last1_calibrator.pkl",
            calibration_dir / "eva_last1_calibrator.json",
        ),
    ]
    ood_results = [
        migrate_ood(
            calibration_dir / "chexfound_ood_model.pkl",
            calibration_dir / "chexfound_ood_model.pkl",
            args.chex_features,
            args.chex_index,
            gate=1.1,
        ),
        migrate_ood(
            calibration_dir / "eva_ood_model.pkl",
            calibration_dir / "eva_ood_model.pkl",
            args.eva_features,
            args.eva_index,
            gate=1.25,
        ),
    ]
    metadata = {
        "schema_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "calibrators": calibration_results,
        "ood_models": ood_results,
        "policy": "Exact split-preserving refit; release is blocked on score/gate parity failure.",
    }
    metadata_path = calibration_dir / "ood_artifact_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if args.remove_legacy_calibrators:
        (calibration_dir / "chexfound_head_platt_calibrator.pkl").unlink()
        (calibration_dir / "eva_last1_calibrator.pkl").unlink()
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
