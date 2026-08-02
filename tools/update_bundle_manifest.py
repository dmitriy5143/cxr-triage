"""Refresh generated model-bundle metadata and checksums for a release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model_bundle"
MANIFEST_PATH = BUNDLE / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.update(
        {
            "delivery_bundle_version": "2026-08-02-v2",
            "clinical_deployment_status": "target_site_ood_validation_required",
            "automatic_retraining_enabled": False,
            "runtime_contract": {
                "python": ">=3.11,<3.13",
                "numpy": "2.0.2",
                "scikit_learn": "1.8.0",
                "joblib": "1.5.3",
                "calibrator_format": "portable_json_v1",
                "ood_format": "joblib_sklearn_1.8.0",
            },
            "why_not_direct_backend_ready": (
                "Direct image inference is implemented and research-router parity is locked. "
                "Clinical auto-negative use still requires a clinic-local OOD profile, local labeled "
                "validation, governance approval, and production observability."
            ),
        }
    )
    files = manifest.setdefault("files", {})
    files["eva_last1_calibrator"] = {
        "path": "model_bundle/calibration/eva_last1_calibrator.json",
        "format": "portable_json_v1",
    }
    files["chexfound_head_calibrator"] = {
        "path": "model_bundle/calibration/chexfound_head_platt_calibrator.json",
        "format": "portable_json_v1",
    }
    files["ood_runtime_metadata"] = {
        "path": "model_bundle/calibration/ood_artifact_metadata.json",
        "format": "json",
    }
    files["site_ood_policy"] = {
        "path": "model_bundle/calibration/site_ood_policy.json",
        "format": "json",
    }
    files["eva_ood_model"]["bytes"] = (BUNDLE / "calibration" / "eva_ood_model.pkl").stat().st_size
    files["chexfound_ood_model"]["bytes"] = (
        BUNDLE / "calibration" / "chexfound_ood_model.pkl"
    ).stat().st_size
    files["preprocessing_config"]["bytes"] = (BUNDLE / "preprocessing_config.json").stat().st_size
    files["eva_x_source_snapshot"]["scope"] = "minimal_inference_source_and_license"
    files["chexfound_source_snapshot"]["scope"] = "minimal_inference_source_and_license"

    checksum_paths = {
        *BUNDLE.joinpath("calibration").glob("*"),
        *BUNDLE.joinpath("models").glob("*.pt"),
        *BUNDLE.joinpath("models", "eva_x").glob("*.pt"),
        *BUNDLE.joinpath("reports").glob("*"),
        BUNDLE / "preprocessing_config.json",
        BUNDLE / "external" / "CheXFound" / "LICENSE",
        BUNDLE / "external" / "CheXFound" / "README.md",
        BUNDLE / "external" / "CheXFound" / "chexfound" / "models" / "vision_transformer.py",
        BUNDLE / "external" / "EVA-X" / "LICENSE.txt",
        BUNDLE / "external" / "EVA-X" / "README.md",
        BUNDLE / "external" / "EVA-X" / "eva_x.py",
        BUNDLE / "external" / "chexfound_hf" / "README.md",
        BUNDLE / "external" / "chexfound_hf" / "config.json",
        BUNDLE / "external" / "chexfound_hf" / "model.safetensors",
    }
    checksums = []
    for path in sorted(checksum_paths):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        checksums.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest["artifact_checksums"] = checksums
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    print(f"Updated {MANIFEST_PATH} with {len(checksums)} checksums")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
