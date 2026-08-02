"""Fit a draft OOD reference profile locally from clinic radiographs.

Images and extracted features stay on the machine running this command. The
result is deliberately marked as a draft; clinical approval is a separate
governed action and is never inferred from this script succeeding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluoro_mvp_backend.image_scoring import (  # noqa: E402
    EnsembleImageScorer,
    load_image_pixels,
    quality_checks,
    robust_normalize,
)
from fluoro_mvp_backend.ood import fit_ood_model, save_ood_model  # noqa: E402


SUPPORTED_SUFFIXES = {".dcm", ".dicom", ".png", ".jpg", ".jpeg"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "model_bundle")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--min-images", type=int, default=50)
    parser.add_argument("--max-images", type=int, default=500)
    return parser


def anonymous_image_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:20]


def main() -> int:
    args = build_parser().parse_args()
    paths = sorted(
        path
        for path in args.image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )[: args.max_images]
    if len(paths) < args.min_images:
        raise RuntimeError(f"Found {len(paths)} supported images; at least {args.min_images} are required.")

    scorer = EnsembleImageScorer(args.bundle, device=args.device)
    eva_features: list[np.ndarray] = []
    chex_features: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        raw, metadata, source_type = load_image_pixels(path)
        arr01 = robust_normalize(raw)
        quality_score, qa_flags, critical_qa = quality_checks(arr01, metadata)
        if critical_qa:
            audit_rows.append(
                {
                    "image_id": anonymous_image_id(path),
                    "included": False,
                    "reason": "critical_qa",
                    "source_type": source_type,
                    "quality_score": quality_score,
                    "qa_flags": qa_flags,
                }
            )
            continue
        _, eva = scorer.eva.extract_outputs(arr01)
        _, chex = scorer.chex.extract_outputs(arr01)
        eva_features.append(eva[0])
        chex_features.append(chex[0])
        audit_rows.append(
            {
                "image_id": anonymous_image_id(path),
                "included": True,
                "reason": "eligible",
                "source_type": source_type,
                "quality_score": quality_score,
                "qa_flags": qa_flags,
            }
        )
        if index % 10 == 0:
            print(f"Processed {index}/{len(paths)} images")

    if len(eva_features) < args.min_images:
        raise RuntimeError(
            f"Only {len(eva_features)} images passed QA; at least {args.min_images} are required."
        )
    eva_matrix = np.asarray(eva_features, dtype=np.float32)
    chex_matrix = np.asarray(chex_features, dtype=np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_ood_model(fit_ood_model(eva_matrix), args.out_dir / "eva_ood_model.pkl")
    save_ood_model(fit_ood_model(chex_matrix), args.out_dir / "chexfound_ood_model.pkl")

    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    artifact_metadata = {
        "schema_version": 1,
        "runtime": runtime,
        "site_id": args.site_id,
        "included_images": len(eva_features),
        "eva_feature_dim": int(eva_matrix.shape[1]),
        "chexfound_feature_dim": int(chex_matrix.shape[1]),
    }
    (args.out_dir / "ood_artifact_metadata.json").write_text(
        json.dumps(artifact_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    profile = {
        "schema_version": 1,
        "profile_id": f"site-{args.site_id}-draft",
        "site_id": args.site_id,
        "status": "draft_requires_clinical_validation",
        "included_images": len(eva_features),
        "contains_source_images": False,
        "automatic_approval": False,
        "required_next_steps": [
            "Run held-out local score/OOD audit.",
            "Validate routing with labeled local cases and clinical reviewers.",
            "Record an approved_for_clinical_validation status through the governed release process.",
        ],
    }
    (args.out_dir / "site_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "site_profile_audit.json").write_text(
        json.dumps(audit_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
