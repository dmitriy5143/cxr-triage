"""Pack backend model artifacts for GitHub Release upload.

The code repository intentionally excludes heavy binary model files from normal
git history. This script creates two tar archives small enough to handle as
release artifacts and preserves paths relative to the repository root, so users
can extract them directly over a cloned checkout.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "release_artifacts"
VERSION = "v0.1.0"
SPLIT_SIZE_BYTES = 256 * 1024 * 1024

ARTIFACT_GROUPS: dict[str, list[str]] = {
    f"cxr-triage-eva-artifacts-{VERSION}.tar": [
        "model_bundle/models/eva_base_partial_unfreeze_last1_best.pt",
        "model_bundle/models/eva_x/eva_x_base_patch16_merged520k_mim.pt",
        "model_bundle/calibration/eva_last1_calibrator.pkl",
        "model_bundle/calibration/eva_ood_model.pkl",
    ],
    f"cxr-triage-chexfound-artifacts-{VERSION}.tar": [
        "model_bundle/external/chexfound_hf/model.safetensors",
        "model_bundle/models/chexfound_frozen_head_h512_do20_lr8e4_wd1e4.pt",
        "model_bundle/calibration/chexfound_head_platt_calibrator.pkl",
        "model_bundle/calibration/chexfound_ood_model.pkl",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def pack_archive(archive_name: str, members: list[str]) -> dict[str, object]:
    archive_path = OUT_DIR / archive_name
    missing = [rel for rel in members if not (ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifacts:\n" + "\n".join(f" - {m}" for m in missing)
        )

    with tarfile.open(archive_path, "w") as tar:
        for rel in members:
            tar.add(ROOT / rel, arcname=rel)

    size = archive_path.stat().st_size
    parts = split_archive(archive_path)
    return {
        "path": str(archive_path.relative_to(ROOT)),
        "bytes": size,
        "size": format_size(size),
        "sha256": sha256_file(archive_path),
        "split_size_bytes": SPLIT_SIZE_BYTES,
        "parts": parts,
        "members": members,
    }


def split_archive(archive_path: Path) -> list[dict[str, object]]:
    for old_part in OUT_DIR.glob(f"{archive_path.name}.part-*"):
        old_part.unlink()

    parts: list[dict[str, object]] = []
    index = 0
    with archive_path.open("rb") as src:
        while True:
            chunk = src.read(SPLIT_SIZE_BYTES)
            if not chunk:
                break
            suffix = chr(ord("a") + index // 26) + chr(ord("a") + index % 26)
            part_path = OUT_DIR / f"{archive_path.name}.part-{suffix}"
            part_path.write_bytes(chunk)
            parts.append(
                {
                    "path": str(part_path.relative_to(ROOT)),
                    "bytes": part_path.stat().st_size,
                    "size": format_size(part_path.stat().st_size),
                    "sha256": sha256_file(part_path),
                }
            )
            index += 1
    return parts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": VERSION,
        "root": str(ROOT),
        "archives": {},
    }
    for archive_name, members in ARTIFACT_GROUPS.items():
        info = pack_archive(archive_name, members)
        summary["archives"][archive_name] = info
        print(f"{archive_name}: {info['size']} sha256={info['sha256']}")

    summary_path = OUT_DIR / "model_artifacts_sha256.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
