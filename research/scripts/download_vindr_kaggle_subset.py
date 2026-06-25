from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "data" / "vindr_cxr"

COMPETITION = "vinbigdata-chest-xray-abnormalities-detection"
PNG_DATASET_512 = "xhlulu/vinbigdata"
PNG_DATASET_256 = "xhlulu/vinbigdata-chest-xray-resized-png-256x256"
METADATA_DATASET = "sunhwan/vinbigdata-chest-xray-dicom-metadata"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def ensure_kaggle_cli() -> str:
    exe = shutil.which("kaggle")
    if exe:
        return exe
    run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    exe = shutil.which("kaggle")
    if not exe:
        raise RuntimeError("Kaggle CLI was not found after installation.")
    return exe


def configure_kaggle_token() -> dict[str, str]:
    env = os.environ.copy()
    token = env.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        access_token = kaggle_dir / "access_token"
        access_token.write_text(token, encoding="utf-8")
        access_token.chmod(0o600)
        print("Kaggle access token configured from KAGGLE_API_TOKEN.")
        return env

    access_token = Path.home() / ".kaggle" / "access_token"
    legacy_json = Path.home() / ".kaggle" / "kaggle.json"
    if access_token.exists() or legacy_json.exists():
        print("Using existing Kaggle credentials from ~/.kaggle.")
        return env

    raise RuntimeError(
        "Set KAGGLE_API_TOKEN in the environment, or create ~/.kaggle/access_token / ~/.kaggle/kaggle.json."
    )


def unzip_archives(target: Path, keep_zips: bool = False) -> None:
    for zip_path in sorted(target.glob("*.zip")):
        print("Unzipping:", zip_path.name, flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        if not keep_zips:
            zip_path.unlink()


def kaggle_competition_file(kaggle: str, filename: str, target: Path, env: dict[str, str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if (target / Path(filename).name).exists():
        print("Already present:", target / Path(filename).name)
        return
    run([kaggle, "competitions", "download", "-c", COMPETITION, "-f", filename, "-p", str(target)], env=env)
    unzip_archives(target)


def kaggle_dataset_file(kaggle: str, dataset: str, filename: str | None, target: Path, env: dict[str, str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle, "datasets", "download", "-d", dataset, "-p", str(target)]
    if filename:
        cmd.extend(["-f", filename])
    run(cmd, env=env)
    unzip_archives(target)


def sample_vindr_ids(train_csv: Path, max_studies: int, normal_fraction: float, seed: int) -> pd.DataFrame:
    labels = pd.read_csv(train_csv)
    labels.columns = [c.strip() for c in labels.columns]
    if "image_id" not in labels.columns or "class_name" not in labels.columns:
        raise ValueError("train.csv must contain image_id and class_name columns.")

    tmp = labels[["image_id", "class_name"]].copy()
    tmp["image_id"] = tmp["image_id"].astype(str)
    tmp["is_abnormal_row"] = tmp["class_name"].fillna("").astype(str).str.lower().ne("no finding")
    image_level = tmp.groupby("image_id", as_index=False)["is_abnormal_row"].max()
    image_level["y_attention"] = image_level["is_abnormal_row"].astype(int)

    normal = image_level[image_level["y_attention"].eq(0)]
    abnormal = image_level[image_level["y_attention"].eq(1)]
    n_normal = min(len(normal), int(round(max_studies * normal_fraction)))
    n_abnormal = min(len(abnormal), max_studies - n_normal)
    if n_normal + n_abnormal < max_studies:
        remaining = max_studies - n_normal - n_abnormal
        if len(normal) > n_normal:
            add = min(len(normal) - n_normal, remaining)
            n_normal += add
            remaining -= add
        if remaining and len(abnormal) > n_abnormal:
            n_abnormal += min(len(abnormal) - n_abnormal, remaining)

    selected = pd.concat(
        [
            normal.sample(n=n_normal, random_state=seed) if n_normal else normal.head(0),
            abnormal.sample(n=n_abnormal, random_state=seed) if n_abnormal else abnormal.head(0),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed)
    return selected[["image_id", "y_attention"]].reset_index(drop=True)


def write_subset_annotations(train_csv: Path, manifest: pd.DataFrame, target: Path) -> None:
    labels = pd.read_csv(train_csv)
    labels["image_id"] = labels["image_id"].astype(str)
    selected_ids = set(manifest["image_id"].astype(str))
    subset_labels = labels[labels["image_id"].isin(selected_ids)].copy()
    subset_labels.to_csv(target / "vindr_subset_annotations.csv", index=False)
    print("Subset annotations:", target / "vindr_subset_annotations.csv", subset_labels.shape)


def image_exists(target: Path, image_id: str, ext: str) -> bool:
    return any(
        p.exists()
        for p in [
            target / f"{image_id}.{ext}",
            target / "train" / f"{image_id}.{ext}",
        ]
    )


def download_subset_images(
    kaggle: str,
    target: Path,
    manifest: pd.DataFrame,
    *,
    dataset: str,
    ext: str,
    env: dict[str, str],
) -> None:
    total = len(manifest)
    for idx, row in manifest.iterrows():
        image_id = str(row["image_id"])
        if image_exists(target, image_id, ext):
            continue
        if idx == 0 or (idx + 1) % 100 == 0:
            print(f"Downloading image {idx + 1}/{total}: {image_id}", flush=True)
        kaggle_dataset_file(kaggle, dataset, f"train/{image_id}.{ext}", target, env)


def validate_download(target: Path, manifest: pd.DataFrame, ext: str) -> dict[str, object]:
    image_paths = list(target.rglob(f"*.{ext}"))
    selected = set(manifest["image_id"].astype(str))
    present_ids = {p.stem for p in image_paths}
    missing = sorted(selected - present_ids)
    train_csv = target / "train.csv"
    ann_csv = target / "vindr_subset_annotations.csv"
    summary = {
        "target": str(target),
        "selected_studies": int(len(manifest)),
        "downloaded_images_total": int(len(image_paths)),
        "selected_images_present": int(len(selected) - len(missing)),
        "selected_images_missing": int(len(missing)),
        "missing_examples": missing[:10],
        "train_csv_exists": train_csv.exists(),
        "subset_annotations_exists": ann_csv.exists(),
        "class_counts": {str(k): int(v) for k, v in manifest["y_attention"].value_counts().sort_index().items()},
    }
    (target / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--max-studies", type=int, default=5000)
    parser.add_argument("--normal-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-mode", choices=["png512", "png256"], default="png512")
    parser.add_argument("--download-full-png-dataset", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    kaggle = ensure_kaggle_cli()
    env = configure_kaggle_token()
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    image_dataset = PNG_DATASET_512 if args.image_mode == "png512" else PNG_DATASET_256
    image_ext = "png"

    print("Target:", target)
    print("Mode:", args.image_mode, "max_studies:", args.max_studies)

    kaggle_competition_file(kaggle, "train.csv", target, env)
    train_csv = target / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)

    metadata_marker = target / ".metadata_download_attempted"
    if not metadata_marker.exists():
        try:
            kaggle_dataset_file(kaggle, METADATA_DATASET, None, target, env)
        except Exception as exc:
            print("Metadata download failed; bbox interpretation can still use train.csv, but bbox scaling may need metadata:", repr(exc))
        metadata_marker.write_text("1", encoding="utf-8")

    manifest = sample_vindr_ids(
        train_csv,
        max_studies=int(args.max_studies),
        normal_fraction=float(args.normal_fraction),
        seed=int(args.seed),
    )
    manifest_path = target / "vindr_subset_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("Subset manifest:", manifest_path, manifest["y_attention"].value_counts().to_dict())
    write_subset_annotations(train_csv, manifest, target)

    if not args.skip_images:
        if args.download_full_png_dataset:
            print("Downloading full PNG dataset mirror. This can be large.")
            kaggle_dataset_file(kaggle, image_dataset, None, target, env)
        else:
            download_subset_images(kaggle, target, manifest, dataset=image_dataset, ext=image_ext, env=env)

    validate_download(target, manifest, image_ext)
    print("Done. VINDR_ROOT=", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
