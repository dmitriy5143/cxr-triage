from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - notebook install cell handles this in Colab.
    torch = None
    nn = None
    F = None

try:
    import pydicom
except Exception:  # pragma: no cover - pydicom is optional until DICOM cells run.
    pydicom = None

from scipy import ndimage
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class NotebookConfig:
    project_dir: str = "/content/fluoro_mvp"
    data_root: str | None = None
    labels_csv: str | None = None
    vindr_root: str | None = None
    max_studies: int | None = 5000
    max_vindr_studies: int | None = 5000
    random_state: int = 42
    target_npv: float = 0.99
    max_fn_per_1000: float = 5.0
    cxr_foundation_full_size: int = 1024
    eva_image_size: int = 224
    batch_size: int = 8
    run_real_google_cxr: bool = True
    run_real_eva_x: bool = True
    run_exp2_lora: bool = False
    run_primary_track: bool = True
    run_vindr_track: bool = True
    run_vindr_exp2: bool = True
    run_exp3_chexfound: bool = False
    pca_components: int | None = 256
    cache_preprocessed_to_disk: bool = True
    preprocessed_cache_dir: str | None = None
    preprocess_progress_every: int = 250
    device: str = "cuda"
    artifacts_dir: str = field(init=False)
    reports_dir: str = field(init=False)
    checkpoints_dir: str = field(init=False)

    def __post_init__(self) -> None:
        if torch is None:
            self.device = "cpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.artifacts_dir = str(Path(self.project_dir) / "artifacts")
        self.reports_dir = str(Path(self.project_dir) / "reports")
        self.checkpoints_dir = str(Path(self.project_dir) / "checkpoints")
        if self.preprocessed_cache_dir is None:
            if Path("/content").exists():
                self.preprocessed_cache_dir = "/content/fluoro_mvp_preprocessed_cache"
            else:
                self.preprocessed_cache_dir = str(Path(self.project_dir) / "preprocessed_cache")


@dataclass
class PreprocessResult:
    study_id: str
    source_path: str
    y_attention: int
    source_dataset: str
    image_full: Image.Image | None
    image_roi: Image.Image | None
    image_eva: Image.Image | None
    image_raw_preview: np.ndarray | None
    lung_mask: np.ndarray | None
    quality_score: float
    qa_flags: list[str]
    metadata: dict[str, Any]
    preprocess_log: list[str]
    preprocessing_version: str = "preproc_v1"
    source_type: str = "unknown"
    roi_status: str = "not_available"
    critical_qa: bool = False
    image_full_path: str | None = None
    image_roi_path: str | None = None
    image_eva_path: str | None = None
    image_raw_preview_path: str | None = None
    lung_mask_path: str | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "study_id": self.study_id,
            "source_path": self.source_path,
            "y_attention": int(self.y_attention),
            "source_dataset": self.source_dataset,
            "quality_score": float(self.quality_score),
            "qa_flags": "|".join(self.qa_flags),
            "preprocess_log": "|".join(self.preprocess_log),
            "preprocessing_version": self.preprocessing_version,
            "source_type": self.source_type,
            "roi_status": self.roi_status,
            "critical_qa": bool(self.critical_qa),
            "image_full_path": self.image_full_path,
            "image_roi_path": self.image_roi_path,
            "image_eva_path": self.image_eva_path,
            "image_raw_preview_path": self.image_raw_preview_path,
            "lung_mask_path": self.lung_mask_path,
        }
        for key in [
            "modality",
            "view_position",
            "photometric_interpretation",
            "bits_stored",
            "manufacturer",
            "rows",
            "columns",
            "patient_id_hash",
            "study_uid_hash",
        ]:
            record[key] = self.metadata.get(key)
        return record


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: NotebookConfig) -> None:
    for path in [
        cfg.project_dir,
        cfg.artifacts_dir,
        cfg.reports_dir,
        cfg.checkpoints_dir,
        Path(cfg.artifacts_dir) / "previews",
        Path(cfg.artifacts_dir) / "embeddings",
        Path(cfg.artifacts_dir) / "models",
        Path(cfg.artifacts_dir) / "calibration",
        Path(cfg.artifacts_dir) / "router",
        cfg.preprocessed_cache_dir,
    ]:
        if path is None:
            continue
        Path(path).mkdir(parents=True, exist_ok=True)


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:n]


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def attach_content_hashes(df: pd.DataFrame, column: str = "content_sha256") -> pd.DataFrame:
    if "path" not in df.columns:
        raise ValueError("Dataset index has no path column.")
    out = df.copy()
    out[column] = [file_sha256(path) for path in out["path"].astype(str)]
    return out


def save_table(df: pd.DataFrame, path_no_ext: str | Path) -> str:
    path_no_ext = Path(path_no_ext)
    try:
        path = path_no_ext.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except Exception:
        path = path_no_ext.with_suffix(".csv")
        df.to_csv(path, index=False)
    return str(path)


def save_pickle(obj: Any, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cloudpickle

        with open(path, "wb") as f:
            cloudpickle.dump(obj, f)
    except Exception:
        joblib.dump(obj, path)
    return str(path)


def _find_first_existing(root: Path, names: list[str]) -> Path | None:
    lower_to_path = {p.name.lower(): p for p in root.rglob("*") if p.is_file()}
    for name in names:
        hit = lower_to_path.get(name.lower())
        if hit is not None:
            return hit
    return None


def _image_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def discover_vindr_dataset(
    vindr_root: str | Path | None,
    max_studies: int | None = 5000,
    cfg: NotebookConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load VinDr-CXR-like DICOM dataset and bbox annotations.

    Supports PhysioNet folder names and Kaggle competition-like CSV layout.
    Returns image-level dataframe plus bbox dataframe.
    """
    if vindr_root is None:
        raise ValueError("Set VINDR_ROOT before running the VinDr track.")
    root = Path(vindr_root)
    if not root.exists():
        raise FileNotFoundError(f"VinDr root not found: {root}")
    image_paths = []
    for pattern in ["*.dicom", "*.dcm", "*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(root.rglob(pattern))
    path_by_id = {_image_id_from_path(p): str(p) for p in image_paths}
    if not path_by_id:
        raise ValueError(f"No images found under {root}")

    annotation_files = []
    for filename in ["annotations_train.csv", "annotations_test.csv", "train.csv", "test.csv"]:
        p = _find_first_existing(root, [filename])
        if p is not None:
            annotation_files.append(p)
    image_label_files = []
    for filename in ["image_labels_train.csv", "image_labels_test.csv"]:
        p = _find_first_existing(root, [filename])
        if p is not None:
            image_label_files.append(p)
    metadata_files = []
    for filename in ["images.csv", "train_meta.csv", "test_meta.csv", "metadata.csv", "dicom_metadata.csv"]:
        p = _find_first_existing(root, [filename])
        if p is not None:
            metadata_files.append(p)

    original_dim_map: dict[str, tuple[float, float]] = {}
    for p in metadata_files:
        tmp = pd.read_csv(p)
        cols = {c.lower(): c for c in tmp.columns}
        image_id_col = cols.get("image_id")
        if image_id_col is None and "fname" in cols:
            image_id_col = cols["fname"]
        rows_col = cols.get("rows") or cols.get("height") or cols.get("original_height")
        cols_col = cols.get("columns") or cols.get("width") or cols.get("original_width")
        if image_id_col is None or rows_col is None or cols_col is None:
            continue
        for _, meta_row in tmp.iterrows():
            image_id = _image_id_from_path(meta_row[image_id_col])
            rows_value = pd.to_numeric(pd.Series([meta_row[rows_col]]), errors="coerce").iloc[0]
            cols_value = pd.to_numeric(pd.Series([meta_row[cols_col]]), errors="coerce").iloc[0]
            if pd.notna(rows_value) and pd.notna(cols_value) and float(rows_value) > 0 and float(cols_value) > 0:
                original_dim_map[str(image_id)] = (float(rows_value), float(cols_value))

    bbox_frames = []
    for p in annotation_files:
        tmp = pd.read_csv(p)
        tmp.columns = [c.strip() for c in tmp.columns]
        if "image_id" not in tmp.columns:
            continue
        if "class_name" not in tmp.columns and "class_id" in tmp.columns:
            tmp["class_name"] = tmp["class_id"].astype(str)
        for col in ["x_min", "y_min", "x_max", "y_max"]:
            if col not in tmp.columns:
                tmp[col] = np.nan
        tmp["study_id"] = tmp["image_id"].astype(str)
        tmp["source_split"] = "test" if "test" in p.name.lower() else "train"
        bbox_frames.append(tmp[["image_id", "study_id", "class_name", "x_min", "y_min", "x_max", "y_max", "source_split"] + ([c for c in ["rad_id", "rad_ID"] if c in tmp.columns])])
    bboxes = pd.concat(bbox_frames, ignore_index=True) if bbox_frames else pd.DataFrame(
        columns=["image_id", "study_id", "class_name", "x_min", "y_min", "x_max", "y_max", "source_split"]
    )
    if original_dim_map and not bboxes.empty:
        bboxes["bbox_original_rows"] = bboxes["image_id"].astype(str).map(
            lambda x: original_dim_map.get(x, (np.nan, np.nan))[0]
        )
        bboxes["bbox_original_columns"] = bboxes["image_id"].astype(str).map(
            lambda x: original_dim_map.get(x, (np.nan, np.nan))[1]
        )

    label_map: dict[str, int] = {}
    if image_label_files:
        for p in image_label_files:
            labels = pd.read_csv(p)
            labels.columns = [c.strip() for c in labels.columns]
            if "image_id" not in labels.columns:
                continue
            label_cols = [c for c in labels.columns if c not in {"image_id", "rad_id", "rad_ID"}]
            for image_id, part in labels.groupby("image_id"):
                if "No finding" in label_cols:
                    abnormal = bool((part.drop(columns=[c for c in ["image_id", "rad_id", "rad_ID"] if c in part.columns]).sum(axis=1) > part.get("No finding", 0)).any())
                else:
                    numeric = part[label_cols].select_dtypes(include=[np.number])
                    abnormal = bool((numeric.sum(axis=1) > 0).any()) if not numeric.empty else False
                label_map[str(image_id)] = int(abnormal)
    if bboxes is not None and not bboxes.empty:
        for image_id, part in bboxes.groupby("image_id"):
            class_names = part["class_name"].fillna("").astype(str).str.lower()
            has_box = part[["x_min", "y_min", "x_max", "y_max"]].notna().all(axis=1)
            abnormal = bool(((class_names != "no finding") & has_box).any())
            label_map[str(image_id)] = max(label_map.get(str(image_id), 0), int(abnormal))

    rows = []
    for image_id, path in path_by_id.items():
        if image_id not in label_map:
            continue
        original_rows, original_columns = original_dim_map.get(str(image_id), (np.nan, np.nan))
        source_split = "test" if "test" in Path(path).parts else "train"
        rows.append(
            {
                "study_id": image_id,
                "path": path,
                "label_text": "abnormal" if label_map[image_id] else "normal",
                "y_attention": int(label_map[image_id]),
                "source_dataset": "vindr_cxr",
                "patient_id_hash": stable_hash(image_id),
                "official_split": source_split,
                "bbox_original_rows": original_rows,
                "bbox_original_columns": original_columns,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No VinDr images could be matched to labels/annotations.")
    if max_studies is not None and len(df) > max_studies:
        df = df.groupby("y_attention", group_keys=False).sample(
            frac=min(1.0, max_studies / len(df)), random_state=42
        )
        if len(df) > max_studies:
            df = df.sample(max_studies, random_state=42)
    selected_ids = set(df["study_id"].astype(str))
    bboxes = bboxes[bboxes["study_id"].astype(str).isin(selected_ids)].reset_index(drop=True) if not bboxes.empty else bboxes
    return df.reset_index(drop=True), bboxes


def _normalize_label_token(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def infer_label_from_path(path: str | Path) -> int | None:
    parts = []
    for part in Path(path).parts:
        parts.append(_normalize_label_token(part))
    stem = _normalize_label_token(Path(path).stem)
    parts.append(stem)
    normal_words = {"normal", "no_finding", "no_findings", "negative", "no_attention_required", "healthy", "clear"}
    abnormal_words = {
        "abnormal",
        "positive",
        "requires_attention",
        "attention_required",
        "tb",
        "suspicious",
        "pathology",
        "pathological",
        "disease",
        "diseased",
    }
    if any(part in abnormal_words for part in parts):
        return 1
    if any(part in normal_words for part in parts):
        return 0
    if any(f"_{word}" in stem or f"{word}_" in stem or stem.endswith(word) for word in abnormal_words):
        return 1
    if any(f"_{word}" in stem or f"{word}_" in stem or stem.endswith(word) for word in normal_words):
        return 0
    return None


def label_value_to_attention(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    except ValueError:
        pass
    return infer_label_from_path(text)


def discover_dataset(
    data_root: str | Path | None,
    labels_csv: str | Path | None = None,
    max_studies: int | None = None,
    cfg: NotebookConfig | None = None,
) -> pd.DataFrame:
    if labels_csv:
        labels = pd.read_csv(labels_csv)
        path_col = "path" if "path" in labels.columns else "image_path"
        label_col = "y_attention" if "y_attention" in labels.columns else "label"
        rows = []
        for _, row in labels.iterrows():
            p = Path(row[path_col])
            if not p.is_absolute() and data_root:
                p = Path(data_root) / p
            y = label_value_to_attention(row[label_col])
            if y not in {0, 1}:
                raise ValueError(f"Could not infer binary y_attention from label value: {row[label_col]!r}")
            rows.append(
                {
                    "study_id": row.get("study_id", stable_hash(p)),
                    "path": str(p),
                    "label_text": row[label_col],
                    "y_attention": int(y),
                    "source_dataset": row.get("source_dataset", "incxr_or_local"),
                    "patient_id_hash": row.get("patient_id_hash", stable_hash(p.stem)),
                }
            )
        df = pd.DataFrame(rows)
    else:
        if data_root is None:
            raise ValueError("Set data_root or labels_csv before running the IN-CXR track.")
        data_root = Path(data_root)
        patterns = ["*.dcm", "*.dicom", "*.png", "*.jpg", "*.jpeg"]
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(data_root.rglob(pattern))
        rows = []
        for path in sorted(paths):
            y = infer_label_from_path(path)
            if y is None:
                continue
            rows.append(
                {
                    "study_id": stable_hash(path),
                    "path": str(path),
                    "label_text": "abnormal" if y else "normal",
                    "y_attention": int(y),
                    "source_dataset": "incxr_or_local",
                    "patient_id_hash": stable_hash(path.stem),
                }
            )
        df = pd.DataFrame(rows)
    if max_studies is not None and len(df) > max_studies:
        df = df.groupby("y_attention", group_keys=False).sample(
            frac=min(1.0, max_studies / len(df)), random_state=42
        )
        if len(df) > max_studies:
            df = df.sample(max_studies, random_state=42)
    if df.empty:
        raise ValueError("No labeled images discovered. Expected normal/abnormal folders or labels_csv.")
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, seed: int = 42, group_col: str = "patient_id_hash") -> pd.DataFrame:
    df = df.copy()
    y = df["y_attention"].astype(int)
    if len(df) < 12 or y.nunique() < 2 or y.value_counts().min() < 4:
        df["split"] = "train"
        return df
    if group_col in df.columns and df[group_col].notna().all():
        group_df = (
            df.groupby(group_col, as_index=False)
            .agg(y_attention=("y_attention", "max"), n_images=("y_attention", "size"))
            .reset_index(drop=True)
        )
        gy = group_df["y_attention"].astype(int)
        if len(group_df) >= 12 and gy.nunique() == 2 and gy.value_counts().min() >= 4:
            g_train, g_temp = train_test_split(
                group_df.index,
                test_size=0.40,
                random_state=seed,
                stratify=gy,
            )
            temp_y = gy.loc[g_temp]
            if temp_y.value_counts().min() >= 3:
                g_calib, g_hold = train_test_split(
                    g_temp,
                    test_size=0.50,
                    random_state=seed,
                    stratify=temp_y,
                )
                hold_y = gy.loc[g_hold]
                if hold_y.value_counts().min() >= 2:
                    g_val, g_test = train_test_split(
                        g_hold,
                        test_size=0.50,
                        random_state=seed,
                        stratify=hold_y,
                    )
                else:
                    g_val, g_test = g_hold, []
                df["split"] = "train"
                group_to_split = {}
                for idx in g_train:
                    group_to_split[group_df.loc[idx, group_col]] = "train"
                for idx in g_calib:
                    group_to_split[group_df.loc[idx, group_col]] = "calibration"
                for idx in g_val:
                    group_to_split[group_df.loc[idx, group_col]] = "validation"
                for idx in g_test:
                    group_to_split[group_df.loc[idx, group_col]] = "final_test"
                df["split"] = df[group_col].map(group_to_split).fillna("train")
                return df.reset_index(drop=True)
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=0.40,
        random_state=seed,
        stratify=y,
    )
    temp_y = y.loc[temp_idx]
    if temp_y.value_counts().min() < 3:
        df.loc[train_idx, "split"] = "train"
        df.loc[temp_idx, "split"] = "validation"
        return df
    calib_idx, hold_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        stratify=temp_y,
    )
    hold_y = y.loc[hold_idx]
    if hold_y.value_counts().min() < 2:
        val_idx, test_idx = hold_idx, []
    else:
        val_idx, test_idx = train_test_split(
            hold_idx,
            test_size=0.50,
            random_state=seed,
            stratify=hold_y,
        )
    df["split"] = "train"
    df.loc[calib_idx, "split"] = "calibration"
    df.loc[val_idx, "split"] = "validation"
    if len(test_idx):
        df.loc[test_idx, "split"] = "final_test"
    return df.reset_index(drop=True)


def validate_binary_dataset_contract(
    df: pd.DataFrame,
    *,
    group_col: str = "patient_id_hash",
    require_existing_files: bool = True,
) -> dict[str, Any]:
    required = {"study_id", "path", "y_attention"}
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset index is missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset index is empty.")

    labels = pd.to_numeric(df["y_attention"], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, 1}):
        raise ValueError("y_attention must contain only binary values: 0=no attention, 1=requires attention.")
    if set(labels.astype(int).unique()) != {0, 1}:
        raise ValueError("Both y_attention classes must be present.")
    if df["study_id"].astype(str).duplicated().any():
        duplicates = int(df["study_id"].astype(str).duplicated().sum())
        raise ValueError(f"study_id must be unique at image level; found {duplicates} duplicates.")
    if df["path"].astype(str).duplicated().any():
        duplicates = int(df["path"].astype(str).duplicated().sum())
        raise ValueError(f"Image paths must be unique; found {duplicates} duplicates.")

    missing_files = []
    if require_existing_files:
        missing_files = [path for path in df["path"].astype(str) if not Path(path).is_file()]
        if missing_files:
            raise FileNotFoundError(f"{len(missing_files)} indexed image files do not exist; first: {missing_files[0]}")

    conflicting_groups = 0
    if group_col in df.columns:
        group_labels = df.assign(_label=labels.astype(int)).groupby(group_col)["_label"].nunique()
        conflicting_groups = int((group_labels > 1).sum())
        if conflicting_groups:
            raise ValueError(
                f"{conflicting_groups} {group_col} groups contain conflicting binary labels. "
                "Resolve the target contract before splitting."
            )

    return {
        "n_images": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in labels.astype(int).value_counts().sort_index().items()},
        "prevalence_requires_attention": float(labels.mean()),
        "unique_studies": int(df["study_id"].astype(str).nunique()),
        "unique_groups": int(df[group_col].nunique()) if group_col in df.columns else None,
        "conflicting_groups": conflicting_groups,
        "missing_files": len(missing_files),
        "target_contract": "0=no_attention_required; 1=requires_attention; N/A is router output only",
    }


def validate_split_integrity(
    df: pd.DataFrame,
    *,
    group_col: str = "patient_id_hash",
    required_splits: tuple[str, ...] = ("train", "calibration", "validation", "final_test"),
) -> dict[str, Any]:
    if "split" not in df.columns:
        raise ValueError("Dataset index has no split column.")
    present = set(df["split"].dropna().astype(str).unique())
    missing = [split for split in required_splits if split not in present]
    if missing:
        raise ValueError(f"Production run requires all splits; missing: {missing}")

    split_counts: dict[str, dict[str, int]] = {}
    for split in required_splits:
        part = df[df["split"].astype(str) == split]
        counts = part["y_attention"].astype(int).value_counts()
        if set(counts.index) != {0, 1}:
            raise ValueError(f"Split {split!r} must contain both target classes; counts={counts.to_dict()}")
        split_counts[split] = {str(k): int(v) for k, v in counts.sort_index().items()}

    overlap_pairs: list[str] = []
    if group_col in df.columns:
        group_splits = df.groupby(group_col)["split"].nunique()
        overlapping = group_splits[group_splits > 1]
        if not overlapping.empty:
            overlap_pairs = overlapping.index.astype(str).tolist()[:10]
            raise ValueError(
                f"{len(overlapping)} {group_col} groups leak across splits; examples={overlap_pairs}"
            )

    return {
        "split_counts": split_counts,
        "group_overlap_count": 0,
        "required_splits": list(required_splits),
    }


def train_internal_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    val_fraction: float = 0.15,
    min_per_class: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    y = np.asarray(y).astype(int)
    X = np.asarray(X)
    idx = np.arange(len(y))
    if len(y) < 20 or len(np.unique(y)) < 2 or pd.Series(y).value_counts().min() < min_per_class * 2:
        return X, y, None, None
    try:
        train_idx, val_idx = train_test_split(
            idx,
            test_size=val_fraction,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        return X, y, None, None
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def index_internal_validation_split(
    indices: np.ndarray | list[int],
    labels: np.ndarray,
    seed: int = 42,
    val_fraction: float = 0.15,
    min_per_class: int = 3,
) -> tuple[np.ndarray, np.ndarray | None]:
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(labels).astype(int)
    y = labels[indices]
    if len(indices) < 20 or len(np.unique(y)) < 2 or pd.Series(y).value_counts().min() < min_per_class * 2:
        return indices, None
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        return indices, None
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _first_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0]
    try:
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return list(value)[0]
    except Exception:
        pass
    return value


def load_image_pixels(path: str | Path) -> tuple[np.ndarray, dict[str, Any], str, list[str]]:
    path = Path(path)
    log: list[str] = []
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        if pydicom is None:
            raise RuntimeError("pydicom is required to read DICOM files.")
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr
            log.append("inverted_MONOCHROME1")
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            wc = float(_first_value(wc))
            ww = float(_first_value(ww))
            lo, hi = wc - ww / 2, wc + ww / 2
            if hi > lo:
                arr = np.clip(arr, lo, hi)
                log.append("applied_dicom_window")
        metadata = {
            "modality": str(getattr(ds, "Modality", "")),
            "view_position": str(getattr(ds, "ViewPosition", "")),
            "photometric_interpretation": photometric,
            "bits_stored": getattr(ds, "BitsStored", None),
            "manufacturer": str(getattr(ds, "Manufacturer", "")),
            "rows": int(getattr(ds, "Rows", arr.shape[0])),
            "columns": int(getattr(ds, "Columns", arr.shape[1])),
            "patient_id_hash": stable_hash(getattr(ds, "PatientID", str(path.parent))),
            "study_uid_hash": stable_hash(getattr(ds, "StudyInstanceUID", str(path))),
        }
        return arr, metadata, "incxr_dicom", log
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32)
    metadata = {
        "rows": int(arr.shape[0]),
        "columns": int(arr.shape[1]),
        "patient_id_hash": stable_hash(path.parent),
        "study_uid_hash": stable_hash(path),
    }
    return arr, metadata, "png_jpeg", log


def robust_normalize(arr: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def quality_checks(arr01: np.ndarray, metadata: dict[str, Any]) -> tuple[float, list[str], bool]:
    flags: list[str] = []
    h, w = arr01.shape
    if h < 256 or w < 256:
        flags.append("low_resolution")
    contrast = float(np.percentile(arr01, 99) - np.percentile(arr01, 1))
    if contrast < 0.08:
        flags.append("low_contrast")
    if float(np.mean(arr01 < 0.01)) > 0.55:
        flags.append("large_black_border_or_empty_area")
    if float(np.mean(arr01 > 0.99)) > 0.25:
        flags.append("large_white_saturation")
    border = np.concatenate([arr01[:10, :].ravel(), arr01[-10:, :].ravel(), arr01[:, :10].ravel(), arr01[:, -10:].ravel()])
    if float(np.mean(border < 0.01)) > 0.80:
        flags.append("black_frame")
    modality = str(metadata.get("modality", "")).upper()
    if modality and modality not in {"DX", "CR", "DR", "OT"}:
        flags.append("wrong_or_unknown_modality")
    view = str(metadata.get("view_position", "")).upper()
    if view and view not in {"PA", "AP"}:
        flags.append("non_frontal_or_unknown_projection")
    score = 1.0
    penalty = {
        "low_resolution": 0.25,
        "low_contrast": 0.30,
        "large_black_border_or_empty_area": 0.25,
        "large_white_saturation": 0.25,
        "black_frame": 0.10,
        "wrong_or_unknown_modality": 0.30,
        "non_frontal_or_unknown_projection": 0.30,
    }
    for flag in flags:
        score -= penalty.get(flag, 0.15)
    score = float(np.clip(score, 0.0, 1.0))
    critical = any(flag in flags for flag in ["wrong_or_unknown_modality", "non_frontal_or_unknown_projection"]) or score < 0.35
    return score, flags, critical


def make_thorax_roi(arr01: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, str, list[str]]:
    log: list[str] = []
    h, w = arr01.shape
    body = arr01 > max(0.04, float(np.percentile(arr01, 12)))
    body = ndimage.binary_opening(body, iterations=1)
    body = ndimage.binary_closing(body, iterations=3)
    labeled, n = ndimage.label(body)
    if n == 0:
        return None, None, "invalid", ["roi_no_foreground"]
    objects = ndimage.find_objects(labeled)
    areas = []
    for idx, sl in enumerate(objects, start=1):
        if sl is None:
            areas.append(0)
        else:
            areas.append(int(np.sum(labeled[sl] == idx)))
    component = int(np.argmax(areas) + 1)
    mask = labeled == component
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, "invalid", ["roi_empty_component"]
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad_y = int(0.04 * h)
    pad_x = int(0.04 * w)
    y0, y1 = max(0, y0 - pad_y), min(h - 1, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(w - 1, x1 + pad_x)
    if (y1 - y0) < h * 0.35 or (x1 - x0) < w * 0.25:
        return None, mask.astype(np.uint8), "invalid", ["roi_bbox_too_small"]
    crop = arr01[y0 : y1 + 1, x0 : x1 + 1]
    log.append(f"thorax_roi_bbox={x0},{y0},{x1},{y1}")
    return crop, mask.astype(np.uint8), "valid", log


def resize_pad_array(arr01: np.ndarray, target: int) -> np.ndarray:
    arr01 = np.clip(arr01, 0, 1)
    img = Image.fromarray((arr01 * 255).astype(np.uint8), mode="L")
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (target, target), 0)
    canvas.paste(img, ((target - new_w) // 2, (target - new_h) // 2))
    return np.asarray(canvas).astype(np.float32) / 255.0


def array_to_pil(arr01: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8), mode="L")


def _save_pil_image(img: Image.Image | None, path: Path) -> str | None:
    if img is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, compress_level=1)
    return str(path)


def _save_array_as_png(arr: np.ndarray | None, path: Path) -> str | None:
    if arr is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path, compress_level=1)
    return str(path)


def cache_preprocessed_result(result: PreprocessResult, cfg: NotebookConfig) -> PreprocessResult:
    cache_root = Path(cfg.preprocessed_cache_dir or Path(cfg.artifacts_dir) / "preprocessed_cache")
    study_id = stable_hash(f"{result.source_dataset}:{result.study_id}", n=20)
    result.image_full_path = _save_pil_image(result.image_full, cache_root / "full" / f"{study_id}.png")
    result.image_roi_path = _save_pil_image(result.image_roi, cache_root / "roi" / f"{study_id}.png")
    result.image_eva_path = _save_pil_image(result.image_eva, cache_root / "eva" / f"{study_id}.png")
    result.image_raw_preview_path = _save_array_as_png(result.image_raw_preview, cache_root / "preview" / f"{study_id}.png")
    result.lung_mask_path = None
    result.image_full = None
    result.image_roi = None
    result.image_eva = None
    result.image_raw_preview = None
    result.lung_mask = None
    return result


def _load_pil_image(path: str | None) -> Image.Image | None:
    if not path:
        return None
    with Image.open(path) as img:
        return img.convert("L").copy()


def get_result_image_full(result: PreprocessResult) -> Image.Image:
    if result.image_full is not None:
        return result.image_full
    img = _load_pil_image(result.image_full_path)
    if img is None:
        raise RuntimeError(f"Missing cached full image for study {result.study_id}")
    return img


def get_result_image_roi(result: PreprocessResult) -> Image.Image | None:
    if result.image_roi is not None:
        return result.image_roi
    return _load_pil_image(result.image_roi_path)


def get_result_image_eva(result: PreprocessResult) -> Image.Image:
    if result.image_eva is not None:
        return result.image_eva
    img = _load_pil_image(result.image_eva_path)
    if img is None:
        raise RuntimeError(f"Missing cached EVA image for study {result.study_id}")
    return img


def get_result_raw_preview(result: PreprocessResult) -> np.ndarray:
    if result.image_raw_preview is not None:
        return result.image_raw_preview
    img = _load_pil_image(result.image_raw_preview_path)
    if img is None:
        return np.asarray(get_result_image_full(result)).astype(np.float32) / 255.0
    return np.asarray(img).astype(np.float32) / 255.0


def preprocess_one(row: pd.Series | dict[str, Any], cfg: NotebookConfig) -> PreprocessResult:
    path = str(row["path"])
    raw, metadata, source_type, log = load_image_pixels(path)
    arr01 = robust_normalize(raw)
    quality_score, qa_flags, critical = quality_checks(arr01, metadata)
    roi_arr, mask, roi_status, roi_log = make_thorax_roi(arr01)
    qa_flags.extend([flag for flag in roi_log if flag.startswith("roi_")])
    log.extend(roi_log)
    full = resize_pad_array(arr01, cfg.cxr_foundation_full_size)
    eva = resize_pad_array(arr01, cfg.eva_image_size)
    roi_pil = None
    if roi_arr is not None and roi_status == "valid":
        roi_pil = array_to_pil(resize_pad_array(roi_arr, cfg.cxr_foundation_full_size))
    metadata["rows"] = int(raw.shape[0])
    metadata["columns"] = int(raw.shape[1])
    metadata.setdefault("patient_id_hash", row.get("patient_id_hash", stable_hash(path)))
    return PreprocessResult(
        study_id=str(row.get("study_id", stable_hash(path))),
        source_path=path,
        y_attention=int(row["y_attention"]),
        source_dataset=str(row.get("source_dataset", "unknown")),
        image_full=array_to_pil(full),
        image_roi=roi_pil,
        image_eva=array_to_pil(eva),
        image_raw_preview=arr01,
        lung_mask=mask,
        quality_score=quality_score,
        qa_flags=qa_flags,
        metadata=metadata,
        preprocess_log=log,
        source_type=source_type,
        roi_status=roi_status,
        critical_qa=critical,
    )


def preprocess_dataframe(df: pd.DataFrame, cfg: NotebookConfig, limit: int | None = None) -> tuple[list[PreprocessResult], pd.DataFrame]:
    rows = df.head(limit).to_dict("records") if limit is not None else df.to_dict("records")
    results = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        result = preprocess_one(row, cfg)
        if cfg.cache_preprocessed_to_disk:
            result = cache_preprocessed_result(result, cfg)
        results.append(result)
        if cfg.preprocess_progress_every and (idx == 1 or idx % cfg.preprocess_progress_every == 0 or idx == total):
            print(f"Preprocessed {idx}/{total} studies")
    meta = pd.DataFrame([r.to_record() for r in results])
    meta = meta.merge(df[["study_id", "split"]], on="study_id", how="left")
    return results, meta


def coerce_embedding_output(output: Any) -> np.ndarray:
    if isinstance(output, np.ndarray):
        return output.astype(np.float32)
    if hasattr(output, "error") and getattr(output, "error") not in (None, ""):
        raise RuntimeError(f"CXR Foundation returned an embedding error: {getattr(output, 'error')}")
    if hasattr(output, "general_img_emb") and getattr(output, "general_img_emb") is not None:
        return coerce_embedding_output(getattr(output, "general_img_emb"))
    if hasattr(output, "contrastive_img_emb") and getattr(output, "contrastive_img_emb") is not None:
        return coerce_embedding_output(getattr(output, "contrastive_img_emb"))
    if hasattr(output, "contrastive_txt_emb") and getattr(output, "contrastive_txt_emb") is not None:
        return coerce_embedding_output(getattr(output, "contrastive_txt_emb"))
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            return np.empty((0, 32, 768), dtype=np.float32)
        if any(
            hasattr(x, "general_img_emb")
            or hasattr(x, "contrastive_img_emb")
            or hasattr(x, "contrastive_txt_emb")
            or hasattr(x, "error")
            for x in output
        ):
            return np.asarray([coerce_embedding_output(x) for x in output], dtype=np.float32)
        return np.asarray(output, dtype=np.float32)
    if isinstance(output, dict):
        if output.get("error"):
            raise RuntimeError(f"CXR Foundation returned an embedding error: {output['error']}")
        for key in [
            "general_img_emb",
            "contrastive_img_emb",
            "contrastive_txt_emb",
            "image_embeddings",
            "embeddings",
            "elixr",
            "cxr_model",
        ]:
            if key in output:
                return coerce_embedding_output(output[key])
        arrays = [
            coerce_embedding_output(v)
            for v in output.values()
            if isinstance(v, (list, tuple, np.ndarray, dict))
            or hasattr(v, "general_img_emb")
            or hasattr(v, "contrastive_img_emb")
            or hasattr(v, "contrastive_txt_emb")
        ]
        if arrays:
            return arrays[0]
    raise TypeError(f"Cannot coerce embedding output of type {type(output)!r}")


def pool_tokens(tokens: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim == 1:
        return tokens
    if tokens.ndim == 2:
        mean = tokens.mean(axis=0)
        maxv = tokens.max(axis=0)
        std = tokens.std(axis=0)
        return np.concatenate([mean, maxv, std]).astype(np.float32)
    if tokens.ndim == 3:
        return np.vstack([pool_tokens(x) for x in tokens])
    raise ValueError(f"Unexpected token shape: {tokens.shape}")


def qa_feature_matrix(meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    flag_counts = meta["qa_flags"].fillna("").apply(lambda s: 0 if not s else len([x for x in str(s).split("|") if x]))
    roi_missing = (meta["roi_status"] != "valid").astype(float)
    critical = meta["critical_qa"].astype(float)
    rows = pd.to_numeric(meta["rows"], errors="coerce").fillna(0)
    cols = pd.to_numeric(meta["columns"], errors="coerce").fillna(0)
    X = np.vstack(
        [
            meta["quality_score"].astype(float).values,
            flag_counts.astype(float).values,
            roi_missing.values,
            critical.values,
            np.log1p(rows.values),
            np.log1p(cols.values),
        ]
    ).T.astype(np.float32)
    names = ["quality_score", "qa_flag_count", "roi_missing", "critical_qa", "log_rows", "log_cols"]
    return X, names


def build_fusion_features(
    full_tokens: np.ndarray,
    roi_tokens: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    full = pool_tokens(full_tokens)
    roi = pool_tokens(roi_tokens)
    qa, qa_names = qa_feature_matrix(meta)
    X = np.concatenate([full, roi, qa], axis=1).astype(np.float32)
    names = (
        [f"full_pool_{i}" for i in range(full.shape[1])]
        + [f"roi_pool_{i}" for i in range(roi.shape[1])]
        + [f"qa_{n}" for n in qa_names]
    )
    return X, names


def make_logistic_pipeline(pca_components: int | None, n_samples: int, n_features: int, seed: int = 42) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if pca_components is not None:
        n_comp = min(int(pca_components), max(1, n_samples - 1), n_features)
        if n_comp < n_features:
            steps.append(("pca", PCA(n_components=n_comp, random_state=seed)))
    clf = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.10,
        solver="saga",
        C=1.0,
        class_weight="balanced",
        max_iter=4000,
        random_state=seed,
        n_jobs=-1,
    )
    steps.append(("clf", clf))
    return Pipeline(steps)


class ConstantProbabilityModel:
    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full(len(X), self.p, dtype=np.float32)
        return np.vstack([1 - p, p]).T


class SigmoidCalibratedModel:
    def __init__(self, base_model: Any):
        self.base_model = base_model
        self.calibrator = LogisticRegression(max_iter=1000)
        self.ready = False

    def fit(self, X_calib: np.ndarray, y_calib: np.ndarray) -> "SigmoidCalibratedModel":
        y_calib = np.asarray(y_calib).astype(int)
        p = np.clip(self.base_model.predict_proba(X_calib)[:, 1], 1e-5, 1 - 1e-5)
        if len(y_calib) >= 4 and len(np.unique(y_calib)) == 2:
            logits = np.log(p / (1 - p)).reshape(-1, 1)
            self.calibrator.fit(logits, y_calib)
            self.ready = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.base_model.predict_proba(X)[:, 1], 1e-5, 1 - 1e-5)
        if self.ready:
            logits = np.log(p / (1 - p)).reshape(-1, 1)
            p = self.calibrator.predict_proba(logits)[:, 1]
        return np.vstack([1 - p, p]).T


def calibrate_prefit(model: Any, X_calib: np.ndarray, y_calib: np.ndarray, method: str = "sigmoid") -> Any:
    y_calib = np.asarray(y_calib).astype(int)
    if len(y_calib) < 4 or len(np.unique(y_calib)) < 2:
        return model
    return SigmoidCalibratedModel(model).fit(X_calib, y_calib)


def train_logistic_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray | None = None,
    y_calib: np.ndarray | None = None,
    pca_components: int | None = 256,
    seed: int = 42,
) -> Any:
    y_train = np.asarray(y_train).astype(int)
    if len(np.unique(y_train)) < 2:
        return ConstantProbabilityModel(float(y_train.mean()))
    model = make_logistic_pipeline(pca_components, len(X_train), X_train.shape[1], seed)
    model.fit(X_train, y_train)
    if X_calib is not None and y_calib is not None:
        model = calibrate_prefit(model, X_calib, y_calib, method="sigmoid")
    return model


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(32, hidden // 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden // 4), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    hidden: int = 256,
    dropout: float = 0.20,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> TorchMLP:
    if torch is None:
        raise RuntimeError("torch is required for MLP training.")
    set_seed(seed)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32) if X_val is not None else None
    model = TorchMLP(X_train.shape[1], hidden=hidden, dropout=dropout).to(device)
    model.scaler = scaler  # type: ignore[attr-defined]
    x = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=device)
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_loss = float("inf")
    patience = 12
    bad = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
        val_loss = float(loss.detach().cpu())
        if X_val_s is not None and y_val is not None and len(y_val) > 0:
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val_s, dtype=torch.float32, device=device)
                yv = torch.tensor(y_val.astype(np.float32), dtype=torch.float32, device=device)
                val_loss = float(F.binary_cross_entropy_with_logits(model(xv), yv).detach().cpu())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_torch_mlp(model: TorchMLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    scaler = getattr(model, "scaler", None)
    Xs = scaler.transform(X).astype(np.float32) if scaler is not None else X.astype(np.float32)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(Xs, dtype=torch.float32, device=device)
        p = torch.sigmoid(model.to(device)(x)).detach().cpu().numpy()
    return p.astype(np.float32)


class ProbabilityCalibrator:
    def __init__(self, method: str = "platt"):
        self.method = str(method).lower()
        self.model = None
        self.ready = False

    def fit(self, p: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
        y = np.asarray(y).astype(int)
        if self.method in {"none", "identity", "raw"}:
            self.ready = False
            return self
        if len(y) < 4 or len(np.unique(y)) != 2:
            self.ready = False
            return self
        if self.method == "platt":
            logits = np.log(p / (1 - p)).reshape(-1, 1)
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(logits, y)
            self.ready = True
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(p, y)
            self.ready = True
        else:
            raise ValueError(f"Unknown calibration method: {self.method!r}")
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
        if not self.ready or self.model is None:
            return p
        if self.method == "isotonic":
            return np.asarray(self.model.predict(p), dtype=np.float32)
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


class PlattCalibrator(ProbabilityCalibrator):
    def __init__(self):
        super().__init__(method="platt")


def predict_proba_any(model: Any, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    if isinstance(model, TorchMLP):
        p = predict_torch_mlp(model, X, device=device)
        return np.vstack([1 - p, p]).T
    return model.predict_proba(X)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(ece)


def metrics_summary(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    out = {
        "n": float(len(y)),
        "prevalence": float(np.mean(y)) if len(y) else float("nan"),
        "auroc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ece": expected_calibration_error(y, p),
        "balanced_accuracy@0.5": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
    }
    return out


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return float("nan")
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return float(max(0.0, (centre - margin) / denom))


def threshold_report(y: np.ndarray, p: np.ndarray, target_npv: float = 0.99, n_grid: int = 501) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    rows = []
    thresholds = np.unique(np.quantile(p, np.linspace(0, 1, int(n_grid))))
    for t in thresholds:
        selected = p <= t
        n_sel = int(np.sum(selected))
        if n_sel == 0:
            continue
        fn = int(np.sum((y == 1) & selected))
        tn = int(np.sum((y == 0) & selected))
        npv = tn / max(tn + fn, 1)
        npv_ci95_low = wilson_lower_bound(tn, tn + fn, z=1.96)
        coverage = n_sel / len(y)
        rows.append(
            {
                "T_negative": float(t),
                "selected_count": n_sel,
                "TN_count": tn,
                "no_attention_required_coverage": float(coverage),
                "NPV": float(npv),
                "NPV_ci95_low": float(npv_ci95_low),
                "FN_count": fn,
                "FN_per_1000": float(fn / max(n_sel, 1) * 1000.0),
                "N/A_rate_if_positive_threshold_unused": float(1.0 - coverage),
                "meets_target_NPV": bool(npv >= target_npv),
            }
        )
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["meets_target_NPV", "no_attention_required_coverage"], ascending=[False, False])
    return report.reset_index(drop=True)


def choose_negative_threshold(report: pd.DataFrame, fallback_quantile: float = 0.15) -> float:
    if report.empty:
        return 0.05
    ok = report[report["meets_target_NPV"]]
    if not ok.empty:
        return float(ok.iloc[0]["T_negative"])
    return float(report["T_negative"].quantile(fallback_quantile))


def route_decisions(
    p: np.ndarray,
    meta: pd.DataFrame,
    t_negative: float,
    t_positive: float = 0.80,
    t_quality: float = 0.35,
    ood_score: np.ndarray | None = None,
    t_ood: float = 0.95,
    t_uncertainty: float = 0.65,
) -> pd.DataFrame:
    rows = []
    for i, prob in enumerate(np.asarray(p)):
        quality = float(meta.iloc[i]["quality_score"])
        critical = bool(meta.iloc[i]["critical_qa"])
        uncertainty = float(1.0 - abs(prob - 0.5) * 2.0)
        ood = float(ood_score[i]) if ood_score is not None else None
        if quality < t_quality or critical:
            route, reason = "N/A", "bad_quality_or_critical_qa"
        elif ood is not None and ood > t_ood:
            route, reason = "N/A", "out_of_distribution"
        elif prob <= t_negative:
            route, reason = "no_attention_required", "confident_no_attention_required"
        elif prob >= t_positive:
            route, reason = "requires_attention", "suspicious_requires_attention"
        elif uncertainty > t_uncertainty:
            route, reason = "N/A", "high_uncertainty"
        else:
            route, reason = "N/A", "gray_zone"
        rows.append(
            {
                "study_id": meta.iloc[i]["study_id"],
                "p_requires_attention": float(prob),
                "quality_score": quality,
                "ood_score": ood,
                "uncertainty_score": uncertainty,
                "route": route,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def transform_bbox_to_padded_square(
    bbox: dict[str, Any] | pd.Series,
    original_h: int,
    original_w: int,
    target: int,
) -> tuple[float, float, float, float]:
    scale = target / max(float(original_w), float(original_h))
    new_w = float(original_w) * scale
    new_h = float(original_h) * scale
    pad_x = (target - new_w) / 2.0
    pad_y = (target - new_h) / 2.0
    x0 = float(bbox["x_min"]) * scale + pad_x
    y0 = float(bbox["y_min"]) * scale + pad_y
    x1 = float(bbox["x_max"]) * scale + pad_x
    y1 = float(bbox["y_max"]) * scale + pad_y
    return (
        float(np.clip(x0, 0, target - 1)),
        float(np.clip(y0, 0, target - 1)),
        float(np.clip(x1, 0, target - 1)),
        float(np.clip(y1, 0, target - 1)),
    )


def bbox_mask_for_result(
    result: PreprocessResult,
    bbox_df: pd.DataFrame,
    target: int,
) -> np.ndarray:
    mask = np.zeros((target, target), dtype=np.float32)
    if bbox_df is None or bbox_df.empty:
        return mask
    raw_preview = get_result_raw_preview(result)
    fallback_rows = int(result.metadata.get("rows", raw_preview.shape[0]))
    fallback_cols = int(result.metadata.get("columns", raw_preview.shape[1]))
    part = bbox_df[bbox_df["study_id"].astype(str) == str(result.study_id)]
    for _, bbox in part.iterrows():
        if not pd.notna(bbox.get("x_min")):
            continue
        class_name = str(bbox.get("class_name", "")).lower()
        if class_name == "no finding":
            continue
        bbox_rows = pd.to_numeric(pd.Series([bbox.get("bbox_original_rows", np.nan)]), errors="coerce").iloc[0]
        bbox_cols = pd.to_numeric(pd.Series([bbox.get("bbox_original_columns", np.nan)]), errors="coerce").iloc[0]
        rows = int(bbox_rows) if pd.notna(bbox_rows) and float(bbox_rows) > 0 else fallback_rows
        cols = int(bbox_cols) if pd.notna(bbox_cols) and float(bbox_cols) > 0 else fallback_cols
        x0, y0, x1, y1 = transform_bbox_to_padded_square(bbox, rows, cols, target)
        x0i, y0i, x1i, y1i = map(lambda v: int(round(v)), [x0, y0, x1, y1])
        if x1i > x0i and y1i > y0i:
            mask[y0i : y1i + 1, x0i : x1i + 1] = 1.0
    return mask


def heatmap_localization_metrics(heatmap: np.ndarray, bbox_mask: np.ndarray) -> dict[str, float]:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    bbox_mask = np.asarray(bbox_mask, dtype=np.float32)
    if heatmap.shape != bbox_mask.shape:
        zoom_y = bbox_mask.shape[0] / heatmap.shape[0]
        zoom_x = bbox_mask.shape[1] / heatmap.shape[1]
        heatmap = ndimage.zoom(heatmap, (zoom_y, zoom_x), order=1)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    has_bbox = bool(np.sum(bbox_mask) > 0)
    if not has_bbox:
        return {
            "has_bbox": 0.0,
            "pointing_game_hit": float("nan"),
            "energy_inside_bbox": float("nan"),
            "bbox_iou_at_top20pct": float("nan"),
        }
    max_y, max_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    pointing = float(bbox_mask[max_y, max_x] > 0)
    energy = float(np.sum(heatmap * bbox_mask) / max(np.sum(heatmap), 1e-8))
    threshold = float(np.quantile(heatmap, 0.80))
    hm_bin = heatmap >= threshold
    bbox_bin = bbox_mask > 0
    inter = float(np.logical_and(hm_bin, bbox_bin).sum())
    union = float(np.logical_or(hm_bin, bbox_bin).sum())
    iou = inter / max(union, 1.0)
    return {
        "has_bbox": 1.0,
        "pointing_game_hit": pointing,
        "energy_inside_bbox": energy,
        "bbox_iou_at_top20pct": iou,
    }


def make_occlusion_heatmap_from_feature_builder(
    result: PreprocessResult,
    base_meta: pd.DataFrame,
    embed_full_fn: Callable[[list[Image.Image]], np.ndarray],
    original_roi_tokens: np.ndarray,
    predict_feature_fn: Callable[[np.ndarray], float],
    grid: int = 8,
    fill_value: int = 0,
) -> np.ndarray:
    full_image = get_result_image_full(result)
    target = full_image.size[0]
    patch = target // grid
    occluded_images = []
    for gy in range(grid):
        for gx in range(grid):
            img = full_image.copy()
            arr = np.asarray(img).copy()
            y0, y1 = gy * patch, target if gy == grid - 1 else (gy + 1) * patch
            x0, x1 = gx * patch, target if gx == grid - 1 else (gx + 1) * patch
            arr[y0:y1, x0:x1] = fill_value
            occluded_images.append(Image.fromarray(arr.astype(np.uint8), mode="L"))
    base_full = embed_full_fn([full_image])
    base_X, _ = build_fusion_features(base_full, original_roi_tokens[None, ...], base_meta)
    base_score = float(predict_feature_fn(base_X)[0])
    occ_full = embed_full_fn(occluded_images)
    roi_repeated = np.repeat(original_roi_tokens[None, ...], len(occluded_images), axis=0)
    meta_rep = pd.concat([base_meta] * len(occluded_images), ignore_index=True)
    occ_X, _ = build_fusion_features(occ_full, roi_repeated, meta_rep)
    occ_scores = np.asarray(predict_feature_fn(occ_X), dtype=np.float32)
    drops = np.clip(base_score - occ_scores, 0, None).reshape(grid, grid)
    if drops.max() > 0:
        drops = drops / drops.max()
    heatmap = ndimage.zoom(drops, (target / grid, target / grid), order=1)
    return heatmap[:target, :target].astype(np.float32)


def fit_ood_model(X_train: np.ndarray) -> dict[str, Any]:
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    nn_model = NearestNeighbors(n_neighbors=min(5, len(Xs))).fit(Xs)
    dists, _ = nn_model.kneighbors(Xs)
    ref = dists[:, -1]
    iso = IsolationForest(random_state=42, contamination="auto").fit(Xs)
    iso_raw = -iso.score_samples(Xs)
    return {
        "scaler": scaler,
        "nn": nn_model,
        "ref95": float(np.percentile(ref, 95)),
        "iso": iso,
        "iso_p5": float(np.percentile(iso_raw, 5)),
        "iso_p95": float(np.percentile(iso_raw, 95)),
    }


def ood_score(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    required = {"scaler", "nn", "ref95", "iso", "iso_p5", "iso_p95"}
    missing = sorted(required.difference(model))
    if missing:
        raise ValueError(f"OOD model is incomplete; missing training-reference fields: {missing}")
    Xs = model["scaler"].transform(X)
    dists, _ = model["nn"].kneighbors(Xs)
    knn = dists[:, -1] / max(model["ref95"], 1e-6)
    iso_raw = -model["iso"].score_samples(Xs)
    iso = (iso_raw - model["iso_p5"]) / max(model["iso_p95"] - model["iso_p5"], 1e-6)
    return np.clip(0.5 * knn + 0.5 * iso, 0, 2)


def image_to_eva_tensor(img: Image.Image, image_size: int = 224) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("torch is required.")
    img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)


class EVAEndToEndClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, hidden: int = 128, dropout: float = 0.20):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            z = self.encoder.forward_features(x)
            if z.ndim == 3:
                z = z[:, 1:, :].mean(dim=1) if z.shape[1] > 1 else z.mean(dim=1)
        else:
            z = self.encoder(x)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)


EVA_X_VARIANTS: dict[str, dict[str, str]] = {
    "tiny": {
        "factory": "eva_x_tiny_patch16",
        "filename": "eva_x_tiny_patch16_merged520k_mim.pt",
    },
    "ti": {
        "factory": "eva_x_tiny_patch16",
        "filename": "eva_x_tiny_patch16_merged520k_mim.pt",
    },
    "small": {
        "factory": "eva_x_small_patch16",
        "filename": "eva_x_small_patch16_merged520k_mim.pt",
    },
    "s": {
        "factory": "eva_x_small_patch16",
        "filename": "eva_x_small_patch16_merged520k_mim.pt",
    },
    "base": {
        "factory": "eva_x_base_patch16",
        "filename": "eva_x_base_patch16_merged520k_mim.pt",
    },
    "b": {
        "factory": "eva_x_base_patch16",
        "filename": "eva_x_base_patch16_merged520k_mim.pt",
    },
}


def load_real_eva_x(
    project_dir: str,
    variant: str = "small",
    device: str = "cpu",
    weights_path: str | None = None,
) -> Any:
    variant_key = str(variant).lower().strip()
    if variant_key not in EVA_X_VARIANTS:
        raise ValueError(f"Unknown EVA-X variant {variant!r}. Use one of: {sorted(EVA_X_VARIANTS)}")
    spec = EVA_X_VARIANTS[variant_key]
    repo_dir = Path(project_dir) / "external" / "EVA-X"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/hustvl/EVA-X.git", str(repo_dir)], check=True)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    try:
        import timm  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm==0.9.0"], check=True)
    if weights_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
            from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id="MapleF/eva_x",
            filename=spec["filename"],
            local_dir=str(Path(project_dir) / "models" / "eva_x"),
        )
    import eva_x

    factory = getattr(eva_x, spec["factory"])

    original_torch_load = torch.load

    def torch_load_eva_x_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        try:
            return original_torch_load(*args, **kwargs)
        except TypeError as exc:
            if "weights_only" in str(exc):
                kwargs.pop("weights_only", None)
                return original_torch_load(*args, **kwargs)
            raise

    torch.load = torch_load_eva_x_compat
    try:
        model = factory(pretrained=weights_path)
    finally:
        torch.load = original_torch_load
    _patch_eva_x_forward_features_compat(model)
    model.eval().to(device)
    return model


def load_real_eva_x_s(project_dir: str, device: str = "cpu", weights_path: str | None = None) -> Any:
    return load_real_eva_x(project_dir=project_dir, variant="small", device=device, weights_path=weights_path)


def _patch_eva_x_forward_features_compat(model: Any) -> None:
    if torch is None or hasattr(model, "_pos_embed"):
        return

    def forward_features_compat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if getattr(self, "cls_token", None) is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = getattr(self, "pos_embed", None)
        if pos_embed is not None:
            x = x + pos_embed
        pos_drop = getattr(self, "pos_drop", None)
        if pos_drop is not None:
            x = pos_drop(x)
        rope = getattr(self, "rope", None)
        rot_pos_embed = rope.get_embed() if rope is not None else None
        patch_drop = getattr(self, "patch_drop", None)
        if patch_drop is not None:
            dropped = patch_drop(x)
            if isinstance(dropped, tuple):
                x = dropped[0]
            else:
                x = dropped
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    model.forward_features = forward_features_compat.__get__(model, model.__class__)


def extract_eva_features_from_images(
    model: Any,
    images: list[Image.Image],
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required.")
    feats = []
    model.eval().to(device)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch = torch.stack([
                image_to_eva_tensor(img, image_size=image_size)
                for img in batch_images
            ]).to(device)
            if hasattr(model, "forward_features"):
                z = model.forward_features(batch)
                if z.ndim == 3:
                    z = z[:, 1:, :].mean(dim=1) if z.shape[1] > 1 else z.mean(dim=1)
            else:
                z = model(batch)
            feats.append(z.detach().float().cpu().numpy())
            del batch, z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_eva_features_real(
    model: Any,
    results: list[PreprocessResult],
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required.")
    feats = []
    model.eval().to(device)
    with torch.no_grad():
        for start in range(0, len(results), batch_size):
            batch_results = results[start : start + batch_size]
            batch = torch.stack([
                image_to_eva_tensor(get_result_image_eva(r), image_size=image_size)
                for r in batch_results
            ]).to(device)
            if hasattr(model, "forward_features"):
                z = model.forward_features(batch)
                if z.ndim == 3:
                    z = z[:, 1:, :].mean(dim=1) if z.shape[1] > 1 else z.mean(dim=1)
            else:
                z = model(batch)
            feats.append(z.detach().float().cpu().numpy())
            del batch, z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0).astype(np.float32)


def make_eva_occlusion_heatmap(
    model: Any,
    predict_feature_fn: Callable[[np.ndarray], np.ndarray],
    result: PreprocessResult,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cpu",
    grid: int = 8,
    fill_value: int = 0,
) -> np.ndarray:
    img = get_result_image_eva(result).convert("L").resize((image_size, image_size), Image.BILINEAR)
    target = img.size[0]
    patch = max(1, target // grid)
    occluded_images = []
    for gy in range(grid):
        for gx in range(grid):
            arr = np.asarray(img).copy()
            y0, y1 = gy * patch, target if gy == grid - 1 else (gy + 1) * patch
            x0, x1 = gx * patch, target if gx == grid - 1 else (gx + 1) * patch
            arr[y0:y1, x0:x1] = fill_value
            occluded_images.append(Image.fromarray(arr.astype(np.uint8), mode="L"))
    base_features = extract_eva_features_from_images(
        model,
        [img],
        image_size=image_size,
        batch_size=1,
        device=device,
    )
    base_score = float(np.asarray(predict_feature_fn(base_features))[0])
    occ_features = extract_eva_features_from_images(
        model,
        occluded_images,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
    )
    occ_scores = np.asarray(predict_feature_fn(occ_features), dtype=np.float32)
    drops = np.clip(base_score - occ_scores, 0, None).reshape(grid, grid)
    if drops.max() > 0:
        drops = drops / drops.max()
    heatmap = ndimage.zoom(drops, (target / grid, target / grid), order=1)
    return heatmap[:target, :target].astype(np.float32)


def predict_eva_end_to_end_images(
    model: nn.Module,
    images: list[Image.Image],
    image_size: int = 224,
    batch_size: int = 4,
    device: str = "cpu",
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required.")
    preds = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            xb = torch.stack([image_to_eva_tensor(img, image_size) for img in batch_images]).to(device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xb)
            preds.append(torch.sigmoid(logits).detach().float().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def make_eva_end_to_end_occlusion_heatmap(
    model: nn.Module,
    result: PreprocessResult,
    image_size: int = 224,
    batch_size: int = 4,
    device: str = "cpu",
    grid: int = 8,
    fill_value: int = 0,
) -> np.ndarray:
    img = get_result_image_eva(result).convert("L").resize((image_size, image_size), Image.BILINEAR)
    target = img.size[0]
    patch = max(1, target // grid)
    occluded_images = []
    for gy in range(grid):
        for gx in range(grid):
            arr = np.asarray(img).copy()
            y0, y1 = gy * patch, target if gy == grid - 1 else (gy + 1) * patch
            x0, x1 = gx * patch, target if gx == grid - 1 else (gx + 1) * patch
            arr[y0:y1, x0:x1] = fill_value
            occluded_images.append(Image.fromarray(arr.astype(np.uint8), mode="L"))
    base_score = float(
        predict_eva_end_to_end_images(
            model,
            [img],
            image_size=image_size,
            batch_size=1,
            device=device,
        )[0]
    )
    occ_scores = predict_eva_end_to_end_images(
        model,
        occluded_images,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
    )
    drops = np.clip(base_score - occ_scores, 0, None).reshape(grid, grid)
    if drops.max() > 0:
        drops = drops / drops.max()
    heatmap = ndimage.zoom(drops, (target / grid, target / grid), order=1)
    return heatmap[:target, :target].astype(np.float32)


def make_eva_end_to_end_calibrated_occlusion_heatmap(
    model: nn.Module,
    calibrator: Any,
    result: PreprocessResult,
    image_size: int = 224,
    batch_size: int = 4,
    device: str = "cpu",
    grid: int = 8,
    fill_value: int = 0,
) -> np.ndarray:
    img = get_result_image_eva(result).convert("L").resize((image_size, image_size), Image.BILINEAR)
    target = img.size[0]
    patch = max(1, target // grid)
    occluded_images = []
    for gy in range(grid):
        for gx in range(grid):
            arr = np.asarray(img).copy()
            y0, y1 = gy * patch, target if gy == grid - 1 else (gy + 1) * patch
            x0, x1 = gx * patch, target if gx == grid - 1 else (gx + 1) * patch
            arr[y0:y1, x0:x1] = fill_value
            occluded_images.append(Image.fromarray(arr.astype(np.uint8), mode="L"))

    base_raw = predict_eva_end_to_end_images(
        model,
        [img],
        image_size=image_size,
        batch_size=1,
        device=device,
    )
    occ_raw = predict_eva_end_to_end_images(
        model,
        occluded_images,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
    )
    if calibrator is not None:
        base_score = float(np.asarray(calibrator.transform(base_raw), dtype=np.float32)[0])
        occ_scores = np.asarray(calibrator.transform(occ_raw), dtype=np.float32)
    else:
        base_score = float(base_raw[0])
        occ_scores = np.asarray(occ_raw, dtype=np.float32)
    drops = np.clip(base_score - occ_scores, 0, None).reshape(grid, grid)
    if drops.max() > 0:
        drops = drops / drops.max()
    heatmap = ndimage.zoom(drops, (target / grid, target / grid), order=1)
    return heatmap[:target, :target].astype(np.float32)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.dropout = nn.Dropout(dropout)
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros((r, base.in_features), device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros((base.out_features, r), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        for p in self.base.parameters():
            p.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return base + update


def inject_lora_last_blocks(
    model: nn.Module,
    n_last_blocks: int = 2,
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_names: tuple[str, ...] = ("qkv", "q_proj", "v_proj", "proj", "fc1", "fc2"),
) -> int:
    if torch is None:
        raise RuntimeError("torch is required.")
    for p in model.parameters():
        p.requires_grad = False
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        blocks = [model]
    target_blocks = list(blocks)[-n_last_blocks:]
    replaced = 0

    def replace_in_module(module: nn.Module, prefix: str = "") -> None:
        nonlocal replaced
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(key in full for key in target_names):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
            else:
                replace_in_module(child, full)

    for block in target_blocks:
        replace_in_module(block)
    return replaced


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def load_eva_end_to_end_checkpoint(
    project_dir: str,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 224,
) -> tuple[nn.Module, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("torch is required.")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    variant = str(state.get("variant") or state.get("adapt_cfg", {}).get("variant") or "").lower()
    kind = str(state.get("kind") or state.get("mode") or "").lower()
    adapt_cfg = dict(state.get("adapt_cfg") or {})
    mode = str(state.get("mode") or ("lora" if "lora" in kind else "partial_unfreeze" if "partial" in kind else ""))
    if variant not in EVA_X_VARIANTS:
        raise ValueError(f"Checkpoint has no supported EVA variant: {variant!r}")
    if mode not in {"lora", "partial_unfreeze"}:
        raise ValueError(f"Checkpoint is not an EVA adapted model: kind={kind!r}, mode={mode!r}")

    encoder = load_real_eva_x(project_dir, variant=variant, device=device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if mode == "lora":
        replaced = inject_lora_last_blocks(
            encoder,
            n_last_blocks=int(adapt_cfg["n_last_blocks"]),
            r=int(adapt_cfg["rank"]),
            alpha=float(adapt_cfg["alpha"]),
            dropout=float(adapt_cfg.get("dropout", 0.0)),
        )
        if replaced <= 0:
            raise RuntimeError("No EVA Linear modules were replaced while reconstructing the LoRA checkpoint.")

    state_dict = state["state_dict"]
    head_weight = state_dict.get("head.1.weight")
    if head_weight is None:
        raise KeyError("EVA checkpoint is missing head.1.weight; cannot infer head dimensions.")
    hidden, feature_dim = map(int, head_weight.shape)
    model = EVAEndToEndClassifier(
        encoder,
        feature_dim=feature_dim,
        hidden=hidden,
        dropout=float(adapt_cfg.get("head_dropout", 0.20)),
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"EVA checkpoint structure mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, {
        "variant": variant,
        "kind": kind,
        "mode": mode,
        "adapt_cfg": adapt_cfg,
        "image_size": int(image_size),
    }


def benchmark_predict(fn: Callable[[], Any], repeats: int = 20) -> dict[str, float]:
    times = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return {
        "latency_ms_mean": float(np.mean(times) * 1000),
        "latency_ms_p95": float(np.percentile(times, 95) * 1000),
    }


def export_json(data: dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return str(path)


def split_arrays_by_meta(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    require_all_splits: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for split in ["train", "calibration", "validation", "final_test"]:
        mask = meta["split"].fillna("train").values == split
        if np.any(mask):
            out[split] = (X[mask], y[mask], np.where(mask)[0])
    if require_all_splits:
        missing = [split for split in ["train", "calibration", "validation", "final_test"] if split not in out]
        if missing:
            raise ValueError(f"Missing required array splits: {missing}")
        for split, (_, y_split, _) in out.items():
            if set(np.asarray(y_split).astype(int).tolist()) != {0, 1}:
                raise ValueError(f"Split {split!r} does not contain both classes.")
        return out
    if "validation" not in out:
        out["validation"] = out.get("calibration", out["train"])
    if "calibration" not in out:
        out["calibration"] = out["validation"]
    if "final_test" not in out:
        out["final_test"] = out["validation"]
    return out


def make_model_report(name: str, y: np.ndarray, p: np.ndarray, meta: pd.DataFrame, target_npv: float) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    metrics = metrics_summary(y, p)
    thr = threshold_report(y, p, target_npv=target_npv)
    t_neg = choose_negative_threshold(thr)
    routes = route_decisions(p, meta, t_negative=t_neg)
    metrics["selected_T_negative"] = float(t_neg)
    metrics["auto_negative_coverage"] = float(np.mean(routes["route"] == "no_attention_required"))
    metrics["N/A_rate"] = float(np.mean(routes["route"] == "N/A"))
    metrics["requires_attention_rate"] = float(np.mean(routes["route"] == "requires_attention"))
    metrics["model"] = name
    return metrics, thr, routes


def route_metrics(y: np.ndarray, routes: pd.DataFrame) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    auto = routes["route"].values == "no_attention_required"
    manual = routes["route"].values == "N/A"
    attention = routes["route"].values == "requires_attention"
    fn_auto = int(np.sum((y == 1) & auto))
    tn_auto = int(np.sum((y == 0) & auto))
    fp_attention = int(np.sum((y == 0) & attention))
    fn_manual = int(np.sum((y == 1) & manual))
    return {
        "route_n": float(len(y)),
        "auto_negative_coverage": float(np.mean(auto)) if len(y) else float("nan"),
        "N/A_rate": float(np.mean(manual)) if len(y) else float("nan"),
        "requires_attention_rate": float(np.mean(attention)) if len(y) else float("nan"),
        "auto_negative_NPV": float(tn_auto / max(tn_auto + fn_auto, 1)),
        "unsafe_FN_auto_negative": float(fn_auto),
        "unsafe_FN_per_1000_auto_negative": float(fn_auto / max(np.sum(auto), 1) * 1000.0),
        "safe_FN_in_N/A": float(fn_manual),
        "workload_FP_requires_attention": float(fp_attention),
    }


def fixed_threshold_evaluation(
    name: str,
    y: np.ndarray,
    p: np.ndarray,
    meta: pd.DataFrame,
    t_negative: float,
    t_positive: float = 0.80,
    t_quality: float = 0.35,
    t_uncertainty: float = 0.65,
    t_ood: float = 0.95,
    ood_score_values: np.ndarray | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    routes = route_decisions(
        p,
        meta,
        t_negative=t_negative,
        t_positive=t_positive,
        t_quality=t_quality,
        ood_score=ood_score_values,
        t_ood=t_ood,
        t_uncertainty=t_uncertainty,
    )
    metrics = metrics_summary(y, p)
    metrics.update(route_metrics(y, routes))
    y_arr = np.asarray(y).astype(int)
    auto = routes["route"].values == "no_attention_required"
    selected_count = int(np.sum(auto))
    fn = int(np.sum((y_arr == 1) & auto))
    tn = int(np.sum((y_arr == 0) & auto))
    metrics["fixed_threshold_selected_count"] = selected_count
    metrics["fixed_threshold_TN_count"] = tn
    metrics["fixed_threshold_FN_count"] = fn
    metrics["fixed_threshold_NPV_ci95_low"] = wilson_lower_bound(tn, tn + fn, z=1.96)
    metrics["model"] = name
    metrics["fixed_T_negative"] = float(t_negative)
    return metrics, routes


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        try:
            value = float(metric_fn(y[idx], p[idx]))
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            values.append(value)
    if not values:
        point = float(metric_fn(y, p))
        return point, float("nan"), float("nan")
    point = float(metric_fn(y, p))
    return point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def npv_at_threshold(y: np.ndarray, p: np.ndarray, t_negative: float) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    selected = p <= t_negative
    if not np.any(selected):
        return float("nan")
    fn = np.sum((y == 1) & selected)
    tn = np.sum((y == 0) & selected)
    return float(tn / max(tn + fn, 1))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    rows = []
    edges = np.linspace(0, 1, n_bins + 1)
    for bin_id, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        rows.append(
            {
                "bin": bin_id,
                "prob_low": float(lo),
                "prob_high": float(hi),
                "n": int(np.sum(mask)),
                "mean_pred": float(np.mean(p[mask])) if np.any(mask) else float("nan"),
                "empirical_rate": float(np.mean(y[mask])) if np.any(mask) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def subgroup_route_report(
    y: np.ndarray,
    p: np.ndarray,
    routes: pd.DataFrame,
    meta: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    rows = []
    work = meta.reset_index(drop=True).copy()
    work["_y"] = np.asarray(y).astype(int)
    work["_p"] = np.asarray(p).astype(float)
    work["_route"] = routes["route"].values
    for group_value, part in work.groupby(group_col, dropna=False):
        idx = part.index.values
        sub_routes = routes.iloc[idx].reset_index(drop=True)
        sub_y = part["_y"].values
        sub_p = part["_p"].values
        row = {
            "group_col": group_col,
            "group_value": str(group_value),
            "n": int(len(part)),
            "prevalence": float(np.mean(sub_y)) if len(part) else float("nan"),
            "auroc": safe_auc(sub_y, sub_p),
            "auprc": safe_auprc(sub_y, sub_p),
        }
        row.update(route_metrics(sub_y, sub_routes))
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_group_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    group_slices: dict[str, slice],
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = safe_auc,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_p = predict_fn(X)
    base_metric = float(metric_fn(y, base_p))
    rows = []
    for group, sl in group_slices.items():
        values = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, sl] = Xp[perm, sl]
            values.append(float(metric_fn(y, predict_fn(Xp))))
        rows.append(
            {
                "feature_group": group,
                "base_metric": base_metric,
                "permuted_metric_mean": float(np.nanmean(values)),
                "metric_drop": float(base_metric - np.nanmean(values)),
                "n_repeats": int(n_repeats),
            }
        )
    return pd.DataFrame(rows).sort_values("metric_drop", ascending=False).reset_index(drop=True)
