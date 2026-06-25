from __future__ import annotations

import textwrap
import copy
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
CORE_SOURCE = (ROOT / "fluoro_mvp_core.py").read_text(encoding="utf-8")
MODEL_SELECTION_PATH = ROOT / "fluoro_mvp_model_selection.ipynb"
VINDR_INTERPRETATION_PATH = ROOT / "fluoro_mvp_vindr_interpretation.ipynb"
RANKING_CLOSURE_PATH = ROOT / "fluoro_mvp_ranking_closure.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def notebook(cells: list):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    }
    nb["cells"] = cells
    return nb


COMMON_SETUP = """
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def pip_install(package: str) -> None:
    print(f"Installing {package} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)

def pip_install_upgrade(package: str) -> None:
    print(f"Upgrading {package} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", package], check=True)

def ensure_distribution_min_version(distribution: str, package_spec: str, min_version: str) -> None:
    try:
        current = version(distribution)
    except PackageNotFoundError:
        pip_install(package_spec)
        return
    if Version(current) < Version(min_version):
        pip_install_upgrade(package_spec)

BASE_DEPS = [
    ("pydicom", "pydicom>=2.4.4"),
    ("joblib", "joblib>=1.3.0"),
    ("scipy", "scipy>=1.10.0"),
    ("sklearn", "scikit-learn>=1.2.0"),
    ("PIL", "Pillow>=9.5.0"),
    ("tqdm", "tqdm>=4.66.0"),
    ("matplotlib", "matplotlib>=3.7.0"),
    ("huggingface_hub", "huggingface_hub>=0.23.0"),
    ("safetensors", "safetensors>=0.4.0"),
    ("cloudpickle", "cloudpickle>=2.2.0"),
    ("tabulate", "tabulate>=0.9.0"),
]

if os.environ.get("FLUORO_SKIP_INSTALLS", "0") != "1":
    for import_name, package in BASE_DEPS:
        if not has_module(import_name):
            pip_install(package)
    ensure_distribution_min_version("ml-dtypes", "ml-dtypes>=0.5.0", "0.5.0")

print("Base environment is ready.")
"""


AUTH_AND_DOWNLOAD = """
import pandas as pd

def get_colab_secret(name: str):
    try:
        from google.colab import userdata
        value = userdata.get(name)
        return value if value else None
    except Exception:
        return None

def setup_hf_token():
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_TOKET")
        or get_colab_secret("HF_TOKEN")
        or get_colab_secret("HF_TOKET")
    )
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token)
            print("HF token configured.")
        except Exception as exc:
            print("HF token present, but login skipped/failed:", exc)
    else:
        print("HF_TOKEN is not set. Public EVA-X checkpoints still work.")

def setup_kaggle_token():
    username = os.environ.get("KAGGLE_USERNAME") or get_colab_secret("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or get_colab_secret("KAGGLE_KEY")
    token_value = (
        os.environ.get("KAGGLE_API_TOKEN")
        or os.environ.get("KAGGLE_JSON")
        or get_colab_secret("KAGGLE_API_TOKEN")
        or get_colab_secret("KAGGLE_JSON")
    )
    if token_value:
        token_value = str(token_value).strip()
    if token_value and token_value.startswith("{") and not (username and key):
        creds = json.loads(token_value)
        username = creds.get("username")
        key = creds.get("key")
        if os.environ.get("KAGGLE_API_TOKEN", "").strip().startswith("{"):
            os.environ.pop("KAGGLE_API_TOKEN", None)
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
        kaggle_json.chmod(0o600)
        print("Kaggle legacy credentials configured.")
        return True
    if token_value:
        os.environ["KAGGLE_API_TOKEN"] = token_value
        access_token_file = kaggle_dir / "access_token"
        access_token_file.write_text(token_value, encoding="utf-8")
        access_token_file.chmod(0o600)
        print("Kaggle access token configured.")
        return True
    print("Kaggle credentials are not set.")
    return False

def kaggle_cli_executable() -> str:
    exe = shutil.which("kaggle")
    if exe is None:
        raise RuntimeError("Kaggle CLI executable was not found after installation.")
    return exe

def ensure_kaggle_cli():
    if not has_module("kaggle"):
        pip_install("kaggle")
    elif os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_USERNAME"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle"], check=True)

def unzip_downloads(target: Path, keep_zips: bool = False) -> None:
    for zip_path in sorted(target.glob("*.zip")):
        print("Unzipping:", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        if not keep_zips:
            zip_path.unlink()

def download_kaggle_dataset_file(dataset: str, filename: str | None, target: Path, env: dict | None = None) -> None:
    ensure_kaggle_cli()
    cmd = [kaggle_cli_executable(), "datasets", "download", "-d", dataset, "-p", str(target)]
    if filename:
        cmd.extend(["-f", filename])
    subprocess.run(cmd, check=True, env=env)
    unzip_downloads(target, keep_zips=os.environ.get("VINDR_KEEP_ZIPS", "0") == "1")

def download_kaggle_competition_file(competition: str, filename: str, target: Path, env: dict | None = None) -> None:
    ensure_kaggle_cli()
    subprocess.run(
        [kaggle_cli_executable(), "competitions", "download", "-c", competition, "-f", filename, "-p", str(target)],
        check=True,
        env=env,
    )
    unzip_downloads(target, keep_zips=os.environ.get("VINDR_KEEP_ZIPS", "0") == "1")

def maybe_download_incxr_from_kaggle(target_dir="/content/incxr_png"):
    if os.environ.get("DOWNLOAD_IN_CXR_FROM_KAGGLE", "1") != "1":
        print("DOWNLOAD_IN_CXR_FROM_KAGGLE=0; skipping IN-CXR download.")
        return None
    if not setup_kaggle_token():
        raise RuntimeError("Set Kaggle credentials before downloading IN-CXR.")
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    dataset = os.environ.get("IN_CXR_KAGGLE_DATASET", "arjav007/in-cxr-dataset-png")
    marker = target / ".incxr_kaggle_download_complete"
    if not marker.exists():
        print("Downloading IN-CXR Kaggle PNG mirror. Expected compressed size is about 380 MB.")
        download_kaggle_dataset_file(dataset, None, target, env=os.environ.copy())
        marker.write_text(dataset, encoding="utf-8")
    else:
        print("IN-CXR Kaggle mirror already exists:", target)
    print("IN-CXR PNG files found:", len(list(target.rglob("*.png"))))
    os.environ["IN_CXR_ROOT"] = str(target)
    print("IN_CXR_ROOT=", target)
    return str(target)

def sample_vindr_ids_from_train_csv(train_csv: Path, max_studies: int, normal_fraction: float = 0.50, seed: int = 42) -> pd.DataFrame:
    labels = pd.read_csv(train_csv)
    labels.columns = [c.strip() for c in labels.columns]
    if "image_id" not in labels.columns or "class_name" not in labels.columns:
        raise ValueError("VinDr/VinBigData train.csv must contain image_id and class_name columns.")
    tmp = labels[["image_id", "class_name"]].copy()
    tmp["image_id"] = tmp["image_id"].astype(str)
    tmp["is_abnormal_row"] = tmp["class_name"].fillna("").astype(str).str.lower().ne("no finding")
    image_level = tmp.groupby("image_id", as_index=False)["is_abnormal_row"].max()
    image_level["y_attention"] = image_level["is_abnormal_row"].astype(int)
    normal = image_level[image_level["y_attention"] == 0]
    abnormal = image_level[image_level["y_attention"] == 1]
    n_normal = min(len(normal), int(round(max_studies * normal_fraction)))
    n_abnormal = min(len(abnormal), max_studies - n_normal)
    if n_abnormal < max_studies - n_normal and len(normal) > n_normal:
        n_normal = min(len(normal), max_studies - n_abnormal)
    normal_s = normal.sample(n=n_normal, random_state=seed) if n_normal else normal.head(0)
    abnormal_s = abnormal.sample(n=n_abnormal, random_state=seed) if n_abnormal else abnormal.head(0)
    selected = pd.concat([normal_s, abnormal_s], ignore_index=True).sample(frac=1.0, random_state=seed)
    return selected[["image_id", "y_attention"]].reset_index(drop=True)

def maybe_download_vindr_png_subset_from_kaggle(target_dir="/content/vindr_cxr", max_studies: int | None = None):
    if os.environ.get("DOWNLOAD_VINDR_FROM_KAGGLE", "0") != "1":
        print("DOWNLOAD_VINDR_FROM_KAGGLE=0; skipping VinDr download.")
        return None
    if not setup_kaggle_token():
        raise RuntimeError("Set Kaggle credentials before downloading VinDr.")
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    competition = os.environ.get("VINDR_KAGGLE_COMPETITION", "vinbigdata-chest-xray-abnormalities-detection")
    image_dataset = os.environ.get("VINDR_KAGGLE_PNG_DATASET", "xhlulu/vinbigdata")
    metadata_dataset = os.environ.get("VINDR_KAGGLE_METADATA_DATASET", "sunhwan/vinbigdata-chest-xray-dicom-metadata")
    image_ext = os.environ.get("VINDR_KAGGLE_IMAGE_EXT", "png")
    max_studies = int(max_studies or os.environ.get("VINDR_DOWNLOAD_MAX_STUDIES", "5000"))
    normal_fraction = float(os.environ.get("VINDR_DOWNLOAD_NORMAL_FRACTION", "0.50"))
    seed = int(os.environ.get("VINDR_DOWNLOAD_SEED", "42"))

    train_csv = target / "train.csv"
    if not train_csv.exists():
        print("Downloading VinDr/VinBigData train.csv ...")
        download_kaggle_competition_file(competition, "train.csv", target, env=os.environ.copy())
    if not any((target / name).exists() for name in ["images.csv", "train_meta.csv", "metadata.csv", "dicom_metadata.csv"]):
        print("Downloading original DICOM metadata for bbox scaling ...")
        download_kaggle_dataset_file(metadata_dataset, None, target, env=os.environ.copy())

    selected = sample_vindr_ids_from_train_csv(train_csv, max_studies=max_studies, normal_fraction=normal_fraction, seed=seed)
    manifest_path = target / "vindr_subset_manifest.csv"
    selected.to_csv(manifest_path, index=False)
    print(f"Selected {len(selected)} VinDr studies:", selected["y_attention"].value_counts().to_dict())

    for idx, row in selected.iterrows():
        image_id = str(row["image_id"])
        image_path = target / f"{image_id}.{image_ext}"
        nested_image_path = target / "train" / f"{image_id}.{image_ext}"
        if image_path.exists() or nested_image_path.exists():
            continue
        if idx % 100 == 0:
            print(f"Downloading PNG {idx + 1}/{len(selected)} ...")
        download_kaggle_dataset_file(image_dataset, f"train/{image_id}.{image_ext}", target, env=os.environ.copy())

    os.environ["VINDR_ROOT"] = str(target)
    print("VINDR_ROOT=", target)
    return str(target)
"""


MODEL_SELECTION_CELLS = [
    md(
        """
        # Fluoro MVP Model Selection

        Clean research notebook for closing the model experiments before backend MVP.

        Main path:

        - IN-CXR normal/abnormal screening-like dataset.
        - EVA-X-S frozen features + tuned MLP head as current baseline.
        - EVA-X-S long LoRA and partial last-block unfreeze with early stopping.
        - EVA-X-B frozen features + EVA-X-B LoRA when T4 memory allows it.
        - CheXFound frozen inference and optional LoRA when repository checkpoints are supplied.
        - Quality-first model selection plus a separate deployment safety gate.

        Google CXR Foundation is intentionally not part of this notebook.
        """
    ),
    md("## 0. Central Run Config"),
    code(
        """
        import os
        from pathlib import Path

        try:
            import google.colab  # noqa: F401
            IN_COLAB_BOOTSTRAP = True
        except Exception:
            IN_COLAB_BOOTSTRAP = False

        # One runtime = one experiment. Override before this cell or edit here:
        # full_sweep, eva_small_frozen_mlp, eva_small_lora, eva_small_partial_unfreeze,
        # eva_base_frozen_mlp, eva_base_lora, eva_base_lora_strong,
        # eva_base_partial_unfreeze, chexfound_frozen, chexfound_lora.
        EXPERIMENT_PROFILE = os.environ.get("EXPERIMENT_PROFILE", "eva_small_frozen_mlp")

        RUN = {
            "run_name": "model_selection_incxr_t4",
            "max_studies": 12600,
            "download_incxr_from_kaggle": True,
            "eva_variants": ["small", "base"],
            "run_frozen_heads": True,
            "run_lora_sweep": True,
            "run_partial_unfreeze": True,
            "run_eva_base_lora": True,
            "run_chexfound_feasibility": True,
            "run_chexfound_inference": True,
            "run_chexfound_lora": True,
            "selection_objective": "quality_first",
            "selected_threshold_policy": "auto",
            "target_npv": 0.99,
            "calibration_methods": ["platt", "isotonic", "none"],
            "batch_size": 4,
            "eva_image_size": 224,
            "lora_checkpoint_interval": 15,
            "cache_preprocessed_to_disk": True,
            "preprocessed_cache_dir": "/content/fluoro_mvp_preprocessed_cache" if IN_COLAB_BOOTSTRAP else str(Path.cwd() / "fluoro_mvp_preprocessed_cache"),
            "project_base_dir": "/content/fluoro_mvp_runs" if IN_COLAB_BOOTSTRAP else str(Path.cwd() / "fluoro_mvp_outputs"),
            "lora_sweep": [
                {"name": "small_lora_last2_r4_e150", "variant": "small", "epochs": 150, "patience": 20, "rank": 4, "alpha": 8.0, "dropout": 0.05, "n_last_blocks": 2, "lr": 2e-4, "weight_decay": 1e-4, "batch_size": 4, "grad_accum_steps": 1},
                {"name": "small_lora_last3_r8_e150", "variant": "small", "epochs": 150, "patience": 20, "rank": 8, "alpha": 16.0, "dropout": 0.05, "n_last_blocks": 3, "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 4, "grad_accum_steps": 1},
                {"name": "base_lora_last1_r4_e150", "variant": "base", "epochs": 150, "patience": 15, "rank": 4, "alpha": 8.0, "dropout": 0.05, "n_last_blocks": 1, "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 1, "grad_accum_steps": 4},
                {"name": "base_lora_last2_r4_e150", "variant": "base", "epochs": 150, "patience": 15, "rank": 4, "alpha": 8.0, "dropout": 0.05, "n_last_blocks": 2, "lr": 7e-5, "weight_decay": 1e-4, "batch_size": 1, "grad_accum_steps": 4},
                {"name": "base_lora_last2_r8_e150", "variant": "base", "epochs": 150, "patience": 20, "rank": 8, "alpha": 16.0, "dropout": 0.05, "n_last_blocks": 2, "lr": 5e-5, "weight_decay": 1e-4, "batch_size": 1, "grad_accum_steps": 4},
            ],
            "partial_unfreeze_sweep": [
                {"name": "small_unfreeze_last1_e150", "variant": "small", "epochs": 150, "patience": 18, "n_last_blocks": 1, "lr": 1e-5, "head_lr": 5e-4, "weight_decay": 1e-4, "batch_size": 2, "grad_accum_steps": 2},
                {"name": "small_unfreeze_last2_e150", "variant": "small", "epochs": 150, "patience": 18, "n_last_blocks": 2, "lr": 5e-6, "head_lr": 5e-4, "weight_decay": 1e-4, "batch_size": 2, "grad_accum_steps": 2},
                {"name": "base_unfreeze_last1_e150", "variant": "base", "epochs": 150, "patience": 8, "n_last_blocks": 1, "lr": 5e-6, "head_lr": 4e-4, "weight_decay": 1e-4, "batch_size": 1, "grad_accum_steps": 4},
                {"name": "base_unfreeze_last2_e150", "variant": "base", "epochs": 150, "patience": 8, "n_last_blocks": 2, "lr": 2e-6, "head_lr": 3e-4, "weight_decay": 1e-4, "batch_size": 1, "grad_accum_steps": 4},
            ],
            "mlp_search_space": [
                {"name": "baseline_h128", "hidden": 128, "dropout": 0.20, "epochs": 80, "lr": 1e-3, "weight_decay": 1e-4, "seed_offset": 0},
                {"name": "strong_h256", "hidden": 256, "dropout": 0.20, "epochs": 120, "lr": 8e-4, "weight_decay": 5e-5, "seed_offset": 1},
                {"name": "regularized_h256", "hidden": 256, "dropout": 0.30, "epochs": 140, "lr": 8e-4, "weight_decay": 1e-4, "seed_offset": 2},
                {"name": "wide_h384", "hidden": 384, "dropout": 0.25, "epochs": 140, "lr": 5e-4, "weight_decay": 5e-5, "seed_offset": 3},
                {"name": "compact_low_wd", "hidden": 192, "dropout": 0.15, "epochs": 120, "lr": 1e-3, "weight_decay": 1e-5, "seed_offset": 4},
                {"name": "conservative_h256", "hidden": 256, "dropout": 0.35, "epochs": 160, "lr": 5e-4, "weight_decay": 2e-4, "seed_offset": 5},
                {"name": "wide_h512", "hidden": 512, "dropout": 0.30, "epochs": 160, "lr": 3e-4, "weight_decay": 1e-4, "seed_offset": 6},
                {"name": "strong_h256_seed2", "hidden": 256, "dropout": 0.20, "epochs": 140, "lr": 8e-4, "weight_decay": 5e-5, "seed_offset": 7},
            ],
        }

        EXPERIMENT_PRESETS = {
            "full_sweep": {
                "run_name": "incxr_full_sweep_t4",
            },
            "eva_small_frozen_mlp": {
                "run_name": "incxr_eva_small_frozen_mlp_t4",
                "eva_variants": ["small"],
                "run_frozen_heads": True,
                "run_lora_sweep": False,
                "run_partial_unfreeze": False,
                "run_eva_base_lora": False,
                "run_chexfound_feasibility": False,
                "run_chexfound_inference": False,
                "run_chexfound_lora": False,
            },
            "eva_small_lora": {
                "run_name": "incxr_eva_small_lora_t4",
                "eva_variants": ["small"],
                "run_frozen_heads": False,
                "run_lora_sweep": True,
                "run_partial_unfreeze": False,
                "run_eva_base_lora": False,
                "run_chexfound_feasibility": False,
                "run_chexfound_inference": False,
                "run_chexfound_lora": False,
                "enabled_lora_names": ["small_lora_last2_r4_e150", "small_lora_last3_r8_e150"],
            },
            "eva_small_partial_unfreeze": {
                "run_name": "incxr_eva_small_partial_unfreeze_t4",
                "eva_variants": ["small"],
                "run_frozen_heads": False,
                "run_lora_sweep": False,
                "run_partial_unfreeze": True,
                "run_eva_base_lora": False,
                "run_chexfound_feasibility": False,
                "run_chexfound_inference": False,
                "run_chexfound_lora": False,
            },
            "eva_base_frozen_mlp": {
                "run_name": "incxr_eva_base_frozen_mlp_t4",
                "eva_variants": ["base"],
                "run_frozen_heads": True,
                "run_lora_sweep": False,
                "run_partial_unfreeze": False,
                "run_eva_base_lora": False,
                "run_chexfound_feasibility": False,
                "run_chexfound_inference": False,
                "run_chexfound_lora": False,
            },
            "eva_base_lora": {
                "run_name": "incxr_eva_base_lora_t4",
                "eva_variants": ["base"],
                "run_frozen_heads": False,
                "run_lora_sweep": True,
                "run_partial_unfreeze": False,
                "run_eva_base_lora": True,
                "run_chexfound_feasibility": False,
                "run_chexfound_inference": False,
                "run_chexfound_lora": False,
                "enabled_lora_names": ["base_lora_last1_r4_e150", "base_lora_last2_r4_e150"],
            },
            "eva_base_lora_strong": {
                "run_name": "incxr_eva_base_lora_strong_t4",
                "eva_variants": ["base"],
                "run_frozen_heads": False,
                "run_lora_sweep": True,
                "run_partial_unfreeze": False,
                "run_eva_base_lora": True,
                "run_chexfound_feasibility": False,
                "run_chexfound_inference": False,
                "run_chexfound_lora": False,
                "enabled_lora_names": ["base_lora_last2_r8_e150"],
            },
            "eva_base_partial_unfreeze": {
                "run_name": "incxr_eva_base_partial_unfreeze_t4",
                "eva_variants": ["base"],
                "run_frozen_heads": False,
                "run_lora_sweep": False,
                "run_partial_unfreeze": True,
                "run_eva_base_lora": False,
                "run_chexfound_feasibility": False,
                "run_chexfound_inference": False,
                "run_chexfound_lora": False,
                "enabled_partial_unfreeze_names": ["base_unfreeze_last1_e150", "base_unfreeze_last2_e150"],
            },
            "chexfound_frozen": {
                "run_name": "incxr_chexfound_frozen_t4",
                "eva_variants": [],
                "run_frozen_heads": False,
                "run_lora_sweep": False,
                "run_partial_unfreeze": False,
                "run_eva_base_lora": False,
                "run_chexfound_feasibility": True,
                "run_chexfound_inference": True,
                "run_chexfound_lora": False,
            },
            "chexfound_lora": {
                "run_name": "incxr_chexfound_lora_t4",
                "eva_variants": [],
                "run_frozen_heads": False,
                "run_lora_sweep": False,
                "run_partial_unfreeze": False,
                "run_eva_base_lora": False,
                "run_chexfound_feasibility": True,
                "run_chexfound_inference": True,
                "run_chexfound_lora": True,
            },
        }
        if EXPERIMENT_PROFILE not in EXPERIMENT_PRESETS:
            raise ValueError(f"Unknown EXPERIMENT_PROFILE={EXPERIMENT_PROFILE!r}. Available: {sorted(EXPERIMENT_PRESETS)}")
        RUN.update(EXPERIMENT_PRESETS[EXPERIMENT_PROFILE])
        RUN["experiment_profile"] = EXPERIMENT_PROFILE
        if "enabled_lora_names" in RUN:
            allowed = set(RUN["enabled_lora_names"])
            RUN["lora_sweep"] = [cfg0 for cfg0 in RUN["lora_sweep"] if cfg0["name"] in allowed]
        if "enabled_partial_unfreeze_names" in RUN:
            allowed = set(RUN["enabled_partial_unfreeze_names"])
            RUN["partial_unfreeze_sweep"] = [cfg0 for cfg0 in RUN["partial_unfreeze_sweep"] if cfg0["name"] in allowed]
        if not RUN["run_lora_sweep"]:
            RUN["lora_sweep"] = []
        if not RUN["run_partial_unfreeze"]:
            RUN["partial_unfreeze_sweep"] = []

        PROJECT_DIR = str(Path(RUN["project_base_dir"]) / RUN["run_name"])
        os.environ["PROJECT_DIR"] = PROJECT_DIR
        os.environ["DOWNLOAD_IN_CXR_FROM_KAGGLE"] = "1" if RUN["download_incxr_from_kaggle"] else "0"
        os.environ["MAX_STUDIES"] = str(RUN["max_studies"])
        os.environ["PREPROCESSED_CACHE_DIR"] = RUN["preprocessed_cache_dir"]
        os.environ["LORA_CHECKPOINT_INTERVAL"] = str(RUN["lora_checkpoint_interval"])
        os.environ["FLUORO_NO_GOOGLE_CXR"] = "1"

        if IN_COLAB_BOOTSTRAP:
            os.environ["MOUNT_DRIVE"] = "0"
        else:
            os.environ.setdefault("MOUNT_DRIVE", "0")

        print("Run config:")
        for key, value in RUN.items():
            if key not in {"lora_sweep", "partial_unfreeze_sweep", "mlp_search_space"}:
                print(f"  {key}: {value}")
        print("PROJECT_DIR:", PROJECT_DIR)
        """
    ),
    md("## 1. Environment Setup"),
    code(COMMON_SETUP),
    md("## 2. Tokens and IN-CXR Download"),
    code(
        AUTH_AND_DOWNLOAD
        + """
setup_hf_token()
setup_kaggle_token()
if RUN["download_incxr_from_kaggle"]:
    maybe_download_incxr_from_kaggle(os.environ.get("IN_CXR_DOWNLOAD_DIR", "/content/incxr_png"))
"""
    ),
    md("## 3. Core Code"),
    code(CORE_SOURCE),
    md("## 4. Runtime Config and Paths"),
    code(
        """
        import gc
        import warnings
        from dataclasses import asdict

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB and os.environ.get("MOUNT_DRIVE", "0") == "1":
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)

        cfg = NotebookConfig(
            project_dir=PROJECT_DIR,
            data_root=os.environ.get("IN_CXR_ROOT") or None,
            labels_csv=os.environ.get("IN_CXR_LABELS_CSV") or None,
            max_studies=int(RUN["max_studies"]),
            random_state=42,
            target_npv=float(RUN["target_npv"]),
            eva_image_size=int(RUN["eva_image_size"]),
            batch_size=int(RUN["batch_size"]),
            run_real_google_cxr=False,
            run_real_eva_x=True,
            run_exp2_lora=bool(RUN["run_lora_sweep"]),
            run_primary_track=True,
            run_vindr_track=False,
            run_vindr_exp2=False,
            run_exp3_chexfound=bool(RUN["run_chexfound_feasibility"]),
            pca_components=256,
            cache_preprocessed_to_disk=bool(RUN["cache_preprocessed_to_disk"]),
            preprocessed_cache_dir=RUN["preprocessed_cache_dir"],
        )
        if not cfg.data_root and not cfg.labels_csv:
            raise RuntimeError("IN_CXR_ROOT or IN_CXR_LABELS_CSV is missing. Run the download cell or set the path manually.")
        if IN_COLAB and str(cfg.preprocessed_cache_dir).startswith("/content/drive"):
            raise RuntimeError("Use local Colab disk for PREPROCESSED_CACHE_DIR, not Google Drive.")
        ensure_dirs(cfg)
        set_seed(cfg.random_state)
        if torch is not None:
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        print(asdict(cfg))
        """
    ),
    md("## 5. Dataset Loading and Preprocessing"),
    code(
        """
        df = discover_dataset(cfg.data_root, cfg.labels_csv, max_studies=cfg.max_studies, cfg=cfg)
        print("Hashing image bytes to prevent exact duplicate leakage across splits ...")
        df = attach_content_hashes(df)
        source_dataset_audit = validate_binary_dataset_contract(df, group_col="content_sha256")
        duplicate_images_removed = int(df.duplicated("content_sha256").sum())
        if duplicate_images_removed:
            print(f"Removing {duplicate_images_removed} exact duplicate image copies before splitting.")
            df = df.drop_duplicates("content_sha256", keep="first").reset_index(drop=True)
        dataset_audit = validate_binary_dataset_contract(df, group_col="content_sha256")
        df = make_splits(df, seed=cfg.random_state, group_col="content_sha256")
        split_audit = validate_split_integrity(df, group_col="content_sha256")
        dataset_audit.update({
            "source_n_before_exact_deduplication": int(source_dataset_audit["n_images"]),
            "exact_duplicate_images_removed": duplicate_images_removed,
            "source_profile": "IN-CXR Kaggle PNG mirror: preprocessed 224x224 grayscale images",
            "label_scope": "official IN-CXR normal versus abnormal screening label from the Indian National TB Prevalence Survey",
            "patient_grouping_limitation": "Kaggle mirror exposes no verified patient identifier; exact duplicate content is grouped, but patient-level leakage cannot be proven absent.",
        })
        export_json(
            {"dataset": dataset_audit, "splits": split_audit},
            Path(cfg.reports_dir) / "dataset_contract_audit.json",
        )
        index_path = save_table(df, Path(cfg.artifacts_dir) / "data_index")
        print(f"Dataset index saved to: {index_path}")
        display(df["y_attention"].value_counts().rename("count").to_frame())
        display(df["split"].value_counts().rename("count").to_frame())

        results, meta = preprocess_dataframe(df, cfg)
        y = meta["y_attention"].values.astype(int)
        preproc_path = save_table(meta, Path(cfg.artifacts_dir) / "preprocessing_report")
        print(f"Preprocessing report saved to: {preproc_path}")
        display(meta[["quality_score", "roi_status", "critical_qa"]].describe(include="all"))
        if RUN["run_chexfound_inference"] and float(pd.to_numeric(meta["rows"], errors="coerce").median()) < 512:
            print(
                "IMPORTANT: CheXFound will receive upscaled 224px Kaggle-mirror images. "
                "Treat this branch as feasibility/reference only; a definitive CheXFound comparison requires original-resolution DICOM."
            )

        n_show = min(6, len(results))
        fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
        if n_show == 1:
            axes = np.asarray([axes])
        for i, r in enumerate(results[:n_show]):
            axes[i, 0].imshow(get_result_raw_preview(r), cmap="gray")
            axes[i, 0].set_title(f"raw norm | y={r.y_attention}")
            axes[i, 1].imshow(get_result_image_eva(r), cmap="gray")
            axes[i, 1].set_title(f"EVA input | q={r.quality_score:.2f}")
            roi_img = get_result_image_roi(r)
            if roi_img is not None:
                axes[i, 2].imshow(roi_img, cmap="gray")
            else:
                axes[i, 2].text(0.5, 0.5, "ROI missing", ha="center", va="center")
            axes[i, 2].set_title(f"ROI: {r.roi_status}")
            for ax in axes[i]:
                ax.axis("off")
        plt.tight_layout()
        preview_path = Path(cfg.artifacts_dir) / "previews" / "preprocessing_audit.png"
        fig.savefig(preview_path, dpi=160, bbox_inches="tight")
        plt.show()
        print("Saved:", preview_path)
        """
    ),
    md("## 6. Shared Model Selection Helpers"),
    code(
        """
        SELECTED_THRESHOLD_POLICY = str(RUN["selected_threshold_policy"])
        SELECTION_OBJECTIVE = str(RUN["selection_objective"]).lower()

        THRESHOLD_POLICIES = [
            {"name": "target_npv_max_coverage", "require_zero_fn": False, "coverage_cap": 1.00, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.35, "t_positive": 0.80, "t_uncertainty": 0.65},
            {"name": "zero_fn_max_coverage", "require_zero_fn": True, "coverage_cap": 1.00, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.35, "t_positive": 0.80, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_10pct", "require_zero_fn": True, "coverage_cap": 0.10, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.35, "t_positive": 0.80, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_08pct", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.35, "t_positive": 0.80, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_08pct_ood90", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.90, "t_quality": 0.35, "t_positive": 0.80, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_08pct_ood110", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 1.10, "t_quality": 0.35, "t_positive": 0.80, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_08pct_quality25", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.25, "t_positive": 0.80, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_08pct_tpos70", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.35, "t_positive": 0.70, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_08pct_tpos60", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.35, "t_positive": 0.60, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_08pct_tpos90", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.35, "t_positive": 0.90, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_08pct_quality45", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.45, "t_positive": 0.80, "t_uncertainty": 0.65},
            {"name": "zero_fn_cap_05pct", "require_zero_fn": True, "coverage_cap": 0.05, "min_selected": 5, "min_npv_ci95_low": None, "t_ood": 0.95, "t_quality": 0.35, "t_positive": 0.80, "t_uncertainty": 0.65},
            {"name": "ci_guard_cap_10pct", "require_zero_fn": True, "coverage_cap": 0.10, "min_selected": 20, "min_npv_ci95_low": 0.95, "t_ood": 0.95, "t_quality": 0.35, "t_positive": 0.80, "t_uncertainty": 0.65},
        ]

        def fit_probability_calibrator(method, raw_calib, y_calib):
            return ProbabilityCalibrator(method=str(method)).fit(raw_calib, y_calib)

        def choose_calibrator(raw_calib, y_calib, raw_val, y_val, methods=None):
            methods = list(methods or RUN.get("calibration_methods", ["platt"]))
            rows = []
            fitted = {}
            for method in methods:
                try:
                    calibrator = fit_probability_calibrator(method, raw_calib, y_calib)
                    p_val = calibrator.transform(raw_val)
                    row = metrics_summary(y_val, p_val)
                    row.update({"calibration_method": str(method), "calibration_error": ""})
                    rows.append(row)
                    fitted[str(method)] = (calibrator, p_val)
                except Exception as exc:
                    rows.append({"calibration_method": str(method), "calibration_error": repr(exc)})
            table = pd.DataFrame(rows)
            valid = table[table["calibration_error"].fillna("").eq("")].copy()
            if valid.empty:
                raise RuntimeError(f"No calibration method succeeded: {table.to_dict(orient='records')}")
            valid = valid.sort_values(
                ["brier", "ece", "auroc", "auprc"],
                ascending=[True, True, False, False],
                na_position="last",
            )
            selected_method = str(valid.iloc[0]["calibration_method"])
            calibrator, p_val = fitted[selected_method]
            return calibrator, p_val, selected_method, table

        def sort_candidates(rows: pd.DataFrame, objective: str = "quality_first") -> pd.DataFrame:
            out = rows.copy()
            objective = objective.lower()
            if objective in {"coverage_first", "safe_coverage_first", "auto_clear_first"}:
                return out.sort_values(
                    ["auto_negative_coverage", "auto_negative_NPV", "auroc", "auprc", "ece"],
                    ascending=[False, False, False, False, True],
                    na_position="last",
                )
            return out.sort_values(
                ["auroc", "auprc", "brier", "ece", "auto_negative_coverage"],
                ascending=[False, False, True, True, False],
                na_position="last",
            )

        def router_metrics_at_threshold(y_true, p_val, val_meta, policy, t_negative, ood_score_values=None):
            routes = route_decisions(
                p_val,
                val_meta,
                t_negative=float(t_negative),
                t_positive=float(policy.get("t_positive", 0.80)),
                t_quality=float(policy.get("t_quality", 0.35)),
                ood_score=ood_score_values,
                t_ood=float(policy.get("t_ood", 0.95)),
                t_uncertainty=float(policy.get("t_uncertainty", 0.65)),
            )
            y_arr = np.asarray(y_true).astype(int)
            auto = routes["route"].values == "no_attention_required"
            selected_count = int(np.sum(auto))
            if selected_count == 0:
                return None, routes
            fn = int(np.sum((y_arr == 1) & auto))
            tn = int(np.sum((y_arr == 0) & auto))
            npv = float(tn / max(tn + fn, 1))
            npv_ci95_low = wilson_lower_bound(tn, tn + fn, z=1.96)
            metrics = route_metrics(y_true, routes)
            metrics.update({
                "selected_T_negative": float(t_negative),
                "threshold_policy": policy["name"],
                "threshold_selected_count": selected_count,
                "threshold_validation_TN_count": tn,
                "threshold_validation_NPV": npv,
                "threshold_validation_FN_count": fn,
                "NPV_ci95_low": float(npv_ci95_low),
                "selected_t_ood": float(policy.get("t_ood", 0.95)),
                "selected_t_positive": float(policy.get("t_positive", 0.80)),
                "selected_t_quality": float(policy.get("t_quality", 0.35)),
                "selected_t_uncertainty": float(policy.get("t_uncertainty", 0.65)),
            })
            return metrics, routes

        def route_metrics_for_policy(y_true, p_val, val_meta, thr_report, policy, ood_score_values=None):
            if thr_report is not None and not thr_report.empty:
                thresholds = np.asarray(sorted(thr_report["T_negative"].dropna().unique()), dtype=float)
            else:
                thresholds = np.unique(np.quantile(np.asarray(p_val, dtype=float), np.linspace(0, 1, 501)))
            rows = []
            routes_by_row = {}
            for t_negative in thresholds:
                metrics, routes = router_metrics_at_threshold(
                    y_true,
                    p_val,
                    val_meta,
                    policy,
                    t_negative=float(t_negative),
                    ood_score_values=ood_score_values,
                )
                if metrics is None:
                    continue
                selected_count_ok = metrics["threshold_selected_count"] >= int(policy.get("min_selected", 1))
                coverage_ok = metrics["auto_negative_coverage"] <= float(policy.get("coverage_cap", 1.0)) + 1e-12
                target_ok = metrics["threshold_validation_NPV"] >= cfg.target_npv
                zero_fn_ok = (not policy.get("require_zero_fn", False)) or metrics["threshold_validation_FN_count"] == 0
                min_ci = policy.get("min_npv_ci95_low")
                ci_ok = True if min_ci is None else metrics["NPV_ci95_low"] >= float(min_ci)
                row = dict(metrics)
                row["meets_policy_constraints"] = bool(selected_count_ok and coverage_ok and target_ok and zero_fn_ok and ci_ok)
                row["threshold_policy_candidate_selected"] = False
                row_idx = len(rows)
                rows.append(row)
                routes_by_row[row_idx] = routes
            sweep = pd.DataFrame(rows)
            if sweep.empty:
                return None, None, sweep
            valid = sweep[sweep["meets_policy_constraints"]].copy()
            if valid.empty:
                return None, None, sweep
            valid = valid.sort_values(
                ["auto_negative_coverage", "NPV_ci95_low", "selected_T_negative"],
                ascending=[False, False, False],
                na_position="last",
            )
            selected_idx = int(valid.index[0])
            sweep.loc[selected_idx, "threshold_policy_candidate_selected"] = True
            return sweep.loc[selected_idx].to_dict(), routes_by_row[selected_idx], sweep

        candidate_rows = []
        model_registry = {}
        feature_matrices = {}
        split_parts = {}
        eva_models_for_reuse = {}

        def register_feature_candidate(name, variant, kind, model, X, parts, base_metrics, p_val, calibrator=None, threshold_report_df=None):
            val_meta = meta.iloc[parts["validation"][2]].reset_index(drop=True)
            threshold_report_df = threshold_report_df if threshold_report_df is not None else threshold_report(parts["validation"][1], p_val, target_npv=cfg.target_npv)
            selected_rows = []
            selected_routes = {}
            full_sweep_rows = []
            try:
                policy_ood_model = fit_ood_model(parts["train"][0])
                validation_ood_scores = ood_score(policy_ood_model, parts["validation"][0])
            except Exception as exc:
                raise RuntimeError(f"Validation OOD scoring failed for {name}; router tuning cannot continue safely.") from exc
            for policy in THRESHOLD_POLICIES:
                policy_metrics, routes, sweep = route_metrics_for_policy(parts["validation"][1], p_val, val_meta, threshold_report_df, policy, ood_score_values=validation_ood_scores)
                if sweep is not None and not sweep.empty:
                    sweep = sweep.copy()
                    sweep.insert(0, "model", name)
                    sweep.insert(1, "variant", variant)
                    sweep.insert(2, "kind", kind)
                    full_sweep_rows.append(sweep)
                row = dict(base_metrics)
                row.update({
                    "model": name,
                    "variant": variant,
                    "kind": kind,
                    "threshold_policy": policy["name"],
                    "threshold_policy_selected": policy_metrics is not None,
                    "deployment_adapter_supported": str(variant).lower() in EVA_X_VARIANTS,
                })
                if policy_metrics is not None:
                    row.update(policy_metrics)
                    selected_routes[policy["name"]] = routes
                else:
                    row.update({"auto_negative_coverage": 0.0, "auto_negative_NPV": np.nan, "unsafe_FN_auto_negative": np.nan})
                selected_rows.append(row)
            policy_table = pd.DataFrame(selected_rows)
            policy_table.to_csv(Path(cfg.reports_dir) / f"{name}_router_policy_summary.csv", index=False)
            if full_sweep_rows:
                pd.concat(full_sweep_rows, ignore_index=True).to_csv(Path(cfg.reports_dir) / f"{name}_threshold_sweep.csv", index=False)
            else:
                policy_table.to_csv(Path(cfg.reports_dir) / f"{name}_threshold_sweep.csv", index=False)
            if SELECTED_THRESHOLD_POLICY.lower() in {"auto", "best", "best_valid"}:
                pick_rows = policy_table[policy_table["threshold_policy_selected"]].copy()
            else:
                pick_rows = policy_table[(policy_table["threshold_policy"] == SELECTED_THRESHOLD_POLICY) & (policy_table["threshold_policy_selected"])].copy()
                if pick_rows.empty:
                    raise RuntimeError(
                        f"Requested threshold policy {SELECTED_THRESHOLD_POLICY!r} is not valid for {name}. "
                        "Use selected_threshold_policy='auto' for automatic selection."
                    )
            if pick_rows.empty:
                print(f"No valid threshold policy for {name}; candidate kept as research-only.")
                candidate_rows.append(policy_table.iloc[0].to_dict())
                return
            picked = pick_rows.sort_values(
                [
                    "auto_negative_coverage",
                    "auto_negative_NPV",
                    "NPV_ci95_low",
                    "N/A_rate",
                    "workload_FP_requires_attention",
                ],
                ascending=[False, False, False, True, True],
                na_position="last",
            ).iloc[0].to_dict()
            candidate_rows.append(picked)
            model_registry[name] = {
                "name": name,
                "variant": variant,
                "kind": kind,
                "model": model,
                "calibrator": calibrator,
                "X": X,
                "parts": parts,
                "p_val": p_val,
                "selected_T_negative": float(picked["selected_T_negative"]),
                "selected_t_ood": float(picked.get("selected_t_ood", 0.95)),
                "selected_t_positive": float(picked.get("selected_t_positive", 0.80)),
                "selected_t_quality": float(picked.get("selected_t_quality", 0.35)),
                "selected_t_uncertainty": float(picked.get("selected_t_uncertainty", 0.65)),
                "threshold_policy": str(picked["threshold_policy"]),
                "calibration_method": str(picked.get("calibration_method", getattr(calibrator, "method", "n/a"))),
                "threshold_report": threshold_report_df,
                "routes_validation": selected_routes.get(str(picked["threshold_policy"])),
            }
            print("Registered candidate:", name, "policy=", picked["threshold_policy"], "AUROC=", picked.get("auroc"))

        def score_candidate_split(candidate: dict, split_name: str, ood_model=None):
            parts = candidate["parts"]
            Xs, ys, idx = parts[split_name]
            if candidate["kind"] == "torch_mlp":
                raw = predict_torch_mlp(candidate["model"], Xs, device=cfg.device)
                p = candidate["calibrator"].transform(raw)
            elif candidate["kind"] in {"eva_e2e", "lora_e2e", "partial_unfreeze_e2e"}:
                raw = predict_eva_end_to_end(candidate["model"], results, idx, cfg, batch_size=int(candidate.get("adapt_batch_size", candidate.get("lora_batch_size", 4))))
                p = candidate["calibrator"].transform(raw)
            elif candidate["kind"] in {"chexfound_e2e"}:
                raw = predict_chexfound_end_to_end(candidate["model"], results, idx, cfg, batch_size=int(candidate.get("adapt_batch_size", 1)))
                p = candidate["calibrator"].transform(raw)
            else:
                p = predict_proba_any(candidate["model"], Xs)[:, 1]
            ood_s = ood_score(ood_model, Xs) if ood_model is not None else None
            return ys, p, ood_s, idx
        """
    ),
    md("## 7. EVA-X Frozen Feature Experiments"),
    code(
        """
        def dataset_fingerprint(results):
            return stable_hash("|".join(f"{r.study_id}:{r.source_path}:{r.y_attention}" for r in results), n=16)

        def extract_or_load_eva_variant(variant: str):
            variant = str(variant).lower()
            feat_path = Path(cfg.artifacts_dir) / "embeddings" / f"eva_x_{variant}_features.npy"
            fp_path = Path(cfg.artifacts_dir) / "embeddings" / f"eva_x_{variant}_fingerprint.json"
            fp = dataset_fingerprint(results)
            if feat_path.exists() and fp_path.exists():
                info = json.loads(fp_path.read_text(encoding="utf-8"))
                X_cached = np.load(feat_path)
                if info.get("fingerprint") == fp and X_cached.shape[0] == len(results):
                    print(f"Using cached EVA-X-{variant} features:", X_cached.shape)
                    return None, X_cached
                print(f"Cached EVA-X-{variant} features do not match current dataset; recomputing.")
            model = load_real_eva_x(cfg.project_dir, variant=variant, device=cfg.device)
            X = extract_eva_features_real(
                model,
                results,
                image_size=cfg.eva_image_size,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
            np.save(feat_path, X)
            fp_path.write_text(json.dumps({"fingerprint": fp, "n": len(results), "variant": variant}, indent=2), encoding="utf-8")
            print(f"Saved EVA-X-{variant} features:", X.shape)
            return model, X

        def train_frozen_heads_for_variant(variant: str, X: np.ndarray):
            parts = split_arrays_by_meta(X, y, meta, require_all_splits=True)
            feature_matrices[variant] = X
            split_parts[variant] = parts
            val_meta = meta.iloc[parts["validation"][2]].reset_index(drop=True)
            if not RUN.get("run_frozen_heads", True):
                print(f"Prepared EVA-X-{variant} frozen features/splits; frozen head training is disabled for profile {RUN['experiment_profile']}.")
                return

            print(
                f"Skipping EVA-X-{variant} logistic reference; production selection uses tuned MLP heads, calibration, and router sweep."
            )

            # MLP sweep.
            tuning_records = []
            best_by_key = {}
            try:
                tuning_ood_model = fit_ood_model(parts["train"][0])
                tuning_ood_scores = ood_score(tuning_ood_model, parts["validation"][0])
            except Exception as exc:
                raise RuntimeError(
                    f"MLP tuning OOD scoring failed for EVA-X-{variant}; router search cannot continue safely."
                ) from exc
            for hp in RUN["mlp_search_space"]:
                hp = dict(hp)
                hp["seed"] = cfg.random_state + int(hp.pop("seed_offset", 0))
                X_head_train, y_head_train, X_head_es, y_head_es = train_internal_validation_split(
                    parts["train"][0],
                    parts["train"][1],
                    seed=int(hp["seed"]),
                )
                print(f"Training EVA-X-{variant} MLP:", hp)
                model = train_torch_mlp(
                    X_head_train,
                    y_head_train,
                    X_head_es,
                    y_head_es,
                    hidden=int(hp["hidden"]),
                    dropout=float(hp["dropout"]),
                    epochs=int(hp["epochs"]),
                    lr=float(hp["lr"]),
                    weight_decay=float(hp["weight_decay"]),
                    seed=int(hp["seed"]),
                    device=cfg.device,
                )
                raw_calib = predict_torch_mlp(model, parts["calibration"][0], device=cfg.device)
                raw_val = predict_torch_mlp(model, parts["validation"][0], device=cfg.device)
                for calibration_method in RUN.get("calibration_methods", ["platt"]):
                    calibrator = fit_probability_calibrator(calibration_method, raw_calib, parts["calibration"][1])
                    p_val = calibrator.transform(raw_val)
                    base_metrics = metrics_summary(parts["validation"][1], p_val)
                    base_metrics["calibration_method"] = str(calibration_method)
                    thr = threshold_report(parts["validation"][1], p_val, target_npv=cfg.target_npv)
                    local_rows = []
                    head_key = f"{hp['name']}__{calibration_method}"
                    for policy in THRESHOLD_POLICIES:
                        policy_metrics, _, _ = route_metrics_for_policy(
                            parts["validation"][1],
                            p_val,
                            val_meta,
                            thr,
                            policy,
                            ood_score_values=tuning_ood_scores,
                        )
                        row = {
                            **base_metrics,
                            **hp,
                            "head_name": hp["name"],
                            "head_key": head_key,
                            "calibration_method": str(calibration_method),
                            "variant": variant,
                            "threshold_policy": policy["name"],
                            "threshold_policy_selected": policy_metrics is not None,
                        }
                        if policy_metrics is not None:
                            row.update(policy_metrics)
                        local_rows.append(row)
                    tuning_records.extend(local_rows)
                    best_by_key[head_key] = (model, calibrator, p_val, thr, base_metrics)

            tuning = pd.DataFrame(tuning_records)
            tuning = sort_candidates(tuning[tuning["threshold_policy_selected"]].copy(), SELECTION_OBJECTIVE) if tuning["threshold_policy_selected"].any() else tuning
            tuning.to_csv(Path(cfg.reports_dir) / f"eva_{variant}_mlp_tuning_results.csv", index=False)
            display(tuning.head(20))
            if SELECTED_THRESHOLD_POLICY.lower() in {"auto", "best", "best_valid"}:
                selected = tuning[tuning["threshold_policy_selected"]].copy()
            else:
                selected = tuning[(tuning["threshold_policy"] == SELECTED_THRESHOLD_POLICY) & (tuning["threshold_policy_selected"])].copy()
                if selected.empty:
                    raise RuntimeError(
                        f"Requested threshold policy {SELECTED_THRESHOLD_POLICY!r} is invalid for every EVA-X-{variant} MLP head."
                    )
            if selected.empty:
                raise RuntimeError(f"No valid tuned MLP candidate for EVA-X-{variant}.")
            selected = sort_candidates(selected, SELECTION_OBJECTIVE)
            selected_head = str(selected.iloc[0]["head_name"])
            selected_key = str(selected.iloc[0]["head_key"])
            selected_calibration = str(selected.iloc[0]["calibration_method"])
            model, calibrator, p_val, thr, base_metrics = best_by_key[selected_key]
            candidate_name = f"eva_{variant}_frozen_mlp_{selected_head}_{selected_calibration}"
            register_feature_candidate(
                name=candidate_name,
                variant=variant,
                kind="torch_mlp",
                model=model,
                X=X,
                parts=parts,
                base_metrics=base_metrics,
                p_val=p_val,
                calibrator=calibrator,
                threshold_report_df=thr,
            )
            torch.save({"state_dict": model.state_dict(), "scaler": model.scaler, "variant": variant, "head_name": selected_head, "calibration_method": selected_calibration}, Path(cfg.checkpoints_dir) / f"{candidate_name}.pt")
            save_pickle(calibrator, Path(cfg.artifacts_dir) / "calibration" / f"{candidate_name}_calibrator.pkl")

        for variant in RUN["eva_variants"]:
            model, X = extract_or_load_eva_variant(variant)
            if model is not None:
                del model
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            train_frozen_heads_for_variant(variant, X)
        """
    ),
    md("## 8. EVA-X Adaptation: Long LoRA and Partial Unfreeze"),
    code(
        """
        def configure_eva_trainable_params(eva_model, adapt_cfg, mode: str):
            for p in eva_model.parameters():
                p.requires_grad = False
            if mode == "lora":
                replaced = inject_lora_last_blocks(
                    eva_model,
                    n_last_blocks=int(adapt_cfg["n_last_blocks"]),
                    r=int(adapt_cfg["rank"]),
                    alpha=float(adapt_cfg["alpha"]),
                    dropout=float(adapt_cfg.get("dropout", 0.0)),
                )
                if replaced <= 0:
                    raise RuntimeError("No EVA attention/MLP Linear modules matched the requested LoRA targets.")
                print(f"LoRA Linear modules replaced: {replaced}")
                return
            if mode == "partial_unfreeze":
                blocks = list(getattr(eva_model, "blocks", []))
                n_last = int(adapt_cfg["n_last_blocks"])
                if not blocks:
                    raise RuntimeError("EVA model has no .blocks attribute for partial unfreeze.")
                for block in blocks[-n_last:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for attr in ["norm", "fc_norm"]:
                    module = getattr(eva_model, attr, None)
                    if module is not None:
                        for p in module.parameters():
                            p.requires_grad = True
                print(f"Partially unfroze EVA last {n_last} blocks plus final norms.")
                return
            raise ValueError(f"Unknown EVA adaptation mode: {mode}")

        def predict_eva_end_to_end(model, results, indices, cfg, batch_size=4):
            preds = []
            model.eval()
            with torch.no_grad():
                for start in range(0, len(indices), batch_size):
                    batch_idx = indices[start:start + batch_size]
                    xb = torch.stack([image_to_eva_tensor(get_result_image_eva(results[int(i)]), cfg.eva_image_size) for i in batch_idx]).to(cfg.device)
                    with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
                        logits = model(xb)
                    preds.append(torch.sigmoid(logits).detach().float().cpu().numpy())
            return np.concatenate(preds, axis=0).astype(np.float32)

        # Backward-compatible alias used by older report cells.
        predict_lora_end_to_end = predict_eva_end_to_end

        def train_eva_adaptation(eva_model, results, labels, cfg, parts, adapt_cfg, mode: str):
            eva_model.to(cfg.device)
            configure_eva_trainable_params(eva_model, adapt_cfg, mode=mode)
            eva_model.to(cfg.device)
            sample = image_to_eva_tensor(get_result_image_eva(results[0]), cfg.eva_image_size).unsqueeze(0).to(cfg.device)
            with torch.no_grad():
                if hasattr(eva_model, "forward_features"):
                    z = eva_model.forward_features(sample)
                    if z.ndim == 3:
                        z = z[:, 1:, :].mean(dim=1) if z.shape[1] > 1 else z.mean(dim=1)
                else:
                    z = eva_model(sample)
            model = EVAEndToEndClassifier(eva_model, int(z.shape[-1])).to(cfg.device)
            print("EVA adaptation params:", count_parameters(model))
            y_tensor = torch.tensor(labels.astype(np.float32), device=cfg.device)
            encoder_params = []
            head_params = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("head."):
                    head_params.append(p)
                else:
                    encoder_params.append(p)
            param_groups = []
            if encoder_params:
                param_groups.append({"params": encoder_params, "lr": float(adapt_cfg["lr"])})
            if head_params:
                param_groups.append({"params": head_params, "lr": float(adapt_cfg.get("head_lr", adapt_cfg["lr"]))})
            opt = torch.optim.AdamW(param_groups, weight_decay=float(adapt_cfg["weight_decay"]))
            scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))
            batch_size = int(adapt_cfg["batch_size"])
            grad_accum_steps = max(1, int(adapt_cfg.get("grad_accum_steps", 1)))
            patience = int(adapt_cfg.get("patience", 20))
            min_delta = float(adapt_cfg.get("min_delta", 1e-4))
            candidate_dir = Path(cfg.checkpoints_dir) / str(adapt_cfg["name"])
            candidate_dir.mkdir(parents=True, exist_ok=True)
            best_path = candidate_dir / "best.pt"
            last_path = candidate_dir / "last.pt"
            history_path = candidate_dir / "history.csv"
            checkpoint_interval = max(1, int(os.environ.get("LORA_CHECKPOINT_INTERVAL", "15")))
            if best_path.exists() and os.environ.get("RESUME_EXISTING_CHECKPOINTS", "1") == "1":
                print("Loading existing best checkpoint:", best_path)
                state = torch.load(best_path, map_location=cfg.device, weights_only=False)
                model.load_state_dict(state["state_dict"], strict=False)
                return model, pd.DataFrame(state.get("history", []))

            best_score = -np.inf
            best_epoch = 0
            history = []
            train_indices_all = np.asarray(parts["train"][2], dtype=int)
            train_indices, early_stop_indices = index_internal_validation_split(
                train_indices_all,
                labels,
                seed=cfg.random_state,
            )
            if early_stop_indices is None:
                raise RuntimeError(
                    "Training split is too small for a separate internal early-stopping subset. "
                    "Validation reuse is disabled in the production research path."
                )
            early_stop_y = labels[early_stop_indices]
            for epoch in range(int(adapt_cfg["epochs"])):
                model.train()
                order = np.random.permutation(train_indices)
                losses = []
                opt.zero_grad(set_to_none=True)
                for start in range(0, len(order), batch_size):
                    idx = order[start:start + batch_size]
                    xb = torch.stack([image_to_eva_tensor(get_result_image_eva(results[int(i)]), cfg.eva_image_size) for i in idx]).to(cfg.device)
                    yb = y_tensor[idx]
                    with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
                        loss = F.binary_cross_entropy_with_logits(model(xb), yb) / grad_accum_steps
                    scaler.scale(loss).backward()
                    if ((start // batch_size) + 1) % grad_accum_steps == 0 or start + batch_size >= len(order):
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                    losses.append(float(loss.detach().cpu()) * grad_accum_steps)
                epoch_loss = float(np.mean(losses)) if losses else float("nan")
                p_early_raw = predict_eva_end_to_end(model, results, early_stop_indices, cfg, batch_size=batch_size)
                early_auc = safe_auc(early_stop_y, p_early_raw)
                early_auprc = safe_auprc(early_stop_y, p_early_raw)
                try:
                    val_loss = float(F.binary_cross_entropy(
                        torch.tensor(p_early_raw, dtype=torch.float32),
                        torch.tensor(early_stop_y.astype(np.float32), dtype=torch.float32),
                    ))
                except Exception:
                    val_loss = float("nan")
                monitor = early_auc if np.isfinite(early_auc) else -val_loss
                row = {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "early_stop_auroc_raw": early_auc,
                    "early_stop_auprc_raw": early_auprc,
                    "early_stop_loss_raw": val_loss,
                    "monitor": monitor,
                    "mode": mode,
                    **adapt_cfg,
                }
                history.append(row)
                payload = {"epoch": epoch + 1, "state_dict": model.state_dict(), "history": history, "adapt_cfg": adapt_cfg, "mode": mode}
                last_path = candidate_dir / "last.pt"
                torch.save(payload, last_path)
                epoch_path = None
                if (epoch + 1) % checkpoint_interval == 0:
                    epoch_path = candidate_dir / f"epoch_{epoch + 1}.pt"
                    torch.save(payload, epoch_path)
                improved = bool(np.isfinite(monitor) and monitor > best_score + min_delta)
                if improved:
                    best_score = float(monitor)
                    best_epoch = epoch + 1
                    torch.save(payload, best_path)
                pd.DataFrame(history).to_csv(history_path, index=False)
                print(
                    f"{adapt_cfg['name']} epoch {epoch+1}: loss={epoch_loss:.4f}; "
                    f"early_auc={early_auc:.4f}; early_auprc={early_auprc:.4f}; best_epoch={best_epoch}; "
                    f"last_checkpoint={last_path}; periodic_checkpoint={epoch_path}"
                )
                if epoch + 1 - best_epoch >= patience:
                    print(f"Early stopping {adapt_cfg['name']} at epoch {epoch+1}; best_epoch={best_epoch}.")
                    break
            if best_path.exists():
                state = torch.load(best_path, map_location=cfg.device, weights_only=False)
                model.load_state_dict(state["state_dict"], strict=False)
            return model, pd.DataFrame(history)

        def run_eva_adaptation_candidate(adapt_cfg, mode: str):
            adapt_cfg = dict(adapt_cfg)
            variant = str(adapt_cfg["variant"]).lower()
            if variant == "base" and mode == "lora" and not RUN.get("run_eva_base_lora", True):
                print("Skipping EVA-X-B LoRA because run_eva_base_lora=False.")
                return
            if variant not in split_parts:
                print(f"Skipping {adapt_cfg['name']}: frozen features for EVA-X-{variant} are not available.")
                return
            try:
                print(f"Running EVA adaptation mode={mode} config:", adapt_cfg)
                base_model = load_real_eva_x(cfg.project_dir, variant=variant, device=cfg.device)
                parts = split_parts[variant]
                model, history = train_eva_adaptation(base_model, results, y, cfg, parts, adapt_cfg, mode=mode)
                final_path = Path(cfg.checkpoints_dir) / f"{adapt_cfg['name']}_final.pt"
                torch.save({"state_dict": model.state_dict(), "adapt_cfg": adapt_cfg, "mode": mode}, final_path)
                prefix = "lora" if mode == "lora" else "partial_unfreeze"
                candidate_name = f"eva_{variant}_{prefix}_{adapt_cfg['name']}"
                raw_calib = predict_eva_end_to_end(model, results, parts["calibration"][2], cfg, batch_size=int(adapt_cfg["batch_size"]))
                raw_val = predict_eva_end_to_end(model, results, parts["validation"][2], cfg, batch_size=int(adapt_cfg["batch_size"]))
                calibrator, p_val, calibration_method, calibration_search_table = choose_calibrator(
                    raw_calib,
                    parts["calibration"][1],
                    raw_val,
                    parts["validation"][1],
                )
                calibration_search_table.to_csv(Path(cfg.reports_dir) / f"{candidate_name}_calibration_tuning.csv", index=False)
                base_metrics = metrics_summary(parts["validation"][1], p_val)
                base_metrics["calibration_method"] = calibration_method
                register_feature_candidate(
                    name=candidate_name,
                    variant=variant,
                    kind="lora_e2e" if mode == "lora" else "partial_unfreeze_e2e",
                    model=model,
                    X=feature_matrices[variant],
                    parts=parts,
                    base_metrics=base_metrics,
                    p_val=p_val,
                    calibrator=calibrator,
                    threshold_report_df=threshold_report(parts["validation"][1], p_val, target_npv=cfg.target_npv),
                )
                if candidate_name in model_registry:
                    model_registry[candidate_name]["adapt_batch_size"] = int(adapt_cfg["batch_size"])
                    model_registry[candidate_name]["adapt_cfg"] = adapt_cfg
                    model_registry[candidate_name]["adapt_history"] = history.to_dict(orient="records") if history is not None and not history.empty else []
                    model_registry[candidate_name]["calibration_method"] = calibration_method
                    save_pickle(calibrator, Path(cfg.artifacts_dir) / "calibration" / f"{candidate_name}_calibrator.pkl")
                if history is not None:
                    history.to_csv(Path(cfg.reports_dir) / f"{candidate_name}_training_history.csv", index=False)
                del base_model
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                    print(f"Skipping {adapt_cfg['name']} after resource/runtime error:", exc)
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return
                raise

        if RUN["run_lora_sweep"]:
            for lora_cfg in RUN["lora_sweep"]:
                run_eva_adaptation_candidate(lora_cfg, mode="lora")
        else:
            print("LoRA sweep disabled by RUN['run_lora_sweep'].")

        if RUN.get("run_partial_unfreeze", False):
            for unfreeze_cfg in RUN["partial_unfreeze_sweep"]:
                run_eva_adaptation_candidate(unfreeze_cfg, mode="partial_unfreeze")
        else:
            print("Partial unfreeze disabled by RUN['run_partial_unfreeze'].")
        """
    ),
    md("## 9. CheXFound Feasibility Check"),
    code(
        """
        def run_chexfound_feasibility():
            report = {
                "enabled": bool(RUN["run_chexfound_feasibility"]),
                "repo_url": "https://github.com/RPIDIAL/CheXFound",
                "paper": "https://arxiv.org/abs/2502.05142",
                "license": "MIT in official GitHub repository",
                "status": "not_run",
            }
            if not RUN["run_chexfound_feasibility"]:
                report["status"] = "disabled"
                return report
            repo_dir = Path(cfg.project_dir) / "external" / "CheXFound"
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            if not repo_dir.exists():
                subprocess.run(["git", "clone", "--depth", "1", "https://github.com/RPIDIAL/CheXFound.git", str(repo_dir)], check=True)
            ckpt = os.environ.get("CHEXFOUND_CKPT", "")
            report.update({
                "status": "repo_available",
                "repo_dir": str(repo_dir),
                "checkpoint_path": ckpt,
                "checkpoint_exists": bool(ckpt and Path(ckpt).exists()),
                "decision": "Use as optional frozen/GLoRI reference only after a real checkpoint/config is supplied. LoRA is not mandatory before frozen/GLoRI proves stable on T4.",
            })
            if not report["checkpoint_exists"]:
                report["next_step"] = "Set CHEXFOUND_CKPT and, if needed, CHEXFOUND_CONFIG before attempting real inference."
            else:
                report["next_step"] = "Wire repository-specific frozen encoder/GLoRI adapter and run the same head/router protocol."
            return report

        chexfound_feasibility = run_chexfound_feasibility()
        chexfound_path = export_json(chexfound_feasibility, Path(cfg.reports_dir) / "chexfound_feasibility_report.json")
        print(json.dumps(chexfound_feasibility, indent=2, ensure_ascii=False))
        print("Saved:", chexfound_path)
        """
    ),
    md("## 9.1 Optional CheXFound Frozen Inference and LoRA"),
    code(
        """
        os.environ.setdefault("XFORMERS_DISABLED", "1")

        def ensure_chexfound_repo():
            repo_dir = Path(cfg.project_dir) / "external" / "CheXFound"
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            if not repo_dir.exists():
                subprocess.run(["git", "clone", "--depth", "1", "https://github.com/RPIDIAL/CheXFound.git", str(repo_dir)], check=True)
            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))
            return repo_dir

        def ensure_chexfound_deps():
            dep_report = {}
            for dep in ["omegaconf", "fvcore", "iopath"]:
                if not has_module(dep):
                    try:
                        pip_install(dep)
                    except Exception as exc:
                        dep_report[dep] = f"install_failed: {exc}"
                        continue
                dep_report[dep] = "ok" if has_module(dep) else "missing"
            return dep_report

        def chexfound_image_tensor(img, image_size=512):
            img = img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            lo = arr.reshape(3, -1).min(axis=1).reshape(3, 1, 1)
            hi = arr.reshape(3, -1).max(axis=1).reshape(3, 1, 1)
            arr = (arr - lo) / np.maximum(hi - lo, 1e-6)
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            return torch.tensor((arr - mean) / std, dtype=torch.float32)

        def chexfound_features_from_outputs(outputs):
            cls_tokens = []
            patch_tokens = None
            for item in outputs:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    patch_tokens, cls_token = item[0], item[1]
                elif isinstance(item, dict):
                    cls_token = item.get("x_norm_clstoken")
                    patch_tokens = item.get("x_norm_patchtokens")
                else:
                    patch_tokens = item
                    cls_token = item[:, 0] if item.ndim == 3 else item
                    if item.ndim == 3 and item.shape[1] > 1:
                        patch_tokens = item[:, 1:, :]
                cls_tokens.append(cls_token)
            if patch_tokens is not None and patch_tokens.ndim == 3:
                cls_tokens.append(patch_tokens.mean(dim=1))
            return torch.cat(cls_tokens, dim=1)

        def load_chexfound_hf_safetensors():
            # Practical fallback for the public HF snapshot. The official repo example
            # expects config.yaml + teacher_checkpoint.pth, while HF publishes a
            # ViT-L/16 safetensors snapshot with "model."-prefixed keys.
            os.environ.setdefault("XFORMERS_DISABLED", "1")
            ensure_chexfound_repo()
            try:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
                from chexfound.models.vision_transformer import vit_large
            except Exception as exc:
                raise RuntimeError(f"Could not import CheXFound HF fallback dependencies: {exc}") from exc

            repo_id = os.environ.get("CHEXFOUND_HF_REPO", "DIAL-RPI/CheXFound")
            filename = os.environ.get("CHEXFOUND_HF_FILENAME", "model.safetensors")
            weights_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(Path(cfg.artifacts_dir) / "hf_cache"),
            )
            state = load_file(weights_path, device="cpu")
            mapped = {}
            for key, value in state.items():
                if key.startswith("model."):
                    key = key[len("model."):]
                key = key.replace(".ls1.weight", ".ls1.gamma").replace(".ls2.weight", ".ls2.gamma")
                mapped[key] = value
            del state

            model = vit_large(
                img_size=512,
                patch_size=16,
                num_register_tokens=4,
                init_values=1e-5,
                ffn_layer="swiglufused",
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                block_chunks=0,
                interpolate_antialias=False,
                interpolate_offset=0.1,
            )
            missing, unexpected = model.load_state_dict(mapped, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"CheXFound HF safetensors load mismatch: missing={list(missing)[:10]}, "
                    f"unexpected={list(unexpected)[:10]}"
                )
            model.eval().to(cfg.device)
            return {
                "model": model,
                "autocast_dtype": torch.float16 if cfg.device == "cuda" else torch.float32,
                "source": "hf_safetensors",
                "weights_path": weights_path,
                "hf_repo": repo_id,
            }

        def load_chexfound_model_if_possible():
            report = dict(chexfound_feasibility)
            report["inference_enabled"] = bool(RUN.get("run_chexfound_inference", False))
            if not RUN.get("run_chexfound_inference", False):
                report["inference_status"] = "disabled"
                return None, report
            config_file = os.environ.get("CHEXFOUND_CONFIG", "")
            ckpt = os.environ.get("CHEXFOUND_CKPT", "")
            report.update({
                "config_file": config_file,
                "config_exists": bool(config_file and Path(config_file).exists()),
                "checkpoint_path": ckpt,
                "checkpoint_exists": bool(ckpt and Path(ckpt).exists()),
                "xformers_disabled": os.environ.get("XFORMERS_DISABLED"),
            })
            if not report["config_exists"] or not report["checkpoint_exists"]:
                try:
                    chex_obj = load_chexfound_hf_safetensors()
                    report.update({
                        "inference_status": "loaded_hf_safetensors",
                        "source": chex_obj.get("source"),
                        "hf_repo": chex_obj.get("hf_repo"),
                        "weights_path": chex_obj.get("weights_path"),
                        "autocast_dtype": str(chex_obj.get("autocast_dtype")),
                    })
                    return chex_obj, report
                except Exception as exc:
                    report["inference_status"] = "missing_config_or_checkpoint_and_hf_fallback_failed"
                    report["next_step"] = "Set CHEXFOUND_CONFIG and CHEXFOUND_CKPT to official CheXFound files, or fix HF fallback access."
                    report["hf_fallback_error"] = repr(exc)
                    return None, report
            if cfg.device != "cuda":
                try:
                    chex_obj = load_chexfound_hf_safetensors()
                    report.update({
                        "inference_status": "loaded_hf_safetensors_non_cuda",
                        "source": chex_obj.get("source"),
                        "hf_repo": chex_obj.get("hf_repo"),
                        "weights_path": chex_obj.get("weights_path"),
                        "autocast_dtype": str(chex_obj.get("autocast_dtype")),
                    })
                    return chex_obj, report
                except Exception as exc:
                    report["inference_status"] = "skipped_non_cuda_runtime_and_hf_fallback_failed"
                    report["next_step"] = "Official CheXFound setup calls CUDA directly; run in Colab T4 or fix HF fallback."
                    report["hf_fallback_error"] = repr(exc)
                    return None, report
            try:
                ensure_chexfound_repo()
                report["dependency_report"] = ensure_chexfound_deps()
                import argparse
                from chexfound.eval.setup import setup_and_build_model
                args = argparse.Namespace(
                    config_file=config_file,
                    pretrained_weights=ckpt,
                    output_dir=str(Path(cfg.artifacts_dir) / "chexfound"),
                    opts=[],
                )
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                model, autocast_dtype = setup_and_build_model(args)
                model.eval()
                report.update({"inference_status": "loaded", "autocast_dtype": str(autocast_dtype)})
                return {"model": model, "autocast_dtype": autocast_dtype}, report
            except Exception as exc:
                report.update({"inference_status": "load_failed", "error": repr(exc)})
                return None, report

        def extract_chexfound_features_real(chex_obj, results, batch_size=1, image_size=512, n_last_blocks=4):
            model = chex_obj["model"]
            autocast_dtype = chex_obj.get("autocast_dtype", torch.float16)
            feats = []
            model.eval()
            with torch.no_grad():
                for start in range(0, len(results), batch_size):
                    batch_results = results[start:start + batch_size]
                    xb = torch.stack([
                        chexfound_image_tensor(get_result_image_full(r), image_size=image_size)
                        for r in batch_results
                    ]).to(cfg.device)
                    with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda"), dtype=autocast_dtype if cfg.device == "cuda" else torch.float32):
                        z = chexfound_features_from_outputs(model.get_intermediate_layers(xb, n=n_last_blocks, return_class_token=True))
                    feats.append(z.detach().float().cpu().numpy())
            return np.concatenate(feats, axis=0).astype(np.float32)

        class CheXFoundEndToEndClassifier(nn.Module):
            def __init__(self, backbone, feature_dim: int, image_size: int = 512):
                super().__init__()
                self.backbone = backbone
                self.image_size = image_size
                self.head = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, 256),
                    nn.GELU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 1),
                )

            def encode(self, x):
                return chexfound_features_from_outputs(self.backbone.get_intermediate_layers(x, n=4, return_class_token=True))

            def forward(self, x):
                return self.head(self.encode(x)).squeeze(-1)

        def predict_chexfound_end_to_end(model, results, indices, cfg, batch_size=1):
            preds = []
            model.eval()
            with torch.no_grad():
                for start in range(0, len(indices), batch_size):
                    batch_idx = indices[start:start + batch_size]
                    xb = torch.stack([
                        chexfound_image_tensor(get_result_image_full(results[int(i)]), image_size=int(model.image_size))
                        for i in batch_idx
                    ]).to(cfg.device)
                    with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
                        logits = model(xb)
                    preds.append(torch.sigmoid(logits).detach().float().cpu().numpy())
            return np.concatenate(preds, axis=0).astype(np.float32)

        def try_train_chexfound_lora(chex_obj, parts):
            if not RUN.get("run_chexfound_lora", False):
                print("CheXFound LoRA disabled.")
                return
            lora_cfg = {
                "name": "chexfound_lora_last1_r4_e80",
                "epochs": 80,
                "patience": 10,
                "rank": 4,
                "alpha": 8.0,
                "dropout": 0.05,
                "n_last_blocks": 1,
                "lr": 5e-5,
                "weight_decay": 1e-4,
                "batch_size": 1,
                "grad_accum_steps": 4,
                "image_size": 512,
            }
            try:
                backbone = chex_obj["model"]
                replaced = inject_lora_last_blocks(
                    backbone,
                    n_last_blocks=int(lora_cfg["n_last_blocks"]),
                    r=int(lora_cfg["rank"]),
                    alpha=float(lora_cfg["alpha"]),
                    dropout=float(lora_cfg["dropout"]),
                    target_names=("qkv", "proj", "fc1", "fc2", "w12", "w3"),
                )
                if replaced <= 0:
                    raise RuntimeError("No CheXFound Linear modules matched the requested LoRA targets.")
                print(f"CheXFound LoRA Linear modules replaced: {replaced}")
                sample = chexfound_image_tensor(get_result_image_full(results[0]), image_size=int(lora_cfg["image_size"])).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    z0 = chexfound_features_from_outputs(backbone.get_intermediate_layers(sample, n=4, return_class_token=True))
                model = CheXFoundEndToEndClassifier(backbone, int(z0.shape[-1]), image_size=int(lora_cfg["image_size"])).to(cfg.device)
                print("CheXFound LoRA params:", count_parameters(model))
                y_tensor = torch.tensor(y.astype(np.float32), device=cfg.device)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lora_cfg["lr"]), weight_decay=float(lora_cfg["weight_decay"]))
                scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))
                train_indices_all = np.asarray(parts["train"][2], dtype=int)
                train_indices, early_stop_indices = index_internal_validation_split(
                    train_indices_all,
                    y,
                    seed=cfg.random_state,
                )
                if early_stop_indices is None:
                    raise RuntimeError(
                        "CheXFound training split is too small for an independent early-stopping subset. "
                        "Validation reuse is disabled."
                    )
                early_stop_y = y[early_stop_indices]
                candidate_dir = Path(cfg.checkpoints_dir) / lora_cfg["name"]
                candidate_dir.mkdir(parents=True, exist_ok=True)
                best_path = candidate_dir / "best.pt"
                last_path = candidate_dir / "last.pt"
                checkpoint_interval = max(1, int(os.environ.get("LORA_CHECKPOINT_INTERVAL", "15")))
                best_score, best_epoch, history = -np.inf, 0, []
                for epoch in range(int(lora_cfg["epochs"])):
                    model.train()
                    order = np.random.permutation(train_indices)
                    losses = []
                    opt.zero_grad(set_to_none=True)
                    for start in range(0, len(order), int(lora_cfg["batch_size"])):
                        idx = order[start:start + int(lora_cfg["batch_size"])]
                        xb = torch.stack([chexfound_image_tensor(get_result_image_full(results[int(i)]), image_size=int(lora_cfg["image_size"])) for i in idx]).to(cfg.device)
                        yb = y_tensor[idx]
                        with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
                            loss = F.binary_cross_entropy_with_logits(model(xb), yb) / int(lora_cfg["grad_accum_steps"])
                        scaler.scale(loss).backward()
                        if ((start // int(lora_cfg["batch_size"])) + 1) % int(lora_cfg["grad_accum_steps"]) == 0 or start + int(lora_cfg["batch_size"]) >= len(order):
                            scaler.step(opt)
                            scaler.update()
                            opt.zero_grad(set_to_none=True)
                        losses.append(float(loss.detach().cpu()) * int(lora_cfg["grad_accum_steps"]))
                    p_early_raw = predict_chexfound_end_to_end(model, results, early_stop_indices, cfg, batch_size=int(lora_cfg["batch_size"]))
                    early_auc, early_auprc = safe_auc(early_stop_y, p_early_raw), safe_auprc(early_stop_y, p_early_raw)
                    monitor = early_auc if np.isfinite(early_auc) else early_auprc
                    row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "early_stop_auroc_raw": early_auc, "early_stop_auprc_raw": early_auprc, **lora_cfg}
                    history.append(row)
                    payload = {"epoch": epoch + 1, "state_dict": model.state_dict(), "history": history, "lora_cfg": lora_cfg}
                    torch.save(payload, last_path)
                    if (epoch + 1) % checkpoint_interval == 0:
                        torch.save(payload, candidate_dir / f"epoch_{epoch + 1}.pt")
                    if np.isfinite(monitor) and monitor > best_score + 1e-4:
                        best_score, best_epoch = float(monitor), epoch + 1
                        torch.save(payload, best_path)
                    pd.DataFrame(history).to_csv(Path(cfg.reports_dir) / "chexfound_lora_training_history.csv", index=False)
                    print(f"CheXFound LoRA epoch {epoch+1}: loss={row['train_loss']:.4f}; early_auc={early_auc:.4f}; best_epoch={best_epoch}")
                    if epoch + 1 - best_epoch >= int(lora_cfg["patience"]):
                        print(f"Early stopping CheXFound LoRA at epoch {epoch+1}; best_epoch={best_epoch}.")
                        break
                if best_path.exists():
                    state = torch.load(best_path, map_location=cfg.device, weights_only=False)
                    model.load_state_dict(state["state_dict"], strict=False)
                raw_calib = predict_chexfound_end_to_end(model, results, parts["calibration"][2], cfg, batch_size=int(lora_cfg["batch_size"]))
                raw_val = predict_chexfound_end_to_end(model, results, parts["validation"][2], cfg, batch_size=int(lora_cfg["batch_size"]))
                calibrator, p_val, calibration_method, calibration_search_table = choose_calibrator(
                    raw_calib,
                    parts["calibration"][1],
                    raw_val,
                    parts["validation"][1],
                )
                candidate_name = "chexfound_lora_last1_r4"
                calibration_search_table.to_csv(Path(cfg.reports_dir) / f"{candidate_name}_calibration_tuning.csv", index=False)
                base_metrics = metrics_summary(parts["validation"][1], p_val)
                base_metrics["calibration_method"] = calibration_method
                register_feature_candidate(
                    name=candidate_name,
                    variant="chexfound",
                    kind="chexfound_e2e",
                    model=model,
                    X=feature_matrices["chexfound"],
                    parts=parts,
                    base_metrics=base_metrics,
                    p_val=p_val,
                    calibrator=calibrator,
                    threshold_report_df=threshold_report(parts["validation"][1], p_val, target_npv=cfg.target_npv),
                )
                if candidate_name in model_registry:
                    model_registry[candidate_name]["adapt_batch_size"] = int(lora_cfg["batch_size"])
                    model_registry[candidate_name]["adapt_cfg"] = lora_cfg
                    model_registry[candidate_name]["calibration_method"] = calibration_method
                save_pickle(calibrator, Path(cfg.artifacts_dir) / "calibration" / f"{candidate_name}_calibrator.pkl")
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                    print("CheXFound LoRA skipped after resource/runtime error:", exc)
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return
                raise
            except Exception as exc:
                print("CheXFound LoRA skipped after setup/training error:", repr(exc))

        chex_obj, chexfound_inference_report = load_chexfound_model_if_possible()
        if chex_obj is not None:
            try:
                chex_feat_path = Path(cfg.artifacts_dir) / "embeddings" / "chexfound_frozen_features.npy"
                if chex_feat_path.exists():
                    X_chex = np.load(chex_feat_path)
                    print("Loaded cached CheXFound features:", X_chex.shape)
                else:
                    X_chex = extract_chexfound_features_real(chex_obj, results, batch_size=1, image_size=512, n_last_blocks=4)
                    np.save(chex_feat_path, X_chex)
                    print("Saved CheXFound features:", X_chex.shape)
                feature_matrices["chexfound"] = X_chex
                parts_chex = split_arrays_by_meta(X_chex, y, meta, require_all_splits=True)
                split_parts["chexfound"] = parts_chex
                X_chex_head_train, y_chex_head_train, X_chex_head_es, y_chex_head_es = train_internal_validation_split(
                    parts_chex["train"][0],
                    parts_chex["train"][1],
                    seed=cfg.random_state,
                )
                chex_head = train_torch_mlp(
                    X_chex_head_train,
                    y_chex_head_train,
                    X_chex_head_es,
                    y_chex_head_es,
                    hidden=256,
                    dropout=0.30,
                    epochs=120,
                    lr=8e-4,
                    weight_decay=1e-4,
                    seed=cfg.random_state,
                    device=cfg.device,
                )
                raw_calib = predict_torch_mlp(chex_head, parts_chex["calibration"][0], device=cfg.device)
                raw_val = predict_torch_mlp(chex_head, parts_chex["validation"][0], device=cfg.device)
                chex_calibrator, p_val, chex_calibration_method, calibration_search_table = choose_calibrator(
                    raw_calib,
                    parts_chex["calibration"][1],
                    raw_val,
                    parts_chex["validation"][1],
                )
                calibration_search_table.to_csv(Path(cfg.reports_dir) / "chexfound_frozen_mlp_calibration_tuning.csv", index=False)
                chex_metrics = metrics_summary(parts_chex["validation"][1], p_val)
                chex_metrics["calibration_method"] = chex_calibration_method
                register_feature_candidate(
                    name="chexfound_frozen_mlp",
                    variant="chexfound",
                    kind="torch_mlp",
                    model=chex_head,
                    X=X_chex,
                    parts=parts_chex,
                    base_metrics=chex_metrics,
                    p_val=p_val,
                    calibrator=chex_calibrator,
                    threshold_report_df=threshold_report(parts_chex["validation"][1], p_val, target_npv=cfg.target_npv),
                )
                torch.save({"state_dict": chex_head.state_dict(), "scaler": chex_head.scaler, "calibration_method": chex_calibration_method}, Path(cfg.checkpoints_dir) / "chexfound_frozen_mlp.pt")
                save_pickle(chex_calibrator, Path(cfg.artifacts_dir) / "calibration" / "chexfound_frozen_mlp_calibrator.pkl")
                try_train_chexfound_lora(chex_obj, parts_chex)
            except Exception as exc:
                chexfound_inference_report.update({"inference_status": "inference_failed", "error": repr(exc)})

        chexfound_feasibility.update(chexfound_inference_report)
        chexfound_path = export_json(chexfound_feasibility, Path(cfg.reports_dir) / "chexfound_feasibility_report.json")
        print(json.dumps(chexfound_feasibility, indent=2, ensure_ascii=False))
        print("Saved:", chexfound_path)
        """
    ),
    md("## 10. Validation Model Comparison"),
    code(
        """
        if not candidate_rows:
            raise RuntimeError("No model candidates were registered.")
        required_candidate_token = {
            "eva_small_lora": "eva_small_lora_",
            "eva_small_partial_unfreeze": "eva_small_partial_unfreeze_",
            "eva_base_lora": "eva_base_lora_",
            "eva_base_lora_strong": "eva_base_lora_",
            "eva_base_partial_unfreeze": "eva_base_partial_unfreeze_",
            "chexfound_lora": "chexfound_lora_",
        }.get(RUN["experiment_profile"])
        if required_candidate_token and not any(required_candidate_token in name for name in model_registry):
            raise RuntimeError(
                f"Experiment profile {RUN['experiment_profile']!r} did not produce its required candidate "
                f"matching {required_candidate_token!r}. A frozen or partial fallback will not be exported."
            )
        model_comparison = pd.DataFrame(candidate_rows)
        model_comparison = sort_candidates(model_comparison, SELECTION_OBJECTIVE)
        model_comparison.to_csv(Path(cfg.reports_dir) / "model_comparison.csv", index=False)
        display(model_comparison)

        registered_model_comparison = model_comparison[
            model_comparison["model"].astype(str).isin(model_registry.keys())
        ].copy()
        if registered_model_comparison.empty:
            raise RuntimeError("No registered candidate has a complete scoring path for final-test evaluation.")

        research_best_name = str(registered_model_comparison.iloc[0]["model"])

        def deployment_safe_mask(df: pd.DataFrame) -> pd.Series:
            return (
                (df["auto_negative_NPV"].fillna(0.0) >= cfg.target_npv)
                & (df["unsafe_FN_auto_negative"].fillna(1e9) == 0)
                & (df["threshold_policy_selected"].fillna(False).astype(bool))
                & (df["deployment_adapter_supported"].fillna(False).astype(bool))
            )

        # The research-best router can be a high-coverage target-NPV policy with nonzero FN.
        # For deployment we scan every saved router policy for every registered model and
        # choose the best strict zero-FN policy when one exists.
        deployment_policy_rows = []
        for model_name in registered_model_comparison["model"].astype(str).tolist():
            policy_path = Path(cfg.reports_dir) / f"{model_name}_router_policy_summary.csv"
            if policy_path.exists():
                policy_table = pd.read_csv(policy_path)
                if "model" not in policy_table.columns:
                    policy_table.insert(0, "model", model_name)
                deployment_policy_rows.append(policy_table)
        if deployment_policy_rows:
            deployment_policy_table = pd.concat(deployment_policy_rows, ignore_index=True)
            deployment_policy_table = deployment_policy_table[
                deployment_policy_table["model"].astype(str).isin(model_registry.keys())
            ].copy()
        else:
            deployment_policy_table = registered_model_comparison.copy()

        if "deployment_adapter_supported" not in deployment_policy_table.columns:
            deployment_policy_table["deployment_adapter_supported"] = deployment_policy_table["model"].astype(str).map(
                lambda name: str(model_registry.get(name, {}).get("variant", "")).lower() in EVA_X_VARIANTS
            )

        safe_rows = deployment_policy_table[deployment_safe_mask(deployment_policy_table)].copy()
        deployment_candidates = sort_candidates(safe_rows, SELECTION_OBJECTIVE) if not safe_rows.empty else safe_rows

        if deployment_candidates.empty:
            deployment_ready = False
            deployment_best_name = research_best_name
            print(
                "No strict deployment-safe router was found on validation "
                "(requires NPV >= target and 0 unsafe FN on auto-negative route). "
                "Continuing with a research-only bundle so the run can be inspected."
            )
            gate_debug_cols = [
                c for c in [
                    "model",
                    "threshold_policy",
                    "auto_negative_coverage",
                    "auto_negative_NPV",
                    "unsafe_FN_auto_negative",
                    "threshold_validation_FN_count",
                    "NPV_ci95_low",
                    "deployment_adapter_supported",
                ] if c in deployment_policy_table.columns
            ]
            deployment_policy_table[gate_debug_cols].to_csv(
                Path(cfg.reports_dir) / "deployment_gate_debug.csv",
                index=False,
            )
            display(deployment_policy_table[gate_debug_cols].sort_values(
                [c for c in ["unsafe_FN_auto_negative", "auto_negative_NPV", "auto_negative_coverage"] if c in gate_debug_cols],
                ascending=[True, False, False][:len([c for c in ["unsafe_FN_auto_negative", "auto_negative_NPV", "auto_negative_coverage"] if c in gate_debug_cols])],
                na_position="last",
            ).head(20))
        else:
            deployment_ready = True
            deployment_best_name = str(deployment_candidates.iloc[0]["model"])
            selected_deployment_policy = deployment_candidates.iloc[0].to_dict()
            target_candidate = model_registry[deployment_best_name]
            for key in [
                "selected_T_negative",
                "selected_t_ood",
                "selected_t_positive",
                "selected_t_quality",
                "selected_t_uncertainty",
                "threshold_policy",
                "calibration_method",
            ]:
                if key in selected_deployment_policy and pd.notna(selected_deployment_policy[key]):
                    if key.startswith("selected_t") or key == "selected_T_negative":
                        target_candidate[key] = float(selected_deployment_policy[key])
                    else:
                        target_candidate[key] = str(selected_deployment_policy[key])
        research_best = model_registry[research_best_name]
        best = model_registry[deployment_best_name]
        print("Research best:", research_best_name)
        print("Deployment candidate:" if deployment_ready else "Research-only final candidate:", deployment_best_name)
        deployment_candidates.to_csv(Path(cfg.reports_dir) / "deployment_candidates.csv", index=False)
        """
    ),
    md("## 11. Fixed-threshold Final Test"),
    code(
        """
        def evaluate_candidate_final(candidate_name, candidate):
            candidate_ood = fit_ood_model(candidate["parts"]["train"][0])
            y_val, p_val, ood_val, val_idx = score_candidate_split(candidate, "validation", ood_model=candidate_ood)
            val_meta = meta.iloc[val_idx].reset_index(drop=True)
            val_routes = route_decisions(
                p_val,
                val_meta,
                t_negative=float(candidate["selected_T_negative"]),
                t_positive=float(candidate.get("selected_t_positive", 0.80)),
                t_quality=float(candidate.get("selected_t_quality", 0.35)),
                ood_score=ood_val,
                t_ood=float(candidate.get("selected_t_ood", 0.95)),
                t_uncertainty=float(candidate.get("selected_t_uncertainty", 0.65)),
            )
            y_test, p_test, ood_test, test_idx = score_candidate_split(candidate, "final_test", ood_model=candidate_ood)
            test_meta = meta.iloc[test_idx].reset_index(drop=True)
            final_metrics, final_routes = fixed_threshold_evaluation(
                candidate_name + " final_test",
                y_test,
                p_test,
                test_meta,
                t_negative=float(candidate["selected_T_negative"]),
                t_positive=float(candidate.get("selected_t_positive", 0.80)),
                t_quality=float(candidate.get("selected_t_quality", 0.35)),
                t_uncertainty=float(candidate.get("selected_t_uncertainty", 0.65)),
                ood_score_values=ood_test,
                t_ood=float(candidate.get("selected_t_ood", 0.95)),
            )
            final_metrics.update({
                "candidate_model": candidate_name,
                "threshold_policy": candidate.get("threshold_policy"),
                "selected_T_negative": float(candidate["selected_T_negative"]),
                "selected_t_ood": float(candidate.get("selected_t_ood", 0.95)),
                "selected_t_positive": float(candidate.get("selected_t_positive", 0.80)),
                "selected_t_quality": float(candidate.get("selected_t_quality", 0.35)),
                "selected_t_uncertainty": float(candidate.get("selected_t_uncertainty", 0.65)),
                "variant": candidate.get("variant"),
                "kind": candidate.get("kind"),
            })
            return final_metrics, final_routes, val_routes, val_idx, test_idx

        final_metrics, final_test_routes, best_routes, best_val_idx, best_test_idx = evaluate_candidate_final(deployment_best_name, best)
        final_safety_pass = (
            float(final_metrics.get("auto_negative_NPV", 0.0)) >= float(cfg.target_npv)
            and float(final_metrics.get("unsafe_FN_auto_negative", 1e9)) == 0.0
        )
        if deployment_ready and not final_safety_pass:
            deployment_ready = False
            final_safety_downgrade_reason = (
                "Validation-selected router did not pass the fixed final-test safety check: "
                f"auto_negative_NPV={final_metrics.get('auto_negative_NPV')}, "
                f"unsafe_FN_auto_negative={final_metrics.get('unsafe_FN_auto_negative')}."
            )
            print(final_safety_downgrade_reason)
        else:
            final_safety_downgrade_reason = ""
        pd.DataFrame([final_metrics]).to_csv(Path(cfg.reports_dir) / "best_final_test_metrics.csv", index=False)
        final_test_routes.to_csv(Path(cfg.reports_dir) / "best_routes_final_test.csv", index=False)
        best_routes.to_csv(Path(cfg.reports_dir) / "best_routes_validation.csv", index=False)
        display(pd.DataFrame([final_metrics]))
        display(final_test_routes.head())

        all_final_rows = []
        for name, candidate in model_registry.items():
            try:
                row, _, _, _, _ = evaluate_candidate_final(name, candidate)
                all_final_rows.append(row)
            except Exception as exc:
                print(f"Final-test report failed for {name}: {exc}")
        all_candidates_final_test = sort_candidates(pd.DataFrame(all_final_rows), SELECTION_OBJECTIVE)
        all_candidates_final_test.to_csv(Path(cfg.reports_dir) / "all_candidates_final_test_metrics.csv", index=False)
        display(all_candidates_final_test)
        """
    ),
    md("## 12. Interpretation, Statistics, Backend Bundle"),
    code(
        """
        review_dir = Path(cfg.artifacts_dir) / "case_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        for j, row in best_routes.head(min(8, len(best_routes))).iterrows():
            study_id = row["study_id"]
            result_idx = next(i for i, r in enumerate(results) if r.study_id == study_id)
            r = results[result_idx]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(get_result_raw_preview(r), cmap="gray")
            axes[0].set_title("Original normalized")
            axes[1].imshow(get_result_image_eva(r), cmap="gray")
            axes[1].set_title("EVA input")
            roi_img = get_result_image_roi(r)
            axes[2].imshow(roi_img if roi_img is not None else get_result_image_eva(r), cmap="gray")
            axes[2].set_title(f"{row['route']} | p={row['p_requires_attention']:.3f}")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            out = review_dir / f"case_{j}_{str(row['route']).replace('/', '_')}.png"
            fig.savefig(out, dpi=160, bbox_inches="tight")
            plt.show()

        # Calibration and bootstrap CI.
        y_val, p_val, _, val_idx = score_candidate_split(best, "validation", ood_model=None)
        y_test, p_test, _, test_idx = score_candidate_split(best, "final_test", ood_model=None)
        val_calibration = calibration_table(y_val, p_val, n_bins=10)
        test_calibration = calibration_table(y_test, p_test, n_bins=10)
        val_calibration.to_csv(Path(cfg.reports_dir) / "calibration_table_validation.csv", index=False)
        test_calibration.to_csv(Path(cfg.reports_dir) / "calibration_table_final_test.csv", index=False)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for table, label in [(val_calibration, "validation"), (test_calibration, "final_test")]:
            nonempty = table.dropna(subset=["mean_pred", "empirical_rate"])
            ax.plot(nonempty["mean_pred"], nonempty["empirical_rate"], marker="o", label=label)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Empirical requires_attention rate")
        ax.set_title("Reliability diagram")
        ax.legend()
        reliability_path = Path(cfg.reports_dir) / "reliability_diagram.png"
        fig.savefig(reliability_path, dpi=160, bbox_inches="tight")
        plt.show()

        ci_rows = []
        for split_name, yy, pp in [("validation", y_val, p_val), ("final_test", y_test, p_test)]:
            for metric_name, fn in {
                "AUROC": safe_auc,
                "AUPRC": safe_auprc,
                "Brier": lambda y0, p0: brier_score_loss(y0, p0),
                "raw_score_NPV@selected_T_negative": lambda y0, p0: npv_at_threshold(y0, p0, float(best["selected_T_negative"])),
            }.items():
                point, lo, hi = bootstrap_ci(yy, pp, fn, n_boot=1000, seed=cfg.random_state)
                ci_rows.append({"split": split_name, "metric": metric_name, "point": point, "ci95_low": lo, "ci95_high": hi})
        ci_report = pd.DataFrame(ci_rows)
        ci_report.to_csv(Path(cfg.reports_dir) / "bootstrap_ci_report.csv", index=False)
        display(ci_report)

        validation_case_report = best_routes.copy()
        validation_case_report["y_attention"] = y_val
        validation_case_report["split"] = "validation"
        validation_case_report["predicted_at_0.5"] = (p_val >= 0.5).astype(int)
        validation_case_report["classification_error_at_0.5"] = validation_case_report["predicted_at_0.5"] != y_val
        validation_case_report.to_csv(Path(cfg.reports_dir) / "best_case_level_validation.csv", index=False)

        final_case_report = final_test_routes.copy()
        final_case_report["y_attention"] = y_test
        final_case_report["split"] = "final_test"
        final_case_report["predicted_at_0.5"] = (p_test >= 0.5).astype(int)
        final_case_report["classification_error_at_0.5"] = final_case_report["predicted_at_0.5"] != y_test
        final_case_report.to_csv(Path(cfg.reports_dir) / "best_case_level_final_test.csv", index=False)
        final_case_report[final_case_report["classification_error_at_0.5"]].to_csv(
            Path(cfg.reports_dir) / "best_error_analysis_final_test.csv",
            index=False,
        )
        route_reason_summary = (
            final_case_report.groupby(["route", "reason", "y_attention"], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        route_reason_summary.to_csv(Path(cfg.reports_dir) / "best_route_reason_summary_final_test.csv", index=False)
        display(route_reason_summary)

        # Backend bundle.
        if "chexfound_feasibility" not in globals():
            chexfound_feasibility = {
                "enabled": bool(RUN.get("run_chexfound_feasibility", False)),
                "status": "not_run_for_this_experiment_profile",
                "message": "CheXFound optional block was not executed in this run.",
            }
        bundle_dir = Path(cfg.artifacts_dir) / ("backend_bundle" if deployment_ready else "research_bundle")
        bundle_dir.mkdir(parents=True, exist_ok=True)
        best_ood_model = fit_ood_model(best["parts"]["train"][0])
        save_pickle(best_ood_model, bundle_dir / "best_ood_model.pkl")
        save_pickle(best["model"], bundle_dir / "best_model.pkl")
        if torch is not None and isinstance(best["model"], nn.Module):
            torch.save(
                {
                    "state_dict": best["model"].state_dict(),
                    "kind": best.get("kind"),
                    "variant": best.get("variant"),
                    "adapt_cfg": best.get("adapt_cfg"),
                },
                bundle_dir / "best_model_state_dict.pt",
            )
        if best.get("calibrator") is not None:
            save_pickle(best["calibrator"], bundle_dir / "best_calibrator.pkl")
        router_config = {
            "deployment_ready": bool(deployment_ready),
            "research_best_model": research_best_name,
            "deployment_candidate_model": deployment_best_name if deployment_ready else None,
            "research_bundle_model": deployment_best_name if not deployment_ready else None,
            "selected_T_negative": float(best["selected_T_negative"]),
            "selected_t_ood": float(best.get("selected_t_ood", 0.95)),
            "selected_t_positive": float(best.get("selected_t_positive", 0.80)),
            "selected_t_quality": float(best.get("selected_t_quality", 0.35)),
            "selected_t_uncertainty": float(best.get("selected_t_uncertainty", 0.65)),
            "threshold_policy": best.get("threshold_policy"),
            "calibration_method": best.get("calibration_method", getattr(best.get("calibrator"), "method", "n/a")),
            "variant": best.get("variant"),
            "kind": best.get("kind"),
            "router_logic": "quality/OOD checks -> T_negative no_attention_required -> T_positive requires_attention -> gray-zone N/A",
        }
        export_json(router_config, bundle_dir / "router_config.json")
        preprocessing_config = {
            "preprocessing_version": "preproc_v1",
            "target_contract": "0=no_attention_required; 1=requires_attention; N/A is router output only",
            "eva_image_size": int(cfg.eva_image_size),
            "feature_extractor_family": "EVA-X" if str(best.get("variant")) in {"small", "base", "tiny", "s", "b", "ti"} else str(best.get("variant")),
            "variant": best.get("variant"),
            "model_kind": best.get("kind"),
            "input_mode": "frozen_features" if best.get("kind") in {"torch_mlp", "sklearn"} else "image_end_to_end",
            "quality_score_field": "quality_score",
            "ood_feature_space": "frozen encoder features used during model selection",
        }
        export_json(preprocessing_config, bundle_dir / "preprocessing_config.json")
        manifest = {
            "project": "fluoro_cxr_backend_mvp_candidate",
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "experiment_profile": RUN["experiment_profile"],
            "run_name": RUN["run_name"],
            "dataset": "IN-CXR Kaggle PNG mirror or user-provided IN-CXR path",
            "selection_objective": SELECTION_OBJECTIVE,
            "target_npv": cfg.target_npv,
            "deployment_ready": bool(deployment_ready),
            "research_best_model": research_best_name,
            "deployment_candidate_model": deployment_best_name if deployment_ready else None,
            "research_bundle_model": deployment_best_name if not deployment_ready else None,
            "deployment_calibration_method": best.get("calibration_method", getattr(best.get("calibrator"), "method", "n/a")),
            "enabled_heavy_branches": {
                "eva_lora_sweep": bool(RUN.get("run_lora_sweep")),
                "eva_partial_unfreeze": bool(RUN.get("run_partial_unfreeze")),
                "eva_base_lora": bool(RUN.get("run_eva_base_lora")),
                "chexfound_inference": bool(RUN.get("run_chexfound_inference")),
                "chexfound_lora": bool(RUN.get("run_chexfound_lora")),
            },
            "lora_sweep": RUN.get("lora_sweep", []),
            "partial_unfreeze_sweep": RUN.get("partial_unfreeze_sweep", []),
            "backend_bundle": str(bundle_dir),
            "reports_dir": cfg.reports_dir,
            "artifacts_dir": cfg.artifacts_dir,
            "chexfound_feasibility": chexfound_feasibility,
            "final_safety_pass": bool(globals().get("final_safety_pass", False)),
            "final_safety_downgrade_reason": globals().get("final_safety_downgrade_reason", ""),
            "runtime_versions": {
                "python": sys.version,
                "torch": getattr(torch, "__version__", None),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "data_limitations": [
                "IN-CXR Kaggle mirror is preprocessed 224x224 PNG rather than original DICOM.",
                "The mirror exposes no verified patient identifier; exact duplicate leakage is prevented by content hashing.",
                "Normal/abnormal labels come from a tuberculosis prevalence screening context and do not prove universal thoracic abnormality coverage.",
            ],
        }
        export_json(manifest, Path(cfg.artifacts_dir) / "manifest.json")
        export_json(manifest, bundle_dir / "manifest.json")
        print("Backend bundle:" if deployment_ready else "Research-only bundle:", bundle_dir)
        """
    ),
    md("## 13. Final Sanity Checks"),
    code(
        """
        assert len(df) > 0
        assert len(results) == len(meta)
        assert Path(cfg.reports_dir, "model_comparison.csv").exists()
        assert Path(cfg.reports_dir, "all_candidates_final_test_metrics.csv").exists()
        expected_bundle_dir = Path(cfg.artifacts_dir) / ("backend_bundle" if deployment_ready else "research_bundle")
        assert Path(expected_bundle_dir, "manifest.json").exists()
        assert Path(expected_bundle_dir, "best_ood_model.pkl").exists()
        assert Path(expected_bundle_dir, "preprocessing_config.json").exists()
        assert final_test_routes["route"].isin(["no_attention_required", "requires_attention", "N/A"]).all()
        print("MODEL SELECTION NOTEBOOK SANITY CHECKS PASSED")
        print("Artifacts:", cfg.artifacts_dir)
        print("Reports:", cfg.reports_dir)
        """
    ),
    md("## 14. Export Single-Experiment Archive"),
    code(
        """
        import shutil

        candidate_project_dirs = [
            Path(getattr(cfg, "project_dir", "")),
            Path(globals().get("PROJECT_DIR", "")),
            Path(cfg.artifacts_dir).parent,
            Path(cfg.reports_dir).parent,
            Path(cfg.checkpoints_dir).parent,
        ]
        existing_project_dirs = []
        for p in candidate_project_dirs:
            if str(p) and p.exists() and p.is_dir() and p not in existing_project_dirs:
                existing_project_dirs.append(p)
        if not existing_project_dirs:
            raise FileNotFoundError(
                "Could not find an existing project directory to archive. "
                f"Checked: {[str(p) for p in candidate_project_dirs]}"
            )

        export_project_dir = existing_project_dirs[0]
        archive_root = Path(RUN.get("project_base_dir", export_project_dir.parent))
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_base = archive_root / f"{RUN['run_name']}_export"
        print("Archiving project directory:", export_project_dir)
        print("Archive target base:", archive_base)

        archive_path = shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=str(export_project_dir.parent),
            base_dir=export_project_dir.name,
        )
        archive_manifest = {
            "experiment_profile": RUN["experiment_profile"],
            "run_name": RUN["run_name"],
            "project_dir": str(export_project_dir),
            "archive_path": archive_path,
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "note": "One archive per runtime experiment. Temporary preprocessing cache is intentionally outside this archive.",
        }
        export_json(archive_manifest, Path(cfg.artifacts_dir) / "archive_manifest.json")
        print("Single-experiment archive:", archive_path)
        print("Upload/share this zip together with the executed notebook if you want notebook outputs preserved.")
        """
    ),
]


VINDR_CELLS = [
    md(
        """
        # Fluoro MVP VinDr BBox Interpretation

        Separate notebook for localization sanity checks on VinDr/VinBigData.

        This notebook does not replace IN-CXR model selection. It answers a narrower question:
        do EVA-based model scores depend on image regions that overlap radiologist bounding boxes?
        """
    ),
    md("## 0. Central Run Config"),
    code(
        """
        import os
        from pathlib import Path

        try:
            import google.colab  # noqa: F401
            IN_COLAB_BOOTSTRAP = True
        except Exception:
            IN_COLAB_BOOTSTRAP = False

        RUN = {
            "run_name": "vindr_bbox_interpretation_t4",
            "max_vindr_studies": 5000,
            "download_vindr_from_kaggle": True,
            "vindr_download_mode": "png512",
            "interpretation_cases": 30,
            "heatmap_grid": 8,
            # primary = single best last1 model; ensemble = last1+last2 router;
            # both = render and report both tracks in one diagnostic run.
            "interpretation_mode": os.environ.get("VINDR_INTERPRETATION_MODE", "both").lower(),
            "batch_size": 8,
            "eva_image_size": 224,
            "project_base_dir": "/content/fluoro_mvp_runs" if IN_COLAB_BOOTSTRAP else str(Path.cwd() / "fluoro_mvp_outputs"),
            "primary_bundle_dir": os.environ.get("PRIMARY_BUNDLE_DIR", os.environ.get("BEST_BUNDLE_DIR", "")),
            "ensemble_bundle_dir": os.environ.get("ENSEMBLE_BUNDLE_DIR", ""),
        }
        if not RUN["primary_bundle_dir"]:
            local_primary = Path.cwd() / "selected_model_workbench" / "router_workbench" / "primary_deployment_bundle"
            if local_primary.exists():
                RUN["primary_bundle_dir"] = str(local_primary)
        if not RUN["ensemble_bundle_dir"]:
            local_ensemble = Path.cwd() / "selected_model_workbench" / "router_workbench" / "ensemble_candidate_bundle"
            if local_ensemble.exists():
                RUN["ensemble_bundle_dir"] = str(local_ensemble)
        RUN["best_bundle_dir"] = RUN["primary_bundle_dir"]
        PROJECT_DIR = str(Path(RUN["project_base_dir"]) / RUN["run_name"])
        os.environ["PROJECT_DIR"] = PROJECT_DIR
        os.environ["DOWNLOAD_VINDR_FROM_KAGGLE"] = "1" if RUN["download_vindr_from_kaggle"] else "0"
        os.environ["VINDR_DOWNLOAD_MODE"] = RUN["vindr_download_mode"]
        os.environ["VINDR_DOWNLOAD_MAX_STUDIES"] = str(RUN["max_vindr_studies"])
        os.environ["MAX_VINDR_STUDIES"] = str(RUN["max_vindr_studies"])
        os.environ["PREPROCESSED_CACHE_DIR"] = "/content/vindr_preprocessed_cache" if IN_COLAB_BOOTSTRAP else str(Path.cwd() / "vindr_preprocessed_cache")
        if IN_COLAB_BOOTSTRAP:
            os.environ["MOUNT_DRIVE"] = "0"
        else:
            os.environ.setdefault("MOUNT_DRIVE", "0")

        print("Run config:")
        for key, value in RUN.items():
            print(f"  {key}: {value}")
        print("PROJECT_DIR:", PROJECT_DIR)
        """
    ),
    md("## 1. Environment Setup"),
    code(COMMON_SETUP),
    md("## 2. Tokens and VinDr Download"),
    code(
        AUTH_AND_DOWNLOAD
        + """
setup_hf_token()
setup_kaggle_token()
if RUN["download_vindr_from_kaggle"]:
    maybe_download_vindr_png_subset_from_kaggle(
        os.environ.get("VINDR_DOWNLOAD_DIR", "/content/vindr_cxr"),
        max_studies=int(RUN["max_vindr_studies"]),
    )
"""
    ),
    md("## 3. Core Code"),
    code(CORE_SOURCE),
    md("## 4. Runtime Config"),
    code(
        """
        import gc
        import warnings
        from dataclasses import asdict

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB and os.environ.get("MOUNT_DRIVE", "0") == "1":
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)

        cfg = NotebookConfig(
            project_dir=PROJECT_DIR,
            vindr_root=os.environ.get("VINDR_ROOT") or None,
            max_vindr_studies=int(RUN["max_vindr_studies"]),
            random_state=42,
            target_npv=0.99,
            eva_image_size=int(RUN["eva_image_size"]),
            batch_size=int(RUN["batch_size"]),
            run_real_google_cxr=False,
            run_real_eva_x=True,
            run_primary_track=False,
            run_vindr_track=True,
            run_vindr_exp2=True,
            cache_preprocessed_to_disk=True,
            preprocessed_cache_dir=os.environ["PREPROCESSED_CACHE_DIR"],
        )
        if not cfg.vindr_root:
            raise RuntimeError("VINDR_ROOT is missing. Run the VinDr download cell or set it manually.")
        ensure_dirs(cfg)
        set_seed(cfg.random_state)
        if torch is not None:
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        print(asdict(cfg))
        """
    ),
    md("## 5. VinDr Dataset Loading and Preprocessing"),
    code(
        """
        vindr_df, vindr_bboxes = discover_vindr_dataset(cfg.vindr_root, max_studies=cfg.max_vindr_studies, cfg=cfg)
        vindr_dataset_audit = validate_binary_dataset_contract(vindr_df)
        vindr_df = make_splits(vindr_df, seed=cfg.random_state)
        vindr_split_audit = validate_split_integrity(vindr_df)
        export_json(
            {"dataset": vindr_dataset_audit, "splits": vindr_split_audit},
            Path(cfg.reports_dir) / "vindr_dataset_contract_audit.json",
        )
        vindr_results, vindr_meta = preprocess_dataframe(vindr_df, cfg)
        y_vindr = vindr_meta["y_attention"].values.astype(int)
        save_table(vindr_df, Path(cfg.artifacts_dir) / "vindr_data_index")
        save_table(vindr_meta, Path(cfg.artifacts_dir) / "vindr_preprocessing_report")
        vindr_bboxes.to_csv(Path(cfg.artifacts_dir) / "vindr_bboxes.csv", index=False)
        print("VinDr dataframe:", vindr_df.shape)
        print("VinDr bboxes:", vindr_bboxes.shape)
        display(vindr_df["y_attention"].value_counts().rename("count").to_frame())
        display(vindr_df["split"].value_counts().rename("count").to_frame())
        display(vindr_bboxes.head())
        """
    ),
    md("## 6. EVA Model, Features, and Scoring Head"),
    code(
        """
        bundle_dir = Path(RUN["best_bundle_dir"]) if RUN["best_bundle_dir"] else None
        required_bundle_files = [
            "manifest.json",
            "router_config.json",
            "preprocessing_config.json",
            "best_model.pkl",
            "best_ood_model.pkl",
        ]
        if bundle_dir is None or not bundle_dir.exists():
            raise RuntimeError(
                "Set BEST_BUNDLE_DIR to the backend_bundle directory exported by the model-selection notebook."
            )
        missing_bundle_files = [name for name in required_bundle_files if not (bundle_dir / name).exists()]
        if missing_bundle_files:
            raise RuntimeError(f"Backend bundle is incomplete; missing files: {missing_bundle_files}")

        backend_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        router_config = json.loads((bundle_dir / "router_config.json").read_text(encoding="utf-8"))
        preprocessing_config = json.loads((bundle_dir / "preprocessing_config.json").read_text(encoding="utf-8"))
        if not bool(router_config.get("deployment_ready", True)):
            raise RuntimeError("The supplied bundle is marked research-only and cannot drive the product diagnostic notebook.")
        backend_kind = str(router_config.get("kind") or preprocessing_config.get("model_kind") or "")
        backend_variant = str(router_config.get("variant") or preprocessing_config.get("variant") or "").lower()
        preprocessing_kind = str(preprocessing_config.get("model_kind") or "")
        preprocessing_variant = str(preprocessing_config.get("variant") or "").lower()
        input_mode = str(preprocessing_config.get("input_mode") or "")
        if preprocessing_kind and backend_kind and preprocessing_kind != backend_kind:
            raise RuntimeError(
                f"Bundle contract mismatch: router kind={backend_kind!r}, preprocessing model_kind={preprocessing_kind!r}."
            )
        if preprocessing_variant and backend_variant and preprocessing_variant != backend_variant:
            raise RuntimeError(
                f"Bundle contract mismatch: router variant={backend_variant!r}, preprocessing variant={preprocessing_variant!r}."
            )
        if backend_kind in {"eva_e2e", "lora_e2e", "partial_unfreeze_e2e"} and input_mode != "image_end_to_end":
            raise RuntimeError(
                f"Adapted EVA bundle must use input_mode='image_end_to_end'; received {input_mode!r}."
            )
        if backend_variant not in EVA_X_VARIANTS:
            raise RuntimeError(
                f"VinDr diagnostic notebook currently supports EVA bundles only; received variant={backend_variant!r}, "
                f"kind={backend_kind!r}."
            )
        if int(preprocessing_config.get("eva_image_size", cfg.eva_image_size)) != int(cfg.eva_image_size):
            raise RuntimeError(
                "Diagnostic eva_image_size does not match the selected backend bundle preprocessing contract."
            )

        frozen_eva_model = load_real_eva_x(cfg.project_dir, variant=backend_variant, device=cfg.device)
        feat_path = Path(cfg.artifacts_dir) / "embeddings" / f"vindr_eva_{backend_variant}_features.npy"
        if feat_path.exists():
            X_vindr = np.load(feat_path)
            print("Loaded cached VinDr EVA features:", X_vindr.shape)
        else:
            X_vindr = extract_eva_features_real(
                frozen_eva_model,
                vindr_results,
                image_size=cfg.eva_image_size,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
            np.save(feat_path, X_vindr)
            print("Saved VinDr EVA features:", X_vindr.shape)
        parts_vindr = split_arrays_by_meta(X_vindr, y_vindr, vindr_meta, require_all_splits=True)

        backend_calibrator = joblib.load(bundle_dir / "best_calibrator.pkl") if (bundle_dir / "best_calibrator.pkl").exists() else None
        backend_ood_model = joblib.load(bundle_dir / "best_ood_model.pkl")
        loaded_backend_bundle = True
        adapted_eva_model = None
        if backend_kind in {"torch_mlp", "sklearn"}:
            backend_model = joblib.load(bundle_dir / "best_model.pkl")
        elif backend_kind in {"eva_e2e", "lora_e2e", "partial_unfreeze_e2e"}:
            state_path = bundle_dir / "best_model_state_dict.pt"
            if not state_path.exists():
                raise RuntimeError("Adapted EVA bundle is missing best_model_state_dict.pt.")
            adapted_eva_model, adapted_checkpoint_info = load_eva_end_to_end_checkpoint(
                cfg.project_dir,
                state_path,
                device=cfg.device,
                image_size=cfg.eva_image_size,
            )
            backend_model = adapted_eva_model
        else:
            raise RuntimeError(f"Unsupported backend model kind for VinDr interpretation: {backend_kind!r}")

        def predict_vindr_features(X_batch):
            is_torch_head = (
                isinstance(backend_model, TorchMLP)
                or (torch is not None and isinstance(backend_model, nn.Module) and hasattr(backend_model, "scaler"))
            )
            if is_torch_head:
                raw = predict_torch_mlp(backend_model, X_batch, device=cfg.device)
                return backend_calibrator.transform(raw) if backend_calibrator is not None else raw
            if hasattr(backend_model, "predict_proba"):
                return backend_model.predict_proba(X_batch)[:, 1]
            raw = predict_torch_mlp(backend_model, X_batch, device=cfg.device)
            return backend_calibrator.transform(raw) if backend_calibrator is not None else raw

        def predict_vindr_indices(indices):
            indices = np.asarray(indices, dtype=int)
            if backend_kind in {"torch_mlp", "sklearn"}:
                return predict_vindr_features(X_vindr[indices])
            images = [get_result_image_eva(vindr_results[int(i)]) for i in indices]
            raw = predict_eva_end_to_end_images(
                adapted_eva_model,
                images,
                image_size=cfg.eva_image_size,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
            return backend_calibrator.transform(raw) if backend_calibrator is not None else raw

        val_indices = parts_vindr["validation"][2]
        p_vindr_val = predict_vindr_indices(val_indices)
        ood_vindr_val = ood_score(backend_ood_model, parts_vindr["validation"][0])
        vindr_metrics, vindr_routes = fixed_threshold_evaluation(
            "VinDr external diagnostic with fixed IN-CXR router",
            parts_vindr["validation"][1],
            p_vindr_val,
            vindr_meta.iloc[val_indices].reset_index(drop=True),
            t_negative=float(router_config["selected_T_negative"]),
            t_positive=float(router_config["selected_t_positive"]),
            t_quality=float(router_config["selected_t_quality"]),
            t_uncertainty=float(router_config["selected_t_uncertainty"]),
            ood_score_values=ood_vindr_val,
            t_ood=float(router_config["selected_t_ood"]),
        )
        vindr_thr = threshold_report(parts_vindr["validation"][1], p_vindr_val, target_npv=cfg.target_npv)
        pd.DataFrame([vindr_metrics]).to_csv(Path(cfg.reports_dir) / "vindr_model_metrics_validation.csv", index=False)
        vindr_thr.to_csv(Path(cfg.reports_dir) / "vindr_threshold_report.csv", index=False)
        vindr_routes.to_csv(Path(cfg.reports_dir) / "vindr_routes_validation.csv", index=False)
        display(pd.DataFrame([vindr_metrics]))
        display(vindr_thr.head(10))
        """
    ),
    md("## 6.5 Ensemble Bundle Scoring"),
    code(
        """
        ensemble_ready = False
        ensemble_bundle_dir = Path(RUN["ensemble_bundle_dir"]) if RUN.get("ensemble_bundle_dir") else None
        ensemble_context = {}

        def ensemble_route_decisions(p_last1, p_last2, meta_df, router_cfg, ood_score_values=None):
            p_last1 = np.asarray(p_last1, dtype=float)
            p_last2 = np.asarray(p_last2, dtype=float)
            t_l1_neg = float(router_cfg["selected_t_last1_negative"])
            t_l2_neg = float(router_cfg["selected_t_last2_negative"])
            t_l1_veto = float(router_cfg["selected_t_last1_veto"])
            t_l2_veto = float(router_cfg["selected_t_last2_veto"])
            t_quality = float(router_cfg.get("selected_t_quality", 0.35))
            t_uncertainty = float(router_cfg.get("selected_t_uncertainty", 0.65))
            t_ood = float(router_cfg.get("selected_t_ood", 1.25))
            t_positive = float(router_cfg.get("selected_t_positive", 0.80))
            rows = []
            for i, (p1, p2) in enumerate(zip(p_last1, p_last2)):
                quality = float(meta_df.iloc[i]["quality_score"])
                critical = bool(meta_df.iloc[i]["critical_qa"])
                p_pair = float(max(p1, p2))
                uncertainty = float(1.0 - abs(p_pair - 0.5) * 2.0)
                ood = float(ood_score_values[i]) if ood_score_values is not None else None
                last1_accepts = p1 <= t_l1_neg and p2 <= t_l2_veto
                last2_accepts = p2 <= t_l2_neg and p1 <= t_l1_veto
                if quality < t_quality or critical:
                    route, reason = "N/A", "bad_quality_or_critical_qa"
                elif ood is not None and ood > t_ood:
                    route, reason = "N/A", "out_of_distribution"
                elif uncertainty > t_uncertainty:
                    route, reason = "N/A", "high_uncertainty"
                elif last1_accepts or last2_accepts:
                    route, reason = "no_attention_required", "ensemble_confident_no_attention_required"
                elif p_pair >= t_positive:
                    route, reason = "requires_attention", "ensemble_suspicious_requires_attention"
                else:
                    route, reason = "N/A", "gray_zone"
                rows.append({
                    "study_id": meta_df.iloc[i]["study_id"],
                    "p_last1_requires_attention": float(p1),
                    "p_last2_requires_attention": float(p2),
                    "p_requires_attention": p_pair,
                    "quality_score": quality,
                    "ood_score": ood,
                    "uncertainty_score": uncertainty,
                    "route": route,
                    "reason": reason,
                })
            return pd.DataFrame(rows)

        def ensemble_fixed_metrics(name, y_true, p_last1, p_last2, meta_df, router_cfg, ood_score_values=None):
            p_pair = np.maximum(np.asarray(p_last1, dtype=float), np.asarray(p_last2, dtype=float))
            routes = ensemble_route_decisions(p_last1, p_last2, meta_df, router_cfg, ood_score_values=ood_score_values)
            metrics = metrics_summary(y_true, p_pair)
            metrics.update(route_metrics(y_true, routes))
            y_arr = np.asarray(y_true).astype(int)
            auto = routes["route"].values == "no_attention_required"
            selected_count = int(np.sum(auto))
            fn = int(np.sum((y_arr == 1) & auto))
            tn = int(np.sum((y_arr == 0) & auto))
            metrics.update({
                "model": name,
                "score_definition": "max(last1_calibrated_probability,last2_calibrated_probability)",
                "fixed_threshold_selected_count": selected_count,
                "fixed_threshold_TN_count": tn,
                "fixed_threshold_FN_count": fn,
                "fixed_threshold_NPV_ci95_low": wilson_lower_bound(tn, tn + fn, z=1.96),
                "fixed_T_last1_negative": float(router_cfg["selected_t_last1_negative"]),
                "fixed_T_last2_negative": float(router_cfg["selected_t_last2_negative"]),
                "fixed_T_last1_veto": float(router_cfg["selected_t_last1_veto"]),
                "fixed_T_last2_veto": float(router_cfg["selected_t_last2_veto"]),
            })
            return metrics, routes, p_pair

        if RUN["interpretation_mode"] in {"ensemble", "both"}:
            if ensemble_bundle_dir is None or not ensemble_bundle_dir.exists():
                raise RuntimeError(
                    "VINDR_INTERPRETATION_MODE requests ensemble, but ENSEMBLE_BUNDLE_DIR is not set or does not exist."
                )
            ensemble_required = [
                "manifest.json",
                "router_config.json",
                "preprocessing_config.json",
                "best_ood_model.pkl",
                "models/base_unfreeze_last1_e150_best.pt",
                "models/base_unfreeze_last2_e150_best.pt",
                "calibration/last1_calibrator.pkl",
                "calibration/last2_calibrator.pkl",
            ]
            missing = [name for name in ensemble_required if not (ensemble_bundle_dir / name).exists()]
            if missing:
                raise RuntimeError(f"Ensemble bundle is incomplete; missing files: {missing}")

            ensemble_manifest = json.loads((ensemble_bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            ensemble_router_config = json.loads((ensemble_bundle_dir / "router_config.json").read_text(encoding="utf-8"))
            ensemble_preprocessing_config = json.loads((ensemble_bundle_dir / "preprocessing_config.json").read_text(encoding="utf-8"))
            ensemble_kind = str(ensemble_router_config.get("kind") or "")
            ensemble_variant = str(ensemble_router_config.get("variant") or ensemble_preprocessing_config.get("variant") or "").lower()
            if ensemble_kind != "partial_unfreeze_e2e_ensemble":
                raise RuntimeError(f"Unsupported ensemble bundle kind for VinDr interpretation: {ensemble_kind!r}")
            if ensemble_variant != backend_variant:
                raise RuntimeError(
                    f"Primary and ensemble variants differ: primary={backend_variant!r}, ensemble={ensemble_variant!r}. "
                    "Run separate diagnostic notebooks for different EVA variants."
                )

            ensemble_last1_model, ensemble_last1_info = load_eva_end_to_end_checkpoint(
                cfg.project_dir,
                ensemble_bundle_dir / "models" / "base_unfreeze_last1_e150_best.pt",
                device=cfg.device,
                image_size=cfg.eva_image_size,
            )
            ensemble_last2_model, ensemble_last2_info = load_eva_end_to_end_checkpoint(
                cfg.project_dir,
                ensemble_bundle_dir / "models" / "base_unfreeze_last2_e150_best.pt",
                device=cfg.device,
                image_size=cfg.eva_image_size,
            )
            ensemble_last1_calibrator = joblib.load(ensemble_bundle_dir / "calibration" / "last1_calibrator.pkl")
            ensemble_last2_calibrator = joblib.load(ensemble_bundle_dir / "calibration" / "last2_calibrator.pkl")
            ensemble_ood_model = joblib.load(ensemble_bundle_dir / "best_ood_model.pkl")

            def predict_ensemble_indices(indices):
                images = [get_result_image_eva(vindr_results[int(i)]) for i in np.asarray(indices, dtype=int)]
                raw1 = predict_eva_end_to_end_images(
                    ensemble_last1_model,
                    images,
                    image_size=cfg.eva_image_size,
                    batch_size=cfg.batch_size,
                    device=cfg.device,
                )
                raw2 = predict_eva_end_to_end_images(
                    ensemble_last2_model,
                    images,
                    image_size=cfg.eva_image_size,
                    batch_size=cfg.batch_size,
                    device=cfg.device,
                )
                p1 = np.asarray(ensemble_last1_calibrator.transform(raw1), dtype=np.float32)
                p2 = np.asarray(ensemble_last2_calibrator.transform(raw2), dtype=np.float32)
                return p1, p2

            p_ensemble_last1_val, p_ensemble_last2_val = predict_ensemble_indices(val_indices)
            ood_ensemble_val = ood_score(ensemble_ood_model, parts_vindr["validation"][0])
            vindr_ensemble_metrics, vindr_ensemble_routes, p_ensemble_pair_val = ensemble_fixed_metrics(
                "VinDr external diagnostic with fixed IN-CXR ensemble router",
                parts_vindr["validation"][1],
                p_ensemble_last1_val,
                p_ensemble_last2_val,
                vindr_meta.iloc[val_indices].reset_index(drop=True),
                ensemble_router_config,
                ood_score_values=ood_ensemble_val,
            )
            vindr_ensemble_thr = threshold_report(
                parts_vindr["validation"][1],
                p_ensemble_pair_val,
                target_npv=cfg.target_npv,
            )
            pd.DataFrame([vindr_ensemble_metrics]).to_csv(Path(cfg.reports_dir) / "vindr_ensemble_model_metrics_validation.csv", index=False)
            vindr_ensemble_thr.to_csv(Path(cfg.reports_dir) / "vindr_ensemble_threshold_report.csv", index=False)
            vindr_ensemble_routes.to_csv(Path(cfg.reports_dir) / "vindr_ensemble_routes_validation.csv", index=False)
            ensemble_context = {
                "bundle_dir": str(ensemble_bundle_dir),
                "manifest": ensemble_manifest,
                "router_config": ensemble_router_config,
                "preprocessing_config": ensemble_preprocessing_config,
                "last1_model": ensemble_last1_model,
                "last2_model": ensemble_last2_model,
                "last1_calibrator": ensemble_last1_calibrator,
                "last2_calibrator": ensemble_last2_calibrator,
                "last1_info": ensemble_last1_info,
                "last2_info": ensemble_last2_info,
                "metrics": vindr_ensemble_metrics,
                "routes": vindr_ensemble_routes,
                "threshold_report": vindr_ensemble_thr,
            }
            ensemble_ready = True
            print("Ensemble bundle loaded:", ensemble_bundle_dir)
            display(pd.DataFrame([vindr_ensemble_metrics]))
            display(vindr_ensemble_thr.head(10))
        else:
            print("Ensemble interpretation disabled by VINDR_INTERPRETATION_MODE.")
        """
    ),
    md("## 7. BBox-aware Occlusion Heatmaps"),
    code(
        """
        interp_dir = Path(cfg.artifacts_dir) / "vindr_interpretation"
        interp_dir.mkdir(parents=True, exist_ok=True)
        val_indices = parts_vindr["validation"][2]
        candidate_indices = [
            int(i) for i in val_indices
            if y_vindr[int(i)] == 1 and not vindr_bboxes[vindr_bboxes["study_id"].astype(str) == str(vindr_results[int(i)].study_id)].empty
        ]
        if not candidate_indices:
            candidate_indices = [int(i) for i in val_indices[: min(int(RUN["interpretation_cases"]), len(val_indices))]]
        candidate_indices = candidate_indices[: int(RUN["interpretation_cases"])]

        metric_rows = []
        for rank, idx in enumerate(candidate_indices):
            result = vindr_results[idx]
            heatmap = make_eva_occlusion_heatmap(
                frozen_eva_model,
                predict_vindr_features,
                result,
                image_size=cfg.eva_image_size,
                batch_size=cfg.batch_size,
                device=cfg.device,
                grid=int(RUN["heatmap_grid"]),
                fill_value=0,
            ) if backend_kind in {"torch_mlp", "sklearn"} else make_eva_end_to_end_calibrated_occlusion_heatmap(
                adapted_eva_model,
                backend_calibrator,
                result,
                image_size=cfg.eva_image_size,
                batch_size=cfg.batch_size,
                device=cfg.device,
                grid=int(RUN["heatmap_grid"]),
                fill_value=0,
            )
            bbox_mask = bbox_mask_for_result(result, vindr_bboxes, target=cfg.eva_image_size)
            loc_metrics = heatmap_localization_metrics(heatmap, bbox_mask)
            loc_metrics.update({"study_id": result.study_id, "rank": rank, "y_attention": int(result.y_attention)})
            metric_rows.append(loc_metrics)

            full_img = np.asarray(get_result_image_eva(result).resize((cfg.eva_image_size, cfg.eva_image_size)))
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.ravel()
            axes[0].imshow(get_result_raw_preview(result), cmap="gray")
            axes[0].set_title("Original normalized")
            axes[1].imshow(full_img, cmap="gray")
            axes[1].set_title("EVA input")
            axes[2].imshow(full_img, cmap="gray")
            axes[2].contour(bbox_mask, levels=[0.5], colors="cyan", linewidths=2)
            axes[2].set_title("Radiologist bbox")
            axes[3].imshow(full_img, cmap="gray")
            axes[3].imshow(heatmap, cmap="magma", alpha=0.55)
            axes[3].contour(bbox_mask, levels=[0.5], colors="cyan", linewidths=2)
            axes[3].set_title("Occlusion heatmap + bbox")
            axes[4].imshow(heatmap, cmap="magma")
            axes[4].contour(bbox_mask, levels=[0.5], colors="cyan", linewidths=2)
            axes[4].set_title(
                f"Energy={loc_metrics['energy_inside_bbox']:.2f} | "
                f"Pointing={loc_metrics['pointing_game_hit']:.0f} | "
                f"IoU={loc_metrics['bbox_iou_at_top20pct']:.2f}"
            )
            axes[5].axis("off")
            axes[5].text(
                0,
                1,
                "How to read:\\n"
                "Magma = areas where occlusion reduced model score.\\n"
                "Cyan = radiologist bbox.\\n"
                "Energy inside bbox shows attribution mass inside bbox.\\n"
                "Pointing game checks whether the hottest point lands inside bbox.\\n"
                "This validates spatial plausibility, not clinical correctness.",
                va="top",
                fontsize=12,
            )
            for ax in axes[:5]:
                ax.axis("off")
            fig.suptitle(f"VinDr case {rank}: {result.study_id}", fontsize=16)
            plt.tight_layout()
            out = interp_dir / f"vindr_case_{rank:03d}_{result.study_id}.png"
            fig.savefig(out, dpi=180, bbox_inches="tight")
            plt.show()
            print("Saved:", out)

        heatmap_metrics = pd.DataFrame(metric_rows)
        heatmap_metrics.to_csv(Path(cfg.reports_dir) / "vindr_heatmap_localization_metrics.csv", index=False)
        display(heatmap_metrics.describe(include="all"))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(heatmap_metrics["energy_inside_bbox"].dropna(), bins=12)
        axes[0].set_title("Energy inside bbox")
        axes[1].bar(["hit", "miss"], [
            float((heatmap_metrics["pointing_game_hit"] == 1).sum()),
            float((heatmap_metrics["pointing_game_hit"] == 0).sum()),
        ])
        axes[1].set_title("Pointing game")
        axes[2].hist(heatmap_metrics["bbox_iou_at_top20pct"].dropna(), bins=12)
        axes[2].set_title("Top-20% heatmap IoU")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        summary_png = Path(cfg.reports_dir) / "vindr_interpretation_summary.png"
        fig.savefig(summary_png, dpi=180, bbox_inches="tight")
        plt.show()
        print("Saved:", summary_png)
        """
    ),
    md("## 7.5 Ensemble BBox-aware Heatmaps"),
    code(
        """
        ensemble_heatmap_metrics = pd.DataFrame()
        ensemble_interp_dir = Path(cfg.artifacts_dir) / "vindr_ensemble_interpretation"
        if ensemble_ready:
            ensemble_interp_dir.mkdir(parents=True, exist_ok=True)
            metric_rows = []
            for rank, idx in enumerate(candidate_indices):
                result = vindr_results[idx]
                heatmap_last1 = make_eva_end_to_end_calibrated_occlusion_heatmap(
                    ensemble_context["last1_model"],
                    ensemble_context["last1_calibrator"],
                    result,
                    image_size=cfg.eva_image_size,
                    batch_size=cfg.batch_size,
                    device=cfg.device,
                    grid=int(RUN["heatmap_grid"]),
                    fill_value=0,
                )
                heatmap_last2 = make_eva_end_to_end_calibrated_occlusion_heatmap(
                    ensemble_context["last2_model"],
                    ensemble_context["last2_calibrator"],
                    result,
                    image_size=cfg.eva_image_size,
                    batch_size=cfg.batch_size,
                    device=cfg.device,
                    grid=int(RUN["heatmap_grid"]),
                    fill_value=0,
                )
                heatmap_combined = np.maximum(heatmap_last1, heatmap_last2)
                if heatmap_combined.max() > 0:
                    heatmap_combined = heatmap_combined / heatmap_combined.max()
                bbox_mask = bbox_mask_for_result(result, vindr_bboxes, target=cfg.eva_image_size)
                route_row = ensemble_context["routes"][ensemble_context["routes"]["study_id"].astype(str) == str(result.study_id)]
                route_text = "not_in_validation_routes"
                if not route_row.empty:
                    rr = route_row.iloc[0]
                    route_text = (
                        f"route={rr['route']} | reason={rr['reason']}\\n"
                        f"p_last1={rr['p_last1_requires_attention']:.3f} | "
                        f"p_last2={rr['p_last2_requires_attention']:.3f} | "
                        f"p_pair={rr['p_requires_attention']:.3f}"
                    )

                metrics_last1 = heatmap_localization_metrics(heatmap_last1, bbox_mask)
                metrics_last2 = heatmap_localization_metrics(heatmap_last2, bbox_mask)
                metrics_combined = heatmap_localization_metrics(heatmap_combined, bbox_mask)
                row = {"study_id": result.study_id, "rank": rank, "y_attention": int(result.y_attention)}
                for prefix, metrics in [
                    ("last1", metrics_last1),
                    ("last2", metrics_last2),
                    ("combined", metrics_combined),
                ]:
                    for key, value in metrics.items():
                        row[f"{prefix}_{key}"] = value
                metric_rows.append(row)

                full_img = np.asarray(get_result_image_eva(result).resize((cfg.eva_image_size, cfg.eva_image_size)))
                fig, axes = plt.subplots(2, 4, figsize=(22, 10))
                axes = axes.ravel()
                axes[0].imshow(get_result_raw_preview(result), cmap="gray")
                axes[0].set_title("Original normalized")
                axes[1].imshow(full_img, cmap="gray")
                axes[1].set_title("EVA input")
                axes[2].imshow(full_img, cmap="gray")
                axes[2].contour(bbox_mask, levels=[0.5], colors="cyan", linewidths=2)
                axes[2].set_title("Radiologist bbox")
                axes[3].axis("off")
                axes[3].text(0, 1, route_text, va="top", fontsize=12)

                panels = [
                    ("last1 heatmap + bbox", heatmap_last1, metrics_last1),
                    ("last2 heatmap + bbox", heatmap_last2, metrics_last2),
                    ("combined max heatmap + bbox", heatmap_combined, metrics_combined),
                ]
                for ax, (title, heatmap, metrics) in zip(axes[4:7], panels):
                    ax.imshow(full_img, cmap="gray")
                    ax.imshow(heatmap, cmap="magma", alpha=0.55)
                    ax.contour(bbox_mask, levels=[0.5], colors="cyan", linewidths=2)
                    ax.set_title(
                        f"{title}\\n"
                        f"Energy={metrics['energy_inside_bbox']:.2f} | "
                        f"Pointing={metrics['pointing_game_hit']:.0f} | "
                        f"IoU={metrics['bbox_iou_at_top20pct']:.2f}"
                    )
                axes[7].axis("off")
                axes[7].text(
                    0,
                    1,
                    "How to read:\\n"
                    "Magma = regions that reduce calibrated risk when occluded.\\n"
                    "Cyan = VinDr radiologist bbox.\\n"
                    "last1/last2 show each model independently.\\n"
                    "combined shows consensus/highest sensitivity.",
                    va="top",
                    fontsize=12,
                )
                for ax in [axes[0], axes[1], axes[2], axes[4], axes[5], axes[6]]:
                    ax.axis("off")
                fig.suptitle(f"VinDr ensemble case {rank}: {result.study_id}", fontsize=16)
                plt.tight_layout()
                out = ensemble_interp_dir / f"vindr_ensemble_case_{rank:03d}_{result.study_id}.png"
                fig.savefig(out, dpi=180, bbox_inches="tight")
                plt.show()
                print("Saved:", out)

            ensemble_heatmap_metrics = pd.DataFrame(metric_rows)
            ensemble_heatmap_metrics.to_csv(Path(cfg.reports_dir) / "vindr_ensemble_heatmap_localization_metrics.csv", index=False)
            display(ensemble_heatmap_metrics.describe(include="all"))

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].hist(ensemble_heatmap_metrics["combined_energy_inside_bbox"].dropna(), bins=12)
            axes[0].set_title("Combined energy inside bbox")
            axes[1].bar(["hit", "miss"], [
                float((ensemble_heatmap_metrics["combined_pointing_game_hit"] == 1).sum()),
                float((ensemble_heatmap_metrics["combined_pointing_game_hit"] == 0).sum()),
            ])
            axes[1].set_title("Combined pointing game")
            axes[2].hist(ensemble_heatmap_metrics["combined_bbox_iou_at_top20pct"].dropna(), bins=12)
            axes[2].set_title("Combined top-20% IoU")
            for ax in axes:
                ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            ensemble_summary_png = Path(cfg.reports_dir) / "vindr_ensemble_interpretation_summary.png"
            fig.savefig(ensemble_summary_png, dpi=180, bbox_inches="tight")
            plt.show()
            print("Saved:", ensemble_summary_png)
        else:
            print("Ensemble heatmaps skipped because ensemble bundle was not loaded.")
        """
    ),
    md("## 8. Interpretation Report Export"),
    code(
        """
        report_path = Path(cfg.reports_dir) / "vindr_interpretation_report.md"
        metrics_summary_text = heatmap_metrics[[
            "energy_inside_bbox",
            "pointing_game_hit",
            "bbox_iou_at_top20pct",
        ]].describe().to_markdown()
        ensemble_metrics_summary_text = ""
        if ensemble_ready and not ensemble_heatmap_metrics.empty:
            ensemble_metrics_summary_text = ensemble_heatmap_metrics[[
                "last1_energy_inside_bbox",
                "last1_pointing_game_hit",
                "last1_bbox_iou_at_top20pct",
                "last2_energy_inside_bbox",
                "last2_pointing_game_hit",
                "last2_bbox_iou_at_top20pct",
                "combined_energy_inside_bbox",
                "combined_pointing_game_hit",
                "combined_bbox_iou_at_top20pct",
            ]].describe().to_markdown()
        report = "\\n".join([
            "# VinDr BBox Interpretation Report",
            "",
            "This report checks spatial plausibility of the EVA-based scoring path on VinDr/VinBigData.",
            "",
            "It does not prove the final IN-CXR screening model is clinically localized. It shows whether the score is sensitive to regions that overlap radiologist bounding boxes on a dataset where boxes are available.",
            "",
            "## Run",
            "",
            f"- EVA variant from backend bundle: `{backend_variant}`",
            f"- Backend model kind: `{backend_kind}`",
            f"- Studies requested: `{RUN['max_vindr_studies']}`",
            f"- Interpretation cases rendered: `{len(heatmap_metrics)}`",
            f"- Backend bundle loaded: `{loaded_backend_bundle}`",
            f"- Ensemble bundle loaded: `{ensemble_ready}`",
            "",
            "## Metrics",
            "",
            "### Primary single-model bundle",
            "",
            metrics_summary_text,
            "",
            "### Ensemble bundle",
            "",
            ensemble_metrics_summary_text if ensemble_metrics_summary_text else "Ensemble interpretation was not run.",
            "",
            "## How to interpret",
            "",
            "- `energy_inside_bbox`: fraction of attribution heatmap mass inside radiologist boxes.",
            "- `pointing_game_hit`: 1 if the hottest heatmap point falls inside a bbox, otherwise 0.",
            "- `bbox_iou_at_top20pct`: overlap between the hottest 20% heatmap area and bbox mask.",
            "- For ensemble panels, `last1` and `last2` are separate EVA-X-B partial-unfreeze models; `combined` is their maximum normalized heatmap.",
            "",
            f"Panels are saved in `{interp_dir}`.",
            f"Ensemble panels are saved in `{ensemble_interp_dir}`." if ensemble_ready else "",
            "",
        ])
        report_path.write_text(report, encoding="utf-8")
        print("Saved:", report_path)

        manifest = {
            "project": "fluoro_vindr_bbox_interpretation",
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "eva_variant": backend_variant,
            "backend_kind": backend_kind,
            "loaded_backend_bundle": loaded_backend_bundle,
            "backend_manifest": backend_manifest,
            "ensemble_ready": bool(ensemble_ready),
            "ensemble_manifest": ensemble_context.get("manifest") if ensemble_ready else None,
            "interpretation_mode": RUN["interpretation_mode"],
            "reports_dir": cfg.reports_dir,
            "artifacts_dir": cfg.artifacts_dir,
            "n_interpretation_cases": int(len(heatmap_metrics)),
            "n_ensemble_interpretation_cases": int(len(ensemble_heatmap_metrics)) if ensemble_ready else 0,
        }
        export_json(manifest, Path(cfg.artifacts_dir) / "manifest.json")

        assert Path(cfg.reports_dir, "vindr_heatmap_localization_metrics.csv").exists()
        assert Path(cfg.reports_dir, "vindr_interpretation_summary.png").exists()
        if ensemble_ready:
            assert Path(cfg.reports_dir, "vindr_ensemble_heatmap_localization_metrics.csv").exists()
            assert Path(cfg.reports_dir, "vindr_ensemble_interpretation_summary.png").exists()
        assert report_path.exists()
        print("VINDR INTERPRETATION NOTEBOOK SANITY CHECKS PASSED")
        """
    ),
    md("## 9. Export Interpretation Archive"),
    code(
        """
        import shutil

        archive_base = Path(RUN["project_base_dir"]) / f"{RUN['run_name']}_export"
        archive_path = shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=str(Path(cfg.project_dir).parent),
            base_dir=Path(cfg.project_dir).name,
        )
        print("VinDr interpretation archive:", archive_path)
        print("Download this zip from the Colab file browser before resetting the runtime.")
        """
    ),
]


RANKING_CLOSURE_CONFIG = """
import os
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB_BOOTSTRAP = True
except Exception:
    IN_COLAB_BOOTSTRAP = False

# One runtime = one ranking experiment. Edit this value before running the notebook:
# eva_base_lora_strong, eva_base_partial_unfreeze, chexfound_frozen.
EXPERIMENT_PROFILE = os.environ.get("EXPERIMENT_PROFILE", "eva_base_lora_strong")

# For the real CheXFound run, set these before the CheXFound profile if the files
# are not already available in your runtime:
# os.environ["CHEXFOUND_CONFIG"] = "/content/path/to/chexfound_config.yaml"
# os.environ["CHEXFOUND_CKPT"] = "/content/path/to/chexfound_checkpoint.pth"

RUN = {
    "run_name": "ranking_closure_incxr_t4",
    "max_studies": 12600,
    "download_incxr_from_kaggle": True,
    "eva_variants": [],
    "run_frozen_heads": False,
    "run_lora_sweep": False,
    "run_partial_unfreeze": False,
    "run_eva_base_lora": False,
    "run_chexfound_feasibility": False,
    "run_chexfound_inference": False,
    "run_chexfound_lora": False,
    "selection_objective": "quality_first",
    "selected_threshold_policy": "auto",
    "target_npv": 0.99,
    "calibration_methods": ["platt", "isotonic", "none"],
    "batch_size": 4,
    "eva_image_size": 224,
    "lora_checkpoint_interval": 15,
    "cache_preprocessed_to_disk": True,
    "preprocessed_cache_dir": "/content/fluoro_mvp_preprocessed_cache" if IN_COLAB_BOOTSTRAP else str(Path.cwd() / "fluoro_mvp_preprocessed_cache"),
    "project_base_dir": "/content/fluoro_mvp_runs" if IN_COLAB_BOOTSTRAP else str(Path.cwd() / "fluoro_mvp_outputs"),
    "lora_sweep": [
        {
            "name": "base_lora_last2_r8_e150",
            "variant": "base",
            "epochs": 150,
            "patience": 20,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.05,
            "n_last_blocks": 2,
            "lr": 5e-5,
            "weight_decay": 1e-4,
            "batch_size": 1,
            "grad_accum_steps": 4,
        },
    ],
    "partial_unfreeze_sweep": [
        {
            "name": "base_unfreeze_last1_e150",
            "variant": "base",
            "epochs": 150,
            "patience": 8,
            "n_last_blocks": 1,
            "lr": 5e-6,
            "head_lr": 4e-4,
            "weight_decay": 1e-4,
            "batch_size": 1,
            "grad_accum_steps": 4,
        },
        {
            "name": "base_unfreeze_last2_e150",
            "variant": "base",
            "epochs": 150,
            "patience": 8,
            "n_last_blocks": 2,
            "lr": 2e-6,
            "head_lr": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 1,
            "grad_accum_steps": 4,
        },
    ],
    "mlp_search_space": [
        {"name": "baseline_h128", "hidden": 128, "dropout": 0.20, "epochs": 80, "lr": 1e-3, "weight_decay": 1e-4, "seed_offset": 0},
        {"name": "strong_h256", "hidden": 256, "dropout": 0.20, "epochs": 120, "lr": 8e-4, "weight_decay": 5e-5, "seed_offset": 1},
        {"name": "regularized_h256", "hidden": 256, "dropout": 0.30, "epochs": 140, "lr": 8e-4, "weight_decay": 1e-4, "seed_offset": 2},
        {"name": "wide_h384", "hidden": 384, "dropout": 0.25, "epochs": 140, "lr": 5e-4, "weight_decay": 5e-5, "seed_offset": 3},
        {"name": "compact_low_wd", "hidden": 192, "dropout": 0.15, "epochs": 120, "lr": 1e-3, "weight_decay": 1e-5, "seed_offset": 4},
        {"name": "conservative_h256", "hidden": 256, "dropout": 0.35, "epochs": 160, "lr": 5e-4, "weight_decay": 2e-4, "seed_offset": 5},
        {"name": "wide_h512", "hidden": 512, "dropout": 0.30, "epochs": 160, "lr": 3e-4, "weight_decay": 1e-4, "seed_offset": 6},
        {"name": "strong_h256_seed2", "hidden": 256, "dropout": 0.20, "epochs": 140, "lr": 8e-4, "weight_decay": 5e-5, "seed_offset": 7},
    ],
}

EXPERIMENT_PRESETS = {
    "eva_base_lora_strong": {
        "run_name": "incxr_eva_base_lora_strong_t4",
        "eva_variants": ["base"],
        "run_frozen_heads": False,
        "run_lora_sweep": True,
        "run_partial_unfreeze": False,
        "run_eva_base_lora": True,
        "run_chexfound_feasibility": False,
        "run_chexfound_inference": False,
        "run_chexfound_lora": False,
    },
    "eva_base_partial_unfreeze": {
        "run_name": "incxr_eva_base_partial_unfreeze_t4",
        "eva_variants": ["base"],
        "run_frozen_heads": False,
        "run_lora_sweep": False,
        "run_partial_unfreeze": True,
        "run_eva_base_lora": False,
        "run_chexfound_feasibility": False,
        "run_chexfound_inference": False,
        "run_chexfound_lora": False,
    },
    "chexfound_frozen": {
        "run_name": "incxr_chexfound_frozen_t4",
        "eva_variants": [],
        "run_frozen_heads": False,
        "run_lora_sweep": False,
        "run_partial_unfreeze": False,
        "run_eva_base_lora": False,
        "run_chexfound_feasibility": True,
        "run_chexfound_inference": True,
        "run_chexfound_lora": False,
    },
}

if EXPERIMENT_PROFILE not in EXPERIMENT_PRESETS:
    raise ValueError(f"Unknown EXPERIMENT_PROFILE={EXPERIMENT_PROFILE!r}. Available: {sorted(EXPERIMENT_PRESETS)}")
RUN.update(EXPERIMENT_PRESETS[EXPERIMENT_PROFILE])
RUN["experiment_profile"] = EXPERIMENT_PROFILE
if not RUN["run_lora_sweep"]:
    RUN["lora_sweep"] = []
if not RUN["run_partial_unfreeze"]:
    RUN["partial_unfreeze_sweep"] = []

PROJECT_DIR = str(Path(RUN["project_base_dir"]) / RUN["run_name"])
os.environ["PROJECT_DIR"] = PROJECT_DIR
os.environ["DOWNLOAD_IN_CXR_FROM_KAGGLE"] = "1" if RUN["download_incxr_from_kaggle"] else "0"
os.environ["MAX_STUDIES"] = str(RUN["max_studies"])
os.environ["PREPROCESSED_CACHE_DIR"] = RUN["preprocessed_cache_dir"]
os.environ["LORA_CHECKPOINT_INTERVAL"] = str(RUN["lora_checkpoint_interval"])
os.environ["FLUORO_NO_GOOGLE_CXR"] = "1"

if IN_COLAB_BOOTSTRAP:
    os.environ["MOUNT_DRIVE"] = "0"
else:
    os.environ.setdefault("MOUNT_DRIVE", "0")

print("Ranking closure config:")
for key, value in RUN.items():
    if key not in {"lora_sweep", "partial_unfreeze_sweep", "mlp_search_space"}:
        print(f"  {key}: {value}")
print("PROJECT_DIR:", PROJECT_DIR)
"""


def ranking_closure_cells() -> list:
    cells = [copy.deepcopy(cell) for cell in MODEL_SELECTION_CELLS]
    cells[0] = md(
        """
        # Fluoro MVP Ranking Closure

        Clean notebook for the final model-ranking experiments before router-only optimization and backend MVP packaging.

        Run one profile per Colab runtime:

        - `eva_base_lora_strong`: one stronger EVA-X-B LoRA candidate, last 2 blocks, rank 8, 150 epochs with early stopping.
        - `eva_base_partial_unfreeze`: EVA-X-B partial unfreeze for last 1 and last 2 blocks.
        - `chexfound_frozen`: CheXFound feasibility plus frozen features/head/reference evaluation.

        Outputs are written to local Colab disk under `/content/fluoro_mvp_runs/<run_name>` and zipped at the end.
        """
    )
    for cell in cells:
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "EXPERIMENT_PRESETS" in src and "PROJECT_DIR" in src:
            cell["source"] = textwrap.dedent(RANKING_CLOSURE_CONFIG).strip()
            break
    else:
        raise RuntimeError("Could not locate model-selection config cell to customize ranking notebook.")
    return cells


def main() -> None:
    nbf.write(notebook(MODEL_SELECTION_CELLS), MODEL_SELECTION_PATH)
    nbf.write(notebook(VINDR_CELLS), VINDR_INTERPRETATION_PATH)
    nbf.write(notebook(ranking_closure_cells()), RANKING_CLOSURE_PATH)
    print(f"Wrote {MODEL_SELECTION_PATH}")
    print(f"Wrote {VINDR_INTERPRETATION_PATH}")
    print(f"Wrote {RANKING_CLOSURE_PATH}")


if __name__ == "__main__":
    main()
