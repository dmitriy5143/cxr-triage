from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
CORE_SOURCE = (ROOT / "fluoro_mvp_core.py").read_text(encoding="utf-8")
NOTEBOOK_PATH = ROOT / "fluoro_mvp_single_notebook.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # Fluoro / CXR MVP Research Notebook

        Safety-first research pipeline for chest screening triage:

        - `no_attention_required`
        - `requires_attention`
        - `N/A / manual_review`

        This notebook implements the first branch of two main experiments:

        - **EXP-1:** Google CXR Foundation full image + ROI embeddings, shallow calibrated heads.
        - **EXP-2:** EVA-X-S frozen encoder + head, with optional LoRA/adapters extension.

        Optional **EXP-3 CheXFound** is included as a scaffold/reference path and is off by default.

        Product note: this is not an automatic doctor. The notebook tests whether a model can safely auto-clear a controlled subset of studies while sending suspicious or uncertain cases to manual review.
        """
    ),
    md(
        """
        ## 0. Run Config

        Edit this cell first. It is the single control panel for dataset choice, model branches, sample size, and output location.

        Default profile for the first Colab machine:

        - IN-CXR only.
        - EXP-1 Google CXR Foundation frozen embeddings + calibrated heads.
        - EXP-2 EVA-X-S frozen encoder + calibrated heads.
        - LoRA is optional and disabled by default.
        - VinDr/VinBigData is disabled here and should be run in a separate runtime/profile.
        """
    ),
    code(
        """
        import os
        from pathlib import Path

        try:
            import google.colab  # noqa: F401
            IN_COLAB_BOOTSTRAP = True
        except Exception:
            IN_COLAB_BOOTSTRAP = False

        # Dataset profile.
        os.environ["DOWNLOAD_IN_CXR_FROM_KAGGLE"] = "1"
        os.environ["RUN_PRIMARY_TRACK"] = "1"
        os.environ["MAX_STUDIES"] = "5000"

        # Keep the second dataset off on this machine.
        os.environ["DOWNLOAD_VINDR_FROM_KAGGLE"] = "0"
        os.environ["RUN_VINDR_TRACK"] = "0"
        os.environ["RUN_VINDR_EXP2"] = "0"
        os.environ["VINDR_DOWNLOAD_MODE"] = "png512"
        os.environ["VINDR_DOWNLOAD_MAX_STUDIES"] = "5000"
        os.environ["MAX_VINDR_STUDIES"] = "5000"

        # Main model branches.
        os.environ["RUN_REAL_GOOGLE_CXR"] = "1"
        os.environ["RUN_REAL_EVA_X"] = "1"
        os.environ["RUN_EXP2_LORA"] = "0"
        os.environ["RUN_EXP3_CHEXFOUND"] = "0"

        # Download guard defaults. The helper cell only reads these values.
        os.environ["DOWNLOAD_VINDR_FULL_FROM_KAGGLE"] = "0"
        os.environ["VINDR_KEEP_ZIPS"] = "0"

        # Resource-sensitive options.
        os.environ["CXR_FOUNDATION_FULL_SIZE"] = "512"
        os.environ["EXP2_LORA_EPOCHS"] = "2"
        os.environ["EXP2_LORA_BATCH_SIZE"] = "4"
        os.environ["EXP2_SELECTED_THRESHOLD_POLICY"] = "zero_fn_cap_08pct"
        os.environ["EXP2_SELECTION_OBJECTIVE"] = "quality_first"
        os.environ["CACHE_PREPROCESSED_TO_DISK"] = "1"
        os.environ["PREPROCESSED_CACHE_DIR"] = "/content/fluoro_mvp_preprocessed_cache" if IN_COLAB_BOOTSTRAP else str(Path.cwd() / "fluoro_mvp_preprocessed_cache")

        # Persist outputs to Drive in Colab so a runtime reset does not erase results.
        if IN_COLAB_BOOTSTRAP:
            os.environ["MOUNT_DRIVE"] = "1"
            os.environ["PROJECT_DIR"] = "/content/drive/MyDrive/fluoro_mvp_runs/incxr_t4"
        else:
            os.environ.setdefault("MOUNT_DRIVE", "0")
            os.environ.setdefault("PROJECT_DIR", str(Path.cwd() / "fluoro_mvp_outputs"))

        # If data is already mounted, set these instead of downloading:
        # os.environ["IN_CXR_ROOT"] = "/content/path/to/incxr"
        # os.environ["IN_CXR_LABELS_CSV"] = "/content/path/to/labels.csv"
        # os.environ["VINDR_ROOT"] = "/content/path/to/vindr"

        print("Run profile configured:")
        for key in [
            "PROJECT_DIR",
            "MAX_STUDIES",
            "DOWNLOAD_IN_CXR_FROM_KAGGLE",
            "RUN_PRIMARY_TRACK",
            "DOWNLOAD_VINDR_FROM_KAGGLE",
            "RUN_VINDR_TRACK",
            "RUN_REAL_GOOGLE_CXR",
            "RUN_REAL_EVA_X",
            "RUN_EXP2_LORA",
            "RUN_EXP3_CHEXFOUND",
            "EXP2_SELECTED_THRESHOLD_POLICY",
            "EXP2_SELECTION_OBJECTIVE",
            "CACHE_PREPROCESSED_TO_DISK",
            "PREPROCESSED_CACHE_DIR",
        ]:
            print(f"  {key}={os.environ.get(key)}")
        """
    ),
    md(
        """
        ## 1. Colab Setup

        Run this notebook on a Google Colab GPU runtime. Tesla T4 16GB is enough for the default first-machine path because the heavy encoders are frozen and embeddings are cached.

        Default mode is **production IN-CXR mode**. It downloads the IN-CXR Kaggle mirror, uses up to **5000 IN-CXR studies**, preprocesses real images, trains the first-branch heads, calibrates the router, exports reports, and builds interpretation artifacts.

        The heavy preprocessed image cache is stored on local Colab disk through `PREPROCESSED_CACHE_DIR=/content/fluoro_mvp_preprocessed_cache`. Google Drive keeps compact reports, embeddings, checkpoints, and final artifacts; it is not used for tens of thousands of temporary image-cache files.

        Production run checklist:

        1. Add `KAGGLE_API_TOKEN` in Colab Secrets.
        2. Accept Hugging Face terms for Google CXR Foundation and add `HF_TOKEN` in Colab Secrets.
        3. Run cells from the top. The notebook downloads the bounded IN-CXR subset and sets `IN_CXR_ROOT` automatically.
        4. EVA-X-S weights are downloaded from the official Hugging Face checkpoint link used by the EVA-X repo when `RUN_REAL_EVA_X=1`.

        For the separate VinDr/VinBigData machine, also accept Kaggle terms for `vinbigdata-chest-xray-abnormalities-detection`.

        About tokens:

        - `HF_TOKEN` is used for gated Hugging Face models such as Google CXR Foundation after terms are accepted.
        - Kaggle can use a modern `KAGGLE_API_TOKEN` access token, or legacy `KAGGLE_USERNAME` + `KAGGLE_KEY` / Kaggle JSON.
        - Production defaults enable bounded Kaggle downloads.

        First production target: use **5000 IN-CXR studies** for screening behavior. VinDr/VinBigData is intentionally off in this first-machine profile.
        """
    ),
    code(
        """
        # Install only lightweight dependencies by default.
        # Heavy/optional model packages are installed later only when the corresponding real-model flag is enabled.
        import importlib.util
        import os
        import subprocess
        import sys
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
            ("cloudpickle", "cloudpickle>=2.2.0"),
        ]

        if os.environ.get("FLUORO_SKIP_INSTALLS", "0") != "1":
            for import_name, package in BASE_DEPS:
                if not has_module(import_name):
                    pip_install(package)
            ensure_distribution_min_version("ml-dtypes", "ml-dtypes>=0.5.0", "0.5.0")

        print("Setup cell finished.")
        """
    ),
    md(
        """
        ## 2. Tokens and Dataset Download Helpers

        This cell reads Colab Secrets, prepares Kaggle/Hugging Face authentication, and downloads only the datasets enabled in **Run Config**.

        Recommended Colab pattern:

        ```python
        import os
        from google.colab import userdata

        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or ""
        os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN") or ""
        ```

        Legacy Kaggle username/key JSON is also supported:

        ```python
        os.environ["KAGGLE_API_TOKEN"] = '{"username":"...","key":"..."}'
        ```

        Current first-machine defaults used by the notebook:

        ```python
        os.environ["DOWNLOAD_IN_CXR_FROM_KAGGLE"] = "1"
        os.environ["DOWNLOAD_VINDR_FROM_KAGGLE"] = "0"
        os.environ["MAX_STUDIES"] = "5000"
        os.environ["VINDR_DOWNLOAD_MAX_STUDIES"] = "5000"
        os.environ["MAX_VINDR_STUDIES"] = "5000"
        os.environ["VINDR_DOWNLOAD_MODE"] = "png512"
        os.environ["CXR_FOUNDATION_FULL_SIZE"] = "512"
        os.environ["RUN_REAL_GOOGLE_CXR"] = "1"
        os.environ["RUN_REAL_EVA_X"] = "1"
        os.environ["RUN_PRIMARY_TRACK"] = "1"
        os.environ["RUN_VINDR_TRACK"] = "0"
        os.environ["RUN_VINDR_EXP2"] = "0"
        os.environ["RUN_EXP2_LORA"] = "0"
        os.environ["EXP2_LORA_EPOCHS"] = "2"
        os.environ["EXP2_LORA_BATCH_SIZE"] = "4"
        os.environ["EXP2_SELECTED_THRESHOLD_POLICY"] = "zero_fn_cap_08pct"
        os.environ["EXP2_SELECTION_OBJECTIVE"] = "quality_first"
        os.environ["MOUNT_DRIVE"] = "1"
        os.environ["PROJECT_DIR"] = "/content/drive/MyDrive/fluoro_mvp_runs/incxr_t4"
        os.environ["PREPROCESSED_CACHE_DIR"] = "/content/fluoro_mvp_preprocessed_cache"
        os.environ["RUN_EXP3_CHEXFOUND"] = "0"
        ```

        The IN-CXR helper downloads the Kaggle PNG mirror and sets `IN_CXR_ROOT`. The VinDr helper stays idle in this profile; if enabled later, it downloads `train.csv`, samples image IDs, then downloads only selected PNG512 files plus original DICOM metadata needed to scale bboxes.

        Resource profile for a single Colab runtime:

        ```python
        # IN-CXR only: lighter first run
        os.environ["DOWNLOAD_VINDR_FROM_KAGGLE"] = "0"
        os.environ["RUN_VINDR_TRACK"] = "0"
        ```

        The default first-machine profile is IN-CXR only. Use a separate notebook copy/profile for the VinDr machine so the two runs do not compete for Colab disk/RAM.

        Note: the Kaggle IN-CXR mirror is preprocessed 224x224 PNG, not the official original DICOM release.
        """
    ),
    code(
        """
        import json
        import os
        import shutil
        import subprocess
        import sys
        import zipfile
        from pathlib import Path

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
                print("HF_TOKEN not set. Real gated HF models will not load until you set it.")

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
                # The Kaggle CLI treats KAGGLE_API_TOKEN as an access token first,
                # so remove JSON-shaped content after converting it to legacy env vars.
                if os.environ.get("KAGGLE_API_TOKEN", "").strip().startswith("{"):
                    os.environ.pop("KAGGLE_API_TOKEN", None)

            if username and key:
                os.environ["KAGGLE_USERNAME"] = username
                os.environ["KAGGLE_KEY"] = key
                kaggle_dir = Path.home() / ".kaggle"
                kaggle_dir.mkdir(exist_ok=True)
                kaggle_json = kaggle_dir / "kaggle.json"
                kaggle_json.write_text(json.dumps({
                    "username": os.environ["KAGGLE_USERNAME"],
                    "key": os.environ["KAGGLE_KEY"],
                }), encoding="utf-8")
                kaggle_json.chmod(0o600)
                print("Kaggle legacy credentials configured.")
                return True

            if token_value:
                os.environ["KAGGLE_API_TOKEN"] = token_value
                kaggle_dir = Path.home() / ".kaggle"
                kaggle_dir.mkdir(exist_ok=True)
                access_token_file = kaggle_dir / "access_token"
                access_token_file.write_text(token_value, encoding="utf-8")
                access_token_file.chmod(0o600)
                print("Kaggle access token configured from KAGGLE_API_TOKEN.")
                return True

            print("Kaggle credentials not set. Kaggle download helper will be skipped.")
            return False

        def kaggle_cli_executable() -> str:
            exe = shutil.which("kaggle")
            if exe is None:
                raise RuntimeError("Kaggle CLI executable was not found after installation.")
            return exe

        def maybe_download_kaggle_competition(
            competition: str,
            target_dir: str | Path,
            unzip: bool = True,
            env_root_var: str | None = None,
        ):
            if not setup_kaggle_token():
                raise RuntimeError("Set Kaggle credentials before downloading from Kaggle.")
            if not has_module("kaggle"):
                pip_install("kaggle")
            elif os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_USERNAME"):
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle"], check=True)
            kaggle_cli = kaggle_cli_executable()

            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [kaggle_cli, "competitions", "download", "-c", competition, "-p", str(target)],
                check=True,
            )

            if unzip:
                zip_paths = sorted(target.glob("*.zip"))
                if not zip_paths:
                    print("No zip archives found after Kaggle download.")
                for zip_path in zip_paths:
                    print("Unzipping:", zip_path.name)
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(target)

            sample_files = [p for p in target.rglob("*") if p.is_file()][:20]
            print("Kaggle data ready at:", target)
            for p in sample_files:
                print(" -", p.relative_to(target))

            if env_root_var:
                os.environ[env_root_var] = str(target)
                print(f"{env_root_var}={target}")
            return str(target)

        def download_kaggle_competition_file(competition: str, filename: str, target: Path, env: dict | None = None) -> None:
            kaggle_cli = kaggle_cli_executable()
            subprocess.run(
                [kaggle_cli, "competitions", "download", "-c", competition, "-f", filename, "-p", str(target)],
                check=True,
                env=env,
            )
            for zip_path in sorted(target.glob("*.zip")):
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(target)
                if os.environ.get("VINDR_KEEP_ZIPS", "0") != "1":
                    zip_path.unlink()

        def download_kaggle_dataset_file(dataset: str, filename: str | None, target: Path, env: dict | None = None) -> None:
            kaggle_cli = kaggle_cli_executable()
            cmd = [kaggle_cli, "datasets", "download", "-d", dataset, "-p", str(target)]
            if filename:
                cmd.extend(["-f", filename])
            subprocess.run(cmd, check=True, env=env)
            for zip_path in sorted(target.glob("*.zip")):
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(target)
                if os.environ.get("VINDR_KEEP_ZIPS", "0") != "1":
                    zip_path.unlink()

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

        def maybe_download_vindr_subset_from_kaggle(target_dir="/content/vindr_cxr", max_studies: int | None = None):
            if not setup_kaggle_token():
                raise RuntimeError("Set Kaggle credentials before downloading VinDr from Kaggle.")
            if not has_module("kaggle"):
                pip_install("kaggle")
            elif os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_USERNAME"):
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle"], check=True)

            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            competition = os.environ.get("VINDR_KAGGLE_COMPETITION", "vinbigdata-chest-xray-abnormalities-detection")
            max_studies = int(max_studies or os.environ.get("VINDR_DOWNLOAD_MAX_STUDIES", "5000"))
            normal_fraction = float(os.environ.get("VINDR_DOWNLOAD_NORMAL_FRACTION", "0.50"))
            seed = int(os.environ.get("VINDR_DOWNLOAD_SEED", "42"))

            train_csv = target / "train.csv"
            if not train_csv.exists():
                print("Downloading VinDr/VinBigData train.csv ...")
                download_kaggle_competition_file(competition, "train.csv", target, env=os.environ.copy())

            selected = sample_vindr_ids_from_train_csv(train_csv, max_studies=max_studies, normal_fraction=normal_fraction, seed=seed)
            manifest_path = target / "vindr_subset_manifest.csv"
            selected.to_csv(manifest_path, index=False)
            print(f"Selected {len(selected)} studies:", selected["y_attention"].value_counts().to_dict())
            print("Subset manifest:", manifest_path)

            for idx, row in selected.iterrows():
                image_id = str(row["image_id"])
                dicom_path = target / f"{image_id}.dicom"
                nested_dicom_path = target / "train" / f"{image_id}.dicom"
                if dicom_path.exists() or nested_dicom_path.exists():
                    continue
                if idx % 25 == 0:
                    print(f"Downloading DICOM {idx + 1}/{len(selected)} ...")
                download_kaggle_competition_file(competition, f"train/{image_id}.dicom", target, env=os.environ.copy())

            os.environ["VINDR_ROOT"] = str(target)
            print("VinDr subset ready at:", target)
            print("VINDR_ROOT=", target)
            return str(target)

        def maybe_download_vindr_png_subset_from_kaggle(target_dir="/content/vindr_cxr", max_studies: int | None = None):
            if not setup_kaggle_token():
                raise RuntimeError("Set Kaggle credentials before downloading VinDr from Kaggle.")
            if not has_module("kaggle"):
                pip_install("kaggle")
            elif os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_USERNAME"):
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle"], check=True)

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
            print(f"Selected {len(selected)} studies:", selected["y_attention"].value_counts().to_dict())
            print("Subset manifest:", manifest_path)

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
            print("VinDr PNG subset ready at:", target)
            print("VINDR_ROOT=", target)
            return str(target)

        def maybe_download_vindr_from_kaggle(target_dir="/content/vindr_cxr"):
            if os.environ.get("DOWNLOAD_VINDR_FROM_KAGGLE", "0") != "1":
                print("DOWNLOAD_VINDR_FROM_KAGGLE=0, skipping Kaggle download.")
                return None
            if os.environ.get("DOWNLOAD_VINDR_FULL_FROM_KAGGLE", "0") != "1":
                mode = os.environ.get("VINDR_DOWNLOAD_MODE", "png512").lower()
                if mode in {"png", "png512", "png_512", "512"}:
                    return maybe_download_vindr_png_subset_from_kaggle(target_dir=target_dir)
                if mode in {"png256", "png_256", "256"}:
                    os.environ.setdefault("VINDR_KAGGLE_PNG_DATASET", "xhlulu/vinbigdata-chest-xray-resized-png-256x256")
                    return maybe_download_vindr_png_subset_from_kaggle(target_dir=target_dir)
                if mode == "dicom":
                    return maybe_download_vindr_subset_from_kaggle(target_dir=target_dir)
                raise ValueError(f"Unknown VINDR_DOWNLOAD_MODE={mode!r}; use png512, png256, or dicom.")
            competition = os.environ.get("VINDR_KAGGLE_COMPETITION", "vinbigdata-chest-xray-abnormalities-detection")
            return maybe_download_kaggle_competition(
                competition=competition,
                target_dir=target_dir,
                unzip=True,
                env_root_var="VINDR_ROOT",
            )

        def maybe_download_incxr_from_kaggle(target_dir="/content/incxr_png"):
            if os.environ.get("DOWNLOAD_IN_CXR_FROM_KAGGLE", "0") != "1":
                print("DOWNLOAD_IN_CXR_FROM_KAGGLE=0, skipping IN-CXR Kaggle download.")
                return None
            if not setup_kaggle_token():
                raise RuntimeError("Set Kaggle credentials before downloading IN-CXR from Kaggle.")
            if not has_module("kaggle"):
                pip_install("kaggle")
            elif os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_USERNAME"):
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle"], check=True)

            target = Path(target_dir)
            target.mkdir(parents=True, exist_ok=True)
            dataset = os.environ.get("IN_CXR_KAGGLE_DATASET", "arjav007/in-cxr-dataset-png")
            marker = target / ".incxr_kaggle_download_complete"
            if not marker.exists():
                print("Downloading IN-CXR Kaggle PNG mirror. Expected compressed size is about 380 MB.")
                download_kaggle_dataset_file(dataset, None, target, env=os.environ.copy())
                marker.write_text(dataset, encoding="utf-8")
            else:
                print("IN-CXR Kaggle mirror already downloaded:", target)

            sample_files = [p for p in target.rglob("*.png")][:10]
            print("IN-CXR PNG files found:", len(list(target.rglob("*.png"))))
            for p in sample_files:
                print(" -", p.relative_to(target))
            os.environ["IN_CXR_ROOT"] = str(target)
            print("IN_CXR_ROOT=", target)
            return str(target)

        setup_hf_token()
        setup_kaggle_token()
        if os.environ.get("RUN_PRIMARY_TRACK", "1") == "1":
            maybe_download_incxr_from_kaggle(os.environ.get("IN_CXR_DOWNLOAD_DIR", "/content/incxr_png"))
        if os.environ.get("RUN_VINDR_TRACK", "0") == "1":
            maybe_download_vindr_from_kaggle(os.environ.get("VINDR_DOWNLOAD_DIR", "/content/vindr_cxr"))
        """
    ),
    md(
        """
        ## 3. Core Pipeline Code

        The next cell contains the self-contained implementation used by the notebook: dataset discovery, DICOM/PNG preprocessing, ROI preparation, feature extraction helpers, model heads, calibration, routing, metrics, LoRA utilities, and quantization helpers.
        """
    ),
    code(CORE_SOURCE),
    md(
        """
        ## 4. Runtime Validation and Project Paths

        This cell converts the top-level run profile into a typed `NotebookConfig`, mounts Drive when requested, validates that required datasets/tokens are available, and creates artifact/report directories.

        - Production defaults use real Kaggle dataset mirrors and real frozen encoders.
        - `MAX_STUDIES=5000` controls IN-CXR sample size.
        - `MAX_VINDR_STUDIES=5000` controls VinDr/VinBigData sample size.
        """
    ),
    code(
        """
        import os
        import warnings
        from pathlib import Path

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

        DEFAULT_PROJECT_DIR = os.environ.get("PROJECT_DIR")
        if not DEFAULT_PROJECT_DIR:
            DEFAULT_PROJECT_DIR = (
                "/content/drive/MyDrive/fluoro_mvp_runs/incxr_t4"
                if IN_COLAB and os.environ.get("MOUNT_DRIVE", "0") == "1"
                else ("/content/fluoro_mvp" if IN_COLAB else str(Path.cwd() / "fluoro_mvp_outputs"))
            )

        # --- Main switches ---
        DATA_ROOT = os.environ.get("IN_CXR_ROOT") or None
        LABELS_CSV = os.environ.get("IN_CXR_LABELS_CSV") or None
        VINDR_ROOT = os.environ.get("VINDR_ROOT") or None
        MAX_STUDIES = int(os.environ.get("MAX_STUDIES", "5000"))
        MAX_VINDR_STUDIES = int(os.environ.get("MAX_VINDR_STUDIES", os.environ.get("VINDR_DOWNLOAD_MAX_STUDIES", "5000")))
        CXR_FOUNDATION_FULL_SIZE = int(os.environ.get("CXR_FOUNDATION_FULL_SIZE", "512"))
        CACHE_PREPROCESSED_TO_DISK = os.environ.get("CACHE_PREPROCESSED_TO_DISK", "1") == "1"
        PREPROCESSED_CACHE_DIR = os.environ.get("PREPROCESSED_CACHE_DIR") or None

        RUN_REAL_GOOGLE_CXR = os.environ.get("RUN_REAL_GOOGLE_CXR", "1") == "1"
        RUN_REAL_EVA_X = os.environ.get("RUN_REAL_EVA_X", "1") == "1"
        RUN_EXP2_LORA = os.environ.get("RUN_EXP2_LORA", "0") == "1"
        RUN_PRIMARY_TRACK = os.environ.get("RUN_PRIMARY_TRACK", "1") == "1"
        RUN_VINDR_TRACK = os.environ.get("RUN_VINDR_TRACK", "0") == "1"
        RUN_VINDR_EXP2 = os.environ.get("RUN_VINDR_EXP2", "0") == "1"
        RUN_EXP3_CHEXFOUND = os.environ.get("RUN_EXP3_CHEXFOUND", "0") == "1"

        cfg = NotebookConfig(
            project_dir=DEFAULT_PROJECT_DIR,
            data_root=DATA_ROOT,
            labels_csv=LABELS_CSV,
            vindr_root=VINDR_ROOT,
            max_studies=MAX_STUDIES,
            max_vindr_studies=MAX_VINDR_STUDIES,
            random_state=42,
            target_npv=0.99,
            max_fn_per_1000=5.0,
            cxr_foundation_full_size=CXR_FOUNDATION_FULL_SIZE,
            eva_image_size=224,
            batch_size=8,
            run_real_google_cxr=RUN_REAL_GOOGLE_CXR,
            run_real_eva_x=RUN_REAL_EVA_X,
            run_exp2_lora=RUN_EXP2_LORA,
            run_primary_track=RUN_PRIMARY_TRACK,
            run_vindr_track=RUN_VINDR_TRACK,
            run_vindr_exp2=RUN_VINDR_EXP2,
            run_exp3_chexfound=RUN_EXP3_CHEXFOUND,
            pca_components=256,
            cache_preprocessed_to_disk=CACHE_PREPROCESSED_TO_DISK,
            preprocessed_cache_dir=PREPROCESSED_CACHE_DIR,
        )
        missing_inputs = []
        if cfg.run_primary_track and not cfg.data_root and not cfg.labels_csv:
            missing_inputs.append("IN_CXR_ROOT or IN_CXR_LABELS_CSV")
        if cfg.run_vindr_track and not cfg.vindr_root:
            missing_inputs.append("VINDR_ROOT")
        if cfg.run_real_google_cxr and not os.environ.get("HF_TOKEN"):
            missing_inputs.append("HF_TOKEN")
        if missing_inputs:
            raise RuntimeError(
                "Production run is missing: "
                + ", ".join(missing_inputs)
                + ". Run the token/download helper cell first, or override the corresponding flags."
            )
        if IN_COLAB and cfg.cache_preprocessed_to_disk and str(cfg.preprocessed_cache_dir).startswith("/content/drive"):
            raise RuntimeError(
                "PREPROCESSED_CACHE_DIR points to Google Drive. Use local Colab disk, for example "
                "os.environ['PREPROCESSED_CACHE_DIR']='/content/fluoro_mvp_preprocessed_cache'."
            )
        ensure_dirs(cfg)
        set_seed(cfg.random_state)
        HEAD_EPOCHS = 80
        if torch is not None:
            torch.set_num_threads(min(4, os.cpu_count() or 1))

        print(asdict(cfg))
        print("HEAD_EPOCHS:", HEAD_EPOCHS)
        """
    ),
    code(
        """
        print("Project directory:", cfg.project_dir)
        print("Preprocessed cache directory:", cfg.preprocessed_cache_dir)
        """
    ),
    md(
        """
        ## 5. Dataset Loading and Split

        For IN-CXR first branch:

        - `normal -> y_attention=0 -> no_attention_required`
        - `abnormal -> y_attention=1 -> requires_attention`
        - `N/A` is not a dataset label; it is produced later by the router.

        For the production run, the intended dataset size is **5000 IN-CXR studies**. This keeps the run inside free Colab limits while preserving enough signal for a meaningful first comparison.

        The loader accepts:

        - folder structure with `normal/` and `abnormal/`;
        - CSV with `path,y_attention`.
        """
    ),
    code(
        """
        if cfg.run_primary_track:
            df = discover_dataset(cfg.data_root, cfg.labels_csv, max_studies=cfg.max_studies, cfg=cfg)
        else:
            raise RuntimeError("RUN_PRIMARY_TRACK=0 is not supported in the production notebook. Run the IN-CXR track for model comparison.")
        df = make_splits(df, seed=cfg.random_state)

        index_path = save_table(df, Path(cfg.artifacts_dir) / "data_index")
        print(f"Dataset index saved to: {index_path}")
        display(df.head())
        display(df["y_attention"].value_counts().rename("count").to_frame())
        display(df["split"].value_counts().rename("count").to_frame())
        """
    ),
    md(
        """
        ## 6. Detailed Preprocessing

        Implemented according to the plan:

        - DICOM/PNG/JPEG read path, including the practical IN-CXR PNG mirror.
        - DICOM photometric interpretation, rescale/window handling when original DICOM is used.
        - Robust clipping and normalization.
        - Aspect-ratio-preserving resize/padding.
        - Quality checks and QA flags.
        - Thorax/lung ROI heuristic for EXP-1 full/ROI fusion.
        - Traceability metadata for analysis and backend export.
        - A visual audit sheet saved to artifacts for quick review before trusting metrics.
        - Disk-backed preprocessed image cache, so full/ROI/EVA inputs do not stay duplicated in RAM.
        """
    ),
    code(
        """
        results, meta = preprocess_dataframe(df, cfg)
        y = meta["y_attention"].values.astype(int)
        preproc_path = save_table(meta, Path(cfg.artifacts_dir) / "preprocessing_report")

        print(f"Preprocessing report saved to: {preproc_path}")
        print("Target vector:", y.shape, "positive_rate=", float(y.mean()))
        display(meta.head())
        display(meta[["quality_score", "roi_status", "critical_qa"]].describe(include="all"))
        display(meta["qa_flags"].value_counts().head(10).rename("count").to_frame())
        """
    ),
    code(
        """
        # Visual audit: original normalized preview, full model input, and ROI input.
        n_show = min(6, len(results))
        fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
        if n_show == 1:
            axes = np.asarray([axes])
        for i, r in enumerate(results[:n_show]):
            axes[i, 0].imshow(get_result_raw_preview(r), cmap="gray")
            axes[i, 0].set_title(f"raw norm | y={r.y_attention}")
            axes[i, 1].imshow(get_result_image_full(r), cmap="gray")
            axes[i, 1].set_title(f"full | q={r.quality_score:.2f}")
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
        print(f"Saved preview: {preview_path}")
        plt.show()
        """
    ),
    md(
        """
        ## 7. EXP-1: Google CXR Foundation Full/ROI Embeddings

        First branch variants:

        - **EXP-1-FIRST-A:** full + ROI embeddings, QA features, elastic-net logistic regression, calibration, N/A-router.
        - **EXP-1-FIRST-B:** same features, small bottleneck MLP + dropout, calibration, N/A-router. Use this only if logistic underfits.

        Diagnostic path:

        - **EXP-1-ABLATION:** full-only, ROI-only, full+ROI, full+ROI+QA. This is not a separate deployment model; it explains whether ROI/QA features really help.

        Real model note:

        Google CXR Foundation requires accepting Hugging Face terms. The notebook uses the official `clientside.clients.make_hugging_face_client('cxr_model')` path.
        """
    ),
    md(
        """
        ### 7.1 EXP-1 Embedding Extraction

        This cell loads Google CXR Foundation, computes full-image and ROI token embeddings, and caches them with a dataset fingerprint. Cached embeddings are reused only when the current dataset matches the saved fingerprint.
        """
    ),
    code(
        """
        def maybe_install_google_cxr_foundation():
            if not cfg.run_real_google_cxr:
                return
            ensure_distribution_min_version("ml-dtypes", "ml-dtypes>=0.5.0", "0.5.0")
            os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            if not has_module("clientside"):
                pip_install("git+https://github.com/Google-Health/cxr-foundation.git#subdirectory=python")
                ensure_distribution_min_version("ml-dtypes", "ml-dtypes>=0.5.0", "0.5.0")
            if os.environ.get("HF_TOKEN"):
                from huggingface_hub import login
                login(token=os.environ["HF_TOKEN"])

        def normalize_embedding_batch(arr, expected_n):
            arr = coerce_embedding_output(arr)
            if arr.ndim == 2 and expected_n == 1:
                return arr[None, :, :]
            if arr.ndim == 2 and expected_n > 1:
                return arr[:, None, :]
            if arr.ndim == 3:
                return arr
            if arr.ndim == 1:
                return arr[None, None, :]
            raise ValueError(f"Unexpected embedding shape {arr.shape}")

        _GOOGLE_CXR_CLIENT = None

        def get_google_cxr_client():
            global _GOOGLE_CXR_CLIENT
            if _GOOGLE_CXR_CLIENT is None:
                maybe_install_google_cxr_foundation()
                from clientside.clients import make_hugging_face_client
                _GOOGLE_CXR_CLIENT = make_hugging_face_client("cxr_model")
            return _GOOGLE_CXR_CLIENT

        def embed_images_exp1(images, cfg, seed=13):
            if not cfg.run_real_google_cxr:
                raise RuntimeError("RUN_REAL_GOOGLE_CXR=0 disables EXP-1 embeddings. Enable it for the production notebook.")
            cxr_client = get_google_cxr_client()
            out = cxr_client.get_image_embeddings_from_images(images)
            return normalize_embedding_batch(out, len(images)).astype(np.float32)

        def extract_exp1_embeddings(results, cfg, prefix="exp1_google"):
            full_path = Path(cfg.artifacts_dir) / "embeddings" / f"{prefix}_full_tokens.npy"
            roi_path = Path(cfg.artifacts_dir) / "embeddings" / f"{prefix}_roi_tokens.npy"
            fingerprint_path = Path(cfg.artifacts_dir) / "embeddings" / f"{prefix}_fingerprint.json"
            fingerprint = stable_hash(
                "|".join(f"{r.study_id}:{r.source_path}:{r.y_attention}" for r in results),
                n=16,
            )
            if full_path.exists() and roi_path.exists() and fingerprint_path.exists():
                cached_full = np.load(full_path)
                cached_roi = np.load(roi_path)
                cached_info = json.loads(fingerprint_path.read_text(encoding="utf-8"))
                cache_ok = (
                    cached_full.shape[0] == len(results)
                    and cached_roi.shape[0] == len(results)
                    and cached_info.get("fingerprint") == fingerprint
                )
                if cache_ok:
                    return cached_full, cached_roi
                print("Cached EXP-1 embeddings do not match the current dataset; recomputing.")

            full_batches, roi_batches = [], []
            for start in range(0, len(results), cfg.batch_size):
                batch = results[start : start + cfg.batch_size]
                full_imgs = [get_result_image_full(r) for r in batch]
                roi_imgs = [get_result_image_roi(r) or get_result_image_full(r) for r in batch]
                full_batches.append(embed_images_exp1(full_imgs, cfg, seed=13))
                roi_batches.append(embed_images_exp1(roi_imgs, cfg, seed=17))
            full_tokens = np.concatenate(full_batches, axis=0).astype(np.float32)
            roi_tokens = np.concatenate(roi_batches, axis=0).astype(np.float32)

            np.save(full_path, full_tokens)
            np.save(roi_path, roi_tokens)
            fingerprint_path.write_text(
                json.dumps({"fingerprint": fingerprint, "n": len(results)}, indent=2),
                encoding="utf-8",
            )
            print(f"Saved EXP-1 embeddings: {full_path}, {roi_path}")
            return full_tokens, roi_tokens

        full_tokens, roi_tokens = extract_exp1_embeddings(results, cfg, prefix="incxr_exp1_google")
        print("EXP-1 full tokens:", full_tokens.shape)
        print("EXP-1 ROI tokens:", roi_tokens.shape)

        # EXP-1 embeddings are now cached, so the TensorFlow client is no longer needed in RAM/VRAM.
        _GOOGLE_CXR_CLIENT = None
        try:
            import gc
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception as exc:
            print("Google CXR cleanup skipped:", exc)
        """
    ),
    md(
        """
        ### 7.2 EXP-1-FIRST-A: Calibrated Logistic Head

        This is the primary lightweight head for high-dimensional Google CXR Foundation embeddings. It fuses full image, ROI, and QA features, then calibrates risk scores and builds the first threshold/router report.
        """
    ),
    code(
        """
        # Build pooled full+ROI+QA feature matrix.
        X_exp1, exp1_feature_names = build_fusion_features(full_tokens, roi_tokens, meta)
        y = meta["y_attention"].values.astype(int)
        parts_exp1 = split_arrays_by_meta(X_exp1, y, meta)

        # EXP-1-FIRST-A: elastic-net logistic regression.
        exp1a = train_logistic_classifier(
            parts_exp1["train"][0],
            parts_exp1["train"][1],
            parts_exp1["calibration"][0],
            parts_exp1["calibration"][1],
            pca_components=cfg.pca_components,
            seed=cfg.random_state,
        )
        p_exp1a_val = predict_proba_any(exp1a, parts_exp1["validation"][0], device=cfg.device)[:, 1]
        exp1a_metrics, exp1a_thr, exp1a_routes = make_model_report(
            "EXP-1-FIRST-A logistic",
            parts_exp1["validation"][1],
            p_exp1a_val,
            meta.iloc[parts_exp1["validation"][2]].reset_index(drop=True),
            cfg.target_npv,
        )

        print("EXP-1-FIRST-A metrics")
        display(pd.DataFrame([exp1a_metrics]))
        display(exp1a_thr.head(10))
        display(exp1a_routes.head())

        save_pickle(exp1a, Path(cfg.artifacts_dir) / "models" / "exp1a_logistic_calibrated.pkl")
        exp1a_thr.to_csv(Path(cfg.reports_dir) / "exp1a_threshold_report.csv", index=False)
        exp1a_routes.to_csv(Path(cfg.reports_dir) / "exp1a_routes_validation.csv", index=False)
        """
    ),
    md(
        """
        ### 7.3 EXP-1 Diagnostic Ablation

        This diagnostic checks whether full-image embeddings, ROI embeddings, and QA metadata contribute useful signal. It is for interpretation and model selection sanity, not a separate deployment candidate.
        """
    ),
    code(
        """
        # EXP-1 stream ablation for interpretation: full-only, ROI-only, full+ROI, full+ROI+QA.
        full_pool = pool_tokens(full_tokens)
        roi_pool = pool_tokens(roi_tokens)
        qa_X, qa_names = qa_feature_matrix(meta)

        ablation_sets = {
            "full_only": full_pool,
            "roi_only": roi_pool,
            "full_roi": np.concatenate([full_pool, roi_pool], axis=1),
            "full_roi_qa": X_exp1,
        }
        ablation_rows = []
        for name, X_ab in ablation_sets.items():
            parts = split_arrays_by_meta(X_ab.astype(np.float32), y, meta)
            model = train_logistic_classifier(
                parts["train"][0], parts["train"][1],
                parts["calibration"][0], parts["calibration"][1],
                pca_components=min(128, max(1, parts["train"][0].shape[0] - 1)),
                seed=cfg.random_state,
            )
            p_val = predict_proba_any(model, parts["validation"][0])[:, 1]
            row = metrics_summary(parts["validation"][1], p_val)
            row["feature_set"] = name
            ablation_rows.append(row)

        exp1_ablation = pd.DataFrame(ablation_rows).sort_values("auroc", ascending=False)
        exp1_ablation.to_csv(Path(cfg.reports_dir) / "exp1_stream_ablation.csv", index=False)
        display(exp1_ablation)
        """
    ),
    md(
        """
        ### 7.4 EXP-1-FIRST-B: Bottleneck MLP Head

        This trains a small regularized neural head on the same frozen EXP-1 features. It is useful when the logistic head underfits, while still avoiding encoder fine-tuning.
        """
    ),
    code(
        """
        # EXP-1-FIRST-B: small bottleneck MLP on the same features.
        hidden = 128 if len(X_exp1) < 200 else 256
        exp1b = train_torch_mlp(
            parts_exp1["train"][0],
            parts_exp1["train"][1],
            parts_exp1["calibration"][0],
            parts_exp1["calibration"][1],
            hidden=hidden,
            epochs=HEAD_EPOCHS,
            lr=1e-3,
            weight_decay=1e-4,
            seed=cfg.random_state,
            device=cfg.device,
        )

        raw_calib = predict_torch_mlp(exp1b, parts_exp1["calibration"][0], device=cfg.device)
        exp1b_platt = PlattCalibrator().fit(raw_calib, parts_exp1["calibration"][1])
        p_exp1b_val_raw = predict_torch_mlp(exp1b, parts_exp1["validation"][0], device=cfg.device)
        p_exp1b_val = exp1b_platt.transform(p_exp1b_val_raw)

        exp1b_metrics, exp1b_thr, exp1b_routes = make_model_report(
            "EXP-1-FIRST-B MLP",
            parts_exp1["validation"][1],
            p_exp1b_val,
            meta.iloc[parts_exp1["validation"][2]].reset_index(drop=True),
            cfg.target_npv,
        )
        print("EXP-1-FIRST-B metrics")
        display(pd.DataFrame([exp1b_metrics]))
        display(exp1b_thr.head(10))

        torch.save({"state_dict": exp1b.state_dict(), "scaler": exp1b.scaler}, Path(cfg.checkpoints_dir) / "exp1b_mlp.pt")
        save_pickle(exp1b_platt, Path(cfg.artifacts_dir) / "calibration" / "exp1b_platt.pkl")
        exp1b_thr.to_csv(Path(cfg.reports_dir) / "exp1b_threshold_report.csv", index=False)
        """
    ),
    md(
        """
        ## 8. EXP-2: EVA-X-S Frozen Encoder + Head

        First branch variants:

        - **EXP-2-FIRST-A1:** frozen EVA-X-S features + calibrated logistic/linear head.
        - **EXP-2-FIRST-A2:** frozen EVA-X-S features + tuned MLP head and conservative threshold policy search.
        - **EXP-2-FIRST-B:** optional LoRA/adapters on the last blocks, only if real EVA-X-S loads and memory is stable.

        Production mode loads the real EVA-X-S encoder and caches frozen features.
        """
    ),
    md(
        """
        ### 8.1 EXP-2 Feature Extraction

        This cell loads EVA-X-S, freezes the encoder, extracts one feature vector per study, and stores the feature matrix. The encoder is not fine-tuned in the default first run.
        """
    ),
    code(
        """
        if cfg.run_real_eva_x:
            eva_model = load_real_eva_x_s(cfg.project_dir, device=cfg.device)
            X_eva = extract_eva_features_real(
                eva_model,
                results,
                image_size=cfg.eva_image_size,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )
        else:
            raise RuntimeError("RUN_REAL_EVA_X=0 disables EXP-2. Enable it for the production notebook.")

        np.save(Path(cfg.artifacts_dir) / "embeddings" / "exp2_eva_features.npy", X_eva)
        print("EXP-2 EVA feature matrix:", X_eva.shape)
        y = meta["y_attention"].values.astype(int)
        parts_eva = split_arrays_by_meta(X_eva, y, meta)
        """
    ),
    md(
        """
        ### 8.2 EXP-2-FIRST-A1: Frozen EVA-X-S Linear Head

        This is the strongest low-risk EVA-X-S baseline: frozen encoder features plus a calibrated logistic head.
        """
    ),
    code(
        """
        # EXP-2-FIRST-A1: frozen EVA-X-S features + linear/logistic head.
        exp2a1 = train_logistic_classifier(
            parts_eva["train"][0],
            parts_eva["train"][1],
            parts_eva["calibration"][0],
            parts_eva["calibration"][1],
            pca_components=min(128, max(1, parts_eva["train"][0].shape[0] - 1)),
            seed=cfg.random_state,
        )
        p_exp2a1_val = predict_proba_any(exp2a1, parts_eva["validation"][0])[:, 1]
        exp2a1_metrics, exp2a1_thr, exp2a1_routes = make_model_report(
            "EXP-2-FIRST-A1 EVA frozen logistic",
            parts_eva["validation"][1],
            p_exp2a1_val,
            meta.iloc[parts_eva["validation"][2]].reset_index(drop=True),
            cfg.target_npv,
        )
        display(pd.DataFrame([exp2a1_metrics]))
        display(exp2a1_thr.head(10))
        save_pickle(exp2a1, Path(cfg.artifacts_dir) / "models" / "exp2a1_eva_logistic.pkl")
        """
    ),
    md(
        """
        ### 8.3 EXP-2-FIRST-A2: EVA MLP Head and Threshold Tuning

        This is the main EVA-X-S head-selection block. EVA-X-S features are already cached, so the expensive encoder is not rerun. The cell retrains several MLP heads, calibrates each one, evaluates AUROC/AUPRC/calibration, and tests multiple `T_negative` policies for the `no_attention_required` router.

        The default selected threshold policy is conservative: it requires zero false negatives on validation and caps validation auto-negative coverage. This trades some automation volume for a safer MVP router. `EXP2_SELECTION_OBJECTIVE=quality_first` keeps the highest-ranking model by AUROC/AUPRC; `coverage_first` instead prefers the largest safe auto-clear volume among candidates that pass the selected safety policy. The full tuning table is saved, and the selected head is registered for final-test, interpretation, quantization, manifest, and archive export.
        """
    ),
    code(
        """
        # EVA-X-S MLP tuning grid. Keep this modest: feature extraction is expensive, but head tuning is cheap.
        # You can add/remove rows after the first full-dataset run.
        EXP2_SELECTED_THRESHOLD_POLICY = os.environ.get("EXP2_SELECTED_THRESHOLD_POLICY", "zero_fn_cap_08pct")
        EXP2_SELECTION_OBJECTIVE = os.environ.get("EXP2_SELECTION_OBJECTIVE", "quality_first").lower()
        EXP2_MLP_SEARCH_SPACE = [
            {"name": "baseline_h128", "hidden": 128, "dropout": 0.20, "epochs": 80, "lr": 1e-3, "weight_decay": 1e-4, "seed": cfg.random_state},
            {"name": "strong_h256", "hidden": 256, "dropout": 0.20, "epochs": 120, "lr": 8e-4, "weight_decay": 5e-5, "seed": cfg.random_state},
            {"name": "regularized_h256", "hidden": 256, "dropout": 0.30, "epochs": 140, "lr": 8e-4, "weight_decay": 1e-4, "seed": cfg.random_state + 1},
            {"name": "wide_h384", "hidden": 384, "dropout": 0.25, "epochs": 140, "lr": 5e-4, "weight_decay": 5e-5, "seed": cfg.random_state + 2},
            {"name": "compact_low_wd", "hidden": 192, "dropout": 0.15, "epochs": 120, "lr": 1e-3, "weight_decay": 1e-5, "seed": cfg.random_state + 3},
            {"name": "conservative_h256", "hidden": 256, "dropout": 0.35, "epochs": 160, "lr": 5e-4, "weight_decay": 2e-4, "seed": cfg.random_state + 4},
            {"name": "wide_h512", "hidden": 512, "dropout": 0.30, "epochs": 160, "lr": 3e-4, "weight_decay": 1e-4, "seed": cfg.random_state + 5},
            {"name": "strong_h256_seed2", "hidden": 256, "dropout": 0.20, "epochs": 140, "lr": 8e-4, "weight_decay": 5e-5, "seed": cfg.random_state + 6},
        ]

        THRESHOLD_POLICIES = [
            {"name": "target_npv_max_coverage", "require_zero_fn": False, "coverage_cap": 1.00, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95},
            {"name": "zero_fn_max_coverage", "require_zero_fn": True, "coverage_cap": 1.00, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95},
            {"name": "zero_fn_cap_10pct", "require_zero_fn": True, "coverage_cap": 0.10, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95},
            {"name": "zero_fn_cap_08pct", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95},
            {"name": "zero_fn_cap_08pct_ood90", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.90},
            {"name": "zero_fn_cap_05pct", "require_zero_fn": True, "coverage_cap": 0.05, "min_selected": 5, "min_npv_ci95_low": None, "t_ood": 0.95},
            {"name": "ci_guard_cap_10pct", "require_zero_fn": True, "coverage_cap": 0.10, "min_selected": 20, "min_npv_ci95_low": 0.95, "t_ood": 0.95},
        ]

        def select_threshold_by_policy(thr_report: pd.DataFrame, policy: dict) -> pd.Series | None:
            if thr_report.empty:
                return None
            candidates = thr_report.copy()
            candidates = candidates[candidates["NPV"] >= cfg.target_npv]
            candidates = candidates[candidates["selected_count"] >= int(policy.get("min_selected", 1))]
            candidates = candidates[candidates["no_attention_required_coverage"] <= float(policy.get("coverage_cap", 1.0)) + 1e-12]
            if policy.get("require_zero_fn", False):
                candidates = candidates[candidates["FN_count"] == 0]
            min_ci = policy.get("min_npv_ci95_low")
            if min_ci is not None:
                candidates = candidates[candidates["NPV_ci95_low"] >= float(min_ci)]
            if candidates.empty:
                return None
            candidates = candidates.sort_values(
                ["no_attention_required_coverage", "NPV_ci95_low", "T_negative"],
                ascending=[False, False, False],
            )
            return candidates.iloc[0]

        def sort_selected_candidate_rows(rows: pd.DataFrame, objective: str | None = None) -> pd.DataFrame:
            objective = (objective or EXP2_SELECTION_OBJECTIVE or "quality_first").lower()
            rows = rows.copy()
            if objective in {"coverage_first", "safe_coverage_first", "auto_clear_first"}:
                return rows.sort_values(
                    ["auto_negative_coverage", "NPV_ci95_low", "auroc", "auprc", "ece"],
                    ascending=[False, False, False, False, True],
                    na_position="last",
                )
            if objective in {"balanced", "balanced_safe"}:
                rows["_balanced_score"] = (
                    rows["auto_negative_coverage"].fillna(0.0) * 2.0
                    + rows["auroc"].fillna(0.0)
                    + rows["auprc"].fillna(0.0)
                    + rows["NPV_ci95_low"].fillna(0.0) * 0.25
                    - rows["ece"].fillna(0.0) * 0.25
                )
                return rows.sort_values(
                    ["_balanced_score", "auto_negative_coverage", "auroc", "auprc"],
                    ascending=[False, False, False, False],
                    na_position="last",
                )
            return rows.sort_values(
                ["auroc", "auprc", "auto_negative_coverage", "NPV_ci95_low", "ece"],
                ascending=[False, False, False, False, True],
                na_position="last",
            )

        tuning_records = []
        exp2_mlp_tuned_models = {}
        exp2_mlp_tuned_calibrators = {}
        exp2_mlp_tuned_predictions = {}
        exp2_mlp_tuned_threshold_reports = {}
        exp2_mlp_tuned_routes = {}

        val_meta_eva = meta.iloc[parts_eva["validation"][2]].reset_index(drop=True)

        for hp in EXP2_MLP_SEARCH_SPACE:
            print("Training EVA MLP head:", hp)
            model = train_torch_mlp(
                parts_eva["train"][0],
                parts_eva["train"][1],
                parts_eva["calibration"][0],
                parts_eva["calibration"][1],
                hidden=int(hp["hidden"]),
                dropout=float(hp["dropout"]),
                epochs=int(hp["epochs"]),
                lr=float(hp["lr"]),
                weight_decay=float(hp["weight_decay"]),
                seed=int(hp["seed"]),
                device=cfg.device,
            )
            raw_calib = predict_torch_mlp(model, parts_eva["calibration"][0], device=cfg.device)
            calibrator = PlattCalibrator().fit(raw_calib, parts_eva["calibration"][1])
            p_raw_val = predict_torch_mlp(model, parts_eva["validation"][0], device=cfg.device)
            p_val = calibrator.transform(p_raw_val)

            base_metrics = metrics_summary(parts_eva["validation"][1], p_val)
            thr_report = threshold_report(parts_eva["validation"][1], p_val, target_npv=cfg.target_npv)

            key = str(hp["name"])
            exp2_mlp_tuned_models[key] = model
            exp2_mlp_tuned_calibrators[key] = calibrator
            exp2_mlp_tuned_predictions[key] = p_val
            exp2_mlp_tuned_threshold_reports[key] = thr_report

            for policy in THRESHOLD_POLICIES:
                selected = select_threshold_by_policy(thr_report, policy)
                if selected is None:
                    record = {**base_metrics, **hp}
                    record.update({
                        "head_name": key,
                        "threshold_policy": policy["name"],
                        "threshold_policy_selected": False,
                        "selected_T_negative": np.nan,
                        "auto_negative_coverage": 0.0,
                        "N/A_rate": np.nan,
                        "requires_attention_rate": np.nan,
                        "auto_negative_NPV": np.nan,
                        "unsafe_FN_auto_negative": np.nan,
                        "unsafe_FN_per_1000_auto_negative": np.nan,
                        "NPV_ci95_low": np.nan,
                        "selected_t_ood": float(policy.get("t_ood", 0.95)),
                    })
                    tuning_records.append(record)
                    continue

                t_neg = float(selected["T_negative"])
                routes = route_decisions(p_val, val_meta_eva, t_negative=t_neg, t_ood=float(policy.get("t_ood", 0.95)))
                r_metrics = route_metrics(parts_eva["validation"][1], routes)
                record = {**base_metrics, **hp, **r_metrics}
                record.update({
                    "head_name": key,
                    "threshold_policy": policy["name"],
                    "threshold_policy_selected": True,
                    "selected_T_negative": t_neg,
                    "threshold_selected_count": int(selected["selected_count"]),
                    "threshold_validation_NPV": float(selected["NPV"]),
                    "threshold_validation_FN_count": int(selected["FN_count"]),
                    "NPV_ci95_low": float(selected["NPV_ci95_low"]),
                    "selected_t_ood": float(policy.get("t_ood", 0.95)),
                    "model": "EXP-2-FIRST-A2-TUNED EVA frozen MLP",
                })
                tuning_records.append(record)
                if policy["name"] == EXP2_SELECTED_THRESHOLD_POLICY:
                    exp2_mlp_tuned_routes[key] = routes

        exp2_mlp_tuning_results = pd.DataFrame(tuning_records)
        exp2_mlp_tuning_results = exp2_mlp_tuning_results.sort_values(
            ["threshold_policy_selected", "threshold_policy", "auroc", "auprc", "auto_negative_coverage"],
            ascending=[False, True, False, False, False],
        )
        exp2_mlp_tuning_results.to_csv(Path(cfg.reports_dir) / "exp2_mlp_tuning_results.csv", index=False)
        export_json(
            {
                "search_space": EXP2_MLP_SEARCH_SPACE,
                "threshold_policies": THRESHOLD_POLICIES,
                "selected_threshold_policy": EXP2_SELECTED_THRESHOLD_POLICY,
                "selection_objective": EXP2_SELECTION_OBJECTIVE,
            },
            Path(cfg.artifacts_dir) / "models" / "exp2_mlp_tuning_config.json",
        )
        display(exp2_mlp_tuning_results.head(30))

        selected_policy_rows = exp2_mlp_tuning_results[
            (exp2_mlp_tuning_results["threshold_policy"] == EXP2_SELECTED_THRESHOLD_POLICY)
            & (exp2_mlp_tuning_results["threshold_policy_selected"])
        ].copy()
        if selected_policy_rows.empty:
            print(f"No rows satisfied {EXP2_SELECTED_THRESHOLD_POLICY}; falling back to any selected threshold policy.")
            selected_policy_rows = exp2_mlp_tuning_results[exp2_mlp_tuning_results["threshold_policy_selected"]].copy()
        if selected_policy_rows.empty:
            raise RuntimeError("No valid EVA MLP threshold policy was found. Lower min_selected or coverage constraints.")

        selected_policy_rows = sort_selected_candidate_rows(selected_policy_rows, EXP2_SELECTION_OBJECTIVE)
        best_tuned_row = selected_policy_rows.iloc[0].to_dict()
        exp2a2_tuned_name = str(best_tuned_row["head_name"])
        exp2a2_tuned = exp2_mlp_tuned_models[exp2a2_tuned_name]
        exp2a2_tuned_platt = exp2_mlp_tuned_calibrators[exp2a2_tuned_name]
        p_exp2a2_tuned_val = exp2_mlp_tuned_predictions[exp2a2_tuned_name]
        exp2a2_tuned_thr = exp2_mlp_tuned_threshold_reports[exp2a2_tuned_name]
        exp2a2_tuned_routes = route_decisions(
            p_exp2a2_tuned_val,
            val_meta_eva,
            t_negative=float(best_tuned_row["selected_T_negative"]),
            t_ood=float(best_tuned_row.get("selected_t_ood", 0.95)),
        )

        exp2a2_tuned_metrics = metrics_summary(parts_eva["validation"][1], p_exp2a2_tuned_val)
        exp2a2_tuned_metrics.update({
            "selected_T_negative": float(best_tuned_row["selected_T_negative"]),
            "auto_negative_coverage": float(best_tuned_row["auto_negative_coverage"]),
            "N/A_rate": float(best_tuned_row["N/A_rate"]),
            "requires_attention_rate": float(best_tuned_row["requires_attention_rate"]),
            "auto_negative_NPV": float(best_tuned_row["auto_negative_NPV"]),
            "unsafe_FN_auto_negative": float(best_tuned_row["unsafe_FN_auto_negative"]),
            "unsafe_FN_per_1000_auto_negative": float(best_tuned_row["unsafe_FN_per_1000_auto_negative"]),
            "threshold_policy": str(best_tuned_row["threshold_policy"]),
            "selected_t_ood": float(best_tuned_row.get("selected_t_ood", 0.95)),
            "head_name": exp2a2_tuned_name,
            "model": "EXP-2-FIRST-A2-TUNED EVA frozen MLP",
        })

        print("Selected tuned EVA MLP:", exp2a2_tuned_name)
        print("Selected threshold policy:", exp2a2_tuned_metrics["threshold_policy"])
        print("Selection objective:", EXP2_SELECTION_OBJECTIVE)
        display(pd.DataFrame([exp2a2_tuned_metrics]))
        display(exp2a2_tuned_thr.head(15))

        torch.save({"state_dict": exp2a2_tuned.state_dict(), "scaler": exp2a2_tuned.scaler}, Path(cfg.checkpoints_dir) / "exp2a2_tuned_eva_mlp.pt")
        save_pickle(exp2a2_tuned_platt, Path(cfg.artifacts_dir) / "calibration" / "exp2a2_tuned_platt.pkl")
        exp2a2_tuned_thr.to_csv(Path(cfg.reports_dir) / "exp2a2_tuned_threshold_report.csv", index=False)
        exp2a2_tuned_routes.to_csv(Path(cfg.reports_dir) / "exp2a2_tuned_routes_validation.csv", index=False)
        """
    ),
    md(
        """
        ### 8.4 EXP-2-FIRST-B: Optional EVA-X-S LoRA

        LoRA is off by default. If enabled, it trains adapters only on the train split, saves a checkpoint after each epoch, calibrates on the calibration split, then runs the same threshold-policy search as the MLP branch. That means a 20- or 40-epoch LoRA run is automatically compared against the tuned MLP with the same safety rules.
        """
    ),
    code(
        """
        # EXP-2-FIRST-B: optional LoRA/adapters controlled extension.
        # This cell defines a real end-to-end training path but does not run unless both flags are true.
        class EVAEndToEndClassifier(nn.Module):
            def __init__(self, encoder, feature_dim: int):
                super().__init__()
                self.encoder = encoder
                self.head = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, 128),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1),
                )

            def encode(self, x):
                if hasattr(self.encoder, "forward_features"):
                    z = self.encoder.forward_features(x)
                    if z.ndim == 3:
                        z = z[:, 1:, :].mean(dim=1) if z.shape[1] > 1 else z.mean(dim=1)
                else:
                    z = self.encoder(x)
                return z

            def forward(self, x):
                return self.head(self.encode(x)).squeeze(-1)

        def train_exp2_lora_end_to_end(eva_model, results, labels, cfg, train_indices, epochs=2):
            replaced = inject_lora_last_blocks(eva_model, n_last_blocks=2, r=4, alpha=8.0)
            print(f"LoRA Linear modules replaced: {replaced}")
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
            print("EVA LoRA params:", count_parameters(model))
            y_tensor = torch.tensor(labels.astype(np.float32), device=cfg.device)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)
            scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))
            lora_batch_size = int(os.environ.get("EXP2_LORA_BATCH_SIZE", "4"))
            checkpoint_dir = Path(cfg.checkpoints_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.train()
            for epoch in range(epochs):
                order = np.random.permutation(np.asarray(train_indices, dtype=int))
                losses = []
                for start in range(0, len(order), lora_batch_size):
                    idx = order[start:start + lora_batch_size]
                    xb = torch.stack([image_to_eva_tensor(get_result_image_eva(results[int(i)]), cfg.eva_image_size) for i in idx]).to(cfg.device)
                    yb = y_tensor[idx]
                    opt.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
                        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    losses.append(float(loss.detach().cpu()))
                epoch_loss = float(np.mean(losses)) if losses else float("nan")
                epoch_path = checkpoint_dir / f"exp2b_eva_lora_epoch_{epoch + 1}.pt"
                torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(), "loss": epoch_loss}, epoch_path)
                print(f"LoRA epoch {epoch+1}: loss={epoch_loss:.4f}; checkpoint={epoch_path}")
            return model

        def predict_lora_end_to_end(model, results, indices, cfg):
            lora_batch_size = int(os.environ.get("EXP2_LORA_BATCH_SIZE", "4"))
            preds = []
            model.eval()
            with torch.no_grad():
                for start in range(0, len(indices), lora_batch_size):
                    batch_idx = indices[start:start + lora_batch_size]
                    xb = torch.stack([image_to_eva_tensor(get_result_image_eva(results[int(i)]), cfg.eva_image_size) for i in batch_idx]).to(cfg.device)
                    with torch.cuda.amp.autocast(enabled=(cfg.device == "cuda")):
                        logits = model(xb)
                    preds.append(torch.sigmoid(logits).detach().float().cpu().numpy())
            return np.concatenate(preds, axis=0).astype(np.float32)

        exp2_lora_model = None
        exp2_lora_metrics = None
        if "EXP2_SELECTION_OBJECTIVE" not in globals():
            EXP2_SELECTION_OBJECTIVE = os.environ.get("EXP2_SELECTION_OBJECTIVE", "quality_first").lower()
        if "THRESHOLD_POLICIES" not in globals():
            THRESHOLD_POLICIES = [
                {"name": "target_npv_max_coverage", "require_zero_fn": False, "coverage_cap": 1.00, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95},
                {"name": "zero_fn_max_coverage", "require_zero_fn": True, "coverage_cap": 1.00, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95},
                {"name": "zero_fn_cap_10pct", "require_zero_fn": True, "coverage_cap": 0.10, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95},
                {"name": "zero_fn_cap_08pct", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.95},
                {"name": "zero_fn_cap_08pct_ood90", "require_zero_fn": True, "coverage_cap": 0.08, "min_selected": 10, "min_npv_ci95_low": None, "t_ood": 0.90},
                {"name": "zero_fn_cap_05pct", "require_zero_fn": True, "coverage_cap": 0.05, "min_selected": 5, "min_npv_ci95_low": None, "t_ood": 0.95},
                {"name": "ci_guard_cap_10pct", "require_zero_fn": True, "coverage_cap": 0.10, "min_selected": 20, "min_npv_ci95_low": 0.95, "t_ood": 0.95},
            ]
        if "EXP2_SELECTED_THRESHOLD_POLICY" not in globals():
            EXP2_SELECTED_THRESHOLD_POLICY = os.environ.get("EXP2_SELECTED_THRESHOLD_POLICY", "zero_fn_cap_08pct")
        if "select_threshold_by_policy" not in globals():
            def select_threshold_by_policy(thr_report: pd.DataFrame, policy: dict) -> pd.Series | None:
                if thr_report.empty:
                    return None
                candidates = thr_report.copy()
                candidates = candidates[candidates["NPV"] >= cfg.target_npv]
                candidates = candidates[candidates["selected_count"] >= int(policy.get("min_selected", 1))]
                candidates = candidates[candidates["no_attention_required_coverage"] <= float(policy.get("coverage_cap", 1.0)) + 1e-12]
                if policy.get("require_zero_fn", False):
                    candidates = candidates[candidates["FN_count"] == 0]
                min_ci = policy.get("min_npv_ci95_low")
                if min_ci is not None:
                    candidates = candidates[candidates["NPV_ci95_low"] >= float(min_ci)]
                if candidates.empty:
                    return None
                candidates = candidates.sort_values(
                    ["no_attention_required_coverage", "NPV_ci95_low", "T_negative"],
                    ascending=[False, False, False],
                )
                return candidates.iloc[0]
        if "sort_selected_candidate_rows" not in globals():
            def sort_selected_candidate_rows(rows: pd.DataFrame, objective: str | None = None) -> pd.DataFrame:
                objective = (objective or EXP2_SELECTION_OBJECTIVE or "quality_first").lower()
                rows = rows.copy()
                if objective in {"coverage_first", "safe_coverage_first", "auto_clear_first"}:
                    return rows.sort_values(
                        ["auto_negative_coverage", "NPV_ci95_low", "auroc", "auprc", "ece"],
                        ascending=[False, False, False, False, True],
                        na_position="last",
                    )
                return rows.sort_values(
                    ["auroc", "auprc", "auto_negative_coverage", "NPV_ci95_low", "ece"],
                    ascending=[False, False, False, False, True],
                    na_position="last",
                )
        if cfg.run_real_eva_x and cfg.run_exp2_lora:
            split_to_indices = {
                split: meta.index[meta["split"] == split].to_numpy(dtype=int)
                for split in ["train", "calibration", "validation", "final_test"]
            }
            lora_epochs = int(os.environ.get("EXP2_LORA_EPOCHS", "2"))
            exp2_lora_model = train_exp2_lora_end_to_end(
                eva_model,
                results,
                y,
                cfg,
                train_indices=split_to_indices["train"],
                epochs=lora_epochs,
            )
            ensure_dirs(cfg)
            torch.save(exp2_lora_model.state_dict(), Path(cfg.checkpoints_dir) / "exp2b_eva_lora_final.pt")
            raw_lora_calib = predict_lora_end_to_end(exp2_lora_model, results, split_to_indices["calibration"], cfg)
            exp2_lora_platt = PlattCalibrator().fit(raw_lora_calib, y[split_to_indices["calibration"]])
            raw_lora_val = predict_lora_end_to_end(exp2_lora_model, results, split_to_indices["validation"], cfg)
            p_exp2_lora_val = exp2_lora_platt.transform(raw_lora_val)
            y_lora_val = y[split_to_indices["validation"]]
            val_meta_lora = meta.iloc[split_to_indices["validation"]].reset_index(drop=True)
            lora_descriptor = f"lora_last2_r4_e{lora_epochs}"
            lora_base_metrics = metrics_summary(y_lora_val, p_exp2_lora_val)
            exp2_lora_thr = threshold_report(y_lora_val, p_exp2_lora_val, target_npv=cfg.target_npv)

            lora_policy_records = []
            lora_policy_routes = {}
            for policy in THRESHOLD_POLICIES:
                selected = select_threshold_by_policy(exp2_lora_thr, policy)
                if selected is None:
                    record = {**lora_base_metrics}
                    record.update({
                        "head_name": lora_descriptor,
                        "threshold_policy": policy["name"],
                        "threshold_policy_selected": False,
                        "selected_T_negative": np.nan,
                        "auto_negative_coverage": 0.0,
                        "N/A_rate": np.nan,
                        "requires_attention_rate": np.nan,
                        "auto_negative_NPV": np.nan,
                        "unsafe_FN_auto_negative": np.nan,
                        "unsafe_FN_per_1000_auto_negative": np.nan,
                        "NPV_ci95_low": np.nan,
                        "selected_t_ood": float(policy.get("t_ood", 0.95)),
                        "model": "EXP-2-FIRST-B EVA LoRA",
                    })
                    lora_policy_records.append(record)
                    continue

                t_neg = float(selected["T_negative"])
                routes = route_decisions(p_exp2_lora_val, val_meta_lora, t_negative=t_neg, t_ood=float(policy.get("t_ood", 0.95)))
                r_metrics = route_metrics(y_lora_val, routes)
                record = {**lora_base_metrics, **r_metrics}
                record.update({
                    "head_name": lora_descriptor,
                    "threshold_policy": policy["name"],
                    "threshold_policy_selected": True,
                    "selected_T_negative": t_neg,
                    "threshold_selected_count": int(selected["selected_count"]),
                    "threshold_validation_NPV": float(selected["NPV"]),
                    "threshold_validation_FN_count": int(selected["FN_count"]),
                    "NPV_ci95_low": float(selected["NPV_ci95_low"]),
                    "selected_t_ood": float(policy.get("t_ood", 0.95)),
                    "model": "EXP-2-FIRST-B EVA LoRA",
                })
                lora_policy_records.append(record)
                if policy["name"] == EXP2_SELECTED_THRESHOLD_POLICY:
                    lora_policy_routes[lora_descriptor] = routes

            exp2_lora_threshold_policy_results = pd.DataFrame(lora_policy_records)
            exp2_lora_threshold_policy_results = exp2_lora_threshold_policy_results.sort_values(
                ["threshold_policy_selected", "threshold_policy", "auroc", "auprc", "auto_negative_coverage"],
                ascending=[False, True, False, False, False],
                na_position="last",
            )

            selected_lora_rows = exp2_lora_threshold_policy_results[
                (exp2_lora_threshold_policy_results["threshold_policy"] == EXP2_SELECTED_THRESHOLD_POLICY)
                & (exp2_lora_threshold_policy_results["threshold_policy_selected"])
            ].copy()
            if selected_lora_rows.empty:
                print(f"No LoRA rows satisfied {EXP2_SELECTED_THRESHOLD_POLICY}; falling back to any selected threshold policy.")
                selected_lora_rows = exp2_lora_threshold_policy_results[
                    exp2_lora_threshold_policy_results["threshold_policy_selected"]
                ].copy()
            if selected_lora_rows.empty:
                raise RuntimeError("No valid LoRA threshold policy was found. Lower min_selected or coverage constraints.")

            selected_lora_rows = sort_selected_candidate_rows(selected_lora_rows, EXP2_SELECTION_OBJECTIVE)
            best_lora_row = selected_lora_rows.iloc[0].to_dict()
            exp2_lora_routes = route_decisions(
                p_exp2_lora_val,
                val_meta_lora,
                t_negative=float(best_lora_row["selected_T_negative"]),
                t_ood=float(best_lora_row.get("selected_t_ood", 0.95)),
            )
            exp2_lora_metrics = metrics_summary(y_lora_val, p_exp2_lora_val)
            exp2_lora_metrics.update({
                "selected_T_negative": float(best_lora_row["selected_T_negative"]),
                "auto_negative_coverage": float(best_lora_row["auto_negative_coverage"]),
                "N/A_rate": float(best_lora_row["N/A_rate"]),
                "requires_attention_rate": float(best_lora_row["requires_attention_rate"]),
                "auto_negative_NPV": float(best_lora_row["auto_negative_NPV"]),
                "unsafe_FN_auto_negative": float(best_lora_row["unsafe_FN_auto_negative"]),
                "unsafe_FN_per_1000_auto_negative": float(best_lora_row["unsafe_FN_per_1000_auto_negative"]),
                "threshold_policy": str(best_lora_row["threshold_policy"]),
                "selected_t_ood": float(best_lora_row.get("selected_t_ood", 0.95)),
                "head_name": lora_descriptor,
                "model": "EXP-2-FIRST-B EVA LoRA",
            })
            ensure_dirs(cfg)
            save_pickle(exp2_lora_platt, Path(cfg.artifacts_dir) / "calibration" / "exp2b_eva_lora_platt.pkl")
            exp2_lora_thr.to_csv(Path(cfg.reports_dir) / "exp2b_lora_threshold_report.csv", index=False)
            exp2_lora_threshold_policy_results.to_csv(Path(cfg.reports_dir) / "exp2b_lora_threshold_policy_results.csv", index=False)
            exp2_lora_routes.to_csv(Path(cfg.reports_dir) / "exp2b_lora_routes_validation.csv", index=False)
            export_json(exp2_lora_metrics, Path(cfg.artifacts_dir) / "models" / "exp2b_lora_metrics_validation.json")
            print("Selected LoRA threshold policy:", exp2_lora_metrics["threshold_policy"])
            print("LoRA selection objective:", EXP2_SELECTION_OBJECTIVE)
            display(pd.DataFrame([exp2_lora_metrics]))
            display(exp2_lora_threshold_policy_results.head(20))
            display(exp2_lora_thr.head(10))
        else:
            print("RUN_EXP2_LORA=0: skipping optional EVA-X-S LoRA branch.")
        """
    ),
    md(
        """
        ## 9. Optional VinDr-CXR 5000-Study Localization Track

        VinDr-CXR is not replacing IN-CXR in this notebook. It is used as a second dataset track for the two main model paths:

        - sample up to **5000 VinDr/VinBigData studies**;
        - map image-level labels to `y_attention`;
        - keep bounding boxes for interpretation validation;
        - run **EXP-1-FIRST-A** style full/ROI embedding + calibrated logistic head;
        - run **EXP-2** EVA-X-S frozen features + calibrated head when `RUN_VINDR_EXP2=1`;
        - produce bbox-aware heatmap/attribution diagnostics.

        Important data note:

        - Official VinDr/VinBigData DICOM data has bbox coordinates in the same pixel space as the source image.
        - Public resized PNG mirrors can keep bbox coordinates from the original DICOM. In that case, put metadata such as `images.csv` / `train_meta.csv` with original `Rows` and `Columns` in `VINDR_ROOT`; the loader uses it to scale bboxes correctly.
        - Without original-size metadata, classification can still run, but bbox-based interpretation should be treated as invalid.

        Product explanation:

        IN-CXR checks screening-like triage quality. VinDr-CXR checks whether explanations are spatially plausible because it has radiologist bounding boxes. This gives us two different kinds of evidence without mixing their claims.
        """
    ),
    md(
        """
        ### 9.1 VinDr Dataset Loading

        This cell is skipped in the first-machine IN-CXR profile. When enabled in a separate profile, it loads VinDr/VinBigData images, labels, metadata, and bounding boxes, then applies the same preprocessing and split logic.
        """
    ),
    code(
        """
        vindr_ready = False
        if cfg.run_vindr_track:
            vindr_df, vindr_bboxes = discover_vindr_dataset(
                cfg.vindr_root,
                max_studies=cfg.max_vindr_studies,
                cfg=cfg,
            )
            vindr_df = make_splits(vindr_df, seed=cfg.random_state)
            vindr_results, vindr_meta = preprocess_dataframe(vindr_df, cfg)
            save_table(vindr_df, Path(cfg.artifacts_dir) / "vindr_data_index")
            save_table(vindr_meta, Path(cfg.artifacts_dir) / "vindr_preprocessing_report")
            vindr_bboxes.to_csv(Path(cfg.artifacts_dir) / "vindr_bboxes.csv", index=False)
            print("VinDr dataframe:", vindr_df.shape)
            print("VinDr bboxes:", vindr_bboxes.shape)
            display(vindr_df["y_attention"].value_counts().rename("count").to_frame())
            display(vindr_df["split"].value_counts().rename("count").to_frame())
            display(vindr_bboxes.head())
            vindr_ready = True
        else:
            print("RUN_VINDR_TRACK=0, skipping VinDr-CXR track.")
        """
    ),
    md(
        """
        ### 9.2 VinDr Model Runs

        When VinDr is enabled, this cell trains the same first-branch heads on the VinDr track and writes a separate VinDr comparison table. These outputs are kept separate from IN-CXR reports.
        """
    ),
    code(
        """
        if vindr_ready:
            vindr_full_tokens, vindr_roi_tokens = extract_exp1_embeddings(
                vindr_results,
                cfg,
                prefix="vindr_exp1_google",
            )
            X_vindr_exp1, vindr_feature_names = build_fusion_features(vindr_full_tokens, vindr_roi_tokens, vindr_meta)
            y_vindr = vindr_meta["y_attention"].values.astype(int)
            parts_vindr = split_arrays_by_meta(X_vindr_exp1, y_vindr, vindr_meta)

            vindr_exp1 = train_logistic_classifier(
                parts_vindr["train"][0],
                parts_vindr["train"][1],
                parts_vindr["calibration"][0],
                parts_vindr["calibration"][1],
                pca_components=cfg.pca_components,
                seed=cfg.random_state,
            )
            p_vindr_val = predict_proba_any(vindr_exp1, parts_vindr["validation"][0])[:, 1]
            vindr_metrics, vindr_thr, vindr_routes = make_model_report(
                "VinDr EXP-1-FIRST-A localization sanity",
                parts_vindr["validation"][1],
                p_vindr_val,
                vindr_meta.iloc[parts_vindr["validation"][2]].reset_index(drop=True),
                cfg.target_npv,
            )
            save_pickle(vindr_exp1, Path(cfg.artifacts_dir) / "models" / "vindr_exp1a_logistic_calibrated.pkl")
            vindr_thr.to_csv(Path(cfg.reports_dir) / "vindr_exp1_threshold_report.csv", index=False)
            vindr_routes.to_csv(Path(cfg.reports_dir) / "vindr_exp1_routes_validation.csv", index=False)
            display(pd.DataFrame([vindr_metrics]))
            display(vindr_thr.head(10))

            vindr_exp2_metrics = None
            if cfg.run_vindr_exp2:
                if cfg.run_real_eva_x and eva_model is not None:
                    X_vindr_eva = extract_eva_features_real(
                        eva_model,
                        vindr_results,
                        image_size=cfg.eva_image_size,
                        batch_size=cfg.batch_size,
                        device=cfg.device,
                    )
                else:
                    raise RuntimeError("RUN_REAL_EVA_X=0 disables VinDr EXP-2. Enable it or set RUN_VINDR_EXP2=0.")
                np.save(Path(cfg.artifacts_dir) / "embeddings" / "vindr_exp2_eva_features.npy", X_vindr_eva)
                parts_vindr_eva = split_arrays_by_meta(X_vindr_eva, y_vindr, vindr_meta)
                vindr_exp2 = train_logistic_classifier(
                    parts_vindr_eva["train"][0],
                    parts_vindr_eva["train"][1],
                    parts_vindr_eva["calibration"][0],
                    parts_vindr_eva["calibration"][1],
                    pca_components=cfg.pca_components,
                    seed=cfg.random_state,
                )
                p_vindr_exp2_val = predict_proba_any(vindr_exp2, parts_vindr_eva["validation"][0])[:, 1]
                vindr_exp2_metrics, vindr_exp2_thr, vindr_exp2_routes = make_model_report(
                    "VinDr EXP-2 EVA-X-S frozen logistic",
                    parts_vindr_eva["validation"][1],
                    p_vindr_exp2_val,
                    vindr_meta.iloc[parts_vindr_eva["validation"][2]].reset_index(drop=True),
                    cfg.target_npv,
                )
                save_pickle(vindr_exp2, Path(cfg.artifacts_dir) / "models" / "vindr_exp2_eva_logistic.pkl")
                vindr_exp2_thr.to_csv(Path(cfg.reports_dir) / "vindr_exp2_threshold_report.csv", index=False)
                vindr_exp2_routes.to_csv(Path(cfg.reports_dir) / "vindr_exp2_routes_validation.csv", index=False)
                display(pd.DataFrame([vindr_exp2_metrics]))
                display(vindr_exp2_thr.head(10))

                vindr_model_comparison = pd.DataFrame([vindr_metrics, vindr_exp2_metrics]).sort_values(
                    ["auto_negative_coverage", "auroc"],
                    ascending=[False, False],
                )
            else:
                print("RUN_VINDR_EXP2=0, skipping VinDr EXP-2.")
                vindr_model_comparison = pd.DataFrame([vindr_metrics])
            vindr_model_comparison.to_csv(Path(cfg.reports_dir) / "vindr_model_comparison.csv", index=False)
            display(vindr_model_comparison)
        """
    ),
    md(
        """
        ### 9.3 VinDr BBox-aware Interpretation

        This cell builds occlusion heatmaps and compares them to radiologist bounding boxes. It is the localization sanity check; it does not run for IN-CXR because IN-CXR has no lesion masks or boxes.
        """
    ),
    code(
        """
        if vindr_ready:
            # BBox-aware occlusion heatmaps for a few abnormal validation cases.
            vindr_interp_dir = Path(cfg.artifacts_dir) / "vindr_interpretation"
            vindr_interp_dir.mkdir(parents=True, exist_ok=True)

            val_indices = parts_vindr["validation"][2]
            candidate_indices = [
                int(i) for i in val_indices
                if y_vindr[int(i)] == 1 and not vindr_bboxes[vindr_bboxes["study_id"].astype(str) == str(vindr_results[int(i)].study_id)].empty
            ]
            if not candidate_indices:
                candidate_indices = [int(i) for i in val_indices[: min(4, len(val_indices))]]

            def predict_vindr_exp1(X_batch):
                return predict_proba_any(vindr_exp1, X_batch)[:, 1]

            def embed_full_for_vindr_heatmap(images):
                return embed_images_exp1(images, cfg, seed=13)

            heatmap_metric_rows = []
            for rank, idx in enumerate(candidate_indices[:4]):
                result = vindr_results[idx]
                base_meta_one = vindr_meta.iloc[[idx]].reset_index(drop=True)
                heatmap = make_occlusion_heatmap_from_feature_builder(
                    result,
                    base_meta_one,
                    embed_full_for_vindr_heatmap,
                    vindr_roi_tokens[idx],
                    predict_vindr_exp1,
                    grid=6,
                    fill_value=0,
                )
                result_full = get_result_image_full(result)
                result_roi = get_result_image_roi(result)
                target = result_full.size[0]
                bbox_mask = bbox_mask_for_result(result, vindr_bboxes, target=target)
                loc_metrics = heatmap_localization_metrics(heatmap, bbox_mask)
                loc_metrics.update({"study_id": result.study_id, "rank": rank})
                heatmap_metric_rows.append(loc_metrics)

                fig, axes = plt.subplots(2, 3, figsize=(16, 10))
                axes = axes.ravel()
                full_img = np.asarray(result_full)
                axes[0].imshow(get_result_raw_preview(result), cmap="gray")
                axes[0].set_title("Original normalized DICOM")
                axes[1].imshow(full_img, cmap="gray")
                axes[1].set_title("Model full input")
                axes[2].imshow(result_roi if result_roi is not None else result_full, cmap="gray")
                axes[2].set_title(f"ROI stream: {result.roi_status}")
                axes[3].imshow(full_img, cmap="gray")
                axes[3].imshow(heatmap, cmap="magma", alpha=0.55)
                axes[3].contour(bbox_mask, levels=[0.5], colors="cyan", linewidths=2)
                axes[3].set_title("Occlusion attribution + radiologist bbox")
                axes[4].imshow(heatmap, cmap="magma")
                axes[4].contour(bbox_mask, levels=[0.5], colors="cyan", linewidths=2)
                axes[4].set_title(
                    "Heatmap only\\n"
                    f"Energy in bbox={loc_metrics['energy_inside_bbox']:.2f}, "
                    f"Pointing hit={loc_metrics['pointing_game_hit']:.0f}"
                )
                axes[5].axis("off")
                explanation = (
                    "How to read this panel:\\n"
                    "1. Magma heatmap = areas where occlusion reduced requires_attention score.\\n"
                    "2. Cyan contour = VinDr radiologist bbox.\\n"
                    "3. Energy inside bbox measures how much attribution mass falls inside bbox.\\n"
                    "4. Pointing game checks whether the hottest point is inside bbox.\\n"
                    "5. This validates spatial plausibility, not diagnosis correctness."
                )
                axes[5].text(0.0, 1.0, explanation, va="top", fontsize=12)
                for ax in axes[:5]:
                    ax.axis("off")
                fig.suptitle(f"VinDr Interpretation Case {rank}: {result.study_id}", fontsize=16)
                plt.tight_layout()
                out = vindr_interp_dir / f"vindr_case_{rank}_{result.study_id}.png"
                fig.savefig(out, dpi=180, bbox_inches="tight")
                plt.show()
                print("Saved:", out)

            vindr_heatmap_metrics = pd.DataFrame(heatmap_metric_rows)
            vindr_heatmap_metrics.to_csv(Path(cfg.reports_dir) / "vindr_heatmap_localization_metrics.csv", index=False)
            display(vindr_heatmap_metrics)

            if not vindr_heatmap_metrics.empty:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                axes[0].bar(vindr_heatmap_metrics["study_id"].astype(str), vindr_heatmap_metrics["energy_inside_bbox"])
                axes[0].set_title("Energy Inside Radiologist BBox")
                axes[0].tick_params(axis="x", rotation=45)
                axes[1].bar(vindr_heatmap_metrics["study_id"].astype(str), vindr_heatmap_metrics["pointing_game_hit"])
                axes[1].set_title("Pointing Game Hit")
                axes[1].tick_params(axis="x", rotation=45)
                axes[2].bar(vindr_heatmap_metrics["study_id"].astype(str), vindr_heatmap_metrics["bbox_iou_at_top20pct"])
                axes[2].set_title("BBox IoU at Top 20% Heatmap")
                axes[2].tick_params(axis="x", rotation=45)
                for ax in axes:
                    ax.set_ylim(0, 1)
                    ax.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                summary_path = Path(cfg.reports_dir) / "vindr_interpretation_summary.png"
                fig.savefig(summary_path, dpi=180, bbox_inches="tight")
                plt.show()
                print("Saved:", summary_path)
        """
    ),
    md(
        """
        ## 10. Optional EXP-3: CheXFound Frozen Reference

        EXP-3 is intentionally optional. It is useful as a strong CXR reference, but it should not block the MVP because real CheXFound setup may require repository-specific dependencies and checkpoints.

        This notebook keeps the path as:

        - clone/check CheXFound repository if enabled;
        - use a checkpoint path from `CHEXFOUND_CKPT` if available;
        - stop with a clear message until the real checkpoint adapter is implemented.
        """
    ),
    code(
        """
        exp3_metrics = None
        if cfg.run_exp3_chexfound:
            try:
                chexfound_repo = Path(cfg.project_dir) / "external" / "CheXFound"
                chexfound_repo.parent.mkdir(parents=True, exist_ok=True)
                if not chexfound_repo.exists():
                    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/RPIDIAL/CheXFound.git", str(chexfound_repo)], check=True)
                ckpt = os.environ.get("CHEXFOUND_CKPT")
                if ckpt and Path(ckpt).exists():
                    print("CheXFound checkpoint detected:", ckpt)
                    print("Real CheXFound feature extraction is repository/checkpoint specific; adapter is intentionally not wired in the first production run.")
                else:
                    raise RuntimeError("CHEXFOUND_CKPT must point to a real checkpoint when RUN_EXP3_CHEXFOUND=1.")
                raise NotImplementedError("Real CheXFound feature adapter is not wired yet. Keep RUN_EXP3_CHEXFOUND=0 for the first production run.")
            except Exception as exc:
                print("EXP-3 skipped after setup error:", exc)
                exp3_metrics = None
        else:
            print("RUN_EXP3_CHEXFOUND=0, optional EXP-3 skipped.")
        """
    ),
    md(
        """
        ## 11. Calibration, N/A-router, Model Comparison

        The main product metric is not accuracy. We compare models by:

        - auto-negative coverage at target NPV;
        - false negatives per 1000 inside the auto-negative zone;
        - N/A rate;
        - calibration quality.

        Postprocessing follows the plan:

        - Platt calibration for neural heads where needed.
        - Negative threshold search for safe `no_attention_required` routing.
        - Positive threshold and gray-zone routing to `requires_attention` / `N/A`.
        - QA and OOD gates before automatic clearance.
        - `final_test` evaluated with the fixed selected validation threshold, so it stays a held-out check rather than another tuning split.
        """
    ),
    md(
        """
        ### 11.1 Validation Comparison

        This cell combines EXP-1 and EXP-2 validation reports and registers the selected model for final evaluation. By default it ranks candidates by AUROC/AUPRC first, then safe auto-negative coverage. Set `EXP2_SELECTION_OBJECTIVE=coverage_first` in Run Config when the goal is to maximize safe auto-clear among candidates that satisfy the selected threshold policy.
        """
    ),
    code(
        """
        model_rows = []
        model_registry = {}
        MODEL_SELECTION_OBJECTIVE = os.environ.get(
            "EXP2_SELECTION_OBJECTIVE",
            globals().get("EXP2_SELECTION_OBJECTIVE", "quality_first"),
        ).lower()

        def register_candidate(name, metrics_name, model_name, X_name, parts_name, kind, calibrator_name=None):
            if metrics_name not in globals() or model_name not in globals() or X_name not in globals() or parts_name not in globals():
                print(f"Skipping {name}: required variables are not available in this run.")
                return
            metrics = globals()[metrics_name]
            if metrics is None:
                print(f"Skipping {name}: metrics are None.")
                return
            row = dict(metrics)
            row["model"] = name
            model_rows.append(row)
            entry = {
                "model": globals()[model_name],
                "X": globals()[X_name],
                "parts": globals()[parts_name],
                "meta": meta,
                "kind": kind,
            }
            if calibrator_name is not None and calibrator_name in globals():
                entry["calibrator"] = globals()[calibrator_name]
            if "selected_T_negative" in row and pd.notna(row["selected_T_negative"]):
                entry["selected_T_negative"] = float(row["selected_T_negative"])
            if "threshold_policy" in row and pd.notna(row["threshold_policy"]):
                entry["threshold_policy"] = str(row["threshold_policy"])
            if "selected_t_ood" in row and pd.notna(row["selected_t_ood"]):
                entry["selected_t_ood"] = float(row["selected_t_ood"])
            if "head_name" in row and pd.notna(row["head_name"]):
                entry["head_name"] = str(row["head_name"])
            model_registry[name] = entry

        def sort_model_candidates(df: pd.DataFrame, objective: str | None = None) -> pd.DataFrame:
            objective = (objective or MODEL_SELECTION_OBJECTIVE or "quality_first").lower()
            out = df.copy()
            if objective in {"coverage_first", "safe_coverage_first", "auto_clear_first"}:
                out["_unsafe_sort"] = out.get("unsafe_FN_auto_negative", pd.Series(np.nan, index=out.index)).fillna(1e9)
                out["_npv_sort"] = out.get("auto_negative_NPV", pd.Series(np.nan, index=out.index)).fillna(-1.0)
                out["_coverage_sort"] = out.get("auto_negative_coverage", pd.Series(np.nan, index=out.index)).fillna(-1.0)
                return out.sort_values(
                    ["_unsafe_sort", "_npv_sort", "_coverage_sort", "auroc", "auprc"],
                    ascending=[True, False, False, False, False],
                    na_position="last",
                ).drop(columns=[c for c in ["_unsafe_sort", "_npv_sort", "_coverage_sort"] if c in out.columns])
            return out.sort_values(
                ["auroc", "auprc", "auto_negative_coverage"],
                ascending=[False, False, False],
                na_position="last",
            )

        register_candidate(
            "EXP-1-FIRST-A logistic",
            "exp1a_metrics",
            "exp1a",
            "X_exp1",
            "parts_exp1",
            "sklearn",
        )
        register_candidate(
            "EXP-1-FIRST-B MLP",
            "exp1b_metrics",
            "exp1b",
            "X_exp1",
            "parts_exp1",
            "torch_mlp",
            "exp1b_platt",
        )
        register_candidate(
            "EXP-2-FIRST-A1 EVA frozen logistic",
            "exp2a1_metrics",
            "exp2a1",
            "X_eva",
            "parts_eva",
            "sklearn",
        )
        register_candidate(
            "EXP-2-FIRST-A2-TUNED EVA frozen MLP",
            "exp2a2_tuned_metrics",
            "exp2a2_tuned",
            "X_eva",
            "parts_eva",
            "torch_mlp",
            "exp2a2_tuned_platt",
        )
        register_candidate(
            "EXP-2-FIRST-A2 EVA frozen MLP",
            "exp2a2_metrics",
            "exp2a2",
            "X_eva",
            "parts_eva",
            "torch_mlp",
            "exp2a2_platt",
        )
        register_candidate(
            "EXP-2-FIRST-A2-STRONG EVA frozen MLP",
            "exp2a2_strong_metrics",
            "exp2a2_strong",
            "X_eva",
            "parts_eva",
            "torch_mlp",
            "exp2a2_strong_platt",
        )
        register_candidate(
            "EXP-2-FIRST-B EVA LoRA",
            "exp2_lora_metrics",
            "exp2_lora_model",
            "X_eva",
            "parts_eva",
            "lora_e2e",
            "exp2_lora_platt",
        )

        if not model_rows:
            raise RuntimeError("No completed model metrics found. Run at least one EXP-1 or EXP-2 head before model comparison.")

        primary_model_names = [row["model"] for row in model_rows]
        extended_rows = list(model_rows)
        if globals().get("exp3_metrics") is not None:
            extended_rows.append(exp3_metrics)
        if globals().get("vindr_ready", False) and "vindr_model_comparison" in globals():
            for _, row in vindr_model_comparison.iterrows():
                extended_rows.append(row.to_dict())
        model_comparison = pd.DataFrame(model_rows)
        model_comparison = sort_model_candidates(model_comparison, MODEL_SELECTION_OBJECTIVE)
        model_comparison.to_csv(Path(cfg.reports_dir) / "model_comparison.csv", index=False)
        display(model_comparison)

        extended_experiment_report = sort_model_candidates(pd.DataFrame(extended_rows), MODEL_SELECTION_OBJECTIVE)
        extended_experiment_report.to_csv(Path(cfg.reports_dir) / "extended_experiment_report.csv", index=False)
        print("Extended report includes optional EXP-3 and VinDr track when available:")
        display(extended_experiment_report)

        best_name = str(model_comparison.iloc[0]["model"])
        print("Model selection objective:", MODEL_SELECTION_OBJECTIVE)
        print("Selected best candidate by validation table:", best_name)
        best = model_registry[best_name]
        """
    ),
    md(
        """
        ### 11.2 Fixed-threshold Final Test

        This cell freezes the selected validation threshold, applies OOD and QA routing, and evaluates the chosen candidate on `final_test` without retuning.
        """
    ),
    code(
        """
        # OOD-aware routes for the selected model.
        best_X = best["X"]
        best_parts = best["parts"]
        ood = fit_ood_model(best_parts["train"][0])

        def score_candidate_split(candidate: dict, split_name: str, ood_model=None):
            parts = candidate["parts"]
            Xs, ys, idx = parts[split_name]
            if candidate["kind"] == "torch_mlp":
                raw = predict_torch_mlp(candidate["model"], Xs, device=cfg.device)
                p = candidate["calibrator"].transform(raw)
            elif candidate["kind"] == "lora_e2e":
                raw = predict_lora_end_to_end(candidate["model"], results, idx, cfg)
                p = candidate["calibrator"].transform(raw)
            else:
                p = predict_proba_any(candidate["model"], Xs)[:, 1]
            ood_s = ood_score(ood_model, Xs) if ood_model is not None else None
            return ys, p, ood_s, idx

        y_val_best, p_val_best, ood_val_best, best_val_idx = score_candidate_split(best, "validation", ood_model=ood)
        best_thr = threshold_report(y_val_best, p_val_best, target_npv=cfg.target_npv)
        if "selected_T_negative" in best and pd.notna(best["selected_T_negative"]):
            selected_T_negative = float(best["selected_T_negative"])
            print("Using validation-selected threshold from model registry:", selected_T_negative)
        else:
            selected_T_negative = choose_negative_threshold(best_thr)
            print("Using default max-coverage target-NPV threshold:", selected_T_negative)
        selected_t_ood = float(best.get("selected_t_ood", 0.95))
        print("Threshold policy:", best.get("threshold_policy", "target_npv_max_coverage"))
        print("OOD threshold:", selected_t_ood)
        best_val_meta = meta.iloc[best_val_idx].reset_index(drop=True)
        best_routes = route_decisions(
            p_val_best,
            best_val_meta,
            t_negative=selected_T_negative,
            ood_score=ood_val_best,
            t_ood=selected_t_ood,
        )

        y_test_best, p_test_best, ood_test_best, best_test_idx = score_candidate_split(best, "final_test", ood_model=ood)
        best_test_meta = meta.iloc[best_test_idx].reset_index(drop=True)
        final_test_metrics, final_test_routes = fixed_threshold_evaluation(
            best_name + " final_test",
            y_test_best,
            p_test_best,
            best_test_meta,
            t_negative=selected_T_negative,
            ood_score_values=ood_test_best,
            t_ood=selected_t_ood,
        )

        print("Best threshold report")
        display(best_thr.head(10))
        print("Best validation routes")
        display(best_routes.head())
        print("Final test metrics with fixed validation threshold")
        display(pd.DataFrame([final_test_metrics]))
        display(final_test_routes.head())
        best_routes.to_csv(Path(cfg.reports_dir) / "best_routes_validation.csv", index=False)
        best_thr.to_csv(Path(cfg.reports_dir) / "best_threshold_report.csv", index=False)
        final_test_routes.to_csv(Path(cfg.reports_dir) / "best_routes_final_test.csv", index=False)
        pd.DataFrame([final_test_metrics]).to_csv(Path(cfg.reports_dir) / "best_final_test_metrics.csv", index=False)

        # Reporting-only final-test check for every registered candidate. This does not retune
        # thresholds and does not change the selected model; it helps compare LoRA vs MLP honestly.
        all_candidate_final_rows = []
        for candidate_name, candidate in model_registry.items():
            if "selected_T_negative" not in candidate:
                print(f"Skipping final-test candidate report for {candidate_name}: no validation-selected threshold.")
                continue
            try:
                candidate_ood = fit_ood_model(candidate["parts"]["train"][0])
                yy, pp, ood_values, idx = score_candidate_split(candidate, "final_test", ood_model=candidate_ood)
                candidate_meta = meta.iloc[idx].reset_index(drop=True)
                candidate_metrics, _ = fixed_threshold_evaluation(
                    candidate_name + " final_test",
                    yy,
                    pp,
                    candidate_meta,
                    t_negative=float(candidate["selected_T_negative"]),
                    ood_score_values=ood_values,
                    t_ood=float(candidate.get("selected_t_ood", 0.95)),
                )
                candidate_metrics["candidate_model"] = candidate_name
                candidate_metrics["threshold_policy"] = candidate.get("threshold_policy", "target_npv_max_coverage")
                candidate_metrics["selected_t_ood"] = float(candidate.get("selected_t_ood", 0.95))
                candidate_metrics["selected_T_negative"] = float(candidate["selected_T_negative"])
                candidate_metrics["head_name"] = candidate.get("head_name")
                all_candidate_final_rows.append(candidate_metrics)
            except Exception as exc:
                print(f"Candidate final-test report failed for {candidate_name}: {exc}")

        all_candidates_final_test = pd.DataFrame(all_candidate_final_rows)
        if not all_candidates_final_test.empty:
            all_candidates_final_test = sort_model_candidates(all_candidates_final_test, MODEL_SELECTION_OBJECTIVE)
            all_candidates_final_test.to_csv(Path(cfg.reports_dir) / "all_candidates_final_test_metrics.csv", index=False)
            print("All-candidate final-test metrics with fixed validation thresholds:")
            display(all_candidates_final_test)
        """
    ),
    md(
        """
        ## 12. Interpretation and Case Review

        Interpretation here is not a medical diagnosis. It is a safety/debug artifact:

        - show original normalized image;
        - show model input;
        - show ROI;
        - show route reason and QA flags;
        - show whether full/ROI/QA fusion helped.

        Current interpretation level:

        - EXP-1 is interpretable through stream ablation and feature-group permutation: full image vs ROI vs QA.
        - EXP-2 is less transparent in frozen-feature mode; this notebook reports feature-level permutation and case review. Strong clinical heatmaps/Grad-CAM should be added only after the real EVA-X-S encoder path is stable.

        Important limitation:

        IN-CXR first branch is a **binary normal/abnormal dataset without lesion masks/bounding boxes**. That does not make heatmaps impossible: Grad-CAM, attention rollout, occlusion maps, or token attribution can still produce model-attribution heatmaps for the binary decision. But these heatmaps are **not validated localization**. They can show where the model is sensitive, not prove where the pathology is.

        Therefore first-branch interpretation focuses on auditable evidence we can defend:

        - preprocessing/ROI visual audit;
        - route reason and QA flags;
        - full vs ROI vs QA ablation;
        - permutation importance by feature group;
        - qualitative attribution heatmaps only as optional diagnostics after the real encoder path is stable.
        """
    ),
    md(
        """
        ### 12.1 Case Review Panels

        This cell renders a few routed cases with original normalized image, model input, ROI input, probability, route, and QA context. These panels are for audit and product review.
        """
    ),
    code(
        """
        def render_case(result, route_row=None):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(get_result_raw_preview(result), cmap="gray")
            axes[0].set_title("Original normalized")
            full_img = get_result_image_full(result)
            roi_img = get_result_image_roi(result)
            axes[1].imshow(full_img, cmap="gray")
            axes[1].set_title("Full input")
            if roi_img is not None:
                axes[2].imshow(roi_img, cmap="gray")
            else:
                axes[2].text(0.5, 0.5, "ROI missing", ha="center", va="center")
            title = f"ROI: {result.roi_status}"
            if route_row is not None:
                title += f"\\nroute={route_row['route']} | p={route_row['p_requires_attention']:.3f}"
            axes[2].set_title(title)
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            return fig

        review_dir = Path(cfg.artifacts_dir) / "case_review"
        review_dir.mkdir(parents=True, exist_ok=True)

        # Render a few routed cases.
        for j, row in best_routes.head(min(6, len(best_routes))).iterrows():
            study_id = row["study_id"]
            result_idx = next(i for i, r in enumerate(results) if r.study_id == study_id)
            fig = render_case(results[result_idx], row)
            safe_route = str(row["route"]).replace("/", "_")
            out = review_dir / f"case_{j}_{safe_route}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.show()
            print("Saved:", out)

        if "exp1_ablation" in globals():
            display(exp1_ablation)
        else:
            print("EXP-1 ablation is not available in this run; skipping EXP-1 ablation display.")
        """
    ),
    md(
        """
        ### 12.2 Feature-group Interpretation

        This cell estimates which feature groups matter by permutation: full-image stream, ROI stream, QA metadata, and EVA frozen features.
        """
    ),
    code(
        """
        # Feature-group interpretation for EXP-1 selected feature set, when EXP-1 was run.
        if all(name in globals() for name in ["full_tokens", "roi_tokens", "exp1a", "parts_exp1"]):
            full_dim = pool_tokens(full_tokens).shape[1]
            roi_dim = pool_tokens(roi_tokens).shape[1]
            qa_dim = qa_feature_matrix(meta)[0].shape[1]
            exp1_group_slices = {
                "full_embedding": slice(0, full_dim),
                "roi_embedding": slice(full_dim, full_dim + roi_dim),
                "qa_metadata": slice(full_dim + roi_dim, full_dim + roi_dim + qa_dim),
            }

            def predict_exp1a_group(X_batch):
                return predict_proba_any(exp1a, X_batch)[:, 1]

            exp1_group_importance = permutation_group_importance(
                predict_exp1a_group,
                parts_exp1["validation"][0],
                parts_exp1["validation"][1],
                exp1_group_slices,
                n_repeats=5,
                seed=cfg.random_state,
            )
            exp1_group_importance.to_csv(Path(cfg.reports_dir) / "exp1_group_importance.csv", index=False)
            display(exp1_group_importance)
        else:
            print("EXP-1 feature-group interpretation skipped: EXP-1 variables are not available in this run.")

        # Coarse EXP-2 interpretation: permutation of all frozen EVA features.
        if "parts_eva" in globals():
            def predict_exp2_group(X_batch):
                if best["kind"] == "torch_mlp" and best.get("X") is X_eva:
                    raw = predict_torch_mlp(best["model"], X_batch, device=cfg.device)
                    return best["calibrator"].transform(raw)
                if best["kind"] == "sklearn" and best.get("X") is X_eva:
                    return predict_proba_any(best["model"], X_batch)[:, 1]
                if "exp2a1" in globals():
                    return predict_proba_any(exp2a1, X_batch)[:, 1]
                raise RuntimeError("No EXP-2 predictor is available for group interpretation.")

            exp2_group_importance = permutation_group_importance(
                predict_exp2_group,
                parts_eva["validation"][0],
                parts_eva["validation"][1],
                {"eva_frozen_features": slice(0, parts_eva["validation"][0].shape[1])},
                n_repeats=5,
                seed=cfg.random_state,
            )
            exp2_group_importance.to_csv(Path(cfg.reports_dir) / "exp2_group_importance.csv", index=False)
            display(exp2_group_importance)
        else:
            print("EXP-2 feature-group interpretation skipped: parts_eva is not available.")
        """
    ),
    md(
        """
        ## 13. Statistics and Error Analysis

        Route-aware error analysis:

        - false negative in `no_attention_required` is the critical unsafe error;
        - false negative in `N/A` is less dangerous because it still goes to manual review;
        - false positive in `requires_attention` increases doctor workload but is safer than an unsafe auto-negative.
        """
    ),
    md(
        """
        ### 13.1 Route-aware Error Analysis

        This cell separates unsafe false negatives in `no_attention_required` from safer manual-review false negatives in `N/A`, which matches the product risk model.
        """
    ),
    code(
        """
        val_truth = pd.DataFrame({
            "study_id": meta.iloc[best_val_idx]["study_id"].values,
            "y_attention": y_val_best,
            "p_requires_attention": p_val_best,
        })
        error_analysis = best_routes.merge(val_truth, on=["study_id", "p_requires_attention"], how="left")
        error_analysis["unsafe_false_negative"] = (
            (error_analysis["route"] == "no_attention_required") & (error_analysis["y_attention"] == 1)
        )
        error_analysis["safe_manual_review_false_negative"] = (
            (error_analysis["route"] == "N/A") & (error_analysis["y_attention"] == 1)
        )
        error_analysis["workload_false_positive"] = (
            (error_analysis["route"] == "requires_attention") & (error_analysis["y_attention"] == 0)
        )
        error_analysis.to_csv(Path(cfg.reports_dir) / "error_analysis_validation.csv", index=False)
        display(error_analysis)
        display(error_analysis[["unsafe_false_negative", "safe_manual_review_false_negative", "workload_false_positive"]].sum().rename("count").to_frame())
        """
    ),
    md(
        """
        ### 13.2 Calibration, Confidence Intervals, and Subgroups

        This cell writes reliability curves, bootstrap confidence intervals, and subgroup route reports using available QA/source metadata.
        """
    ),
    code(
        """
        # Calibration tables, bootstrap confidence intervals, and subgroup analysis.
        val_calibration = calibration_table(y_val_best, p_val_best, n_bins=10)
        test_calibration = calibration_table(y_test_best, p_test_best, n_bins=10)
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
        calibration_plot_path = Path(cfg.reports_dir) / "reliability_diagram.png"
        fig.savefig(calibration_plot_path, dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved:", calibration_plot_path)

        ci_rows = []
        ci_specs = {
            "AUROC": safe_auc,
            "AUPRC": safe_auprc,
            "Brier": lambda yy, pp: brier_score_loss(yy, pp),
            "NPV@selected_T_negative": lambda yy, pp: npv_at_threshold(yy, pp, selected_T_negative),
        }
        for split_name, yy, pp in [
            ("validation", y_val_best, p_val_best),
            ("final_test", y_test_best, p_test_best),
        ]:
            for metric_name, fn in ci_specs.items():
                point, lo, hi = bootstrap_ci(yy, pp, fn, n_boot=1000, seed=cfg.random_state)
                ci_rows.append({"split": split_name, "metric": metric_name, "point": point, "ci95_low": lo, "ci95_high": hi})
        ci_report = pd.DataFrame(ci_rows)
        ci_report.to_csv(Path(cfg.reports_dir) / "bootstrap_ci_report.csv", index=False)
        display(ci_report)

        # Simple subgroup checks available before real clinical metadata arrives.
        best_val_meta_for_groups = best_val_meta.copy()
        if best_val_meta_for_groups["quality_score"].nunique() > 1:
            best_val_meta_for_groups["quality_bin"] = pd.qcut(
                best_val_meta_for_groups["quality_score"].rank(method="first"),
                q=min(3, len(best_val_meta_for_groups)),
                duplicates="drop",
            ).astype(str)
        else:
            best_val_meta_for_groups["quality_bin"] = "all_same_quality"

        subgroup_reports = []
        for col in ["roi_status", "quality_bin", "source_type"]:
            if col in best_val_meta_for_groups.columns:
                subgroup_reports.append(subgroup_route_report(y_val_best, p_val_best, best_routes, best_val_meta_for_groups, col))
        subgroup_report = pd.concat(subgroup_reports, ignore_index=True) if subgroup_reports else pd.DataFrame()
        subgroup_report.to_csv(Path(cfg.reports_dir) / "subgroup_route_report_validation.csv", index=False)
        display(subgroup_report)
        """
    ),
    md(
        """
        ## 14. Quantization Candidate for Best Head

        Quantization is a deployment check, not a way to improve research metrics. After quantization we must re-check calibration/thresholds.
        """
    ),
    code(
        """
        quant_report = {
            "best_model": best_name,
            "threshold_policy": best.get("threshold_policy", "target_npv_max_coverage"),
            "selected_t_ood": best.get("selected_t_ood", 0.95),
        }
        if best["kind"] == "torch_mlp":
            X_sample = best_parts["validation"][0][: min(16, len(best_parts["validation"][0]))]
            fp_bench = benchmark_predict(lambda: predict_torch_mlp(best["model"], X_sample, device="cpu"), repeats=20)
            q_model = quantize_torch_mlp(best["model"])
            scaler = getattr(best["model"], "scaler", None)
            def q_predict():
                Xs = scaler.transform(X_sample).astype(np.float32) if scaler is not None else X_sample.astype(np.float32)
                with torch.no_grad():
                    return torch.sigmoid(q_model(torch.tensor(Xs))).numpy()
            q_bench = benchmark_predict(q_predict, repeats=20)
            quant_report.update({"fp32_head": fp_bench, "int8_dynamic_head": q_bench})
            torch.save(q_model.state_dict(), Path(cfg.artifacts_dir) / "quantization" / "best_head_dynamic_int8.pt")
        elif best["kind"] == "lora_e2e":
            sample_idx = best_parts["validation"][2][: min(8, len(best_parts["validation"][2]))]
            quant_report["note"] = "Best model is EVA-X-S LoRA end-to-end; dynamic head-only quantization is not applied in this notebook."
            quant_report["lora_predict"] = benchmark_predict(
                lambda: predict_lora_end_to_end(best["model"], results, sample_idx, cfg),
                repeats=5,
            )
        else:
            X_sample = best_parts["validation"][0][: min(16, len(best_parts["validation"][0]))]
            quant_report["note"] = "Best model is sklearn/logistic; quantization is not needed for this shallow head."
            quant_report["sklearn_predict"] = benchmark_predict(lambda: best["model"].predict_proba(X_sample), repeats=20)

        quant_path = export_json(quant_report, Path(cfg.artifacts_dir) / "quantization" / "quantization_report.json")
        print("Quantization report:", quant_path)
        print(json.dumps(quant_report, indent=2))
        """
    ),
    md(
        """
        ## 15. Export Backend Artifacts

        This cell saves a compact manifest with model choice, preprocessing version, reports, and router artifacts. Backend should reproduce exactly this logic after the research notebook is accepted.
        """
    ),
    code(
        """
        manifest = {
            "project": "fluoro_cxr_safety_first_mvp",
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "task": "no_attention_required / requires_attention / N/A",
            "dataset_mode": "real",
            "datasets": {
                "primary_quality": "IN-CXR/local screening-like normal-abnormal" if cfg.run_primary_track else "disabled",
                "localization_sanity": "VinDr/VinBigData 5000-study bbox-aware track" if cfg.run_vindr_track else "disabled",
            },
            "run_primary_track": bool(cfg.run_primary_track),
            "run_vindr_exp2": bool(cfg.run_vindr_exp2),
            "best_model": best_name,
            "best_threshold_policy": best.get("threshold_policy", "target_npv_max_coverage"),
            "best_t_ood": best.get("selected_t_ood", 0.95),
            "best_head_name": best.get("head_name"),
            "optional_exp3_enabled": bool(cfg.run_exp3_chexfound),
            "target_npv": cfg.target_npv,
            "preprocessing_version": "preproc_v1",
            "artifacts_dir": cfg.artifacts_dir,
            "reports_dir": cfg.reports_dir,
            "model_comparison": model_comparison.to_dict(orient="records"),
            "extended_experiment_report": extended_experiment_report.to_dict(orient="records"),
        }
        manifest_path = export_json(manifest, Path(cfg.artifacts_dir) / "manifest.json")
        router_path = export_json(
            {
                "best_model": best_name,
                "threshold_policy": best.get("threshold_policy", "target_npv_max_coverage"),
                "threshold_table": best_thr.head(20).to_dict(orient="records"),
                "selected_T_negative": float(selected_T_negative),
                "selected_t_ood": float(selected_t_ood),
                "router_logic": "quality/OOD checks -> T_negative auto-negative -> T_positive requires_attention -> gray-zone N/A",
            },
            Path(cfg.artifacts_dir) / "router" / "router_config.json",
        )
        print("Manifest:", manifest_path)
        print("Router config:", router_path)
        """
    ),
    md(
        """
        ## 16. Final Sanity Checks

        These checks are intentionally simple but cover the critical chain:

        data -> image preprocessing -> ROI/QA -> enabled experiments -> calibration -> router -> artifacts.
        """
    ),
    code(
        """
        assert len(df) > 0, "Dataset index is empty"
        assert len(results) == len(meta), "Preprocessing results/meta mismatch"
        assert set(meta["roi_status"]).issubset({"valid", "invalid", "not_available"})
        if "X_exp1" in globals():
            assert X_exp1.shape[0] == len(meta)
        if "X_eva" in globals():
            assert X_eva.shape[0] == len(meta)
        assert "X_exp1" in globals() or "X_eva" in globals(), "No primary feature matrix is available"
        assert not model_comparison.empty
        assert not extended_experiment_report.empty
        assert best_routes["route"].isin(["no_attention_required", "requires_attention", "N/A"]).all()
        assert Path(cfg.artifacts_dir, "manifest.json").exists()
        assert Path(cfg.reports_dir, "model_comparison.csv").exists()
        if cfg.run_vindr_track and vindr_ready:
            assert Path(cfg.reports_dir, "vindr_heatmap_localization_metrics.csv").exists()
            assert Path(cfg.reports_dir, "vindr_interpretation_summary.png").exists()
            assert Path(cfg.reports_dir, "vindr_model_comparison.csv").exists()

        print("FINAL SANITY CHECKS PASSED")
        print("Artifacts:", cfg.artifacts_dir)
        print("Reports:", cfg.reports_dir)
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    "colab": {"provenance": [], "gpuType": "T4"},
    "accelerator": "GPU",
}
nb["cells"] = cells

nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
