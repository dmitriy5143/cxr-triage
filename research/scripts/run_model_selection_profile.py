from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
import time
import traceback
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent


def display(obj=None, *args, **kwargs):
    """Small notebook-display fallback for headless local execution."""
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            print(obj.head(30).to_string())
            if len(obj) > 30:
                print(f"... [{len(obj)} rows x {len(obj.columns)} columns]")
            return
        if isinstance(obj, pd.Series):
            print(obj.head(30).to_string())
            if len(obj) > 30:
                print(f"... [{len(obj)} rows]")
            return
    except Exception:
        pass
    print(repr(obj))


def prepare_env(profile: str) -> None:
    os.environ["EXPERIMENT_PROFILE"] = profile
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MOUNT_DRIVE", "0")
    os.environ.setdefault("IN_CXR_DOWNLOAD_DIR", str(ROOT / "data" / "incxr_png"))
    os.environ.setdefault("RESUME_EXISTING_CHECKPOINTS", "1")
    os.environ.setdefault("LORA_CHECKPOINT_INTERVAL", "15")

    access_token = Path.home() / ".kaggle" / "access_token"
    if access_token.exists() and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = access_token.read_text(encoding="utf-8").strip()

    # Avoid runaway thread oversubscription in sklearn/BLAS on Apple Silicon.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")


def execute_notebook(profile: str, notebook_path: Path) -> None:
    prepare_env(profile)
    builtins.display = display

    nb = nbformat.read(notebook_path, as_version=4)
    ns: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
        "display": display,
    }
    started = time.time()
    print(json.dumps({
        "event": "profile_start",
        "profile": profile,
        "notebook": str(notebook_path),
        "cwd": str(ROOT),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False), flush=True)

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = str(cell.source).strip()
        if not source:
            continue
        cell_started = time.time()
        print(f"\n===== EXEC CELL {idx} =====", flush=True)
        try:
            code = compile(source, f"{notebook_path.name}:cell_{idx}", "exec")
            exec(code, ns, ns)
            only_lora_name = os.environ.get("ONLY_LORA_NAME")
            if idx == 2 and only_lora_name and "RUN" in ns:
                run = ns["RUN"]
                if isinstance(run, dict) and "lora_sweep" in run:
                    before = len(run["lora_sweep"])
                    run["lora_sweep"] = [
                        cfg for cfg in run["lora_sweep"] if cfg.get("name") == only_lora_name
                    ]
                    print(
                        f"ONLY_LORA_NAME={only_lora_name}: filtered lora_sweep "
                        f"{before} -> {len(run['lora_sweep'])}",
                        flush=True,
                    )
                    if not run["lora_sweep"]:
                        raise RuntimeError(f"ONLY_LORA_NAME={only_lora_name!r} did not match any lora_sweep config.")
        except Exception:
            print(f"===== CELL {idx} FAILED after {time.time() - cell_started:.1f}s =====", flush=True)
            traceback.print_exc()
            raise
        print(f"===== CELL {idx} DONE in {time.time() - cell_started:.1f}s =====", flush=True)

    print(json.dumps({
        "event": "profile_complete",
        "profile": profile,
        "elapsed_sec": round(time.time() - started, 1),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--notebook", default=str(ROOT / "fluoro_mvp_model_selection.ipynb"))
    args = parser.parse_args()
    execute_notebook(args.profile, Path(args.notebook))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
