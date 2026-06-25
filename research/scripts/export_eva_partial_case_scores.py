from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluoro_mvp_core import (  # noqa: E402
    EVAEndToEndClassifier,
    fixed_threshold_evaluation,
    image_to_eva_tensor,
    load_real_eva_x,
    ood_score,
    route_decisions,
    split_arrays_by_meta,
)


RUN_ROOT = ROOT / "fluoro_mvp_outputs" / "incxr_eva_base_partial_unfreeze_t4"
OUT_DIR = ROOT / "selected_model_workbench" / "case_scores"

CANDIDATES = {
    "last1": {
        "model_name": "eva_base_partial_unfreeze_base_unfreeze_last1_e150",
        "checkpoint": RUN_ROOT / "checkpoints" / "base_unfreeze_last1_e150" / "best.pt",
        "calibrator": RUN_ROOT / "artifacts" / "calibration" / "eva_base_partial_unfreeze_base_unfreeze_last1_e150_calibrator.pkl",
        "router": ROOT
        / "selected_model_workbench"
        / "router_workbench"
        / "router_configs"
        / "primary_last1_deployment_zero_fn_router.json",
    },
    "last2": {
        "model_name": "eva_base_partial_unfreeze_base_unfreeze_last2_e150",
        "checkpoint": RUN_ROOT / "checkpoints" / "base_unfreeze_last2_e150" / "best.pt",
        "calibrator": RUN_ROOT / "artifacts" / "calibration" / "eva_base_partial_unfreeze_base_unfreeze_last2_e150_calibrator.pkl",
        "router": ROOT
        / "selected_model_workbench"
        / "router_workbench"
        / "router_configs"
        / "challenger_last2_strict_zero_fn_router.json",
    },
}


def choose_device(requested: str) -> str:
    requested = requested.lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_candidate_model(checkpoint_path: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapt_cfg = dict(state.get("adapt_cfg") or {})
    variant = str(adapt_cfg.get("variant") or state.get("variant") or "base").lower()
    weights_path = RUN_ROOT / "models" / "eva_x" / "eva_x_base_patch16_merged520k_mim.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing local EVA-X-B weights at {weights_path}. "
            "This script is intentionally offline and will not download model weights."
        )
    encoder = load_real_eva_x(
        str(RUN_ROOT),
        variant=variant,
        device=device,
        weights_path=str(weights_path),
    )
    state_dict = state["state_dict"]
    head_weight = state_dict.get("head.1.weight")
    if head_weight is None:
        raise KeyError(f"{checkpoint_path} is missing head.1.weight.")
    hidden, feature_dim = map(int, head_weight.shape)
    model = EVAEndToEndClassifier(
        encoder,
        feature_dim=feature_dim,
        hidden=hidden,
        dropout=float(adapt_cfg.get("head_dropout", 0.20)),
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, {"variant": variant, "adapt_cfg": adapt_cfg}


def predict_image_paths(
    model: torch.nn.Module,
    image_paths: list[str],
    *,
    image_size: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    n = len(image_paths)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                img = Image.open(path).convert("L")
                tensors.append(image_to_eva_tensor(img, image_size=image_size))
            xb = torch.stack(tensors).to(device)
            logits = model(xb)
            preds.append(torch.sigmoid(logits).detach().float().cpu().numpy())
            if start == 0 or (start // batch_size) % 25 == 0:
                print(f"  predicted {min(start + len(batch_paths), n)}/{n}", flush=True)
            del xb, tensors, logits
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
    return np.concatenate(preds, axis=0).astype(np.float32)


def make_case_frame(
    split_meta: pd.DataFrame,
    y: np.ndarray,
    raw: np.ndarray,
    prob: np.ndarray,
    ood_values: np.ndarray,
    *,
    alias: str,
    model_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_id": split_meta["study_id"].astype(str).to_numpy(),
            "split": split_meta["split"].astype(str).to_numpy(),
            "y_attention": y.astype(int),
            "quality_score": split_meta["quality_score"].astype(float).to_numpy(),
            "critical_qa": split_meta["critical_qa"].astype(bool).to_numpy(),
            "qa_flags": split_meta["qa_flags"].astype(str).to_numpy(),
            "source_path": split_meta["source_path"].astype(str).to_numpy(),
            "image_eva_path": split_meta["image_eva_path"].astype(str).to_numpy(),
            f"raw_{alias}": raw.astype(float),
            f"p_{alias}": prob.astype(float),
            "ood_score": ood_values.astype(float),
            f"uncertainty_{alias}": (1.0 - np.abs(prob.astype(float) - 0.5) * 2.0),
            f"model_{alias}": model_name,
        }
    )


def load_router(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def route_for_case_frame(df: pd.DataFrame, alias: str, router: dict[str, Any]) -> pd.DataFrame:
    meta_cols = ["study_id", "quality_score", "critical_qa"]
    meta = df[meta_cols].copy()
    routes = route_decisions(
        df[f"p_{alias}"].to_numpy(),
        meta,
        t_negative=float(router["selected_T_negative"]),
        t_positive=float(router.get("selected_t_positive", 0.8)),
        t_quality=float(router.get("selected_t_quality", 0.35)),
        ood_score=df["ood_score"].to_numpy(),
        t_ood=float(router.get("selected_t_ood", 0.95)),
        t_uncertainty=float(router.get("selected_t_uncertainty", 0.65)),
    )
    out = df[["study_id", "split", "y_attention"]].copy()
    out = pd.concat([out, routes.drop(columns=["study_id"])], axis=1)
    out["alias"] = alias
    out["model"] = router["model"]
    out["threshold_policy"] = router["threshold_policy"]
    return out


def export_candidate(alias: str, cfg: dict[str, Any], meta: pd.DataFrame, parts: dict[str, Any], args: argparse.Namespace) -> None:
    model_name = cfg["model_name"]
    print(f"\n== Exporting {alias}: {model_name}", flush=True)
    model, info = load_candidate_model(cfg["checkpoint"], args.device)
    calibrator = joblib.load(cfg["calibrator"])
    router = load_router(cfg["router"])

    split_frames = []
    route_frames = []
    metric_rows = []
    for split in ["validation", "final_test"]:
        _, y_split, idx = parts[split]
        idx = np.asarray(idx, dtype=int)
        if args.limit and args.limit > 0:
            idx = idx[: int(args.limit)]
            y_split = y_split[: int(args.limit)]
        split_meta = meta.iloc[idx].reset_index(drop=True)
        raw = predict_image_paths(
            model,
            split_meta["image_eva_path"].astype(str).tolist(),
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            device=args.device,
        )
        prob = np.asarray(calibrator.transform(raw), dtype=np.float32)
        ood_values = args.ood_scores[idx]
        case_df = make_case_frame(split_meta, y_split, raw, prob, ood_values, alias=alias, model_name=model_name)
        split_frames.append(case_df)
        case_path = OUT_DIR / f"{alias}_{split}_case_scores.csv"
        case_df.to_csv(case_path, index=False)

        route_df = route_for_case_frame(case_df, alias, router)
        route_frames.append(route_df)
        route_path = OUT_DIR / f"{alias}_{split}_routes.csv"
        route_df.to_csv(route_path, index=False)

        metrics, _ = fixed_threshold_evaluation(
            f"{model_name} {split}",
            y_split,
            prob,
            split_meta,
            t_negative=float(router["selected_T_negative"]),
            t_positive=float(router.get("selected_t_positive", 0.8)),
            t_quality=float(router.get("selected_t_quality", 0.35)),
            t_uncertainty=float(router.get("selected_t_uncertainty", 0.65)),
            t_ood=float(router.get("selected_t_ood", 0.95)),
            ood_score_values=ood_values,
        )
        metrics.update(
            {
                "alias": alias,
                "split": split,
                "checkpoint": str(cfg["checkpoint"].relative_to(ROOT)),
                "calibrator": str(cfg["calibrator"].relative_to(ROOT)),
                "router": str(cfg["router"].relative_to(ROOT)),
                "variant": info["variant"],
            }
        )
        metric_rows.append(metrics)
        print(
            f"{alias} {split}: AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, auto-negative={metrics['auto_negative_coverage']:.2%}, "
            f"NPV={metrics['auto_negative_NPV']:.4f}, FN={metrics['unsafe_FN_auto_negative']}",
            flush=True,
        )

    pd.concat(split_frames, ignore_index=True).to_csv(OUT_DIR / f"{alias}_case_scores.csv", index=False)
    pd.concat(route_frames, ignore_index=True).to_csv(OUT_DIR / f"{alias}_routes.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / f"{alias}_fixed_router_metrics.csv", index=False)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    elif args.device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split smoke-test limit.")
    parser.add_argument("--candidates", nargs="+", default=["last1", "last2"], choices=sorted(CANDIDATES))
    args = parser.parse_args()
    args.device = choose_device(args.device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(RUN_ROOT / "artifacts" / "preprocessing_report.parquet")
    X = np.load(RUN_ROOT / "artifacts" / "embeddings" / "eva_x_base_features.npy")
    y = meta["y_attention"].astype(int).to_numpy()
    parts = split_arrays_by_meta(X, y, meta, require_all_splits=True)
    ood_model = joblib.load(RUN_ROOT / "artifacts" / "backend_bundle" / "best_ood_model.pkl")
    args.ood_scores = ood_score(ood_model, X)

    print(
        f"Exporting candidates={args.candidates}, device={args.device}, "
        f"batch_size={args.batch_size}, limit={args.limit or 'full'}",
        flush=True,
    )
    for alias in args.candidates:
        export_candidate(alias, CANDIDATES[alias], meta, parts, args)

    manifest = {
        "run_root": str(RUN_ROOT.relative_to(ROOT)),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "device": args.device,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "limit": args.limit,
        "candidates": args.candidates,
    }
    (OUT_DIR / "case_score_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Case score export complete:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
