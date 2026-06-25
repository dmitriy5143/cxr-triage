from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluoro_mvp_core import (  # noqa: E402
    ProbabilityCalibrator,
    expected_calibration_error,
    metrics_summary,
    predict_torch_mlp,
    route_metrics,
    train_torch_mlp,
    wilson_lower_bound,
)


CHEX_DIR = ROOT / "CheXFound_frozen"
EVA_CASE_DIR = ROOT / "selected_model_workbench" / "case_scores"
OUT_DIR = CHEX_DIR / "posthoc_head_sweep_workbench"


def chex_index_with_file() -> pd.DataFrame:
    index = pd.read_parquet(CHEX_DIR / "data_index.parquet")
    index["study_id"] = index["study_id"].astype(str)
    index["image_file"] = index["path"].map(lambda x: Path(str(x)).name)
    return index


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_chex_split(split: str) -> pd.DataFrame:
    path = CHEX_DIR / f"best_case_level_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "p_requires_attention": "p_chex",
            "uncertainty_score": "uncertainty_chex",
            "ood_score": "ood_score_chex",
        }
    )
    df["study_id"] = df["study_id"].astype(str)
    df = df.merge(chex_index_with_file()[["study_id", "image_file"]], on="study_id", how="left", validate="one_to_one")
    if df["image_file"].isna().any():
        raise ValueError(f"Could not attach image_file to CheXFound {split} rows.")
    df["critical_qa"] = False
    return df[
        [
            "study_id",
            "image_file",
            "split",
            "y_attention",
            "quality_score",
            "critical_qa",
            "ood_score_chex",
            "p_chex",
            "uncertainty_chex",
        ]
    ].copy()


def load_eva_split(alias: str, split: str) -> pd.DataFrame:
    path = EVA_CASE_DIR / f"{alias}_{split}_case_scores.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["study_id"] = df["study_id"].astype(str)
    df["image_file"] = df["source_path"].map(lambda x: Path(str(x)).name)
    return df[
        [
            "study_id",
            "image_file",
            f"p_{alias}",
            f"uncertainty_{alias}",
            "ood_score",
            "quality_score",
            "critical_qa",
        ]
    ].rename(columns={"ood_score": "ood_score_eva"})


def merge_scores(split: str) -> pd.DataFrame:
    chex = load_chex_split(split)
    last1 = load_eva_split("last1", split)
    last2 = load_eva_split("last2", split)
    df = chex.merge(
        last1[["image_file", "p_last1", "uncertainty_last1", "ood_score_eva"]],
        on="image_file",
        how="inner",
        validate="one_to_one",
    )
    df = df.merge(
        last2[["image_file", "p_last2", "uncertainty_last2"]],
        on="image_file",
        how="inner",
        validate="one_to_one",
    )
    if len(df) != len(chex):
        raise ValueError(f"Merge lost rows for {split}: chex={len(chex)} merged={len(df)}")
    df["p_eva_min"] = df[["p_last1", "p_last2"]].min(axis=1)
    df["p_eva_max"] = df[["p_last1", "p_last2"]].max(axis=1)
    df["p_eva_mean"] = df[["p_last1", "p_last2"]].mean(axis=1)
    df["p_all_max"] = df[["p_chex", "p_last1", "p_last2"]].max(axis=1)
    df["p_all_mean"] = df[["p_chex", "p_last1", "p_last2"]].mean(axis=1)
    return df


def base_mask(
    df: pd.DataFrame,
    *,
    score_cols: list[str],
    t_ood_chex: float,
    t_ood_eva: float,
    t_quality: float,
    t_uncertainty: float,
) -> np.ndarray:
    uncertainty = np.maximum.reduce(
        [1.0 - np.abs(df[col].to_numpy(float) - 0.5) * 2.0 for col in score_cols]
    )
    return (
        (df["quality_score"].to_numpy(float) >= t_quality)
        & (~df["critical_qa"].astype(bool).to_numpy())
        & (df["ood_score_chex"].to_numpy(float) <= t_ood_chex)
        & (df["ood_score_eva"].to_numpy(float) <= t_ood_eva)
        & (uncertainty <= t_uncertainty)
    )


def metrics_for_mask(
    df: pd.DataFrame,
    mask: np.ndarray,
    score_col: str,
    *,
    include_score_metrics: bool = False,
) -> dict[str, Any]:
    y = df["y_attention"].to_numpy(int)
    p = df[score_col].to_numpy(float)
    mask = np.asarray(mask, dtype=bool)
    n_sel = int(mask.sum())
    fn = int(((y == 1) & mask).sum())
    tn = int(((y == 0) & mask).sum())
    out = {
        "n": int(len(df)),
        "selected_count": n_sel,
        "auto_negative_coverage": float(n_sel / max(len(df), 1)),
        "TN_count": tn,
        "FN_count": fn,
        "NPV": float(tn / max(tn + fn, 1)),
        "NPV_ci95_low": wilson_lower_bound(tn, tn + fn, z=1.96),
        "FN_per_1000_selected": float(fn / max(n_sel, 1) * 1000.0),
    }
    if include_score_metrics:
        out.update(
            {
                "auroc": safe_auc(y, p),
                "auprc": safe_auprc(y, p),
                "brier": float(brier_score_loss(y, p)),
                "ece": expected_calibration_error(y, p),
            }
        )
    return out


def add_route_columns(df: pd.DataFrame, mask: np.ndarray, score_col: str, reason: str) -> pd.DataFrame:
    out = df[["study_id", "split", "y_attention", "quality_score", "ood_score_chex", "ood_score_eva"]].copy()
    out["p_requires_attention"] = df[score_col].to_numpy(float)
    out["route"] = np.where(mask, "no_attention_required", "N/A")
    out["reason"] = np.where(mask, reason, "not_auto_negative")
    return out


def make_thresholds(df: pd.DataFrame, col: str, low: float = 0.005, high: float = 0.18, n: int = 36) -> np.ndarray:
    qs = np.linspace(low, high, n)
    values = np.unique(np.quantile(df[col].to_numpy(float), qs))
    return values[np.isfinite(values)]


def single_chex_grid(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    thresholds = make_thresholds(df, "p_chex", low=0.002, high=0.18, n=90)
    for t, t_ood_chex, t_quality, t_unc in product(
        thresholds,
        [0.90, 0.95, 1.05, 1.10, 1.25, 1.50, 2.0],
        [0.25, 0.35, 0.45],
        [0.50, 0.65, 0.80, 1.00],
    ):
        qmask = base_mask(
            df,
            score_cols=["p_chex"],
            t_ood_chex=t_ood_chex,
            t_ood_eva=2.0,
            t_quality=t_quality,
            t_uncertainty=t_unc,
        )
        mask = qmask & (df["p_chex"].to_numpy(float) <= float(t))
        row = metrics_for_mask(df, mask, "p_chex")
        row.update(
            {
                "split": split,
                "rule": "single_chex_threshold",
                "score": "p_chex",
                "t_chex_negative": float(t),
                "t_eva_negative": math.nan,
                "t_chex_veto": math.nan,
                "t_eva_veto": math.nan,
                "t_ood_chex": t_ood_chex,
                "t_ood_eva": math.nan,
                "t_quality": t_quality,
                "t_uncertainty": t_unc,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def ensemble_grid(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    chex_thresholds = make_thresholds(df, "p_chex", low=0.005, high=0.16, n=18)
    eva_score_options = ["p_last1", "p_last2", "p_eva_min", "p_eva_max"]
    veto_chex_options = [0.05, 0.08, 0.12, 0.20]
    veto_eva_options = [0.05, 0.08, 0.12, 0.20]
    t_ood_chex_options = [1.10, 1.25, 1.50]
    t_ood_eva_options = [1.10, 1.25, 1.50]
    t_unc_options = [0.65, 0.80, 1.00]

    for eva_col in eva_score_options:
        eva_thresholds = make_thresholds(df, eva_col, low=0.005, high=0.16, n=16)
        score_col = "p_all_max" if eva_col != "p_eva_min" else "p_all_mean"
        for t_chex, t_eva, t_ood_chex, t_ood_eva, t_unc in product(
            chex_thresholds,
            eva_thresholds,
            t_ood_chex_options,
            t_ood_eva_options,
            t_unc_options,
        ):
            qmask = base_mask(
                df,
                score_cols=["p_chex", eva_col],
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=0.35,
                t_uncertainty=t_unc,
            )
            mask = qmask & (df["p_chex"].to_numpy(float) <= t_chex) & (df[eva_col].to_numpy(float) <= t_eva)
            row = metrics_for_mask(df, mask, score_col)
            row.update(
                {
                    "split": split,
                    "rule": f"chex_and_{eva_col}_both_low",
                    "score": score_col,
                    "eva_score_col": eva_col,
                    "t_chex_negative": float(t_chex),
                    "t_eva_negative": float(t_eva),
                    "t_chex_veto": math.nan,
                    "t_eva_veto": math.nan,
                    "t_ood_chex": t_ood_chex,
                    "t_ood_eva": t_ood_eva,
                    "t_quality": 0.35,
                    "t_uncertainty": t_unc,
                }
            )
            rows.append(row)

        for t_chex, t_eva, veto_chex, veto_eva, t_ood_chex, t_ood_eva, t_unc in product(
            chex_thresholds[::2],
            eva_thresholds[::2],
            veto_chex_options,
            veto_eva_options,
            [1.10, 1.50],
            [1.10, 1.50],
            [0.80, 1.00],
        ):
            qmask = base_mask(
                df,
                score_cols=["p_chex", eva_col],
                t_ood_chex=t_ood_chex,
                t_ood_eva=t_ood_eva,
                t_quality=0.35,
                t_uncertainty=t_unc,
            )
            chex_low_eva_no_veto = (df["p_chex"].to_numpy(float) <= t_chex) & (df[eva_col].to_numpy(float) <= veto_eva)
            eva_low_chex_no_veto = (df[eva_col].to_numpy(float) <= t_eva) & (df["p_chex"].to_numpy(float) <= veto_chex)
            mask = qmask & (chex_low_eva_no_veto | eva_low_chex_no_veto)
            row = metrics_for_mask(df, mask, score_col)
            row.update(
                {
                    "split": split,
                    "rule": f"chex_{eva_col}_one_low_other_veto",
                    "score": score_col,
                    "eva_score_col": eva_col,
                    "t_chex_negative": float(t_chex),
                    "t_eva_negative": float(t_eva),
                    "t_chex_veto": float(veto_chex),
                    "t_eva_veto": float(veto_eva),
                    "t_ood_chex": t_ood_chex,
                    "t_ood_eva": t_ood_eva,
                    "t_quality": 0.35,
                    "t_uncertainty": t_unc,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def validation_posthoc_sweep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val = merge_scores("validation")
    final = merge_scores("final_test")
    val.to_csv(OUT_DIR / "merged_chex_eva_validation.csv", index=False)
    final.to_csv(OUT_DIR / "merged_chex_eva_final_test.csv", index=False)

    print("Running CheXFound single/router grid ...", flush=True)
    val_single = single_chex_grid(val, "validation")
    print("Running CheXFound+EVA ensemble grid ...", flush=True)
    val_ensemble = ensemble_grid(val, "validation")
    val_grid = pd.concat([val_single, val_ensemble], ignore_index=True)
    for score_col in sorted(val_grid["score"].dropna().astype(str).unique()):
        if score_col not in val.columns:
            continue
        mask = val_grid["score"].astype(str).eq(score_col)
        y_val = val["y_attention"].to_numpy(int)
        p_val = val[score_col].to_numpy(float)
        val_grid.loc[mask, "auroc"] = safe_auc(y_val, p_val)
        val_grid.loc[mask, "auprc"] = safe_auprc(y_val, p_val)
        val_grid.loc[mask, "brier"] = float(brier_score_loss(y_val, p_val))
        val_grid.loc[mask, "ece"] = expected_calibration_error(y_val, p_val)
    val_grid["safe_zero_fn"] = val_grid["FN_count"].eq(0) & val_grid["NPV"].ge(0.99) & val_grid["selected_count"].ge(10)
    val_grid.to_csv(OUT_DIR / "validation_chex_posthoc_grid.csv", index=False)

    safe = val_grid[val_grid["safe_zero_fn"]].copy()
    safe = safe.sort_values(
        ["auto_negative_coverage", "NPV_ci95_low", "auroc", "auprc"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    safe.to_csv(OUT_DIR / "validation_chex_posthoc_safe_rules.csv", index=False)

    final_rows: list[dict[str, Any]] = []
    for _, rule in safe.head(100).iterrows():
        mask = apply_posthoc_rule(final, rule)
        row = metrics_for_mask(final, mask, str(rule["score"]), include_score_metrics=True)
        rule_dict = rule.to_dict()
        for key, value in rule_dict.items():
            if key in {"rule", "score", "eva_score_col"} or str(key).startswith("t_"):
                row[key] = value
            else:
                row[f"validation_{key}"] = value
        row["split"] = "final_test"
        row["validation_auto_negative_coverage"] = float(rule["auto_negative_coverage"])
        row["validation_selected_count"] = int(rule["selected_count"])
        row["validation_FN_count"] = int(rule["FN_count"])
        row["validation_NPV"] = float(rule["NPV"])
        final_rows.append(row)
    final_results = pd.DataFrame(final_rows)
    if not final_results.empty:
        final_results = final_results.sort_values(
            ["FN_count", "auto_negative_coverage", "NPV_ci95_low", "auroc"],
            ascending=[True, False, False, False],
            na_position="last",
        )
    final_results.to_csv(OUT_DIR / "final_test_fixed_chex_posthoc_results.csv", index=False)
    return val_grid, safe, final_results


def apply_posthoc_rule(df: pd.DataFrame, rule: pd.Series) -> np.ndarray:
    rule_name = str(rule["rule"])
    score = str(rule.get("eva_score_col", ""))
    t_ood_eva = 2.0 if pd.isna(rule.get("t_ood_eva")) else float(rule.get("t_ood_eva"))
    qmask = base_mask(
        df,
        score_cols=["p_chex"] if rule_name == "single_chex_threshold" else ["p_chex", score],
        t_ood_chex=float(rule["t_ood_chex"]),
        t_ood_eva=t_ood_eva,
        t_quality=float(rule["t_quality"]),
        t_uncertainty=float(rule["t_uncertainty"]),
    )
    if rule_name == "single_chex_threshold":
        return qmask & (df["p_chex"].to_numpy(float) <= float(rule["t_chex_negative"]))
    if "both_low" in rule_name:
        return (
            qmask
            & (df["p_chex"].to_numpy(float) <= float(rule["t_chex_negative"]))
            & (df[score].to_numpy(float) <= float(rule["t_eva_negative"]))
        )
    if "one_low_other_veto" in rule_name:
        chex_low_eva_no_veto = (
            (df["p_chex"].to_numpy(float) <= float(rule["t_chex_negative"]))
            & (df[score].to_numpy(float) <= float(rule["t_eva_veto"]))
        )
        eva_low_chex_no_veto = (
            (df[score].to_numpy(float) <= float(rule["t_eva_negative"]))
            & (df["p_chex"].to_numpy(float) <= float(rule["t_chex_veto"]))
        )
        return qmask & (chex_low_eva_no_veto | eva_low_chex_no_veto)
    raise ValueError(f"Unsupported rule: {rule_name}")


def calibrator_options(raw_calib: np.ndarray, y_calib: np.ndarray, raw_val: np.ndarray, y_val: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for method in ["platt", "isotonic", "none"]:
        cal = ProbabilityCalibrator(method).fit(raw_calib, y_calib)
        p_val = cal.transform(raw_val)
        row = metrics_summary(y_val, p_val)
        row.update({"calibration_method": method, "calibrator": cal, "p_val": p_val})
        rows.append(row)
    return rows


def best_single_router_for_scores(
    base_df: pd.DataFrame,
    p: np.ndarray,
    *,
    split: str,
    score_name: str,
    target_npv: float = 0.99,
) -> tuple[pd.DataFrame, pd.Series | None]:
    df = base_df.copy()
    df[score_name] = np.asarray(p, dtype=float)
    if "ood_score_eva" not in df.columns:
        df["ood_score_eva"] = 0.0
    route_df = df.copy()
    route_df["p_chex"] = df[score_name].to_numpy(float)
    rows = []
    thresholds = make_thresholds(df, score_name, low=0.002, high=0.18, n=120)
    for t, t_ood, t_unc in product(thresholds, [0.95, 1.10, 1.25, 1.50], [0.50, 0.65, 0.80, 1.00]):
        qmask = base_mask(
            route_df,
            score_cols=["p_chex"],
            t_ood_chex=t_ood,
            t_ood_eva=2.0,
            t_quality=0.35,
            t_uncertainty=t_unc,
        )
        mask = qmask & (route_df["p_chex"].to_numpy(float) <= float(t))
        row = metrics_for_mask(route_df, mask, "p_chex")
        row.update(
            {
                "split": split,
                "score_name": score_name,
                "t_negative": float(t),
                "t_ood_chex": t_ood,
                "t_uncertainty": t_unc,
                "t_quality": 0.35,
            }
        )
        rows.append(row)
    sweep = pd.DataFrame(rows)
    safe = sweep[sweep["FN_count"].eq(0) & sweep["NPV"].ge(target_npv) & sweep["selected_count"].ge(10)].copy()
    if safe.empty:
        return sweep, None
    selected = safe.sort_values(
        ["auto_negative_coverage", "NPV_ci95_low", "selected_count"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]
    return sweep, selected


def load_feature_splits() -> dict[str, Any]:
    X = np.load(CHEX_DIR / "chexfound_frozen_features.npy", mmap_mode="r")
    index = chex_index_with_file()
    y = index["y_attention"].to_numpy(int)
    splits: dict[str, Any] = {"X": X, "index": index, "y": y}
    for split in ["train", "calibration", "validation", "final_test"]:
        idx = np.flatnonzero(index["split"].astype(str).to_numpy() == split)
        splits[split] = (idx, y[idx])
    return splits


def run_head_sweep(device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_feature_splits()
    X = splits["X"]
    train_idx, y_train = splits["train"]
    calib_idx, y_calib = splits["calibration"]
    val_idx, y_val = splits["validation"]
    test_idx, y_test = splits["final_test"]

    X_train = np.asarray(X[train_idx], dtype=np.float32)
    X_calib = np.asarray(X[calib_idx], dtype=np.float32)
    X_val = np.asarray(X[val_idx], dtype=np.float32)
    X_test = np.asarray(X[test_idx], dtype=np.float32)

    val_order = splits["index"][["study_id", "image_file"]].iloc[val_idx].reset_index(drop=True)
    test_order = splits["index"][["study_id", "image_file"]].iloc[test_idx].reset_index(drop=True)
    val_base = val_order.merge(load_chex_split("validation"), on=["study_id", "image_file"], how="left")
    test_base = test_order.merge(load_chex_split("final_test"), on=["study_id", "image_file"], how="left")
    if val_base[["quality_score", "ood_score_chex"]].isna().any().any():
        raise ValueError("Could not align validation case metadata for head sweep.")
    if test_base[["quality_score", "ood_score_chex"]].isna().any().any():
        raise ValueError("Could not align final-test case metadata for head sweep.")

    configs = [
        {"name": "h256_do10_lr8e4_wd1e4", "hidden": 256, "dropout": 0.10, "lr": 8e-4, "weight_decay": 1e-4, "epochs": 100},
        {"name": "h256_do20_lr8e4_wd1e4", "hidden": 256, "dropout": 0.20, "lr": 8e-4, "weight_decay": 1e-4, "epochs": 100},
        {"name": "h256_do35_lr8e4_wd3e4", "hidden": 256, "dropout": 0.35, "lr": 8e-4, "weight_decay": 3e-4, "epochs": 100},
        {"name": "h512_do10_lr3e4_wd1e4", "hidden": 512, "dropout": 0.10, "lr": 3e-4, "weight_decay": 1e-4, "epochs": 120},
        {"name": "h512_do20_lr3e4_wd1e4", "hidden": 512, "dropout": 0.20, "lr": 3e-4, "weight_decay": 1e-4, "epochs": 120},
        {"name": "h512_do35_lr3e4_wd3e4", "hidden": 512, "dropout": 0.35, "lr": 3e-4, "weight_decay": 3e-4, "epochs": 120},
        {"name": "h512_do20_lr8e4_wd1e4", "hidden": 512, "dropout": 0.20, "lr": 8e-4, "weight_decay": 1e-4, "epochs": 100},
        {"name": "h384_do20_lr5e4_wd1e4", "hidden": 384, "dropout": 0.20, "lr": 5e-4, "weight_decay": 1e-4, "epochs": 110},
    ]

    rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    model_dir = OUT_DIR / "head_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        print("Training CheXFound frozen head:", cfg, flush=True)
        model = train_torch_mlp(
            X_train,
            y_train,
            X_calib,
            y_calib,
            hidden=int(cfg["hidden"]),
            dropout=float(cfg["dropout"]),
            epochs=int(cfg["epochs"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            seed=42,
            device=device,
        )
        raw_calib = predict_torch_mlp(model, X_calib, device=device)
        raw_val = predict_torch_mlp(model, X_val, device=device)
        raw_test = predict_torch_mlp(model, X_test, device=device)

        cal_rows = calibrator_options(raw_calib, y_calib, raw_val, y_val)
        cal_table = pd.DataFrame([{k: v for k, v in row.items() if k not in {"calibrator", "p_val"}} for row in cal_rows])
        cal_table = cal_table.sort_values(["brier", "ece", "auroc", "auprc"], ascending=[True, True, False, False])
        selected_method = str(cal_table.iloc[0]["calibration_method"])
        selected = next(row for row in cal_rows if row["calibration_method"] == selected_method)
        calibrator = selected["calibrator"]
        p_val = np.asarray(selected["p_val"], dtype=np.float32)
        p_test = np.asarray(calibrator.transform(raw_test), dtype=np.float32)

        val_sweep, val_router = best_single_router_for_scores(
            val_base,
            p_val,
            split="validation",
            score_name="p_head",
        )
        val_sweep.to_csv(OUT_DIR / f"head_{cfg['name']}_validation_router_sweep.csv", index=False)

        val_metrics = metrics_summary(y_val, p_val)
        val_metrics.update(cfg)
        val_metrics["calibration_method"] = selected_method
        val_metrics["model_name"] = f"chexfound_head_{cfg['name']}"
        if val_router is not None:
            for key in [
                "auto_negative_coverage",
                "selected_count",
                "TN_count",
                "FN_count",
                "NPV",
                "NPV_ci95_low",
                "t_negative",
                "t_ood_chex",
                "t_uncertainty",
            ]:
                val_metrics[f"router_validation_{key}"] = val_router.get(key)
            test_mask = apply_head_router(test_base, p_test, val_router)
            test_eval = metrics_for_head(test_base, p_test, test_mask)
            test_eval.update(
                {
                    "model_name": val_metrics["model_name"],
                    "calibration_method": selected_method,
                    "selected_from_validation_auto_negative_coverage": val_router["auto_negative_coverage"],
                    "selected_from_validation_FN_count": val_router["FN_count"],
                }
            )
            final_rows.append(test_eval)
        rows.append(val_metrics)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "scaler": getattr(model, "scaler", None),
                "config": cfg,
                "calibration_method": selected_method,
            },
            model_dir / f"{cfg['name']}.pt",
        )
        joblib.dump(calibrator, model_dir / f"{cfg['name']}_{selected_method}_calibrator.pkl")

    val_results = pd.DataFrame(rows).sort_values(
        ["auroc", "auprc", "brier", "router_validation_auto_negative_coverage"],
        ascending=[False, False, True, False],
        na_position="last",
    )
    final_results = pd.DataFrame(final_rows)
    if not final_results.empty:
        final_results = final_results.sort_values(
            ["FN_count", "auroc", "auprc", "auto_negative_coverage"],
            ascending=[True, False, False, False],
            na_position="last",
        )
    val_results.to_csv(OUT_DIR / "chexfound_frozen_head_sweep_validation.csv", index=False)
    final_results.to_csv(OUT_DIR / "chexfound_frozen_head_sweep_final_test.csv", index=False)
    return val_results, final_results


def apply_head_router(df: pd.DataFrame, p: np.ndarray, router: pd.Series) -> np.ndarray:
    tmp = df.copy()
    tmp["p_chex"] = np.asarray(p, dtype=float)
    if "ood_score_eva" not in tmp.columns:
        tmp["ood_score_eva"] = 0.0
    return base_mask(
        tmp,
        score_cols=["p_chex"],
        t_ood_chex=float(router["t_ood_chex"]),
        t_ood_eva=2.0,
        t_quality=float(router["t_quality"]),
        t_uncertainty=float(router["t_uncertainty"]),
    ) & (tmp["p_chex"].to_numpy(float) <= float(router["t_negative"]))


def metrics_for_head(df: pd.DataFrame, p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    tmp = df.copy()
    tmp["p_chex"] = np.asarray(p, dtype=float)
    return metrics_for_mask(tmp, mask, "p_chex", include_score_metrics=True)


def write_report(
    posthoc_safe: pd.DataFrame,
    posthoc_final: pd.DataFrame,
    head_val: pd.DataFrame,
    head_final: pd.DataFrame,
) -> None:
    report_path = OUT_DIR / "chexfound_posthoc_head_sweep_report.md"
    lines = [
        "# CheXFound Posthoc Router and Frozen Head Sweep",
        "",
        "This workbench uses already exported CheXFound frozen artifacts. No backbone retraining or feature extraction is performed.",
        "",
        "## Best Validation-Safe Posthoc Rules",
        "",
        posthoc_safe.head(15).to_markdown(index=False) if not posthoc_safe.empty else "No safe posthoc rule found.",
        "",
        "## Fixed Final-Test Results for Top Validation Rules",
        "",
        posthoc_final.head(15).to_markdown(index=False) if not posthoc_final.empty else "No final-test rows.",
        "",
        "## Frozen Feature Head Sweep: Validation",
        "",
        head_val.head(15).to_markdown(index=False) if not head_val.empty else "Head sweep did not run.",
        "",
        "## Frozen Feature Head Sweep: Fixed Final Test",
        "",
        head_final.head(15).to_markdown(index=False) if not head_final.empty else "Head sweep final-test table is empty.",
        "",
        "## Interpretation",
        "",
        "- A posthoc rule is useful only if it is selected on validation and keeps FN=0 on final test.",
        "- Head-sweep candidates are selected by ranking quality first, then checked with validation-selected routers.",
        "- If CheXFound improves ranking but not safe auto-negative coverage, it is best used as a consensus/veto model with EVA.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", report_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Output:", OUT_DIR)
    val_grid, posthoc_safe, posthoc_final = validation_posthoc_sweep()
    device = choose_device()
    print("Head sweep device:", device)
    head_val, head_final = run_head_sweep(device)
    write_report(posthoc_safe, posthoc_final, head_val, head_final)
    manifest = {
        "posthoc_validation_grid_rows": int(len(val_grid)),
        "posthoc_safe_rows": int(len(posthoc_safe)),
        "head_sweep_rows": int(len(head_val)),
        "device": device,
        "inputs": {
            "chexfound_case_scores": str(CHEX_DIR),
            "eva_case_scores": str(EVA_CASE_DIR),
            "chexfound_features": str(CHEX_DIR / "chexfound_frozen_features.npy"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
