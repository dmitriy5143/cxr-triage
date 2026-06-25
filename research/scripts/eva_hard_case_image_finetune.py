from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Hugging Face's Xet download backend can hang silently on some local networks.
# The regular HTTP path is slower but much easier to diagnose and resume.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fluoro_mvp_core import (  # noqa: E402
    PlattCalibrator,
    fixed_threshold_evaluation,
    image_to_eva_tensor,
    load_eva_end_to_end_checkpoint,
)


RUN_ROOT = ROOT / "fluoro_mvp_outputs" / "incxr_eva_base_partial_unfreeze_t4"
MERGED_DIR = ROOT / "selected_model_workbench" / "deep_router_with_chexfound_lora"
OUT_DIR = ROOT / "selected_model_workbench" / "eva_hard_case_image_finetune"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_device(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def image_tensor(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L")
    return image_to_eva_tensor(img, image_size=image_size)


class ImageRows(torch.utils.data.Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_size: int,
        weight_col: str | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.weight_col = weight_col

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        x = image_tensor(str(row["source_path"]), self.image_size)
        y = torch.tensor(float(row["y_attention"]), dtype=torch.float32)
        if self.weight_col:
            w = torch.tensor(float(row[self.weight_col]), dtype=torch.float32)
        else:
            w = torch.tensor(1.0, dtype=torch.float32)
        return x, y, w


def predict_logits(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    *,
    image_size: int,
    batch_size: int,
    device: str,
    label: str,
) -> np.ndarray:
    dataset = ImageRows(frame, image_size=image_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i, (xb, _, _) in enumerate(loader):
            xb = xb.to(device)
            logits = model(xb)
            out.append(logits.detach().float().cpu().numpy())
            if i == 0 or (i + 1) % 25 == 0:
                done = min((i + 1) * batch_size, len(frame))
                print(f"[predict:{label}] {done}/{len(frame)}", flush=True)
            del xb, logits
            clear_device(device)
    return np.concatenate(out).astype(np.float32)


def set_last_blocks_trainable(model: torch.nn.Module, n_last_blocks: int) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    blocks = getattr(model.encoder, "blocks", None)
    if blocks is None:
        raise RuntimeError("EVA encoder does not expose .blocks; cannot partially unfreeze.")
    for block in blocks[-int(n_last_blocks) :]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    frame["_sample_row_id"] = np.arange(len(frame))
    if n <= 0 or len(frame) <= n:
        return frame.sample(frac=1.0, random_state=seed).drop(columns=["_sample_row_id"]).reset_index(drop=True)
    per_class = max(1, n // 2)
    parts = []
    for label in [0, 1]:
        part = frame[frame["y_attention"].astype(int) == label]
        take = min(len(part), per_class)
        parts.append(part.sample(n=take, random_state=seed + label))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) < n:
        sampled_ids = set(sampled["_sample_row_id"].tolist())
        rest = frame[~frame["_sample_row_id"].isin(sampled_ids)]
        if len(rest):
            sampled = pd.concat(
                [sampled, rest.sample(n=min(len(rest), n - len(sampled)), random_state=seed + 7)],
                ignore_index=True,
            )
    return sampled.drop(columns=["_sample_row_id"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def mine_hard_case_training_frame(
    train_pool: pd.DataFrame,
    baseline_p: np.ndarray,
    *,
    max_train: int,
    seed: int,
) -> pd.DataFrame:
    frame = train_pool.copy().reset_index(drop=True)
    frame["baseline_p"] = baseline_p.astype(float)
    y = frame["y_attention"].astype(int).to_numpy()
    p = frame["baseline_p"].to_numpy()
    hard_positive = (y == 1) & (p < 0.35)
    very_hard_positive = (y == 1) & (p < 0.20)
    hard_negative = (y == 0) & (p > 0.65)
    boundary_negative = (y == 0) & (p > 0.45)
    frame["sample_weight"] = (
        1.0
        + 3.0 * hard_positive.astype(float)
        + 3.0 * very_hard_positive.astype(float)
        + 2.0 * hard_negative.astype(float)
        + 1.0 * boundary_negative.astype(float)
    )
    frame["hard_case_group"] = np.select(
        [very_hard_positive, hard_positive, hard_negative, boundary_negative],
        ["positive_very_low_score", "positive_low_score", "negative_high_score", "negative_boundary_score"],
        default="ordinary",
    )

    # Keep a deliberately hard-enriched, still class-balanced sample.
    hard = frame[frame["hard_case_group"] != "ordinary"]
    ordinary = frame[frame["hard_case_group"] == "ordinary"]
    hard_take = min(len(hard), max_train // 2)
    ordinary_take = max_train - hard_take
    selected = []
    if hard_take:
        selected.append(stratified_sample(hard, hard_take, seed + 1))
    if ordinary_take:
        selected.append(stratified_sample(ordinary, ordinary_take, seed + 2))
    if not selected:
        selected = [stratified_sample(frame, max_train, seed)]
    out = pd.concat(selected, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def train_weighted(
    model: torch.nn.Module,
    train_frame: pd.DataFrame,
    *,
    image_size: int,
    batch_size: int,
    device: str,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    epochs: int,
    max_steps: int,
) -> pd.DataFrame:
    params = [
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": lr_backbone},
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": lr_head},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
    dataset = ImageRows(train_frame, image_size=image_size, weight_col="sample_weight")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    history: list[dict[str, float]] = []
    step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        start_time = time.time()
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_vec = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (loss_vec * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step == 1 or step % 25 == 0:
                print(f"[train] epoch={epoch} step={step} loss={losses[-1]:.4f}", flush=True)
            del xb, yb, wb, logits, loss, loss_vec
            clear_device(device)
            if max_steps > 0 and step >= max_steps:
                break
        history.append(
            {
                "epoch": float(epoch),
                "steps_seen": float(step),
                "weighted_loss": float(np.mean(losses)) if losses else float("nan"),
                "seconds": float(time.time() - start_time),
            }
        )
        if max_steps > 0 and step >= max_steps:
            break
    model.eval()
    return pd.DataFrame(history)


def simple_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    return {
        "n": float(len(y)),
        "prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def vector_route_metrics(
    y: np.ndarray,
    p: np.ndarray,
    quality: np.ndarray,
    critical: np.ndarray,
    ood: np.ndarray,
    *,
    t_negative: float,
    t_positive: float,
    t_quality: float,
    t_uncertainty: float,
    t_ood: float,
) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    quality = np.asarray(quality).astype(float)
    critical = np.asarray(critical).astype(bool)
    ood = np.asarray(ood).astype(float)
    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0
    bad_quality = (quality < t_quality) | critical
    out_of_distribution = (~bad_quality) & (ood > t_ood)
    auto = (~bad_quality) & (~out_of_distribution) & (p <= t_negative)
    attention = (~bad_quality) & (~out_of_distribution) & (~auto) & (p >= t_positive)
    high_uncertainty = (~bad_quality) & (~out_of_distribution) & (~auto) & (~attention) & (uncertainty > t_uncertainty)
    manual = ~(auto | attention)
    fn = int(np.sum((y == 1) & auto))
    tn = int(np.sum((y == 0) & auto))
    fp_attention = int(np.sum((y == 0) & attention))
    return {
        **simple_metrics(y, p),
        "route_n": float(len(y)),
        "auto_negative_coverage": float(np.mean(auto)),
        "N/A_rate": float(np.mean(manual)),
        "requires_attention_rate": float(np.mean(attention)),
        "auto_negative_NPV": float(tn / max(tn + fn, 1)),
        "unsafe_FN_auto_negative": float(fn),
        "unsafe_FN_per_1000_auto_negative": float(fn / max(np.sum(auto), 1) * 1000.0),
        "workload_FP_requires_attention": float(fp_attention),
        "fixed_threshold_selected_count": int(np.sum(auto)),
        "fixed_threshold_TN_count": tn,
        "fixed_threshold_FN_count": fn,
        "fixed_threshold_NPV_ci95_low": float(wilson_lower_bound_local(tn, tn + fn)),
        "fixed_T_negative": float(t_negative),
        "bad_quality_count": int(np.sum(bad_quality)),
        "ood_count": int(np.sum(out_of_distribution)),
        "high_uncertainty_count": int(np.sum(high_uncertainty)),
    }


def wilson_lower_bound_local(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return float("nan")
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return float(max(0.0, (centre - margin) / denom))


def vector_routes(
    frame: pd.DataFrame,
    *,
    prefix: str,
    t_negative: float,
    t_positive: float,
    t_quality: float,
    t_uncertainty: float,
    t_ood: float,
) -> pd.DataFrame:
    p = frame[f"p_{prefix}"].astype(float).to_numpy()
    quality = frame["quality_score"].astype(float).to_numpy()
    critical = frame["critical_qa"].astype(bool).to_numpy()
    ood = frame["ood_score_eva"].astype(float).to_numpy()
    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0
    bad_quality = (quality < t_quality) | critical
    out_of_distribution = (~bad_quality) & (ood > t_ood)
    auto = (~bad_quality) & (~out_of_distribution) & (p <= t_negative)
    attention = (~bad_quality) & (~out_of_distribution) & (~auto) & (p >= t_positive)
    reason = np.full(len(frame), "gray_zone", dtype=object)
    route = np.full(len(frame), "N/A", dtype=object)
    reason[bad_quality] = "bad_quality_or_critical_qa"
    reason[out_of_distribution] = "out_of_distribution"
    reason[(~bad_quality) & (~out_of_distribution) & (~auto) & (~attention) & (uncertainty > t_uncertainty)] = "high_uncertainty"
    route[auto] = "no_attention_required"
    reason[auto] = "confident_no_attention_required"
    route[attention] = "requires_attention"
    reason[attention] = "suspicious_requires_attention"
    return pd.DataFrame(
        {
            "study_id": frame["study_id"].astype(str).to_numpy(),
            "y_attention": frame["y_attention"].astype(int).to_numpy(),
            "p_requires_attention": p,
            "quality_score": quality,
            "ood_score": ood,
            "uncertainty_score": uncertainty,
            "route": route,
            "reason": reason,
        }
    )


def load_ood_scores(split: str) -> pd.DataFrame:
    path = MERGED_DIR / f"merged_scores_{split}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["study_id", "ood_score_eva"])
    return pd.read_csv(path)[["study_id", "ood_score_eva"]].drop_duplicates("study_id")


def score_split_with_calibrator(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    calibrator: PlattCalibrator,
    *,
    split: str,
    image_size: int,
    batch_size: int,
    device: str,
    prefix: str,
) -> pd.DataFrame:
    logits = predict_logits(
        model,
        frame,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        label=f"{prefix}_{split}",
    )
    p_raw = 1.0 / (1.0 + np.exp(-logits))
    p_cal = calibrator.transform(p_raw)
    out = frame[["study_id", "split", "source_path", "y_attention", "quality_score", "critical_qa", "qa_flags"]].copy()
    out[f"logit_{prefix}"] = logits.astype(float)
    out[f"p_raw_{prefix}"] = p_raw.astype(float)
    out[f"p_{prefix}"] = p_cal.astype(float)
    ood = load_ood_scores(split)
    out = out.merge(ood, how="left", on="study_id")
    out["ood_score_eva"] = out["ood_score_eva"].fillna(0.0)
    return out


def tune_router(
    val_df: pd.DataFrame,
    *,
    prefix: str,
    target_npv: float,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    y = val_df["y_attention"].astype(int).to_numpy()
    p = val_df[f"p_{prefix}"].astype(float).to_numpy()
    quantiles = np.unique(np.quantile(p, np.linspace(0.001, 0.35, 220)))
    quality = val_df["quality_score"].astype(float).to_numpy()
    critical = val_df["critical_qa"].astype(bool).to_numpy()
    ood = val_df["ood_score_eva"].astype(float).to_numpy()
    rows = []
    for t_neg in quantiles:
        for t_ood in [0.95, 1.10, 1.25, 2.00]:
            for t_quality in [0.25, 0.35, 0.50]:
                for t_uncertainty in [0.50, 0.65, 0.80]:
                    row = vector_route_metrics(
                        y,
                        p,
                        quality,
                        critical,
                        ood,
                        t_negative=float(t_neg),
                        t_positive=0.80,
                        t_quality=float(t_quality),
                        t_uncertainty=float(t_uncertainty),
                        t_ood=float(t_ood),
                    )
                    row.update(
                        {
                            "selected_T_negative": float(t_neg),
                            "selected_t_ood": float(t_ood),
                            "selected_t_quality": float(t_quality),
                            "selected_t_uncertainty": float(t_uncertainty),
                        }
                    )
                    rows.append(row)
    sweep = pd.DataFrame(rows)
    sweep["passes_gate"] = (
        (sweep["fixed_threshold_FN_count"] == 0)
        & (sweep["auto_negative_NPV"] >= target_npv)
        & (sweep["fixed_threshold_selected_count"] >= 10)
    )
    ok = sweep[sweep["passes_gate"]].copy()
    if ok.empty:
        ok = sweep.sort_values(["fixed_threshold_FN_count", "fixed_threshold_selected_count"], ascending=[True, False]).head(1)
    else:
        ok = ok.sort_values(["fixed_threshold_selected_count", "auroc"], ascending=[False, False]).head(1)
    best = ok.iloc[0].to_dict()
    val_routes = vector_routes(
        val_df,
        prefix=prefix,
        t_negative=float(best["selected_T_negative"]),
        t_positive=0.80,
        t_quality=float(best["selected_t_quality"]),
        t_uncertainty=float(best["selected_t_uncertainty"]),
        t_ood=float(best["selected_t_ood"]),
    )
    return best, sweep, val_routes


def write_report(
    *,
    args: argparse.Namespace,
    train_frame: pd.DataFrame,
    history: pd.DataFrame,
    baseline_val_metrics: dict[str, float],
    baseline_test_metrics: dict[str, float],
    finetune_val_metrics: dict[str, float],
    finetune_test_metrics: dict[str, float],
    router_best: dict[str, float],
    final_metrics: dict[str, float],
    archive_path: Path,
) -> None:
    hard_case_mix = train_frame["hard_case_group"].value_counts().rename_axis("group").reset_index(name="n")
    hard_case_mix_md = hard_case_mix.to_string(index=False)
    report = f"""# EVA-X-B Image-Level Hard-Case Fine-Tune Pilot

Это короткая разведка, а не финальная production-модель. Цель: проверить, может ли
image-level дообучение на трудных случаях улучшить score-модель сильнее, чем
обычная настройка порогов.

## Setup

- Стартовый checkpoint: `{args.checkpoint}`
- Train pool for mining: `{args.mine_pool}` samples
- Fine-tune train sample: `{len(train_frame)}` samples
- Epochs / max steps: `{args.epochs}` / `{args.max_steps}`
- Batch size: `{args.batch_size}`
- Device: `{args.device}`
- Last EVA blocks trainable: `{args.n_last_blocks}`

## Hard-Case Mix

```text
{hard_case_mix_md}
```

## Baseline Checkpoint Metrics

| split | AUROC | AUPRC | Brier |
|---|---:|---:|---:|
| validation | {baseline_val_metrics['auroc']:.4f} | {baseline_val_metrics['auprc']:.4f} | {baseline_val_metrics['brier']:.4f} |
| final_test | {baseline_test_metrics['auroc']:.4f} | {baseline_test_metrics['auprc']:.4f} | {baseline_test_metrics['brier']:.4f} |

## Fine-Tuned Metrics

| split | AUROC | AUPRC | Brier |
|---|---:|---:|---:|
| validation | {finetune_val_metrics['auroc']:.4f} | {finetune_val_metrics['auprc']:.4f} | {finetune_val_metrics['brier']:.4f} |
| final_test | {finetune_test_metrics['auroc']:.4f} | {finetune_test_metrics['auprc']:.4f} | {finetune_test_metrics['brier']:.4f} |

## Selected Router On Validation

- `T_negative={router_best['selected_T_negative']:.6f}`
- `t_ood={router_best['selected_t_ood']:.2f}`
- `t_quality={router_best['selected_t_quality']:.2f}`
- `t_uncertainty={router_best['selected_t_uncertainty']:.2f}`
- validation selected: `{int(router_best['fixed_threshold_selected_count'])}`
- validation FN: `{int(router_best['fixed_threshold_FN_count'])}`

## Fixed Router On Final Test

- final selected: `{int(final_metrics['fixed_threshold_selected_count'])}`
- final auto-negative coverage: `{final_metrics['auto_negative_coverage']:.4%}`
- final FN: `{int(final_metrics['fixed_threshold_FN_count'])}`
- final NPV: `{final_metrics['auto_negative_NPV']:.4f}`
- final NPV CI95 low: `{final_metrics['fixed_threshold_NPV_ci95_low']:.4f}`

## Interpretation

Если fine-tuned модель не поднимает final-safe coverage относительно текущего
ансамбля 9.95%, значит проблема не в простом локальном дообучении последних
слоев, а в данных/таргете и более специализированном адаптере. Если поднимает,
эту ветку надо повторить уже полным протоколом в Colab с несколькими seed и
без ограничения `max_steps`.

Archive: `{archive_path}`
"""
    (OUT_DIR / "eva_hard_case_image_finetune_report_ru.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=RUN_ROOT / "checkpoints" / "base_unfreeze_last1_e150" / "best.pt")
    parser.add_argument("--meta", type=Path, default=RUN_ROOT / "artifacts" / "preprocessing_report.parquet")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mine-pool", type=int, default=1200)
    parser.add_argument("--max-train", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--n-last-blocks", type=int, default=1)
    parser.add_argument("--lr-backbone", type=float, default=2e-6)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-npv", type=float, default=0.99)
    parser.add_argument(
        "--resume-existing-scores",
        action="store_true",
        help="Finish router/report from existing hard_case score CSVs without rerunning EVA image inference.",
    )
    args = parser.parse_args()

    args.device = choose_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scores").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    cached_val_scores = OUT_DIR / "scores" / "hard_case_scores_validation.csv"
    cached_test_scores = OUT_DIR / "scores" / "hard_case_scores_final_test.csv"
    if args.resume_existing_scores and cached_val_scores.exists() and cached_test_scores.exists():
        print("Resuming from cached hard-case score CSVs.", flush=True)
        val_scores = pd.read_csv(cached_val_scores)
        test_scores = pd.read_csv(cached_test_scores)
        history_path = OUT_DIR / "reports" / "training_history.csv"
        sample_path = OUT_DIR / "reports" / "hard_case_train_sample.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        hard_train = pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame({"hard_case_group": ["cached"]})
        finetune_val_metrics = simple_metrics(val_scores["y_attention"].to_numpy(), val_scores["p_hard_case"].to_numpy())
        finetune_test_metrics = simple_metrics(test_scores["y_attention"].to_numpy(), test_scores["p_hard_case"].to_numpy())
        router_best, router_sweep, val_routes = tune_router(val_scores, prefix="hard_case", target_npv=args.target_npv)
        router_sweep.to_csv(OUT_DIR / "reports" / "hard_case_router_sweep_validation.csv", index=False)
        val_routes.to_csv(OUT_DIR / "reports" / "hard_case_routes_validation.csv", index=False)
        final_metrics = vector_route_metrics(
            test_scores["y_attention"].to_numpy(),
            test_scores["p_hard_case"].to_numpy(),
            test_scores["quality_score"].to_numpy(),
            test_scores["critical_qa"].to_numpy(),
            test_scores["ood_score_eva"].to_numpy(),
            t_negative=float(router_best["selected_T_negative"]),
            t_positive=0.80,
            t_quality=float(router_best["selected_t_quality"]),
            t_uncertainty=float(router_best["selected_t_uncertainty"]),
            t_ood=float(router_best["selected_t_ood"]),
        )
        final_routes = vector_routes(
            test_scores,
            prefix="hard_case",
            t_negative=float(router_best["selected_T_negative"]),
            t_positive=0.80,
            t_quality=float(router_best["selected_t_quality"]),
            t_uncertainty=float(router_best["selected_t_uncertainty"]),
            t_ood=float(router_best["selected_t_ood"]),
        )
        pd.DataFrame([final_metrics]).to_csv(OUT_DIR / "reports" / "hard_case_fixed_final_metrics.csv", index=False)
        final_routes.to_csv(OUT_DIR / "reports" / "hard_case_routes_final_test.csv", index=False)
        manifest = {
            "kind": "image_level_hard_case_finetune_pilot",
            "resumed_from_cached_scores": True,
            "args": vars(args),
            "finetune_validation": finetune_val_metrics,
            "finetune_final_test": finetune_test_metrics,
            "selected_router": router_best,
            "fixed_final": final_metrics,
        }
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        archive_path = OUT_DIR.parent / "eva_hard_case_image_finetune_export.zip"
        baseline_placeholder = {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
        write_report(
            args=args,
            train_frame=hard_train,
            history=history,
            baseline_val_metrics=baseline_placeholder,
            baseline_test_metrics=baseline_placeholder,
            finetune_val_metrics=finetune_val_metrics,
            finetune_test_metrics=finetune_test_metrics,
            router_best=router_best,
            final_metrics=final_metrics,
            archive_path=archive_path,
        )
        if archive_path.exists():
            archive_path.unlink()
        archive_path = Path(shutil.make_archive(str(archive_path.with_suffix("")), "zip", OUT_DIR))
        print("Fine-tuned validation:", finetune_val_metrics)
        print("Fine-tuned final:", finetune_test_metrics)
        print("Fixed final router:", final_metrics)
        print("Saved:", OUT_DIR)
        print("Archive:", archive_path)
        return

    meta = pd.read_parquet(args.meta)
    meta = meta[meta["source_path"].astype(str).map(lambda p: Path(p).exists())].reset_index(drop=True)
    for col in ["quality_score", "critical_qa", "qa_flags"]:
        if col not in meta:
            meta[col] = False if col == "critical_qa" else ""
    train_pool = stratified_sample(meta[meta["split"] == "train"], args.mine_pool, args.seed)
    validation = meta[meta["split"] == "validation"].reset_index(drop=True)
    final_test = meta[meta["split"] == "final_test"].reset_index(drop=True)

    print(f"Device: {args.device}", flush=True)
    print(f"Meta rows: {len(meta)} | train_pool={len(train_pool)} validation={len(validation)} final_test={len(final_test)}", flush=True)

    model_cache_dir = ROOT / ".model_cache" / "eva_hard_case"
    model, info = load_eva_end_to_end_checkpoint(str(model_cache_dir), args.checkpoint, device=args.device)
    counts = set_last_blocks_trainable(model, args.n_last_blocks)
    print("Trainable parameters:", counts, flush=True)

    baseline_train_logits = predict_logits(
        model,
        train_pool,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        label="baseline_train_pool",
    )
    baseline_train_p = 1.0 / (1.0 + np.exp(-baseline_train_logits))
    hard_train = mine_hard_case_training_frame(
        train_pool,
        baseline_train_p,
        max_train=args.max_train,
        seed=args.seed,
    )
    hard_train.to_csv(OUT_DIR / "reports" / "hard_case_train_sample.csv", index=False)

    baseline_val_logits = predict_logits(model, validation, image_size=args.image_size, batch_size=args.batch_size, device=args.device, label="baseline_validation")
    baseline_test_logits = predict_logits(model, final_test, image_size=args.image_size, batch_size=args.batch_size, device=args.device, label="baseline_final_test")
    baseline_cal = PlattCalibrator().fit(1.0 / (1.0 + np.exp(-baseline_val_logits)), validation["y_attention"].to_numpy())
    baseline_val_p = baseline_cal.transform(1.0 / (1.0 + np.exp(-baseline_val_logits)))
    baseline_test_p = baseline_cal.transform(1.0 / (1.0 + np.exp(-baseline_test_logits)))
    baseline_val_metrics = simple_metrics(validation["y_attention"].to_numpy(), baseline_val_p)
    baseline_test_metrics = simple_metrics(final_test["y_attention"].to_numpy(), baseline_test_p)

    history = train_weighted(
        model,
        hard_train,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    history.to_csv(OUT_DIR / "reports" / "training_history.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "source_checkpoint": str(args.checkpoint),
            "info": info,
            "hard_case_args": vars(args),
        },
        OUT_DIR / "checkpoints" / "eva_base_last1_hard_case_pilot.pt",
    )

    finetune_val_logits = predict_logits(model, validation, image_size=args.image_size, batch_size=args.batch_size, device=args.device, label="finetune_validation")
    finetune_cal = PlattCalibrator().fit(1.0 / (1.0 + np.exp(-finetune_val_logits)), validation["y_attention"].to_numpy())
    joblib.dump(finetune_cal, OUT_DIR / "checkpoints" / "eva_base_last1_hard_case_platt.pkl")
    val_scores = score_split_with_calibrator(
        model,
        validation,
        finetune_cal,
        split="validation",
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        prefix="hard_case",
    )
    test_scores = score_split_with_calibrator(
        model,
        final_test,
        finetune_cal,
        split="final_test",
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        prefix="hard_case",
    )
    val_scores.to_csv(OUT_DIR / "scores" / "hard_case_scores_validation.csv", index=False)
    test_scores.to_csv(OUT_DIR / "scores" / "hard_case_scores_final_test.csv", index=False)

    finetune_val_metrics = simple_metrics(val_scores["y_attention"].to_numpy(), val_scores["p_hard_case"].to_numpy())
    finetune_test_metrics = simple_metrics(test_scores["y_attention"].to_numpy(), test_scores["p_hard_case"].to_numpy())

    router_best, router_sweep, val_routes = tune_router(val_scores, prefix="hard_case", target_npv=args.target_npv)
    router_sweep.to_csv(OUT_DIR / "reports" / "hard_case_router_sweep_validation.csv", index=False)
    val_routes.to_csv(OUT_DIR / "reports" / "hard_case_routes_validation.csv", index=False)
    final_metrics = vector_route_metrics(
        test_scores["y_attention"].to_numpy(),
        test_scores["p_hard_case"].to_numpy(),
        test_scores["quality_score"].to_numpy(),
        test_scores["critical_qa"].to_numpy(),
        test_scores["ood_score_eva"].to_numpy(),
        t_negative=float(router_best["selected_T_negative"]),
        t_positive=0.80,
        t_quality=float(router_best["selected_t_quality"]),
        t_uncertainty=float(router_best["selected_t_uncertainty"]),
        t_ood=float(router_best["selected_t_ood"]),
    )
    final_routes = vector_routes(
        test_scores,
        prefix="hard_case",
        t_negative=float(router_best["selected_T_negative"]),
        t_positive=0.80,
        t_quality=float(router_best["selected_t_quality"]),
        t_uncertainty=float(router_best["selected_t_uncertainty"]),
        t_ood=float(router_best["selected_t_ood"]),
    )
    pd.DataFrame([final_metrics]).to_csv(OUT_DIR / "reports" / "hard_case_fixed_final_metrics.csv", index=False)
    final_routes.to_csv(OUT_DIR / "reports" / "hard_case_routes_final_test.csv", index=False)

    manifest = {
        "kind": "image_level_hard_case_finetune_pilot",
        "args": vars(args),
        "trainable_parameters": counts,
        "baseline_validation": baseline_val_metrics,
        "baseline_final_test": baseline_test_metrics,
        "finetune_validation": finetune_val_metrics,
        "finetune_final_test": finetune_test_metrics,
        "selected_router": router_best,
        "fixed_final": final_metrics,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    archive_path = OUT_DIR.parent / "eva_hard_case_image_finetune_export.zip"
    write_report(
        args=args,
        train_frame=hard_train,
        history=history,
        baseline_val_metrics=baseline_val_metrics,
        baseline_test_metrics=baseline_test_metrics,
        finetune_val_metrics=finetune_val_metrics,
        finetune_test_metrics=finetune_test_metrics,
        router_best=router_best,
        final_metrics=final_metrics,
        archive_path=archive_path,
    )
    if archive_path.exists():
        archive_path.unlink()
    archive_path = Path(shutil.make_archive(str(archive_path.with_suffix("")), "zip", OUT_DIR))
    print("Baseline validation:", baseline_val_metrics)
    print("Baseline final:", baseline_test_metrics)
    print("Fine-tuned validation:", finetune_val_metrics)
    print("Fine-tuned final:", finetune_test_metrics)
    print("Fixed final router:", final_metrics)
    print("Saved:", OUT_DIR)
    print("Archive:", archive_path)


if __name__ == "__main__":
    main()
