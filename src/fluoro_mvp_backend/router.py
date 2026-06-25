from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .schemas import RouteDecision, ScorePayload


AUTO_NEGATIVE = "no_attention_required"
MANUAL_REVIEW = "N/A"
REQUIRES_ATTENTION = "requires_attention"


def load_router_config(bundle_dir: str | Path) -> dict[str, Any]:
    """Load the selected research router from a bundle directory."""

    bundle = Path(bundle_dir)
    path = bundle / "reports" / "selected_mass_router_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Router config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def selected_rule(config: dict[str, Any]) -> dict[str, Any]:
    rule = config.get("selected_validation_rule")
    if not isinstance(rule, dict):
        raise ValueError("Router config does not contain selected_validation_rule.")
    return rule


def route_payload(payload: ScorePayload, config: dict[str, Any]) -> RouteDecision:
    return route_record(payload.to_record(), config)


def route_record(record: dict[str, Any], config: dict[str, Any]) -> RouteDecision:
    """Apply the selected deployment router to one score record.

    The current production candidate is a conservative ensemble rule:
    CheXFound tuned head and EVA-X-B partial-unfreeze last1 must either both
    be low risk, or one must be very low while the other does not veto it.
    """

    rule = selected_rule(config)
    if rule.get("rule") != "pair_one_low_other_veto":
        raise NotImplementedError(f"Unsupported router rule: {rule.get('rule')!r}")

    model_a = str(rule["model_a"])
    model_b = str(rule["model_b"])
    p_a = _as_float(_field(record, model_a))
    p_b = _as_float(_field(record, model_b))
    t_a_negative = _as_float(rule["t_a_negative"])
    t_b_negative = _as_float(rule["t_b_negative"])
    t_a_veto = _as_float(rule["t_a_veto"])
    t_b_veto = _as_float(rule["t_b_veto"])
    t_ood_chex = _as_float(rule.get("t_ood_chex", math.inf))
    t_ood_eva = _as_float(rule.get("t_ood_eva", math.inf))
    t_quality = _as_float(rule.get("t_quality", 0.0))
    t_uncertainty = _as_float(rule.get("t_uncertainty", math.inf))

    thresholds = {
        "t_a_negative": t_a_negative,
        "t_b_negative": t_b_negative,
        "t_a_veto": t_a_veto,
        "t_b_veto": t_b_veto,
        "t_ood_chex": t_ood_chex,
        "t_ood_eva": t_ood_eva,
        "t_quality": t_quality,
        "t_uncertainty": t_uncertainty,
    }

    p_requires_attention = max(p_a, p_b)
    quality_score = _as_float(_field(record, "quality_score", 1.0))
    ood_chex = _as_float(_field(record, "ood_score_chex", 0.0))
    ood_eva = _as_float(_field(record, "ood_score_eva", 0.0))
    critical_qa = _as_bool(_field(record, "critical_qa_bool", _field(record, "critical_qa", False)))
    uncertainty = _uncertainty(record, model_a, model_b)

    if critical_qa:
        return RouteDecision(MANUAL_REVIEW, "critical_qa", p_requires_attention, False, thresholds)
    if quality_score < t_quality:
        return RouteDecision(MANUAL_REVIEW, "low_quality", p_requires_attention, False, thresholds)
    if ood_chex > t_ood_chex or ood_eva > t_ood_eva:
        return RouteDecision(MANUAL_REVIEW, "out_of_distribution", p_requires_attention, False, thresholds)
    if uncertainty > t_uncertainty:
        return RouteDecision(MANUAL_REVIEW, "high_uncertainty", p_requires_attention, False, thresholds)

    a_low_b_veto = p_a <= t_a_negative and p_b <= t_b_veto
    b_low_a_veto = p_b <= t_b_negative and p_a <= t_a_veto
    if a_low_b_veto or b_low_a_veto:
        return RouteDecision(AUTO_NEGATIVE, "pair_one_low_other_veto", p_requires_attention, True, thresholds)

    positive_threshold = _as_float(rule.get("t_positive", 0.8))
    if p_requires_attention >= positive_threshold:
        return RouteDecision(REQUIRES_ATTENTION, "suspicious_requires_attention", p_requires_attention, False, thresholds)
    return RouteDecision(MANUAL_REVIEW, "gray_zone", p_requires_attention, False, thresholds)


def route_dataframe(frame: Any, config: dict[str, Any]) -> Any:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        decision = route_record(record, config)
        out = dict(record)
        out.update(decision.to_dict())
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_routes(frame: Any) -> dict[str, Any]:
    auto = frame["route"].eq(AUTO_NEGATIVE)
    n = int(len(frame))
    selected_count = int(auto.sum())
    y = frame.get("y_attention")
    summary: dict[str, Any] = {
        "n": n,
        "selected_count": selected_count,
        "auto_negative_coverage": selected_count / n if n else 0.0,
    }
    if y is not None:
        y_auto = frame.loc[auto, "y_attention"].astype(int)
        fn_count = int((y_auto == 1).sum())
        tn_count = int((y_auto == 0).sum())
        npv = tn_count / selected_count if selected_count else float("nan")
        summary.update(
            {
                "TN_count": tn_count,
                "FN_count": fn_count,
                "NPV": npv,
                "NPV_ci95_low": wilson_lower_bound(tn_count, selected_count),
                "FN_per_1000_selected": fn_count / selected_count * 1000 if selected_count else float("nan"),
            }
        )
    return summary


def wilson_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    if n <= 0:
        return float("nan")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def _uncertainty(record: dict[str, Any], model_a: str, model_b: str) -> float:
    values = []
    for key in [f"uncertainty_{model_a}", f"uncertainty_{model_b}"]:
        if key in record:
            values.append(_as_float(record[key]))
    if not values and "uncertainty_core_max" in record:
        return _as_float(record["uncertainty_core_max"])
    if not values:
        for key in [model_a, model_b]:
            p = _as_float(record[key])
            values.append(1.0 - abs(p - 0.5) * 2.0)
    return max(values)


def _field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in record:
        return record[key]
    if default is not None:
        return default
    raise KeyError(f"Required score field is missing: {key}")


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float) and math.isnan(value):
        return value
    text = str(value).strip()
    if text == "":
        return float("nan")
    return float(text)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
