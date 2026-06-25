from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScorePayload:
    """Model scores and QA signals needed by the deployment router."""

    p_chex_head: float
    p_last1: float
    study_id: str | None = None
    quality_score: float = 1.0
    ood_score_chex: float = 0.0
    ood_score_eva: float = 0.0
    critical_qa: bool = False
    y_attention: int | None = None
    image_file: str | None = None
    split: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ScorePayload":
        known = {
            "p_chex_head",
            "p_last1",
            "study_id",
            "quality_score",
            "ood_score_chex",
            "ood_score_eva",
            "critical_qa",
            "y_attention",
            "image_file",
            "split",
        }
        critical = data.get("critical_qa_bool", data.get("critical_qa", False))
        return cls(
            p_chex_head=float(data["p_chex_head"]),
            p_last1=float(data["p_last1"]),
            study_id=_optional_str(data.get("study_id")),
            quality_score=float(data.get("quality_score", 1.0)),
            ood_score_chex=float(data.get("ood_score_chex", 0.0)),
            ood_score_eva=float(data.get("ood_score_eva", 0.0)),
            critical_qa=_as_bool(critical),
            y_attention=_optional_int(data.get("y_attention")),
            image_file=_optional_str(data.get("image_file")),
            split=_optional_str(data.get("split")),
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        extra = record.pop("extra", {}) or {}
        record.update(extra)
        record["critical_qa_bool"] = self.critical_qa
        return record


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    p_requires_attention: float
    selected_by_rule: bool
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text.lower() in {"", "nan", "none"} else text


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "nan", "none"}:
        return None
    return int(float(text))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}
