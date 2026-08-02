from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PortableCalibrator:
    """Version-neutral probability calibration used by production inference."""

    schema_version: int
    artifact_type: str
    method: str
    ready: bool
    coef: tuple[float, ...] = ()
    intercept: float = 0.0
    x_thresholds: tuple[float, ...] = ()
    y_thresholds: tuple[float, ...] = ()
    source_runtime: dict[str, str] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "PortableCalibrator":
        if payload.get("artifact_type") != "probability_calibrator":
            raise ValueError("Unsupported calibration artifact type.")
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported calibration artifact schema version.")
        return cls(
            schema_version=1,
            artifact_type="probability_calibrator",
            method=str(payload.get("method", "identity")).lower(),
            ready=bool(payload.get("ready", False)),
            coef=tuple(float(v) for v in payload.get("coef", [])),
            intercept=float(payload.get("intercept", 0.0)),
            x_thresholds=tuple(float(v) for v in payload.get("x_thresholds", [])),
            y_thresholds=tuple(float(v) for v in payload.get("y_thresholds", [])),
            source_runtime=dict(payload.get("source_runtime") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PortableCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["coef"] = list(self.coef)
        payload["x_thresholds"] = list(self.x_thresholds)
        payload["y_thresholds"] = list(self.y_thresholds)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def transform(self, p: np.ndarray | list[float] | float) -> np.ndarray:
        values = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1 - 1e-5)
        if not self.ready or self.method in {"none", "identity", "raw"}:
            return values.astype(np.float32)
        if self.method == "platt":
            if len(self.coef) != 1:
                raise ValueError("Platt calibrator must contain exactly one coefficient.")
            logits = np.log(values / (1 - values))
            decision = logits * self.coef[0] + self.intercept
            calibrated = 1.0 / (1.0 + np.exp(-np.clip(decision, -80.0, 80.0)))
            return calibrated.astype(np.float32)
        if self.method == "isotonic":
            if len(self.x_thresholds) < 2 or len(self.x_thresholds) != len(self.y_thresholds):
                raise ValueError("Isotonic calibrator thresholds are incomplete.")
            calibrated = np.interp(values, self.x_thresholds, self.y_thresholds)
            return calibrated.astype(np.float32)
        raise ValueError(f"Unsupported calibration method: {self.method!r}")


def calibrate_probabilities(calibrator: Any, p: np.ndarray | list[float] | float) -> np.ndarray:
    """Apply the version-neutral production calibration contract."""

    values = np.clip(np.asarray(p, dtype=np.float32), 1e-5, 1 - 1e-5)
    if calibrator is None:
        return values
    if isinstance(calibrator, PortableCalibrator):
        return calibrator.transform(values)
    raise TypeError("Production calibration accepts PortableCalibrator artifacts only.")


def load_portable_calibrator(path: str | Path) -> PortableCalibrator:
    return PortableCalibrator.load(path)
