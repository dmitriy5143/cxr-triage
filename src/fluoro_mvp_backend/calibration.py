from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Compatibility class for calibrators saved by the research notebooks."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self.model: Any | None = None
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

    def transform(self, p: np.ndarray | list[float] | float) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float32), 1e-5, 1 - 1e-5)
        if not self.ready or self.model is None:
            return p.astype(np.float32)
        if self.method == "isotonic":
            return np.asarray(self.model.predict(p), dtype=np.float32)
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1].astype(np.float32)


class PlattCalibrator(ProbabilityCalibrator):
    def __init__(self) -> None:
        super().__init__(method="platt")


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


def portable_calibrator_from_legacy(
    calibrator: Any,
    *,
    source_runtime: dict[str, str] | None = None,
) -> PortableCalibrator:
    """Extract only calibration mathematics from a notebook-era object."""

    method = str(getattr(calibrator, "method", "identity")).lower()
    ready = bool(getattr(calibrator, "ready", False))
    model = getattr(calibrator, "model", None)
    if not ready or model is None:
        return PortableCalibrator(1, "probability_calibrator", method, False, source_runtime=source_runtime)
    if method == "platt":
        coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        intercept = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
        classes = np.asarray(model.classes_).reshape(-1)
        if coef.size != 1 or intercept.size != 1 or classes.tolist() != [0, 1]:
            raise ValueError("Only binary [0, 1] Platt calibrators are supported.")
        return PortableCalibrator(
            1,
            "probability_calibrator",
            "platt",
            True,
            coef=(float(coef[0]),),
            intercept=float(intercept[0]),
            source_runtime=source_runtime,
        )
    if method == "isotonic":
        return PortableCalibrator(
            1,
            "probability_calibrator",
            "isotonic",
            True,
            x_thresholds=tuple(float(v) for v in np.asarray(model.X_thresholds_).reshape(-1)),
            y_thresholds=tuple(float(v) for v in np.asarray(model.y_thresholds_).reshape(-1)),
            source_runtime=source_runtime,
        )
    raise ValueError(f"Unsupported legacy calibration method: {method!r}")


def calibrate_probabilities(calibrator: Any, p: np.ndarray | list[float] | float) -> np.ndarray:
    """Calibrate without invoking notebook-serialized ``transform`` bytecode."""

    values = np.clip(np.asarray(p, dtype=np.float32), 1e-5, 1 - 1e-5)
    if calibrator is None or not bool(getattr(calibrator, "ready", False)):
        return values
    if isinstance(calibrator, PortableCalibrator):
        return calibrator.transform(values)

    model = getattr(calibrator, "model", None)
    if model is None:
        return values
    method = str(getattr(calibrator, "method", "platt")).lower()
    if method == "isotonic":
        return np.asarray(model.predict(values), dtype=np.float32)
    if method == "platt":
        logits = np.log(values / (1 - values)).reshape(-1, 1)
        return np.asarray(model.predict_proba(logits)[:, 1], dtype=np.float32)
    raise ValueError(f"Unsupported calibration method: {method!r}")


def install_research_pickle_shims() -> None:
    """Expose notebook-era class names before loading joblib/pickle artifacts."""

    shim = sys.modules.get("fluoro_mvp_core")
    if shim is None:
        shim = types.ModuleType("fluoro_mvp_core")
        sys.modules["fluoro_mvp_core"] = shim
    setattr(shim, "ProbabilityCalibrator", ProbabilityCalibrator)
    setattr(shim, "PlattCalibrator", PlattCalibrator)

    main = sys.modules.get("__main__")
    if main is not None:
        setattr(main, "ProbabilityCalibrator", ProbabilityCalibrator)
        setattr(main, "PlattCalibrator", PlattCalibrator)


def load_research_artifact(path: str | Path) -> Any:
    install_research_pickle_shims()
    return joblib.load(path)


def load_portable_calibrator(path: str | Path) -> PortableCalibrator:
    return PortableCalibrator.load(path)
