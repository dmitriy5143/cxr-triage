from __future__ import annotations

import sys
import types
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
