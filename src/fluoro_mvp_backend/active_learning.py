from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_review_candidates(bundle_dir: str | Path) -> pd.DataFrame:
    path = Path(bundle_dir) / "reports" / "target_review_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Review-candidate table not found: {path}")
    return pd.read_csv(path)


def select_review_batch(candidates: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """Select cases for physician review from the research target-review table."""

    frame = candidates.copy()
    score_cols = [
        c
        for c in [
            "target_review_priority",
            "router_blocker_count",
            "p_all_core_max",
            "uncertainty_core_max",
        ]
        if c in frame.columns
    ]
    if score_cols:
        ascending = [False for _ in score_cols]
        frame = frame.sort_values(score_cols, ascending=ascending)
    return frame.head(limit).reset_index(drop=True)


def build_retraining_plan(
    feedback_count: int,
    min_feedback_for_run: int = 200,
    preferred_profile: str = "eva_base_partial_unfreeze_last1_refresh",
) -> dict[str, Any]:
    ready = feedback_count >= min_feedback_for_run
    return {
        "ready_for_retraining": ready,
        "feedback_count": feedback_count,
        "min_feedback_for_run": min_feedback_for_run,
        "recommended_profile": preferred_profile if ready else None,
    }
