from pathlib import Path

import pandas as pd

from fluoro_mvp_backend.active_learning import build_retraining_plan, select_review_batch
from fluoro_mvp_backend.storage import FeedbackStore


def test_feedback_store_roundtrip(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite")
    payload = {"study_id": "demo", "p_chex_head": 0.01, "p_last1": 0.01}
    decision = {"route": "no_attention_required", "reason": "pair_one_low_other_veto", "p_requires_attention": 0.01}
    prediction_id = store.log_prediction(payload, decision)
    feedback_id = store.add_review_feedback("demo", 0, prediction_id=prediction_id, reviewer_id="qa")
    run_id = store.create_training_run("scheduled-refresh", {"profile": "eva_base"})
    store.update_training_run(run_id, "complete", metrics={"auroc": 0.94}, artifact_uri="s3://demo")

    assert prediction_id > 0
    assert feedback_id > 0
    assert run_id > 0


def test_active_learning_helpers():
    frame = pd.DataFrame(
        {
            "study_id": ["a", "b", "c"],
            "router_blocker_count": [1, 3, 2],
            "p_all_core_max": [0.3, 0.1, 0.2],
        }
    )
    selected = select_review_batch(frame, limit=2)
    assert selected["study_id"].tolist() == ["b", "c"]

    plan = build_retraining_plan(feedback_count=250, min_feedback_for_run=200)
    assert plan["ready_for_retraining"] is True
