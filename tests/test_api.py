import json
from pathlib import Path

from fastapi.testclient import TestClient

from fluoro_mvp_backend import api as api_module
from fluoro_mvp_backend.api import create_app


ROOT = Path(__file__).resolve().parents[1]


def test_api_prediction_feedback_and_training_flow(tmp_path):
    app = create_app(ROOT / "model_bundle", db_path=tmp_path / "api.sqlite")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    artifacts = client.get("/model/artifacts")
    assert artifacts.status_code == 200
    artifact_status = artifacts.json()["artifacts"]
    assert artifact_status["chexfound_hf_model_safetensors"]["exists"] is True
    assert artifact_status["eva_x_base_last1_checkpoint"]["exists"] is True
    assert artifact_status["eva_x_base_frozen_ood_weights"]["exists"] is True
    assert artifact_status["chexfound_external_code"]["exists"] is True
    assert artifact_status["eva_x_external_code"]["exists"] is True
    assert artifact_status["full_image_adapter_wired"] is True

    scores = json.loads((ROOT / "examples" / "demo_scores_auto_negative.json").read_text(encoding="utf-8"))
    pred = client.post("/predict-scores", json={"scores": scores})
    assert pred.status_code == 200
    pred_body = pred.json()
    assert pred_body["route"] == "no_attention_required"
    assert pred_body["prediction_id"] > 0

    feedback = client.post(
        "/review-feedback",
        json={
            "study_id": scores["study_id"],
            "prediction_id": pred_body["prediction_id"],
            "label_attention": 0,
            "reviewer_id": "test",
            "comment": "confirmed",
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["status"] == "stored"

    stats = client.get("/db/stats")
    assert stats.status_code == 200
    assert stats.json()["counts"] == {"predictions": 1, "review_feedback": 1, "training_runs": 0}

    candidates = client.get("/review-candidates?limit=3")
    assert candidates.status_code == 200
    assert len(candidates.json()["items"]) <= 3

    run = client.post(
        "/training-runs",
        json={"run_name": "scheduled-refresh", "config": {"profile": "eva_base"}, "status": "planned"},
    )
    assert run.status_code == 200
    assert run.json()["training_run_id"] > 0

    runs = client.get("/training-runs")
    assert runs.status_code == 200
    assert runs.json()["items"][0]["run_name"] == "scheduled-refresh"


def test_predict_image_endpoint_reports_missing_image_without_loading_models(tmp_path):
    app = create_app(ROOT / "model_bundle", db_path=tmp_path / "api.sqlite")
    client = TestClient(app)

    response = client.post("/predict-image", json={"image_path": "demo.png"})
    assert response.status_code == 400
    assert "Image was not found" in response.json()["detail"]


def test_predict_image_endpoint_routes_scored_image_payload(tmp_path, monkeypatch):
    demo_scores = json.loads((ROOT / "examples" / "demo_scores_auto_negative.json").read_text(encoding="utf-8"))

    def fake_score_image_with_metadata(self, image_path):
        return {
            "scores": demo_scores,
            "preprocessing": {
                "quality_score": demo_scores["quality_score"],
                "critical_qa": demo_scores["critical_qa_bool"],
                "qa_flags": [],
            },
        }

    monkeypatch.setattr(
        api_module.ImageModelScoreProvider,
        "score_image_with_metadata",
        fake_score_image_with_metadata,
    )
    image_path = tmp_path / "demo.png"
    image_path.write_bytes(b"placeholder")
    app = create_app(ROOT / "model_bundle", db_path=tmp_path / "api.sqlite")
    client = TestClient(app)

    response = client.post("/predict-image", json={"image_path": str(image_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "no_attention_required"
    assert body["scores"]["p_chex_head"] == demo_scores["p_chex_head"]
    assert body["prediction_id"] > 0
