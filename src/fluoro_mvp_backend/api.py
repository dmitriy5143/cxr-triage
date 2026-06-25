from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .active_learning import load_review_candidates, select_review_batch
from .inference import ImageModelScoreProvider, predict_from_scores
from .storage import FeedbackStore


try:  # Keep FastAPI optional for non-API installs.
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment,misc]


class ScoreRequest(BaseModel):  # type: ignore[misc]
    scores: dict[str, Any]


class FeedbackRequest(BaseModel):  # type: ignore[misc]
    study_id: str
    label_attention: int
    prediction_id: int | None = None
    reviewer_id: str | None = None
    comment: str | None = None
    metadata: dict[str, Any] | None = None


class TrainingRunRequest(BaseModel):  # type: ignore[misc]
    run_name: str
    config: dict[str, Any]
    status: str = "planned"


class ImagePredictionRequest(BaseModel):  # type: ignore[misc]
    image_path: str


def create_app(bundle_dir: str | Path | None = None, db_path: str | Path | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Install fluoro-mvp-delivery[api] to use the FastAPI app.") from exc

    bundle = Path(bundle_dir or os.environ.get("FLUORO_BUNDLE_DIR", "model_bundle"))
    store = FeedbackStore(db_path or os.environ.get("FLUORO_DB_PATH") or bundle.parent / "feedback.sqlite")
    image_provider = ImageModelScoreProvider(bundle)
    app = FastAPI(title="Fluoro MVP Backend", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "bundle_dir": str(bundle), "db_path": str(store.db_path)}

    @app.get("/model/artifacts")
    def model_artifacts() -> dict[str, Any]:
        return {"status": "ok", "artifacts": image_provider.artifact_status()}

    @app.post("/predict-scores")
    def predict_scores(request: ScoreRequest) -> dict[str, Any]:
        decision = predict_from_scores(request.scores, bundle)
        prediction_id = store.log_prediction(request.scores, decision)
        decision["prediction_id"] = prediction_id
        return decision

    @app.post("/predict-image")
    def predict_image(request: ImagePredictionRequest) -> dict[str, Any]:
        try:
            scored = image_provider.score_image_with_metadata(request.image_path)
            decision = predict_from_scores(scored["scores"], bundle)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Image inference failed: {exc}") from exc
        prediction_id = store.log_prediction(scored["scores"], decision)
        decision["prediction_id"] = prediction_id
        decision["scores"] = scored["scores"]
        decision["preprocessing"] = scored["preprocessing"]
        return decision

    @app.post("/review-feedback")
    def review_feedback(request: FeedbackRequest) -> dict[str, Any]:
        feedback_id = store.add_review_feedback(
            study_id=request.study_id,
            label_attention=request.label_attention,
            prediction_id=request.prediction_id,
            reviewer_id=request.reviewer_id,
            comment=request.comment,
            metadata=request.metadata,
        )
        return {"feedback_id": feedback_id, "status": "stored"}

    @app.get("/db/stats")
    def db_stats() -> dict[str, Any]:
        return {"status": "ok", "counts": store.counts()}

    @app.get("/predictions")
    def predictions(limit: int = 100) -> dict[str, Any]:
        return {"items": store.list_predictions(limit=limit)}

    @app.get("/review-candidates")
    def review_candidates(limit: int = 50) -> dict[str, Any]:
        frame = select_review_batch(load_review_candidates(bundle), limit=limit)
        frame = frame.astype(object).where(frame.notna(), None)
        return {"items": frame.to_dict(orient="records")}

    @app.post("/training-runs")
    def create_training_run(request: TrainingRunRequest) -> dict[str, Any]:
        run_id = store.create_training_run(
            run_name=request.run_name,
            config=request.config,
            status=request.status,
        )
        return {"training_run_id": run_id, "status": "stored"}

    @app.get("/training-runs")
    def training_runs(limit: int = 100) -> dict[str, Any]:
        return {"items": store.list_training_runs(limit=limit)}

    return app
