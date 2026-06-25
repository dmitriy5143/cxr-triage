# Plan Completion Audit

## Status

The delivery package is ready as a backend MVP package and research handoff folder.

It now includes the direct image scorer adapter that wires preprocessing into both heavy backbones and emits the two routed scores:

- `p_chex_head`
- `p_last1`

`/predict-image` and `/predict-scores` are both implemented. Heavy model loading remains isolated behind `ImageModelScoreProvider`.

## Checklist

| Planned item | Status | Evidence |
|---|---|---|
| Fix the best model/ensemble | Done | `model_bundle/reports/selected_mass_router_config.json` |
| Include ensemble + two strongest candidates | Done | `model_bundle/manifest.json`, `model_bundle/models/` |
| Prevent quality drift from research | Done | `tests/test_router_research_drift.py` |
| Validate artifact integrity | Done | `tests/test_artifact_integrity.py` |
| CLI/demo inference | Done for score-router, image smoke, and real-data parity smoke | `src/fluoro_mvp_backend/cli.py`, `examples/*.json`, `tools/image_inference_smoke.py`, `tools/real_data_parity_smoke.py` |
| Preprocessing service | Done as deterministic image normalization layer | `src/fluoro_mvp_backend/preprocessing.py`, `tests/test_preprocessing.py` |
| Inference service boundary | Done | `src/fluoro_mvp_backend/inference.py` |
| Full image scoring adapter | Done | `src/fluoro_mvp_backend/image_scoring.py`, `/predict-image`, `tools/image_inference_smoke.py` |
| API endpoints | Done and tested | `src/fluoro_mvp_backend/api.py`, `tests/test_api.py` |
| Logging/storage | Done and tested | `src/fluoro_mvp_backend/storage.py`, SQLite schema |
| Review feedback storage | Done and tested | `/review-feedback`, `review_feedback` table |
| Active learning backend contour | Done as bookkeeping + review queue helpers | `src/fluoro_mvp_backend/active_learning.py` |
| Scheduled retraining bookkeeping | Done | `training_runs` table and API endpoints |
| Research code transfer | Done | copied notebooks/scripts match originals byte-for-byte |
| Model card / MVP report | Done | `docs/MODEL_CARD.md`, `docs/MVP_REPORT.md` |
| Fresh environment smoke test | Done | `tools/fresh_env_smoke.py` |
| Real-data image parity smoke | Done on 20 local IN-CXR cases | `reports/real_data_parity/real_data_parity_summary.json` |
| UI scale-up spec | Done | `docs/UI_REVIEW_SPEC.md` |

## Locked Research Metrics

The backend router reproduces the final-test research output:

| Metric | Value |
|---|---:|
| Final test size | 1256 |
| Auto-negative selected | 125 |
| Auto-negative coverage | 9.95% |
| Auto-negative false negatives | 0 |
| NPV | 1.000 |
| NPV 95% CI low | 0.970 |

## Remaining Production Hardening Work

Before an image-upload backend can be called production-complete in a customer environment, add:

1. Customer-local data validation before clinical rollout.
2. Production observability: latency, memory, OOD-rate, route distribution, and reviewer disagreement dashboards.

The current code deliberately does not provide a fake fallback for model scores. `/model/artifacts` reports artifact presence, and `/predict-image` uses the packaged EVA-X-B and CheXFound artifacts.
