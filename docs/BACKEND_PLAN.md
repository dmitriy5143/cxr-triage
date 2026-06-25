# Backend Plan

## v1 Components

1. Preprocessing service
   - Load image.
   - Normalize to the model input contract.
   - Produce QA metadata.

2. Inference service
   - Compute CheXFound score.
   - Compute EVA-X-B score.
   - Apply calibrators/OOD models.
   - Pass scores to the router.
   - Current status: score-router is implemented; image scorer adapter is implemented behind `ImageModelScoreProvider`; full image smoke test is available in `tools/image_inference_smoke.py`.

3. Router service
   - Apply `selected_mass_router_config.json`.
   - Return route, reason, score, and thresholds.
   - Keep route behavior covered by regression tests.

4. API
   - `/health`
   - `/predict-scores`
   - `/predict-image`
   - `/model/artifacts`
   - `/review-feedback`
   - `/db/stats`
   - `/predictions`
   - `/review-candidates`
   - `/training-runs`

5. Logging and storage
   - Store every prediction payload.
   - Store every decision.
   - Store physician feedback and labels.

6. Review UI
   - Show image, route, reason, model scores, and QA flags.
   - Allow reviewer to confirm/correct label.
   - Export review queue for retraining.

7. Active learning
   - Select near-boundary and blocked cases.
   - Track training runs.
   - Schedule retraining only after enough reviewed cases arrive.

## Required Tests Before Backend MVP

- Artifact checksum and manifest tests.
- Router drift tests against research score tables.
- CLI score-routing tests.
- API contract tests.
- Preprocessing deterministic-output tests.
- Inference parity tests on fixed research/customer images.
- Storage and feedback roundtrip tests.
- Fresh-environment smoke test.

## Current Boundary

This package already covers router, CLI, API, full image scorer loading, real-data image parity smoke, logging, feedback storage, and active-learning bookkeeping. The remaining production hardening work is customer-local validation and production observability: latency, memory, OOD-rate, route distribution, and reviewer disagreement monitoring.
