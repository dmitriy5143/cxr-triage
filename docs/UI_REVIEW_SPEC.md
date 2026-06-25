# UI / Review Spec

## Goal

Build a review UI around the MVP backend so the product team can inspect model decisions, collect physician feedback, and prepare future active-learning runs.

The UI should treat the model as a routed decision-support system, not as a simple binary classifier.

## Main Screens

### 1. Worklist

Purpose: scan incoming studies and model routes.

Columns:

- study id;
- thumbnail;
- route: `no_attention_required`, `N/A`, `requires_attention`;
- reason;
- model score;
- CheXFound score;
- EVA score;
- quality score;
- OOD flags;
- uncertainty;
- created time;
- review status.

Filters:

- route;
- reason;
- review status;
- high uncertainty;
- OOD;
- low quality;
- near-threshold cases;
- date range.

Primary actions:

- open case;
- mark reviewed;
- send to physician queue;
- export selected cases.

### 2. Case Review

Purpose: inspect one study and store feedback.

Layout:

- image viewer;
- route and reason;
- score block;
- preprocessing/QA block;
- model explanation block;
- physician decision form.

Feedback fields:

- final label: `no attention required` / `requires attention`;
- optional clinical comment;
- reviewer id;
- confidence;
- reason tags: poor image, label uncertainty, subtle finding, artifact, out of distribution, other.

Backend call:

`POST /review-feedback`

Payload:

```json
{
  "study_id": "string",
  "prediction_id": 1,
  "label_attention": 0,
  "reviewer_id": "doctor_1",
  "comment": "confirmed",
  "metadata": {
    "confidence": "high",
    "tags": ["near_threshold"]
  }
}
```

### 3. Active Learning Queue

Purpose: collect useful cases for later retraining.

Sources:

- near-threshold auto-negative blockers;
- high uncertainty;
- OOD but clinically normal;
- physician-disagreed cases;
- cases with repeated quality problems.

Backend call:

`GET /review-candidates?limit=50`

Displayed columns:

- study id;
- priority;
- reason for selection;
- model scores;
- route blocker count;
- current label if available.

### 4. Training Runs

Purpose: track periodic model refresh attempts.

Backend calls:

- `POST /training-runs`
- `GET /training-runs`

Run fields:

- run name;
- status: planned, running, failed, complete;
- config;
- metrics;
- artifact URI;
- created time.

## API Contract

### Health

`GET /health`

Returns backend status, bundle path, and DB path.

### Score Prediction

`POST /predict-scores`

Current tested production boundary. Receives already computed model scores and applies the fixed router.

Required fields:

- `p_chex_head`;
- `p_last1`;
- `quality_score`;
- `ood_score_chex`;
- `ood_score_eva`.

Recommended fields:

- `study_id`;
- `image_file`;
- `critical_qa_bool`;
- uncertainty columns.

### Image Prediction

`POST /predict-image`

Active endpoint for full image inference. It loads the packaged EVA-X-B and CheXFound artifacts, computes model scores, applies the fixed router, stores the prediction, and returns route, reason, scores, thresholds, and preprocessing metadata.

The UI should treat backend `4xx/5xx` responses as operational errors, not as model decisions.

## Route Semantics

`no_attention_required`:

The model selected the case for auto-clear. This is the only route used for MVP automation metrics.

`N/A`:

Manual review. Usually uncertainty, OOD, low quality, or gray-zone score.

`requires_attention`:

Suspicious model score. For MVP this should still be treated as a decision-support route.

## UI Safety Rules

- Always show route reason next to route.
- Do not hide OOD/quality blockers.
- Do not allow auto-clear without a stored backend prediction id.
- Store every physician correction.
- Never overwrite raw model scores after review.
- Keep model version and router config version visible in case detail.

## Scaling Notes

The first UI can be a simple internal review console. For the next stage:

- add authentication;
- add role-based access;
- add audit logs;
- add DICOM/PACS integration;
- add asynchronous image inference jobs;
- add dashboard charts for route distribution and review disagreement rate.
