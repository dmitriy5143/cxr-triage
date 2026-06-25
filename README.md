# Fluoro MVP Delivery

This folder is the backend-ready delivery package for the current FLG/CXR MVP candidate.

It contains:

- `model_bundle/` - selected candidate artifacts, calibrators, router config, reports, and checksums.
- `src/fluoro_mvp_backend/` - backend-facing preprocessing, routing, inference boundary, CLI, API skeleton, logging, feedback, and active-learning storage.
- `tests/` - regression tests that verify artifact integrity and prevent router drift against the research outputs.
- `research/` - notebooks and scripts used to reproduce the model-selection and interpretation contour.
- `docs/` - model card, MVP report, backend plan, and reproducibility notes.
- `runtime/` - local SQLite runtime data, initialized by `tools/init_db.py`.
- `.env.example` - environment-variable template for local/API deployment.

Heavy model binaries are not committed as regular git blobs. Put them back into
`model_bundle/` from the release archives described in
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md).

## Fixed MVP Candidate

Primary deployment candidate:

`CheXFound tuned frozen head + EVA-X-B partial-unfreeze last1 + conservative ensemble router`

The router auto-clears a study as `no_attention_required` only when one model is very low-risk and the second model does not veto it, with quality/OOD/uncertainty gates.

Locked final-test metrics from the research contour:

| Metric | Value |
|---|---:|
| Final test size | 1256 |
| Auto-negative count | 125 |
| Auto-negative coverage | 9.95% |
| False negatives on auto-negative route | 0 |
| NPV on auto-negative route | 1.000 |
| NPV 95% CI low | 0.970 |

Two strongest single-model candidates are also packaged:

- `EVA-X-B partial unfreeze last1`
- `CheXFound frozen tuned MLP head`

## Quick Start

Run tests from this folder:

```bash
cd fluoro_mvp_delivery
python3 -m pytest
```

Run CLI inference on a saved demo score payload:

```bash
cd fluoro_mvp_delivery
PYTHONPATH=src python3 -m fluoro_mvp_backend.cli \
  --bundle model_bundle \
  --scores-json examples/demo_scores_auto_negative.json \
  --pretty
```

Run a compact smoke check after moving the folder to a fresh machine:

```bash
cd fluoro_mvp_delivery
python3 tools/fresh_env_smoke.py
```

Run the optional full image-inference smoke test after installing image extras:

```bash
cd fluoro_mvp_delivery
python3 -m pip install -e ".[image]"
PYTHONPATH=src python3 tools/image_inference_smoke.py
```

Run real-data parity smoke when the local IN-CXR PNG folder is available:

```bash
cd fluoro_mvp_delivery
PYTHONPATH=src python3 tools/real_data_parity_smoke.py \
  --data-root ../data/incxr_png/'IN-CXR (pre-processed)'
```

Initialize the local MVP SQLite database:

```bash
cd fluoro_mvp_delivery
python3 tools/init_db.py
```

Run the optional API after installing API dependencies:

```bash
cd fluoro_mvp_delivery
python3 -m pip install -e ".[api]"
uvicorn fluoro_mvp_backend.api:create_app --factory
```

The CLI/API supports both score-router inference and full image inference. Full image inference is intentionally isolated behind `ImageModelScoreProvider` because it loads heavy EVA/CheXFound backbones. The bundle includes CheXFound HF safetensors, external EVA-X/CheXFound model code, and exposes artifact status through `/model/artifacts`.

CLI image inference:

```bash
cd fluoro_mvp_delivery
PYTHONPATH=src python3 -m fluoro_mvp_backend.cli \
  --bundle model_bundle \
  --image /path/to/radiograph.png \
  --pretty
```

## Why This Shape

The research notebooks are excellent for experimentation, but the backend needs a smaller, testable boundary. The core production decision is the router, so this package freezes the selected router and adds tests that replay the research final-test score table. If a future code change alters the 125 selected auto-negative cases, tests fail.
