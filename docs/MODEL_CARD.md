# Model Card

## Model

Primary candidate:

`CheXFound tuned frozen head + EVA-X-B partial-unfreeze last1 + ensemble router`

The model is used for binary screening support:

- `no_attention_required`: the model is confident the study can be auto-cleared.
- `N/A`: the case should remain in manual review.
- `requires_attention`: suspicious/high-risk model score.

For MVP safety, the product decision is driven by the auto-negative route. The model is not intended to replace physician review.

## Data

Primary model-selection data: IN-CXR style binary image-level labels for screening logic.

Interpretation and localization sanity checks: VinDr/VinBigData with bounding boxes.

The target is `y_attention`:

- `0` means no attention required.
- `1` means requires attention.

## Candidate Artifacts

Packaged candidates:

- Ensemble router using CheXFound head and EVA-X-B partial-unfreeze last1.
- EVA-X-B partial-unfreeze last1 checkpoint.
- CheXFound tuned frozen head checkpoint.
- Calibrators and OOD models used in the research contour.
- CheXFound HF backbone snapshot from `DIAL-RPI/CheXFound` at revision `41b966c3d4fcd2c9b5c8bd760f2df17a7b569dd3`.
- External EVA-X and CheXFound source snapshots needed to restore the image inference architectures.
- Frozen EVA-X-B base weights used to reproduce the deployed EVA OOD feature space.

## Final-Test Metrics

| Metric | Value |
|---|---:|
| N | 1256 |
| Auto-negative selected | 125 |
| Auto-negative coverage | 9.95% |
| False negatives on auto-negative route | 0 |
| NPV | 1.000 |
| NPV 95% CI low | 0.970 |

## Safety Policy

The router applies conservative gates:

- image quality gate;
- CheXFound OOD gate;
- EVA OOD gate;
- uncertainty gate;
- model-pair low-risk/veto rule.

A case is auto-cleared only when it passes all gates and the model-pair rule.

## Limitations

- Backend image inference is implemented and real-data parity-smoked on local IN-CXR samples; customer-local data validation is still required before deployment.
- Metrics are based on the available open dataset and should be revalidated on customer-local data before clinical use.
- Edge cases near the auto-negative boundary should be prioritized for physician review and active learning.
- The model is a decision-support component, not a standalone medical device.
