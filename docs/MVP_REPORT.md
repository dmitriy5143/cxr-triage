# Final MVP Report

## What We Have

The strongest current MVP candidate is an ensemble:

`CheXFound tuned head + EVA-X-B partial-unfreeze last1 + conservative router`.

The model does not make an aggressive binary decision for every image. Instead, it uses a safe routing policy:

- clearly low-risk cases go to `no_attention_required`;
- uncertain or unusual cases go to `N/A`;
- high-risk cases can be marked as `requires_attention`.

This is the right shape for an MVP screening product because the safest automation point is the low-risk auto-clear route.

## Result

On the locked final test split:

| Metric | Result |
|---|---:|
| Studies | 1256 |
| Auto-cleared as no attention required | 125 |
| False negatives among auto-cleared cases | 0 |
| NPV among auto-cleared cases | 1.000 |
| Conservative lower CI for NPV | 0.970 |

The key product interpretation: every study auto-cleared by the selected router was truly negative in the final-test labels.

## How The Router Works

Two independent model signals are used:

- CheXFound tuned frozen head;
- EVA-X-B partial-unfreeze last1.

The case is auto-cleared only when one signal is very confidently low-risk and the other signal does not object. The router also blocks auto-clear when the image looks out-of-distribution, low-quality, or too uncertain.

## Why This Is Backend-Ready

The delivery package contains:

- selected model artifacts;
- calibrators and OOD models;
- preprocessing config;
- router config;
- metric reports;
- integrity manifest with checksums;
- CLI demo;
- API with score and full-image inference;
- storage for predictions and review feedback;
- active-learning training-run bookkeeping;
- regression tests that prevent drift from research metrics.

## Next Product Step

Validate on customer-local data with the same locked router protocol. Review feedback should be stored from day one so the model can be refreshed on real product data.
