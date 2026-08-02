# Production Runtime Contract

## Locked Environment

The supported release environment is Python 3.11 or 3.12 with the exact
versions in `pyproject.toml`. Python 3.13/3.14 and unpinned upgrades are outside
the validated contract. Dependency updates are reviewed and tested one at a
time before a new release is cut.

The two probability calibrators are JSON artifacts containing only the Platt
coefficient and intercept. Runtime inference evaluates this math with NumPy and
never executes a method serialized from a notebook. Both OOD models were
refitted from the preserved 7,534-row training feature matrices under
scikit-learn 1.8.0. Validation parity was release-gated:

| Artifact | Max validation score delta | OOD gate changes |
|---|---:|---:|
| CheXFound OOD | 5.1e-8 | 0 / 1,256 |
| EVA OOD | 0 | 0 / 1,256 |

`load_ood_model` checks the installed scikit-learn version against
`ood_artifact_metadata.json` and fails clearly on mismatch.

## Preprocessing Contract

CheXFound follows the research path exactly: robust normalization, BILINEAR
letterbox to 1024, BICUBIC downsample to 512, per-channel min-max scaling, then
ImageNet normalization. EVA uses the locked 224 input path.

DICOM loading supports native pixel arrays beyond 8 bit, signed/unsigned pixel
representation through pydicom, rescale slope/intercept, MONOCHROME1 inversion,
and WindowCenter/WindowWidth clipping. These operations are regression-tested.

## Release Rule

Run unit tests, full artifact checks, locked router replay, a full image smoke,
and real-data parity before publishing an artifact release. A changed
calibrator, preprocessing tensor, OOD gate decision, or locked router result
blocks the release until reviewed.

Full image inference also requires an approved target-site OOD profile before
the auto-negative route is enabled. `FLUORO_ALLOW_REFERENCE_OOD_FOR_RESEARCH`
exists only for locked parity tooling and must remain `0` in production.
