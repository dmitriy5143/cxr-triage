# Model Artifacts

The repository keeps code, tests, configs, reports, notebooks, and checksums in
git. Heavy binary model artifacts are intentionally not committed as regular git
blobs.

Use release artifacts for the backend-ready model bundle:

- `cxr-triage-eva-artifacts-v0.1.0.tar`
- `cxr-triage-chexfound-artifacts-v0.1.0.tar`
- `model_artifacts_sha256.json`

For GitHub Release upload, the tar files may also be published as split parts:

- `cxr-triage-eva-artifacts-v0.1.0.tar.part-aa`, `.part-ab`, ...
- `cxr-triage-chexfound-artifacts-v0.1.0.tar.part-aa`, `.part-ab`, ...

Reassemble split artifacts first:

```bash
cat cxr-triage-eva-artifacts-v0.1.0.tar.part-* > cxr-triage-eva-artifacts-v0.1.0.tar
cat cxr-triage-chexfound-artifacts-v0.1.0.tar.part-* > cxr-triage-chexfound-artifacts-v0.1.0.tar
```

Both tar archives are extracted over the repository root:

```bash
cd cxr-triage
tar -xf /path/to/cxr-triage-eva-artifacts-v0.1.0.tar
tar -xf /path/to/cxr-triage-chexfound-artifacts-v0.1.0.tar
```

After extraction, verify that all required large artifacts match the manifest:

```bash
cd cxr-triage
CHECK_LARGE_ARTIFACTS=1 PYTHONPATH=src python3 -m pytest tests/test_artifact_integrity.py
PYTHONPATH=src python3 tools/fresh_env_smoke.py
PYTHONPATH=src python3 tools/image_inference_smoke.py
```

Create or refresh the release artifacts from a complete local delivery folder:

```bash
cd fluoro_mvp_delivery
python3 tools/pack_model_artifacts.py
```

The script writes archives into `release_artifacts/`. This directory is ignored
by git and should be uploaded to a GitHub Release, not committed.

Git LFS can also be used later if the project has enough LFS quota, but release
artifacts are safer for this bundle because the complete model payload is
multiple gigabytes.
