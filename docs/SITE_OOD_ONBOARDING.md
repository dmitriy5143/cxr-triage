# Site OOD Onboarding

The bundled OOD detectors describe the IN-CXR research distribution. A high OOD
rate on a new clinic or device is a valid domain-shift signal, not a threshold
bug. The safe default route is `N/A`. Full image inference is release-gated:
without an approved site profile, the router returns
`target_site_ood_not_validated` even if model scores happen to be low.

Fit a draft profile entirely inside the clinic network:

```bash
PYTHONPATH=src python3 tools/fit_site_ood_profile.py \
  --image-dir /secure/local/validation_images \
  --bundle model_bundle \
  --out-dir /secure/local/site_profiles/clinic-a \
  --site-id clinic-a \
  --device cuda
```

The command stores OOD artifacts, anonymous content hashes, QA summaries, and
runtime metadata. It does not copy source images and marks the result
`draft_requires_clinical_validation`.

Before approval:

1. Use a held-out local sample covering every device/protocol in scope.
2. Review OOD-rate, QA failures, score distributions, route distribution, and
   labeled FN/NPV behavior with clinical owners.
3. Recalibrate probability heads/router on a separate labeled validation split
   when score shift is present. Do not use the final test split for tuning.
4. Record model, data, device, software, reviewer, and approval versions.
5. Only the governed release process may change profile status to
   `approved_for_clinical_validation`.

After approval, configure `FLUORO_SITE_OOD_DIR` to that profile directory. The
API reports the active profile and clinical gate status through
`GET /model/artifacts`.
