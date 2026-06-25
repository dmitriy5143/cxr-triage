# Research Reproducibility

## Included Files

Notebook contour:

- `research/notebooks/fluoro_mvp_model_selection.ipynb`
- `research/notebooks/fluoro_mvp_ranking_closure.ipynb`
- `research/notebooks/fluoro_mvp_vindr_interpretation.ipynb`
- `research/notebooks/fluoro_mvp_single_notebook.ipynb`

Script contour:

- `research/scripts/run_model_selection_profile.py`
- `research/scripts/mass_router_meta_analysis.py`
- `research/scripts/prepare_ensemble_router_bundle.py`
- `research/scripts/chexfound_posthoc_and_head_sweep.py`
- `research/scripts/run_vindr_interpretation_local.py`

Technical plan:

- `docs/technical_plan_v04.md`

## Reproduction Protocol

1. Run model-selection experiments on IN-CXR.
2. Export candidate score tables for validation and final test.
3. Run head/router sweeps.
4. Select the deployment candidate by validation-safe routing.
5. Apply the frozen router to final test without retuning.
6. Export candidate bundle.
7. Run backend drift tests in this delivery package.

## Locked Candidate

The current selected router is stored in:

`model_bundle/reports/selected_mass_router_config.json`

The final-test score replay table is stored in:

`model_bundle/reports/input_scores_final_test.csv`

The expected final-test route table is stored in:

`model_bundle/reports/selected_routes_final_test.csv`

These files are the regression baseline for future backend work.
