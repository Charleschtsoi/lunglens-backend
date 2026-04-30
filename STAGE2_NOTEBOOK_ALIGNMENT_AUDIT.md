# Stage-2 Notebook Alignment Audit

Source of truth:

- `/Users/charlescht/Downloads/main.ipynb` output summary provided by team.

## Aligned

- Class order default in backend matches notebook:
  - index 0 `Normal`
  - index 1 `Lung Opacity`
  - index 2 `Viral Pneumonia`
- Stage-2 preprocessing in backend inference path matches notebook:
  - resize `224x224`
  - RGB conversion
  - normalization `/255.0`
- Backend does not apply `resnet_v2.preprocess_input` for stage-2 H5 path.

## Misaligned (before this adjustment)

- No explicit uncertainty policy for borderline outputs (e.g., low top confidence or small top-2 margin).

## Missing (before this adjustment)

- Docs did not clearly state notebook-confirmed preprocessing and class-order assumptions as canonical stage-2 defaults.

## Fixes Applied

- Added optional uncertainty gating for Stage 2 behind env flags:
  - `STAGE2_UNCERTAINTY_ENABLED` (default `false`)
  - `STAGE2_UNCERTAINTY_MIN_CONFIDENCE` (default `0.55`)
  - `STAGE2_UNCERTAINTY_MIN_MARGIN` (default `0.1`)
- Kept response contract unchanged.
- When enabled, uncertain stage-2 outputs map to `stage2.label = "Other"` with top score confidence.
- Added documentation updates to keep team alignment with notebook semantics.
