# LungLens Architecture Alignment (v1.5)

This document captures the agreed architecture and integration contract based on:

- `Machine Learning 2 project v1.5-2.pdf`

It is intended to keep backend, frontend, and ML integration aligned during implementation.

## End-to-end pipeline (authoritative flow)

1. **ML Model 1: Binary pneumonia detection**
   - Input: uploaded chest X-ray
   - Output: `Pneumonia` vs `Normal` + confidence
2. **ML Model 2: Multi-class disease classification**
   - Runs on the same uploaded image
   - Output: class label + confidence
3. **Grad-CAM**
   - Generated from image model outputs
   - Always produced for explainability
4. **Gate check**
   - If both ML Model 1 and ML Model 2 are negative -> early stop
   - If either is positive -> continue
5. **ML Model 3: Clinical Q&A (conditional)**
   - Triggered only when gate route is `continue`
   - Uses questionnaire inputs for severity/risk/recovery
6. **ML Model 4: Report synthesis**
   - Aggregates outputs from ML Models 1–3 + Grad-CAM
   - Produces human-readable summary, actions, and disclaimer

## Deployment architecture decision

- Frontend runs on Vercel.
- Model/pipeline backend runs separately (Hugging Face Space Docker).
- Frontend calls backend API over HTTPS.
- This separation is required due to model/runtime size limits and latency constraints.

## Frontend-facing API contract (must remain stable)

Current backend response shape that frontend depends on:

- `success`
- `predictions` (14-label probability object)
- `gradcam`:
  - `heatmap_base64`
  - `top_prediction`
  - `confidence`
- `model1`
- `model2`
- `gate`
- `model3`
- `model4`
- `timing_ms`
- `requires_questionnaire`

Any future model integration (ResNet152 or others) should replace internals only and preserve these keys.

## Gate and model-output contract

- If gate route is `early_stop`:
  - Backend may return minimal downstream data and stop early semantics.
- If gate route is `continue` and questionnaire is missing:
  - `requires_questionnaire: true`
  - `model4: null`
- If questionnaire is present:
  - `requires_questionnaire: false`
  - include `model3` and `model4`

## ML integration checkpoints

When replacing mock inference with trained model inference:

1. Keep class order mapping deterministic (from `labels.json`).
2. Keep image preprocessing identical to training pipeline.
3. Load model once at startup, not per request.
4. Generate real Grad-CAM overlay and return base64 PNG.
5. Preserve existing response keys and type formats.
6. Maintain security controls (API key, CORS, upload limits).

## Known class-label alignment note

The proposal text describes ML Model 2 classes as `Normal / Bacterial / Viral / Other`.
Current backend uses `Normal / Lung Opacity / Viral Pneumonia / Other`.

Team should confirm one canonical ML Model 2 label taxonomy and use it consistently across:

- training labels
- backend response
- frontend display text
- report templates

## Verification checklist before release

- Frontend can call `/api/v1/analyze` and `/pipeline/analyze`.
- `X-API-Key` is included in production requests.
- CORS allows Vercel production domain(s).
- Grad-CAM renders correctly in frontend overlay.
- Gate route and questionnaire flow match UX design.
- Error states handled in frontend (`401`, `413`, `415`, `400`, `500`).
