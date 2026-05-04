---
title: Lunglens Backend
emoji: 🌖
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
license: mit
---

# LungLens Backend (Week 1 Skeleton)

Minimal FastAPI backend skeleton for LungLens frontend integration and mock pipeline behavior.

**Local-first:** develop against this repo on your machine (`.env` + `uvicorn`). Point the Next.js app at `http://localhost:7860` and defer Hugging Face / Vercel production env until models and contracts are stable.

See **`FRONTEND_E2E_CHECKLIST.md`** for frontend end-to-end tests (questionnaire gate + education report).
See **`DEBUGGING_BACKEND.md`** for backend-side troubleshooting (model load, provenance, and fallback diagnosis).

See `ARCHITECTURE_ALIGNMENT.md` for the shared backend/frontend/ML pipeline contract derived from the v1.5 project architecture document.
See `MULTI_DATASET_ADOPTION.md` for business validation, cost bands, and 30-day rollout guidance for multi-dataset expansion.

## Project structure

- `main.py` - FastAPI app, health endpoints, and mock analyze pipeline.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - Container build and run configuration.

## Production environment variables

Deploy only after models and integration are ready. On any host (including Hugging Face Spaces), set `ENVIRONMENT=production` so API key enforcement defaults on.

Set these before deployment:

- `ENVIRONMENT=production`
- `ALLOWED_ORIGINS=https://your-frontend.vercel.app,https://your-custom-domain.com`
- `API_KEY=<strong-random-secret>`
- `REQUIRE_API_KEY=true`
- `MAX_UPLOAD_MB=10`
- `ALLOWED_IMAGE_MIME_TYPES=image/jpeg,image/png,image/webp`

Notes:

- `ALLOWED_ORIGINS` must be explicit in production (wildcard is blocked).
- Analyze endpoints require `X-API-Key` when `REQUIRE_API_KEY=true`.
- Upload size and MIME type checks are enforced before image validation.

## Run locally

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: `cp .env.example .env` and edit paths (especially `H5_MODEL2_PATH` / legacy `H5_MODEL_PATH` if using the real ML Model 2 `.h5`). Variables load automatically via `python-dotenv` when present.

4. Start the server (defaults are already local-friendly when `ENVIRONMENT` is unset or `development`):

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

Equivalent explicit flags:

```bash
ENVIRONMENT=development REQUIRE_API_KEY=false uvicorn main:app --host 0.0.0.0 --port 7860
```

**Frontend:** set `BACKEND_API_BASE_URL=http://localhost:7860` (and `BACKEND_API_KEY` only if you set `REQUIRE_API_KEY=true` locally).

## Run with Docker

Build:

```bash
docker build -t lunglens-backend .
```

Run:

```bash
docker run --rm -p 7860:7860 \
  -e ENVIRONMENT=production \
  -e ALLOWED_ORIGINS=https://your-frontend.vercel.app \
  -e API_KEY=change-me \
  -e REQUIRE_API_KEY=true \
  -e MAX_UPLOAD_MB=10 \
  lunglens-backend
```

## Connect to Vercel frontend

1. Deploy this backend (Hugging Face Space Docker or another container host).
2. In Vercel project settings, add:
   - `NEXT_PUBLIC_API_BASE_URL=https://your-backend-host`
   - `LUNGLENS_API_KEY=<same value as backend API_KEY>` (server-side use only)
3. Update frontend calls to include `X-API-Key` header.
4. Add your Vercel production URL to backend `ALLOWED_ORIGINS`.

## Frontend API contract (sync reference)

This section mirrors the current frontend/backend integration contract.

- **Browser -> Next.js**: `POST /api/analyze` (multipart form)
- **Next.js -> Backend**: `POST {BACKEND_API_BASE_URL}/pipeline/analyze`
- **Header**: `X-API-Key: <BACKEND_API_KEY>`
- **Multipart fields**:
  - `image` (required file, jpg/png/webp)
  - `questionnaire` (optional stringified JSON) — see **`FRONTEND_E2E_CHECKLIST.md`** for Zustand **camelCase** (`coughDurationDays`, `breathingDifficulty`, …) and nested `patient_data`; both are accepted.

Backend always returns JSON:

- Error:
  - `{"success": false, "error": "human readable message", "error_code": "...", "stage": "...", "retryable": false}`
- Success:
  - includes `success`, `predictions` (all 14 labels), `gradcam`, `model1`, `model2`, `gate`, `requires_questionnaire`, `model3`, `model4`, `timing_ms`, `warnings`, `provenance`

Required enums:

- `model1.label`: with **`ENABLE_MODEL1_PYTORCH`**: `Normal | Pneumonia-bacteria | Pneumonia-virus`; otherwise mock/rules: `Pneumonia | Normal`
- `model2.label`: `Normal | Lung Opacity | Viral Pneumonia | Other`
- `gate.route`: `early_stop | continue`
- `gate.reason`: `both_negative | positive_detected`
- `model3.risk_level`: `low | medium | high`
- `model3.severity`: `low | moderate | high`

Flow rules:

- If `gate.route == continue` and questionnaire is missing:
  - `requires_questionnaire: true`
  - `model4: null`
- If questionnaire is present:
  - `requires_questionnaire: false`
  - include `model3` and `model4`
- If `gate.route == early_stop`:
  - backend may return immediate final output with `requires_questionnaire: false`

Transport/auth/validation:

- Invalid or missing API key -> `401`
- Missing/invalid file -> `400`
- Oversized upload -> `413`
- Unsupported MIME type -> `415`
- Error response includes `error_code`, `stage`, and `retryable` for frontend handling.

Provenance and warnings:

- `provenance.run_mode`: `real | mock | hybrid`
- `provenance.model{1..4}`: source/status/model metadata
- `warnings[]`: degraded or scaffold notices for explainability messaging

## Architecture compatibility endpoints

To align with architecture notes while preserving the current frontend contract, the backend exposes both:

- Current production endpoints:
  - `GET /healthz`
  - `POST /pipeline/analyze`
  - `POST /api/v1/analyze`
- Compatibility aliases:
  - `GET /health` -> `{"status":"ok","models_loaded":<bool>}`
  - `POST /predict` -> stage output + gate format
  - `POST /assess` -> model3/model4 follow-up format

Notes:

- Existing frontend integration should continue using `/pipeline/analyze`.
- Compatibility aliases are additive and non-breaking.

## ML Model 2 — real `.h5` pilot (3-class only)

You can enable only the uploaded `.h5` for **ML Model 2** (`Normal | Lung Opacity | Viral Pneumonia`) while keeping other outputs mock- or rule-assisted.

Notebook-confirmed ML Model 2 assumptions:

- Class order must match training (see `H5_MODEL2_LABELS`; often alphabetical folder order).
- Preprocessing: RGB -> resize to `224x224` -> normalize `/255.0`
- No `resnet_v2.preprocess_input` in this path
- **TensorFlow:** use **`tensorflow==2.18.0`** from `requirements.txt` for Colab-exported `.h5` graphs.

Set these backend env vars:

```env
ENABLE_MODEL2_H5=true
H5_MODEL2_PATH=models/resnet152v2_lung_disease_final.h5
H5_MODEL2_LABELS=Normal,Lung_Opacity,Viral_Pneumonia
```

(Legacy: `ENABLE_H5_MODEL`, `H5_MODEL_PATH`, `H5_STAGE2_LABELS` are still read if the new names are unset.)

Notes:

- ML Model 2 will use real inference when the `.h5` **loads successfully**; otherwise it falls back to rule/mock behavior and responses may include warnings.
- `predictions` (14 labels), gate, and non-Model-2 fields remain mock-assisted for this pilot.
- If `ENABLE_MODEL2_H5` / `ENABLE_H5_MODEL` is false (default), backend uses full mock behavior for Model 2.

**`provenance.run_mode` (per request):**

- `mock` — ML Model 2 H5 did not run successfully on that request (flag off, load failure, or inference error).
- `hybrid` — ML Model 2 H5 inference succeeded; ML Model 1 / 14-label path remain mock-assisted (default honest label).
- `real` — Set `STAGE2_COUNTS_AS_FULL_REAL_RUN_MODE=true` if the product should report `real` when only ML Model 2 is live (optional).

Check `GET /health` for `models.model2_h5.loaded` and load errors before E2E.

Optional uncertainty handling (disabled by default):

```env
STAGE2_UNCERTAINTY_ENABLED=false
STAGE2_UNCERTAINTY_MIN_CONFIDENCE=0.55
STAGE2_UNCERTAINTY_MIN_MARGIN=0.1
```

When uncertainty handling is enabled, low-confidence or borderline ML Model 2 outputs are mapped to `model2.label = Other`.

## Example API calls

Health:

```bash
curl http://127.0.0.1:7860/healthz
```

Analyze without questionnaire:

```bash
curl -X POST "http://127.0.0.1:7860/api/v1/analyze" \
  -H "X-API-Key: change-me" \
  -F "image=@/path/to/chest_xray.png"
```

Analyze with questionnaire:

```bash
curl -X POST "http://127.0.0.1:7860/pipeline/analyze" \
  -H "X-API-Key: change-me" \
  -F "image=@/path/to/chest_xray.png" \
  -F 'questionnaire={"patient_data":{"age":62,"fever":true,"cough_duration_days":4},"notes":"sample"}'
```

Flat Zustand-style JSON is also accepted, for example:
`'questionnaire={"age":62,"fever":true,"coughDurationDays":4,"smoking":"never","breathingDifficulty":"none"}'`
(`smoking`: `never` | `former` | `current`; `breathingDifficulty`: `none` | `mild` | `severe`).
