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

See `ARCHITECTURE_ALIGNMENT.md` for the shared backend/frontend/ML pipeline contract derived from the v1.5 project architecture document.
See `MULTI_DATASET_ADOPTION.md` for business validation, cost bands, and 30-day rollout guidance for multi-dataset expansion.

## Project structure

- `main.py` - FastAPI app, health endpoints, and mock analyze pipeline.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - Container build and run configuration.

## Production environment variables

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

3. Start the server:

```bash
ENVIRONMENT=development \
REQUIRE_API_KEY=false \
uvicorn main:app --host 0.0.0.0 --port 7860
```

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
  - `questionnaire` (optional stringified JSON)

Backend always returns JSON:

- Error:
  - `{"success": false, "error": "human readable message"}`
- Success:
  - includes `success`, `predictions` (all 14 labels), `gradcam`, `stage1`, `stage2`, `gate`, `requires_questionnaire`, `stage3`, `report`, `timing_ms`

Required enums:

- `stage1.label`: `Pneumonia | Normal`
- `stage2.label`: `Normal | Lung Opacity | Viral Pneumonia | Other`
- `gate.route`: `early_stop | continue`
- `gate.reason`: `both_negative | positive_detected`
- `stage3.risk_level`: `low | medium | high`
- `stage3.severity`: `low | moderate | high`

Flow rules:

- If `gate.route == continue` and questionnaire is missing:
  - `requires_questionnaire: true`
  - `report: null`
- If questionnaire is present:
  - `requires_questionnaire: false`
  - include `stage3` and `report`
- If `gate.route == early_stop`:
  - backend may return immediate final output with `requires_questionnaire: false`

Transport/auth/validation:

- Invalid or missing API key -> `401`
- Missing/invalid file -> `400`
- Oversized upload -> `413`
- Unsupported MIME type -> `415`
- Error response shape always remains `{"success": false, "error": "<message>"}`.

## Architecture compatibility endpoints

To align with architecture notes while preserving the current frontend contract, the backend exposes both:

- Current production endpoints:
  - `GET /healthz`
  - `POST /pipeline/analyze`
  - `POST /api/v1/analyze`
- Compatibility aliases:
  - `GET /health` -> `{"status":"ok","models_loaded":<bool>}`
  - `POST /predict` -> stage output + gate format
  - `POST /assess` -> stage3/stage4 follow-up format

Notes:

- Existing frontend integration should continue using `/pipeline/analyze`.
- Compatibility aliases are additive and non-breaking.

## Stage-2 real model pilot (3-class only)

You can enable only the uploaded `.h5` model for Stage 2 (`Normal | Lung Opacity | Viral Pneumonia`) while keeping the rest of the pipeline in mock mode.

Set these backend env vars:

```env
ENABLE_H5_MODEL=true
H5_MODEL_PATH=models/resnet152v2_lung_disease_final.h5
H5_STAGE2_LABELS=Normal,Lung Opacity,Viral Pneumonia
```

Notes:

- Stage 2 will use real model inference.
- `predictions` (14 labels), gate/report scaffolding, and non-Stage-2 components remain mock-assisted for this pilot.
- If `ENABLE_H5_MODEL=false` (default), backend uses full mock behavior.

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
