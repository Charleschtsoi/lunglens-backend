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
