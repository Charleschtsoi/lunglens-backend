---
title: Lunglens Backend
emoji: 🌖
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
license: mit
---

# LungLens Backend

FastAPI backend for LungLens image analysis, rule-based gating, questionnaire-assisted risk context, and report synthesis.

This README is written as a local onboarding guide for teammates: clone -> run -> test -> debug.

## What this backend does

- Accepts chest X-ray uploads (`jpg/png/webp`) through `multipart/form-data`.
- Runs model inference when enabled and loaded:
  - Model 1: PyTorch ResNet50 (3-class)
  - Model 2: Keras H5 (3-class)
  - Model 3: DenseNet-121 (3-class + Grad-CAM)
- Aggregates findings with strict rules:
  - Build `predictions` only from non-`Normal` positives from successful models.
  - If no positives, return `{"Normal": 1.0}`.
- Uses gate logic:
  - `continue` when any non-`Normal` finding exists.
  - `early_stop` when only `Normal` is present.
- Uses questionnaire to enrich contextual risk/report text (does not change raw model logits/probabilities).

## Tech stack

- Python 3.10+
- FastAPI + Uvicorn
- Pydantic v2
- TensorFlow (for `.h5` model)
- PyTorch + torchvision
- `grad-cam`
- pytest + httpx + pytest-asyncio

## Repository layout

- `main.py`: API server, model loaders, inference/rules pipeline
- `requirements.txt`: runtime dependencies
- `tests/test_api.py`: contract and endpoint tests
- `tests/requirements.txt`: test-only dependencies
- `.env.example`: local environment template
- `Dockerfile`: container runtime

## Endpoints

- `GET /` -> liveness
- `GET /healthz` -> lightweight health
- `GET /health` -> detailed model health/load status
- `GET /debug` -> provenance/debug diagnostics
- `POST /api/v1/analyze` -> primary analyze endpoint
- `POST /pipeline/analyze` -> alias of analyze endpoint
- `POST /predict/densenet` -> standalone DenseNet inference + Grad-CAM
- `POST /api/v1/generate-questions` -> question generation helper

## 1) Local setup

### Prerequisites

- Python 3.10 or 3.11 recommended
- `pip` available
- Optional: model files on local disk if you want live inference

### Create venv and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r tests/requirements.txt
```

## 2) Environment configuration

Copy and edit `.env`:

```bash
cp .env.example .env
```

Minimum local-safe defaults:

```env
ENVIRONMENT=development
REQUIRE_API_KEY=false
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,*
MAX_UPLOAD_MB=10
```

### Enable models (optional)

If model files exist locally, set the relevant flags and paths:

```env
ENABLE_MODEL1=true
MODEL1_PATH=models/best_resnet50_lunglens_cleaner.pth
MODEL1_LABELS=Normal,Pneumonia-Bacteria,Pneumonia-Virus

ENABLE_MODEL2_H5=true
H5_MODEL2_PATH=/absolute/path/to/resnet152v2_lung_disease_final.h5
H5_MODEL2_LABELS=Normal,Lung_Opacity,Viral_Pneumonia

ENABLE_DENSENET121=true
DENSENET121_PATH=models/best_densenet121_lunglens.pth
DENSENET121_LABELS=COVID-19,Normal,Pneumonia
```

Notes:

- Model files are intentionally not committed to Git.
- Keep label order exactly aligned to training checkpoints.
- If a model is disabled or fails to load, pipeline still returns JSON with fallback/rules behavior.

## 3) Run the server locally

```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Service URL: `http://127.0.0.1:7860`

Quick check:

```bash
curl http://127.0.0.1:7860/healthz
```

## 4) Manual API testing

### Analyze (image only)

```bash
curl -X POST "http://127.0.0.1:7860/api/v1/analyze" \
  -F "image=@testfile/Lung Xray.jpeg"
```

### Analyze (image + questionnaire)

```bash
curl -X POST "http://127.0.0.1:7860/api/v1/analyze" \
  -F "image=@testfile/Lung Xray.jpeg" \
  -F 'questionnaire={"patient_data":{"age":45,"fever":false,"cough_duration_days":3,"smoking":"never","breathing_difficulty":"none"}}'
```

### DenseNet standalone endpoint

```bash
curl -X POST "http://127.0.0.1:7860/predict/densenet" \
  -F "image=@testfile/Lung Xray.jpeg"
```

### When API key is enabled

If `REQUIRE_API_KEY=true`, include:

```bash
-H "X-API-Key: <your-api-key>"
```

## 5) Run automated tests

```bash
source .venv/bin/activate
pytest -q
```

Run only API contract tests:

```bash
pytest tests/test_api.py -q
```

## 6) Debugging workflow for teammates

1. Check load status:
   - `GET /health`
   - `GET /debug`
2. Validate request payload:
   - multipart has `image`
   - `questionnaire` is valid JSON string when sent
3. Confirm gate behavior:
   - `gate.route == "continue"` -> questionnaire may be required
   - `gate.route == "early_stop"` -> final output without questionnaire is possible
4. Verify provenance:
   - `provenance.run_mode`
   - `provenance.model1/model2/model3` source + status

## 7) Frontend integration notes

For local frontend integration:

- Backend base URL: `http://localhost:7860`
- If backend requires key, frontend/SSR must send `X-API-Key`.
- Questionnaire impacts `clinical_risk` and `model4` summary context, not the base model inference outputs.

## 8) Production safety checklist

Before deploying publicly:

- Never commit `.env`, keys, or private assets.
- Use `ENVIRONMENT=production`.
- Set explicit `ALLOWED_ORIGINS` (no wildcard).
- Set `REQUIRE_API_KEY=true` and a strong `API_KEY`.
- Store model files securely in host-managed storage.

## 9) Docker (optional)

Build:

```bash
docker build -t lunglens-backend .
```

Run:

```bash
docker run --rm -p 7860:7860 \
  -e ENVIRONMENT=production \
  -e ALLOWED_ORIGINS=https://your-frontend.vercel.app \
  -e REQUIRE_API_KEY=true \
  -e API_KEY=change-me \
  lunglens-backend
```

## 10) Common teammate pitfalls

- Wrong Python interpreter (tests fail due to missing deps).
- Model path points to non-existent file.
- Label order mismatch vs checkpoint output order.
- Posting questionnaire as JSON body instead of multipart form field.
- Forgetting `X-API-Key` when API key enforcement is on.
