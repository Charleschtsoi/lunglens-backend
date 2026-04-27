# LungLens Backend (Week 1 Skeleton)

Minimal FastAPI backend skeleton for LungLens frontend integration and mock pipeline behavior.

## Project structure

- `main.py` - FastAPI app, health endpoints, and mock analyze pipeline.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - Container build and run configuration.

## Run locally

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Run with Docker

Build:

```bash
docker build -t lunglens-backend .
```

Run:

```bash
docker run --rm -p 7860:7860 lunglens-backend
```

## Example API calls

Health:

```bash
curl http://127.0.0.1:7860/healthz
```

Analyze without questionnaire:

```bash
curl -X POST "http://127.0.0.1:7860/api/v1/analyze" \
  -F "image=@/path/to/chest_xray.png"
```

Analyze with questionnaire:

```bash
curl -X POST "http://127.0.0.1:7860/pipeline/analyze" \
  -F "image=@/path/to/chest_xray.png" \
  -F 'questionnaire={"patient_data":{"age":62,"fever":true,"cough_duration_days":4},"notes":"sample"}'
```
