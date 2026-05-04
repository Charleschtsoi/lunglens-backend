# LungLens Backend Debugging Guide

Use this when the frontend keeps showing mock/fallback behavior and you expect live model inference.

This project uses FastAPI + TensorFlow/Keras `.h5` (not PyTorch), so troubleshoot with the stack below.

## 1) Basics first (LungLens-specific)

### Check backend health

```bash
curl -sS "http://127.0.0.1:7860/health"
```

Interpret key fields:

- `models.model2_h5.enabled=false`: `ENABLE_MODEL2_H5` / `ENABLE_H5_MODEL` is off.
- `models.model2_h5.loaded=false`: model did not load.
- `models.model2_h5.error`: startup load failure reason (read this first).

### Run backend manually and watch logs

```bash
ENVIRONMENT=development \
REQUIRE_API_KEY=false \
ENABLE_MODEL2_H5=true \
H5_MODEL2_PATH="/absolute/path/to/resnet152v2_lung_disease_final.h5" \
H5_MODEL2_LABELS="Normal,Lung_Opacity,Viral_Pneumonia" \
uvicorn main:app --host 127.0.0.1 --port 7860
```

(Legacy env names `ENABLE_H5_MODEL`, `H5_MODEL_PATH`, `H5_STAGE2_LABELS` still work if the new keys are unset.)

Watch startup and first request logs. Do not diagnose from stale background processes.

## 2) Most common causes in this repo

### Wrong model path

`H5_MODEL2_PATH` (or legacy `H5_MODEL_PATH`) is resolved from process context; use an absolute path during debugging.

```bash
ls -la "/absolute/path/to/resnet152v2_lung_disease_final.h5"
```

### File missing or not synced

If model is not in repo, confirm local file exists and is non-trivial size.

### Missing dependencies

Typical errors:

- `No module named 'tensorflow'`
- `No module named 'h5py'`

Fix:

```bash
pip install -r requirements.txt
```

### Keras/TensorFlow compatibility mismatch

Colab-exported `.h5` often uses Keras 3-style metadata (`__keras_tensor__` inbound nodes). This repo pins:

- `tensorflow==2.18.0`
- `h5py==3.11.0`

in [`requirements.txt`](requirements.txt). Older TensorFlow versions (for example 2.15.x) often fail to deserialize these files.

### Multiple uvicorn servers

If two local servers run on different ports, frontend may hit the wrong one (old code or mock-only config). Keep one canonical process.

```bash
pkill -f "uvicorn main:app" 2>/dev/null
```

## 3) Why frontend can appear “always mock”

### Confirm frontend is calling the intended backend

Check `BACKEND_API_BASE_URL` and Network tab requests.

### Confirm response contains provenance

`POST /pipeline/analyze` should include:

- `provenance.run_mode`
- `provenance.model2.source`
- `provenance.model2.status`
- optional `warnings[]`

If `provenance` is missing, frontend is likely hitting an older backend build or wrong URL.

### Interpret run mode

- `run_mode=mock`: ML Model 2 did not run successfully (flag off, load failed, or inference fallback).
- `run_mode=hybrid`: ML Model 2 ran successfully, while other models are still mock/rule assisted.
- `run_mode=real`: only when backend is configured to count live ML Model 2 as full real mode.

Correlate with `/health` and `warnings[]` for degraded-mode notices.

## 4) Standalone TensorFlow sanity test

Run this from backend root:

```python
import os
import tensorflow as tf

model_path = "/absolute/path/to/resnet152v2_lung_disease_final.h5"
print("exists:", os.path.exists(model_path))
print("size_mb:", round(os.path.getsize(model_path) / 1e6, 2) if os.path.exists(model_path) else "n/a")

model = tf.keras.models.load_model(model_path, compile=False)
print("loaded_ok:", type(model).__name__)
```

If this fails, use the exact exception as the primary debugging signal.

## 5) Quick E2E smoke for provenance

```bash
python3 -c "from PIL import Image; Image.new('RGB',(224,224),(90,90,90)).save('/tmp/lunglens_e2e.png')"

curl -sS -X POST "http://127.0.0.1:7860/pipeline/analyze" \
  -F "image=@/tmp/lunglens_e2e.png"
```

Expected when ML Model 2 is truly live:

- `provenance.model2.source == "model"`
- `provenance.model2.status == "ok"`
- `provenance.run_mode` transitions out of `mock` (usually `hybrid` by default).

## 6) Source locations in this codebase

See [`main.py`](main.py):

- `_load_model2_h5()`
- `_load_h5_model_compat()`
- `_run_h5_model2()`
- `GET /health`
- `POST /pipeline/analyze`
