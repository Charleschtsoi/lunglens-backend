# LungLens Model Owner Spec

Use this spec for any teammate training a model for LungLens.  
Goal: ensure every model can be integrated into backend/frontend without breaking contract.

## Quick Step Navigation

[Step 1](#step-1-api-contract-essentials) | [Step 2](#step-2-per-model-output-requirements) | [Step 3](#step-3-mandatory-handoff-package) | [Step 4](#step-4-validation-and-submission)

## Step 1: API Contract Essentials

### 1) Non-Negotiable Integration Rule

Model internals can change, but API contract to frontend must remain stable.

Backend response keys that must remain present:

- `success`
- `predictions`
- `gradcam`
- `model1`
- `model2`
- `gate`
- `model3`
- `model4`
- `timing_ms`
- `requires_questionnaire`

### 2) API Request/Response Contract

#### Request (`multipart/form-data`)

- `image`: file (required)
- `questionnaire`: JSON string (optional)

Example questionnaire:

```json
{
  "patient_data": {
    "age": 52,
    "fever": true,
    "cough_duration_days": 5
  },
  "gender": "female",
  "smoking_status": "former",
  "lung_capacity": 72,
  "hospital_visits": 2
}
```

#### Response behavior

- If `gate.route == "continue"` and questionnaire is missing:
  - `requires_questionnaire: true`
  - `model4: null`
- If questionnaire is present:
  - `requires_questionnaire: false`
  - include `model3` + `model4`

[Back to Steps](#quick-step-navigation) | [Next: Step 2](#step-2-per-model-output-requirements)

## Step 2: Per-Model Output Requirements

### A) ML Model 1 — Binary Pneumonia Model

Dataset: Kaggle pediatric X-ray (Normal vs Pneumonia).

Accepted raw outputs:
- Sigmoid: `p_pneumonia`
- Softmax: `[p_normal, p_pneumonia]`

Must map to:
- `model1.label` in `{"Pneumonia", "Normal"}`
- `model1.confidence` in `[0,1]`

Recommended label order:

```json
["Normal", "Pneumonia"]
```

### B) ML Model 2 — Multi-Class Image Model

Dataset: Normal / Lung Opacity / Viral Pneumonia.

Required raw output:
- 3-class softmax in fixed index order.

Canonical order:

```json
["Normal", "Lung Opacity", "Viral Pneumonia"]
```

Canonical preprocessing for current ML Model 2 `.h5` integration:

- RGB input
- resize to `224x224`
- normalize with `/255.0` to `[0,1]`
- do not apply `resnet_v2.preprocess_input` in this path

Must map to:
- `model2.label` in `{"Normal", "Lung Opacity", "Viral Pneumonia", "Other"}`
- `model2.confidence` in `[0,1]`

### C) ML Model 3 — Structured Clinical Model

Dataset: tabular records (age, gender, smoking status, lung capacity, treatment/recovery-related fields).

Required outputs:
- `severity` in `{"low", "moderate", "high"}`
- `risk_level` in `{"low", "medium", "high"}`
- `recovery_outlook` in `{"favorable", "guarded", "uncertain"}`

This becomes backend `model3`.

### D) Optional MIMIC-CXR Track

If a model is used to populate backend `predictions`, it must output all 14 probabilities in `[0,1]`:

- `Atelectasis`
- `Cardiomegaly`
- `Effusion`
- `Infiltration`
- `Mass`
- `Nodule`
- `Pneumonia`
- `Pneumothorax`
- `Consolidation`
- `Edema`
- `Emphysema`
- `Fibrosis`
- `Pleural_Thickening`
- `Hernia`

### Grad-CAM Requirement

- Grad-CAM must be generated for image model inference.
- Backend returns:
  - `gradcam.heatmap_base64` (PNG base64)
  - `gradcam.top_prediction`
  - `gradcam.confidence`
- Heatmap should resemble standard Grad-CAM overlays (high activation in warm colors).

[Back: Step 1](#step-1-api-contract-essentials) | [Next: Step 3](#step-3-mandatory-handoff-package)

## Step 3: Mandatory Handoff Package

Every teammate must submit all items below:

1. Model artifact (`.h5`, `.pth`, `.safetensors`, etc.)
2. `labels.json` (exact index-to-class order)
3. `preprocess.json`:
   - image size
   - color mode (RGB/BGR)
   - normalization/scaling
   - any model-specific preprocessing
4. `inference_example.json`:
   - one sample input reference
   - raw model output vector
   - mapped class output
5. Framework/runtime versions:
   - TensorFlow or PyTorch version
   - key dependency versions

No integration starts until all 5 are provided.

[Back: Step 2](#step-2-per-model-output-requirements) | [Next: Step 4](#step-4-validation-and-submission)

## Step 4: Validation and Submission

### Validation Checklist Before Merge

- Label order verified against `labels.json`
- Preprocessing exactly matches training pipeline
- Output maps correctly to model1–model4 schema
- API contract keys unchanged
- Error handling still valid (`401`, `413`, `415`, `400`, `500`)
- End-to-end sample request works on:
  - `/api/v1/analyze`
  - `/pipeline/analyze`

### Submission Template (copy and fill)

```md
Model owner:
Model name/version:
Task model slot (1/2/3/optional):
Dataset used:

Class order (exact):
- 0:
- 1:
- 2:

Preprocess:
- input_size:
- color_mode:
- normalization:
- extra transforms:

Artifact path:
Framework versions:

Sample inference:
- sample input:
- raw output:
- mapped output:
```

[Back: Step 3](#step-3-mandatory-handoff-package) | [Back to Steps](#quick-step-navigation)
