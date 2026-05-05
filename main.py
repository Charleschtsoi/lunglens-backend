import json
import os
import time
from io import BytesIO
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    from torchvision import transforms as T

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class PatientData(BaseModel):
    age: int
    fever: bool
    cough_duration_days: int


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_origins(value: str) -> list[str]:
    cleaned = [origin.strip() for origin in value.split(",") if origin.strip()]
    return cleaned if cleaned else ["*"]


app = FastAPI(title="LungLens API")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
ALLOWED_ORIGINS = _parse_origins(
    os.getenv("ALLOWED_ORIGINS", "*" if ENVIRONMENT != "production" else "")
)
REQUIRE_API_KEY = _parse_bool_env(
    "REQUIRE_API_KEY", default=(ENVIRONMENT == "production")
)
API_KEY = os.getenv("API_KEY", "").strip()
MAX_UPLOAD_MB = max(int(os.getenv("MAX_UPLOAD_MB", "10")), 1)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_IMAGE_MIME_TYPES = {
    item.strip().lower()
    for item in os.getenv(
        "ALLOWED_IMAGE_MIME_TYPES", "image/jpeg,image/png,image/webp"
    ).split(",")
    if item.strip()
}

MODEL1_PATH = os.getenv("MODEL1_PATH", "").strip()
MODEL1_LABELS = ["Normal", "Pneumonia-Bacteria", "Pneumonia-Virus"]

if ENVIRONMENT == "production" and ("*" in ALLOWED_ORIGINS or not ALLOWED_ORIGINS):
    raise RuntimeError(
        "Production requires explicit ALLOWED_ORIGINS (comma-separated, no wildcard)."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PLACEHOLDER_HEATMAP_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5WkR8AAAAASUVORK5CYII="
)

# Lazy-loaded model — populated on first request
_model1: Any = None
_model1_loaded: bool = False

_MODEL1_TRANSFORM = (
    T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if _TORCH_AVAILABLE
    else None
)


def _load_model1() -> Any:
    global _model1, _model1_loaded
    if _model1_loaded:
        return _model1
    _model1_loaded = True

    if not _TORCH_AVAILABLE or not MODEL1_PATH:
        return None
    if not os.path.isfile(MODEL1_PATH):
        return None

    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(MODEL1_LABELS))
        checkpoint = torch.load(MODEL1_PATH, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        _model1 = model
        return _model1
    except Exception:
        return None


def _run_model1(image_bytes: bytes) -> tuple[dict[str, float] | None, int]:
    model = _load_model1()
    if model is None or _MODEL1_TRANSFORM is None:
        return None, 0
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = _MODEL1_TRANSFORM(img).unsqueeze(0)
        t0 = time.monotonic()
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].tolist()
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return {label: round(float(p), 4) for label, p in zip(MODEL1_LABELS, probs)}, elapsed_ms
    except Exception:
        return None, 0


def _error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": message},
    )


def _safe_parse_questionnaire(questionnaire: str | None) -> dict[str, Any] | None:
    if questionnaire is None or questionnaire.strip() == "":
        return None

    try:
        parsed = json.loads(questionnaire)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid questionnaire JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Questionnaire must be a JSON object.")

    patient_data = parsed.get("patient_data")
    if patient_data is not None:
        try:
            parsed["patient_data"] = PatientData.model_validate(patient_data).model_dump()
        except ValidationError as exc:
            raise ValueError(f"Invalid patient_data schema: {exc.errors()}") from exc

    return parsed


def _validate_api_key(x_api_key: str | None) -> JSONResponse | None:
    if not REQUIRE_API_KEY:
        return None
    if not API_KEY:
        return _error_response("Server API key is not configured.", 500)
    if x_api_key != API_KEY:
        return _error_response("Unauthorized.", 401)
    return None


def _build_pipeline_outputs_real(
    model1_preds: dict[str, float],
    questionnaire_data: dict[str, Any] | None,
    inference_ms: int,
) -> dict[str, Any]:
    normal_score = model1_preds.get("Normal", 0.0)
    bact_score = model1_preds.get("Pneumonia-Bacteria", 0.0)
    virus_score = model1_preds.get("Pneumonia-Virus", 0.0)

    top_label = max(model1_preds, key=model1_preds.get)  # type: ignore[arg-type]
    top_confidence = model1_preds[top_label]

    if top_label == "Normal":
        stage1_label = "Normal"
        stage1_confidence = round(normal_score, 3)
    else:
        stage1_label = top_label
        stage1_confidence = round(top_confidence, 3)

    if stage1_label == "Normal":
        stage2_label = "Normal"
        stage2_confidence = round(normal_score, 3)
    elif stage1_label == "Pneumonia-Bacteria":
        stage2_label = "Pneumonia"
        stage2_confidence = round(bact_score, 3)
    else:
        stage2_label = "Viral Pneumonia"
        stage2_confidence = round(virus_score, 3)

    gate_route = "early_stop" if stage1_label == "Normal" else "continue"
    gate_reason = "both_negative" if gate_route == "early_stop" else "positive_detected"
    requires_questionnaire = gate_route == "continue" and questionnaire_data is None

    stage3 = None
    report = None
    positive_score = max(bact_score, virus_score)

    if not requires_questionnaire:
        if gate_route == "early_stop":
            severity, risk_level, recovery_outlook = "low", "low", "favorable"
        elif positive_score >= 0.8:
            severity, risk_level, recovery_outlook = "high", "high", "uncertain"
        elif positive_score >= 0.6:
            severity, risk_level, recovery_outlook = "moderate", "medium", "guarded"
        else:
            severity, risk_level, recovery_outlook = "low", "low", "favorable"

        stage3 = {
            "enabled": True,
            "severity": severity,
            "risk_level": risk_level,
            "recovery_outlook": recovery_outlook,
        }
        report = {
            "summary": (
                f"AI analysis indicates {stage2_label} "
                f"(confidence: {top_confidence:.0%}). "
                "Findings are for educational review only — consult a radiologist."
            ),
            "recommended_actions": [
                "Discuss these findings with your doctor or a licensed radiologist.",
                "Correlate with your symptoms, vitals, and clinical history.",
                "Follow up with repeat imaging if clinically indicated.",
            ],
            "disclaimer": "This output is AI-generated and not a medical diagnosis.",
        }

    stage3_ms = 28 if stage3 else 0
    stage4_ms = 35 if report else 0

    return {
        "success": True,
        "predictions": model1_preds,
        "gradcam": {
            "heatmap_base64": PLACEHOLDER_HEATMAP_BASE64,
            "top_prediction": top_label,
            "confidence": round(top_confidence, 3),
        },
        "stage1": {"label": stage1_label, "confidence": stage1_confidence},
        "stage2": {"label": stage2_label, "confidence": stage2_confidence},
        "gate": {"route": gate_route, "reason": gate_reason},
        "stage3": stage3,
        "report": report,
        "timing_ms": {
            "stage1": inference_ms,
            "stage2": 5,
            "stage3": stage3_ms,
            "stage4": stage4_ms,
            "total": inference_ms + 5 + stage3_ms + stage4_ms,
        },
        "requires_questionnaire": requires_questionnaire,
        "provenance": {
            "run_mode": "model",
            "model1": {"source": "model", "status": "ok", "model_id": "resnet50-lunglens", "model_version": "v1"},
            "model2": {"source": "rules", "status": "derived"},
            "model3": {"source": "mock", "status": "pending"},
            "clinical_risk": (
                {"source": "rules", "status": "ok"} if stage3 else {"source": "rules", "status": "skipped"}
            ),
            "model4": (
                {"source": "rules", "status": "ok"} if report else {"source": "rules", "status": "skipped"}
            ),
        },
    }


# Mock fallback — used when torch is unavailable or MODEL1_PATH is not set

_MOCK_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def _build_mock_predictions(image_bytes: bytes) -> dict[str, float]:
    checksum = sum(image_bytes) % 1000
    return {
        label: round(((checksum + (idx + 1) * 37) % 100) / 100.0, 3)
        for idx, label in enumerate(_MOCK_LABELS)
    }


def _build_pipeline_outputs_mock(
    predictions: dict[str, float], questionnaire_data: dict[str, Any] | None
) -> dict[str, Any]:
    top_prediction = max(predictions, key=predictions.get)  # type: ignore[arg-type]
    top_confidence = float(predictions[top_prediction])

    pneumonia_score = float(predictions["Pneumonia"])
    edema_score = float(predictions["Edema"])
    infiltration_score = float(predictions["Infiltration"])

    stage1_positive_score = max(pneumonia_score, infiltration_score)
    stage1_label = "Pneumonia" if stage1_positive_score >= 0.5 else "Normal"
    stage1_confidence = round(
        stage1_positive_score if stage1_label == "Pneumonia" else 1 - stage1_positive_score, 3
    )

    if stage1_label == "Normal":
        stage2_label = "Normal"
        stage2_confidence = round(max(0.55, 1 - edema_score), 3)
    elif edema_score > 0.65:
        stage2_label = "Lung Opacity"
        stage2_confidence = round(edema_score, 3)
    elif pneumonia_score >= infiltration_score:
        stage2_label = "Viral Pneumonia"
        stage2_confidence = round(pneumonia_score, 3)
    else:
        stage2_label = "Other"
        stage2_confidence = round(infiltration_score, 3)

    gate_route = "continue" if stage1_label != "Normal" or stage2_label != "Normal" else "early_stop"
    gate_reason = "positive_detected" if gate_route == "continue" else "both_negative"
    requires_questionnaire = gate_route == "continue" and questionnaire_data is None

    stage3 = None
    report = None
    if not requires_questionnaire:
        if gate_route == "early_stop":
            severity, risk_level, recovery_outlook = "low", "low", "favorable"
        elif stage1_positive_score >= 0.8:
            severity, risk_level, recovery_outlook = "high", "high", "uncertain"
        elif stage1_positive_score >= 0.6:
            severity, risk_level, recovery_outlook = "moderate", "medium", "guarded"
        else:
            severity, risk_level, recovery_outlook = "low", "low", "favorable"

        stage3 = {"enabled": True, "severity": severity, "risk_level": risk_level, "recovery_outlook": recovery_outlook}
        report = {
            "summary": (
                f"Mock analysis suggests {stage2_label} with top finding "
                f"{top_prediction} ({top_confidence:.2f})."
            ),
            "recommended_actions": [
                "Review this result with a licensed radiologist.",
                "Correlate with patient symptoms and vitals.",
                "Repeat imaging or follow-up per clinical protocol if needed.",
            ],
            "disclaimer": "This is a PoC mock output and not a medical diagnosis.",
        }

    stage3_ms = 28 if stage3 else 0
    stage4_ms = 40 if report else 0

    return {
        "success": True,
        "predictions": predictions,
        "gradcam": {
            "heatmap_base64": PLACEHOLDER_HEATMAP_BASE64,
            "top_prediction": top_prediction,
            "confidence": round(top_confidence, 3),
        },
        "stage1": {"label": stage1_label, "confidence": stage1_confidence},
        "stage2": {"label": stage2_label, "confidence": stage2_confidence},
        "gate": {"route": gate_route, "reason": gate_reason},
        "stage3": stage3,
        "report": report,
        "timing_ms": {
            "stage1": 18,
            "stage2": 22,
            "stage3": stage3_ms,
            "stage4": stage4_ms,
            "total": 18 + 22 + stage3_ms + stage4_ms,
        },
        "requires_questionnaire": requires_questionnaire,
    }


async def _analyze_internal(
    image: UploadFile, questionnaire: str | None, x_api_key: str | None
) -> JSONResponse:
    try:
        auth_error = _validate_api_key(x_api_key)
        if auth_error is not None:
            return auth_error

        if not image.filename:
            return _error_response("Missing uploaded image filename.", 400)

        content_type = (image.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_MIME_TYPES:
            return _error_response(
                f"Unsupported image content type: {content_type or 'unknown'}.", 415
            )

        image_bytes = await image.read()
        if not image_bytes:
            return _error_response("Uploaded image is empty.", 400)
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            return _error_response(
                f"Uploaded image exceeds max size of {MAX_UPLOAD_MB} MB.", 413
            )

        try:
            Image.open(BytesIO(image_bytes)).verify()
        except Exception:
            return _error_response("Uploaded file is not a valid image.", 400)

        questionnaire_data = _safe_parse_questionnaire(questionnaire)

        model1_preds, inference_ms = _run_model1(image_bytes)
        if model1_preds is not None:
            payload = _build_pipeline_outputs_real(model1_preds, questionnaire_data, inference_ms)
        else:
            mock_preds = _build_mock_predictions(image_bytes)
            payload = _build_pipeline_outputs_mock(mock_preds, questionnaire_data)

        return JSONResponse(status_code=200, content=payload)
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except Exception:
        return _error_response("Internal server error.", 500)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/health")
async def health() -> dict[str, Any]:
    model = _load_model1()
    return {
        "status": "healthy",
        "torch_available": _TORCH_AVAILABLE,
        "model1": {
            "loaded": model is not None,
            "path_configured": bool(MODEL1_PATH),
            "path_exists": os.path.isfile(MODEL1_PATH) if MODEL1_PATH else False,
            "labels": MODEL1_LABELS,
        },
    }


@app.post("/predict/densenet")
async def predict_densenet(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    auth_error = _validate_api_key(x_api_key)
    if auth_error is not None:
        return auth_error
    return JSONResponse(
        status_code=503,
        content={"success": False, "error": "DenseNet-121 model not loaded."},
    )


@app.post("/api/v1/analyze")
async def analyze_v1(
    image: UploadFile = File(...),
    questionnaire: str | None = Form(None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    return await _analyze_internal(
        image=image, questionnaire=questionnaire, x_api_key=x_api_key
    )


@app.post("/pipeline/analyze")
async def analyze_pipeline_alias(
    image: UploadFile = File(...),
    questionnaire: str | None = Form(None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    return await _analyze_internal(
        image=image, questionnaire=questionnaire, x_api_key=x_api_key
    )
