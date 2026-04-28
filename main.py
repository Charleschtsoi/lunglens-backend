import json
import os
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from PIL import Image


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

if ENVIRONMENT == "production" and ("*" in ALLOWED_ORIGINS or not ALLOWED_ORIGINS):
    raise RuntimeError(
        "Production requires explicit ALLOWED_ORIGINS (comma-separated, no wildcard)."
    )

# TODO: Restrict CORS origins/methods/headers before production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

PLACEHOLDER_HEATMAP_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5WkR8AAAAASUVORK5CYII="
)


def _error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": message,
        },
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


def _build_mock_predictions(image_bytes: bytes) -> dict[str, float]:
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    checksum = sum(image_bytes) % 1000
    predictions: dict[str, float] = {}
    for idx, label in enumerate(LABELS):
        score = ((checksum + (idx + 1) * 37) % 100) / 100.0
        predictions[label] = round(score, 3)

    return predictions


def _build_pipeline_outputs(
    predictions: dict[str, float], questionnaire_data: dict[str, Any] | None
) -> dict[str, Any]:
    top_prediction = max(predictions, key=predictions.get)
    top_confidence = float(predictions[top_prediction])

    pneumonia_score = float(predictions["Pneumonia"])
    edema_score = float(predictions["Edema"])
    infiltration_score = float(predictions["Infiltration"])

    stage1_positive_score = max(pneumonia_score, infiltration_score)
    stage1_label = "Pneumonia" if stage1_positive_score >= 0.5 else "Normal"
    stage1_confidence = round(stage1_positive_score if stage1_label == "Pneumonia" else 1 - stage1_positive_score, 3)

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
            severity = "low"
            risk_level = "low"
            recovery_outlook = "favorable"
        else:
            if stage1_positive_score >= 0.8:
                severity = "high"
                risk_level = "high"
                recovery_outlook = "uncertain"
            elif stage1_positive_score >= 0.6:
                severity = "moderate"
                risk_level = "medium"
                recovery_outlook = "guarded"
            else:
                severity = "low"
                risk_level = "low"
                recovery_outlook = "favorable"

        stage3 = {
            "enabled": True,
            "severity": severity,
            "risk_level": risk_level,
            "recovery_outlook": recovery_outlook,
        }

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
            "disclaimer": (
                "This is a PoC mock output and not a medical diagnosis."
            ),
        }

    stage3_timing = 28 if stage3 else 0
    stage4_timing = 40 if report else 0
    timing_ms = {
        "stage1": 18,
        "stage2": 22,
        "stage3": stage3_timing,
        "stage4": stage4_timing,
        "total": 18 + 22 + stage3_timing + stage4_timing,
    }

    return {
        "success": True,
        "predictions": predictions,
        "gradcam": {
            "heatmap_base64": PLACEHOLDER_HEATMAP_BASE64,
            "top_prediction": top_prediction,
            "confidence": round(top_confidence, 3),
        },
        "stage1": {
            "label": stage1_label,
            "confidence": stage1_confidence,
        },
        "stage2": {
            "label": stage2_label,
            "confidence": stage2_confidence,
        },
        "gate": {
            "route": gate_route,
            "reason": gate_reason,
        },
        "stage3": stage3,
        "report": report,
        "timing_ms": timing_ms,
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
        predictions = _build_mock_predictions(image_bytes)
        payload = _build_pipeline_outputs(predictions, questionnaire_data)
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
