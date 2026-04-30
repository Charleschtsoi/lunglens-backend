import json
import os
import base64
import random
import math
import logging
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from PIL import Image


class PatientData(BaseModel):
    age: int
    fever: bool
    cough_duration_days: int


class AssessRequest(BaseModel):
    clinical_data: dict[str, Any]
    stage1_label: str
    stage1_confidence: float
    stage2_label: str
    stage2_confidence: float


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_origins(value: str) -> list[str]:
    cleaned = [origin.strip() for origin in value.split(",") if origin.strip()]
    return cleaned if cleaned else ["*"]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value.strip())
    except ValueError:
        return default


app = FastAPI(title="LungLens API")
logger = logging.getLogger("lunglens.backend")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
ALLOWED_ORIGINS = _parse_origins(
    os.getenv("ALLOWED_ORIGINS", "*" if ENVIRONMENT != "production" else "")
)
REQUIRE_API_KEY = _parse_bool_env(
    "REQUIRE_API_KEY", default=True
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
ENABLE_H5_MODEL = _parse_bool_env("ENABLE_H5_MODEL", default=False)
H5_MODEL_PATH = os.getenv(
    "H5_MODEL_PATH", "models/resnet152v2_lung_disease_final.h5"
).strip()
H5_STAGE2_LABELS = _parse_csv(
    os.getenv("H5_STAGE2_LABELS", "Normal,Lung Opacity,Viral Pneumonia")
)
STAGE2_UNCERTAINTY_ENABLED = _parse_bool_env(
    "STAGE2_UNCERTAINTY_ENABLED", default=False
)
STAGE2_UNCERTAINTY_MIN_CONFIDENCE = _parse_float_env(
    "STAGE2_UNCERTAINTY_MIN_CONFIDENCE", 0.55
)
STAGE2_UNCERTAINTY_MIN_MARGIN = _parse_float_env("STAGE2_UNCERTAINTY_MIN_MARGIN", 0.1)

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

MOCK_PROFILES = [
    {
        "name": "pneumonia_focus",
        "offset": 11,
        "boost_labels": {"Pneumonia", "Infiltration", "Consolidation"},
        "suppress_labels": {"Hernia", "Fibrosis"},
        "hotspots": [
            (0.42, 0.36, 1.0, 0.14),
            (0.58, 0.52, 0.75, 0.19),
        ],
    },
    {
        "name": "edema_focus",
        "offset": 37,
        "boost_labels": {"Edema", "Effusion", "Cardiomegaly"},
        "suppress_labels": {"Pneumothorax", "Mass"},
        "hotspots": [
            (0.50, 0.45, 1.0, 0.18),
            (0.34, 0.57, 0.65, 0.16),
        ],
    },
    {
        "name": "mostly_normal",
        "offset": 73,
        "boost_labels": {"Nodule", "Pleural_Thickening"},
        "suppress_labels": {"Pneumonia", "Edema", "Infiltration", "Effusion"},
        "hotspots": [
            (0.46, 0.42, 0.55, 0.11),
            (0.62, 0.35, 0.45, 0.09),
        ],
    },
]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _gradcam_color(intensity: float) -> tuple[int, int, int]:
    # Grad-CAM-like colormap: deep blue -> cyan -> yellow -> red.
    t = _clamp01(intensity)
    if t < 0.35:
        p = t / 0.35
        r = int(10 + p * 30)
        g = int(20 + p * 140)
        b = int(100 + p * 155)
    elif t < 0.7:
        p = (t - 0.35) / 0.35
        r = int(40 + p * 215)
        g = int(160 + p * 70)
        b = int(255 - p * 185)
    else:
        p = (t - 0.7) / 0.3
        r = 255
        g = int(230 - p * 170)
        b = int(70 - p * 45)
    return r, g, b


def _create_placeholder_heatmap_base64(
    hotspots: list[tuple[float, float, float, float]], size: int = 256
) -> str:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    for y in range(size):
        for x in range(size):
            nx = x / (size - 1)
            ny = y / (size - 1)
            intensity = 0.0
            for cx, cy, amplitude, sigma in hotspots:
                dx = nx - cx
                dy = ny - cy
                gaussian = amplitude * math.exp(-((dx * dx + dy * dy) / (2 * sigma * sigma)))
                intensity += gaussian

            intensity = _clamp01(intensity)
            if intensity < 0.05:
                continue

            alpha = int(20 + intensity * 210)
            red, green, blue = _gradcam_color(intensity)
            image.putpixel((x, y), (red, green, blue, alpha))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


MOCK_HEATMAPS = {
    profile["name"]: _create_placeholder_heatmap_base64(profile["hotspots"])
    for profile in MOCK_PROFILES
}
H5_MODEL: Any = None
H5_MODEL_LOAD_ERROR: str | None = None


def _error_response(
    message: str,
    status_code: int,
    error_code: str = "internal_error",
    stage: str = "pipeline",
    retryable: bool = False,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": message,
            "error_code": error_code,
            "stage": stage,
            "retryable": retryable,
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
        return _error_response(
            "Server API key is not configured.",
            500,
            error_code="invalid_api_key",
            stage="pipeline",
            retryable=False,
        )
    if x_api_key != API_KEY:
        return _error_response(
            "Unauthorized.",
            401,
            error_code="invalid_api_key",
            stage="pipeline",
            retryable=False,
        )
    return None


def _models_loaded_status() -> bool:
    if not ENABLE_H5_MODEL:
        return True
    return H5_MODEL is not None and H5_MODEL_LOAD_ERROR is None


def _load_h5_model() -> None:
    global H5_MODEL, H5_MODEL_LOAD_ERROR

    if not ENABLE_H5_MODEL:
        H5_MODEL = None
        H5_MODEL_LOAD_ERROR = None
        return

    if len(H5_STAGE2_LABELS) != 3:
        H5_MODEL = None
        H5_MODEL_LOAD_ERROR = "H5_STAGE2_LABELS must contain exactly 3 labels."
        return

    try:
        import tensorflow as tf  # type: ignore

        H5_MODEL = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
        H5_MODEL_LOAD_ERROR = None
    except Exception as exc:
        initial_error = str(exc)
        try:
            H5_MODEL = _load_h5_model_with_inputlayer_compat(tf, H5_MODEL_PATH)
            H5_MODEL_LOAD_ERROR = None
            logger.warning(
                "Loaded H5 model with compatibility fallback after initial error: %s",
                initial_error,
            )
            return
        except Exception as compat_exc:
            H5_MODEL = None
            H5_MODEL_LOAD_ERROR = (
                f"Initial load failed: {initial_error}. "
                f"Compatibility fallback failed: {compat_exc}"
            )


def _sanitize_inputlayer_config(node: Any) -> Any:
    if isinstance(node, dict):
        config = node.get("config")
        if isinstance(config, dict):
            if "batch_shape" in config and "batch_input_shape" not in config:
                config["batch_input_shape"] = config["batch_shape"]
            config.pop("batch_shape", None)
            config.pop("optional", None)
            config.pop("quantization_config", None)
            dtype_policy = config.get("dtype")
            if isinstance(dtype_policy, dict):
                # Older runtimes cannot deserialize newer policy objects.
                # We only need inference compatibility, so use a plain dtype string.
                policy_name = (
                    dtype_policy.get("config", {}).get("name")
                    if isinstance(dtype_policy.get("config"), dict)
                    else None
                )
                config["dtype"] = (
                    "float32" if policy_name == "mixed_float16" else policy_name
                ) or "float32"
        for value in node.values():
            _sanitize_inputlayer_config(value)
    elif isinstance(node, list):
        for item in node:
            _sanitize_inputlayer_config(item)
    return node


def _load_h5_model_with_inputlayer_compat(tf: Any, model_path: str) -> Any:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise ValueError(
            f"H5 model compatibility fallback requires h5py: {exc}"
        ) from exc

    with h5py.File(model_path, "r") as h5_file:
        raw_config = h5_file.attrs.get("model_config")

    if raw_config is None:
        raise ValueError("H5 model has no model_config metadata.")

    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")

    model_config = json.loads(raw_config)
    sanitized_config = _sanitize_inputlayer_config(model_config)
    model_json = json.dumps(sanitized_config)
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(model_path)
    return model


def _run_h5_stage2_prediction(image_bytes: bytes) -> tuple[str, float, list[float]]:
    if H5_MODEL is None:
        if H5_MODEL_LOAD_ERROR:
            raise ValueError(f"H5 model unavailable: {H5_MODEL_LOAD_ERROR}")
        raise ValueError("H5 model is not loaded.")

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise ValueError(f"NumPy import failed for H5 inference: {exc}") from exc

    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype="float32") / 255.0
    batch = np.expand_dims(array, axis=0)

    raw = H5_MODEL.predict(batch, verbose=0)
    probs = raw[0].tolist()
    if len(probs) != len(H5_STAGE2_LABELS):
        raise ValueError(
            f"H5 output size mismatch: got {len(probs)}, expected {len(H5_STAGE2_LABELS)}."
        )

    max_idx = max(range(len(probs)), key=lambda idx: probs[idx])
    label = H5_STAGE2_LABELS[max_idx]
    confidence = round(float(probs[max_idx]), 3)
    return label, confidence, [float(item) for item in probs]


def _apply_stage2_uncertainty(
    label: str, confidence: float, probs: list[float]
) -> tuple[str, float]:
    if not STAGE2_UNCERTAINTY_ENABLED:
        return label, confidence

    sorted_probs = sorted(probs, reverse=True)
    top_score = float(sorted_probs[0]) if sorted_probs else 0.0
    second_score = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_score - second_score
    if top_score < STAGE2_UNCERTAINTY_MIN_CONFIDENCE or margin < STAGE2_UNCERTAINTY_MIN_MARGIN:
        return "Other", round(top_score, 3)

    return label, confidence


def _apply_stage2_signal_to_predictions(
    predictions: dict[str, float], stage2_label: str, stage2_confidence: float
) -> dict[str, float]:
    adjusted = dict(predictions)
    if stage2_label == "Viral Pneumonia":
        adjusted["Pneumonia"] = round(max(adjusted["Pneumonia"], stage2_confidence), 3)
        adjusted["Infiltration"] = round(
            max(adjusted["Infiltration"], stage2_confidence * 0.92), 3
        )
        adjusted["Consolidation"] = round(
            max(adjusted["Consolidation"], stage2_confidence * 0.85), 3
        )
    elif stage2_label == "Lung Opacity":
        adjusted["Infiltration"] = round(
            max(adjusted["Infiltration"], stage2_confidence * 0.9), 3
        )
        adjusted["Effusion"] = round(max(adjusted["Effusion"], stage2_confidence * 0.82), 3)
        adjusted["Edema"] = round(max(adjusted["Edema"], stage2_confidence * 0.75), 3)
    elif stage2_label == "Normal":
        adjusted["Pneumonia"] = round(min(adjusted["Pneumonia"], 0.2), 3)
        adjusted["Infiltration"] = round(min(adjusted["Infiltration"], 0.2), 3)
        adjusted["Effusion"] = round(min(adjusted["Effusion"], 0.2), 3)

    return adjusted


def _build_mock_predictions(image_bytes: bytes, profile: dict[str, Any]) -> dict[str, float]:
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    checksum = sum(image_bytes) % 1000
    predictions: dict[str, float] = {}
    offset = int(profile["offset"])
    boost_labels = set(profile["boost_labels"])
    suppress_labels = set(profile["suppress_labels"])

    for idx, label in enumerate(LABELS):
        score = ((checksum + (idx + 1) * (29 + offset)) % 100) / 100.0
        if label in boost_labels:
            score = score * 0.5 + 0.42
        if label in suppress_labels:
            score = score * 0.45
        score = _clamp01(score + random.uniform(-0.03, 0.03))
        predictions[label] = round(score, 3)

    return predictions


def _build_assess_predictions(
    stage1_label: str,
    stage1_confidence: float,
    stage2_label: str,
    stage2_confidence: float,
) -> dict[str, float]:
    predictions = {label: 0.05 for label in LABELS}

    if stage1_label == "Pneumonia":
        predictions["Pneumonia"] = max(0.1, min(1.0, stage1_confidence))
        predictions["Infiltration"] = max(
            predictions["Infiltration"], min(1.0, stage1_confidence * 0.9)
        )
        predictions["Consolidation"] = max(
            predictions["Consolidation"], min(1.0, stage1_confidence * 0.85)
        )

    if stage2_label == "Lung Opacity":
        predictions["Infiltration"] = max(
            predictions["Infiltration"], min(1.0, stage2_confidence * 0.95)
        )
        predictions["Effusion"] = max(
            predictions["Effusion"], min(1.0, stage2_confidence * 0.85)
        )
        predictions["Edema"] = max(
            predictions["Edema"], min(1.0, stage2_confidence * 0.78)
        )
    elif stage2_label == "Viral Pneumonia":
        predictions["Pneumonia"] = max(
            predictions["Pneumonia"], min(1.0, stage2_confidence)
        )
        predictions["Consolidation"] = max(
            predictions["Consolidation"], min(1.0, stage2_confidence * 0.82)
        )
    elif stage2_label == "Normal":
        predictions["Pneumonia"] = min(predictions["Pneumonia"], 0.2)
        predictions["Infiltration"] = min(predictions["Infiltration"], 0.2)

    return {label: round(float(value), 3) for label, value in predictions.items()}


def _select_mock_profile(
    image_bytes: bytes, questionnaire_data: dict[str, Any] | None
) -> dict[str, Any]:
    # Keep no-questionnaire flow predictably testable for frontend.
    # Use only positive-leaning profiles when questionnaire is absent.
    if questionnaire_data is None:
        candidate_profiles = MOCK_PROFILES[:2]
    else:
        candidate_profiles = MOCK_PROFILES

    profile_idx = sum(image_bytes) % len(candidate_profiles)
    return candidate_profiles[profile_idx]


def _build_pipeline_outputs(
    predictions: dict[str, float],
    questionnaire_data: dict[str, Any] | None,
    heatmap_base64: str,
    stage2_override: tuple[str, float] | None = None,
    run_mode: str = "mock",
    stage2_status: str = "ok",
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    top_prediction = max(predictions, key=predictions.get)
    top_confidence = float(predictions[top_prediction])

    pneumonia_score = float(predictions["Pneumonia"])
    edema_score = float(predictions["Edema"])
    infiltration_score = float(predictions["Infiltration"])

    stage1_positive_score = max(pneumonia_score, infiltration_score)
    stage1_label = "Pneumonia" if stage1_positive_score >= 0.5 else "Normal"
    stage1_confidence = round(stage1_positive_score if stage1_label == "Pneumonia" else 1 - stage1_positive_score, 3)

    if stage2_override is not None:
        stage2_label, stage2_confidence = stage2_override
    elif stage1_label == "Normal":
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
            "heatmap_base64": heatmap_base64,
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
        "warnings": warnings or [],
        "provenance": {
            "run_mode": run_mode,
            "stage1": {
                "source": "mock",
                "status": "ok",
                "model_id": "mock-stage1",
                "model_version": "demo-v1",
            },
            "stage2": {
                "source": "model" if run_mode == "hybrid" else "mock",
                "status": stage2_status,
                "model_id": "resnet152v2_lung_disease_final.h5"
                if run_mode == "hybrid"
                else "mock-stage2",
                "model_version": "pilot-v1" if run_mode == "hybrid" else "demo-v1",
            },
            "stage3": {
                "source": "rule",
                "status": "ok" if stage3 else "skipped",
                "model_id": "clinical-rule-engine",
                "model_version": "v1",
            },
            "stage4": {
                "source": "rule",
                "status": "ok" if report else "skipped",
                "model_id": "report-template-engine",
                "model_version": "v1",
            },
            "explanations": [
                {
                    "section": "pipeline-summary",
                    "stage_keys": ["stage1", "stage2", "stage3"],
                    "source_type": "model" if run_mode == "hybrid" else "mock",
                },
                {
                    "section": "report-summary",
                    "stage_keys": ["stage4"],
                    "source_type": "rule",
                },
                {
                    "section": "anatomy-guide",
                    "stage_keys": ["pipeline"],
                    "source_type": "static",
                },
            ],
        },
    }


async def _analyze_internal(
    image: UploadFile, questionnaire: str | None, x_api_key: str | None
) -> JSONResponse:
    try:
        auth_error = _validate_api_key(x_api_key)
        if auth_error is not None:
            return auth_error

        if not image.filename:
            return _error_response(
                "Missing uploaded image filename.",
                400,
                error_code="missing_image",
                stage="pipeline",
                retryable=False,
            )

        content_type = (image.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_MIME_TYPES:
            return _error_response(
                f"Unsupported image content type: {content_type or 'unknown'}.",
                415,
                error_code="unsupported_file_type",
                stage="pipeline",
                retryable=False,
            )

        image_bytes = await image.read()
        if not image_bytes:
            return _error_response(
                "Uploaded image is empty.",
                400,
                error_code="missing_image",
                stage="pipeline",
                retryable=False,
            )
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            return _error_response(
                f"Uploaded image exceeds max size of {MAX_UPLOAD_MB} MB.",
                413,
                error_code="payload_too_large",
                stage="pipeline",
                retryable=False,
            )

        try:
            Image.open(BytesIO(image_bytes)).verify()
        except Exception:
            return _error_response(
                "Uploaded file is not a valid image.",
                400,
                error_code="invalid_request",
                stage="pipeline",
                retryable=False,
            )

        questionnaire_data = _safe_parse_questionnaire(questionnaire)
        selected_profile = _select_mock_profile(image_bytes, questionnaire_data)
        predictions = _build_mock_predictions(image_bytes, selected_profile)
        stage2_override: tuple[str, float] | None = None
        run_mode = "mock"
        stage2_status = "ok"
        warnings: list[dict[str, Any]] = [
            {
                "code": "mock_scaffold_active",
                "message": "Parts of this report use mock/rule-based scaffolding for educational output.",
                "stage": "pipeline",
            }
        ]
        if ENABLE_H5_MODEL:
            run_mode = "hybrid"
            try:
                stage2_label, stage2_confidence, stage2_probs = _run_h5_stage2_prediction(
                    image_bytes
                )
                stage2_label, stage2_confidence = _apply_stage2_uncertainty(
                    stage2_label, stage2_confidence, stage2_probs
                )
                stage2_override = (stage2_label, stage2_confidence)
                predictions = _apply_stage2_signal_to_predictions(
                    predictions, stage2_label, stage2_confidence
                )
            except ValueError as exc:
                stage2_status = "fallback"
                warnings.append(
                    {
                        "code": "stage2_model_unavailable",
                        "message": f"Stage 2 model unavailable; using fallback behavior. {exc}",
                        "stage": "stage2",
                    }
                )
        payload = _build_pipeline_outputs(
            predictions,
            questionnaire_data,
            heatmap_base64=MOCK_HEATMAPS[selected_profile["name"]],
            stage2_override=stage2_override,
            run_mode=run_mode,
            stage2_status=stage2_status,
            warnings=warnings,
        )
        return JSONResponse(status_code=200, content=payload)
    except ValueError as exc:
        return _error_response(
            str(exc),
            400,
            error_code="invalid_request",
            stage="pipeline",
            retryable=False,
        )
    except Exception:
        return _error_response(
            "Internal server error.",
            500,
            error_code="internal_error",
            stage="pipeline",
            retryable=True,
        )


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/health")
async def health() -> dict[str, Any]:
    h5_loaded = H5_MODEL is not None and H5_MODEL_LOAD_ERROR is None
    return {
        "status": "ok",
        "models_loaded": _models_loaded_status(),
        "h5": {
            "enabled": ENABLE_H5_MODEL,
            "path": H5_MODEL_PATH,
            "loaded": h5_loaded,
            "error": H5_MODEL_LOAD_ERROR,
        },
    }


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Any, exc: RequestValidationError
) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    error_loc = ".".join(str(part) for part in first_error.get("loc", []))
    error_msg = first_error.get("msg", "Invalid request input.")
    return _error_response(
        f"{error_loc}: {error_msg}",
        400,
        error_code="invalid_request",
        stage="pipeline",
        retryable=False,
    )


@app.on_event("startup")
async def startup_model_loader() -> None:
    _load_h5_model()


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


@app.post("/predict")
async def predict_compat(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    base_response = await _analyze_internal(
        image=image, questionnaire=None, x_api_key=x_api_key
    )
    if base_response.status_code >= 400:
        return base_response

    payload = json.loads(base_response.body.decode("utf-8"))
    gate_positive = payload["gate"]["route"] == "continue"
    heatmap = payload["gradcam"]["heatmap_base64"]
    return JSONResponse(
        status_code=200,
        content={
            "stage1": payload["stage1"],
            "stage2": payload["stage2"],
            "gradcam": {
                "stage1_heatmap": heatmap,
                "stage2_heatmap": heatmap,
                "overlay": heatmap,
            },
            "gate_result": "positive" if gate_positive else "negative",
            "requires_clinical_qa": gate_positive,
        },
    )


@app.post("/assess")
async def assess_compat(
    request: AssessRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    auth_error = _validate_api_key(x_api_key)
    if auth_error is not None:
        return auth_error

    predictions = _build_assess_predictions(
        request.stage1_label,
        request.stage1_confidence,
        request.stage2_label,
        request.stage2_confidence,
    )

    merged_questionnaire = {"clinical_data": request.clinical_data}
    payload = _build_pipeline_outputs(
        predictions=predictions,
        questionnaire_data=merged_questionnaire,
        heatmap_base64=MOCK_HEATMAPS["pneumonia_focus"],
        stage2_override=(request.stage2_label, round(request.stage2_confidence, 3)),
    )

    report = payload.get("report")
    if report is None:
        return _error_response(
            "Unable to produce final report from assess inputs.",
            400,
            error_code="invalid_request",
            stage="stage4",
            retryable=False,
        )

    return JSONResponse(
        status_code=200,
        content={
            "stage3": payload["stage3"],
            "stage4": {
                "report": report["summary"],
                "recommended_actions": report["recommended_actions"],
                "disclaimer": report["disclaimer"],
            },
        },
    )
