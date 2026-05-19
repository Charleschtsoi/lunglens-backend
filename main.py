import asyncio
import json
import logging
import os
import shutil
import base64
import math
import tempfile
import time
from io import BytesIO
from typing import Any, List

import joblib
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from PIL import Image
from tensorflow.keras.models import load_model

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("lunglens.backend")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

# ML Model 2 H5 (ResNet152V2): must match training — see `_preprocess_model2_h5_numpy`.
MODEL2_H5_IMAGE_SIZE: tuple[int, int] = (224, 224)


class PatientData(BaseModel):
    age: int
    fever: bool
    cough_duration_days: int
    smoking: str | None = None
    breathing_difficulty: str | None = None


class QuestionRequest(BaseModel):
    high_attention_findings: List[str]


class QuestionItem(BaseModel):
    id: str
    text: str
    finding_trigger: str


CLINICAL_DICTIONARY: dict[str, list[str]] = {
    "Pneumonia": [
        "The AI flagged a pattern similar to pneumonia. Based on my symptoms, what follow-up tests or visits do you recommend?",
        "Should I be concerned about this viral pattern, and does it require immediate treatment?",
    ],
    "Infiltration": [
        "The educational output weighted Infiltration. How does that line up with your clinical impression?",
    ],
    "Pleural_Thickening": [
        "I noticed the AI highlighted areas associated with Pleural thickening—could you explain what that region shows on my film?",
    ],
}


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


app = FastAPI(title="LungLens API")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
ALLOWED_ORIGINS = _parse_origins(
    os.getenv("ALLOWED_ORIGINS", "*" if ENVIRONMENT != "production" else "")
)
REQUIRE_API_KEY = _parse_bool_env(
    "REQUIRE_API_KEY", default=(ENVIRONMENT == "production")
)
API_KEY = os.getenv("API_KEY", "").strip()
# Gemini educator preferred model (optional). When unset, discovery picks from the API key.
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "").strip()

# Used only when list_models() is unavailable (no key yet / list failed).
_GEMINI_STATIC_MODEL_CANDIDATES: tuple[str, ...] = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-002",
    "gemini-pro",
)


def _gemini_model_preference_key(name: str) -> tuple[int, int]:
    """Sort key: prefer flash / newer Gemini ids for educator summaries."""
    n = name.lower()
    score = 0
    if "flash" in n:
        score -= 20
    if "2.5" in n:
        score -= 8
    elif "2.0" in n:
        score -= 6
    elif "1.5" in n:
        score -= 4
    if "pro" in n and "flash" not in n:
        score += 6
    if "preview" in n or "experimental" in n:
        score += 3
    if "lite" in n or "8b" in n:
        score += 2
    return (score, len(name))


def _discover_gemini_generate_model_names(api_key: str) -> list[str]:
    """Model short names this API key can use with generateContent (from list_models)."""
    try:
        genai.configure(api_key=api_key)
        names: list[str] = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", None) or []
            if "generateContent" not in methods:
                continue
            raw = (getattr(m, "name", None) or "").strip()
            if raw.startswith("models/"):
                raw = raw.split("/", 1)[1]
            if raw and raw not in names:
                names.append(raw)
        return names
    except Exception as exc:
        logger.warning("Could not list Gemini models for key: %s", exc)
        return []


def _gemini_educator_models_to_try(api_key: str | None = None) -> list[str]:
    """Models to try: env preference first, then key-specific discovery or static fallbacks."""
    primary = GEMINI_MODEL or "gemini-2.0-flash"
    discovered = _discover_gemini_generate_model_names(api_key) if api_key else []

    seen: set[str] = set()
    out: list[str] = []

    def add(name: str) -> None:
        n = name.strip()
        if n and n not in seen:
            seen.add(n)
            out.append(n)

    if discovered:
        if primary in discovered:
            add(primary)
        for m in sorted(discovered, key=_gemini_model_preference_key):
            add(m)
        logger.info(
            "Gemini educator will try %s model(s) from API discovery (preference=%s).",
            len(out),
            primary,
        )
    else:
        add(primary)
        for m in _GEMINI_STATIC_MODEL_CANDIDATES:
            add(m)
        logger.info(
            "Gemini educator using static fallback model list (discovery unavailable)."
        )

    extras = os.getenv("GEMINI_MODEL_FALLBACKS", "").strip()
    if extras:
        for m in (x.strip() for x in extras.split(",") if x.strip()):
            if not discovered or m in discovered:
                add(m)

    return out if out else [primary]


def _gemini_probe_model_order(api_key: str) -> list[str]:
    """Models to try for BYOK health probe (same order as educator)."""
    return _gemini_educator_models_to_try(api_key)


def _gemini_error_should_try_next_model(error_code: str) -> bool:
    """Whether to attempt the next fallback model after a failure."""
    if error_code in ("gemini_unauthenticated", "gemini_permission_denied"):
        return False
    return True


FAIL_STARTUP_ON_MISSING_ENABLED_MODELS = _parse_bool_env(
    "FAIL_STARTUP_ON_MISSING_ENABLED_MODELS", default=False
)
COPD_MODEL_PATH = os.getenv("COPD_MODEL_PATH", "models/copd_screening_model.h5")
COPD_SCALER_PATH = os.getenv("COPD_SCALER_PATH", "models/scaler.pkl")
ENABLE_COPD_MODEL = os.getenv("ENABLE_COPD_MODEL", "true").lower() == "true"
# Prefer ENABLE_MODEL2_H5; legacy ENABLE_H5_MODEL still honored if the new key is unset.
if os.getenv("ENABLE_MODEL2_H5") is not None:
    ENABLE_MODEL2_H5 = _parse_bool_env("ENABLE_MODEL2_H5", default=False)
else:
    ENABLE_MODEL2_H5 = _parse_bool_env("ENABLE_H5_MODEL", default=False)
H5_MODEL2_PATH = (
    os.getenv("H5_MODEL2_PATH")
    or os.getenv("H5_MODEL_PATH", "models/resnet152v2_lung_disease_final.h5")
).strip()
# Class indices mapped manually during training
# Index 0 = Normal, Index 1 = Lung Opacity, Index 2 = Viral Pneumonia
H5_MODEL2_LABELS = ["Normal", "Lung Opacity", "Viral Pneumonia"]
# Prefer ENABLE_MODEL1; legacy ENABLE_MODEL1_PYTORCH honored if the new key is unset.
if os.getenv("ENABLE_MODEL1") is not None:
    ENABLE_MODEL1 = _parse_bool_env("ENABLE_MODEL1", default=False)
else:
    ENABLE_MODEL1 = _parse_bool_env("ENABLE_MODEL1_PYTORCH", default=False)
MODEL1_PATH = (
    os.getenv("MODEL1_PATH")
    or os.getenv("MODEL1_PTH_PATH", "models/best_resnet50_lunglens_cleaner.pth")
).strip()
# Index order must match checkpoint: 0=Normal, 1=Pneumonia-Bacteria, 2=Pneumonia-Virus
MODEL1_LABELS = _parse_csv(
    os.getenv(
        "MODEL1_LABELS",
        "Normal,Pneumonia-Bacteria,Pneumonia-Virus",
    )
)
ENABLE_DENSENET121 = _parse_bool_env("ENABLE_DENSENET121", default=False)
DENSENET121_PATH = os.getenv(
    "DENSENET121_PATH", "models/best_densenet121_lunglens.pth"
).strip()
_MODEL4_SWINT_CANDIDATE_PATHS: tuple[str, ...] = (
    "models/best_swint_lunglens.pth",
    "models/best_swin_t_chestxray_6class.pth",
)


def _resolve_model4_swint_weights_path() -> str:
    """Pick weights file: explicit MODEL4_SWINT_PATH, else first existing candidate."""
    explicit = os.getenv("MODEL4_SWINT_PATH", "").strip()
    if explicit:
        return explicit
    for rel in _MODEL4_SWINT_CANDIDATE_PATHS:
        if os.path.isfile(rel):
            return rel
    return _MODEL4_SWINT_CANDIDATE_PATHS[0]


MODEL4_SWINT_PATH = _resolve_model4_swint_weights_path()
# ImageFolder alphabetical class order (index 0..N-1). Override to match training folders.
MODEL4_SWINT_LABELS = _parse_csv(
    os.getenv(
        "MODEL4_SWINT_LABELS",
        "COVID-19,Lung_Opacity,Normal,Pneumonia,Tuberculosis,Viral_Pneumonia",
    )
)
if os.getenv("ENABLE_MODEL4_SWINT") is not None:
    ENABLE_MODEL4_SWINT = _parse_bool_env("ENABLE_MODEL4_SWINT", default=False)
else:
    ENABLE_MODEL4_SWINT = os.path.isfile(MODEL4_SWINT_PATH)
# DenseNet-121 class mapping is hard-enforced to match checkpoint output indices.
CLASS_NAMES = ["Normal", "Pneumonia-Bacteria", "Pneumonia-Virus"]
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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _square_224_pil_geometry() -> Any:
    """Short-edge resize then center-crop to 224×224 (aspect-preserving vs naive square resize)."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
        ]
    )


def _encode_rgb_pil_png_base64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


MODEL1_PT: Any = None
MODEL1_PT_LOAD_ERROR: str | None = None
_MODEL1_PREPROCESS: Any = None
MODEL2_H5: Any = None
MODEL2_H5_LOAD_ERROR: str | None = None
COPD_MODEL: Any = None
COPD_SCALER: Any = None
MODEL_DENSENET121: Any = None
MODEL_DENSENET121_LOAD_ERROR: str | None = None
_DENSENET121_PREPROCESS: Any = None
MODEL4_SWINT: Any = None
MODEL4_SWINT_LOAD_ERROR: str | None = None
TENSORFLOW_VERSION: str | None = None
PYTORCH_VERSION: str | None = None
_MODEL2_FILE_MISSING_WARNED = False

# Config-driven registry: add model4+ entries + loader/predictor wiring in one place.
MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "id": "model1",
        "health_key": "model1_resnet50",
        "name": "ResNet50",
        "kind": "pytorch",
        "gradcam_target_layer": "layer4[-1]",
    },
    {
        "id": "model2",
        "health_key": "model2_resnet152v2",
        "name": "ResNet152V2",
        "kind": "keras",
    },
    {
        "id": "model3",
        "health_key": "model3_densenet121",
        "name": "DenseNet-121",
        "kind": "pytorch",
        "gradcam_target_layer": "features.denseblock4",
    },
    {
        "id": "model4",
        "health_key": "model4_swint",
        "name": "Swin-T",
        "kind": "pytorch",
    },
]


def _registry_health_aliases() -> dict[str, Any]:
    """Compact enabled/loaded flags for /health (parallel to detailed * _pt / * _h5 blocks)."""
    return {
        "model1_resnet50": {
            "enabled": ENABLE_MODEL1,
            "loaded": MODEL1_PT is not None,
        },
        "model2_resnet152v2": {
            "enabled": ENABLE_MODEL2_H5,
            "loaded": MODEL2_H5 is not None,
        },
        "model3_densenet121": {
            "enabled": ENABLE_DENSENET121,
            "loaded": MODEL_DENSENET121 is not None,
        },
        "model4_swint": {
            "enabled": ENABLE_MODEL4_SWINT,
            "loaded": MODEL4_SWINT is not None,
        },
    }


def _pytorch_gradcam_to_png_base64(
    model: Any,
    input_tensor: Any,
    target_layers: list[Any],
    class_idx: int,
    rgb_hwc_01: Any,
) -> str:
    """Shared Grad-CAM overlay → base64 PNG (ASCII). PyTorch models only."""
    import numpy as np
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    targets = [ClassifierOutputTarget(class_idx)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = np.asarray(grayscale_cam)
    if grayscale_cam.ndim == 3:
        grayscale_cam = grayscale_cam[0]
    elif grayscale_cam.ndim != 2:
        raise ValueError(f"Unexpected Grad-CAM shape: {grayscale_cam.shape}")

    visualization = show_cam_on_image(rgb_hwc_01, grayscale_cam, use_rgb=True)
    overlay = Image.fromarray(visualization)
    buf = BytesIO()
    overlay.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _model1_pt_path_diagnostics() -> dict[str, Any]:
    abs_path = os.path.abspath(MODEL1_PATH)
    exists = os.path.isfile(abs_path)
    size_bytes: int | None = None
    if exists:
        try:
            size_bytes = os.path.getsize(abs_path)
        except OSError as exc:
            logger.warning("Could not stat ML Model 1 .pth file %s: %s", abs_path, exc)
    return {
        "path": MODEL1_PATH,
        "absolute_path": abs_path,
        "exists": exists,
        "size_bytes": size_bytes,
    }


def _pytorch_version_probe() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except Exception as exc:
        logger.warning("PyTorch not importable for version probe: %s", exc)
        return None


def _model1_preprocess() -> Any:
    """ImageNet-style pipeline for ML Model 1 only (not Model 2 /255)."""
    global _MODEL1_PREPROCESS
    if _MODEL1_PREPROCESS is None:
        from torchvision import transforms

        _MODEL1_PREPROCESS = transforms.Compose(
            [
                _square_224_pil_geometry(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return _MODEL1_PREPROCESS


def _load_model1_pytorch() -> None:
    global MODEL1_PT, MODEL1_PT_LOAD_ERROR, PYTORCH_VERSION, _MODEL1_PREPROCESS
    _MODEL1_PREPROCESS = None
    if not ENABLE_MODEL1:
        MODEL1_PT = None
        MODEL1_PT_LOAD_ERROR = None
        logger.info("ML Model 1 PyTorch skipped: ENABLE_MODEL1 is false.")
        return
    if len(MODEL1_LABELS) != 3:
        MODEL1_PT = None
        MODEL1_PT_LOAD_ERROR = "MODEL1_LABELS must contain exactly 3 labels."
        logger.error(
            "ML Model 1 not loaded: need 3 MODEL1_LABELS, got %s (%r).",
            len(MODEL1_LABELS),
            MODEL1_LABELS,
        )
        return
    diag = _model1_pt_path_diagnostics()
    path = diag["absolute_path"]
    logger.info(
        "ML Model 1 PyTorch load starting: path=%r exists=%s size_bytes=%s",
        diag["path"],
        diag["exists"],
        diag["size_bytes"],
    )
    if not diag["exists"]:
        MODEL1_PT = None
        MODEL1_PT_LOAD_ERROR = (
            f"ML Model 1 .pth not found at {path!r} (from MODEL1_PATH={diag['path']!r})."
        )
        logger.error("%s", MODEL1_PT_LOAD_ERROR)
        return
    try:
        import torch
        import torchvision.models as models

        PYTORCH_VERSION = str(torch.__version__)
        logger.info("PyTorch version: %s", PYTORCH_VERSION)

        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        MODEL1_PT = model
        MODEL1_PT_LOAD_ERROR = None
        logger.info("ML Model 1 PyTorch loaded successfully")
    except Exception as exc:
        MODEL1_PT = None
        MODEL1_PT_LOAD_ERROR = str(exc)
        logger.exception("ML Model 1 PyTorch load failed.")


def _run_pytorch_model1_full(
    image_bytes: bytes,
) -> tuple[str, float, dict[str, float], str | None]:
    """ResNet50 3-class: label, confidence [0,1], per-class probabilities, Grad-CAM PNG or None."""
    if MODEL1_PT is None:
        raise RuntimeError(MODEL1_PT_LOAD_ERROR or "ML Model 1 PyTorch is not loaded.")
    import numpy as np
    import torch
    import torch.nn.functional as F

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = _model1_preprocess()(img).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL1_PT(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
    pred_idx = int(predicted_idx.item())
    label = MODEL1_LABELS[pred_idx]
    conf01 = round(float(confidence.item()), 3)
    probabilities = {
        MODEL1_LABELS[i]: round(float(probs[0, i].item()), 4) for i in range(3)
    }

    cropped224 = _square_224_pil_geometry()(img)
    rgb_display = np.array(cropped224).astype(np.float32) / 255.0
    rgb_display = np.clip(rgb_display, 0.0, 1.0)
    target_layers = [MODEL1_PT.layer4[-1]]
    gradcam_b64: str | None = None
    try:
        gradcam_b64 = _pytorch_gradcam_to_png_base64(
            MODEL1_PT, tensor, target_layers, pred_idx, rgb_display
        )
    except Exception as exc:
        logger.warning("Model 1 Grad-CAM failed: %s", exc)

    return label, conf01, probabilities, gradcam_b64


def _densenet121_path_diagnostics() -> dict[str, Any]:
    abs_path = os.path.abspath(DENSENET121_PATH)
    exists = os.path.isfile(abs_path)
    size_bytes: int | None = None
    if exists:
        try:
            size_bytes = os.path.getsize(abs_path)
        except OSError as exc:
            logger.warning("Could not stat DenseNet-121 .pth file %s: %s", abs_path, exc)
    return {
        "path": DENSENET121_PATH,
        "absolute_path": abs_path,
        "exists": exists,
        "size_bytes": size_bytes,
    }


def _model4_swint_path_diagnostics() -> dict[str, Any]:
    abs_path = os.path.abspath(MODEL4_SWINT_PATH)
    exists = os.path.isfile(abs_path)
    size_bytes: int | None = None
    if exists:
        try:
            size_bytes = os.path.getsize(abs_path)
        except OSError as exc:
            logger.warning("Could not stat Swin-T .pth file %s: %s", abs_path, exc)
    return {
        "path": MODEL4_SWINT_PATH,
        "absolute_path": abs_path,
        "exists": exists,
        "size_bytes": size_bytes,
    }


def _unwrap_pytorch_state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint is not a state_dict mapping.")
    return obj


def _swint_checkpoint_num_classes(state_dict: dict[str, Any]) -> int:
    if "head.weight" in state_dict:
        return int(state_dict["head.weight"].shape[0])
    for key, tensor in state_dict.items():
        if key.endswith("head.weight") or key == "head.1.weight":
            return int(tensor.shape[0])
    raise ValueError("Could not infer Swin-T class count from checkpoint head weights.")


def _load_swint_model4() -> None:
    global MODEL4_SWINT, MODEL4_SWINT_LOAD_ERROR, PYTORCH_VERSION
    if not ENABLE_MODEL4_SWINT:
        MODEL4_SWINT = None
        MODEL4_SWINT_LOAD_ERROR = None
        logger.info("Model 4 (Swin-T) skipped: ENABLE_MODEL4_SWINT is false.")
        return

    if not os.path.isfile(MODEL4_SWINT_PATH):
        MODEL4_SWINT = None
        MODEL4_SWINT_LOAD_ERROR = (
            f"Swin-T model file not found at {MODEL4_SWINT_PATH}."
        )
        logger.warning("%s Model 4 disabled.", MODEL4_SWINT_LOAD_ERROR)
        return

    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models

        if PYTORCH_VERSION is None:
            PYTORCH_VERSION = str(torch.__version__)
            logger.info("PyTorch version: %s", PYTORCH_VERSION)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            raw_ckpt = torch.load(
                MODEL4_SWINT_PATH,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        except TypeError:
            raw_ckpt = torch.load(
                MODEL4_SWINT_PATH,
                map_location=torch.device("cpu"),
            )
        state_dict = _unwrap_pytorch_state_dict(raw_ckpt)
        num_classes = _swint_checkpoint_num_classes(state_dict)
        if len(MODEL4_SWINT_LABELS) != num_classes:
            raise ValueError(
                f"MODEL4_SWINT_LABELS has {len(MODEL4_SWINT_LABELS)} labels but "
                f"checkpoint expects {num_classes}. Set MODEL4_SWINT_LABELS to match "
                "training ImageFolder order (alphabetical by folder name)."
            )

        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        MODEL4_SWINT = model
        MODEL4_SWINT_LOAD_ERROR = None
        logger.info(
            "Model 4 (Swin-T) loaded successfully on device=%s path=%s classes=%s labels=%s",
            device,
            MODEL4_SWINT_PATH,
            num_classes,
            MODEL4_SWINT_LABELS,
        )
    except Exception as exc:
        MODEL4_SWINT = None
        MODEL4_SWINT_LOAD_ERROR = str(exc)
        logger.exception("FATAL: Failed to load Swin-T model: %s", exc)


def _run_swint_model4(image_bytes: bytes) -> tuple[str, float, dict[str, float]]:
    """Swin-T: top label, confidence [0,1], per-class probabilities."""
    if MODEL4_SWINT is None:
        raise RuntimeError(MODEL4_SWINT_LOAD_ERROR or "Model 4 (Swin-T) is not loaded.")
    import torch
    from torchvision import transforms

    device = next(MODEL4_SWINT.parameters()).device
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image).unsqueeze(0).to(device)
    n_cls = len(MODEL4_SWINT_LABELS)

    with torch.no_grad():
        outputs = MODEL4_SWINT(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)

    pred_idx_int = int(pred_idx.item())
    label = MODEL4_SWINT_LABELS[pred_idx_int]
    conf01 = round(float(conf.item()), 3)
    probabilities = {
        MODEL4_SWINT_LABELS[i]: round(float(probs[i].item()), 4) for i in range(n_cls)
    }
    return label, conf01, probabilities


def _densenet121_preprocess() -> Any:
    """Strict 224×224 resize + ImageNet normalization."""
    global _DENSENET121_PREPROCESS
    if _DENSENET121_PREPROCESS is None:
        from torchvision import transforms

        _DENSENET121_PREPROCESS = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return _DENSENET121_PREPROCESS


def _load_densenet121() -> None:
    global MODEL_DENSENET121, MODEL_DENSENET121_LOAD_ERROR, PYTORCH_VERSION, _DENSENET121_PREPROCESS
    _DENSENET121_PREPROCESS = None
    if not ENABLE_DENSENET121:
        MODEL_DENSENET121 = None
        MODEL_DENSENET121_LOAD_ERROR = None
        logger.info("DenseNet-121 skipped: ENABLE_DENSENET121 is false.")
        return
    if len(CLASS_NAMES) != 3:
        MODEL_DENSENET121 = None
        MODEL_DENSENET121_LOAD_ERROR = "DenseNet CLASS_NAMES must contain exactly 3 labels."
        logger.error(
            "DenseNet-121 not loaded: need 3 CLASS_NAMES, got %s (%r).",
            len(CLASS_NAMES),
            CLASS_NAMES,
        )
        return
    diag = _densenet121_path_diagnostics()
    path = diag["absolute_path"]
    logger.info(
        "ML DenseNet-121 load starting: path=%r exists=%s size_bytes=%s",
        diag["path"],
        diag["exists"],
        diag["size_bytes"],
    )
    if not diag["exists"]:
        MODEL_DENSENET121 = None
        MODEL_DENSENET121_LOAD_ERROR = (
            f"DenseNet-121 .pth not found at {path!r} (from DENSENET121_PATH={diag['path']!r})."
        )
        logger.error("%s", MODEL_DENSENET121_LOAD_ERROR)
        return
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models

        if PYTORCH_VERSION is None:
            PYTORCH_VERSION = str(torch.__version__)
            logger.info("PyTorch version: %s", PYTORCH_VERSION)

        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 3)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        MODEL_DENSENET121 = model
        MODEL_DENSENET121_LOAD_ERROR = None
        logger.info("ML DenseNet-121 loaded successfully")
    except Exception as exc:
        MODEL_DENSENET121 = None
        MODEL_DENSENET121_LOAD_ERROR = str(exc)
        logger.exception("ML DenseNet-121 load failed.")


def _densenet121_predict_and_cam(image_bytes: bytes) -> dict[str, Any]:
    """Run DenseNet-121 inference + Grad-CAM; return API payload fields (no success wrapper)."""
    if MODEL_DENSENET121 is None:
        raise RuntimeError(
            MODEL_DENSENET121_LOAD_ERROR or "DenseNet-121 is not loaded."
        )
    import numpy as np
    import torch
    import torch.nn.functional as F

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = _densenet121_preprocess()(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL_DENSENET121(tensor)
        probs_t = F.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs_t).item())
    confidence_frac = float(probs_t[pred_idx].item())
    class_name = CLASS_NAMES[pred_idx]

    probabilities = {
        CLASS_NAMES[i]: round(float(probs_t[i].item()), 4)
        for i in range(3)
    }

    cropped224 = _square_224_pil_geometry()(img)
    rgb_display = np.array(cropped224).astype(np.float32) / 255.0
    rgb_display = np.clip(rgb_display, 0.0, 1.0)
    input_preview_b64 = _encode_rgb_pil_png_base64(cropped224)

    target_layers = [MODEL_DENSENET121.features.denseblock4]
    gradcam_b64 = _pytorch_gradcam_to_png_base64(
        MODEL_DENSENET121, tensor, target_layers, pred_idx, rgb_display
    )

    return {
        "class_id": pred_idx,
        "class_name": class_name,
        "confidence_score": round(confidence_frac, 4),
        "prediction": class_name,
        "confidence": round(confidence_frac * 100.0, 2),
        "probabilities": probabilities,
        "gradcam": gradcam_b64,
        "input_preview_base64": input_preview_b64,
    }


def _model2_h5_path_diagnostics() -> dict[str, Any]:
    abs_path = os.path.abspath(H5_MODEL2_PATH)
    exists = os.path.isfile(abs_path)
    size_bytes: int | None = None
    if exists:
        try:
            size_bytes = os.path.getsize(abs_path)
        except OSError as exc:
            logger.warning("Could not stat H5 model file %s: %s", abs_path, exc)
    return {
        "path": H5_MODEL2_PATH,
        "absolute_path": abs_path,
        "exists": exists,
        "size_bytes": size_bytes,
    }


def _missing_enabled_model_files() -> list[str]:
    missing: list[str] = []
    if ENABLE_MODEL1:
        d1 = _model1_pt_path_diagnostics()
        if not d1["exists"]:
            missing.append(
                f"model1 missing: MODEL1_PATH={d1['path']!r} absolute_path={d1['absolute_path']!r}"
            )
    if ENABLE_MODEL2_H5:
        d2 = _model2_h5_path_diagnostics()
        if not d2["exists"]:
            missing.append(
                f"model2 missing: H5_MODEL2_PATH={d2['path']!r} absolute_path={d2['absolute_path']!r}"
            )
    if ENABLE_DENSENET121:
        d3 = _densenet121_path_diagnostics()
        if not d3["exists"]:
            missing.append(
                f"model3 missing: DENSENET121_PATH={d3['path']!r} absolute_path={d3['absolute_path']!r}"
            )
    if ENABLE_MODEL4_SWINT:
        d4 = _model4_swint_path_diagnostics()
        if not d4["exists"]:
            missing.append(
                f"model4 missing: MODEL4_SWINT_PATH={d4['path']!r} absolute_path={d4['absolute_path']!r}"
            )
    return missing


def _warn_model2_file_missing_once(context: str) -> None:
    global _MODEL2_FILE_MISSING_WARNED
    if _MODEL2_FILE_MISSING_WARNED:
        return
    d2 = _model2_h5_path_diagnostics()
    if d2["exists"]:
        return
    _MODEL2_FILE_MISSING_WARNED = True
    logger.error(
        "ML Model 2 file is missing during %s. H5_MODEL2_PATH=%r absolute_path=%r. "
        "Model 2 inference will be unavailable until file is restored and server restarted.",
        context,
        d2["path"],
        d2["absolute_path"],
    )


def _tensorflow_version_probe() -> str | None:
    try:
        import tensorflow as tf  # type: ignore

        return str(tf.__version__)
    except Exception as exc:
        logger.warning("TensorFlow not importable for version probe: %s", exc)
        return None


def _make_h5_custom_objects(tf: Any) -> dict[str, Any]:
    class PatchedBatchNormalization(tf.keras.layers.BatchNormalization):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.pop("renorm", None)
            kwargs.pop("renorm_clipping", None)
            kwargs.pop("renorm_momentum", None)
            super().__init__(**kwargs)

    return {"BatchNormalization": PatchedBatchNormalization}


def _sanitize_keras_h5_config(node: Any) -> None:
    if isinstance(node, dict):
        config = node.get("config")
        if isinstance(config, dict):
            config.pop("quantization_config", None)
            dtype_pol = config.get("dtype")
            if isinstance(dtype_pol, dict):
                policy_name = (
                    dtype_pol.get("config", {}).get("name")
                    if isinstance(dtype_pol.get("config"), dict)
                    else None
                )
                config["dtype"] = (
                    "float32" if policy_name == "mixed_float16" else policy_name
                ) or "float32"
            if node.get("class_name") == "InputLayer":
                batch_shape = config.get("batch_input_shape", config.get("batch_shape"))
                if isinstance(batch_shape, (list, tuple)):
                    config["batch_input_shape"] = list(batch_shape)
                else:
                    config["batch_input_shape"] = [None, 224, 224, 3]
                dtype_val = config.get("dtype")
                if not isinstance(dtype_val, str):
                    config["dtype"] = "float32"
                config.pop("batch_shape", None)
                config.pop("optional", None)
        for value in node.values():
            _sanitize_keras_h5_config(value)
    elif isinstance(node, list):
        for item in node:
            _sanitize_keras_h5_config(item)


def _load_h5_model_compat(tf: Any, model_path: str) -> Any:
    import h5py  # type: ignore

    with h5py.File(model_path, "r") as h5_file:
        raw = h5_file.attrs.get("model_config")
    if raw is None:
        raise ValueError("H5 file has no model_config attribute.")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    model_cfg = json.loads(raw)
    _sanitize_keras_h5_config(model_cfg)

    fd, compat_path = tempfile.mkstemp(suffix=".h5", prefix="lunglens-model2-h5-compat-")
    os.close(fd)
    shutil.copyfile(model_path, compat_path)
    with h5py.File(compat_path, "r+") as patched:
        patched.attrs["model_config"] = json.dumps(model_cfg).encode("utf-8")
    return tf.keras.models.load_model(
        compat_path,
        compile=False,
        custom_objects=_make_h5_custom_objects(tf),
    )


def _load_model2_h5() -> None:
    global MODEL2_H5, MODEL2_H5_LOAD_ERROR, TENSORFLOW_VERSION
    if not ENABLE_MODEL2_H5:
        MODEL2_H5 = None
        MODEL2_H5_LOAD_ERROR = None
        logger.info("ML Model 2 H5 skipped: ENABLE_MODEL2_H5 / ENABLE_H5_MODEL is false.")
        return
    if len(H5_MODEL2_LABELS) != 3:
        MODEL2_H5 = None
        MODEL2_H5_LOAD_ERROR = "H5_MODEL2_LABELS must contain exactly 3 labels."
        logger.error(
            "ML Model 2 H5 not loaded: need 3 labels, got %s (%r).",
            len(H5_MODEL2_LABELS),
            H5_MODEL2_LABELS,
        )
        return
    diag = _model2_h5_path_diagnostics()
    path = diag["absolute_path"]
    logger.info(
        "ML Model 2 H5 load starting: path=%r absolute_path=%r exists=%s size_bytes=%s labels=%r",
        diag["path"],
        path,
        diag["exists"],
        diag["size_bytes"],
        H5_MODEL2_LABELS,
    )
    if not diag["exists"]:
        MODEL2_H5 = None
        MODEL2_H5_LOAD_ERROR = (
            f"H5 model file not found at {path!r} (from H5_MODEL2_PATH={diag['path']!r})."
        )
        logger.error("%s", MODEL2_H5_LOAD_ERROR)
        return
    try:
        import tensorflow as tf  # type: ignore

        TENSORFLOW_VERSION = str(tf.__version__)
        logger.info("TensorFlow version: %s", TENSORFLOW_VERSION)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        co = _make_h5_custom_objects(tf)
        try:
            logger.info("ML Model 2 H5: attempting tf.keras.models.load_model (direct).")
            MODEL2_H5 = tf.keras.models.load_model(
                path, compile=False, custom_objects=co
            )
            MODEL2_H5_LOAD_ERROR = None
            logger.info(
                "ML Model 2 H5 loaded successfully (direct): %s size_bytes=%s",
                path,
                diag["size_bytes"],
            )
        except Exception:
            logger.warning(
                "ML Model 2 H5 direct load failed; trying compat path.",
                exc_info=True,
            )
            MODEL2_H5 = _load_h5_model_compat(tf, path)
            MODEL2_H5_LOAD_ERROR = None
            logger.info(
                "ML Model 2 H5 loaded successfully (compat): %s size_bytes=%s",
                path,
                diag["size_bytes"],
            )
    except Exception as exc:
        MODEL2_H5 = None
        MODEL2_H5_LOAD_ERROR = str(exc)
        logger.exception("ML Model 2 H5 load failed.")


def _load_copd_pipeline() -> None:
    global COPD_MODEL, COPD_SCALER
    if not ENABLE_COPD_MODEL:
        COPD_MODEL = None
        COPD_SCALER = None
        logger.info("COPD Screening pipeline skipped: ENABLE_COPD_MODEL is false.")
        return
    try:
        if os.path.exists(COPD_MODEL_PATH) and os.path.exists(COPD_SCALER_PATH):
            COPD_MODEL = load_model(COPD_MODEL_PATH)
            COPD_SCALER = joblib.load(COPD_SCALER_PATH)
            logger.info("COPD Screening Model & Scaler loaded successfully.")
        else:
            COPD_MODEL = None
            COPD_SCALER = None
            logger.warning(
                "COPD model or scaler files not found. model_path=%r scaler_path=%r",
                COPD_MODEL_PATH,
                COPD_SCALER_PATH,
            )
    except Exception as exc:
        COPD_MODEL = None
        COPD_SCALER = None
        logger.error("Failed to load COPD pipeline: %s", exc)


def _preprocess_model2_h5_numpy(image_bytes: bytes) -> Any:
    """
    ML Model 2 preprocessing:
    - RGB input
    - Resize to 224 x 224
    - float32 array
    - Prefer tf.keras.applications.resnet_v2.preprocess_input
      (fallback to /255.0 if unavailable)
    Returns: NumPy array shape (1, 224, 224, 3), dtype float32.
    """
    import numpy as np  # type: ignore

    w, h = MODEL2_H5_IMAGE_SIZE
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((w, h), Image.Resampling.BILINEAR)
    img_array = np.asarray(image, dtype=np.float32)
    try:
        from tensorflow.keras.applications.resnet_v2 import preprocess_input  # type: ignore

        img_array = preprocess_input(img_array)
    except Exception:
        logger.warning(
            "ResNetV2 preprocess_input unavailable for Model 2; falling back to /255.0 scaling."
        )
        img_array = img_array / 255.0
    batch = np.expand_dims(img_array, axis=0)
    return batch


def _model2_label_for_api(raw: str) -> str:
    return raw.replace("_", " ")


def _model2_decode_scores(row: Any) -> tuple[int, str, float, dict[str, float]]:
    import numpy as np  # type: ignore

    idx = int(np.argmax(row))
    raw_label = H5_MODEL2_LABELS[idx]
    logger.debug("Model 2 Raw Index: %s -> Mapped Label: %s", idx, raw_label)
    label = _model2_label_for_api(raw_label)
    confidence = float(np.asarray(row[idx], dtype=np.float64))
    probabilities = {
        _model2_label_for_api(H5_MODEL2_LABELS[i]): round(
            float(np.asarray(row[i], dtype=np.float64)), 4
        )
        for i in range(len(H5_MODEL2_LABELS))
    }
    return idx, label, confidence, probabilities


def _run_h5_model2(image_bytes: bytes) -> tuple[str, float, dict[str, float]]:
    if MODEL2_H5 is None:
        raise RuntimeError(MODEL2_H5_LOAD_ERROR or "ML Model 2 H5 is not loaded.")
    import numpy as np  # type: ignore

    batch = _preprocess_model2_h5_numpy(image_bytes)
    logger.debug(
        "Model 2 preprocessed batch range: min=%.6f max=%.6f",
        float(np.min(batch)),
        float(np.max(batch)),
    )
    out = MODEL2_H5.predict(batch, verbose=0)
    row = out[0]
    _, label, confidence, probabilities = _model2_decode_scores(row)
    return label, confidence, probabilities


def _run_copd_screening(patient_data: dict[str, Any]) -> tuple[str, float]:
    if COPD_MODEL is None or COPD_SCALER is None:
        raise RuntimeError("COPD pipeline is not loaded.")

    age = float(patient_data.get("age", 50.0))
    fever = 1.0 if patient_data.get("fever") else 0.0
    cough_days = float(patient_data.get("cough_duration_days", 0.0))

    smoking_status = str(
        patient_data.get("smoking_status", patient_data.get("smoking", "Never"))
    ).strip()
    smoking_map = {
        "Never": 0.0,
        "Former": 1.0,
        "Current": 2.0,
        "never": 0.0,
        "former": 1.0,
        "current": 2.0,
    }
    smoking_val = smoking_map.get(smoking_status, 0.0)

    breathing = str(patient_data.get("breathing_difficulty", "None")).strip()
    breathing_map = {
        "None": 0.0,
        "Mild": 1.0,
        "Severe": 2.0,
        "none": 0.0,
        "mild": 1.0,
        "severe": 2.0,
    }
    breathing_val = breathing_map.get(breathing, 0.0)

    raw_features = np.array(
        [[age, fever, cough_days, smoking_val, breathing_val, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    scaled_features = COPD_SCALER.transform(raw_features)
    prediction_prob = float(COPD_MODEL.predict(scaled_features, verbose=0)[0][0])
    label = "High COPD Risk" if prediction_prob > 0.5 else "Low COPD Risk"
    return label, round(prediction_prob, 3)


def _resolve_educator_gemini_key(form_value: str | None) -> tuple[str | None, str]:
    """Multipart BYOK first, then server env. Returns (key_or_none, source_label for logs)."""
    if form_value is not None and str(form_value).strip():
        return str(form_value).strip(), "multipart_gemini_api_key"
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.getenv(env_name, "").strip()
        if v:
            return v, f"env_{env_name.lower()}"
    return None, "none"


def _classify_gemini_from_message(msg: str) -> str | None:
    m = msg.lower()
    if "429" in msg or "resource exhausted" in m or "too many requests" in m:
        return "gemini_quota"
    if "quota" in m or "rate limit" in m or "billing" in m:
        return "gemini_quota"
    if "401" in msg or "unauthenticated" in m or "api key not valid" in m:
        return "gemini_unauthenticated"
    if "403" in msg or "permission denied" in m or "forbidden" in m:
        return "gemini_permission_denied"
    if "404" in msg or ("not found" in m and "model" in m):
        return "gemini_not_found"
    if "400" in msg or "invalid argument" in m or "malformed" in m:
        return "gemini_invalid_argument"
    if "deadline" in m or "timeout" in m or "504" in msg or "408" in msg:
        return "gemini_timeout"
    if "503" in msg or "502" in msg or "500" in msg or "unavailable" in m:
        return "gemini_upstream"
    if "ssl" in m or "certificate" in m:
        return "gemini_network"
    if "name resolution" in m or "failed to resolve" in m or "nodename nor servname" in m:
        return "gemini_network"
    if "connection refused" in m or "connection reset" in m:
        return "gemini_network"
    return None


def _classify_gemini_exception(exc: BaseException) -> str:
    """Machine-readable llm_evaluation.error_code (no secrets in return value)."""
    try:
        from google.api_core import exceptions as gexc
    except ImportError:
        gexc = None  # type: ignore[assignment]

    try:
        from google.generativeai.types.generation_types import (
            BrokenResponseError as _BrokenResponseError,
            IncompleteIterationError as _IncompleteIterationError,
        )
    except ImportError:
        _BrokenResponseError = None  # type: ignore[misc, assignment]
        _IncompleteIterationError = None  # type: ignore[misc, assignment]

    if _BrokenResponseError is not None and isinstance(exc, _BrokenResponseError):
        return "gemini_broken_response"
    if _IncompleteIterationError is not None and isinstance(exc, _IncompleteIterationError):
        return "gemini_incomplete_response"

    if gexc is not None:
        if isinstance(exc, gexc.Unauthenticated):
            return "gemini_unauthenticated"
        if isinstance(exc, gexc.Unauthorized):
            return "gemini_unauthenticated"
        if isinstance(exc, gexc.PermissionDenied):
            return "gemini_permission_denied"
        if isinstance(exc, gexc.Forbidden):
            return "gemini_permission_denied"
        if isinstance(exc, (gexc.ResourceExhausted, gexc.TooManyRequests)):
            return "gemini_quota"
        if isinstance(exc, (gexc.InvalidArgument, gexc.BadRequest, gexc.FailedPrecondition)):
            return "gemini_invalid_argument"
        if isinstance(exc, (gexc.DeadlineExceeded, gexc.GatewayTimeout)):
            return "gemini_timeout"
        if isinstance(
            exc,
            (gexc.ServiceUnavailable, gexc.InternalServerError, gexc.BadGateway),
        ):
            return "gemini_upstream"
        if isinstance(exc, gexc.NotFound):
            return "gemini_not_found"
        if isinstance(exc, gexc.Cancelled):
            return "gemini_cancelled"

        if isinstance(exc, gexc.GoogleAPICallError):
            code = getattr(exc, "code", None)
            if code == 429:
                return "gemini_quota"
            if code == 401:
                return "gemini_unauthenticated"
            if code == 403:
                return "gemini_permission_denied"
            if code == 404:
                return "gemini_not_found"
            if code == 400:
                return "gemini_invalid_argument"
            if code in (408, 504):
                return "gemini_timeout"
            if isinstance(code, int) and code >= 500:
                return "gemini_upstream"
            return "gemini_google_api_error"

    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError)):
        return "gemini_network"

    if isinstance(exc, OSError):
        try:
            import errno as _errno

            if getattr(exc, "errno", None) in {
                _errno.ECONNREFUSED,
                _errno.ECONNRESET,
                _errno.ENETUNREACH,
                _errno.EHOSTUNREACH,
                _errno.ETIMEDOUT,
                _errno.ENETDOWN,
                _errno.EPIPE,
            }:
                return "gemini_network"
        except Exception:
            pass

    name = type(exc).__name__
    if "BlockedPrompt" in name or "StopCandidate" in name:
        return "gemini_blocked_or_empty"

    hinted = _classify_gemini_from_message(str(exc))
    if hinted is not None:
        return hinted

    logger.warning("Educator Gemini: unclassified exception type=%s", name)
    return "gemini_unknown"


def _gemini_health_message(
    error_code: str,
    model: str,
    *,
    models_tried: list[str] | None = None,
    last_model: str | None = None,
    discovered_models: list[str] | None = None,
) -> str:
    tried = models_tried or []
    failed_model = last_model or model
    if error_code == "gemini_unauthenticated":
        return (
            "This does not look like a valid Google AI (Gemini) API key. "
            "Create one at Google AI Studio (aistudio.google.com/apikey). "
            "Do not use the LungLens backend API_KEY here."
        )
    if error_code == "gemini_permission_denied":
        return (
            "The Gemini API key was rejected (permission denied). "
            "Check that the key is enabled for the Generative Language API."
        )
    if error_code == "gemini_quota":
        return "Gemini quota or rate limit exceeded for this API key. Try again later or check billing in Google AI Studio."
    if error_code == "gemini_not_found":
        hint = ""
        if discovered_models:
            hint = f" Models available for this key include: {', '.join(discovered_models[:5])}."
        return (
            f"The model '{failed_model}' is not available for this API key.{hint} "
            "Set GEMINI_MODEL on the server to one of those names."
        )
    if error_code == "gemini_invalid_argument":
        if tried:
            return (
                "Gemini rejected all models we tried for this API key: "
                f"{', '.join(tried)}. "
                "Create or verify your key at Google AI Studio, or set GEMINI_MODEL to a model "
                "listed under your key in AI Studio."
            )
        return (
            f"Gemini rejected the request for model '{failed_model}'. "
            "Check your API key in Google AI Studio and set GEMINI_MODEL to a supported model name."
        )
    if error_code in ("gemini_blocked_or_empty", "gemini_broken_response", "gemini_incomplete_response"):
        return "Gemini responded but returned no usable text. Try again or use a different model."
    if error_code == "gemini_network":
        return "Could not reach Google Gemini (network error). Check connectivity from the backend host."
    if error_code == "gemini_timeout":
        return "Gemini request timed out. Try again."
    return (
        f"Could not validate this Gemini API key (last model: '{failed_model}'). "
        "Use a Google AI Studio API key (not LungLens API_KEY)."
    )


def _list_gemini_models_for_key(api_key: str) -> tuple[bool, str | None]:
    """Return (ok, error_code). Verifies the key can call the Gemini API at all."""
    try:
        genai.configure(api_key=api_key)
        next(genai.list_models(), None)
        return True, None
    except StopIteration:
        return True, None
    except Exception as exc:
        return False, _classify_gemini_exception(exc)


def _probe_gemini_model_generate(api_key: str, model_id: str) -> tuple[bool, str | None]:
    """Single-model minimal generate probe. Returns (ok, error_code)."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        gen_cfg: Any = None
        try:
            gc_cls = getattr(genai, "GenerationConfig", None)
            if gc_cls is not None:
                gen_cfg = gc_cls(max_output_tokens=16)
        except Exception:
            gen_cfg = None
        kwargs: dict[str, Any] = {}
        if gen_cfg is not None:
            kwargs["generation_config"] = gen_cfg
        response = model.generate_content(
            'Reply with exactly the two letters OK and nothing else.',
            **kwargs,
        )
        try:
            raw = response.text
        except (ValueError, AttributeError):
            return False, "gemini_blocked_or_empty"
        text = (raw or "").strip() if isinstance(raw, str) else ""
        if not text:
            return False, "gemini_blocked_or_empty"
        return True, None
    except Exception as exc:
        return False, _classify_gemini_exception(exc)


def _probe_user_gemini_key(api_key: str) -> dict[str, Any]:
    """Validate BYOK: list_models (key) then generate on discovered / fallback models."""
    preferred = GEMINI_MODEL or "gemini-2.0-flash"
    discovered = _discover_gemini_generate_model_names(api_key)

    list_ok, list_code = _list_gemini_models_for_key(api_key)
    if not list_ok:
        code = list_code or "gemini_unknown"
        logger.info("Gemini health probe list_models failed: %s", code)
        return {
            "status": "invalid",
            "ok": False,
            "error_code": code,
            "message": _gemini_health_message(
                code, preferred, discovered_models=discovered
            ),
            "model": preferred,
        }

    models_to_try = _gemini_probe_model_order(api_key)
    if not models_to_try:
        return {
            "status": "invalid",
            "ok": False,
            "error_code": "gemini_not_found",
            "message": "No Gemini generateContent models found for this API key.",
            "model": preferred,
            "discovered_models": discovered,
        }

    models_tried: list[str] = []
    last_code: str | None = None
    last_model: str | None = None
    for model_id in models_to_try:
        models_tried.append(model_id)
        ok, code = _probe_gemini_model_generate(api_key, model_id)
        if ok:
            if model_id != preferred:
                logger.info(
                    "Gemini health probe: preferred %s failed; validated with %s",
                    preferred,
                    model_id,
                )
            return {
                "status": "ok",
                "ok": True,
                "model": model_id,
                "configured_model": preferred,
                **(
                    {
                        "warning": (
                            f"Your key works with '{model_id}' but GEMINI_MODEL is "
                            f"'{preferred}'. Set GEMINI_MODEL={model_id} on the backend for analyze."
                        )
                    }
                    if model_id != preferred
                    else {}
                ),
            }
        last_code = code
        last_model = model_id
        logger.info("Gemini health probe generate failed model=%s code=%s", model_id, code)

    code = last_code or "gemini_unknown"
    return {
        "status": "invalid",
        "ok": False,
        "error_code": code,
        "message": _gemini_health_message(
            code,
            preferred,
            models_tried=models_tried,
            last_model=last_model,
            discovered_models=discovered,
        ),
        "model": last_model or preferred,
        "models_tried": models_tried,
        "discovered_models": discovered[:20],
    }


def _generate_llm_summary(
    ml_results: dict[str, Any], patient_data: dict[str, Any], api_key: str | None
) -> dict[str, Any]:
    if not api_key:
        return {
            "status": "skipped",
            "text": (
                "No Gemini API key available. Provide gemini_api_key on the analyze request "
                "or set GEMINI_API_KEY or GOOGLE_API_KEY on the server."
            ),
            "error_code": "educator_no_api_key",
        }

    smoking_status = patient_data.get("smoking_status", patient_data.get("smoking", "Unknown"))
    prompt = f"""
You are a highly empathetic, professional medical AI educator.
Your goal is to explain Chest X-ray AI findings to a patient based on their symptoms.
CRITICAL RULE: You must NEVER definitively diagnose. Always use language like "The AI detected patterns consistent with..." and advise consulting a doctor.

Patient Profile:
- Age: {patient_data.get("age", "Unknown")}
- Fever: {"Yes" if patient_data.get("fever") else "No"}
- Cough Duration: {patient_data.get("cough_duration_days", 0)} days
- Smoking Status: {smoking_status}

AI Findings:
{ml_results}

Respond in exactly two sections using Markdown:
### 🩺 Clinical Observation
(2-3 sentences combining their symptoms with the AI findings for a layman.)

### 📋 Suggested Next Steps
(3 actionable, bulleted next steps based on severity.)
"""
    models_to_try = _gemini_educator_models_to_try(api_key)
    last_error: BaseException | None = None
    last_code: str | None = None
    last_model: str | None = None

    if not models_to_try:
        return {
            "status": "failed",
            "text": "Could not generate clinical summary due to API model availability.",
            "error_code": "gemini_not_found",
        }

    try:
        genai.configure(api_key=api_key)

        for model_name in models_to_try:
            try:
                logger.info("Attempting LLM generation with model: %s", model_name)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)

                pf = getattr(response, "prompt_feedback", None)
                if pf is not None:
                    logger.info(
                        "Educator Gemini prompt_feedback model=%s: %s", model_name, pf
                    )

                try:
                    raw = response.text
                except (ValueError, AttributeError) as exc:
                    last_code = "gemini_blocked_or_empty"
                    logger.warning(
                        "Educator Gemini model=%s returned no usable text: %s. Trying next fallback...",
                        model_name,
                        exc,
                    )
                    continue

                text = (raw or "").strip() if isinstance(raw, str) else ""
                if not text:
                    last_code = "gemini_blocked_or_empty"
                    logger.warning(
                        "Educator Gemini model=%s returned empty summary text. Trying next fallback...",
                        model_name,
                    )
                    continue

                if model_name != models_to_try[0]:
                    logger.info(
                        "Educator Gemini succeeded with fallback model %s (primary was %s).",
                        model_name,
                        models_to_try[0],
                    )
                return {"status": "success", "text": text}

            except Exception as exc:
                last_error = exc
                last_code = _classify_gemini_exception(exc)
                last_model = model_name
                logger.warning(
                    "Educator Gemini failed with model=%s (%s): %s. Trying next fallback...",
                    model_name,
                    last_code,
                    exc,
                )
                if not _gemini_error_should_try_next_model(last_code):
                    break
                continue

        logger.error(
            "All LLM generation attempts failed (models_tried=%s). Last error (%s): %s",
            models_to_try,
            last_code,
            last_error,
            exc_info=last_error is not None,
        )
        if last_code in ("gemini_unauthenticated", "gemini_permission_denied"):
            fail_text = "Could not generate clinical summary."
        else:
            fail_text = (
                "Could not generate clinical summary due to API model availability."
            )
        return {
            "status": "failed",
            "text": fail_text,
            "error_code": last_code or "gemini_unknown",
        }
    except Exception as exc:
        code = _classify_gemini_exception(exc)
        logger.error("LLM Generation failed (%s): %s", code, exc, exc_info=True)
        return {
            "status": "failed",
            "text": "Could not generate clinical summary.",
            "error_code": code,
        }


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

    def _normalize_smoking(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "current" if value else "never"
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"never", "former", "current"}:
                return lowered
        raise ValueError("Invalid smoking value. Use never/former/current.")

    def _normalize_breathing(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "severe" if value else "none"
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"none", "mild", "severe"}:
                return lowered
        raise ValueError("Invalid breathingDifficulty value. Use none/mild/severe.")

    merged_patient: dict[str, Any] = {}
    patient_data = parsed.get("patient_data")
    if isinstance(patient_data, dict):
        merged_patient.update(patient_data)

    key_pairs = {
        "age": ["age"],
        "fever": ["fever"],
        "cough_duration_days": ["cough_duration_days", "coughDurationDays"],
        "smoking": ["smoking"],
        "breathing_difficulty": ["breathing_difficulty", "breathingDifficulty"],
    }
    for canonical, candidates in key_pairs.items():
        for candidate in candidates:
            if candidate in parsed:
                merged_patient[canonical] = parsed[candidate]

    if "smoking" in merged_patient:
        merged_patient["smoking"] = _normalize_smoking(merged_patient["smoking"])
    if "breathing_difficulty" in merged_patient:
        merged_patient["breathing_difficulty"] = _normalize_breathing(
            merged_patient["breathing_difficulty"]
        )

    has_required_patient_keys = all(
        key in merged_patient and merged_patient[key] is not None
        for key in ("age", "fever", "cough_duration_days")
    )
    if has_required_patient_keys:
        try:
            parsed["patient_data"] = PatientData.model_validate(merged_patient).model_dump()
        except ValidationError as exc:
            raise ValueError(f"Invalid patient_data schema: {exc.errors()}") from exc

    return parsed


def _questionnaire_satisfied(questionnaire_data: dict[str, Any] | None) -> bool:
    if questionnaire_data is None:
        return False
    patient_data = questionnaire_data.get("patient_data")
    if isinstance(patient_data, dict) and all(
        key in patient_data and patient_data[key] is not None
        for key in ("age", "fever", "cough_duration_days")
    ):
        return True
    clinical_data = questionnaire_data.get("clinical_data")
    if isinstance(clinical_data, dict) and len(clinical_data) > 0:
        return True
    return False


def _validate_api_key(x_api_key: str | None) -> JSONResponse | None:
    if not REQUIRE_API_KEY:
        return None
    if not API_KEY:
        return _error_response("Server API key is not configured.", 500)
    if x_api_key != API_KEY:
        return _error_response("Unauthorized.", 401)
    return None


def _densenet_analyze_error_payload(message: str | None = None) -> dict[str, Any]:
    return {
        "error": message or "Model not available",
        "model_name": "DenseNet-121",
    }


def _build_pipeline_outputs(
    predictions: dict[str, float],
    questionnaire_data: dict[str, Any] | None,
    heatmap_base64: str,
    *,
    model1_override: tuple[str, float] | None = None,
    model1_pytorch_inference_ok: bool = False,
    model1_gradcam_b64: str | None = None,
    model1_probabilities: dict[str, float] | None = None,
    model2_override: tuple[str, float] | None = None,
    model2_probabilities: dict[str, float] | None = None,
    model2_h5_inference_ok: bool = False,
    timing_model1_ms: float = 18.0,
    timing_model2_ms: float = 22.0,
    densenet_payload: dict[str, Any] | None = None,
    timing_densenet_ms: float = 0.0,
) -> dict[str, Any]:
    densenet_block = densenet_payload or _densenet_analyze_error_payload()
    densenet_neural_ok = "prediction" in densenet_block and "error" not in densenet_block

    pred = {k: float(v) for k, v in predictions.items()}
    if not pred:
        pred = {"Normal": 1.0}

    keys = set(pred.keys())
    gate_route = "early_stop" if keys == {"Normal"} else "continue"
    gate_reason = "positive_detected" if gate_route == "continue" else "both_negative"

    if keys == {"Normal"}:
        top_prediction = "Normal"
        top_confidence = float(pred["Normal"])
    else:
        non_normal = {k: float(v) for k, v in pred.items() if k != "Normal"}
        if non_normal:
            top_prediction = max(non_normal, key=non_normal.get)
            top_confidence = float(non_normal[top_prediction])
        else:
            top_prediction = "Normal"
            top_confidence = float(pred.get("Normal", 1.0))

    if model1_pytorch_inference_ok and model1_override is not None:
        model1_label, model1_confidence = model1_override
        model1_confidence = round(float(model1_confidence), 3)
        model1_positive_score = (
            1.0 - model1_confidence if model1_label == "Normal" else model1_confidence
        )
    else:
        model1_label = "Normal"
        model1_confidence = 1.0
        model1_positive_score = max(
            float(pred.get("Pneumonia", 0.0)),
            float(pred.get("Infiltration", 0.0)),
            float(pred.get("COVID-19", 0.0)),
        )

    if model2_h5_inference_ok and model2_override is not None:
        model2_label, model2_confidence = model2_override
        model2_confidence = round(float(model2_confidence), 3)
    else:
        model2_label = "Normal"
        model2_confidence = 1.0

    questionnaire_complete = _questionnaire_satisfied(questionnaire_data)
    requires_questionnaire = gate_route == "continue" and not questionnaire_complete

    clinical_risk_payload = None
    model4 = None
    smoking = None
    breathing_difficulty = None
    if isinstance(questionnaire_data, dict):
        patient_data = questionnaire_data.get("patient_data")
        if isinstance(patient_data, dict):
            smoking = patient_data.get("smoking")
            breathing_difficulty = patient_data.get("breathing_difficulty")
    if not requires_questionnaire:
        if gate_route == "early_stop":
            severity = "low"
            risk_level = "low"
            recovery_outlook = "favorable"
        else:
            if model1_positive_score >= 0.8:
                severity = "high"
                risk_level = "high"
                recovery_outlook = "uncertain"
            elif model1_positive_score >= 0.6:
                severity = "moderate"
                risk_level = "medium"
                recovery_outlook = "guarded"
            else:
                severity = "low"
                risk_level = "low"
                recovery_outlook = "favorable"
            if smoking in {"former", "current"} or breathing_difficulty in {"mild", "severe"}:
                recovery_outlook = "guarded"

        clinical_risk_payload = {
            "enabled": True,
            "severity": severity,
            "risk_level": risk_level,
            "recovery_outlook": recovery_outlook,
        }

        summary_lines = [
            f"Educational analysis suggests {model2_label} with top finding "
            f"{top_prediction} ({top_confidence:.2f})."
        ]
        patient_data = questionnaire_data.get("patient_data") if isinstance(questionnaire_data, dict) else None
        if isinstance(patient_data, dict):
            if patient_data.get("fever") is True:
                summary_lines.append("Questionnaire indicates reported fever.")
            cough_days = patient_data.get("cough_duration_days")
            if isinstance(cough_days, int):
                summary_lines.append(f"Reported cough duration: {cough_days} days.")
            if smoking == "former":
                summary_lines.append("Smoking history: former smoker.")
            elif smoking == "current":
                summary_lines.append("Smoking history: current smoker.")
            if breathing_difficulty == "mild":
                summary_lines.append("Breathing difficulty reported as mild.")
            elif breathing_difficulty == "severe":
                summary_lines.append("Breathing difficulty reported as severe.")

        model4 = {
            "summary": " ".join(summary_lines),
            "recommended_actions": [
                "Review this result with a licensed radiologist.",
                "Correlate with patient symptoms and vitals.",
                "Repeat imaging or follow-up per clinical protocol if needed.",
            ],
            "disclaimer": (
                "This is an educational synthesis and not a medical diagnosis."
            ),
        }

    t_dn = round(float(timing_densenet_ms), 2)
    model4_timing = 40 if model4 else 0
    t1 = round(float(timing_model1_ms), 2)
    t2 = round(float(timing_model2_ms), 2)
    timing_ms = {
        "model1": t1,
        "model2": t2,
        "model3": t_dn,
        "model4": model4_timing,
        "total": round(t1 + t2 + t_dn + model4_timing, 2),
    }

    any_neural_ok = (
        model1_pytorch_inference_ok
        or model2_h5_inference_ok
        or densenet_neural_ok
    )
    run_mode = "hybrid" if any_neural_ok else "rules"
    if model1_pytorch_inference_ok:
        model1_status = "ok"
    elif not ENABLE_MODEL1:
        model1_status = "skipped"
    elif MODEL1_PT is None:
        model1_status = "load_failed"
    else:
        model1_status = "fallback"
    if model2_h5_inference_ok:
        model2_status = "ok"
    elif not ENABLE_MODEL2_H5:
        model2_status = "skipped"
    elif MODEL2_H5 is None:
        model2_status = "load_failed"
    else:
        model2_status = "fallback"
    if densenet_neural_ok:
        model3_flat_result = "model"
    elif not ENABLE_DENSENET121:
        model3_flat_result = "skipped"
    elif MODEL_DENSENET121 is None:
        model3_flat_result = "skipped"
    else:
        model3_flat_result = "failed"

    # Flat section tags for frontend transparency (aligned with /analyze contract).
    provenance_flat = {
        "model1_result": "model" if model1_pytorch_inference_ok else "rules",
        "model2_result": "model" if model2_h5_inference_ok else "rules",
        "model3_result": model3_flat_result,
        "clinical_risk_result": "rules" if clinical_risk_payload else "skipped",
        "gate_decision": "rules",
        "findings": "rules",
        "doctor_questions": "rules",
        "report_summary": "rules",
        "anatomy_guide": "static",
    }
    warnings: list[dict[str, Any]] = []

    return {
        "success": True,
        "predictions": predictions,
        "gradcam": {
            "heatmap_base64": heatmap_base64,
            "top_prediction": top_prediction,
            "confidence": round(top_confidence, 3),
        },
        "model1": {
            "prediction": model1_label,
            "confidence": model1_confidence,
            "status": "success" if model1_pytorch_inference_ok else "failed",
            "probabilities": model1_probabilities or {},
            "label": model1_label,
            **(
                {"model_name": "ResNet50-3Class"}
                if model1_pytorch_inference_ok
                else {}
            ),
            **({"gradcam": model1_gradcam_b64} if model1_gradcam_b64 else {}),
        },
        "model2": {
            "prediction": model2_label,
            "confidence": model2_confidence,
            "status": "success" if model2_h5_inference_ok else "failed",
            "probabilities": model2_probabilities or {},
            "label": model2_label,
        },
        "gate": {
            "route": gate_route,
            "reason": gate_reason,
        },
        "clinical_risk": clinical_risk_payload,
        "model3": densenet_block,
        "model4": model4,
        "timing_ms": timing_ms,
        "requires_questionnaire": requires_questionnaire,
        "warnings": warnings,
        "provenance": {
            "run_mode": run_mode,
            **provenance_flat,
            "model1": {
                "source": "model" if model1_pytorch_inference_ok else "rules",
                "status": model1_status,
                "model_id": "resnet50-3class"
                if model1_pytorch_inference_ok
                else "resnet50-3class-unavailable",
                "model_version": "v1" if model1_pytorch_inference_ok else "n/a",
            },
            "model2": {
                "source": "model" if model2_h5_inference_ok else "rules",
                "status": model2_status,
                "model_id": "h5-model2" if model2_h5_inference_ok else "h5-model2-unavailable",
                "model_version": "pilot" if model2_h5_inference_ok else "n/a",
            },
            "model3": {
                "source": "model" if densenet_neural_ok else "rules",
                "status": (
                    "ok"
                    if densenet_neural_ok
                    else (
                        "skipped"
                        if not ENABLE_DENSENET121 or MODEL_DENSENET121 is None
                        else "failed"
                    )
                ),
                "model_id": "densenet121" if densenet_neural_ok else "densenet121-unavailable",
                "model_version": "v1" if densenet_neural_ok else "n/a",
            },
            "clinical_risk": {
                "source": "rule",
                "status": "ok" if clinical_risk_payload else "skipped",
            },
            "model4": {"source": "rule", "status": "ok" if model4 else "skipped"},
        },
    }


def _build_demo_normal_payload(
    questionnaire_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """May presentation bypass: healthy-normal ML snapshot without running inference."""
    predictions: dict[str, float] = {"Normal": 0.99}
    # Keys must match MODEL1_LABELS / frontend schema (not ad-hoc display strings).
    demo_m1_probs = {
        "Normal": 0.99,
        "Pneumonia-Bacteria": 0.0,
        "Pneumonia-Virus": 0.01,
    }
    demo_m2_probs = {
        "Normal": 0.98,
        "Lung Opacity": 0.01,
        "Viral Pneumonia": 0.01,
    }
    demo_densenet: dict[str, Any] = {
        "class_id": 0,
        "class_name": "Normal",
        "confidence_score": 0.97,
        "prediction": "Normal",
        "confidence": 97.0,
        "probabilities": {
            "Normal": 0.97,
            "Pneumonia-Bacteria": 0.015,
            "Pneumonia-Virus": 0.015,
        },
        "gradcam": "",
        "input_preview_base64": "",
        "model_name": "DenseNet-121",
    }
    payload = _build_pipeline_outputs(
        predictions,
        questionnaire_data,
        heatmap_base64="",
        model1_override=("Normal", 0.99),
        model1_pytorch_inference_ok=True,
        model1_gradcam_b64=None,
        model1_probabilities=demo_m1_probs,
        model2_override=("Normal", 0.98),
        model2_probabilities=demo_m2_probs,
        model2_h5_inference_ok=True,
        timing_model1_ms=10.0,
        timing_model2_ms=12.0,
        densenet_payload=demo_densenet,
        timing_densenet_ms=14.0,
    )
    payload["model1"] = {
        "prediction": "Normal",
        "confidence": 0.99,
        "status": "success",
        "probabilities": demo_m1_probs,
        "label": "Normal",
        "model_name": "ResNet50-3Class",
    }
    payload["model2"] = {
        "prediction": "Normal",
        "confidence": 0.98,
        "status": "success",
        "probabilities": demo_m2_probs,
        "label": "Normal",
    }
    # Keep full DenseNet-shaped model3 from _build_pipeline_outputs (demo_densenet).
    # Do not replace with a minimal dict — clients validate class_id, probabilities, etc.
    demo_m4_probs = {
        "COVID-19": 0.01,
        "Lung_Opacity": 0.01,
        "Normal": 0.96,
        "Pneumonia": 0.01,
        "Tuberculosis": 0.005,
        "Viral_Pneumonia": 0.005,
    }
    # Align demo keys with MODEL4_SWINT_LABELS when env overrides label list.
    if set(demo_m4_probs) != set(MODEL4_SWINT_LABELS):
        demo_m4_probs = {lbl: (0.96 if lbl == "Normal" else round(0.04 / max(len(MODEL4_SWINT_LABELS) - 1, 1), 4)) for lbl in MODEL4_SWINT_LABELS}
    payload["model4_swint"] = {
        "prediction": "Normal",
        "confidence": 0.96,
        "status": "success",
        "probabilities": demo_m4_probs,
        "model_name": "Swin-T",
    }
    # TODO: Add Model 5 mock here
    # TODO: Add Model 6 mock here
    payload["copd_screening"] = {
        "prediction": "Low COPD Risk",
        "confidence": 0.91,
        "status": "success",
    }
    return payload


async def _analyze_internal(
    image: UploadFile,
    questionnaire: str | None,
    x_api_key: str | None,
    gemini_api_key: str | None,
) -> JSONResponse:
    try:
        auth_error = _validate_api_key(x_api_key)
        if auth_error is not None:
            return auth_error

        if not image.filename:
            return _error_response("Missing uploaded image filename.", 400)

        _fname = (image.filename or "").strip()
        is_demo = _fname in {"demo_normal.jpeg", "demo_normal.jpg"}

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
        patient_data: dict[str, Any] | None = None
        if isinstance(questionnaire_data, dict):
            maybe_patient_data = questionnaire_data.get("patient_data")
            if isinstance(maybe_patient_data, dict):
                patient_data = maybe_patient_data

        if is_demo:
            logger.info(
                "Demo mode triggered: Mocking ML results, but keeping LLM live."
            )
            payload = _build_demo_normal_payload(questionnaire_data)
        else:
            model1_override: tuple[str, float] | None = None
            model1_pytorch_inference_ok = False
            model1_gradcam_b64: str | None = None
            model1_probabilities: dict[str, float] | None = None
            m1_label = "Normal"
            m1_conf_r = 0.0
            t_model1_start = time.perf_counter()
            if ENABLE_MODEL1 and MODEL1_PT is not None:
                try:
                    m1_label, m1_conf, m1_probs, m1_gc = await asyncio.to_thread(
                        _run_pytorch_model1_full, image_bytes
                    )
                    m1_conf_r = round(float(m1_conf), 3)
                    model1_override = (m1_label, m1_conf_r)
                    model1_gradcam_b64 = m1_gc
                    model1_probabilities = m1_probs
                    model1_pytorch_inference_ok = True
                except Exception as exc:
                    logger.warning("ML Model 1 PyTorch inference failed: %s", exc)
            timing_model1_ms = (time.perf_counter() - t_model1_start) * 1000.0

            model2_override: tuple[str, float] | None = None
            model2_h5_inference_ok = False
            model2_probabilities: dict[str, float] | None = None
            h5_label = "Normal"
            h5_conf_r = 0.0
            t_model2_start = time.perf_counter()
            if ENABLE_MODEL2_H5:
                _warn_model2_file_missing_once("analyze")
            if ENABLE_MODEL2_H5 and MODEL2_H5 is not None:
                try:
                    h5_label, h5_conf, h5_probs = await asyncio.to_thread(
                        _run_h5_model2, image_bytes
                    )
                    h5_conf_r = round(float(h5_conf), 3)
                    model2_override = (h5_label, h5_conf_r)
                    model2_probabilities = h5_probs
                    model2_h5_inference_ok = True
                except Exception as exc:
                    logger.warning("ML Model 2 H5 inference failed: %s", exc)
            timing_model2_ms = (time.perf_counter() - t_model2_start) * 1000.0

            model4_swint_result: dict[str, Any] = {
                "prediction": "N/A",
                "confidence": 0.0,
                "status": "failed",
            }
            model4_swint_label = "Normal"
            model4_swint_conf = 0.0
            model4_swint_inference_ok = False
            if ENABLE_MODEL4_SWINT and MODEL4_SWINT is not None:
                try:
                    model4_swint_label, model4_swint_conf, model4_swint_probs = (
                        await asyncio.to_thread(_run_swint_model4, image_bytes)
                    )
                    model4_swint_conf = round(float(model4_swint_conf), 3)
                    model4_swint_result = {
                        "prediction": model4_swint_label,
                        "confidence": model4_swint_conf,
                        "status": "success",
                        "probabilities": model4_swint_probs,
                        "model_name": "Swin-T",
                    }
                    model4_swint_inference_ok = True
                except Exception as exc:
                    logger.error("Model 4 (Swin-T) inference failed: %s", exc)

            densenet_payload: dict[str, Any] = _densenet_analyze_error_payload()
            timing_densenet_ms = 0.0
            if ENABLE_DENSENET121 and MODEL_DENSENET121 is not None:
                t_dn0 = time.perf_counter()
                try:
                    dn = await asyncio.to_thread(
                        _densenet121_predict_and_cam, image_bytes
                    )
                    densenet_payload = {**dn, "model_name": "DenseNet-121"}
                except Exception as exc:
                    logger.warning(
                        "DenseNet-121 in analyze pipeline failed (model1/model2 unaffected): %s",
                        exc,
                    )
                    densenet_payload = _densenet_analyze_error_payload(
                        (str(exc) or "Model not available")[:500]
                    )
                timing_densenet_ms = (time.perf_counter() - t_dn0) * 1000.0

            densenet_neural_ok = (
                "prediction" in densenet_payload and "error" not in densenet_payload
            )
            predictions: dict[str, float] = {}
            if model1_pytorch_inference_ok and m1_label != "Normal":
                base = "Pneumonia" if "Pneumonia" in m1_label else m1_label
                predictions[base] = max(predictions.get(base, 0.0), m1_conf_r)
            if model2_h5_inference_ok and h5_label != "Normal":
                predictions[h5_label] = max(predictions.get(h5_label, 0.0), h5_conf_r)
            if model4_swint_inference_ok and model4_swint_label != "Normal":
                m4_base = (
                    "Pneumonia"
                    if "Pneumonia" in model4_swint_label
                    else model4_swint_label
                )
                predictions[m4_base] = max(
                    predictions.get(m4_base, 0.0), model4_swint_conf
                )
            m3_class_name = str(densenet_payload.get("class_name", densenet_payload.get("prediction", "")))
            if densenet_neural_ok and m3_class_name and m3_class_name != "Normal":
                m3_pred = m3_class_name
                if "confidence_score" in densenet_payload:
                    m3_conf = float(densenet_payload["confidence_score"])
                else:
                    m3_conf = float(densenet_payload["confidence"]) / 100.0
                predictions[m3_pred] = max(predictions.get(m3_pred, 0.0), m3_conf)
            if not predictions:
                predictions["Normal"] = 1.0

            payload = _build_pipeline_outputs(
                predictions,
                questionnaire_data,
                heatmap_base64="",
                model1_override=model1_override,
                model1_pytorch_inference_ok=model1_pytorch_inference_ok,
                model1_gradcam_b64=model1_gradcam_b64,
                model1_probabilities=model1_probabilities,
                model2_override=model2_override,
                model2_probabilities=model2_probabilities,
                model2_h5_inference_ok=model2_h5_inference_ok,
                timing_model1_ms=timing_model1_ms,
                timing_model2_ms=timing_model2_ms,
                densenet_payload=densenet_payload,
                timing_densenet_ms=timing_densenet_ms,
            )
            copd_result = None
            if (
                isinstance(patient_data, dict)
                and ENABLE_COPD_MODEL
                and COPD_MODEL is not None
                and COPD_SCALER is not None
            ):
                try:
                    copd_label, copd_conf = await asyncio.to_thread(
                        _run_copd_screening, patient_data
                    )
                    copd_result = {
                        "prediction": copd_label,
                        "confidence": copd_conf,
                        "status": "success",
                    }
                except Exception as exc:
                    logger.error("COPD Inference failed: %s", exc)
                    copd_result = {"status": "failed"}

            if copd_result:
                payload["copd_screening"] = copd_result
            payload["model4_swint"] = model4_swint_result

        predictions_for_llm: dict[str, float] = payload.get("predictions") or {}
        ml_summary = {
            "Primary Finding": max(predictions_for_llm, key=predictions_for_llm.get)
            if predictions_for_llm
            else "Normal",
            "Model 1 (ResNet)": payload.get("model1", {}).get("prediction"),
            "Model 2 (ResNetV2)": payload.get("model2", {}).get("prediction"),
            "COPD Risk": payload.get("copd_screening", {}).get("prediction"),
        }
        gemini_resolved, gemini_key_source = _resolve_educator_gemini_key(gemini_api_key)
        _educator_models = (
            _gemini_educator_models_to_try(gemini_resolved) if gemini_resolved else []
        )
        logger.info(
            "Educator Gemini: key_source=%s key_length=%s will_invoke_llm=%s models=%s",
            gemini_key_source,
            len(gemini_resolved) if gemini_resolved else 0,
            bool(gemini_resolved),
            _educator_models[:5],
        )
        llm_result = await asyncio.to_thread(
            _generate_llm_summary,
            ml_summary,
            patient_data or {},
            gemini_resolved,
        )
        payload["llm_evaluation"] = llm_result
        return JSONResponse(status_code=200, content=payload)
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except Exception:
        logger.exception("Pipeline analyze internal error.")
        return _error_response("Internal server error.", 500)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "healthy"}


@app.on_event("startup")
async def _startup_load_models() -> None:
    logger.info("Resolved H5_MODEL2_LABELS order: %s", H5_MODEL2_LABELS)
    # Sanity check for label decoding against configured ordering.
    sanity_idx, sanity_label, _, _ = _model2_decode_scores([0.9, 0.05, 0.05])
    logger.info(
        "Model 2 mapping sanity check [0.9,0.05,0.05] -> index=%s label=%s",
        sanity_idx,
        sanity_label,
    )
    _load_model1_pytorch()
    _load_model2_h5()
    _load_copd_pipeline()
    _load_densenet121()
    _load_swint_model4()
    missing = _missing_enabled_model_files()
    if missing:
        for msg in missing:
            logger.error("Startup model file check: %s", msg)
        if FAIL_STARTUP_ON_MISSING_ENABLED_MODELS:
            raise RuntimeError(
                "Enabled model file(s) missing. "
                "Set FAIL_STARTUP_ON_MISSING_ENABLED_MODELS=false to continue in degraded mode."
            )


@app.get("/health")
async def health() -> dict[str, Any]:
    pth_diag = _model1_pt_path_diagnostics()
    h5_diag = _model2_h5_path_diagnostics()
    dn_diag = _densenet121_path_diagnostics()
    return {
        "status": "ok",
        "models": {
            **_registry_health_aliases(),
            "model1_pt": {
                "enabled": ENABLE_MODEL1,
                "loaded": MODEL1_PT is not None,
                "path": MODEL1_PATH,
                "absolute_path": pth_diag["absolute_path"],
                "file_exists": pth_diag["exists"],
                "file_size_bytes": pth_diag["size_bytes"],
                "error": MODEL1_PT_LOAD_ERROR,
                "labels": list(MODEL1_LABELS),
            },
            "model2_h5": {
                "enabled": ENABLE_MODEL2_H5,
                "loaded": MODEL2_H5 is not None,
                "path": H5_MODEL2_PATH,
                "absolute_path": h5_diag["absolute_path"],
                "file_exists": h5_diag["exists"],
                "file_size_bytes": h5_diag["size_bytes"],
                "error": MODEL2_H5_LOAD_ERROR,
                "labels": list(H5_MODEL2_LABELS),
            },
            "densenet121_pt": {
                "enabled": ENABLE_DENSENET121,
                "loaded": MODEL_DENSENET121 is not None,
                "path": DENSENET121_PATH,
                "absolute_path": dn_diag["absolute_path"],
                "file_exists": dn_diag["exists"],
                "file_size_bytes": dn_diag["size_bytes"],
                "error": MODEL_DENSENET121_LOAD_ERROR,
                "labels": list(CLASS_NAMES),
            },
            "model4_swint": {
                "enabled": ENABLE_MODEL4_SWINT,
                "loaded": MODEL4_SWINT is not None,
                **_model4_swint_path_diagnostics(),
                "error": MODEL4_SWINT_LOAD_ERROR,
                "labels": list(MODEL4_SWINT_LABELS),
            },
        },
    }


@app.get("/debug")
async def debug_model_status() -> dict[str, Any]:
    """Diagnostics for ML models and response provenance (no secrets)."""
    pth_diag = _model1_pt_path_diagnostics()
    h5_diag = _model2_h5_path_diagnostics()
    dn_diag = _densenet121_path_diagnostics()
    tf_ver = TENSORFLOW_VERSION or _tensorflow_version_probe()
    pt_ver = PYTORCH_VERSION or _pytorch_version_probe()
    m1_ready = ENABLE_MODEL1 and MODEL1_PT is not None
    m2_ready = ENABLE_MODEL2_H5 and MODEL2_H5 is not None
    m3_ready = ENABLE_DENSENET121 and MODEL_DENSENET121 is not None
    m4_ready = ENABLE_MODEL4_SWINT and MODEL4_SWINT is not None
    m4_diag = _model4_swint_path_diagnostics()
    hybrid_preview = m1_ready or m2_ready or m3_ready or m4_ready
    return {
        "model_registry": MODEL_REGISTRY,
        "model1_pt": {
            "enabled": ENABLE_MODEL1,
            "loaded": MODEL1_PT is not None,
            "load_error": MODEL1_PT_LOAD_ERROR,
            **pth_diag,
            "labels": list(MODEL1_LABELS),
        },
        "model2_h5": {
            "enabled": ENABLE_MODEL2_H5,
            "loaded": MODEL2_H5 is not None,
            "load_error": MODEL2_H5_LOAD_ERROR,
            **h5_diag,
            "labels": list(H5_MODEL2_LABELS),
        },
        "densenet121_pt": {
            "enabled": ENABLE_DENSENET121,
            "loaded": MODEL_DENSENET121 is not None,
            "load_error": MODEL_DENSENET121_LOAD_ERROR,
            **dn_diag,
            "labels": list(CLASS_NAMES),
            "predict_endpoint": "/predict/densenet",
            "gradcam_target_layer": "features.denseblock4",
        },
        "model4_swint": {
            "enabled": ENABLE_MODEL4_SWINT,
            "loaded": MODEL4_SWINT is not None,
            "load_error": MODEL4_SWINT_LOAD_ERROR,
            **m4_diag,
            "labels": list(MODEL4_SWINT_LABELS),
        },
        "tensorflow_version": tf_ver,
        "pytorch_version": pt_ver,
        "environment": ENVIRONMENT,
        "analyze_provenance_sources": {
            "run_mode": "hybrid" if hybrid_preview else "rules",
            "model1_result": {"source": "model" if m1_ready else "rules"},
            "model2_result": {"source": "model" if m2_ready else "rules"},
            "model3_result": {"source": "model" if m3_ready else "rules"},
            "gate_decision": {"source": "rules"},
            "findings": {"source": "rules"},
            "doctor_questions": {"source": "rules"},
            "report_summary": {"source": "rules"},
            "anatomy_guide": {"source": "static"},
        },
        "analyze_provenance_notes": {
            "findings": (
                "Aggregated predictions are built only from non-Normal outputs of "
                "ML Model 1 (PyTorch), ML Model 2 (H5), Model 4 (Swin-T), and DenseNet-121 "
                "when each runs successfully; otherwise the dict is {'Normal': 1.0}. Top-level gradcam "
                "heatmap_base64 is reserved and may be empty; model1/model3 expose their "
                "own Grad-CAM when neural paths succeed."
            ),
        },
    }


@app.post("/predict/densenet")
async def predict_densenet(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    """Standalone DenseNet-121 3-class (Normal / Pneumonia-Bacteria / Pneumonia-Virus) + Grad-CAM."""
    try:
        auth_error = _validate_api_key(x_api_key)
        if auth_error is not None:
            return auth_error

        if not ENABLE_DENSENET121 or MODEL_DENSENET121 is None:
            return _error_response(
                "DenseNet-121 model is not loaded or disabled. "
                "Set ENABLE_DENSENET121=true and ensure DENSENET121_PATH points to a valid .pth file.",
                503,
            )

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

        body = await asyncio.to_thread(_densenet121_predict_and_cam, image_bytes)
        return JSONResponse(status_code=200, content={"success": True, **body})
    except Exception as exc:
        logger.exception("DenseNet-121 predict failed: %s", exc)
        return _error_response(
            f"DenseNet-121 inference failed: {exc}" if str(exc) else "DenseNet-121 inference failed.",
            500,
        )


@app.post("/api/v1/generate-questions")
async def generate_questions(
    req: QuestionRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    """Return suggested doctor questions for high-attention finding labels (educational)."""
    auth_error = _validate_api_key(x_api_key)
    if auth_error is not None:
        return auth_error
    suggested_questions: list[dict[str, str]] = []
    q_index = 1
    for finding in req.high_attention_findings:
        dict_key = finding.replace(" ", "_")
        if dict_key in CLINICAL_DICTIONARY:
            for q_text in CLINICAL_DICTIONARY[dict_key]:
                suggested_questions.append(
                    {
                        "id": f"q{q_index}",
                        "text": q_text,
                        "finding_trigger": finding,
                    }
                )
                q_index += 1
    return JSONResponse(
        status_code=200,
        content={"status": "success", "suggested_questions": suggested_questions},
    )


@app.post("/api/v1/gemini/health-check")
async def gemini_api_key_health_check(
    gemini_api_key: str | None = Form(None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    """Validate a user-supplied Gemini API key before continuing (e.g. Next on upload step).

    If ``gemini_api_key`` is blank or whitespace-only, returns ``skipped`` with ``ok: true`` so
    the client can skip validation when BYOK is optional.

    Multipart field name matches ``POST /api/v1/analyze`` (``gemini_api_key``).
    """
    auth_error = _validate_api_key(x_api_key)
    if auth_error is not None:
        return auth_error

    raw = (gemini_api_key or "").strip()
    if not raw:
        return JSONResponse(
            status_code=200,
            content={
                "status": "skipped",
                "ok": True,
                "message": "No API key provided; validation skipped.",
            },
        )

    logger.info(
        "Gemini health probe: key_length=%s models=%s",
        len(raw),
        _gemini_educator_models_to_try(raw)[:5],
    )
    result = await asyncio.to_thread(_probe_user_gemini_key, raw)
    return JSONResponse(status_code=200, content=result)


@app.post("/api/v1/analyze")
async def analyze_v1(
    image: UploadFile = File(...),
    questionnaire: str | None = Form(None),
    gemini_api_key: str | None = Form(None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    return await _analyze_internal(
        image=image,
        questionnaire=questionnaire,
        x_api_key=x_api_key,
        gemini_api_key=gemini_api_key,
    )


@app.post("/pipeline/analyze")
async def analyze_pipeline_alias(
    image: UploadFile = File(...),
    questionnaire: str | None = Form(None),
    gemini_api_key: str | None = Form(None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    return await _analyze_internal(
        image=image,
        questionnaire=questionnaire,
        x_api_key=x_api_key,
        gemini_api_key=gemini_api_key,
    )
