import asyncio
import json
import logging
import os
import shutil
import base64
import math
import tempfile
import time
import uuid
from io import BytesIO
from typing import Any, List, Literal

import joblib
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, File, Form, Header, UploadFile, Path
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

# ML Model 6 (legacy ResNet-152V2 vision H5): optional; was formerly the Model 2 X-ray slot.
MODEL2_VISION_H5_IMAGE_SIZE: tuple[int, int] = (224, 224)


class PatientData(BaseModel):
    age: int
    fever: bool
    cough_duration_days: int
    smoking: str | None = None
    breathing_difficulty: str | None = None


class QuestionRequest(BaseModel):
    high_attention_findings: List[str]
    gemini_api_key: str | None = None
    locale: str | None = None
    scan_summary: str | None = None


class EducationalInsightItem(BaseModel):
    id: str
    title: str
    text: str
    finding_trigger: str
    category: str = "overview"


def _normalize_insight_finding_key(finding: str) -> str:
    raw = finding.strip()
    if not raw:
        return raw
    aliases = {
        "Infiltration": "Lung Opacity",
        "Lung_Opacity": "Lung Opacity",
        "COVID_19": "COVID-19",
    }
    return aliases.get(raw, raw)


def _resolve_insights_fallback_locale(locale: str | None) -> str:
    loc = (locale or "en").strip()
    if loc in ("zh-Hant", "zh-TW", "zh-HK"):
        return "zh-Hant"
    if loc in ("zh-Hans", "zh-CN", "zh"):
        return "zh-Hans"
    return "en"


EDUCATIONAL_INSIGHTS_FALLBACK_BY_LOCALE: dict[str, dict[str, list[dict[str, str]]]] = {
    "en": {
        "Pneumonia": [
            {
                "title": "What this pattern may indicate",
                "text": (
                    "Pneumonia on chest X-ray often shows areas where air spaces look filled or inflamed. "
                    "Your clinician combines imaging with symptoms, exams, and sometimes blood tests or cultures—not AI scores alone."
                ),
                "category": "overview",
            },
            {
                "title": "Treatment and follow-up (general)",
                "text": (
                    "Care may include antibiotics for suspected bacterial infection, rest and fluids, oxygen if needed, "
                    "and repeat imaging when symptoms persist. Vaccines (influenza, pneumococcal, COVID-19 where appropriate) "
                    "help prevent some pneumonias."
                ),
                "category": "treatment",
            },
        ],
        "Lung Opacity": [
            {
                "title": "Understanding lung opacity",
                "text": (
                    "Opacity means part of the lung looks denser or hazier than normal. Causes range from infection and fluid "
                    "to inflammation or scarring; radiology wording and your symptoms guide next steps."
                ),
                "category": "overview",
            },
            {
                "title": "Typical management themes",
                "text": (
                    "Doctors may order follow-up X-rays, CT, or labs depending on context. Treatment targets the underlying "
                    "cause—only your care team should decide what applies to you."
                ),
                "category": "treatment",
            },
        ],
        "COVID-19": [
            {
                "title": "Imaging and COVID-19",
                "text": (
                    "AI may flag patterns seen with viral pneumonia, including COVID-19. Imaging supports assessment "
                    "but does not replace PCR/antigen testing or clinical diagnosis."
                ),
                "category": "overview",
            },
            {
                "title": "Care and prevention context",
                "text": (
                    "Management ranges from home monitoring to antivirals in eligible high-risk patients and hospital care "
                    "when severe. Check official health-authority guidance on vaccines and variants with your clinician."
                ),
                "category": "treatment",
            },
        ],
        "Pleural_Thickening": [
            {
                "title": "Pleural thickening on X-ray",
                "text": (
                    "Thickening along the lung lining can reflect prior inflammation, asbestos exposure, or other causes. "
                    "Many cases are chronic and stable; new or worsening thickening may warrant CT and specialist review."
                ),
                "category": "overview",
            },
        ],
    },
    "zh-Hant": {
        "Pneumonia": [
            {
                "title": "影像可能代表什麼",
                "text": (
                    "胸肺 X 光上的肺炎常見表現是肺泡區域較實或發炎。醫師會結合症狀、理學檢查，"
                    "有時加上血液或培養檢查，而不只依賴 AI 分數。"
                ),
                "category": "overview",
            },
            {
                "title": "治療與跟進（一般說明）",
                "text": (
                    "視病因而定，可能包括抗生素、休息與補充水分、需要時的氧氣治療，以及症狀持續時的覆照影像。"
                    "流感、肺炎鏈球菌、COVID-19 等疫苗有助降低部分肺炎風險。"
                ),
                "category": "treatment",
            },
        ],
        "Lung Opacity": [
            {
                "title": "肺野不透明度的意義",
                "text": (
                    "表示肺部部分區域在 X 光上較濃或較霧，可能與感染、積液、發炎或疤痕等有關，"
                    "需配合放射科報告與臨床情況判斷。"
                ),
                "category": "overview",
            },
            {
                "title": "常見處理方向",
                "text": "醫師可能安排覆照、電腦掃描或化驗；治療針對真正病因，請以你的主診醫師計劃為準。",
                "category": "treatment",
            },
        ],
        "COVID-19": [
            {
                "title": "與 COVID-19 相關的影像表現",
                "text": (
                    "AI 可能標示與病毒性肺炎（包括 COVID-19）相關的模式。"
                    "影像有助評估，但不能取代 PCR/抗原檢測或臨床診斷。"
                ),
                "category": "overview",
            },
            {
                "title": "照護與預防背景",
                "text": (
                    "由居家監測到高風險族群的抗病毒治療，以及需要時的住院支援。"
                    "疫苗與公共衛生建議會更新，請向醫護人員及官方來源查證。"
                ),
                "category": "treatment",
            },
        ],
        "Pleural_Thickening": [
            {
                "title": "肋膜增厚",
                "text": (
                    "肺膜沿線增厚可能與既往發炎、石棉暴露等有關；許多情況屬慢性且穩定，"
                    "若為新出現或惡化，醫師可能安排 CT 或專科評估。"
                ),
                "category": "overview",
            },
        ],
    },
    "zh-Hans": {
        "Pneumonia": [
            {
                "title": "影像可能代表什么",
                "text": (
                    "胸片上的肺炎常表现为肺泡区域较实或发炎。医生会结合症状、查体，"
                    "有时加上血液或培养检查，而不只依赖 AI 分数。"
                ),
                "category": "overview",
            },
            {
                "title": "治疗与随访（一般说明）",
                "text": (
                    "视病因而定，可能包括抗生素、休息与补液、需要时的氧疗，以及症状持续时的复查影像。"
                    "流感、肺炎球菌、COVID-19 等疫苗有助于降低部分肺炎风险。"
                ),
                "category": "treatment",
            },
        ],
        "Lung Opacity": [
            {
                "title": "肺野不透明度的意义",
                "text": (
                    "表示肺部部分区域在 X 光上较浓或较雾，可能与感染、积液、炎症或瘢痕等有关，"
                    "需结合放射科报告与临床情况判断。"
                ),
                "category": "overview",
            },
            {
                "title": "常见处理方向",
                "text": "医生可能安排复查、CT 或化验；治疗针对真正病因，请以你的主治医生方案为准。",
                "category": "treatment",
            },
        ],
        "COVID-19": [
            {
                "title": "与 COVID-19 相关的影像表现",
                "text": (
                    "AI 可能标示与病毒性肺炎（包括 COVID-19）相关的模式。"
                    "影像有助于评估，但不能取代 PCR/抗原检测或临床诊断。"
                ),
                "category": "overview",
            },
            {
                "title": "照护与预防背景",
                "text": (
                    "从居家监测到高风险人群的抗病毒治疗，以及需要时的住院支持。"
                    "疫苗与公共卫生建议会更新，请向医护人员及官方来源查证。"
                ),
                "category": "treatment",
            },
        ],
        "Pleural_Thickening": [
            {
                "title": "胸膜增厚",
                "text": (
                    "肺膜沿线增厚可能与既往炎症、石棉暴露等有关；许多情况属慢性且稳定，"
                    "若为新出现或恶化，医生可能安排 CT 或专科评估。"
                ),
                "category": "overview",
            },
        ],
    },
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
# ML Model 2 — ResNet-152V2 vision H5 (Edward; questionnaire not used).
if os.getenv("ENABLE_MODEL2_VISION_H5") is not None:
    ENABLE_MODEL2_VISION_H5 = _parse_bool_env("ENABLE_MODEL2_VISION_H5", default=False)
elif os.getenv("ENABLE_MODEL6_VISION_H5") is not None:
    ENABLE_MODEL2_VISION_H5 = _parse_bool_env("ENABLE_MODEL6_VISION_H5", default=False)
elif os.getenv("ENABLE_MODEL2_H5") is not None:
    ENABLE_MODEL2_VISION_H5 = _parse_bool_env("ENABLE_MODEL2_H5", default=False)
else:
    ENABLE_MODEL2_VISION_H5 = _parse_bool_env("ENABLE_H5_MODEL", default=False)
MODEL2_VISION_H5_PATH = (
    os.getenv("MODEL2_VISION_H5_PATH")
    or os.getenv("MODEL6_VISION_H5_PATH")
    or os.getenv("H5_MODEL2_PATH")
    or os.getenv("H5_MODEL_PATH", "models/resnet152v2_lung_disease_final.h5")
).strip()

# ML Model 6 — tabular clinical screening (questionnaire → scaler.pkl → Keras H5).
MODEL6_TABULAR_PATH = (
    os.getenv("MODEL6_TABULAR_PATH")
    or os.getenv("MODEL2_TABULAR_PATH")
    or os.getenv("COPD_MODEL_PATH", "models/copd_screening_model.h5")
).strip()
MODEL6_SCALER_PATH = (
    os.getenv("MODEL6_SCALER_PATH")
    or os.getenv("MODEL2_SCALER_PATH")
    or os.getenv("COPD_SCALER_PATH", "models/scaler.pkl")
).strip()
if os.getenv("ENABLE_MODEL6_TABULAR") is not None:
    ENABLE_MODEL6_TABULAR = _parse_bool_env("ENABLE_MODEL6_TABULAR", default=True)
elif os.getenv("ENABLE_MODEL2_TABULAR") is not None:
    ENABLE_MODEL6_TABULAR = _parse_bool_env("ENABLE_MODEL2_TABULAR", default=True)
elif os.getenv("ENABLE_COPD_MODEL") is not None:
    ENABLE_MODEL6_TABULAR = os.getenv("ENABLE_COPD_MODEL", "true").lower() == "true"
else:
    ENABLE_MODEL6_TABULAR = True
MODEL6_TABULAR_LABELS = ("Low COPD Risk", "High COPD Risk")
# Edward ResNet-152V2 (legacy H5_MODEL2 / model2_vision_h5): 3 classes, index order per Edward's training.
# Keras ImageDataGenerator alphabetical order would be Lung_Opacity, Normal, Viral_Pneumonia — confirm with Edward.
H5_MODEL2_LABELS: tuple[str, ...] = ("Normal", "Viral_Pneumonia", "Lung_Opacity")
MODEL2_VISION_LABELS = [
    lbl.replace("_", " ")
    for lbl in _parse_csv(
        os.getenv(
            "MODEL2_VISION_LABELS",
            os.getenv("H5_MODEL2_LABELS", ",".join(H5_MODEL2_LABELS)),
        )
    )
]
# resnet_v2 = Keras ResNetV2 ImageNet scaling; scale_01 = divide by 255 only ([0, 1]).
MODEL2_PREPROCESS_MODE = (
    os.getenv("MODEL2_PREPROCESS_MODE")
    or os.getenv("MODEL6_PREPROCESS_MODE", "resnet_v2")
).strip().lower()
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
MODEL5_DENSENET_PATH = (
    os.getenv("MODEL5_DENSENET_PATH", "models/best_model_DENSENET121.h5").strip()
)
# ML Model 5 (Keras DenseNet-121 H5): softmax size must match len(MODEL5_DENSENET_LABELS).
# Default: NIH ChestX-ray 14 classes in ImageFolder alphabetical order (index 0..13).
# Override via MODEL5_DENSENET_LABELS if Dicky's training used different folder names/order.
MODEL5_DENSENET_LABELS = _parse_csv(
    os.getenv(
        "MODEL5_DENSENET_LABELS",
        "Atelectasis,Cardiomegaly,Consolidation,Edema,Effusion,Emphysema,Fibrosis,Hernia,"
        "Infiltration,Mass,Nodule,Pleural_Thickening,Pneumonia,Pneumothorax",
    )
)
MODEL5_DENSENET_IMAGE_SIZE: tuple[int, int] = (224, 224)
if os.getenv("ENABLE_MODEL5_DENSENET") is not None:
    ENABLE_MODEL5_DENSENET = _parse_bool_env("ENABLE_MODEL5_DENSENET", default=False)
else:
    ENABLE_MODEL5_DENSENET = os.path.isfile(MODEL5_DENSENET_PATH)
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
MODEL6_TABULAR: Any = None
MODEL6_SCALER: Any = None
MODEL6_TABULAR_LOAD_ERROR: str | None = None
MODEL2_VISION_H5: Any = None
MODEL2_VISION_H5_LOAD_ERROR: str | None = None
MODEL_DENSENET121: Any = None
MODEL_DENSENET121_LOAD_ERROR: str | None = None
_DENSENET121_PREPROCESS: Any = None
MODEL4_SWINT: Any = None
MODEL4_SWINT_LOAD_ERROR: str | None = None
MODEL5_DENSENET_H5: Any = None
MODEL5_DENSENET_LOAD_ERROR: str | None = None
TENSORFLOW_VERSION: str | None = None
PYTORCH_VERSION: str | None = None
_MODEL2_VISION_FILE_MISSING_WARNED = False

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
        "name": "ResNet-152V2 (Edward)",
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
    {
        "id": "model5",
        "health_key": "model5_densenet_h5",
        "name": "DenseNet-121 (H5)",
        "kind": "keras",
    },
    {
        "id": "model6",
        "health_key": "model6_tabular",
        "name": "Clinical COPD screening (tabular)",
        "kind": "tabular",
    },
]


def _registry_health_aliases() -> dict[str, Any]:
    """Compact enabled/loaded flags for /health (parallel to detailed * _pt / * _h5 blocks)."""
    return {
        "model1_resnet50": {
            "enabled": ENABLE_MODEL1,
            "loaded": MODEL1_PT is not None,
        },
        "model6_tabular": {
            "enabled": ENABLE_MODEL6_TABULAR,
            "loaded": MODEL6_TABULAR is not None and MODEL6_SCALER is not None,
        },
        "model2_resnet152v2": {
            "enabled": ENABLE_MODEL2_VISION_H5,
            "loaded": MODEL2_VISION_H5 is not None,
        },
        "model3_densenet121": {
            "enabled": ENABLE_DENSENET121,
            "loaded": MODEL_DENSENET121 is not None,
        },
        "model4_swint": {
            "enabled": ENABLE_MODEL4_SWINT,
            "loaded": MODEL4_SWINT is not None,
        },
        "model5_densenet_h5": {
            "enabled": ENABLE_MODEL5_DENSENET,
            "loaded": MODEL5_DENSENET_H5 is not None,
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


def _model5_densenet_path_diagnostics() -> dict[str, Any]:
    abs_path = os.path.abspath(MODEL5_DENSENET_PATH)
    exists = os.path.isfile(abs_path)
    size_bytes: int | None = None
    if exists:
        try:
            size_bytes = os.path.getsize(abs_path)
        except OSError as exc:
            logger.warning("Could not stat Model 5 H5 file %s: %s", abs_path, exc)
    return {
        "path": MODEL5_DENSENET_PATH,
        "absolute_path": abs_path,
        "exists": exists,
        "size_bytes": size_bytes,
    }


def _model5_label_for_api(raw: str) -> str:
    return raw.replace("_", " ")


def _h5_output_num_classes(model: Any) -> int:
    shape = getattr(model, "output_shape", None)
    if shape is None:
        raise ValueError("Keras model has no output_shape.")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        dim = shape[-1]
        if dim is not None:
            return int(dim)
    raise ValueError(f"Could not infer class count from output_shape={shape!r}.")


def _load_densenet_model5() -> None:
    global MODEL5_DENSENET_H5, MODEL5_DENSENET_LOAD_ERROR, TENSORFLOW_VERSION
    if not ENABLE_MODEL5_DENSENET:
        MODEL5_DENSENET_H5 = None
        MODEL5_DENSENET_LOAD_ERROR = None
        logger.info("Model 5 (DenseNet-121 H5) skipped: ENABLE_MODEL5_DENSENET is false.")
        return

    if not MODEL5_DENSENET_LABELS:
        MODEL5_DENSENET_H5 = None
        MODEL5_DENSENET_LOAD_ERROR = "MODEL5_DENSENET_LABELS must not be empty."
        logger.error("%s", MODEL5_DENSENET_LOAD_ERROR)
        return

    diag = _model5_densenet_path_diagnostics()
    path = diag["absolute_path"]
    logger.info(
        "Model 5 (DenseNet-121 H5) load starting: path=%r absolute_path=%r exists=%s "
        "size_bytes=%s labels=%r",
        diag["path"],
        path,
        diag["exists"],
        diag["size_bytes"],
        MODEL5_DENSENET_LABELS,
    )
    if not diag["exists"]:
        MODEL5_DENSENET_H5 = None
        MODEL5_DENSENET_LOAD_ERROR = (
            f"Model 5 H5 file not found at {path!r} "
            f"(from MODEL5_DENSENET_PATH={diag['path']!r})."
        )
        logger.warning("%s Model 5 disabled.", MODEL5_DENSENET_LOAD_ERROR)
        return

    try:
        import tensorflow as tf  # type: ignore

        if TENSORFLOW_VERSION is None:
            TENSORFLOW_VERSION = str(tf.__version__)
            logger.info("TensorFlow version: %s", TENSORFLOW_VERSION)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        co = _make_h5_custom_objects(tf)
        try:
            logger.info("Model 5 H5: attempting tf.keras.models.load_model (direct).")
            model = tf.keras.models.load_model(path, compile=False, custom_objects=co)
        except Exception:
            logger.warning(
                "Model 5 H5 direct load failed; trying compat path.",
                exc_info=True,
            )
            model = _load_h5_model_compat(tf, path)

        num_classes = _h5_output_num_classes(model)
        if len(MODEL5_DENSENET_LABELS) != num_classes:
            raise ValueError(
                f"MODEL5_DENSENET_LABELS has {len(MODEL5_DENSENET_LABELS)} labels but "
                f"checkpoint expects {num_classes}. Set MODEL5_DENSENET_LABELS to match "
                "training class order (index 0..N-1). Default assumes NIH ChestX-ray 14 "
                "classes in ImageFolder alphabetical order — confirm with Dicky."
            )

        MODEL5_DENSENET_H5 = model
        MODEL5_DENSENET_LOAD_ERROR = None
        logger.info(
            "Model 5 (DenseNet-121 H5) loaded successfully: %s classes=%s labels=%s",
            path,
            num_classes,
            MODEL5_DENSENET_LABELS,
        )
    except Exception as exc:
        MODEL5_DENSENET_H5 = None
        MODEL5_DENSENET_LOAD_ERROR = str(exc)
        logger.exception("Failed to load Model 5 (DenseNet-121 H5): %s", exc)


def _preprocess_model5_densenet_numpy(image_bytes: bytes) -> Any:
    """Model 5: RGB 224×224, DenseNet preprocess_input (fallback /255)."""
    import numpy as np  # type: ignore

    w, h = MODEL5_DENSENET_IMAGE_SIZE
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((w, h), Image.Resampling.BILINEAR)
    img_array = np.asarray(image, dtype=np.float32)
    try:
        from tensorflow.keras.applications.densenet import preprocess_input  # type: ignore

        img_array = preprocess_input(img_array)
    except Exception:
        logger.warning(
            "DenseNet preprocess_input unavailable for Model 5; falling back to /255.0."
        )
        img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def _model5_decode_scores(row: Any) -> tuple[str, float, dict[str, float]]:
    import numpy as np  # type: ignore

    arr = np.asarray(row, dtype=np.float64).reshape(-1)
    exp = np.exp(arr - np.max(arr))
    probs = exp / np.sum(exp)
    idx = int(np.argmax(probs))
    raw_label = MODEL5_DENSENET_LABELS[idx]
    label = _model5_label_for_api(raw_label)
    confidence = round(float(probs[idx]), 3)
    probabilities = {
        _model5_label_for_api(MODEL5_DENSENET_LABELS[i]): round(float(probs[i]), 4)
        for i in range(len(MODEL5_DENSENET_LABELS))
    }
    return label, confidence, probabilities


def _run_densenet_model5(image_bytes: bytes) -> dict[str, Any]:
    """Model 5 (Keras DenseNet-121 H5): standard analyze block payload."""
    if MODEL5_DENSENET_H5 is None:
        raise RuntimeError(
            MODEL5_DENSENET_LOAD_ERROR or "Model 5 (DenseNet-121 H5) is not loaded."
        )
    batch = _preprocess_model5_densenet_numpy(image_bytes)
    out = MODEL5_DENSENET_H5.predict(batch, verbose=0)
    label, confidence, probabilities = _model5_decode_scores(out[0])
    return {
        "prediction": label,
        "confidence": confidence,
        "status": "success",
        "probabilities": probabilities,
        "model_name": "Model 5 (DenseNet-121)",
    }


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
    try:
        gradcam_b64 = _pytorch_gradcam_to_png_base64(
            MODEL_DENSENET121, tensor, target_layers, pred_idx, rgb_display
        )
    except Exception as exc:
        logger.warning(
            "DenseNet-121 Grad-CAM failed; returning prediction without heatmap: %s",
            exc,
        )
        gradcam_b64 = ""

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


def _model6_tabular_path_diagnostics() -> dict[str, Any]:
    model_abs = os.path.abspath(MODEL6_TABULAR_PATH)
    scaler_abs = os.path.abspath(MODEL6_SCALER_PATH)
    model_exists = os.path.isfile(model_abs)
    scaler_exists = os.path.isfile(scaler_abs)
    model_size: int | None = None
    scaler_size: int | None = None
    if model_exists:
        try:
            model_size = os.path.getsize(model_abs)
        except OSError as exc:
            logger.warning("Could not stat Model 2 tabular H5 %s: %s", model_abs, exc)
    if scaler_exists:
        try:
            scaler_size = os.path.getsize(scaler_abs)
        except OSError as exc:
            logger.warning("Could not stat Model 2 scaler %s: %s", scaler_abs, exc)
    return {
        "tabular_path": MODEL6_TABULAR_PATH,
        "tabular_absolute_path": model_abs,
        "tabular_exists": model_exists,
        "tabular_size_bytes": model_size,
        "scaler_path": MODEL6_SCALER_PATH,
        "scaler_absolute_path": scaler_abs,
        "scaler_exists": scaler_exists,
        "scaler_size_bytes": scaler_size,
        "exists": model_exists and scaler_exists,
    }


def _model2_vision_h5_path_diagnostics() -> dict[str, Any]:
    abs_path = os.path.abspath(MODEL2_VISION_H5_PATH)
    exists = os.path.isfile(abs_path)
    size_bytes: int | None = None
    if exists:
        try:
            size_bytes = os.path.getsize(abs_path)
        except OSError as exc:
            logger.warning("Could not stat Model 6 vision H5 %s: %s", abs_path, exc)
    return {
        "path": MODEL2_VISION_H5_PATH,
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
    if ENABLE_MODEL6_TABULAR:
        d2 = _model6_tabular_path_diagnostics()
        if not d2["exists"]:
            missing.append(
                f"model6 tabular missing: MODEL6_TABULAR_PATH={d2['tabular_path']!r} "
                f"MODEL6_SCALER_PATH={d2['scaler_path']!r}"
            )
    if ENABLE_MODEL2_VISION_H5:
        d6 = _model2_vision_h5_path_diagnostics()
        if not d6["exists"]:
            missing.append(
                f"model2 vision missing: MODEL2_VISION_H5_PATH={d6['path']!r} "
                f"absolute_path={d6['absolute_path']!r}"
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
    if ENABLE_MODEL5_DENSENET:
        d5 = _model5_densenet_path_diagnostics()
        if not d5["exists"]:
            missing.append(
                f"model5 missing: MODEL5_DENSENET_PATH={d5['path']!r} "
                f"absolute_path={d5['absolute_path']!r}"
            )
    return missing


def _warn_model2_vision_file_missing_once(context: str) -> None:
    global _MODEL2_VISION_FILE_MISSING_WARNED
    if _MODEL2_VISION_FILE_MISSING_WARNED:
        return
    d6 = _model2_vision_h5_path_diagnostics()
    if d6["exists"]:
        return
    _MODEL2_VISION_FILE_MISSING_WARNED = True
    logger.error(
        "Model 2 ResNet-152V2 vision H5 is missing during %s. MODEL2_VISION_H5_PATH=%r "
        "absolute_path=%r.",
        context,
        d6["path"],
        d6["absolute_path"],
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

    class PatchedDense(tf.keras.layers.Dense):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.pop("quantization_config", None)
            super().__init__(**kwargs)

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> Any:
            cfg = dict(config)
            cfg.pop("quantization_config", None)
            return super().from_config(cfg)

    return {
        "BatchNormalization": PatchedBatchNormalization,
        "Dense": PatchedDense,
    }


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


def _load_model6_tabular() -> None:
    global MODEL6_TABULAR, MODEL6_SCALER, MODEL6_TABULAR_LOAD_ERROR
    if not ENABLE_MODEL6_TABULAR:
        MODEL6_TABULAR = None
        MODEL6_SCALER = None
        MODEL6_TABULAR_LOAD_ERROR = None
        logger.info("Model 6 (tabular clinical) skipped: ENABLE_MODEL6_TABULAR is false.")
        return
    diag = _model6_tabular_path_diagnostics()
    try:
        if diag["exists"]:
            import tensorflow as tf  # type: ignore

            path = diag["tabular_absolute_path"]
            logger.info("Model 6 tabular: loading H5 via compat sanitizer.")
            MODEL6_TABULAR = _load_h5_model_compat(tf, path)
            MODEL6_SCALER = joblib.load(diag["scaler_absolute_path"])
            MODEL6_TABULAR_LOAD_ERROR = None
            logger.info(
                "Model 6 tabular loaded: model=%s scaler=%s",
                path,
                diag["scaler_absolute_path"],
            )
        else:
            MODEL6_TABULAR = None
            MODEL6_SCALER = None
            MODEL6_TABULAR_LOAD_ERROR = (
                "Model 6 tabular assets not found "
                f"(model={diag['tabular_path']!r}, scaler={diag['scaler_path']!r})."
            )
            logger.warning("%s", MODEL6_TABULAR_LOAD_ERROR)
    except Exception as exc:
        MODEL6_TABULAR = None
        MODEL6_SCALER = None
        MODEL6_TABULAR_LOAD_ERROR = str(exc)
        logger.exception("Failed to load Model 6 tabular pipeline: %s", exc)


def _load_model2_vision_h5() -> None:
    global MODEL2_VISION_H5, MODEL2_VISION_H5_LOAD_ERROR, TENSORFLOW_VERSION
    if not ENABLE_MODEL2_VISION_H5:
        MODEL2_VISION_H5 = None
        MODEL2_VISION_H5_LOAD_ERROR = None
        logger.info("Model 2 ResNet-152V2 vision H5 skipped: ENABLE_MODEL2_VISION_H5 is false.")
        return
    if len(MODEL2_VISION_LABELS) != 3:
        MODEL2_VISION_H5 = None
        MODEL2_VISION_H5_LOAD_ERROR = "MODEL2_VISION_LABELS must contain exactly 3 labels."
        logger.error(
            "Model 6 vision H5 not loaded: need 3 labels, got %s (%r).",
            len(MODEL2_VISION_LABELS),
            MODEL2_VISION_LABELS,
        )
        return
    diag = _model2_vision_h5_path_diagnostics()
    path = diag["absolute_path"]
    if not diag["exists"]:
        MODEL2_VISION_H5 = None
        MODEL2_VISION_H5_LOAD_ERROR = (
            f"Model 6 vision H5 not found at {path!r} "
            f"(MODEL2_VISION_H5_PATH={diag['path']!r})."
        )
        logger.warning("%s", MODEL2_VISION_H5_LOAD_ERROR)
        return
    try:
        import tensorflow as tf  # type: ignore

        if TENSORFLOW_VERSION is None:
            TENSORFLOW_VERSION = str(tf.__version__)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        co = _make_h5_custom_objects(tf)
        try:
            MODEL2_VISION_H5 = tf.keras.models.load_model(
                path, compile=False, custom_objects=co
            )
        except Exception:
            logger.warning(
                "Model 6 vision H5 direct load failed; trying compat path.",
                exc_info=True,
            )
            MODEL2_VISION_H5 = _load_h5_model_compat(tf, path)
        MODEL2_VISION_H5_LOAD_ERROR = None
        logger.info("Model 2 ResNet-152V2 vision H5 loaded: %s", path)
    except Exception as exc:
        MODEL2_VISION_H5 = None
        MODEL2_VISION_H5_LOAD_ERROR = str(exc)
        logger.exception("Model 6 vision H5 load failed.")


def _preprocess_model2_vision_h5_numpy(image_bytes: bytes) -> Any:
    """Model 6 legacy ResNet-152V2: RGB 224×224; scale per MODEL2_PREPROCESS_MODE."""
    import numpy as np  # type: ignore

    w, h = MODEL2_VISION_H5_IMAGE_SIZE
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((w, h), Image.Resampling.BILINEAR)
    img_array = np.asarray(image, dtype=np.float32)
    if MODEL2_PREPROCESS_MODE == "scale_01":
        img_array = img_array / 255.0
    else:
        try:
            from tensorflow.keras.applications.resnet_v2 import preprocess_input  # type: ignore

            img_array = preprocess_input(img_array)
        except Exception:
            logger.warning(
                "ResNetV2 preprocess_input unavailable for Model 6; falling back to /255.0."
            )
            img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def _model2_vision_label_index_map() -> list[dict[str, Any]]:
    """Indexed labels for /health and /debug (Edward verification)."""
    return [
        {"index": i, "label": _label_for_api(MODEL2_VISION_LABELS[i])}
        for i in range(len(MODEL2_VISION_LABELS))
    ]


def _label_for_api(raw: str) -> str:
    return raw.replace("_", " ")


def _model2_vision_decode_scores(row: Any) -> tuple[int, str, float, dict[str, float]]:
    import numpy as np  # type: ignore

    scores = np.asarray(row, dtype=np.float64).flatten()
    n_labels = len(MODEL2_VISION_LABELS)
    if scores.size < n_labels:
        raise ValueError(
            f"Model 6 H5 returned {scores.size} logits; expected {n_labels} "
            f"for labels {MODEL2_VISION_LABELS!r}."
        )
    if scores.size > n_labels:
        logger.warning(
            "Model 6 H5 returned %s logits; using first %s with labels %r.",
            scores.size,
            n_labels,
            MODEL2_VISION_LABELS,
        )
        scores = scores[:n_labels]
    idx = int(np.argmax(scores))
    raw_label = MODEL2_VISION_LABELS[idx]
    label = _label_for_api(raw_label)
    confidence = float(scores[idx])
    probabilities = {
        _label_for_api(MODEL2_VISION_LABELS[i]): round(float(scores[i]), 4)
        for i in range(n_labels)
    }
    return idx, label, confidence, probabilities


def _find_model6_gradcam_layer(model: Any) -> Any:
    """Last conv block for ResNet152V2-style Keras models (post-ReLU target)."""
    import tensorflow as tf  # type: ignore

    for name in ("post_relu", "conv5_block3_out", "conv5_block3_3_conv"):
        try:
            return model.get_layer(name)
        except ValueError:
            continue
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No suitable Conv2D layer found for Model 6 Grad-CAM.")


def _keras_gradcam_model6_to_png_base64(
    model: Any,
    batch: Any,
    class_idx: int,
    rgb_hwc_01: Any,
) -> str | None:
    """TensorFlow Grad-CAM overlay for Model 6 ResNet-152V2 H5."""
    import numpy as np  # type: ignore
    import tensorflow as tf  # type: ignore

    try:
        target_layer = _find_model6_gradcam_layer(model)
        grad_model = tf.keras.models.Model(
            [model.inputs[0]],
            [target_layer.output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(batch, training=False)
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        conv_out = conv_out[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap_np = heatmap.numpy()
        heatmap_resized = np.uint8(255 * heatmap_np)
        heatmap_img = Image.fromarray(heatmap_resized).resize(
            (rgb_hwc_01.shape[1], rgb_hwc_01.shape[0]),
            Image.Resampling.BILINEAR,
        )
        heatmap_arr = np.asarray(heatmap_img, dtype=np.float32) / 255.0
        heatmap_rgb = np.stack([heatmap_arr, heatmap_arr, heatmap_arr], axis=-1)
        overlay = np.clip(heatmap_rgb * 0.4 + rgb_hwc_01 * 0.6, 0, 1)
        out_img = Image.fromarray((overlay * 255).astype(np.uint8))
        buf = BytesIO()
        out_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("Model 6 Grad-CAM failed: %s", exc)
        return None


def _run_model2_vision_h5(image_bytes: bytes) -> dict[str, Any]:
    if MODEL2_VISION_H5 is None:
        raise RuntimeError(
            MODEL2_VISION_H5_LOAD_ERROR or "Model 2 ResNet-152V2 vision H5 is not loaded."
        )
    import numpy as np  # type: ignore

    w, h = MODEL2_VISION_H5_IMAGE_SIZE
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((w, h), Image.Resampling.BILINEAR)
    rgb_01 = np.asarray(image, dtype=np.float32) / 255.0
    batch = _preprocess_model2_vision_h5_numpy(image_bytes)
    out = MODEL2_VISION_H5.predict(batch, verbose=0)
    idx, label, confidence, probabilities = _model2_vision_decode_scores(out[0])
    gradcam_b64 = _keras_gradcam_model6_to_png_base64(
        MODEL2_VISION_H5, batch, idx, rgb_01
    )
    return {
        "prediction": label,
        "confidence": round(float(confidence), 3),
        "probabilities": probabilities,
        "gradcam": gradcam_b64 or "",
    }


def _model6_tabular_probabilities(prediction_prob: float) -> dict[str, float]:
    """P(High COPD Risk) = prediction_prob; P(Low) = complement."""
    p_high = round(float(prediction_prob), 4)
    p_low = round(max(0.0, 1.0 - p_high), 4)
    return {
        MODEL6_TABULAR_LABELS[0]: p_low,
        MODEL6_TABULAR_LABELS[1]: p_high,
    }


def _run_model6_tabular(patient_data: dict[str, Any]) -> dict[str, Any]:
    """Model 2: questionnaire → scaler.pkl → tabular Keras H5."""
    if MODEL6_TABULAR is None or MODEL6_SCALER is None:
        raise RuntimeError(
            MODEL6_TABULAR_LOAD_ERROR or "Model 6 tabular pipeline is not loaded."
        )

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
    scaled_features = MODEL6_SCALER.transform(raw_features)
    prediction_prob = float(MODEL6_TABULAR.predict(scaled_features, verbose=0)[0][0])
    label = (
        MODEL6_TABULAR_LABELS[1]
        if prediction_prob > 0.5
        else MODEL6_TABULAR_LABELS[0]
    )
    conf = round(prediction_prob, 3)
    probs = _model6_tabular_probabilities(prediction_prob)
    return {
        "prediction": label,
        "confidence": conf,
        "status": "success",
        "input_type": "tabular",
        "model_name": "Chronic Lung Risk (COPD)",
        "label": label,
        "probabilities": probs,
    }


def _model6_tabular_skipped_payload(reason: str = "questionnaire_required") -> dict[str, Any]:
    return {
        "status": "skipped",
        "input_type": "tabular",
        "reason": reason,
        "model_name": "Chronic Lung Risk (COPD)",
    }


def _model6_tabular_failed_payload() -> dict[str, Any]:
    return {
        "status": "failed",
        "input_type": "tabular",
        "model_name": "Chronic Lung Risk (COPD)",
    }


def _model2_vision_skipped_payload(reason: str = "vision_disabled") -> dict[str, Any]:
    return {
        "status": "skipped",
        "input_type": "vision",
        "reason": reason,
        "model_name": "ResNet-152V2 (Edward)",
    }


def _model2_vision_failed_payload() -> dict[str, Any]:
    return {
        "prediction": "N/A",
        "confidence": 0.0,
        "status": "failed",
        "model_name": "ResNet-152V2 (Edward)",
        "input_type": "vision",
    }


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


def _vision_model_confidence_01(block: dict[str, Any] | None) -> float | None:
    """Normalize model block confidence to [0, 1] for educator context."""
    if not block or block.get("status") == "failed":
        return None
    if "confidence_score" in block:
        score = float(block["confidence_score"])
        return score if score <= 1.0 else score / 100.0
    if "confidence" in block:
        score = float(block["confidence"])
        return score if score <= 1.0 else score / 100.0
    return None


def _format_vision_model_for_llm(label: str, block: dict[str, Any] | None) -> str:
    if not block:
        return f"{label}: unavailable"
    if block.get("status") == "skipped":
        reason = block.get("reason", "not run")
        return f"{label}: skipped ({reason})"
    if block.get("status") == "failed":
        return f"{label}: unavailable"
    pred = (
        block.get("prediction")
        or block.get("class_name")
        or block.get("label")
        or "N/A"
    )
    conf = _vision_model_confidence_01(block)
    if conf is not None:
        return f"{label}: {pred} (confidence {conf:.0%})"
    return f"{label}: {pred}"


def _build_educator_ml_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Structured ML context for Gemini: all five vision models + COPD screening."""
    predictions = payload.get("predictions") or {}
    primary = (
        max(predictions, key=predictions.get)
        if predictions
        else "Normal"
    )
    return {
        "Primary Finding (ensemble)": primary,
        "Model 1 (ResNet-50)": _format_vision_model_for_llm(
            "ResNet-50", payload.get("model1")
        ),
        "Model 2 (ResNet-152V2)": _format_vision_model_for_llm(
            "ResNet-152V2", payload.get("model2")
        ),
        "Model 3 (DenseNet-121 PyTorch)": _format_vision_model_for_llm(
            "DenseNet-121", payload.get("model3")
        ),
        "Model 4 (Swin-T)": _format_vision_model_for_llm(
            "Swin-T", payload.get("model4_swint")
        ),
        "Model 5 (DenseNet-121 H5)": _format_vision_model_for_llm(
            "DenseNet-121 H5", payload.get("model5_densenet")
        ),
        "Model 6 (Clinical COPD tabular)": _format_vision_model_for_llm(
            "Clinical COPD", payload.get("model6")
        ),
    }


def _format_patient_profile_for_educator(patient_data: dict[str, Any]) -> str:
    smoking_status = patient_data.get(
        "smoking_status", patient_data.get("smoking", "Unknown")
    )
    fever_val = patient_data.get("fever")
    if fever_val is None:
        fever_line = "Unknown"
    else:
        fever_line = "Yes" if fever_val else "No"
    breathing = patient_data.get("breathing_difficulty", "Unknown")
    cough_days = patient_data.get("cough_duration_days", "Unknown")
    return (
        f"- Age: {patient_data.get('age', 'Unknown')}\n"
        f"- Fever: {fever_line}\n"
        f"- Cough Duration: {cough_days} days\n"
        f"- Smoking Status: {smoking_status}\n"
        f"- Breathing Difficulty: {breathing}"
    )


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

    patient_profile = _format_patient_profile_for_educator(patient_data)
    prompt = f"""
You are a highly empathetic, professional medical AI educator for a medical education platform.
Your goal is to explain chest X-ray AI findings in clear, academically rigorous language suitable for informed patients and learners.
You must NEVER definitively diagnose. Use phrasing such as "The AI detected patterns consistent with..." and note that only a licensed clinician can interpret imaging in full clinical context.

CRITICAL: You are an educational tool, not a doctor. All strategies and developments must be framed as "standard medical practice" or "general clinical knowledge." Never use prescriptive language like "You should do X" or "Your best strategy is Y." Always speak objectively about the condition itself—not direct instructions to this patient.

Patient Profile (clinical intake questionnaire):
{patient_profile}

AI Findings (five vision models + COPD screening):
{ml_results}

Respond in exactly two sections using Markdown (no other sections):

### 🩺 Clinical Observation
Write 2-3 sentences combining questionnaire symptoms with the AI findings for a lay reader. Educational tone only; no diagnosis.

### 💡 Clinical Context & Management Strategy
Write exactly three paragraphs (plain prose, not bullet lists). Personalize using the questionnaire data above and the most prominent condition flagged by the vision models:

**Latest Developments:** A brief, factual summary of modern medical understanding or recent advancements regarding that prominent condition (e.g., standard approaches to viral pneumonia, lung opacities, or other relevant patterns). Frame as general clinical knowledge—not live breaking news unless widely established.

**Standard Strategy:** Explain the typical clinical pathway or standard investigative procedures clinicians usually follow when presented with these X-ray patterns together with the questionnaire symptoms. Describe what is commonly done in practice; do not tell the reader what they personally must do.

**Key Concerns & Limitations:** Highlight specific risk factors and monitoring themes clinicians typically weigh in these scenarios, dynamically referencing the patient's age, symptom duration, smoking history, and breathing difficulty from the questionnaire where relevant. Emphasize limits of AI screening and the need for professional interpretation.
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


def _resolve_gemini_key_json(value: str | None) -> tuple[str | None, str]:
    if value is not None and str(value).strip():
        return str(value).strip(), "json_gemini_api_key"
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.getenv(env_name, "").strip()
        if v:
            return v, f"env_{env_name.lower()}"
    return None, "none"


def _locale_instruction(locale: str | None) -> str:
    loc = (locale or "en").strip()
    if loc in ("zh-Hant", "zh-TW", "zh-HK"):
        return "Write all titles and text in Traditional Chinese (繁體中文)."
    if loc in ("zh-Hans", "zh-CN", "zh"):
        return "Write all titles and text in Simplified Chinese (简体中文)."
    return "Write all titles and text in English."


def _parse_educational_insights_json(raw: str) -> list[dict[str, str]] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    out: list[dict[str, str]] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        body = str(item.get("text") or item.get("body") or "").strip()
        trigger = str(item.get("finding_trigger") or item.get("finding") or "General").strip()
        category = str(item.get("category") or "overview").strip() or "overview"
        if not body:
            continue
        if not title:
            title = "Educational insight"
        out.append(
            {
                "id": str(item.get("id") or f"i{i + 1}"),
                "title": title,
                "text": body,
                "finding_trigger": trigger,
                "category": category,
            }
        )
    return out if out else None


def _fallback_educational_insights(
    findings: list[str], locale: str | None = None
) -> list[dict[str, str]]:
    pack = EDUCATIONAL_INSIGHTS_FALLBACK_BY_LOCALE[_resolve_insights_fallback_locale(locale)]
    rows: list[dict[str, str]] = []
    idx = 1
    seen: set[str] = set()
    for finding in findings:
        key = _normalize_insight_finding_key(finding)
        if not key or key in seen:
            continue
        seen.add(key)
        templates = pack.get(key)
        if not templates:
            continue
        for tpl in templates:
            rows.append(
                {
                    "id": f"i{idx}",
                    "title": tpl["title"],
                    "text": tpl["text"],
                    "finding_trigger": key,
                    "category": tpl.get("category", "overview"),
                }
            )
            idx += 1
    return rows


def _generate_educational_insights_llm(
    findings: list[str],
    api_key: str,
    locale: str | None,
    scan_summary: str | None,
) -> dict[str, Any]:
    normalized = [_normalize_insight_finding_key(f) for f in findings]
    normalized = [f for f in normalized if f]
    locale_line = _locale_instruction(locale)
    summary_block = (
        f"\nScan / model context (educational only):\n{scan_summary}\n"
        if scan_summary and scan_summary.strip()
        else ""
    )
    prompt = f"""
You are a medical health educator helping a patient understand their chest X-ray AI screening output.
{locale_line}

CRITICAL RULES:
- Do NOT diagnose. Use phrasing like "patterns consistent with" or "areas sometimes associated with".
- Do NOT prescribe. Describe general treatment approaches and public-health context only.
- For "latest news": summarize widely accepted clinical themes and guideline directions (vaccines, antibiotics, antivirals, follow-up imaging). State clearly this is general background—not a live news feed—and advise consulting their clinician for personal care.
- Never claim real-time access to breaking news unless you are certain.

High-attention AI finding labels from this scan:
{json.dumps(normalized)}
{summary_block}

Return ONLY a JSON array (no markdown prose outside the array) with 3 to 6 objects. Each object must have:
- "id": string (e.g. "i1")
- "title": short heading
- "text": 2-4 sentences of patient-friendly educational information (treatment options, follow-up tests, or research/guideline context tied to the findings)
- "finding_trigger": one of the finding labels above, or "General"
- "category": one of "overview", "treatment", "research"

Cover at least: (1) what the finding means on X-ray, (2) typical treatment/management themes, (3) recent evidence or guideline themes where relevant.
"""
    models_to_try = _gemini_educator_models_to_try(api_key)
    if not models_to_try:
        return {"status": "failed", "insights": None, "error_code": "gemini_not_found"}

    last_code: str | None = None
    try:
        genai.configure(api_key=api_key)
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                try:
                    raw = response.text
                except (ValueError, AttributeError):
                    last_code = "gemini_blocked_or_empty"
                    continue
                text = (raw or "").strip() if isinstance(raw, str) else ""
                if not text:
                    last_code = "gemini_blocked_or_empty"
                    continue
                parsed = _parse_educational_insights_json(text)
                if parsed:
                    return {"status": "success", "insights": parsed, "model": model_name}
                last_code = "gemini_invalid_json"
            except Exception as exc:
                last_code = _classify_gemini_exception(exc)
                if not _gemini_error_should_try_next_model(last_code):
                    break
                continue
    except Exception as exc:
        last_code = _classify_gemini_exception(exc)

    return {"status": "failed", "insights": None, "error_code": last_code or "gemini_unknown"}


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
    model2_vision_payload: dict[str, Any] | None = None,
    model2_vision_inference_ok: bool = False,
    model6_tabular_payload: dict[str, Any] | None = None,
    model6_tabular_inference_ok: bool = False,
    timing_model1_ms: float = 18.0,
    timing_model2_ms: float = 22.0,
    timing_model6_ms: float = 0.0,
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

    if model6_tabular_inference_ok and model6_tabular_payload is not None:
        model6_label = str(
            model6_tabular_payload.get("prediction", model6_tabular_payload.get("label", ""))
        )
        model6_confidence = round(
            float(model6_tabular_payload.get("confidence", 0.0)), 3
        )
    else:
        model6_label = ""
        model6_confidence = 0.0

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

        imaging_label = model1_label if model1_pytorch_inference_ok else top_prediction
        summary_lines = [
            f"Educational analysis suggests {imaging_label} with top imaging finding "
            f"{top_prediction} ({top_confidence:.2f})."
        ]
        if model6_tabular_inference_ok and model6_label:
            summary_lines.append(f"Clinical intake screening: {model6_label}.")
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
    t6 = round(float(timing_model6_ms), 2)
    timing_ms = {
        "model1": t1,
        "model2": t2,
        "model3": t_dn,
        "model4": model4_timing,
        "model6": t6,
        "total": round(t1 + t2 + t6 + t_dn + model4_timing, 2),
    }

    any_neural_ok = (
        model1_pytorch_inference_ok or densenet_neural_ok
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
    if model2_vision_inference_ok:
        model2_status = "ok"
    elif not ENABLE_MODEL2_VISION_H5:
        model2_status = "skipped"
    elif MODEL2_VISION_H5 is None:
        model2_status = "load_failed"
    else:
        model2_status = "skipped"
    if model6_tabular_inference_ok:
        model6_status = "ok"
    elif not ENABLE_MODEL6_TABULAR:
        model6_status = "skipped"
    elif MODEL6_TABULAR is None or MODEL6_SCALER is None:
        model6_status = "load_failed"
    else:
        model6_status = "skipped"
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
        "model2_result": "model" if model2_vision_inference_ok else "rules",
        "model6_result": "model" if model6_tabular_inference_ok else "rules",
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
        "model2": model2_vision_payload
        if model2_vision_payload is not None
        else _model2_vision_failed_payload(),
        "model6": model6_tabular_payload
        if model6_tabular_payload is not None
        else _model6_tabular_failed_payload(),
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
                "source": "model" if model2_vision_inference_ok else "rules",
                "status": model2_status,
                "model_id": "resnet152v2-edward"
                if model2_vision_inference_ok
                else "resnet152v2-unavailable",
                "model_version": "v1" if model2_vision_inference_ok else "n/a",
            },
            "model6": {
                "source": "model" if model6_tabular_inference_ok else "rules",
                "status": model6_status,
                "model_id": "model6-tabular-copd"
                if model6_tabular_inference_ok
                else "model6-tabular-unavailable",
                "model_version": "pilot" if model6_tabular_inference_ok else "n/a",
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


async def _timed_thread_call(
    fn: Any, *args: Any
) -> tuple[Any, float, BaseException | None]:
    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(fn, *args)
        return result, (time.perf_counter() - start) * 1000.0, None
    except BaseException as exc:
        return None, (time.perf_counter() - start) * 1000.0, exc


async def _analyze_run_vision_models(image_bytes: bytes) -> dict[str, Any]:
    """Run all enabled vision models concurrently (thread pool)."""
    pending: dict[str, Any] = {}

    if ENABLE_MODEL1 and MODEL1_PT is not None:
        pending["model1"] = _timed_thread_call(_run_pytorch_model1_full, image_bytes)
    if ENABLE_MODEL2_VISION_H5:
        _warn_model2_vision_file_missing_once("analyze")
    if ENABLE_MODEL2_VISION_H5 and MODEL2_VISION_H5 is not None:
        pending["model2"] = _timed_thread_call(_run_model2_vision_h5, image_bytes)
    if ENABLE_MODEL4_SWINT and MODEL4_SWINT is not None:
        pending["model4"] = _timed_thread_call(_run_swint_model4, image_bytes)
    if ENABLE_MODEL5_DENSENET and MODEL5_DENSENET_H5 is not None:
        pending["model5"] = _timed_thread_call(_run_densenet_model5, image_bytes)
    if ENABLE_DENSENET121 and MODEL_DENSENET121 is not None:
        pending["model3"] = _timed_thread_call(
            _densenet121_predict_and_cam, image_bytes
        )

    if not pending:
        return {}

    keys = list(pending.keys())
    gathered = await asyncio.gather(*pending.values())
    return dict(zip(keys, gathered, strict=True))


AnalyzeJobStatus = Literal["queued", "processing", "complete", "failed"]
ANALYZE_JOBS: dict[str, dict[str, Any]] = {}
MAX_ANALYZE_JOBS = 200
ANALYZE_JOB_TTL_SECONDS = 3600


def _prune_analyze_jobs() -> None:
    if len(ANALYZE_JOBS) <= MAX_ANALYZE_JOBS:
        return
    now = time.time()
    expired = [
        job_id
        for job_id, job in ANALYZE_JOBS.items()
        if now - float(job.get("created_at", now)) > ANALYZE_JOB_TTL_SECONDS
    ]
    for job_id in expired:
        ANALYZE_JOBS.pop(job_id, None)
    if len(ANALYZE_JOBS) <= MAX_ANALYZE_JOBS:
        return
    oldest = sorted(
        ANALYZE_JOBS.items(),
        key=lambda item: float(item[1].get("created_at", 0.0)),
    )
    for job_id, _ in oldest[: max(0, len(ANALYZE_JOBS) - MAX_ANALYZE_JOBS)]:
        ANALYZE_JOBS.pop(job_id, None)


async def _run_analyze_job(
    job_id: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    questionnaire: str | None,
    gemini_api_key: str | None,
) -> None:
    job = ANALYZE_JOBS.get(job_id)
    if job is None:
        return
    job["status"] = "processing"
    try:
        upload = UploadFile(
            file=BytesIO(image_bytes),
            filename=filename,
            size=len(image_bytes),
            headers={"content-type": content_type},
        )
        response = await _analyze_internal(
            image=upload,
            questionnaire=questionnaire,
            x_api_key=None,
            gemini_api_key=gemini_api_key,
            skip_auth=True,
        )
        raw_body = getattr(response, "body", b"") or b""
        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            body = {}
        if response.status_code == 200 and isinstance(body, dict):
            job["status"] = "complete"
            job["result"] = body
            job["error"] = None
            job["error_code"] = None
            return
        err_msg = (
            body.get("error") if isinstance(body, dict) else None
        ) or f"Analyze failed ({response.status_code})."
        job["status"] = "failed"
        job["result"] = None
        job["error"] = str(err_msg)[:500]
        job["error_code"] = (
            body.get("error_code") if isinstance(body, dict) else "backend_unavailable"
        )
    except Exception as exc:
        logger.exception("Analyze job %s failed", job_id)
        job["status"] = "failed"
        job["result"] = None
        job["error"] = str(exc)[:500]
        job["error_code"] = "internal_error"


async def _analyze_internal(
    image: UploadFile,
    questionnaire: str | None,
    x_api_key: str | None,
    gemini_api_key: str | None,
    *,
    skip_auth: bool = False,
) -> JSONResponse:
    try:
        if not skip_auth:
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
        patient_data: dict[str, Any] | None = None
        if isinstance(questionnaire_data, dict):
            maybe_patient_data = questionnaire_data.get("patient_data")
            if isinstance(maybe_patient_data, dict):
                patient_data = maybe_patient_data

        model1_override: tuple[str, float] | None = None
        model1_pytorch_inference_ok = False
        model1_gradcam_b64: str | None = None
        model1_probabilities: dict[str, float] | None = None
        m1_label = "Normal"
        m1_conf_r = 0.0
        timing_model1_ms = 0.0

        model6_tabular_result: dict[str, Any] = _model6_tabular_skipped_payload()
        model6_tabular_inference_ok = False
        timing_model6_ms = 0.0

        model2_vision_result: dict[str, Any] = {
            "prediction": "N/A",
            "confidence": 0.0,
            "status": "failed",
            "model_name": "ResNet-152V2 (Edward)",
            "input_type": "vision",
        }
        model2_vision_inference_ok = False
        model2_vision_label = "Normal"
        model2_vision_conf = 0.0
        timing_model2_ms = 0.0

        model4_swint_result: dict[str, Any] = {
            "prediction": "N/A",
            "confidence": 0.0,
            "status": "failed",
        }
        model4_swint_label = "Normal"
        model4_swint_conf = 0.0
        model4_swint_inference_ok = False

        model5_densenet_result: dict[str, Any] = {
            "prediction": "N/A",
            "confidence": 0.0,
            "status": "failed",
        }
        model5_label = ""
        model5_conf = 0.0
        model5_inference_ok = False

        densenet_payload: dict[str, Any] = _densenet_analyze_error_payload()
        timing_densenet_ms = 0.0

        vision_results = await _analyze_run_vision_models(image_bytes)

        if "model1" in vision_results:
            m1_data, timing_model1_ms, m1_err = vision_results["model1"]
            if m1_err is None and m1_data is not None:
                m1_label, m1_conf, m1_probs, m1_gc = m1_data
                m1_conf_r = round(float(m1_conf), 3)
                model1_override = (m1_label, m1_conf_r)
                model1_gradcam_b64 = m1_gc
                model1_probabilities = m1_probs
                model1_pytorch_inference_ok = True
            else:
                logger.warning("ML Model 1 PyTorch inference failed: %s", m1_err)

        if isinstance(patient_data, dict) and ENABLE_MODEL6_TABULAR:
            if MODEL6_TABULAR is not None and MODEL6_SCALER is not None:
                t_m6 = time.perf_counter()
                try:
                    model6_tabular_result = await asyncio.to_thread(
                        _run_model6_tabular, patient_data
                    )
                    model6_tabular_inference_ok = True
                except Exception as exc:
                    logger.error("Model 6 tabular inference failed: %s", exc)
                    model6_tabular_result = _model6_tabular_failed_payload()
                timing_model6_ms = (time.perf_counter() - t_m6) * 1000.0
            else:
                model6_tabular_result = _model6_tabular_failed_payload()

        if "model2" in vision_results:
            m2_data, timing_model2_ms, m2_err = vision_results["model2"]
            if m2_err is None and isinstance(m2_data, dict):
                model2_vision_label = str(m2_data.get("prediction", "Normal"))
                model2_vision_conf = round(float(m2_data.get("confidence", 0.0)), 3)
                model2_vision_result = {
                    "prediction": model2_vision_label,
                    "confidence": model2_vision_conf,
                    "status": "success",
                    "probabilities": m2_data.get("probabilities") or {},
                    "gradcam": m2_data.get("gradcam") or "",
                    "model_name": "ResNet-152V2 (Edward)",
                    "input_type": "vision",
                }
                model2_vision_inference_ok = True
            else:
                logger.warning("Model 2 ResNet-152V2 vision inference failed: %s", m2_err)

        if "model4" in vision_results:
            m4_data, _m4_ms, m4_err = vision_results["model4"]
            if m4_err is None and m4_data is not None:
                model4_swint_label, model4_swint_conf, model4_swint_probs = m4_data
                model4_swint_conf = round(float(model4_swint_conf), 3)
                model4_swint_result = {
                    "prediction": model4_swint_label,
                    "confidence": model4_swint_conf,
                    "status": "success",
                    "probabilities": model4_swint_probs,
                    "model_name": "Swin-T",
                }
                model4_swint_inference_ok = True
            else:
                logger.error("Model 4 (Swin-T) inference failed: %s", m4_err)

        if "model5" in vision_results:
            m5_data, _m5_ms, m5_err = vision_results["model5"]
            if m5_err is None and isinstance(m5_data, dict):
                model5_densenet_result = m5_data
                model5_label = str(model5_densenet_result.get("prediction", ""))
                model5_conf = float(model5_densenet_result.get("confidence", 0.0))
                model5_inference_ok = (
                    model5_densenet_result.get("status") == "success"
                )
            else:
                logger.error("Model 5 (DenseNet-121 H5) inference failed: %s", m5_err)

        if "model3" in vision_results:
            m3_data, timing_densenet_ms, m3_err = vision_results["model3"]
            if m3_err is None and isinstance(m3_data, dict):
                densenet_payload = {**m3_data, "model_name": "DenseNet-121"}
            else:
                logger.warning(
                    "DenseNet-121 in analyze pipeline failed (model1/model2 unaffected): %s",
                    m3_err,
                )
                densenet_payload = _densenet_analyze_error_payload(
                    (str(m3_err) or "Model not available")[:500]
                )

        densenet_neural_ok = (
            "prediction" in densenet_payload and "error" not in densenet_payload
        )
        predictions: dict[str, float] = {}
        if model1_pytorch_inference_ok and m1_label != "Normal":
            base = "Pneumonia" if "Pneumonia" in m1_label else m1_label
            predictions[base] = max(predictions.get(base, 0.0), m1_conf_r)
        if model2_vision_inference_ok and model2_vision_label != "Normal":
            m6_base = (
                "Pneumonia"
                if "Pneumonia" in model2_vision_label
                else model2_vision_label
            )
            predictions[m6_base] = max(
                predictions.get(m6_base, 0.0), model2_vision_conf
            )
        if model4_swint_inference_ok and model4_swint_label != "Normal":
            m4_base = (
                "Pneumonia"
                if "Pneumonia" in model4_swint_label
                else model4_swint_label
            )
            predictions[m4_base] = max(
                predictions.get(m4_base, 0.0), model4_swint_conf
            )
        if (
            model5_inference_ok
            and model5_label
            and model5_label != "Normal"
            and model5_conf >= 0.5
        ):
            predictions[model5_label] = max(
                predictions.get(model5_label, 0.0), model5_conf
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
            model2_vision_payload=model2_vision_result,
            model2_vision_inference_ok=model2_vision_inference_ok,
            model6_tabular_payload=model6_tabular_result,
            model6_tabular_inference_ok=model6_tabular_inference_ok,
            timing_model1_ms=timing_model1_ms,
            timing_model2_ms=timing_model2_ms,
            timing_model6_ms=timing_model6_ms,
            densenet_payload=densenet_payload,
            timing_densenet_ms=timing_densenet_ms,
        )
        payload["model4_swint"] = model4_swint_result
        payload["model5_densenet"] = model5_densenet_result

        llm_patient_data: dict[str, Any] = dict(patient_data or {})
        if not llm_patient_data and isinstance(questionnaire_data, dict):
            q_patient = questionnaire_data.get("patient_data")
            if isinstance(q_patient, dict):
                llm_patient_data = q_patient

        ml_summary = _build_educator_ml_summary(payload)
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
            llm_patient_data,
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
    _load_model1_pytorch()
    _load_model6_tabular()
    _load_model2_vision_h5()
    _load_densenet121()
    _load_swint_model4()
    _load_densenet_model5()
    logger.info("Resolved MODEL5_DENSENET_LABELS order: %s", MODEL5_DENSENET_LABELS)
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
    m2_diag = _model6_tabular_path_diagnostics()
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
            "model6_tabular": {
                "enabled": ENABLE_MODEL6_TABULAR,
                "loaded": MODEL6_TABULAR is not None and MODEL6_SCALER is not None,
                "error": MODEL6_TABULAR_LOAD_ERROR,
                "labels": list(MODEL6_TABULAR_LABELS),
                **m2_diag,
            },
            "model2": {
                "enabled": ENABLE_MODEL2_VISION_H5,
                "loaded": MODEL2_VISION_H5 is not None,
                **_model2_vision_h5_path_diagnostics(),
                "error": MODEL2_VISION_H5_LOAD_ERROR,
                "labels": list(MODEL2_VISION_LABELS),
                "label_indices": _model2_vision_label_index_map(),
                "h5_model2_labels_default": list(H5_MODEL2_LABELS),
                "preprocess_mode": MODEL2_PREPROCESS_MODE,
                "image_size": list(MODEL2_VISION_H5_IMAGE_SIZE),
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
            "model5_densenet_h5": {
                "enabled": ENABLE_MODEL5_DENSENET,
                "loaded": MODEL5_DENSENET_H5 is not None,
                **_model5_densenet_path_diagnostics(),
                "error": MODEL5_DENSENET_LOAD_ERROR,
                "labels": list(MODEL5_DENSENET_LABELS),
            },
        },
    }


@app.get("/debug")
async def debug_model_status() -> dict[str, Any]:
    """Diagnostics for ML models and response provenance (no secrets)."""
    pth_diag = _model1_pt_path_diagnostics()
    m2_diag = _model6_tabular_path_diagnostics()
    dn_diag = _densenet121_path_diagnostics()
    tf_ver = TENSORFLOW_VERSION or _tensorflow_version_probe()
    pt_ver = PYTORCH_VERSION or _pytorch_version_probe()
    m1_ready = ENABLE_MODEL1 and MODEL1_PT is not None
    m2_ready = (
        ENABLE_MODEL6_TABULAR and MODEL6_TABULAR is not None and MODEL6_SCALER is not None
    )
    m3_ready = ENABLE_DENSENET121 and MODEL_DENSENET121 is not None
    m4_ready = ENABLE_MODEL4_SWINT and MODEL4_SWINT is not None
    m5_ready = ENABLE_MODEL5_DENSENET and MODEL5_DENSENET_H5 is not None
    m4_diag = _model4_swint_path_diagnostics()
    m5_diag = _model5_densenet_path_diagnostics()
    hybrid_preview = m1_ready or m2_ready or m3_ready or m4_ready or m5_ready
    return {
        "model_registry": MODEL_REGISTRY,
        "model1_pt": {
            "enabled": ENABLE_MODEL1,
            "loaded": MODEL1_PT is not None,
            "load_error": MODEL1_PT_LOAD_ERROR,
            **pth_diag,
            "labels": list(MODEL1_LABELS),
        },
        "model6_tabular": {
            "enabled": ENABLE_MODEL6_TABULAR,
            "loaded": MODEL6_TABULAR is not None and MODEL6_SCALER is not None,
            "load_error": MODEL6_TABULAR_LOAD_ERROR,
            **m2_diag,
            "labels": list(MODEL6_TABULAR_LABELS),
        },
        "model2": {
            "enabled": ENABLE_MODEL2_VISION_H5,
            "loaded": MODEL2_VISION_H5 is not None,
            "load_error": MODEL2_VISION_H5_LOAD_ERROR,
            **_model2_vision_h5_path_diagnostics(),
            "labels": list(MODEL2_VISION_LABELS),
            "label_indices": _model2_vision_label_index_map(),
            "h5_model2_labels_default": list(H5_MODEL2_LABELS),
            "preprocess_mode": MODEL2_PREPROCESS_MODE,
            "image_size": list(MODEL2_VISION_H5_IMAGE_SIZE),
            "gradcam": "keras_gradient_tape",
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
        "model5_densenet_h5": {
            "enabled": ENABLE_MODEL5_DENSENET,
            "loaded": MODEL5_DENSENET_H5 is not None,
            "load_error": MODEL5_DENSENET_LOAD_ERROR,
            **m5_diag,
            "labels": list(MODEL5_DENSENET_LABELS),
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
    """Return AI educational insights (treatment/research context) for high-attention findings."""
    auth_error = _validate_api_key(x_api_key)
    if auth_error is not None:
        return auth_error

    findings = req.high_attention_findings or []
    gemini_key, _key_src = _resolve_gemini_key_json(req.gemini_api_key)
    source = "rules"
    educational_insights: list[dict[str, str]] = []

    if gemini_key and findings:
        llm_result = await asyncio.to_thread(
            _generate_educational_insights_llm,
            findings,
            gemini_key,
            req.locale,
            req.scan_summary,
        )
        if llm_result.get("status") == "success" and llm_result.get("insights"):
            educational_insights = llm_result["insights"]
            source = "llm"

    if not educational_insights:
        educational_insights = _fallback_educational_insights(findings, req.locale)

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "source": source,
            "educational_insights": educational_insights,
        },
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


@app.post("/api/v1/analyze/jobs")
async def create_analyze_job(
    image: UploadFile = File(...),
    questionnaire: str | None = Form(None),
    gemini_api_key: str | None = Form(None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    """Accept image and run analyze in background; poll GET /api/v1/analyze/jobs/{job_id}."""
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

    _prune_analyze_jobs()
    job_id = str(uuid.uuid4())
    ANALYZE_JOBS[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "result": None,
        "error": None,
        "error_code": None,
    }
    asyncio.create_task(
        _run_analyze_job(
            job_id,
            image_bytes,
            image.filename,
            content_type,
            questionnaire,
            gemini_api_key,
        )
    )
    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "job_id": job_id,
            "status": "queued",
        },
    )


@app.get("/api/v1/analyze/jobs/{job_id}")
async def get_analyze_job(
    job_id: str = Path(..., min_length=8, max_length=64),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    auth_error = _validate_api_key(x_api_key)
    if auth_error is not None:
        return auth_error

    job = ANALYZE_JOBS.get(job_id)
    if job is None:
        return _error_response("Analyze job not found.", 404)

    status: AnalyzeJobStatus = job.get("status", "queued")
    if status == "complete" and isinstance(job.get("result"), dict):
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "status": "complete",
                "result": job["result"],
            },
        )
    if status == "failed":
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "status": "failed",
                "error": job.get("error") or "Analyze job failed.",
                "error_code": job.get("error_code") or "backend_unavailable",
                "retryable": True,
            },
        )
    return JSONResponse(
        status_code=200,
        content={"job_id": job_id, "status": status},
    )


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
