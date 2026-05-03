"""
API contract tests for LungLens FastAPI backend.

Works with real ML weights loaded or mock/rule fallback (no env assumptions).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGE_PATH = PROJECT_ROOT / "testfile" / "Lung Xray.jpeg"

# API returns humanized Model 2 labels (underscores → spaces); mock path may use "Other".
MODEL2_LABELS_API = frozenset(
    {"Normal", "Lung Opacity", "Viral Pneumonia", "Other"}
)

# Neural Model 1 uses 3-class checkpoint labels; mock scaffold uses Pneumonia | Normal.
MODEL1_LABELS_ALLOWED = frozenset(
    {"Normal", "Pneumonia-Bacteria", "Pneumonia-Virus", "Pneumonia"}
)


def _assert_analyze_success_payload(data: dict, *, context: str) -> None:
    assert data.get("success") is True, f"{context}: expected success true, got {data!r}"


def _collect_key_paths(obj: object, prefix: str = "") -> set[str]:
    """Dot-paths for dict keys (values ignored); lists are not descended."""
    paths: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            paths.add(p)
            paths |= _collect_key_paths(v, p)
    return paths


def _all_dict_keys(obj: object) -> set[str]:
    """Every dict key at any depth (values ignored)."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        keys |= set(obj.keys())
        for v in obj.values():
            keys |= _all_dict_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _all_dict_keys(item)
    return keys


@pytest.fixture
def test_image_bytes() -> bytes:
    assert TEST_IMAGE_PATH.is_file(), f"Missing test image at {TEST_IMAGE_PATH}"
    return TEST_IMAGE_PATH.read_bytes()


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
async def analyze_json(client: AsyncClient, test_image_bytes: bytes) -> dict:
    response = await client.post(
        "/api/v1/analyze",
        files={
            "image": (
                "Lung Xray.jpeg",
                test_image_bytes,
                "image/jpeg",
            )
        },
    )
    assert response.status_code == 200, (
        f"analyze failed: status={response.status_code} body={response.text[:500]}"
    )
    data = response.json()
    _assert_analyze_success_payload(data, context="POST /api/v1/analyze")
    return data


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_root_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/")
        assert response.status_code == 200, f"GET / expected 200, got {response.status_code}"
        body = response.json()
        assert "status" in body, f"root JSON should include 'status', got keys {body.keys()}"

    @pytest.mark.asyncio
    async def test_healthz_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/healthz")
        assert response.status_code == 200, (
            f"GET /healthz expected 200, got {response.status_code}"
        )

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200, (
            f"GET /health expected 200, got {response.status_code}"
        )
        body = response.json()
        assert "models" in body, f"/health should include 'models', got keys {body.keys()}"
        models = body["models"]
        assert "model1_pt" in models, (
            f"/health models should include 'model1_pt', got {models.keys()}"
        )
        assert "model2_h5" in models, (
            f"/health models should include 'model2_h5', got {models.keys()}"
        )
        for alias in (
            "model1_resnet50",
            "model2_resnet152v2",
            "model3_densenet121",
        ):
            assert alias in models, (
                f"/health models should include alias {alias!r}, got {models.keys()}"
            )
            slot = models[alias]
            assert "enabled" in slot and "loaded" in slot, (
                f"{alias} should have enabled+loaded, got {slot!r}"
            )

    @pytest.mark.asyncio
    async def test_debug_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/debug")
        assert response.status_code == 200, (
            f"GET /debug expected 200, got {response.status_code}"
        )
        body = response.json()
        # /debug exposes model blocks at top level (not under "models").
        assert "model1_pt" in body, (
            f"/debug should include 'model1_pt', got keys {body.keys()}"
        )
        assert "model2_h5" in body, (
            f"/debug should include 'model2_h5', got keys {body.keys()}"
        )
        assert "analyze_provenance_sources" in body, (
            f"/debug should include 'analyze_provenance_sources', got keys {body.keys()}"
        )


class TestResponseShape:
    @pytest.mark.asyncio
    async def test_analyze_returns_model1_key(self, analyze_json: dict) -> None:
        assert "model1" in analyze_json, (
            "response must include 'model1' (not legacy stage1); "
            f"keys={sorted(analyze_json.keys())}"
        )

    @pytest.mark.asyncio
    async def test_analyze_returns_model2_key(self, analyze_json: dict) -> None:
        assert "model2" in analyze_json, (
            "response must include 'model2' (not legacy stage2); "
            f"keys={sorted(analyze_json.keys())}"
        )

    @pytest.mark.asyncio
    async def test_analyze_no_stage_keys(self, analyze_json: dict) -> None:
        forbidden = {"stage1", "stage2", "stage3", "stage4", "report"}
        found = forbidden & _all_dict_keys(analyze_json)
        assert not found, (
            "response must not use legacy stage/report dict keys; "
            f"found {sorted(found)}"
        )

    @pytest.mark.asyncio
    async def test_analyze_has_clinical_risk_and_model3_densenet(
        self, analyze_json: dict
    ) -> None:
        assert "clinical_risk" in analyze_json, (
            "response must include clinical_risk (questionnaire severity block); "
            f"keys={sorted(analyze_json.keys())}"
        )
        m3 = analyze_json.get("model3")
        assert isinstance(m3, dict), f"model3 must be a dict, got {type(m3)}"
        assert m3.get("model_name") == "DenseNet-121", (
            f"model3.model_name must be DenseNet-121, got {m3!r}"
        )
        if "prediction" in m3:
            assert "probabilities" in m3 and "gradcam" in m3, (
                f"successful model3 must include probabilities+gradcam, got keys {m3.keys()}"
            )
        else:
            assert "error" in m3, f"failed model3 must include error, got {m3!r}"

    @pytest.mark.asyncio
    async def test_analyze_has_gate(self, analyze_json: dict) -> None:
        assert "gate" in analyze_json, "response must include 'gate'"
        gate = analyze_json["gate"]
        assert "route" in gate, f"gate must include 'route', got {gate}"
        assert "reason" in gate, f"gate must include 'reason', got {gate}"

    @pytest.mark.asyncio
    async def test_analyze_has_timing(self, analyze_json: dict) -> None:
        assert "timing_ms" in analyze_json, "response must include 'timing_ms'"
        t = analyze_json["timing_ms"]
        for key in ("model1", "model2", "model3", "model4", "total"):
            assert key in t, f"timing_ms must include {key!r}, got {t.keys()}"

    @pytest.mark.asyncio
    async def test_analyze_has_provenance(self, analyze_json: dict) -> None:
        assert "provenance" in analyze_json, "response must include 'provenance'"
        prov = analyze_json["provenance"]
        for key in ("model1", "model2", "model1_result", "model2_result"):
            assert key in prov, (
                f"provenance must include {key!r}, got {prov.keys()}"
            )


class TestModel1:
    @pytest.mark.asyncio
    async def test_model1_has_label(self, analyze_json: dict) -> None:
        label = analyze_json["model1"]["label"]
        assert label in MODEL1_LABELS_ALLOWED, (
            f"model1.label must be a known scaffold or 3-class label, got {label!r}; "
            f"allowed={sorted(MODEL1_LABELS_ALLOWED)}"
        )

    @pytest.mark.asyncio
    async def test_model1_has_confidence(self, analyze_json: dict) -> None:
        conf = analyze_json["model1"]["confidence"]
        assert isinstance(conf, (int, float)), (
            f"model1.confidence must be numeric, got {type(conf)}"
        )
        assert 0.0 <= float(conf) <= 1.0, (
            f"model1.confidence must be in [0,1], got {conf}"
        )

    @pytest.mark.asyncio
    async def test_model1_has_model_name_when_neural(
        self, analyze_json: dict
    ) -> None:
        src = analyze_json["provenance"]["model1"]["source"]
        if src == "model":
            assert analyze_json["model1"].get("model_name") == "ResNet50-3Class", (
                "When Model 1 runs as neural net, model1.model_name must be "
                f"'ResNet50-3Class', got {analyze_json['model1']!r}"
            )

    @pytest.mark.asyncio
    async def test_model1_provenance_source(self, analyze_json: dict) -> None:
        src = analyze_json["provenance"]["model1"]["source"]
        assert src in ("model", "mock"), (
            f"provenance.model1.source must be 'model' or 'mock', got {src!r}"
        )


class TestModel2:
    @pytest.mark.asyncio
    async def test_model2_has_label(self, analyze_json: dict) -> None:
        label = analyze_json["model2"]["label"]
        assert label in MODEL2_LABELS_API, (
            "model2.label must match API labels (spaces, optional Other from mock); "
            f"got {label!r}; allowed={sorted(MODEL2_LABELS_API)}"
        )

    @pytest.mark.asyncio
    async def test_model2_has_confidence(self, analyze_json: dict) -> None:
        conf = analyze_json["model2"]["confidence"]
        assert isinstance(conf, (int, float)), (
            f"model2.confidence must be numeric, got {type(conf)}"
        )
        assert 0.0 <= float(conf) <= 1.0, (
            f"model2.confidence must be in [0,1], got {conf}"
        )

    @pytest.mark.asyncio
    async def test_model2_provenance_source(self, analyze_json: dict) -> None:
        src = analyze_json["provenance"]["model2"]["source"]
        assert src in ("model", "mock"), (
            f"provenance.model2.source must be 'model' or 'mock', got {src!r}"
        )


class TestGateLogic:
    @pytest.mark.asyncio
    async def test_gate_continue_when_abnormal(self, analyze_json: dict) -> None:
        m1 = analyze_json["model1"]["label"]
        m2 = analyze_json["model2"]["label"]
        abnormal = (m1 != "Normal") or (m2 != "Normal")
        route = analyze_json["gate"]["route"]
        if abnormal:
            assert route == "continue", (
                f"If either model is non-Normal, gate.route must be 'continue'; "
                f"model1={m1!r} model2={m2!r} route={route!r}"
            )
        else:
            assert route == "early_stop", (
                f"If both models are Normal, gate.route must be 'early_stop'; "
                f"model1={m1!r} model2={m2!r} route={route!r}"
            )

    @pytest.mark.asyncio
    async def test_gate_reason_positive(self, analyze_json: dict) -> None:
        route = analyze_json["gate"]["route"]
        reason = analyze_json["gate"]["reason"]
        if route == "continue":
            assert reason == "positive_detected", (
                f"When gate continues, reason must be 'positive_detected', got {reason!r}"
            )
        else:
            assert reason == "both_negative", (
                f"When gate early_stops, reason must be 'both_negative', got {reason!r}"
            )


class TestPipelineEndpoint:
    @pytest.mark.asyncio
    async def test_pipeline_analyze_same_shape(
        self,
        client: AsyncClient,
        test_image_bytes: bytes,
    ) -> None:
        files = {
            "image": (
                "Lung Xray.jpeg",
                test_image_bytes,
                "image/jpeg",
            )
        }
        # Identical PRNG stream per request so mock scaffolding matches byte-for-byte shape.
        random.seed(0)
        r_v1 = await client.post("/api/v1/analyze", files=files)
        random.seed(0)
        r_pipe = await client.post("/pipeline/analyze", files=files)

        assert r_v1.status_code == 200, (
            f"POST /api/v1/analyze expected 200, got {r_v1.status_code} "
            f"body={r_v1.text[:500]}"
        )
        assert r_pipe.status_code == 200, (
            f"POST /pipeline/analyze expected 200, got {r_pipe.status_code} "
            f"body={r_pipe.text[:500]}"
        )
        v1 = r_v1.json()
        pipe = r_pipe.json()
        _assert_analyze_success_payload(v1, context="POST /api/v1/analyze")
        _assert_analyze_success_payload(pipe, context="POST /pipeline/analyze")
        keys_v1 = _collect_key_paths(v1)
        keys_pipe = _collect_key_paths(pipe)
        assert keys_v1 == keys_pipe, (
            "/api/v1/analyze and /pipeline/analyze must return the same JSON key tree; "
            f"only_in_v1={sorted(keys_v1 - keys_pipe)} "
            f"only_in_pipe={sorted(keys_pipe - keys_v1)}"
        )


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_analyze_no_image_returns_error(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/analyze")
        assert 400 <= response.status_code < 500, (
            "POST /api/v1/analyze without image should return 4xx; "
            f"got {response.status_code} body={response.text[:300]}"
        )
        body = response.json()
        if response.status_code == 422:
            assert "detail" in body, (
                "FastAPI validation errors should include 'detail'; "
                f"got {body!r}"
            )
        else:
            assert body.get("success") is False, (
                f"app error payload should have success=false, got {body!r}"
            )
            assert "error" in body, (
                f"app error payload should include 'error' key, got {body!r}"
            )

    @pytest.mark.asyncio
    async def test_analyze_invalid_file_returns_error(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/api/v1/analyze",
            files={"image": ("notes.txt", b"not-a-real-image", "text/plain")},
        )
        assert 400 <= response.status_code < 500, (
            "POST with non-image MIME should return 4xx; "
            f"got {response.status_code} body={response.text[:300]}"
        )
        body = response.json()
        assert body.get("success") is False, (
            f"error payload should have success=false, got {body!r}"
        )
        assert "error" in body, f"error payload should include 'error' key, got {body!r}"
