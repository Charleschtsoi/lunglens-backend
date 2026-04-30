# Backend Alignment Audit (Architecture Notes)

Scope audited:

- `main.py`
- `README.md`
- `requirements.txt`

Frontend code audit is deferred (frontend repository path not provided in this workspace).

## Aligned

- Backend is API-only JSON service (no HTML UI endpoints from FastAPI app).
- Image preprocessing/inference stays backend-side; frontend is not required to preprocess tensors.
- CORS is environment-driven and supports domain restriction.
- API key auth exists and can be required (`X-API-Key`).
- File validations exist (presence/type/size) with JSON error shape.
- Gate-driven questionnaire behavior exists in analyze flow (`requires_questionnaire`).
- Stage-2 `.h5` model switch is available via env vars for pilot mode.

## Misaligned

- Architecture notes specify a warm-up endpoint `GET /health` with model readiness metadata.
  - Current implementation exposes `GET /healthz` and `/`.
- Architecture notes define endpoint split:
  - `POST /predict` for image stages + gate
  - `POST /assess` for clinical assessment + report
  - Current production flow uses unified analyze endpoint (`/pipeline/analyze` and `/api/v1/analyze`).
- Architecture notes mention explicit `models_loaded` status in health response.
  - Current health responses do not expose model readiness metadata.

## Missing

- Compatibility endpoints for architecture note consumers:
  - `GET /health` (warm-up target with readiness metadata)
  - `POST /predict` (stage outputs + gate response format)
  - `POST /assess` (clinical-only follow-up format)
- Documentation that maps current production contract and architecture-note compatibility paths side-by-side.

## Adjustment Strategy

- Preserve existing frontend contract (`/pipeline/analyze`) as primary and non-breaking.
- Add compatibility aliases (`/health`, `/predict`, `/assess`) without removing current endpoints.
- Reuse existing stage/gate/report logic to avoid rewriting stable code.
