# Frontend E2E checklist (backend sync)

Use this with the LungLens backend while developing **locally** or against a deployed host. It mirrors the current `/pipeline/analyze` contract and when the **education report** is produced.

**Repo sync:** Run E2E against a process loading this repo’s current `main.py` (e.g. `git pull` on `LungLens - backend`). If the checklist and the running API disagree, refresh the checkout before joint tests.

## Backend changes to be aware of (local-first)

1. **Local-first workflow** — Run: `uvicorn main:app --host 0.0.0.0 --port 7860` after `pip install -r requirements.txt`.
2. **`.env`** — Copy `.env.example` → `.env` for local URLs and optional H5 settings. `.env` is not committed.
3. **`REQUIRE_API_KEY`** — Defaults to **`false` in development** (`ENVIRONMENT` unset or `development`); defaults to **`true` when `ENVIRONMENT=production`**. Send `X-API-Key` only when required.
4. **Production / Hugging Face** — Same JSON contract; when deploying again, set `ENVIRONMENT=production`, explicit `ALLOWED_ORIGINS`, `REQUIRE_API_KEY=true`, and `API_KEY`.

## Point the frontend at local API

- `BACKEND_API_BASE_URL=http://localhost:7860` (or `http://127.0.0.1:7860`).
- **Next.js proxy:** it still reads `BACKEND_API_KEY` to send `X-API-Key`. If the backend does **not** require a key locally (`REQUIRE_API_KEY=false`), the header value can be a **placeholder**; when the backend **does** require a key, the same env value must match `API_KEY` on the server.

If the browser blocks CORS, add your dev origin to backend `ALLOWED_ORIGINS` (see `.env.example`).

## Primary endpoint

- **`POST /pipeline/analyze`**
  - **Multipart:** `image` (file, required), `questionnaire` (optional stringified JSON).

### Questionnaire JSON shape (important)

The backend normalizes and validates **clinical** fields into canonical `patient_data` (snake_case in responses).

**Supported inputs:**

1. **Zustand / flat (frontend default):**  
   `age`, `fever`, `coughDurationDays`, `smoking`, `breathingDifficulty`  
   (`coughDurationDays` / `breathingDifficulty` are accepted as camelCase aliases.)

2. **Nested (legacy / docs):**  
   `{"patient_data": {"age", "fever", "cough_duration_days", ...}}`

3. **Merged:** Top-level clinical keys override the same keys inside `patient_data` when both are present (UI wins).

**Gate rule:** For `gate.route === "continue"`, the backend treats the questionnaire as **complete** only when it has successfully validated **`patient_data`** (or, on **`POST /assess`**, non-empty `clinical_data`). Sending `{}` or JSON without required fields does **not** skip the questionnaire step.

**Joint test:** Run a second `POST` with the **real** app payload (camelCase). Confirm `model4.summary` mentions fever, cough duration, smoking (former/current), and breathing difficulty (mild/severe) when applicable. On the **continue** path, `model3.recovery_outlook` may move from `favorable` to `guarded` when **`smoking` is `former` or `current`** or **`breathingDifficulty` is `mild` or `severe`**.

Required fields for validation: **`age`**, **`fever`**, **`cough_duration_days`** (or **`coughDurationDays`**).

Optional fields (match Zustand enums):

- **`smoking`:** `"never"` | `"former"` | `"current"` (case-insensitive accepted).
- **`breathingDifficulty`:** `"none"` | `"mild"` | `"severe"` (case-insensitive accepted).

**Legacy booleans** in old scripts still work: `smoking` true/false maps to `current` / `never`; `breathingDifficulty` true/false maps to `severe` / `none`.

## Error responses

Errors are always JSON. The app may include:

- `success: false`
- `error` (string, required for errors)
- Optional: `error_code`, `stage`, `retryable` — **use them when present**; older backends omit them and the client should still work.

## `warnings` / `provenance`

When the backend sends `warnings` or `provenance`, the UI (and PDF) can show run mode and degraded-model notices. If omitted, the client may use sensible defaults.

**`provenance.run_mode`:** `mock` unless ML Model 2 H5 **actually runs** on that request (`ENABLE_MODEL2_H5` or legacy `ENABLE_H5_MODEL=true`, model loaded, inference succeeds). Then **`hybrid`** by default (ML Model 1 still mock). Set backend env **`STAGE2_COUNTS_AS_FULL_REAL_RUN_MODE=true`** if the product should report **`real`** when only ML Model 2 is live. Use **`GET /health`** (`models.model2_h5.loaded` / `error`) to confirm the build before joint tests.
For backend diagnosis steps when this does not match expectations, see **`DEBUGGING_BACKEND.md`**.

## Education report: when `report` is present

| `gate.route` | Questionnaire | `requires_questionnaire` | `report` |
|--------------|---------------|---------------------------|----------|
| `early_stop` | not required | `false` | **Present** |
| `continue` | **missing / invalid** (first call) | **`true`** | **`null`** — show questionnaire |
| `continue` | **valid** (second call) | `false` | **Present** |

### E2E paths to test

1. **Early stop** — Image / mock path yields `gate.route === "early_stop"`: **Analyze** → results with **`report` non-null** and questionnaire step **skipped**.

2. **Continue** — Yields `continue` and `requires_questionnaire: true` → questionnaire step → submit with **real Zustand JSON** → results with **full `report`**; confirm copy reflects answers (see above).

3. **Errors** — Missing file (proxy **400**), oversize / bad MIME (per backend), wrong API key (**401**), missing `BACKEND_API_BASE_URL` (**500** from Next).

4. **Prod / HF** — Point `.env.local` at deployed URL + production key; repeat a smoke analyze.

### Assertions (suggested)

- [ ] `success === true` on happy path.
- [ ] When `requires_questionnaire === false`: `report` is non-null with `summary`, `recommended_actions`, `disclaimer` (match frontend types).
- [ ] Disclaimer / non-diagnostic wording still visible in UI.
- [ ] `warnings` / `provenance` handled when present; safe when absent.
- [ ] Invalid questionnaire → clear **400** and does not incorrectly skip the gate.

## Quick curl (local, no API key by default)

```bash
curl -sS -X POST "http://localhost:7860/pipeline/analyze" \
  -F "image=@/path/to/xray.jpg"
```

With **Zustand-shaped** questionnaire:

```bash
curl -sS -X POST "http://localhost:7860/pipeline/analyze" \
  -F "image=@/path/to/xray.jpg" \
  -F 'questionnaire={"age":45,"fever":true,"coughDurationDays":3,"smoking":"former","breathingDifficulty":"mild"}'
```

With nested `patient_data` (still supported):

```bash
curl -sS -X POST "http://localhost:7860/pipeline/analyze" \
  -F "image=@/path/to/xray.jpg" \
  -F 'questionnaire={"patient_data":{"age":45,"fever":true,"cough_duration_days":3}}'
```

Health (optional model diagnostics):

```bash
curl -sS "http://localhost:7860/health"
```

## Before switching back to production

- [ ] `BACKEND_API_BASE_URL` points to deployed backend.
- [ ] `BACKEND_API_KEY` matches server `API_KEY` when `REQUIRE_API_KEY=true`.
- [ ] Re-run early-stop and continue paths against production URL.

## Contract closure (frontend / backend)

**Aligned (no frontend change required for this round):** `Stage3QuestionnaireInput` uses `smoking: "never" | "former" | "current"` and `breathingDifficulty: "none" | "mild" | "severe"`; `analyzeImageFile` JSON-stringifies that object as-is, matching backend `PatientData` and normalization.

**Joint E2E (practical):** Use a backend tree whose `main.py` includes string enums plus `_questionnaire_satisfied` (and related helpers). Restart `uvicorn`, then:

1. **Path A** — Early stop: `report` non-null, questionnaire skipped.
2. **Path B** — Continue → questionnaire → second POST with real payload → spot-check **`model4.summary`** and **`model3.recovery_outlook`** for former/current smoking and mild/severe breathing.

## Related docs

- [README.md](README.md) — run instructions and env vars.
- [ARCHITECTURE_ALIGNMENT.md](ARCHITECTURE_ALIGNMENT.md) — shared pipeline contract.
