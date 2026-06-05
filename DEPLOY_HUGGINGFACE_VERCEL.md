# Deploy backend (Hugging Face) + connect frontend (Vercel)

| | URL |
|---|-----|
| **HF Space** | https://huggingface.co/spaces/Charleschtsoi/lunglens-backend |
| **Public API** | https://charleschtsoi-lunglens-backend.hf.space |
| **Vercel frontend** | https://lung-lens.vercel.app |
| **GitHub backend** | https://github.com/Charleschtsoi/lunglens-backend |
| **Git remote `hf`** | `https://huggingface.co/spaces/Charleschtsoi/lunglens-backend` |

The Space uses the Docker SDK (`README.md` front matter: `sdk: docker`). Uvicorn listens on **port 7860** inside the container.

---

## 1) Install Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```

Use a token with **write** access to your account/spaces.

---

## 2) Generate a shared API key (backend + Vercel)

Use one strong secret for both sides:

```bash
openssl rand -hex 32
```

Save it in a password manager. You will set:

- Hugging Face Space **secret** `API_KEY`
- Vercel **environment variable** `BACKEND_API_KEY` (same value)

---

## 3) Configure the Space (CLI)

From this repo root, copy the templates and edit secrets:

```bash
cp .env.hf.secrets.example .env.hf.secrets
cp .env.hf.variables.example .env.hf.variables
# Edit .env.hf.secrets — set API_KEY=... (and optional GEMINI_API_KEY)
# Edit .env.hf.variables if you change origins or model flags
```

Apply to the Space:

```bash
hf spaces secrets add Charleschtsoi/lunglens-backend --secrets-file .env.hf.secrets
hf spaces variables add Charleschtsoi/lunglens-backend --env-file .env.hf.variables
```

List (secrets show keys only, not values):

```bash
hf spaces secrets ls Charleschtsoi/lunglens-backend
hf spaces variables ls Charleschtsoi/lunglens-backend
```

**UI alternative:** Space → **Settings** → **Variables and secrets**.

After changing secrets/variables, open the Space → **Build** → **Restart Space** (or push a new commit).

---

## 4) Required Space configuration (reference)

### Secrets (encrypted)

| Name | Required | Notes |
|------|----------|--------|
| `API_KEY` | **Yes** | Must match Vercel `BACKEND_API_KEY`. Frontend sends `X-API-Key` via Next.js BFF. |
| `GEMINI_API_KEY` | No | Optional server-side Gemini for `llm_evaluation` when the browser does not send BYOK `gemini_api_key`. |

### Variables (plain)

| Name | Production value |
|------|------------------|
| `ENVIRONMENT` | `production` |
| `REQUIRE_API_KEY` | `true` |
| `ALLOWED_ORIGINS` | `https://lung-lens.vercel.app` (add preview URLs if needed, comma-separated, **no** `*`) |
| `MAX_UPLOAD_MB` | `10` |
| `ENABLE_MODEL1` | `true` |
| `MODEL1_PATH` | `models/best_resnet50_lunglens_cleaner.pth` |
| `ENABLE_MODEL2_VISION_H5` | `true` |
| `MODEL2_VISION_H5_PATH` | `models/resnet152v2_lung_disease_final.h5` |
| `MODEL2_VISION_LABELS` | `Normal,Viral_Pneumonia,Lung_Opacity` |
| `ENABLE_MODEL6_TABULAR` | `true` |
| `MODEL6_TABULAR_PATH` | `models/copd_screening_model.h5` |
| `MODEL6_SCALER_PATH` | `models/scaler.pkl` |
| `ENABLE_DENSENET121` | `true` |
| `DENSENET121_PATH` | `models/best_densenet121_lunglens.pth` |
| `ENABLE_MODEL4_SWINT` | `true` (if weights present) |
| `MODEL4_SWINT_PATH` | `models/best_swin_t_chestxray_6class.pth` |
| `ENABLE_MODEL5_DENSENET` | `true` (if weights present) |
| `MODEL5_DENSENET_PATH` | `models/best_model_DENSENET121.h5` |

Legacy env names (`ENABLE_MODEL2_H5`, `H5_MODEL2_PATH`, etc.) still work but prefer the names above.

---

## 5) Upload model weights to the Space

**Local `models/` is gitignored** (not on GitHub). The live Space currently returns `loaded: false` until weights exist under `/app/models/`.

### Option A — Git LFS push to `hf` only (recommended)

```bash
cd "/Users/charlescht/Desktop/Vibe code products/LungLens - backend"
git lfs install
git add -f models/
git status   # expect LFS pointers for .h5 / .pth / .pkl
git commit -m "Add model weights for Hugging Face Space (LFS)"
git push hf main
```

Do **not** push this commit to `origin` if you want to keep weights off GitHub:

```bash
# Push code-only to GitHub (if needed):
git push origin main
# Push weights only to HF:
git push hf main
```

If GitHub and HF share `main` with a weights commit, GitHub will also receive LFS objects (large). Use a dedicated `hf-release` branch pushed only to `hf` if you need separation.

### Option B — HF CLI upload (no git commit)

```bash
hf upload Charleschtsoi/lunglens-backend ./models models --repo-type space
```

Then restart the Space.

---

## 6) Push application code

When `main` changes locally:

```bash
git push origin main
git push hf main
```

Check sync:

```bash
git rev-parse origin/main hf/main
```

---

## 7) Vercel (frontend) environment variables

**Project:** linked to `Charleschtsoi/LungLens`  
**Settings → Environment Variables** (Production + Preview):

| Variable | Value |
|----------|--------|
| `BACKEND_API_BASE_URL` | `https://charleschtsoi-lunglens-backend.hf.space` |
| `BACKEND_API_KEY` | Same as Space `API_KEY` |
| `NEXT_PUBLIC_API_URL` | Optional: same HF URL (silent warm-up ping) |

Redeploy Vercel after changing env vars.

See also: frontend repo `PRODUCTION_DEPLOY.md`.

---

## 8) Smoke tests

```bash
export HF_API="https://charleschtsoi-lunglens-backend.hf.space"
export API_KEY="<your-api-key>"

curl -sS "$HF_API/healthz"
curl -sS -H "X-API-Key: $API_KEY" "$HF_API/health" | python3 -m json.tool

curl -sS -H "X-API-Key: $API_KEY" -X POST "$HF_API/api/v1/analyze" \
  -F "image=@testfile/Lung Xray.jpeg;type=image/jpeg"
```

Expect `GET /health` → `model1_pt.loaded: true`, `model6_tabular.loaded: true`, etc. after weights are uploaded.

On Vercel: open `/upload`, run analyze, confirm Network shows `POST /api/analyze` → 200 and results page loads.

---

## 9) Troubleshooting

| Symptom | Fix |
|---------|-----|
| `401 Unauthorized` from HF | Set `BACKEND_API_KEY` on Vercel = `API_KEY` on Space; redeploy Vercel. |
| CORS error in browser | Add your Vercel URL to `ALLOWED_ORIGINS` on the Space; restart Space. |
| `loaded: false` for all models | Upload `models/` (section 5); verify paths match env vars. |
| Cold start slow | First request after sleep can take 60s+ on free tier; use `NEXT_PUBLIC_API_URL` warm-up if enabled. |
| `gemini` educator skipped | BYOK on questionnaire step, or set `GEMINI_API_KEY` on Space. |

---

## 10) Security checklist

- Never commit `.env`, `.env.hf.secrets`, or API keys.
- Production: `ENVIRONMENT=production`, explicit `ALLOWED_ORIGINS`, `REQUIRE_API_KEY=true`.
- Rotate `API_KEY` if exposed; update Vercel and HF together.
