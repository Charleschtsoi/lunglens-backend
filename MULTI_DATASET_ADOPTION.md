# Multi-Dataset Architecture Validation and 30-Day Adoption

This document operationalizes the approved plan for moving LungLens to a multi-dataset, multi-model architecture while preserving frontend stability.

## 1) Business Validity Assessment

### Decision validity

Adopting multiple datasets and independently owned model tracks is valid for this project.

- Delivery: parallel model ownership lowers team bottlenecks and improves throughput.
- Product: combining broad-screening models and specialized models increases coverage for clinically relevant findings.
- Architecture: current backend contract already supports multi-model inference (`model1`, `model2`, `gate`, `model3`, `model4`) and can hide internal model growth behind stable output keys.

### Strategy guardrails (required)

To avoid over-complexity from 4+ models, enforce these controls:

1. Stable response contract
   - Keep existing response keys unchanged for frontend compatibility.
   - Add new model outputs only as optional internal inputs to existing stage fields.

2. Model registry and versioning
   - Track each model as `{model_id, dataset, task, version, owner, status}`.
   - Only one active production version per task; others stay in shadow/candidate mode.

3. Label taxonomy freeze
   - Define one canonical label dictionary for ML Model 1/2 display terms.
   - Block deployment if training label order or names diverge from canonical mapping.

4. Routing policy
   - Define explicit rule set for conflicts across models (e.g., tie-break by calibrated confidence and task priority).
   - Keep gate behavior deterministic and auditable.

5. Quality gate before promotion
   - Require model-level metrics, calibration checks, and latency envelope verification.
   - Promote only if it improves aggregate pipeline KPI versus current baseline.

### API contract stability checklist

- Do not remove or rename:
  - `predictions`, `gradcam`, `model1`, `model2`, `gate`, `model3`, `model4`, `timing_ms`, `requires_questionnaire`.
- Preserve `gate`-driven questionnaire behavior.
- Preserve error structure (`success: false`, `error`).
- Expose model expansion internally; frontend should not need to know model count.

## 2) MIMIC-CXR Cost and Feasibility Analysis

The professor-suggested track (MIMIC-CXR scale) is feasible but should be phased with budget caps.

### Access and compliance prerequisites

- MIMIC-CXR is credentialed data through PhysioNet.
- Team members accessing data must complete required training and sign DUA.
- Data sharing restrictions mean only approved workflows and storage locations should be used.

### Cost assumptions

- GPU rates used for budget planning:
  - A100 80GB: USD 1.5-3.0 / hour
  - H100 80GB: USD 3.0-5.0+ / hour
- Additional costs considered:
  - storage + snapshots
  - failed/aborted runs
  - experiment repeats and tuning
  - inference cold-start and serving overhead

### Budget bands (planning estimates)

1. Pilot subset (10k-30k images, 1-2 models, minimal tuning)
   - Compute: ~60-180 GPU hours
   - Cost range:
     - A100: USD 90-540
     - H100: USD 180-900
   - With overhead (20-40%): USD 110-1,260

2. Practical candidate training (~100k images equivalent effort, tuning included)
   - Compute: ~300-900 GPU hours
   - Cost range:
     - A100: USD 450-2,700
     - H100: USD 900-4,500
   - With overhead (30-50%): USD 585-6,750

3. Full experimentation (multiple architectures + sweeps near full scale)
   - Compute: ~1,500-4,000 GPU hours
   - Cost range:
     - A100: USD 2,250-12,000
     - H100: USD 4,500-20,000+
   - With overhead (40-70%): USD 3,150-34,000+

### Governance recommendation

- Start with Pilot subset immediately.
- Set fixed cap (recommended: USD 1,000 initial ceiling).
- Define go/no-go criteria before further spend:
  - measurable uplift versus baseline
  - acceptable latency on deployment target
  - stable calibration and false-negative control

## 3) Frontend Change Scope in 30 Days

Scope should stay moderate by preserving API shape and improving resilience for longer pipelines.

### Must-do frontend changes

1. Async UX states for inference
   - upload/queued/running/completed/failed states.
   - timeout and retry affordances.

2. Route transparency
   - show gate outcome (`early_stop` vs `continue`) and reason.
   - display when questionnaire is required.

3. Confidence and uncertainty presentation
   - clear display of stage confidence values.
   - caution copy when confidence is borderline.

4. Error handling hardening
   - explicit UI for `401`, `413`, `415`, and generic server errors.

5. Performance and payload handling
   - ensure Grad-CAM base64 handling does not block rendering.
   - lazy-render report/details sections after core result appears.

### Defer beyond 30 days

- per-model debug panels for all model tracks
- new analytics dashboards
- major interaction redesign

### Estimated frontend effort (30 days)

- Must-do items: medium effort (about 8-14 engineering days total)
- Deferred items: additional 6-12 days if included

## 4) Four-Week Rollout With Go/No-Go Checkpoints

### Week 1: Contract and KPI freeze

- Freeze canonical ML Model 2 labels and response contract.
- Publish model registry template and promotion policy.
- Finalize spend cap and baseline KPI targets.

Go/No-Go:
- No model training starts without approved labels, contract, and budget cap.

### Week 2: MIMIC pilot and backend adapter

- Train one pilot model track on approved subset.
- Integrate candidate output into backend internals without API changes.
- Measure latency and failure modes.

Go/No-Go:
- Continue only if KPI uplift and latency are acceptable.

### Week 3: Frontend resilience implementation

- Add async status UX, gate transparency, and questionnaire trigger handling.
- Harden error experiences and loading states.

Go/No-Go:
- Continue only if frontend regression tests pass against unchanged API schema.

### Week 4: Stabilization and launch decision

- Run end-to-end regression (contract, latency, UX, error paths).
- Review total spend vs cap and quality uplift.
- Decide launch scope (pilot model on/off, fallback behavior).

Go/No-Go:
- Launch only if KPI uplift is proven and spend remains within approved tolerance.

## 5) Executive Recommendation

- Proceed with multi-dataset approach.
- Treat MIMIC-CXR as phased investment, not immediate full-scale training.
- Keep frontend contract stable and focus 30-day frontend work on resilience, not redesign.
- Enforce governance (labels, versioning, KPI gates, budget gates) to prevent uncontrolled model sprawl.

## References Used for Planning

- PhysioNet MIMIC-CXR: https://physionet.org/content/mimic-cxr/
- PhysioNet MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/
- Vercel backend docs: https://vercel.com/docs/frameworks/backend
