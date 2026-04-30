# LungLens Professor Update Memo (Decision-Focused)

## Objective

Confirm whether LungLens should proceed with a multi-dataset, multi-model architecture in the next 30 days, including a MIMIC-CXR pilot track.

## Current Decision

- **Proceed** with multi-dataset architecture.
- **Preserve frontend API contract** (no breaking schema changes).
- **Adopt phased MIMIC-CXR pilot first**, not full-scale training immediately.

## Why This Is Valid

- Team-level parallelization: model owners can train independently, reducing project bottlenecks.
- Clinical value: separate models capture complementary signals (binary triage + multi-class pathology + conditional clinical risk).
- Technical fit: existing backend contract already supports staged routing (`stage1`, `stage2`, `gate`, `stage3`, `report`).

## Cost and Feasibility Summary (Budgetary)

- Access prerequisites: PhysioNet credentialing + DUA compliance for MIMIC-CXR.
- Estimated spend bands (including overhead risk):
  - **Pilot subset**: ~USD **110-1,260**
  - **Practical candidate training**: ~USD **585-6,750**
  - **Full experimentation**: ~USD **3,150-34,000+**
- Recommendation: set initial cap at **USD 1,000** for pilot go/no-go.

## Frontend Impact (30-Day Scope)

- Moderate if API shape remains stable.
- Required changes:
  - long-running inference states (queued/running/completed/error),
  - route transparency (`early_stop` vs `continue`),
  - questionnaire trigger handling,
  - robust error UX (`401`, `413`, `415`, timeout/server).
- Major redesign and per-model dashboards are deferred.

## 30-Day Plan

- **Week 1**: freeze labels, API contract, KPI targets, and budget cap.
- **Week 2**: train MIMIC pilot subset + backend adapter integration.
- **Week 3**: frontend resilience hardening for async/staged output behavior.
- **Week 4**: regression, KPI/cost review, and launch decision.

## Go/No-Go Criteria

- KPI uplift vs baseline is demonstrated.
- End-to-end latency remains acceptable for user flow.
- Spend stays within approved cap.
- Frontend contract remains backward compatible.

## Request for Approval

Approve the phased pilot approach (subset-first, budget-capped) before any full-scale MIMIC expansion.
