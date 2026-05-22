# LungLens Team Execution Checklist (30 Days)

Use this as the operational checklist for the multi-dataset adoption cycle.

## Week 1 - Freeze and Governance

- [ ] Confirm canonical ML Model 1 and ML Model 2 label taxonomy.
- [ ] Freeze backend response contract keys (no breaking changes).
- [ ] Publish model registry fields: `model_id`, `dataset`, `task`, `version`, `owner`, `status`.
- [ ] Set KPI acceptance thresholds (AUC/F1/latency).
- [ ] Set pilot budget cap (recommended initial cap: USD 1,000).
- [ ] Confirm PhysioNet credential/DUA completion for required members.

## Week 2 - MIMIC Pilot Training + Backend Adapter

- [ ] Train one pilot model track on approved subset.
- [ ] Record reproducible training config and artifact versions.
- [ ] Integrate pilot outputs into backend internals (no API schema change).
- [ ] Validate class mapping and preprocessing consistency.
- [ ] Measure inference latency and memory footprint in target environment.
- [ ] Run gate-path sanity checks (`early_stop` and `continue`).

## Week 3 - Frontend Resilience Upgrade

- [ ] Add explicit async states: upload/queued/running/completed/failed.
- [ ] Display gate route + reason clearly in result flow.
- [ ] Enforce questionnaire UX trigger via `requires_questionnaire`.
- [ ] Add user-friendly handling for `401`, `413`, `415`, and server timeout errors.
- [ ] Verify Grad-CAM rendering remains stable with larger payloads.
- [ ] Regression-test existing result components with unchanged API keys.

## Week 4 - Stabilization and Launch Decision

- [ ] Run end-to-end regression (API contract, gate flow, questionnaire, report).
- [ ] Compare pilot KPI to baseline and document uplift/no-uplift.
- [ ] Review total spend against budget cap and explain variance.
- [ ] Confirm rollback strategy (previous model version) is ready.
- [ ] Decide launch scope: pilot enabled, shadow mode, or hold.
- [ ] Publish final status memo (technical + business decision).

## Release Gate (All Must Pass)

- [ ] Frontend works with unchanged response schema.
- [ ] KPI uplift meets agreed threshold.
- [ ] Latency remains acceptable.
- [ ] Cost remains within approved range.
- [ ] Rollback is tested.

## Owners and Reporting Cadence

- [ ] Assign one owner per workstream (Model, Backend, Frontend, QA, PM).
- [ ] Run 2 checkpoints/week (cost + KPI + blockers).
- [ ] Keep one shared scoreboard: spend, latency, KPI, defects, risk.
