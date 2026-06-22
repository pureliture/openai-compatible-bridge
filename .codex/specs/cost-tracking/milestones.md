# Milestones — cost tracking

## M1 Cost ledger and pricing catalog
- status: completed
- evidence:
  - `uv run pytest tests/test_cost_tracking.py` -> 12 passed
  - `uv run pytest` -> 164 passed, 1 warning
  - Added `cost_tracking.py` with config validation, explicit pricing catalog, cost estimator, SQLite allowlist ledger, retention pruning, and fail-closed ledger health circuit breaker.

## M2 Budget gate in request path
- status: completed
- evidence:
  - `uv run pytest` -> 176 passed, 1 warning
  - Added `BudgetGate` common seam with `preflight`, `finalize_success`, `release_nonbillable`, `finalize_estimated_only`, and fail-closed health handling.
  - Connected the seam to `/v1/embeddings`, `/v1/chat/completions` non-streaming, `/v1/chat/completions` streaming, and `/v1/rerank`.
  - Verified budget block prevents upstream chat call and returns OpenAI-compatible `budget_exceeded`.
  - Verified success response shape remains unchanged while cost events are finalized in the SQLite ledger.

## M3 Admin status and structured logs
- status: completed
- evidence:
  - `uv run pytest` -> 180 passed, 1 warning
  - Added private `/admin/cost/status`, `/admin/cost/events`, and `/admin/cost/reconciliation` routes.
  - Verified admin API is disabled by default, requires `COST_ADMIN_API_KEY`, and rejects `WRAPPER_API_KEY`.
  - Verified admin responses and structured `cost_tracking` logs do not include prompt/raw provider payload fields.
  - Added reconciliation placeholder states: `unavailable` when disabled and `pending` when configured but not yet reconciled.

## M4 BigQuery reconciliation
- status: completed
- evidence:
  - `uv run pytest tests/test_cost_tracking.py` -> 22 passed
  - `uv run pytest` -> 185 passed, 1 warning
  - Added aggregate-level `ReconciliationJob` with local fake adapter coverage for `matched`, `mismatch`, `pending`, `unavailable`, and BigQuery permission error as non-request-path `error`.
  - Reconciliation results persist to `cost_reconciliation_results` and admin reconciliation returns the latest persisted result when available.

## M5 Ubuntu Docker deployment readiness
- status: completed
- evidence:
  - `uv run pytest` -> 185 passed, 1 warning
  - Deployed updated files to `/home/ragflow/vertex-ai-api-wrapper` on `ragflow-ubuntu`.
  - Remote `docker compose up -d --build wrapper-vertex-ai-api` built image `sha256:28f4ae721a31e9d5fbaeb61eb637eb774277bb0beeec45397374abc43cb4767a`.
  - Production wrapper on Ubuntu is healthy at `http://127.0.0.1:8930/healthz`.
  - Production compose mounts `/home/ragflow/vertex-ai-api-wrapper/data` to `/data`.
  - Remote smoke container with cost tracking enabled returned 429 `budget_exceeded`.
  - Remote smoke container recreation with the same `/data` mount preserved the SQLite ledger and `/admin/cost/events` returned the blocked event.
  - Container import sanity passed: `cost_tracking` and `app` import inside `wrapper-vertex-ai-api`.
  - Remote host-level `uv run pytest` was not available because `uv` is not installed on the Ubuntu host; Docker build and runtime smoke covered deployment readiness.
