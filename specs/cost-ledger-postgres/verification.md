# 개발 검증 기록

## 결과

**개발 검증 완료 · 운영 전환 미수행.**

개발 후보 브랜치: `codex/cost-ledger-postgres`, 개발 시작 기준: `d2230dd`.
Worktree: `.worktrees/cost-ledger-postgres`. 아래는 커밋·issue·PR 생성 전의
로컬 개발 검증 기록이다. 이 검증 시점에는 미커밋 상태였으며 push, merge,
이미지 게시, GitOps 동기화를 수행하지 않았다. `main` checkout과 별도의
`feat/foundry-next-model-candidates` worktree는 수정하지 않았다.

Environment: macOS, Python **3.13.13**, psycopg **3.3.6**, disposable native
PostgreSQL **17.11** (Homebrew). No test used the production DB or real cost rows.

## Executed checks

| Check | Actual result |
|---|---|
| Initial cost/package regression | 42 passed |
| New configuration/readiness regression before implementation | 13 expected failures reproduced |
| Cost/backend/request/package regression after implementation | 227 passed |
| PostgreSQL repository integration | 43 passed |
| Backup/import/compare/reverse-export tests | 57 passed |
| PostgreSQL HTTP integration, including streaming | 6 passed |
| `COST_POSTGRES_TEST_REQUIRED=1 uv run pytest -q --tb=short` | **576 passed, no skips**, 57.87s |
| Backend + HTTP tests after test-only style cleanup | 19 passed |
| `uv run python -m compileall -q openai_compatible_bridge` | Passed |
| `uv lock --check` | Passed, 51 packages resolved |
| `git diff --check` | Passed |
| Targeted `ruff check` for new backend/HTTP tests | Passed |

The full suite emitted one existing Starlette/httpx TestClient deprecation
warning; no tests failed or were disabled. PostgreSQL tests fail, rather than
skip, when required binaries are missing under `COST_POSTGRES_TEST_REQUIRED=1`.
The GitHub Actions test definition now installs native PostgreSQL and sets that
flag, but the remote workflow was not triggered.

## Covered behavior

- Independent process/thread budget admission cannot race the shared ledger
  lock; over-limit requests do not reach the provider.
- Exact money, opaque unique reservation IDs, identical retry handling,
  conflicting duplicates, missing reservations, rollback, nested savepoints,
  cancellation, connection recovery and timeouts.
- Runtime schema version/checksum validation, no implicit DDL, read-only mode,
  limited runtime permissions and incompatible schema rejection.
- PostgreSQL reservations survive window boundaries and retention. Failed
  settlement keeps the hold; database recovery does not clear the process latch.
- Real PostgreSQL-backed normal/streaming HTTP requests and admin status;
  startup/settlement connection failure returns unavailable without SQLite
  fallback, leaking credentials, or repeating a paid upstream call.
- WAL-safe SQLite backup, source preservation, no-overwrite destinations,
  restrictive permissions, atomic import and rollback, exact repeated import,
  content/count/money/state comparison, sanitized CLI failures, and reverse
  export that includes records and updates after cutover.

## Explicitly not performed / operational prerequisites

- Production database/schema/role creation, Secret/PVC/Deployment/probe changes,
  real-data backup or migration, paid canary, image publication, protected-branch
  push and GitOps synchronization were **not performed**.
- Live read-only evidence only: bridge one ready Pod with `Recreate`, PostgreSQL
  on HomeLab, and bridge-Pod-to-service TCP reachability rechecked successfully.
  This is not authorization/TLS/throughput/restore evidence.
- Atlas must approve dedicated database/least-privilege provisioning and rehearse
  the [cutover and rollback procedure](atlas-cutover.md) on its target PostgreSQL
  version. Ubuntu production runtime, CI Python 3.12 and deployment validation
  remain distinct from these local development results.
- There is no durable settlement outbox or automatic unknown-charge recovery.
  Uncertain reservations stay held; Atlas must fence all replicas, obtain
  provider/billing evidence, reconcile/replay safely, then restart deliberately.