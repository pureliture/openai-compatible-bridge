# 개발 검증 기록

## 최신 #16 / #17 상태

**갱신된 #16 기준 개발 검증 완료 · 운영 전환 미수행.**

개발 브랜치는 `codex/cost-ledger-postgres`, worktree는
`.worktrees/cost-ledger-postgres`다. 이번 범위는 로컬 개발과 기존 #17 갱신이다.
기준 commit `8ff7a26` 이후의 후속 변경에서 2026-09-26 아래 검증을 수행했다.
환경은 macOS, Python **3.13.13**, psycopg **3.3.6**, 일회용 native
PostgreSQL **17.11**이다. 운영 DSN이나 실제 비용 행을 사용하지 않았다.

| 실제 명령/검증 항목 | 결과 |
|---|---|
| `COST_POSTGRES_TEST_REQUIRED=1 uv run --no-sync pytest -q --tb=short` | **648 passed, 0 skipped**, 최종 64.28초 |
| `COST_POSTGRES_TEST_REQUIRED=1 uv run --no-sync pytest -q --tb=short tests/test_async_cost.py tests/test_cost_postgres_api.py tests/test_cost_schema.py --durations=5` | **29 passed**, 35.17초 |
| `uv lock --check` | 통과, 51 packages |
| `uv run --no-sync python -m compileall -q openai_compatible_bridge` | 통과 |
| `git diff --check` | 통과 |
| `uv tool run ruff check` — 신규 비용 경계/CLI와 관련 테스트 7개 파일 | 통과 |
| 환경 예제 정적 검증 | UTF-8·공백·폐기 CLI 참조 검사 및 Compose YAML·기본값 검사 통과 |

기존 Starlette/httpx TestClient deprecation 경고 1건이 있다. Compose 기동이나
운영 환경 검증을 수행한 결과가 아니며 GitHub Actions 검증과도 구분한다.
기존 57개 이관/역이관 테스트는 기능 자체를 범위에서 제거하면서 schema-only 테스트로
대체했다. 실패를 숨기기 위한 skip은 없다.

## 검증한 계약과 한계

- `test_metered_http.py`: 실제 provider client + `MockTransport`, Foundry 5개 프로토콜,
  병렬 Vertex embedding·stream fallback, Ollama 재호출, 원본 usage와 취소/오류.
- `test_wrapper.py`, `test_ollama_chat.py`: 실제 HTTP 경계에서 budget 거부 시 미전송,
  일반/structured repair 스트림·도구 호출·응답 형태 회귀, 구독형 제외.
- `test_cost_postgres_api.py`: 독립 앱 2개에서 동시 호출 4건 중 정확히 1건만 한도 내 전송,
  실제 PostgreSQL lock 대기 중 health/구독형 스트림 진행, 판정 timeout 시 전송 0건,
  usage 기록 장애에도 응답 유지, DB 회복 후 새 판정, SQLite 미생성.
- `test_async_cost.py`: 슬롯/큐 상한, 기록 실패·종료 유실, 재시작 후 미확정 예약 보존,
  age/prune 경계, 알려진 0/누락 구분, forecast보다 큰 실제 비용의 기록을 검증했다.
- `test_postgres_cost_repository.py`: 기존 실제 독립 프로세스·스레드 직렬화,
  정밀도·중복·rollback·schema/권한·timeout 검증을 전체 suite에서 재실행했다.
- 초과 기록은 실제 invoice 상한을 입증하지 않는다. 실제 운영 provider·DB를 호출하지 않았고,
  sudden process kill의 정확한 유실 건수를 재구성하는 기능도 제공하지 않는다.

- PostgreSQL 단일 authority, 캐시 없음, 여러 pod/process의 advisory-lock budget
  검사와 forecast 예약 원자성. 각 실제 전송 직전 gate와 embedding batch,
  repair/retry/stream fallback의 개별 예약 및 거부 시 유료 전송 없음.
- 명시적 `COST_PROVIDER_BILLING_JSON`, 누락/잘못된 분류 fail-closed,
  provider 전체 credential/baseURL 계약 경계, legacy scope 무시, `false` 명시적 off.
  종량제의 적용 가격 차원 전체 필수 및 유효한 구독형/비과금의 비용 경로 완전 우회.
- Admission worker 4개와 별도 대기열 없음, caller timeout 2초 뒤에도 실행 중 슬롯 유지,
  DB connect/statement/lock 3/5/5초, 포화·장애 fail-closed와 새 시도의 DB 자동 회복.
- I/O 없는 동기 enqueue, queue 256개·record worker 1개·retry 0회,
  shutdown 1초 drain 후 drop, 실행 중 commit 불확실성, 느린 logging 격리.
  Record advisory lock이 다른 유료 admission을 지연시키는 경우의 timeout 차단.
- Missing/invalid usage, upstream 오류·취소·queue 유실의 예약 유지,
  age/prune/restart 이후 PostgreSQL 예약 보존, 알려진 명시적 0 허용과 fake zero 금지.
  Actual price estimate는 확정 청구가 아니며 forecast 오차에 따른 invoice 상한은 없음.
- Non-stream 429/503과 headers 이후 streaming SSE error + `[DONE]`,
  `/healthz` 생존과 `/readyz` admission DB 의미, record health와 readiness 분리,
  `recording` 지표 및 프로세스 종료 시 정확한 누락 건수 관측 불가능성.
- Schema-only 적용, runtime DDL 금지, 전용 DB/schema 권한과 호환성,
  PostgreSQL 선택 시 SQLite 미사용, 기존 SQLite 보존 및 호환 backend 선택.

## 과거 검증 기록 — 최신 계약에는 적용하지 않음

아래는 `8ff7a26`에서 진행한 **이전 계약의 로컬 검증 기록**이다.
당시 문서에 기록된 환경은 macOS, Python 3.13.13, psycopg 3.3.6,
일회성 native PostgreSQL 17.11(Homebrew)이며 운영 DB/실제 원장을 사용하지 않았다.
이전 세션의 기록이며 최신 기준의 결과와 구분한다.

| 과거 항목 | 당시 기록 |
|---|---|
| `COST_POSTGRES_TEST_REQUIRED=1 uv run pytest -q --tb=short` | 576 passed, no skips, 58.05초 |
| 경고 | 기존 Starlette/httpx TestClient deprecation 경고 1건 |
| 개발 경계 | 당시 이미지 게시·배포·GitOps 동기화 미수행 |

이 수치는 이전 sticky latch와 폐기된 데이터 이관 기능을 포함한 suite의 결과다.
최신 #16의 비차단 기록·호출별 admission·schema-only CLI가 통과했다는 근거가 아니다.
폐기된 이관 테스트 수치나 구 계약의 통과 목록은 최신 acceptance evidence로 사용하지 않는다.

## 수행하지 않은 운영 작업

- DB/schema/role 생성, Secret/PVC/Deployment/probe 변경, 실제 데이터 백업/이관/삭제,
  paid canary, 이미지 게시, merge와 GitOps 동기화는 이번 개발에서 수행하지 않았다.
- 최신 요청에는 서버 연결 확인 작업이 없다. 기존 2026-09-26의 단일 ready Pod/`Recreate`
  및 PostgreSQL service TCP 연결 성공 기록은 과거 관측이며 새 연결 검증이 아니다.
  권한·TLS·처리량·복원 가능성도 입증하지 않는다.
- Atlas는 전용 DB/최소 권한, schema-only 적용, 빈 DB 시작 시 이전 비용 의무,
  [전환과 안전한 rollback](atlas-cutover.md)을 별도 승인해야 한다.
  Ubuntu runtime, CI 및 배포 검증은 로컬 개발 결과와 분리한다.
- 전역 `/readyz`를 rollout probe에 자동 연결하지 않는다. 구독형 라우팅 영향도 별도 승인한다.
  Record queue는 durable outbox가 아니며 카운터는 abrupt kill 때 사라진다.
  미정산 DB 예약을 age/count와 증거로 검토하고 임의 TTL로 해제하지 않는다.
- Rollback은 모든 pod의 유료 호출 fence와 PostgreSQL 기록/백업 보존이 우선이다.
  호환 PostgreSQL application revision이 없으면 별도 reconciliation/carry-forward 승인까지
  fence를 유지한다. 과거 SQLite로 자동 전환하거나 reverse export로 우회하지 않는다.
