# PostgreSQL cost ledger: Atlas 전환 초안

## 상태와 작업 경계

이 문서는 최신 #16 계약을 반영하는 #17 개발 후보이며 배포 승인이 아니다.
현재 범위는 **로컬 개발과 기존 PR 갱신**이다. 서버 연결 확인 작업은 포함하지 않는다.
DB/schema/role, Secret, PVC, Deployment/probe, GitOps, 실제 원장 데이터,
이미지 게시와 merge 작업도 이번 범위에 포함하지 않는다.
Provisioning, 배포, 백업, 복구와 전환 승인은 Atlas의 별도 책임이다.
검증을 위해 legacy Jenkins 게시/GitOps 단계를 실행하지 않는다.

이전 2026-09-26 기록의 bridge 단일 ready replica/`Recreate` 및 Pod에서
PostgreSQL service로의 TCP 연결 성공은 **과거 관측**이다. 최신 상태를 재확인한
결과가 아니며 DB 권한·TLS·schema 호환·처리량·복구 가능성을 입증하지 않는다.
Ubuntu 운영 runtime 배포 증거는 Atlas가 별도로 확보해야 한다.

## 저장소와 admission 계약

- PostgreSQL을 선택하면 `bridge_cost.cost_events`가 단일 admission authority다.
  각 실제 provider HTTP 시도 직전에 같은 transaction의 advisory lock으로
  예산 검사와 forecast 예약을 원자적으로 수행한다. 캐시는 사용하지 않는다.
  Embedding batch, repair/retry, stream fallback도 각각 별도 예약을 거친다.
- 모든 pod는 같은 DB·lock 계약·billing 분류·가격·한도·UTC 기준을 사용한다.
  독립 DB, live 임의 쓰기, lock 우회, SQLite/PostgreSQL dual writer는 허용하지 않는다.
- 판정식은 `기간 내 확정 추정치 + 모든 미확정 예약 + 이번 forecast <= limit`이다.
  정확히 같은 금액은 허용한다. 이는 forecast admission 직렬화이며 invoice 상한이 아니다.
  실제 비용의 양의 오차 합 `sum(max(actual - forecast, 0))`만큼 초과할 수 있고,
  알 수 없는 청구·가격 오차·누락 때문에 실제 청구 초과액의 유한한 상한은 없다.
- 유효한 usage는 가격표 기반 actual 추정액으로 finalize한다. 확정 청구액이 아니다.
  명시적 0은 허용하지만 missing/invalid usage를 0으로 대체하지 않는다.
  Upstream 오류·취소·기록 유실도 `reserved` forecast를 남긴다.
- PostgreSQL의 `reserved`는 원래 window/day 이후에도 예산을 소비한다.
  Prune·재시작으로 삭제하거나 임의 TTL로 해제하지 않는다. Finalized/released
  기록의 정상 집계는 날짜 기준이다. 오래된 미정산 예약은 운영자 검토 대상이다.
- 금액은 PostgreSQL `NUMERIC` / Python `Decimal`을 사용한다. Event/reservation
  ID는 기존 opaque prefixed UUID 문자열을 유지한다. 동일 ledger 재적용은
  idempotent하지만 충돌하는 duplicate/terminal settlement는 덮어쓰지 않는다.
  이 저장소 성질은 runtime의 자동 record retry를 의미하지 않는다.
- `COST_LEDGER_BACKEND=postgres`와 `COST_LEDGER_POSTGRES_DSN`을 명시한다.
  PostgreSQL 선택 시 SQLite 파일을 열지 않으며 장애 시 fallback도 없다.
  SQLite는 새 runtime에서 선택 가능한 로컬 호환 기본 backend지만 multi-pod 공유
  예산 보장은 없다. SQLite backend와 PostgreSQL DSN의 혼용은 설정 오류다.

## Billing 분류와 설정 승인

`COST_PROVIDER_BILLING_JSON`은 구현된 `vertex`, `ollama`, `foundry`를 실제 payer
계약에 따라 `metered`, `subscription`, `nonbillable`로 명시한다. 추정 기본값은 없다.
Vertex는 보통 종량제지만 자동 분류하지 않는다. Foundry가 항상 종량제이거나
Ollama가 항상 구독형이라는 가정도 금지한다. OpenRouter는 구현되어 있지 않다.

분류가 누락되거나 잘못되면 fail-closed한다. Map은 provider 전체 credential/baseURL
경로에 적용되므로 같은 provider의 계약을 alias별로 섞을 수 없다. 서로 다른 계약은
별도 deployment로 분리하고 운영자가 map을 승인한다.

모든 `metered` 시도는 적용 가격 차원이 필수다. Chat은 `input_per_million`과
`output_per_million`, embeddings는 `embedding_per_million`, rerank는
`rerank_per_unit`을 모두 명시한다. 유효한 `subscription`/`nonbillable`은 가격·비용
설정 오류·DB 장애와 무관하게 비용 경로 전체를 우회한다. 분류 자체는 유효해야 한다.

`COST_TRACKING_PROVIDERS`의 과거 빈 값=전체 추적/선택 scope는 legacy이며 새 runtime은
무시한다. 종량제 면제나 구독형 기록 수단으로 사용하지 않는다.
`COST_TRACKING_ENABLED=false`는 보호 해제이며 장애 회피나 전환 방법이 아니다.
`unlimited`는 예산 한도 차단만 해제하고 종량제 분류·가격·DB 검사는 유지한다.

## Provisioning 검토 — Atlas 승인 후 수행

1. Neurons와 분리된 **bridge 전용 논리 DB**, 고정 schema `bridge_cost`, 전용 role을
   사용한다. DB는 격리된 no-login owner가 소유한다. Migrator/runtime login을 분리하고
   superuser, `CREATEDB`, `CREATEROLE`, replication 권한을 주지 않는다.
   전용 DB의 public 연결/create 권한을 제거하고 승인된 role/network만 허용한다.
2. Migrator만 schema 생성/소유 및 versioned migration 적용 권한을 갖는다.
   Runtime은 DB `CONNECT`, schema `USAGE`, migration metadata `SELECT`,
   runtime 원장 테이블 3개의 `SELECT, INSERT, UPDATE, DELETE`만 갖는다.
   `DELETE`는 retention용이다. Runtime의 DDL, schema `CREATE`, ownership,
   migrator/owner membership, migration metadata 변경, Neurons 접근은 금지한다.
3. 승인된 비운영 DB의 합성 데이터로 runtime schema 검증·예약·정산·조회·prune 권한과
   DDL/metadata 변경/다른 DB·table 접근 거부를 검증한다. Runtime admission/readiness는 schema
   version/checksum을 검증하며 누락·구버전·비호환 schema는 유료 admission을 막는다.
4. TLS, 인증, network 격리를 승인한다. TLS 환경은 적절한 CA/hostname과
   `sslmode=verify-full`을 우선 검토한다. Private cluster의 평문 연결을 자동 승인하지 않는다.
   DSN은 외부에서 주입하고 명령행·Git·로그·handoff에 값이나 자격증명을 남기지 않는다.
5. PostgreSQL 백업/PITR, 복원 검증, 저장소와 연결 용량, monitoring과 유지보수 창을 승인한다.
   Runtime 자동 migration이나 DB-admin credential 재사용으로 이 절차를 우회하지 않는다.

## Schema-only 적용 — 예시이며 미실행

Operator migration은 `apply_migrations`를 사용하는 schema-only 절차다.
승인된 migrator 자격증명을 환경 변수 `COST_LEDGER_POSTGRES_DSN`으로 주입한 뒤 실행한다.
Runtime에는 별도의 최소 권한 DSN을 주입한다. 명령행에 DSN 값을 넣지 않는다.

```bash
uv run python -m openai_compatible_bridge.core.cost_schema --apply
```

새 PostgreSQL DB는 **빈 상태로 시작**한다. SQLite 데이터 복사·이관·역이관 기능은
제공하지 않으며 기존 SQLite 파일/volume은 그대로 보존하고 삭제하지 않는다.
빈 DB는 이전 사용액이나 미확정 청구가 0이라는 증거가 아니다. 기존 예산 의무를
어떻게 반영할지 별도 승인 없이 새 예산으로 간주해 유료 트래픽을 재개하지 않는다.

## 비차단 기록과 장애 처리

| 항목 | 계약 |
|---|---|
| Admission | 전용 worker 4개, 슬롯 외 대기열 없음, 포화 시 fail-closed |
| Caller timeout | 2초. Timeout 이후 실행 중 작업도 완료까지 슬롯 유지 |
| DB timeout | connect 3초 / statement 5초 / lock 5초 |
| Usage 기록 | 프로세스 내 queue 256개, 전용 worker 1개, 재시도 0회 |
| Enqueue | 동기·I/O 없음. Queue가 가득 차면 즉시 drop |
| Shutdown | 1초 drain 후 대기 job drop. 실행 중 commit 결과는 불확실할 수 있음 |

Record와 admission은 event loop 밖의 별도 executor에서 실행한다. 느린 logging도
별도 executor로 격리하므로 event loop나 응답 전달을 기다리게 하지 않는다.
단, record transaction의 advisory lock은 다른 유료 admission을 지연시킬 수 있다.
이 경우 admission timeout으로 fail-closed한다. 이미 얻은 응답을 기록 때문에 회수하거나
기록 복구를 위해 provider를 다시 호출하지 않는다.

Sticky 전역 latch는 없다. 이후 새 시도는 DB에 다시 판정하므로 건강한 DB가 돌아오면
자동으로 새 admission이 가능하다. 이는 이전의 불확실한 예약/commit을 자동 retry하거나
해제한다는 뜻이 아니다. Queue는 best-effort이며 durable outbox가 아니다.

Non-stream은 예산 초과 시 HTTP 429 `budget_exceeded`, admission 불가 시 HTTP 503
`cost_tracking_unavailable`로 거부한다. Streaming gate는 headers 이후 generator 내부에서
실행하므로 HTTP status 변경 대신 SSE error `budget_exceeded` / `cost_tracking_unavailable`
/ `cost_config_error`와 `[DONE]`을 보낸다. 거부된 시도의 유료 전송은 없다.

## 관측과 미정산 검토

`/healthz`는 프로세스 생존, `/readyz`는 **admission DB 가용성**이다. Record health는
readiness 판정 기준이 아니다. DB 응답에 DSN, raw ledger row, SQL 오류 상세를 노출하지 않는다.
전역 `/readyz`를 rollout probe에 자동 연결하면 유효한 구독형/비과금 라우팅도 차단할 수 있다.
Atlas가 timeout·라우팅 영향을 별도로 승인해야 하며 Compose liveness는 `/healthz`를 유지한다.

상태의 `recording`에는 다음 필드를 노출한다.

| 구분 | 필드 |
|---|---|
| Queue/worker | `queue_depth`, `queue_capacity`, `record_in_flight` |
| 기록 결과 | `records_written`, `records_dropped`, `record_failures`, `usage_missing`, `last_record_success_at` |
| Admission | `admission_in_flight`, `admission_capacity`, `admission_timeouts`, `admission_rejected`, `admission_failures` |

이 값은 프로세스 로컬이며 abrupt kill 시 사라진다. 정확한 유실 건수는 알 수 없으므로
외부 수집과 DB의 outstanding `reserved` age/count를 함께 검토한다. 연결/lock timeout,
queue drop, record 실패, admission 거부, 백업·저장소 상태에 대한 경보도 필요하다.
`records_dropped`에는 종료 시 commit 결과가 불확실한 작업도 포함하므로 확정 유실 건수와
동일하지 않다. `usage_missing`은 확인 가능한 사용량이 없어 예약을 유지한 건수다.

미정산 사고를 검토할 때는 모든 pod의 유료 호출을 fence하고 PostgreSQL 기록/백업과
provider usage/billing 증거를 보존한다. Commit 응답이 유실돼도 DB에서는 완료됐을 수 있다.
원래 reservation ID와 DB 상태를 확인하고 승인된 reconciliation으로 불확실성을 해소한다.
Usage를 입증할 수 없으면 예약을 유지한다. 임의 TTL·prune·직접 table 수정으로 해제하지 않는다.
자동 복구 worker나 settlement replay CLI는 제공하지 않는다. 가용성 회복과 누락 기록 복구는
별개이며 재시작만으로 미정산 비용이 사라지지 않는다.

## 향후 전환 체크리스트 — 이번 작업에서는 실행하지 않음

1. 전용 DB/role, schema version, credential 전달, billing map·가격·한도, 백업과
   복구 담당자·중단 조건을 Atlas가 승인한다. 기존 과거 관측만으로 배포를 승인하지 않는다.
2. 모든 pod의 유료 admission을 fence하고 request/stream/repair를 drain한다.
   기존 SQLite와 PostgreSQL을 동시에 writer로 사용하지 않는다. 이전 사용액/미확정 청구의
   처리 전략을 별도 승인하고 SQLite 원본은 수정·삭제하지 않는다.
3. 승인된 migrator로 schema만 적용한다. Runtime 최소 권한과 호환성을 검증한다.
   Schema 성공은 실제 payer 계약, 가격 정확성, 청구 정산 성공의 증거가 아니다.
4. Atlas 승인 후에만 Secret/config/image/probe를 변경한다. 먼저 단일 replica에서
   runtime identity, readiness, 기록 queue, 예약/finalize 동작을 승인된 canary로 확인한다.
5. 동일 설정과 공유 DB 동시성 증거를 승인한 후에만 replica 확대를 검토한다.
   Lock/latency, budget headroom, 미정산 age/count를 관찰하고 임시 migrator 접근을 회수한다.

## Rollback — 과거 SQLite로 예산 우회 금지

- **과거 SQLite backend로 자동 전환하지 않는다.** PostgreSQL에서 이미 소비한 금액과
  미확정 예약이 빠져 예산이 다시 열릴 수 있다. SQLite 원본 보존은 rollback 승인과 다르다.
- 모든 pod의 유료 호출을 중단/fence하고 PostgreSQL 기록·usage 증거·백업을 보존한다.
  호환되는 PostgreSQL application revision으로만 application rollback을 수행한다.
- 호환 revision이 없거나 DB 상태가 불확실하면 유료 호출을 계속 fence한다.
  Reconciliation/carry-forward 전략은 별도 승인 대상이며 reverse export는 제공하지 않는다.
- PostgreSQL을 사용할 수 없으면 검증된 PostgreSQL 백업/WAL로 복구한다.
  가용성을 위해 unknown charge를 버리거나 accounting을 꺼서는 안 된다.
  Backend 분리나 부분 pod 전환으로 split brain을 만들지 않는다.

## 개발 검증 상태

최신 개발 검증은 전체 **648 passed, 0 skipped**이며 [verification.md](verification.md)에 기록한다.
이 문서의 운영 체크리스트는 실행 완료 증거가 아니다. 개발 테스트는 로컬
일회성 native PostgreSQL 환경과 합성 데이터를 사용하며 운영 DSN/실제 원장은 사용하지 않는다.
과거 576 passed 기록은 최신 #16 동작의 검증 결과가 아니다.
운영 서버 연결·schema 적용·배포 명령은 실행하지 않았다.
