# #16 갱신 계약 — #17 개발 후보

## 분류와 가격

- PostgreSQL `bridge_cost.cost_events`를 단일 admission authority로 사용한다. 캐시나 SQLite 자동 fallback은 사용하지 않는다.
- `COST_PROVIDER_BILLING_JSON`은 구현된 `vertex`, `ollama`, `foundry`의 실제 계약을 `metered`, `subscription`, `nonbillable`로 명시하는 map이다. 추정 기본값은 없으며 빈 값·누락·잘못된 설정·알 수 없는 분류는 fail-closed다. OpenRouter는 미구현이다.
- 실제 payer 계약을 확인한 운영자가 map을 승인한다. Vertex는 보통 종량제이지만 기본 분류로 가정하지 않는다. Foundry가 항상 종량제이거나 Ollama가 항상 구독형인 것도 아니다.
- 분류는 provider 전체 credential/baseURL 경로에 적용된다. 같은 provider의 다른 계약을 alias별로 혼합할 수 없으며 별도 deployment로 분리해야 한다.
- `COST_TRACKING_PROVIDERS`의 빈 값=전체 추적 의미와 provider 선택 scope는 legacy다. 새 runtime은 이 키를 무시한다. 모든 `metered` 시도를 검사하고, 유효한 `subscription`/`nonbillable`은 가격·비용 설정·DB 장애와 무관하게 비용 경로 전체를 우회한다. 분류 자체의 유효성은 필수다.
- `COST_TRACKING_ENABLED=false`는 보호 기능을 명시적으로 끈 상태다. `unlimited`는 예산 한도 차단만 해제하며 종량제의 분류·가격·DB 검사는 유지한다.
- 종량제 chat은 `input_per_million`과 `output_per_million`, embeddings는 `embedding_per_million`, rerank는 `rerank_per_unit`을 모두 명시해야 한다. 적용 가격 차원이 누락되거나 잘못되면 전송 전에 차단한다.

## 전송 직전 판정

- 각 실제 provider HTTP POST/stream 연결 직전에 판정한다. Embedding batch, repair, retry, stream fallback도 각각 별도 예약이다. HTTP transport 내부 자동 재시도는 사용하지 않는다.
- 같은 transaction의 PostgreSQL advisory lock으로 `기간 내 확정 추정치 + 모든 미확정 예약 + 이번 forecast <= limit`을 원자적으로 검사하고 예약한다. 일별 기준은 UTC, short window는 설정 초 수다. 정확히 같은 금액은 허용한다.
- 모든 pod가 같은 DB·lock·분류·가격·한도 설정을 사용해야 한다. Forecast는 가격과 입력 추정치 기반이다. 분할 호출은 원 요청의 forecast 전체를 보수적으로 적용하고 repair는 변경된 모델과 forecast를 사용한다.
- 보장은 **forecast admission의 직렬화**이지 실제 청구 총액의 엄격한 상한이 아니다. 실제 비용의 양의 오차 합 `sum(max(actual - forecast, 0))`만큼 예산을 초과할 수 있다. 알 수 없는 청구·가격 오차·usage 누락 때문에 **invoice 초과액의 유한한 절대 상한은 보장하지 않는다**.
- 알려진 유효한 usage는 가격표 기반 actual 추정액으로 finalize한다. 확정 청구액은 아니다. 명시적 0은 허용하지만 missing/invalid usage를 가짜 0으로 채우지 않는다.
- Usage 누락/오류, upstream 오류, 취소, queue 유실은 `reserved` forecast를 남긴다. PostgreSQL에서는 기간 경계·prune·재시작 이후에도 무기한 유지한다. 이는 장기적인 과잉 차단을 유발할 수 있다.

## 비차단 실행과 실패

| 경로 | 고정된 실행 한계 |
|---|---|
| Admission | 전용 worker **4개**, 슬롯 외 대기열 없음, 포화 시 fail-closed |
| Admission caller | **2초** timeout. 실행 중인 작업은 timeout 후에도 완료까지 슬롯 유지 |
| DB | connect **3초**, statement **5초**, lock **5초** timeout |
| Usage 기록 | 프로세스 내 queue **256개**, 전용 record worker **1개**, 재시도 **0회** |
| Enqueue | 동기 호출, I/O 없음. Queue 포화 시 즉시 drop |
| Shutdown | **1초** drain 후 대기 job drop. 실행 중 commit 결과는 불확실할 수 있음 |

- Admission과 record는 event loop와 분리된 executor에서 실행한다. 느린 logging도 별도 executor로 격리해 event loop와 응답 전달을 막지 않는다.
- Record의 DB advisory lock이 다른 유료 admission을 지연시킬 수 있다. 이때 해당 admission은 timeout으로 fail-closed하며 유료 전송을 하지 않는다.
- Sticky 전역 latch는 없다. 이후 새 시도는 DB에 다시 판정하므로 복구된 DB를 자동으로 사용한다. 이전의 불확실한 예약/commit은 자동 retry·해제하지 않는다. 기록 복구를 위해 upstream을 재호출하지 않는다.
- Queue는 best-effort이며 durable outbox나 exactly-once 기록 보장이 아니다. Queue 포화·DB 실패·종료로 usage가 유실될 수 있지만 예약은 남는다.
- Non-stream 예산 초과는 HTTP 429 `budget_exceeded`, admission 불가는 HTTP 503 `cost_tracking_unavailable`다. Streaming gate는 headers 이후 generator 안에서 실행하므로 SSE error `budget_exceeded` / `cost_tracking_unavailable` / `cost_config_error`와 `[DONE]`을 보낸다. 거부된 시도는 유료 전송하지 않는다.

## 관측

`/healthz`는 프로세스 생존, `/readyz`는 admission DB 가용성이지 record health가 아니다. 전역 readiness를 rollout probe에 그대로 연결하면 구독형/비과금 라우팅까지 차단할 수 있어 Atlas의 별도 승인이 필요하다.

상태의 `recording` 필드는 다음과 같다.

| 구분 | 필드 |
|---|---|
| Queue/worker | `queue_depth`, `queue_capacity`, `record_in_flight` |
| 기록 결과 | `records_written`, `records_dropped`, `record_failures`, `usage_missing`, `last_record_success_at` |
| Admission | `admission_in_flight`, `admission_capacity`, `admission_timeouts`, `admission_rejected`, `admission_failures` |

프로세스 로컬 카운터는 abrupt kill 때 사라지므로 정확한 누락 건수는 알 수 없다. 외부 수집과 DB의 미정산 `reserved` age/count를 함께 검토한다. 오래된 예약이라는 이유만으로 임의 TTL·prune·release를 적용하지 않는다.

## 운영 경계

- 새 PostgreSQL DB는 빈 상태로 시작한다. 기존 SQLite 데이터 복사·이관·역이관·삭제는 수행하지 않는다. SQLite는 새 runtime에서도 로컬 호환 backend로 선택 가능하지만 공유 multi-pod 보장은 없다. PostgreSQL 선택 시 SQLite 파일에 접근하지 않는다.
- 전용 논리 DB 및 고정 schema `bridge_cost`를 사용하며 Neurons DB를 공유하지 않는다. Operator migration은 `apply_migrations` 기반 schema-only 절차다. Migrator 자격증명을 외부 주입 `COST_LEDGER_POSTGRES_DSN`에 사용해 `uv run python -m openai_compatible_bridge.core.cost_schema --apply`를 실행한다. Runtime은 schema 검증만 하며 DDL 권한을 갖지 않는다.
- Rollback은 과거 SQLite 전환으로 PostgreSQL에서 소비한 예산을 우회해서는 안 된다. 모든 pod의 유료 호출을 fence하고 PostgreSQL 기록/백업을 보존한다. 호환되는 PostgreSQL application revision으로만 rollback하거나, 별도 승인된 reconciliation/carry-forward 전략이 마련될 때까지 유료 호출을 막는다. Reverse export는 제공하지 않는다.
- DB/계정, 연결, Secret/PVC, 배포/probe, GitOps, 백업과 rollback은 Atlas의 별도 승인 작업이다. 이번 범위는 로컬 개발과 #17 갱신이며 서버 연결 확인·운영 변경·이미지 게시·merge를 수행하지 않는다. 최신 테스트 결과는 [verification.md](verification.md)에 기록한다.
