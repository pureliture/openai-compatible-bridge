# Wrapper Cost Tracking Requirements

## 승인 대상

- Source of truth: `requirements.md`
- Preview companion: `requirements.html`

## 질문-답변 흐름

### Q: 비용 추적 기능은 어떤 방향으로 가야 하는가?

사용자는 "Best" 접근을 원한다. 즉 wrapper request 단위의 실시간 추정 비용을 남기고, Cloud Billing BigQuery export와 나중에 reconciliation하는 방향이다.

### Q: 비용 추적 기능의 1차 목적은 무엇인가?

예산 통제 중심으로 확정한다. 비용 추적은 단순 관측이 아니라 예산 초과 알림과 차단까지 포함해야 한다.

### Q: 예산 초과 시 차단 강도는 무엇인가?

Hard limit로 확정한다. 예산 기준을 초과한 뒤에는 새 billable 요청을 차단해서 비용 폭주를 막아야 한다.

### Q: 예산은 어떤 경계로 적용하는가?

Wrapper 전체 예산으로 확정한다. 모델별 또는 클라이언트별 세부 할당보다, 전체 wrapper가 일정 기간 동안 쓸 수 있는 총 비용 상한을 먼저 둔다.

### Q: 예산 기간은 무엇인가?

단기 윈도우와 일 단위 예산을 함께 둔다. 단기 윈도우는 몇 분 또는 한 시간 안의 급격한 비용 폭증을 막고, 일 단위 예산은 하루 총액을 제한한다.

### Q: 차단된 요청 응답은 얼마나 자세해야 하는가?

운영 정보 포함으로 확정한다. 차단 응답은 `budget_exceeded`를 명확히 표시하고, 어떤 limit이 걸렸는지, reset 예상 시각, 현재 추정 사용량과 limit을 포함해야 한다. 단, prompt, completion, embedding vector, raw document text, 상세 tenant 정보는 포함하지 않는다.

### Q: 실시간 estimate는 어디에 노출하는가?

기존 OpenAI-compatible 성공 응답 shape는 기본적으로 바꾸지 않는다. 실시간 비용 정보는 private admin 조회 surface와 structured server log에 노출한다. 차단 응답에는 운영자가 즉시 원인을 알 수 있는 최소 운영 정보를 포함한다.

### Q: Reconciliation은 자동화하는가?

Cloud Billing BigQuery export가 설정되어 있으면 자동 reconciliation report를 생성하는 방향으로 확정한다. BigQuery export가 아직 없거나 지연되면 wrapper estimate를 계속 남기고 reconciliation 상태를 `pending` 또는 `unavailable`로 표시한다.

### Q: 저장 보존 기간과 집계 단위는 무엇인가?

Request-level 원장은 90일 보존, 일 단위 aggregate와 reconciliation 결과는 13개월 보존으로 확정한다. 이는 운영 디버깅과 월별 청구 대조를 모두 지원하기 위한 기준이다.

### Q: 예산 금액은 코드 기본값을 둘 것인가?

금액 기본값은 두지 않는다. cost tracking이 활성화된 runtime에서는 단기 윈도우 예산과 일 단위 예산을 명시 config로 제공해야 한다. 예산 config가 없으면 hard limit이 켜진 척 동작하지 않아야 하며, 설정 오류를 운영자가 명확히 볼 수 있어야 한다.

## 현재 맥락

- 대상 시스템은 Google Cloud Vertex AI / Agent Platform을 OpenAI-compatible 형태로 노출하는 private wrapper이다.
- Runtime target은 Ubuntu Docker `wrapper-vertex-ai-api`이며, Mac `launchd` wrapper는 제거되었다.
- Wrapper는 `/v1/embeddings`, `/v1/chat/completions`, `/v1/rerank`를 제공한다.
- `/v1/chat/completions`와 `/v1/embeddings`는 이미 usage/token 정보를 응답 경로에서 다룬다.
- `/v1/rerank`는 현재 token usage 신호가 없다.
- Google 공식 문서 기준으로 실시간 invoice-level exact cost는 불가하고, Cloud Billing export to BigQuery는 지연과 스키마/서비스 보고 주기 차이를 가진다.

## 기능 요구사항

- Wrapper는 성공한 billable request에 대해 request 단위 cost estimate를 기록해야 한다.
- Wrapper는 모델, endpoint, usage, estimated cost, currency, pricing source, timestamp를 감사 가능한 형태로 남겨야 한다.
- Wrapper는 Cloud Billing BigQuery export와 나중에 대조할 수 있도록 reconciliation에 필요한 식별자와 집계 축을 남겨야 한다.
- Wrapper는 cost estimate가 불가능한 요청을 성공 실패와 별도로 표시해야 한다.
- Wrapper는 설정된 예산 기준을 넘는 사용을 감지하고 알림 또는 차단을 수행해야 한다.
- Wrapper는 hard limit 초과 상태에서 새 billable 요청을 차단해야 한다.
- Wrapper는 wrapper 전체 사용량 기준으로 예산을 계산하고 차단해야 한다.
- Wrapper는 단기 윈도우 예산과 일 단위 예산 중 하나라도 hard limit을 초과하면 새 billable 요청을 차단해야 한다.
- Wrapper는 예산 통제 상태에서도 기존 OpenAI-compatible 오류 형식으로 실패를 반환해야 한다.
- 차단 응답은 `budget_exceeded`, limit type, reset time, current estimated spend, configured limit을 포함해야 한다.
- 기존 OpenAI-compatible 성공 응답 contract는 기본적으로 유지해야 한다.
- 실시간 비용 estimate는 private admin 조회 surface와 structured server log를 통해 확인할 수 있어야 한다.
- Cloud Billing BigQuery export가 설정된 경우 wrapper estimate와 실제 billing export를 자동으로 reconciliation할 수 있어야 한다.
- Reconciliation 지연 또는 export 미설정 상태는 별도 상태로 표시되어야 한다.
- Wrapper는 private proxy 운영 전제를 유지해야 하며 public cost dashboard를 기본 제공하지 않는다.
- Cost tracking 활성화 시 budget amount와 pricing config는 명시되어야 하며, 금액 기본값으로 조용히 동작하지 않아야 한다.

## 비기능 요구사항

| 항목 | 요구값 |
| --- | --- |
| 정확도 | 실시간 값은 estimate, Cloud Billing export 값은 invoice reconciliation source로 구분한다. |
| 안정성 | 비용 추적 저장/로그 실패가 정상 API 응답을 깨지 않아야 한다. 단, hard limit 판정 실패는 fail-closed로 차단한다. |
| 보안 | prompt, completion, embedding vector, raw document text는 비용 로그에 저장하지 않는다. |
| 운영성 | Ubuntu Docker runtime에서 재시작 후에도 추적 데이터가 보존되어야 한다. |
| 호환성 | 기존 `/v1/*` 성공 응답 shape는 기본적으로 유지한다. |
| 과금 기준 | 200 응답만 billable estimate 대상이다. |
| 통제 방향 | 비용 추적은 관측뿐 아니라 예산 초과 알림과 차단을 포함한다. |
| 차단 강도 | Hard limit. 예산 초과 뒤 새 billable 요청은 차단한다. |
| 예산 경계 | Wrapper 전체 예산. 모델별/클라이언트별 예산은 초기 범위에서 제외한다. |
| 예산 기간 | 단기 윈도우 + 일 단위. 단기 폭증과 하루 총액을 모두 방어한다. |
| 노출 정책 | 성공 응답은 그대로 두고, 차단 응답과 private admin surface에만 비용 정보를 노출한다. |
| 보존 기간 | Request-level 원장 90일, 일 단위 aggregate와 reconciliation 결과 13개월. |
| 가격 설정 | 코드에 금액 기본값을 두지 않고 runtime config로 명시한다. |

## 사용자 시나리오

- 운영자는 모델 전환 뒤 실제 절감 효과를 wrapper request 단위로 빠르게 보고 싶다.
- 운영자는 하루 또는 일정 기간 단위로 wrapper estimate와 Cloud Billing export를 대조하고 싶다.
- 운영자는 특정 모델이나 endpoint가 비용을 많이 쓰는지 알고 싶다.
- 운영자는 가격표 누락, usage 누락, Billing export 지연을 구분해서 보고 싶다.
- 운영자는 예산을 넘는 사용이 계속될 때 wrapper가 자동으로 더 큰 비용 발생을 막아주길 원한다.
- 운영자는 hard limit에 걸린 요청이 정상적인 provider 장애가 아니라 예산 차단임을 구분하고 싶다.
- 운영자는 특정 모델이나 클라이언트가 아니라 wrapper 전체 비용 폭주를 먼저 막고 싶다.
- 운영자는 몇 분 또는 한 시간 안의 비용 폭증과 하루 누적 비용 초과를 모두 막고 싶다.
- 운영자는 성공 응답의 OpenAI 호환성이 깨지지 않으면서도 private admin surface에서 비용 상태를 보고 싶다.
- 운영자는 Billing export가 늦게 도착해도 wrapper estimate가 사라지지 않기를 원한다.

## 리서치 메모

- Agent Platform pricing 문서는 200 응답만 input/output 과금 대상이라고 설명한다.
- Cloud Billing export to BigQuery는 usage, cost estimate, pricing data를 BigQuery dataset으로 내보낼 수 있다.
- Cloud Billing export는 지연될 수 있으며 delivery latency guarantee가 없다.

## 제외 범위

- Public dashboard 제공.
- 모델별 예산과 클라이언트별 예산.
- Prompt, completion, raw document text 저장.
- Cloud Billing export 미설정 환경에서 invoice-level exact cost를 실시간으로 보장하는 것.
- Discount, credit, tax, currency conversion까지 반영한 실시간 exact invoice 계산.

## 미결정 항목

- 없음. 이 `requirements.md` 승인을 기다린다.
