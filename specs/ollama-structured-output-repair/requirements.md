# Ollama Structured Output Repair Requirements

## 승인 대상

- Source of truth: `requirements.md`
- Preview companion: `requirements.html`
- 승인 상태: 사용자가 `requirements.md`를 승인함. Phase 2 설계를 진행한다.

## 질문-답변 흐름

### Q: Ollama Cloud structured output 실패를 어떤 제품 약속으로 다룰 것인가?

현재 사용자 판단은 다음과 같다.

- bridge에 `structured_output_repair` 옵션을 추가한다.
- 기본값은 OFF다.
- Ollama cloud에만 opt-in한다.
- 최대 3회 repair한다. 이 3회는 같은 모델 반복 호출이 아니라 지정된 Ollama Cloud 후보 모델을 순서대로 호출하는 repair chain이다.
- 실패율, latency, cost metric을 잡는다.
- Graphiti canary로 검증한다.
- 통과 모델만 allowlist한다.

선택지는 다음과 같다.

- Ollama-only repair: Ollama Cloud 응답을 bridge가 검증하고, 지정된 Ollama Cloud 후보 모델을 순서대로 최대 3회 호출해 repair한다. 끝까지 실패하면 `502 invalid_schema_output`으로 닫는다. 사용자가 원하는 "Ollama로 해결"에 가장 가깝고 provider fallback 비용/정책을 섞지 않는다.
- Ollama-first fallback: Ollama Cloud에서 최대 3회 repair 후 실패하면 Vertex/OpenAI 같은 structured-capable provider로 넘긴다. 성공률은 높지만 비용, 데이터 경로, 모델 일관성이 달라진다.
- Fail-fast 유지: 현재처럼 schema mismatch를 즉시 `502 invalid_schema_output`으로 반환하고 Graphiti 후보에서 Ollama Cloud를 제외한다. 가장 단순하지만 Ollama 활용 목표를 달성하지 못한다.

결정: **Ollama-only repair**를 선택한다. repair는 지정된 모델들을 순서대로 호출하는 3회 재시도 chain으로 수행한다. 이유는 이번 목표가 Ollama Cloud를 Graphiti 후보로 재평가하는 것이며, fallback provider를 섞으면 "Ollama가 해결됐는지"와 "fallback이 살렸는지"가 섞이기 때문이다.

repair chain의 기본 후보 순서는 다음과 같다.

1. `ollama:qwen3.5:cloud`
2. `ollama:gemma4:31b-cloud`
3. `ollama:glm-5.2:cloud`

### Q: repair prompt에 실패한 raw output을 포함할 것인가?

자문자답 결정: **포함하지 않는다**.

repair prompt는 original user messages, expected JSON schema, redacted validation error summary만 사용한다. 실패한 raw model output 전문이나 excerpt는 repair prompt, log, error response에 넣지 않는다.

이유는 raw output이 user content, retrieved private context, DSN, path, secret-like token을 포함할 수 있고, repair 모델로 다시 보내는 순간 데이터 노출면이 커지기 때문이다. repair 성공률보다 안전한 service boundary가 우선이다.

### Q: repair metric은 어디에 남길 것인가?

자문자답 결정: **cost accounting과 repair observability를 분리한다**.

각 model attempt의 token/cost accounting은 기존 cost tracking 정책을 우회하지 않는다. 별도로 repair outcome metric은 redacted operational metadata로 남긴다.

최소 metric은 다음을 포함한다.

- repair enabled 여부
- attempt index
- attempted model id
- final status
- validation failure category
- total latency bucket
- raw content가 아닌 content length

### Q: streaming 요청에서 repair를 허용할 것인가?

자문자답 결정: **허용하되 json_schema streaming은 buffering mode로 처리한다**.

`stream=true`와 `response_format.type == "json_schema"`가 함께 들어오고 repair가 켜져 있으면 bridge는 invalid partial content를 먼저 흘리지 않는다. 각 attempt는 내부적으로 buffered validation을 거치고, validation을 통과한 최종 JSON만 OpenAI-compatible stream chunk로 내보낸다. 모든 attempt가 실패하면 content chunk 없이 SSE error로 닫는다.

### Q: Graphiti canary 통과 기준은 무엇인가?

자문자답 결정: **schema contract와 extraction usefulness를 분리해서 본다**.

1차 통과 기준은 schema contract다. 지정된 후보 세션에서 `invalid_schema_output`, JSON parse error, Graphiti semantic extraction schema error가 없어야 한다.

2차 판정은 usefulness다. Gemini/Vertex control이 entity 또는 relation을 만든 세션에서 Ollama repair chain이 계속 empty extraction만 내면 model quality fail로 분리한다.

초기 기준은 모델별 3 sessions, chain 전체 0 schema failure로 둔다. 이 기준을 통과한 모델만 allowlist 후보가 된다.

## 기능 요구사항

- `response_format.type == "json_schema"` 요청에 대해 bridge는 기존 post-validation을 유지해야 한다.
- `structured_output_repair`가 OFF이면 현재 동작을 유지해야 한다.
- `structured_output_repair`가 ON이고 대상이 Ollama Cloud이면 schema mismatch 또는 non-JSON output에 대해 repair chain을 수행해야 한다.
- repair chain은 최대 3회까지만 추가 호출해야 한다.
- repair chain은 같은 모델을 3회 반복하지 않고, configured Ollama Cloud 후보 모델을 순서대로 호출해야 한다.
- 기본 repair chain 순서는 `ollama:qwen3.5:cloud`, `ollama:gemma4:31b-cloud`, `ollama:glm-5.2:cloud`이다.
- repair chain 중 어느 attempt라도 schema validation을 통과하면 즉시 성공으로 종료해야 한다.
- repair chain의 모든 attempt가 실패하면 최종 실패로 처리해야 한다.
- repair loop는 original user request, expected JSON schema, validation error summary를 사용해 모델이 schema-compliant JSON만 다시 생성하도록 유도해야 한다.
- repair prompt는 실패한 raw output 전문이나 excerpt를 포함하지 않아야 한다.
- repair loop는 raw transcript, secret, DSN, private path, raw model output을 로그나 error response에 출력하지 않아야 한다.
- repair loop가 성공하면 bridge는 OpenAI-compatible response를 200으로 반환하되, 반환 content는 최종 validation을 통과한 JSON이어야 한다.
- 최대 repair 이후에도 validation이 실패하면 bridge는 raw content 없이 `502 invalid_schema_output`을 반환해야 한다.
- repair 대상은 우선 dynamic `ollama:<native>:cloud` 모델로 제한해야 한다.
- local Ollama와 registered Ollama alias는 별도 승인 전까지 repair opt-in 대상에서 제외해야 한다.
- streaming 경로에서는 invalid partial content가 downstream으로 먼저 흘러가지 않아야 한다.
- `stream=true`와 `response_format.type == "json_schema"`가 함께 들어오고 repair가 켜진 경우, bridge는 validation을 통과한 최종 JSON만 stream chunk로 내보내야 한다.
- streaming repair가 모두 실패하면 content chunk 없이 SSE error로 실패해야 한다.
- `response_format.type == "json_object"` 기존 동작은 회귀하지 않아야 한다.
- 기존 alias 방식, dynamic Ollama routing, reasoning effort override, `<think>` stripping, cost tracking, embeddings pricing 동작은 회귀하지 않아야 한다.
- Graphiti semantic extraction canary는 repair 기능이 배포된 후에만 다시 실행해야 한다.
- Graphiti canary는 `ollama:qwen3.5:cloud`, `ollama:gemma4:31b-cloud`, `ollama:glm-5.2:cloud`를 후보로 다시 평가해야 한다.
- Graphiti 후보 allowlist는 canary를 통과한 모델만 포함해야 한다.
- Graphiti canary는 schema contract failure와 extraction usefulness failure를 분리해서 보고해야 한다.
- 초기 allowlist 후보 기준은 모델별 3 sessions에서 chain 전체 0 schema failure다.

## 비기능 요구사항

| 항목 | 요구값 |
| --- | --- |
| 기본 안전성 | `structured_output_repair` 기본값은 OFF |
| 대상 제한 | 우선 Ollama Cloud dynamic model에만 opt-in |
| 재시도 경계 | 지정된 Ollama Cloud 후보 모델을 순서대로 최대 3회 추가 호출 |
| 실패 정책 | fail-open 금지. 최종 validation 실패 시 `502 invalid_schema_output` |
| 검증 기준 | 200 응답은 반드시 schema validation을 통과한 JSON만 허용 |
| 관측성 | repair attempt count, final status, latency impact, cost estimate를 redacted metric으로 남김 |
| 보안 | raw prompt, raw completion, secret, DSN, private path 출력 금지 |
| 비용 통제 | repair attempt도 cost tracking 정책을 우회하지 않음 |
| 운영성 | M7 full runner는 명시 승인 전까지 OFF 유지 |
| 회귀 방지 | unit, route, streaming, live smoke, Graphiti canary를 분리해 검증 |

## 사용자 시나리오

- 운영자는 Ollama Cloud 모델을 Graphiti semantic extraction 후보로 살리고 싶지만, invalid schema output이 200으로 흘러가는 것은 원하지 않는다.
- bridge는 Ollama Cloud가 native strict structured output을 보장하지 않는다는 전제에서 validation과 repair를 수행한다.
- Graphiti는 bridge가 200을 반환하면 schema-compliant JSON이라고 믿을 수 있어야 한다.
- 운영자는 모델별 attempt 성공률과 repair chain 최종 성공률을 보고 특정 Ollama Cloud 모델을 allowlist하거나 제외한다.
- 운영자는 M7 full runner를 켜기 전 small live smoke와 Graphiti canary로 모델별 성공률을 확인한다.

## 미결정 항목

- 없음.
