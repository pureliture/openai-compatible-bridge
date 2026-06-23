# Ollama Structured Output Repair Design Spec

## Overview

Ollama Cloud의 `response_format=json_schema` 실패를 bridge boundary에서 `validate -> repair chain -> final validate`로 흡수한다. 200 응답은 schema-compliant JSON일 때만 허용하고, 끝까지 실패하면 raw content 없이 `502 invalid_schema_output`으로 닫는다.

## Requirements Reference

- Phase 1 source: `requirements.md`
- Preview companion: `requirements.html`

핵심 요구사항은 다음과 같다.

- `structured_output_repair` 기본값은 OFF다.
- 대상은 우선 dynamic `ollama:<native>:cloud` chat completion의 `response_format.type == "json_schema"`로 제한한다.
- repair는 provider fallback이 아니라 Ollama Cloud-only chain이다.
- repair chain 순서는 `ollama:qwen3.5:cloud`, `ollama:gemma4:31b-cloud`, `ollama:glm-5.2:cloud`이다.
- repair prompt에는 실패한 raw output 전문이나 excerpt를 넣지 않는다.
- `stream=true + json_schema`는 buffering mode로 처리하고 valid final JSON만 SSE content chunk로 내보낸다.
- cost accounting과 repair observability는 분리한다.
- Graphiti canary는 schema contract failure와 extraction usefulness failure를 분리해서 보고한다.

## Approach Proposal

추천안: **route/service boundary의 repair orchestrator**를 둔다.

이 방식은 `/v1/chat/completions` route가 initial Ollama attempt 실패를 보고 repair chain을 실행한다. 각 attempt는 기존 `OllamaChatClient.generate()`를 재사용하되, attempt model, repair prompt, schema validation, cost preflight, usage aggregation, metric emission을 route 근처의 작은 service가 조율한다.

이유는 repair가 단순 provider retry가 아니기 때문이다. 같은 upstream request 안에서 서로 다른 Ollama Cloud model을 추가 호출하므로, pricing fail-closed, usage aggregation, streaming envelope, response model identity, redacted observability가 provider adapter 내부만으로는 깔끔하게 닫히지 않는다.

대안 1: **Ollama adapter 내부 repair**.

장점은 변경 위치가 provider 파일 하나에 모인다. 단점은 adapter가 user-facing model id, cost policy, stream SSE behavior를 알아야 해서 provider boundary가 비대해진다.

대안 2: **Graphiti 쪽 retry/repair**.

장점은 bridge 변경이 작다. 단점은 OpenAI-compatible endpoint가 200/502 contract를 스스로 보장하지 못하고, 다른 bridge consumer는 같은 실패를 다시 겪는다.

자문자답 결정: 추천안을 선택한다. bridge가 OpenAI-compatible contract를 제공하는 계층이므로 repair도 bridge boundary에서 책임진다.

## Architecture

```text
OpenAI-compatible client
  -> POST /v1/chat/completions
    -> auth, model resolution, response_format validation
    -> cost preflight for initial request
    -> provider dispatch
      -> OllamaChatClient.generate(initial model)
      -> json_schema validation
    -> StructuredOutputRepairOrchestrator
      -> eligibility check
      -> repair chain planning
      -> per-attempt cost pricing validation
      -> per-attempt OllamaChatClient.generate(repair model)
      -> final schema validation
      -> redacted repair metric
    -> OpenAI-compatible response or error
```

The Ollama adapter remains responsible for native request construction, `format` schema forwarding, reasoning effort mapping, `<think>` stripping, and single-attempt schema validation. The repair orchestrator owns cross-attempt policy.

## Data Flow

### Non-Stream Success Without Repair

```text
request model = ollama:qwen3.5:cloud
response_format = json_schema
structured_output_repair = ON

initial attempt -> validation passes
-> no repair chain
-> return 200 with validated JSON
```

### Non-Stream Repair Success

```text
initial attempt -> invalid_schema_output
repair attempt 1: ollama:qwen3.5:cloud -> invalid_schema_output
repair attempt 2: ollama:gemma4:31b-cloud -> validation passes
-> return 200 with attempt 2 validated JSON
-> emit redacted repair metric with winner model and attempt count
```

The OpenAI-compatible `model` field remains the requested `payload.model`. The winning repair model is exposed only through redacted operational metrics, not through response body shape changes.

### Non-Stream Final Failure

```text
initial attempt -> invalid_schema_output
repair attempt 1 -> invalid_schema_output
repair attempt 2 -> invalid_schema_output
repair attempt 3 -> invalid_schema_output
-> return 502 invalid_schema_output
-> no raw model output in response or logs
```

### Streaming Repair

```text
stream=true + response_format=json_schema + repair enabled
-> bridge does not forward invalid partial content
-> repair attempts run in buffered mode
-> if final JSON validates, emit one OpenAI-compatible content chunk and finish
-> if all attempts fail, emit SSE error and [DONE]
```

This intentionally sacrifices token-by-token streaming for contract safety on structured output requests.

## Component Details

### StructuredOutputRepairConfig

Purpose:

- Make repair opt-in explicit and bounded.

Inputs:

- enabled flag, default OFF
- repair model chain, default:
  - `ollama:qwen3.5:cloud`
  - `ollama:gemma4:31b-cloud`
  - `ollama:glm-5.2:cloud`
- max repair attempts, default and cap: 3

Behavior:

- Applies only to dynamic `ollama:<native>:cloud` requests.
- Applies only to `response_format.type == "json_schema"`.
- Does not apply to local Ollama, registered Ollama aliases, Vertex, embeddings, or rerank.

### StructuredOutputRepairOrchestrator

Purpose:

- Convert `invalid_schema_output` from a single Ollama Cloud attempt into a bounded repair chain.

Inputs:

- original payload fields needed for generation
- original messages
- JSON schema
- validation failure category
- Ollama client
- cost accounting handle

Outputs:

- validated final result, or `VertexAPIError(502, code="invalid_schema_output")`
- redacted repair outcome metric

Behavior:

- Runs only after the initial attempt fails with `invalid_schema_output`.
- Does not retry timeout, auth, missing pricing, invalid request, or upstream non-schema errors as repair.
- Uses the configured model chain in order.
- Stops on first schema-valid output.
- Aggregates usage across attempts when usage is available.
- Emits only redacted metadata: enabled state, attempt count, attempted model ids, final status, failure category, latency bucket, content length.

### Repair Prompt Builder

Purpose:

- Ask the next Ollama Cloud model to produce schema-compliant JSON without leaking failed raw output.

Inputs:

- original messages
- JSON schema
- redacted validation error summary

Behavior:

- Prepends or appends a repair instruction that requires JSON only.
- Includes the expected schema.
- Includes validation failure category and schema path when available.
- Does not include failed raw output, raw transcript dump, secrets, DSN, or private path.
- Keeps `temperature` low when the request did not explicitly set it.

### Schema Validation Boundary

Purpose:

- Preserve current contract: 200 only after final validation.

Behavior:

- Existing schema validation stays in Ollama single-attempt path.
- For orchestration, validation errors are classified without exposing raw output.
- Non-JSON and schema mismatch both remain `invalid_schema_output`.

### Cost And Usage Handling

Purpose:

- Keep cost tracking fail-closed even when repair calls multiple models.

Behavior:

- Initial request cost preflight remains unchanged.
- When repair is enabled, each configured repair model must have pricing through exact model pricing or explicit `ollama:*` fallback before its attempt is called.
- Missing pricing for any attempted repair model fails closed before that model is called.
- Successful response finalizes aggregate usage from initial and repair attempts where available.
- Repair observability metric is separate from the billing ledger.

### Streaming Adapter

Purpose:

- Avoid sending invalid partial content before validation.

Behavior:

- For `stream=true + json_schema + repair enabled`, the route uses buffered generation internally.
- On success, it emits standard OpenAI-compatible SSE chunks: role, content, finish, `[DONE]`.
- On failure, it emits an SSE error with `invalid_schema_output` and `[DONE]`.
- Existing streaming behavior remains unchanged when repair is OFF or when `response_format` is not `json_schema`.

## Error Handling

- Repair OFF: current `invalid_schema_output` behavior remains unchanged.
- Initial invalid request: HTTP 400, no repair.
- Invalid JSON schema: HTTP 400, no repair.
- Missing pricing for initial model: existing cost error, no repair.
- Missing pricing for repair model: fail closed before that attempt.
- Initial timeout or connection error: existing upstream error, no repair.
- Initial or repair schema mismatch: eligible for next repair attempt.
- All repair attempts fail schema validation: `502 invalid_schema_output`.
- Streaming final failure: SSE error with `invalid_schema_output`, no content chunk.

All errors remain raw-content-free.

## Testing Strategy

- RED: repair OFF preserves current immediate `502 invalid_schema_output`.
- RED: repair ON for dynamic `ollama:qwen3.5:cloud` calls repair chain in configured order.
- RED: repair chain stops after first schema-valid attempt.
- RED: all repair attempts invalid returns `502 invalid_schema_output`.
- RED: repair prompt excludes failed raw output.
- RED: repair does not run for local Ollama, registered alias, Vertex, `json_object`, invalid schema, timeout, or missing pricing.
- RED: missing pricing for a repair candidate fails closed before calling that candidate.
- RED: `stream=true + json_schema + repair ON` emits no invalid partial content and succeeds with one validated content chunk.
- RED: streaming repair final failure emits SSE error without content.
- Regression: dynamic Ollama routing, alias routing, reasoning effort override, `<think>` stripping, embeddings pricing, and full test suite still pass.
- Live smoke: deploy bridge, run minimal json_schema smoke for `ollama:qwen3.5:cloud`, `ollama:gemma4:31b-cloud`, `ollama:glm-5.2:cloud`, then run Graphiti canary with M7 full runner still OFF.

## Milestones

- M1: Spec approval — `requirements.md` 승인 반영, `design.md` 작성 및 self-review.
- M2: RED tests — repair OFF/ON, chain order, cost fail-closed, streaming buffering, no raw-output leak 테스트를 먼저 작성한다.
- M3: Minimal implementation — config, orchestrator, repair prompt builder, route integration, metrics를 구현한다.
- M4: Local verification — targeted tests와 full `uv run pytest`를 통과시킨다.
- M5: Review and docs — multi-agent review, README/runtime config 문서 갱신, 보안/운영 문구 점검.
- M6: Deploy and bridge smoke — Ubuntu 배포 후 redacted smoke로 200/502 contract와 repair metric을 확인한다.
- M7: Graphiti canary — M7 full runner는 켜지 않고, 지정 3개 모델의 Graphiti semantic extraction canary만 실행해 schema contract와 usefulness를 분리 보고한다.
- M8: Allowlist handoff — canary 통과 모델만 neurons allowlist 후보로 전달한다.

## Deployment and Canary Result

- M6 완료: Ubuntu bridge 배포는 commit `4d22026` 기준으로 완료했고, container health는 healthy였다.
- Bridge smoke 완료: plain chat, minimal `json_schema`, `gemini-embedding-2` embeddings smoke는 통과했다.
- Repair metric 확인 완료: `structured_output_repair` redacted metric은 runtime log에 출력된다.
- M7 완료: M7 full runner는 OFF로 유지하고 synthetic Graphiti semantic extraction canary만 실행했다.
- Graphiti 결과: `ollama:qwen3.5:cloud`, `ollama:gemma4:31b-cloud`, `ollama:glm-5.2:cloud` 모두 단일 synthetic episode에서도 timeout/`504 Gateway Timeout` 계열로 실패했다.
- Allowlist 결과: Graphiti semantic extraction 후보 allowlist에 올릴 Ollama Cloud 모델은 아직 없다.

## Open Questions

- 없음.
