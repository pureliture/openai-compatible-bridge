# Dynamic Ollama Model Routing and Response Normalization Design Spec

## Overview

`/v1/chat/completions`에서 `model: "ollama:<native-model>"`을 Ollama 전용 dynamic route로 해석한다. Ollama adapter는 non-stream과 stream 응답 모두에서 `<think>...</think>` reasoning block을 제거하고, 제거된 reasoning text는 response에 노출하지 않는다.

## Requirements Reference

- Phase 1 source: `requirements.md`
- Preview companion: `requirements.html`

핵심 요구사항은 다음과 같다.

- 동적 모델 지정 범위는 Ollama 전용이다.
- user-facing 호출 형태는 `model: "ollama:<native-model>"`이다.
- `ollama:`처럼 native model이 비어 있으면 provider 호출 전 HTTP 400으로 거절한다.
- `/v1/models`는 dynamic namespace를 열거하지 않고 registry model만 반환한다.
- `<think>...</think>` reasoning block은 non-stream `message.content`와 streaming `delta.content`에서 제거한다.
- 제거된 reasoning text는 버리고 별도 response field로 노출하지 않는다.
- 기존 registry alias, Vertex chat, embeddings, rerank, cost tracking 동작은 회귀하지 않는다.
- production 배포 후 live smoke로 dynamic route와 normalization을 검증한다.

## Approach

선택한 접근은 **route에서 dynamic model resolve + Ollama adapter에서 normalize**다.

- `main.py`의 `/v1/chat/completions` route는 기존 registry lookup 전에 `ollama:` prefix를 인식한다.
- dynamic model이면 provider를 `ollama`로 고정하고 provider-native model은 prefix를 제거한 값으로 둔다.
- dynamic model이 아니면 기존 registry 기반 alias resolution을 그대로 사용한다.
- `<think>` 제거는 `openai_compatible_bridge/providers/ollama.py`의 adapter boundary에서 수행한다.

이 접근은 registry의 의미를 흐리지 않고, provider-specific response cleanup을 provider adapter에 둔다.

## Architecture

```text
OpenAI client
  -> POST /v1/chat/completions
    -> auth check
    -> chat model resolution
      -> dynamic Ollama model: model starts with "ollama:"
      -> registered alias: model_config(model)
    -> response_format validation
    -> cost preflight
    -> provider client dispatch
      -> OllamaChatClient.generate / stream_chat
        -> Ollama /api/chat
        -> strip <think> reasoning block
    -> OpenAI-compatible response
```

## Data Flow

### Non-Stream

```text
payload.model = "ollama:minimax-m3:cloud"
-> resolve provider="ollama", provider_model="minimax-m3:cloud"
-> call OllamaChatClient.generate(model="minimax-m3:cloud")
-> receive content possibly containing "<think>...</think>answer"
-> normalize content to "answer"
-> return ChatCompletion with model="ollama:minimax-m3:cloud"
```

### Stream

```text
payload.model = "ollama:minimax-m3:cloud", stream=true
-> resolve provider="ollama", provider_model="minimax-m3:cloud"
-> call OllamaChatClient.stream_chat(model="minimax-m3:cloud")
-> feed text chunks through stateful think-block stripper
-> emit only visible delta_text
-> finish and usage events remain unchanged
```

## Component Details

### Chat Model Resolution

Purpose:

- Decide whether `payload.model` is a dynamic Ollama model or an existing registry alias.

Behavior:

- If model is exactly `ollama:` or has only whitespace after the prefix, return HTTP 400 `invalid_request_error` with `param="model"`.
- If model starts with `ollama:` and has a non-empty native model, return:
  - `provider="ollama"`
  - `provider_model=<native-model>`
  - `kind="chat"`
  - `resolved_config=None`
- Otherwise, require the model to exist in `current_allowed_models()` and use existing `model_config()`.

Dependencies:

- Existing `openai_error_response()`
- Existing `current_allowed_models()`
- Existing `model_config()`

### Ollama Response Normalizer

Purpose:

- Remove provider-specific `<think>...</think>` reasoning blocks from visible content.

Behavior:

- Non-stream content uses a complete-string normalization helper.
- Streaming content uses a small stateful normalizer because `<think>` and `</think>` can cross chunk boundaries.
- Text outside think blocks is preserved.
- Text inside think blocks is discarded.
- Unclosed trailing `<think>` content is discarded.
- Plain content without think tags is unchanged.

Dependencies:

- No new runtime dependency.
- Python string/state handling only.

### Cost Tracking

Purpose:

- Preserve existing fail-closed cost behavior.

Behavior:

- Cost preflight keeps using the user-facing `payload.model`.
- If cost tracking is enabled and pricing for `ollama:<native-model>` is absent, existing missing-pricing behavior applies.
- No automatic pricing fallback from native model to another alias is added.

### `/v1/models`

Purpose:

- Preserve registry list semantics.

Behavior:

- Dynamic Ollama model namespace is not listed.
- `/v1/models/{model_id}` remains registry-only.
- Dynamic models are accepted only at `/v1/chat/completions`.

## Error Handling

- `model="ollama:"`: HTTP 400 `invalid_request_error`, `code="invalid_model"`, `param="model"`.
- `model="ollama:   "`: HTTP 400 `invalid_request_error`, `code="invalid_model"`, `param="model"`.
- `model="unknown"`: existing HTTP 404 `model_not_found`.
- `model="ollama:<native>"` where Ollama returns 404 or other error: existing upstream `VertexAPIError` mapping.
- Unsupported `response_format.type`: existing HTTP 400 behavior.
- Streaming provider error after stream start: existing SSE error chunk behavior remains.

## Testing Strategy

Targeted tests:

- `model="ollama:minimax-m3:cloud"` bypasses registry 404 and routes to Ollama client with model `minimax-m3:cloud`.
- `model="ollama:"` returns HTTP 400 before provider dispatch.
- Registry alias Ollama route still works.
- Unknown non-dynamic model still returns HTTP 404.
- `/v1/models` does not include dynamic Ollama examples unless they are registered aliases.
- Non-stream Ollama response removes `<think>...</think>` and keeps visible text.
- Non-stream plain text remains unchanged.
- Streaming Ollama response removes think blocks when tags and content are split across chunks.
- Streaming plain deltas remain unchanged.
- Full `uv run pytest` passes.

Production verification:

- Deploy updated bridge image/config to the existing production runtime.
- Smoke `/v1/chat/completions` with `model="ollama:minimax-m3:cloud"` or another available Ollama cloud model and confirm the request reaches Ollama as the native model.
- Smoke a controlled response path or live model response that contains `<think>` and confirm the returned OpenAI-compatible content excludes `<think>` and reasoning text.
- Confirm existing registered alias still responds after deploy.

## Milestones

- M1: Requirements and design approval — `requirements.md` and `design.md` are approved.
- M2: Local implementation — dynamic Ollama route and normalizer are implemented with focused tests.
- M3: Regression verification — full `uv run pytest` passes.
- M4: Production deploy — updated bridge is deployed to the production runtime.
- M5: Production live verification — dynamic Ollama route, `<think>` normalization, and existing alias regression smoke pass.

## Open Questions

- 없음.
