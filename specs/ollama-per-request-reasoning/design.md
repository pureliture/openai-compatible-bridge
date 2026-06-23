# Ollama Per-Request Reasoning Design Spec

## Overview

Ollama dynamic chat path에 OpenAI-compatible per-request reasoning control을 추가한다. 기존 env default, native `/api/chat` forwarding, structured output mapping, reasoning normalization은 유지한다.

## Requirements Reference

- Phase 1 source: `requirements.md`
- Preview companion: `requirements.html`
- 핵심 기능: `reasoning_effort`와 `reasoning.effort`를 요청별로 받아 Ollama native `think`로 변환한다.

## Approach Proposal

추천안: 현재 native `/api/chat` adapter를 유지하고 요청별 reasoning 필드만 `think`로 매핑한다.

이유: 이미 dynamic model id, `response_format.json_schema` to `format` schema, streaming normalization, cost tracking이 이 경로에 붙어 있다. endpoint 전체를 `/v1/chat/completions`로 바꾸면 response normalization과 streaming cost finalization의 blast radius가 커진다.

대안 1: Ollama `/v1/chat/completions`로 provider를 전환한다.

장점은 공식 OpenAI-compatible 필드를 그대로 전달한다는 점이다. 단점은 bridge가 다시 OpenAI-compatible response를 OpenAI-compatible response로 재정규화해야 해서 기존 error/cost/streaming contract가 흔들린다.

대안 2: env-only를 유지한다.

장점은 변경이 없다는 점이다. 단점은 사용자가 지적한 운영 문제, 즉 reasoning level 조정 때 재기동이 필요한 문제가 남는다.

## Architecture

```text
OpenAI client
  -> FastAPI /v1/chat/completions
  -> OpenAIChatRequest(reasoning_effort, reasoning)
  -> OllamaChatClient.generate/stream_chat
  -> _ollama_think_from_reasoning()
  -> Ollama native /api/chat { model, messages, stream, think, format, options }
```

## Data Flow

1. Client가 `reasoning_effort` 또는 `reasoning.effort`를 보낸다.
2. FastAPI request model이 해당 필드를 보존한다.
3. Chat route가 Ollama provider 호출 인자에 reasoning 필드를 포함한다.
4. Ollama provider가 `reasoning.effort`를 우선하고, 없으면 `reasoning_effort`, 둘 다 없으면 `OLLAMA_THINK` default를 사용한다.
5. Provider가 native body의 `think`를 설정한다. 값이 `None`이면 field를 생략한다.
6. invalid 값은 upstream HTTP 호출 전에 `VertexAPIError(400, code="invalid_request")`로 중단한다.

## Component Details

### OpenAIChatRequest

- 입력: `reasoning_effort: str | None`, `reasoning: dict[str, Any] | None`
- 출력: route layer에서 provider kwargs로 전달
- 의존성: Pydantic request parsing

### OllamaChatClient

- 입력: existing chat args plus `reasoning_effort`, `reasoning`
- 출력: Ollama native request body
- 의존성: `OLLAMA_THINK` default, `_ollama_think_from_env`

### Reasoning Mapper

- 입력: request-level fields and env default
- 출력: `bool | str | None`
- 규칙: `high/medium/low` 문자열, `none` -> `False`, missing -> default

## Error Handling

- invalid `reasoning_effort`: HTTP 400 `invalid_request_error`, upstream not called.
- invalid `reasoning.effort`: HTTP 400 `invalid_request_error`, upstream not called.
- malformed `reasoning` non-object: HTTP 400 `invalid_request_error`, upstream not called.
- existing upstream errors, timeout, empty visible content after reasoning normalization은 현재 정책 유지.

## Testing Strategy

- RED: app-level dynamic Ollama request with `reasoning_effort=medium` must capture native `think="medium"`.
- RED: app-level dynamic Ollama request with `reasoning.effort=low` must capture native `think="low"`.
- RED: provider streaming request with `reasoning_effort=none` must capture native `think=false`.
- RED: invalid reasoning value must return 400 and make no upstream call.
- Regression: existing Ollama alias, dynamic model, `json_schema` format, `json_object`, thinking-only error tests still pass.

## Milestones

- M1: Spec and RED tests — 요구사항/설계 산출물 작성, 실패 테스트 확인.
- M2: Minimal implementation — request model, route forwarding, provider mapper를 구현해 RED를 GREEN으로 전환.
- M3: Verification and review — targeted/full tests, multi-agent review, docs update, git state 확인.

## Open Questions

- 없음.
