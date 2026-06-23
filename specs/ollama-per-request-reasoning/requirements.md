# Ollama Per-Request Reasoning Requirements

## 승인 대상

- Source of truth: `requirements.md`
- Preview companion: `requirements.html`
- 승인 상태: 사용자가 이번 턴에서 자문자답 및 승인 진행을 위임함.

## 질문-답변 흐름

### Q: 호출별 reasoning 제어는 어떤 사용자 경험이어야 하는가?

운영자가 bridge를 재기동하지 않고 같은 Ollama dynamic model에 대해 요청마다 reasoning 강도를 조정할 수 있어야 한다. 기본값은 기존 `OLLAMA_THINK` env 설정을 유지하되, OpenAI-compatible 요청 필드가 들어오면 그 요청에만 적용한다.

선택지:

- 호출별 override 유지: env는 default, 요청 필드는 one-shot override가 된다.
- Ollama OpenAI endpoint 전환: `/api/chat` 대신 `/v1/chat/completions`를 사용한다.
- env-only 유지: 운영자가 재기동으로만 제어한다.

결정: 호출별 override 유지. 현재 bridge의 dynamic model, schema forwarding, streaming normalize 경로를 최대한 보존하면서 Ollama 공식 OpenAI-compatible reasoning 필드와 호환된다.

### Q: reasoning을 끌 수 있어야 하는가?

기본 운영 목적은 reasoning을 켜는 것이다. 다만 Ollama 공식 OpenAI-compatible contract가 `none`을 지원하므로, client가 명시적으로 `none`을 보낸 경우에는 요청별로 `think=false`로 전달한다. bridge가 임의로 reasoning을 끄지는 않는다.

### Q: invalid reasoning 값은 어떻게 처리해야 하는가?

잘못된 값은 upstream으로 보내지 않고 HTTP 400 `invalid_request_error`로 거부한다. 이 정책은 fail-open이 아니라 명시적 contract enforcement다.

## 기능 요구사항

- `/v1/chat/completions` 요청의 `reasoning_effort` 값을 Ollama provider에 전달한다.
- `/v1/chat/completions` 요청의 `reasoning.effort` 값을 Ollama provider에 전달한다.
- `reasoning.effort`가 있으면 `reasoning_effort`보다 우선한다.
- 허용값은 `high`, `medium`, `low`, `none`이다.
- `high`, `medium`, `low`는 Ollama native `/api/chat`의 `think` 문자열로 전달한다.
- `none`은 Ollama native `/api/chat`의 `think=false`로 전달한다.
- 요청별 reasoning 필드가 없으면 기존 `OLLAMA_THINK` env-derived default를 그대로 사용한다.
- dynamic `ollama:<native>` model routing, alias routing, `response_format` schema forwarding, think-block normalization은 깨지지 않아야 한다.
- streaming과 non-streaming 모두 동일한 reasoning override 규칙을 적용한다.

## 비기능 요구사항

| 항목 | 요구값 |
| --- | --- |
| 호환성 | Ollama official OpenAI compatibility의 `reasoning_effort`와 `reasoning.effort` field를 수용 |
| 안전성 | invalid reasoning 값은 upstream 호출 전 400 |
| 운영성 | runtime env 변경이나 bridge 재기동 없이 요청별 조정 가능 |
| 보안 | raw prompt, raw completion, secret, private path를 로그/에러에 출력하지 않음 |
| 회귀 방지 | 기존 alias 방식과 dynamic Ollama 방식 유지 |

## 사용자 시나리오

- 운영자가 qwen/glm canary에서 기본 reasoning은 유지하면서 특정 요청만 `reasoning_effort=medium`으로 낮춰 timeout과 thinking-only truncation을 비교한다.
- Graphiti semantic extraction smoke가 `response_format.json_schema`와 `reasoning_effort`를 함께 보내도 Ollama native request에는 schema와 think level이 모두 보존된다.
- 잘못된 reasoning 값이 들어와도 Ollama까지 전달되지 않고 bridge가 400으로 빠르게 차단한다.

## 미결정 항목

- 없음.
