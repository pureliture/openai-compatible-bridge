# Dynamic Ollama Model Routing and Response Normalization Requirements

## 승인 대상

- Source of truth: `requirements.md`
- Preview companion: `requirements.html`

## 질문-답변 흐름

### Q: 동적 모델 지정의 제품 경계는 어디까지인가?

사용자는 **Ollama 전용 동적 모델 지정**을 선택했다.

선택지는 다음과 같다.

- Ollama 전용 동적 모델: registry/env/code alias 없이 호출마다 Ollama native model을 지정한다. 기존 Vertex와 registered alias 동작은 유지한다.
- 모든 provider 동적 모델: Vertex/Ollama 모두 호출마다 native model을 지정한다. 유연하지만 allowlist, cost, 인증, region 책임이 커진다.
- 동적 지정 없음: registry/env/code alias 체계를 유지하고, provider 응답 정규화만 추가한다.

확정된 범위는 Ollama 전용 동적 모델이다. Vertex와 다른 provider의 동적 native model 지정은 이번 범위에서 제외한다.

### Q: 호출별 Ollama native model은 어떻게 표현하는가?

사용자는 `model: "ollama:<native-model>"` 형태를 선택했다.

예시는 다음과 같다.

- `model: "ollama:minimax-m3:cloud"`
- `model: "ollama:deepseek-v4-pro:cloud"`

이 형태는 OpenAI-compatible client가 이미 지원하는 `model` field만 사용한다. 별도 `provider_model` request field는 이번 범위에서 제외한다.

### Q: `<think>...</think>` 안의 reasoning text는 어떻게 처리하는가?

사용자는 reasoning text를 버리는 정책을 선택했다.

`<think>...</think>` block은 OpenAI-compatible response의 `message.content`와 streaming `delta.content`에서 제거한다. 제거된 reasoning text는 별도 response field로 노출하지 않는다.

### Q: 동적 모델 지정 실패는 어떻게 처리하는가?

사용자-facing 추가 결정을 요구하지 않는 보수 기본값으로 닫는다.

- `model` 값이 `ollama:`처럼 native model 없이 비어 있으면 provider 호출 전 HTTP 400 `invalid_request_error`로 거절한다.
- `ollama:<native-model>` 형식은 유효하지만 Ollama가 모델을 찾지 못하거나 실행하지 못하면 기존 upstream error mapping을 유지한다.
- `/v1/models`는 동적 infinite namespace를 열거하지 않는다. Registry에 등록된 모델만 반환한다.

## 기능 요구사항

- `/v1/chat/completions` 호출에서 registry에 없는 모델이라도 승인된 동적 모델 규칙에 해당하면 provider 호출 전 404로 거절하지 않아야 한다.
- `model` 값이 `ollama:<native-model>` 형태이면 bridge는 `<native-model>`을 Ollama native model로 사용해야 한다.
- 동적 Ollama 모델 지정은 기존 registered alias 경로와 공존해야 한다.
- 기존 `MODEL_REGISTRY_JSON` 기반 Ollama alias, Vertex chat, embeddings, rerank, `/v1/models`, cost tracking 동작은 회귀하지 않아야 한다.
- Ollama 응답 본문에 provider-specific reasoning wrapper가 포함되면 OpenAI-compatible `message.content` 또는 streaming `delta.content`에서 정규화해야 한다.
- 최소 정규화 대상은 `<think>...</think>` reasoning block 제거다.
- non-stream 응답과 streaming 응답 모두 정규화 대상이다.
- 정규화로 제거된 reasoning text를 기본 response content에 섞거나 별도 response field로 노출하지 않아야 한다.
- 모델이 plain content만 반환하는 경우 정규화가 내용을 변경하지 않아야 한다.
- `ollama:`처럼 native model이 비어 있는 동적 모델 지정은 provider 호출 전 HTTP 400으로 실패해야 한다.
- `/v1/models`는 동적 Ollama model namespace를 열거하지 않고 기존 registry model 목록만 반환해야 한다.
- production 배포 후 live request로 동적 Ollama 모델 지정과 `<think>` 정규화가 실제 동작함을 검증해야 한다.

## 비기능 요구사항

| 항목 | 요구값 |
| --- | --- |
| 호환성 | 기존 OpenAI-compatible request/response shape를 유지한다. |
| 안전성 | 동적 모델 지정은 명확히 구분되는 provider boundary를 가져야 한다. |
| 회귀 방지 | unit/integration test가 registry alias, dynamic route, non-stream normalization, stream normalization을 모두 커버한다. |
| 배포 검증 | local test만으로 끝내지 않고 production container/runtime에서 smoke evidence를 확보한다. |
| 비용 추적 | cost tracking이 켜진 경우 기존 fail-closed 원칙을 약화하지 않는다. |

## 사용자 시나리오

- 사용자는 bridge code나 environment registry를 바꾸지 않고 요청마다 Ollama cloud model을 바꿔 실험한다.
- 사용자는 `Minimax-M3:cloud` 계열 응답에서 `<think>` block이 섞여도 downstream client에는 정리된 content만 전달되길 기대한다.
- 사용자는 기존 registered alias를 계속 사용하면서, 필요할 때만 동적 Ollama 모델 지정 경로를 사용한다.
- 운영자는 배포 후 `/v1/chat/completions` live smoke로 동적 모델 route와 정규화 결과를 확인한다.

## 미결정 항목

- 없음.
