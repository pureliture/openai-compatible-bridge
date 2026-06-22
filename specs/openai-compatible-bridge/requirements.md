# OpenAI-Compatible Bridge Requirements

## 승인 대상

- Source of truth: `requirements.md`
- Preview companion: `requirements.html`

## 질문-답변 흐름

### Q: 현재 repo를 새 프로젝트로 분리할지, 기존 repo를 승격할지?

기존 `vertex-ai-api-wrapper` repo를 승격한다.

### Q: `gateway`라는 이름이 이 제품 경계에 맞는가?

핵심 역할이 모든 요청의 정책 관문이 아니라 provider-native API를 OpenAI-compatible API shape으로 연결하고 변환하는 것이므로 `gateway`보다 `bridge`가 더 정확하다.

### Q: 승격 후 이름은 무엇으로 둘지?

프로젝트 이름은 `openai-compatible-bridge`로 둔다.

### Q: 첫 목표 범위는 어디까지인가?

1단계에서는 기존 Vertex 기능을 adapter화하고 이름, 문서, 구조를 `openai-compatible-bridge`로 승격한다. 1단계 완료 후 2단계로 Ollama adapter 추가를 이어서 진행한다.

### Q: 1단계 완료 기준은 어디까지인가?

1단계 완료 기준은 구조, 이름, 문서뿐 아니라 내부 import와 모듈 경계까지 전면 정리하는 것이다. 회귀 위험은 더 크지만, 기존 테스트와 adapter 경계 검증으로 제어한다.

### Q: `vertex-wrapper` legacy alias를 얼마나 남길지?

남기지 않는다. 코드, 문서, Docker service, import path에서 `vertex-wrapper` alias를 유지하지 않고 새 이름과 adapter 경계로 정리한다. 호환성은 OpenAI-compatible HTTP API behavior 기준으로만 유지한다.

### Q: 2단계 Ollama adapter가 처음 지원해야 할 API surface는?

2단계 Ollama adapter는 `/v1/chat/completions`만 지원한다. Embeddings와 rerank는 2단계 범위에서 제외하고, 필요하면 이후 별도 요구사항으로 다룬다.

### Q: provider selection은 어떤 방식으로 할지?

Provider selection은 model alias 중심으로 한다. Client는 OpenAI-compatible 요청의 `model` 필드만 보내고, bridge가 model registry를 통해 provider와 provider-native model id를 결정한다. 별도 `provider` 필드나 header를 client 계약으로 요구하지 않는다.

### Q: 1단계 공통 정책 범위는 어디까지인가?

1단계에서는 기존 기능만 보존한다. 기존 timeout, cost/budget, model registry는 새 `openai-compatible-bridge` 구조로 옮기되, retry와 rate-limit 신규 추가는 1단계 범위에 포함하지 않는다.

## 기능 요구사항

- OpenAI-compatible API를 보는 upstream client는 provider별 HTTP/API 차이를 몰라도 되어야 한다.
- 기존 Vertex 기반 embedding, chat completions, rerank 동작은 1단계에서 유지되어야 한다.
- 기존 `vertex-wrapper` 정체성은 유지하지 않는다. Vertex는 `openai-compatible-bridge` 안의 provider adapter로만 표현한다.
- 코드 import path, Docker service name, 문서 이름에 `vertex-wrapper` legacy alias를 남기지 않는다.
- 1단계는 내부 import와 module naming까지 `openai-compatible-bridge` 방향으로 정리해야 한다.
- 1단계 완료 후 2단계에서 Ollama provider를 `/v1/chat/completions` 대상으로 추가할 수 있어야 한다.
- 2단계 Ollama adapter는 embeddings와 rerank를 지원하지 않는다.
- Provider selection은 model alias 중심이어야 한다.
- Client 요청은 별도 provider 필드나 header 없이 OpenAI-compatible shape을 유지해야 한다.
- 1단계는 기존 timeout, cost/budget, model registry 동작을 새 구조로 보존해야 한다.
- 1단계에서 retry와 rate-limit 신규 기능을 추가하지 않는다.
- retry와 rate-limit는 필요하면 1단계 완료 이후 별도 요구사항으로 다룬다.

## 비기능 요구사항

| 항목 | 요구값 |
| --- | --- |
| 제품 경계 | provider-native 모델 호출을 OpenAI-compatible API shape으로 표준화한다. |
| 품질 책임 분리 | semantic extraction 품질 판단, schema validation, entity/relation quality gate, fallback 정책은 consumer인 neurons 같은 상위 시스템에 둔다. |
| 기존 호환성 | 1단계는 내부 import 전면 정리를 포함하되 기존 OpenAI-compatible HTTP API behavior와 Vertex 동작을 깨지 않아야 한다. |
| Legacy alias | `vertex-wrapper` alias는 유지하지 않는다. |
| 검증 기준 | 기존 테스트 통과와 adapter 경계 검증을 1단계 완료 조건에 포함한다. |
| 단계 진행 | 1단계 완료와 검증 후 2단계 Ollama chat completions adapter로 진행한다. |
| 2단계 제외 범위 | Ollama embeddings와 Ollama rerank는 제외한다. |
| Provider selection | Client가 보낸 `model` alias를 model registry가 provider와 provider-native model id로 resolve한다. |
| 1단계 공통 정책 | 기존 timeout, cost/budget, model registry 보존만 포함한다. |
| 1단계 제외 정책 | retry와 rate-limit 신규 추가는 제외한다. |

## 사용자 시나리오

- neurons 또는 유사한 client는 OpenAI-compatible endpoint 하나만 바라보고 Vertex/Ollama 같은 provider 교체를 bridge 설정으로 처리한다.
- 운영자는 기존 Vertex wrapper를 더 일반적인 `openai-compatible-bridge`로 이해하고, Vertex는 첫 provider adapter로 다룬다.
- 1단계 사용자는 기존 Vertex 기반 embedding/chat/rerank 기능을 회귀 없이 계속 사용한다.
- 2단계 사용자는 Ollama를 추가 provider로 연결해 local model 호출을 같은 OpenAI-compatible 표면으로 사용할 수 있다.

## 확정 항목

- 현재 repo를 `openai-compatible-bridge`로 승격한다.
- `gateway`가 아니라 `bridge` 이름을 사용한다.
- 1단계는 Vertex adapter화, 이름/문서/구조 승격, 내부 import 전면 정리를 포함한다.
- `vertex-wrapper` legacy alias는 남기지 않는다.
- 1단계 완료 후 2단계로 Ollama chat completions adapter 추가를 진행한다.
- 2단계에서 Ollama embeddings와 rerank는 제외한다.
- Provider selection은 model alias 중심으로 한다.
- 1단계 공통 정책은 기존 timeout, cost/budget, model registry 보존만 포함한다.
- retry와 rate-limit 신규 추가는 1단계에서 제외한다.

## 미결정 항목

- 없음.
