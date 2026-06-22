# OpenAI-Compatible Bridge Design Spec

## Overview

`vertex-ai-api-wrapper`를 `openai-compatible-bridge`로 승격한다. 기존 Vertex 기반 OpenAI-compatible HTTP behavior를 보존하면서 내부 구조를 provider adapter 기반으로 전면 정리하고, 이후 같은 단일 실행 goal 안에서 Ollama chat completions adapter를 추가한다.

이 설계는 구현자가 `agentic-execution` 루프에서 하나의 장기 goal로 처리할 수 있도록 작성한다. 요구사항이나 설계 변경이 필요해지는 경우에는 구현 루프 안에서 SoT를 수정하지 않고 `grill-to-spec`으로 회귀한다.

단일 goal은 하나의 실행 흐름을 뜻하지만, 단계 경계를 흐리지 않는다. M1-M4 검증이 완료되기 전에는 M5 Ollama 구현을 시작하지 않는다.

## Requirements Reference

- Source of truth: `requirements.md`
- Preview companion: `requirements.html`
- Phase 1/Phase 2 실행 경계는 `requirements.md`와 이 `design.md`의 milestones 안에 정의한다.
- 승인된 핵심 요구사항:
  - 프로젝트 이름은 `openai-compatible-bridge`다.
  - `gateway`가 아니라 `bridge`다.
  - `vertex-wrapper` legacy alias는 남기지 않는다.
  - 1단계는 Vertex adapter화, 이름/문서/구조 승격, 내부 import 전면 정리를 포함한다.
  - 1단계는 기존 timeout, cost/budget, model registry 보존만 포함하고 retry/rate-limit 신규 추가는 제외한다.
  - Provider selection은 model alias 중심이다.
  - 2단계는 Ollama `/v1/chat/completions`만 추가하고 Ollama embeddings/rerank는 제외한다.

## Approach Proposal

### 선택안 A: 단일 장기 goal 안에서 단계형 refactor와 Ollama 추가

1단계에서 repo identity와 내부 module boundary를 `openai-compatible-bridge`로 정리하고, 2단계에서 Ollama chat completions adapter를 추가한다. 하나의 branch/worktree와 하나의 agentic-execution goal에서 milestone 단위로 act/observe/adjust를 반복한다.

이 설계의 선택안이다. 요구사항의 “1단계 완료 후 2단계 진행”을 유지하면서도 사용자 요청대로 단일 goal로 장기 실행할 수 있다.

### 대안 B: 1단계와 2단계를 별도 specs/branches로 분리

회귀 위험은 낮지만 사용자가 요청한 단일 장기 goal과 맞지 않는다. 1단계 완료 후 다시 planning overhead가 생긴다.

### 대안 C: 얇은 rename 후 Ollama를 빠르게 붙이기

빠르지만 내부 import 전면 정리 요구사항을 충족하지 못한다. `bridge` 제품 경계가 코드 구조에 충분히 반영되지 않는다.

## Agentic Execution Goal

단일 goal:

```text
현재 repo를 openai-compatible-bridge로 승격하고, Vertex 기능을 provider adapter 구조로 보존한 뒤, 같은 OpenAI-compatible /v1/chat/completions 표면에 Ollama chat adapter를 추가한다.
```

Goal 완료 조건:

- `main` 또는 `master`가 아닌 전용 branch/worktree에서만 작업한다.
- 제품/패키지/문서/Docker 표면에서 `vertex-wrapper` alias를 남기지 않는다.
- 기존 repo/product 이름인 `vertex-ai-api-wrapper`와 Docker service 이름 `wrapper-vertex-ai-api`도 새 제품 표면의 alias로 남기지 않는다.
- 기존 `/v1/models`, `/v1/embeddings`, `/v1/chat/completions`, `/v1/rerank`, `/healthz`, cost admin endpoints의 behavior가 의도 없이 깨지지 않는다.
- Provider selection은 `model` alias가 model registry를 통해 provider와 provider-native model id로 resolve되는 방식이다.
- Vertex adapter는 기존 embeddings, chat, rerank 기능을 보존한다.
- Ollama adapter는 `/v1/chat/completions`만 지원한다.
- Ollama embeddings/rerank, retry 신규 추가, rate-limit 신규 추가는 구현하지 않는다.
- 모든 milestone은 구체 증거를 남기고 완료된다.

## Architecture

```text
openai-compatible-bridge
  FastAPI app
    -> OpenAI-compatible routes
       -> model registry
          -> provider adapter
             -> vertex adapter
             -> ollama adapter
       -> cost/budget accounting
```

권장 Python package layout:

```text
openai_compatible_bridge/
  __init__.py
  main.py
  api/
    __init__.py
    errors.py
    schemas.py
    routes.py
  core/
    __init__.py
    config.py
    model_registry.py
    cost_tracking.py
  providers/
    __init__.py
    base.py
    vertex/
      __init__.py
      auth.py
      chat.py
      embeddings.py
      rerank.py
    ollama/
      __init__.py
      chat.py
```

Entrypoint:

```text
openai_compatible_bridge.main:app
```

`app.py`와 `vertex.py` 같은 root-level implementation modules는 1단계 완료 시점에 제거하거나 새 package 내부로 이동한다. Legacy import alias는 남기지 않는다.

마이그레이션 순서상 root-level implementation modules를 먼저 제거하지 않는다. 먼저 package shell, app factory/provider injection seam, registry seam을 만든 뒤 기존 route behavior tests가 통과하는 지점에서 제거한다.

## Data Flow

### Vertex Embeddings

```text
POST /v1/embeddings
  -> validate OpenAIEmbeddingsRequest
  -> resolve model alias in model registry
  -> require provider=vertex, kind=embedding
  -> cost preflight
  -> Vertex embeddings adapter
  -> normalize to OpenAI embeddings response
  -> finalize cost
```

### Vertex Chat Completions

```text
POST /v1/chat/completions
  -> validate OpenAIChatRequest
  -> resolve model alias in model registry
  -> require kind=chat
  -> cost preflight
  -> provider adapter selected by registry
  -> normalize to OpenAI chat completion or SSE stream
  -> finalize cost
```

### Vertex Rerank

```text
POST /v1/rerank
  -> validate CohereRerankRequest
  -> resolve model alias in model registry
  -> require provider=vertex, kind=rerank
  -> cost preflight
  -> Vertex rerank adapter
  -> normalize to rerank results
  -> finalize cost
```

### Ollama Chat Completions

```text
POST /v1/chat/completions
  -> validate OpenAIChatRequest
  -> resolve model alias in model registry
  -> require provider=ollama, kind=chat
  -> cost preflight according to existing cost subsystem rules
  -> Ollama native chat adapter
  -> normalize to OpenAI chat completion or SSE stream
```

## Component Details

### FastAPI App

- Input: HTTP requests matching current OpenAI-compatible endpoints.
- Output: OpenAI-compatible JSON/SSE responses and current admin responses.
- Depends on: route modules, model registry, provider adapters, cost accounting.
- Notes: The app title/version/docs should use `openai-compatible-bridge`.
- Notes: An app factory or equivalent provider injection seam should exist before root modules are removed, so tests can replace provider clients without relying on stale global state.

### API Schemas

- Input: OpenAI-compatible request bodies.
- Output: typed request models and normalized response helpers.
- Depends on: Pydantic.
- Notes: Current request tolerance using `extra="allow"` should be preserved unless a test proves it is unsafe.
- Notes: Bridge schema validation is limited to HTTP/OpenAI-compatible request and response shape. Semantic output schema validation remains outside the bridge.

### Model Registry

- Input: built-in model entries plus configured JSON/env overrides.
- Output: resolved model config with at least `provider`, `kind`, and `provider_model`.
- Depends on: environment/config parser.
- Notes:
  - Current `api`-based Vertex registry entries migrate to provider-based entries before Ollama entries are accepted.
  - Existing Vertex entries map to `provider=vertex`.
  - Ollama entries use `provider=ollama`, `kind=chat`, `provider_model=<ollama model name>`.
  - Client-facing provider selection remains `model` alias only.
  - Existing `MODEL_REGISTRY_JSON`, `EXTRA_MODELS`, `openapi_model`, `vertex_model`, and `thinking_budget` behavior must be preserved or explicitly covered by migration tests.

### Provider Adapter Base

- Input: normalized route-level request data.
- Output: normalized provider result objects for route serialization.
- Depends on: no concrete provider.
- Notes: The base boundary should be small. Do not create a generic framework for endpoints not in scope.

### Vertex Adapter

- Input: resolved Vertex model config and normalized request data.
- Output: embeddings, chat, rerank normalized provider results.
- Depends on: Google ADC/OAuth, `httpx`, Vertex endpoint shapes.
- Notes:
  - Existing auth, batching, stream parsing, response_format handling, rerank behavior, and error mapping must be preserved.
  - Vertex-specific env vars such as `VERTEX_PROJECT`, `VERTEX_LOCATION`, `VERTEX_TASK_TYPE_DEFAULT`, and `VERTEX_AUTO_TRUNCATE` remain valid provider config.

### Ollama Adapter

- Input: resolved Ollama chat model config and normalized chat request.
- Output: chat completion normalized provider result, with streaming support if `stream=true` is requested.
- Depends on: Ollama native API base URL.
- Notes:
  - Scope is chat completions only.
  - Streaming chat is part of Phase 2 acceptance when `stream=true`; if implementation discovery shows this cannot be safely normalized, stop and return to `grill-to-spec` instead of weakening acceptance inside the execution loop.
  - No embeddings or rerank methods are added for Ollama.
  - Config should include an Ollama base URL, for example `OLLAMA_BASE_URL`.
  - Provider-native model name comes from `provider_model`.

### Cost/Budget Accounting

- Input: endpoint, model alias, forecast usage, provider result usage.
- Output: allow/block decision and ledger events.
- Depends on: current SQLite ledger and pricing config.
- Notes:
  - Preserve existing behavior.
  - Do not add new pricing source behavior in 1단계.
  - For Ollama, local models can be priced only if explicitly present in pricing config.
  - When cost tracking is enabled, missing Ollama pricing follows the existing fail-closed pricing behavior.

## Error Handling

- Unknown model alias returns current OpenAI-compatible `model_not_found` style response.
- Wrong model kind for an endpoint returns current `invalid_model` style response.
- Provider HTTP timeout and connection errors map to current upstream error shapes.
- Malformed provider responses map to `502` `api_error` / `bad_gateway` style responses.
- Cost subsystem config errors preserve current fail-closed behavior.
- Ollama unavailable maps to provider connection error without fallback to Vertex.
- No semantic quality fallback is implemented in this bridge.
- Any implementation discovery that requires changing approved requirements or this design stops execution and returns to `grill-to-spec`.

## Testing Strategy

- Run the existing full suite with `uv run pytest` before and after refactor milestones.
- Add unit tests for model registry resolution:
  - Vertex model alias resolves to provider `vertex`.
  - Ollama model alias resolves to provider `ollama`.
  - Unknown model remains rejected.
  - Endpoint kind mismatch remains rejected.
  - Existing `MODEL_REGISTRY_JSON`, `EXTRA_MODELS`, `openapi_model`, `vertex_model`, and `thinking_budget` behavior remains covered through migration.
- Add route tests proving existing Vertex endpoint behavior remains compatible through the new package layout.
- Add adapter tests for Ollama chat:
  - non-stream response normalization;
  - stream response normalization when `stream=true`;
  - connection error mapping;
  - auth rejection before provider dispatch;
  - cost success/finalization behavior;
  - missing pricing fail-closed behavior when cost tracking is enabled;
  - `provider_model` mapping from registry.
- Add Docker/docs sanity checks where practical:
  - service/container names use `openai-compatible-bridge`;
  - Docker entrypoint uses `openai_compatible_bridge.main:app`;
  - docs do not describe the product as `vertex-wrapper`;
  - command examples use `openai_compatible_bridge.main:app`.

## Milestones

### M1: Spec SoT and worktree setup

Done when:

- Implementation work happens in a dedicated non-main worktree.
- `requirements.md` and approved `design.md` are present and referenced.
- Baseline `uv run pytest` result is captured.

Evidence:

- `git status --short --branch`
- `uv run pytest`

### M2: Package shell and app factory seam

Done when:

- Python package `openai_compatible_bridge` exists.
- FastAPI app loads from `openai_compatible_bridge.main:app`.
- An app factory or equivalent provider injection seam exists for tests.
- Root implementation modules may still exist temporarily as migration scaffolding, but new package entrypoint works.
- `pyproject.toml` can still use `package = false` if the app runs by module path, but Docker and local commands must point at the new package entrypoint by the end of M6.

Evidence:

- `uv run pytest`
- import smoke for `openai_compatible_bridge.main:app`
- test smoke proving provider clients can be injected/replaced without relying on stale module globals

### M3: Model registry migration and provider boundary

Done when:

- Current `api`-based registry behavior migrates to provider-based registry behavior without import-time failures.
- Model registry resolves model aliases to `provider`, `kind`, and `provider_model`.
- Existing Vertex models are represented as Vertex provider entries.
- Route handlers choose provider adapters through registry resolution, not direct Vertex coupling.
- Existing `MODEL_REGISTRY_JSON`, `EXTRA_MODELS`, `openapi_model`, `vertex_model`, and `thinking_budget` behavior is covered by tests or intentionally migrated with documented equivalent behavior.

Evidence:

- targeted registry tests
- migration tests for existing registry override behavior
- route tests for model kind mismatch and unknown model rejection
- `uv run pytest`

### M4: Vertex adapter preservation

Done when:

- Existing Vertex embeddings, chat, streaming chat, response_format, OpenAI-compatible Agent Platform chat, and rerank behavior pass through the adapter boundary.
- Existing cost/budget behavior still wraps billable endpoints.
- No retry or rate-limit feature is newly introduced.
- Root-level `app.py` and `vertex.py` implementation code is removed or reduced to non-production migration residue only after route behavior tests pass; no legacy import alias remains at M4 exit.

Evidence:

- existing tests migrated and passing
- adapter-focused tests covering current Vertex behavior
- import smoke using only `openai_compatible_bridge.main:app`
- `uv run pytest`

### M5: Ollama chat completions adapter

Done when:

- Ollama model aliases with `provider=ollama`, `kind=chat` route to Ollama chat adapter.
- `/v1/chat/completions` supports Ollama non-stream responses.
- `/v1/chat/completions` supports Ollama streaming responses when `stream=true`.
- Ollama embeddings and rerank are absent and rejected by model kind/provider scope.
- Existing auth behavior applies before Ollama provider dispatch.
- Existing cost preflight/finalization semantics apply to Ollama chat when cost tracking is enabled.

Evidence:

- Ollama adapter unit tests with mocked HTTP responses
- route tests for Ollama chat alias
- non-stream and stream cost success tests
- missing pricing fail-closed test when cost tracking is enabled
- auth rejection test before provider dispatch
- connection error mapping test
- negative tests for Ollama embeddings/rerank scope
- `uv run pytest`

### M6: Documentation and migration cleanup

Done when:

- README explains `openai-compatible-bridge`, provider adapters, model alias selection, Vertex config, Ollama chat config, and excluded scope.
- `.env.example` and Docker Compose reflect new bridge names and no `vertex-wrapper` aliases.
- Dockerfile copies `openai_compatible_bridge/` and starts `uvicorn openai_compatible_bridge.main:app`.
- Docker Compose service/container names use `openai-compatible-bridge`; no `wrapper-vertex-ai-api` DNS alias is retained.
- Product-level env names are explicit: common bridge auth uses `BRIDGE_API_KEY`, while Vertex-specific env vars remain `VERTEX_*`.
- `WRAPPER_API_KEY` is not retained as a fallback alias; operators must update env and client/DNS config during migration.
- Product-facing docs avoid stale `vertex-ai-api-wrapper` and `wrapper-vertex-ai-api` naming except where explicitly explaining historical migration context.
- Architecture assets or text no longer present the repo as Vertex-only.

Evidence:

- docs grep for stale product names
- config grep for `WRAPPER_API_KEY`, `wrapper-vertex-ai-api`, and old product title
- Docker build or import/run smoke proving the new entrypoint loads
- `uv run pytest`
- manual README scan

### M7: Final verification

Done when:

- Full test suite passes.
- `git status --short --branch` shows only intended files.
- The implementation notes include milestone evidence and any divergence notes.
- No SoT changes were made during implementation; if SoT changes were needed, the loop stopped and returned to `grill-to-spec`.

Evidence:

- final `uv run pytest`
- final `git status --short --branch`
- milestone evidence summary

## Open Questions

- 없음.
