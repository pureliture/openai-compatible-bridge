# openai-compatible-bridge

OpenAI-compatible API shape을 유지하면서 provider-native model API를 연결하는 private bridge입니다. Client는 `/v1/...` endpoint와 `model` alias만 보고, bridge가 model registry를 통해 provider와 provider-native model id를 결정합니다.

> Public internet에 노출하지 마세요. Vertex service account, local model endpoint, cost ledger 같은 운영 자원이 연결되므로 local machine 또는 private Docker network 안에서만 운용합니다.

## 지원 범위

| Endpoint | Provider | 상태 |
| --- | --- | --- |
| `/v1/embeddings` | Vertex | 지원 |
| `/v1/chat/completions` | Vertex | 지원 |
| `/v1/chat/completions` | Ollama | 지원 |
| `/v1/rerank` | Vertex | 지원 |

Ollama embeddings와 Ollama rerank는 현재 범위가 아닙니다. Retry와 rate-limit 신규 정책도 이 단계에는 포함하지 않습니다.

## Model Alias

Client 요청에는 provider field를 넣지 않습니다. `model` 값이 registry alias이며, bridge가 provider를 고릅니다.

```json
{
  "llama-local": {
    "provider": "ollama",
    "kind": "chat",
    "provider_model": "llama3.1"
  },
  "gemini-2.5-flash": {
    "provider": "vertex",
    "api": "generateContent",
    "kind": "chat",
    "location": "us-central1"
  }
}
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `BRIDGE_API_KEY` | `""` | Optional Bearer token for client requests. |
| `MODEL_REGISTRY_JSON` | `""` | Model alias registry override JSON. |
| `EXTRA_MODELS` | `""` | Backward-compatible comma-separated Vertex predict model additions. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama native API base URL. |
| `GOOGLE_APPLICATION_CREDENTIALS` | `""` | Vertex service account JSON path. |
| `VERTEX_PROJECT` | required | GCP project id for Vertex. |
| `VERTEX_LOCATION` | `us-central1` | Default Vertex region. |
| `VERTEX_TASK_TYPE_DEFAULT` | `RETRIEVAL_DOCUMENT` | Default Vertex embedding task type. |
| `VERTEX_AUTO_TRUNCATE` | `true` | Vertex embedding auto truncate flag. |
| `MAX_CONCURRENCY` | `8` | Provider HTTP concurrency limit. |
| `HTTP_TIMEOUT_SECONDS` | `60` | Provider HTTP timeout. |

Cost tracking keeps the existing fail-closed behavior. If `COST_TRACKING_ENABLED=true`, every billable model/endpoint must have explicit pricing.

## Run

```bash
uv run uvicorn openai_compatible_bridge.main:app --reload --port 8000
```

Docker:

```bash
docker compose up -d --build
```

Use `http://127.0.0.1:8000` for local clients, or `http://openai-compatible-bridge` inside the configured Docker network.

## Endpoints

| Method | Endpoint |
| --- | --- |
| `GET` | `/healthz` |
| `GET` | `/v1/models` |
| `GET` | `/v1/models/{model_id}` |
| `POST` | `/v1/embeddings` |
| `POST` | `/v1/chat/completions` |
| `POST` | `/v1/rerank` |
| `GET` | `/admin/cost/status` |
| `GET` | `/admin/cost/events` |
| `GET` | `/admin/cost/reconciliation` |
