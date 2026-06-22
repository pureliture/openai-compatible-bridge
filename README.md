<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=openai-compatible-bridge&fontSize=40" width="100%" alt="openai-compatible-bridge"/>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" alt="Google Cloud"/>
  <img src="https://img.shields.io/badge/Ollama-111827?style=for-the-badge&logoColor=white" alt="Ollama"/>
</div>

<br/>
<div align="center">
  <h3>OpenAI-compatible API shape을 유지하면서 Vertex AI와 Ollama 같은 provider-native model API를 연결하는 <b>로컬/사내망 전용 private bridge</b>입니다.</h3>
</div>
<br/>

> [!WARNING]
> **배포 위치 주의**: 이 bridge는 Vertex service account, local model endpoint, cost ledger 같은 운영 자원을 연결합니다. 보안과 과금 보호를 위해 **public internet에 노출하지 말고**, local machine 또는 private Docker network 안에서만 운용하십시오.

<br/>

<div align="center">
  <a href="#-시스템-아키텍처"><img src="https://img.shields.io/badge/🏛️%20시스템%20아키텍처-555555?style=for-the-badge" alt="시스템 아키텍처"/></a> &nbsp;|&nbsp;
  <a href="#-빠른-시작"><img src="https://img.shields.io/badge/🚀%20빠른%20시작-555555?style=for-the-badge" alt="빠른 시작"/></a> &nbsp;|&nbsp;
  <a href="#-환경-변수-설정"><img src="https://img.shields.io/badge/⚙️%20환경%20변수%20설정-555555?style=for-the-badge" alt="환경 변수 설정"/></a> &nbsp;|&nbsp;
  <a href="#-api-참조"><img src="https://img.shields.io/badge/📡%20API%20참조-555555?style=for-the-badge" alt="API 참조"/></a>
</div>

---

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%" alt="section divider"/>

## 💡 개발 배경 (Why?)

OpenAI-compatible client가 여러 provider를 직접 다루게 만들면 인증, payload, streaming, batching, 비용 추적 책임이 client마다 흩어집니다. 이 bridge는 client에는 `/v1/...` endpoint와 `model` alias만 노출하고, 내부에서 provider-native API 호출을 분기합니다.

* **Provider routing 단일화**: client 요청에는 provider field를 넣지 않습니다. `model` alias가 registry를 통해 Vertex 또는 Ollama provider adapter로 해석됩니다.
* **Vertex 운영 복잡도 흡수**: Vertex service account 인증, model별 batching, embeddings/rerank/chat payload 변환을 bridge가 담당합니다.
* **Local model 연결**: Ollama chat completions를 같은 `/v1/chat/completions` 표면으로 연결해 local model과 Vertex model을 같은 client 설정에서 다룰 수 있습니다.
* **비용 방어선 유지**: cost tracking을 켜면 billable request가 upstream 호출 전에 budget gate를 통과해야 합니다.

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%" alt="section divider"/>

## 🏛️ 시스템 아키텍처

<div align="center">
  <img src="./assets/architecture.svg" width="100%" alt="OpenAI-compatible bridge architecture"/>
</div>

### 🎨 핵심 설계 포인트

<table width="100%">
  <tr>
    <td width="50%" valign="top">

#### 🟦 Drop-in API Shape
<p>OpenAI-compatible client는 기존처럼 <code>/v1/embeddings</code>, <code>/v1/chat/completions</code>, <code>/v1/rerank</code>를 호출합니다.</p>
    </td>
    <td width="50%" valign="top">

#### 🟩 Provider Adapter
<p>Bridge 내부에서 model alias를 Vertex 또는 Ollama provider-native model id로 해석합니다.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">

#### 🟪 Vertex Native Support
<p>Vertex embeddings, chat completions, Search Ranking rerank를 OpenAI-compatible response shape으로 변환합니다.</p>
    </td>
    <td width="50%" valign="top">

#### 🟧 Private Cost Gate
<p>Cost tracking이 켜진 경우 가격 설정 없는 billable model은 fail-closed 처리하고 hard budget 초과 요청은 upstream 호출 전에 차단합니다.</p>
    </td>
  </tr>
</table>

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%" alt="section divider"/>

## 🧭 지원 범위

| Endpoint | Provider | 상태 |
|---|---|---|
| `POST /v1/embeddings` | Vertex | 지원 |
| `POST /v1/chat/completions` | Vertex | 지원 |
| `POST /v1/chat/completions` | Ollama | 지원 |
| `POST /v1/rerank` | Vertex | 지원 |

Ollama embeddings와 Ollama rerank는 현재 범위가 아닙니다. Retry와 rate-limit 신규 정책도 이 단계에는 포함하지 않습니다.

### Model Alias

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

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%" alt="section divider"/>

## ⚙️ 환경 변수 설정 (Configuration)

`.env` 파일을 생성하거나 컨테이너 환경 변수로 다음 값을 주입합니다.

| Variable | Default | Description |
|---|---|---|
| `BRIDGE_API_KEY` | `""` | Client request 보호용 선택적 Bearer token. |
| `MODEL_REGISTRY_JSON` | `""` | Model alias registry override JSON. |
| `EXTRA_MODELS` | `""` | Backward-compatible comma-separated Vertex predict model additions. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama native API base URL. Docker에서는 `http://host.docker.internal:11434` 권장. |
| `GOOGLE_APPLICATION_CREDENTIALS` | `""` | Vertex service account JSON path. |
| `VERTEX_PROJECT` | *(Required)* | GCP project id for Vertex. |
| `VERTEX_LOCATION` | `us-central1` | Default Vertex region. |
| `VERTEX_TASK_TYPE_DEFAULT` | `RETRIEVAL_DOCUMENT` | Default Vertex embedding task type. |
| `VERTEX_AUTO_TRUNCATE` | `true` | Vertex embedding auto truncate flag. |
| `MAX_CONCURRENCY` | `8` | Provider HTTP concurrency limit. |
| `HTTP_TIMEOUT_SECONDS` | `60` | Provider HTTP timeout. |
| `DEFAULT_MAX_INSTANCES` | `1` | 알 수 없는 Vertex model 병렬 호출 시 기본 chunk size. |
| `COST_TRACKING_ENABLED` | `false` | 비용 추적과 hard budget gate 활성화 여부. |
| `COST_LEDGER_PATH` | `""` | 비용 원장 SQLite 파일 경로. Docker에서는 `/data/cost-ledger.db` 권장. |
| `COST_PRICING_JSON` | `""` | 모델/endpoint별 가격 JSON. `COST_PRICING_PATH`와 둘 중 하나를 사용. |
| `COST_PRICING_PATH` | `""` | 가격 JSON 파일 경로. |
| `COST_SHORT_WINDOW_SECONDS` | `""` | 단기 budget window 길이(초). 비용 추적 활성화 시 필수. |
| `COST_SHORT_WINDOW_LIMIT_USD` | `""` | 단기 window hard limit. 비용 추적 활성화 시 필수. |
| `COST_DAILY_LIMIT_USD` | `""` | 일 단위 hard limit. 비용 추적 활성화 시 필수. |
| `COST_ADMIN_ENABLED` | `false` | Private cost admin API 활성화 여부. |
| `COST_ADMIN_API_KEY` | `""` | Cost admin 전용 Bearer token. `BRIDGE_API_KEY`와 별도 값이어야 함. |
| `COST_RECONCILIATION_ENABLED` | `false` | Cloud Billing BigQuery reconciliation 활성화 여부. |
| `COST_RETENTION_REQUEST_DAYS` | `90` | Request-level 비용 원장 보존 기간. |
| `COST_RETENTION_AGGREGATE_MONTHS` | `13` | Aggregate/reconciliation 보존 기간. |

<details>
<summary><b>💡 복잡한 모델 라우팅 추가 방법</b></summary>
<p><code>MODEL_REGISTRY_JSON</code>으로 provider, API type, region, provider-native model id를 alias별로 제어할 수 있습니다.</p>
</details>

### 비용 추적과 hard budget gate

<div align="center">
  <img src="./assets/cost-tracking-flow.svg" width="100%" alt="Cost tracking hard budget flow"/>
</div>

<br/>

`COST_TRACKING_ENABLED=true`이면 모든 billable request는 provider 호출 전에 SQLite 원장에 forecast 비용을 예약합니다. 단기 window 또는 일 단위 limit을 넘는 요청은 upstream 호출 없이 HTTP 429 `budget_exceeded`로 차단됩니다. 성공 응답의 OpenAI-compatible shape에는 비용 필드를 추가하지 않습니다.

가격 설정은 코드에 내장하지 않고 `COST_PRICING_JSON` 또는 `COST_PRICING_PATH`로 주입합니다.

```json
{
  "source": "manual",
  "version": "YYYY-MM-DD",
  "currency": "USD",
  "models": {
    "gemma-4-26b-a4b-it-maas": {
      "chat": {
        "input_per_million": "",
        "output_per_million": ""
      }
    },
    "text-embedding-005": {
      "embeddings": {
        "embedding_per_million": ""
      }
    },
    "semantic-ranker-512@latest": {
      "rerank": {
        "rerank_per_unit": ""
      }
    }
  }
}
```

Docker Compose에서는 비용 원장 디렉터리를 `/data`에 mount합니다. 컨테이너를 재생성해도 `COST_LEDGER_DIR`가 유지되면 `/data/cost-ledger.db`가 그대로 남습니다.

```bash
mkdir -p ./data
docker compose up -d --build
```

운영 확인용 private endpoint는 `COST_ADMIN_ENABLED=true`와 `COST_ADMIN_API_KEY`가 모두 설정된 경우에만 열립니다.

| Method | Endpoint | 용도 |
|---|---|---|
| `GET` | `/admin/cost/status` | 현재 spend, limit, reset time, health, reconciliation 상태 |
| `GET` | `/admin/cost/events` | Allowlist 기반 최근 비용 이벤트 |
| `GET` | `/admin/cost/reconciliation` | Cloud Billing export 대조 상태 |

Cloud Billing BigQuery reconciliation은 request path를 막지 않습니다. Export 미설정은 `unavailable`, 최근 billing row 지연은 `pending`, 권한/쿼리 오류는 `error`로 admin API에 노출됩니다.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%" alt="section divider"/>

## 🎯 범용 AI 클라이언트 연동 가이드

Bridge를 로컬/사내망에 띄웠다면, OpenAI API 규격을 지원하는 도구에서 custom base URL만 이 bridge로 지정합니다.

| 설정 항목 | 입력할 값 | 비고 |
|---|---|---|
| **API Provider** | `OpenAI` 또는 compatible provider | 도구별 custom OpenAI-compatible provider 선택 |
| **Base URL** | `http://127.0.0.1:8000` | Docker host debug port를 쓰면 `http://127.0.0.1:8930` |
| **API Key** | `BRIDGE_API_KEY` 값 또는 dummy string | `BRIDGE_API_KEY`가 비어 있으면 인증은 강제되지 않음 |
| **Model Name** | Registry alias | 예: `gemini-2.5-flash`, `llama-local` |

Docker network 내부 client는 service DNS를 사용할 수 있습니다.

```text
http://openai-compatible-bridge
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%" alt="section divider"/>

## 🚀 빠른 시작

### ⚡ 요구사항

- uv
- Docker 및 Docker Compose
- Vertex 사용 시 GCP service account JSON key file (`roles/aiplatform.user`)
- Ollama 사용 시 local Ollama server

### 🧪 로컬 환경 (uv 사용)

```bash
uv run uvicorn openai_compatible_bridge.main:app --reload --port 8000
```

### 🐳 Docker Compose 환경

```bash
docker compose up -d --build
```

Use `http://127.0.0.1:8000` for local clients, or `http://openai-compatible-bridge` inside the configured Docker network.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%" alt="section divider"/>

## 📡 API 참조

| Method | Endpoint | 호환 규격 | Provider |
|---|---|---|---|
| <img src="https://img.shields.io/badge/GET-2563EB?style=flat-square" alt="GET"/> | `/healthz` | Health check | Bridge |
| <img src="https://img.shields.io/badge/GET-2563EB?style=flat-square" alt="GET"/> | `/v1/models` | OpenAI-compatible | Registry |
| <img src="https://img.shields.io/badge/GET-2563EB?style=flat-square" alt="GET"/> | `/v1/models/{model_id}` | OpenAI-compatible | Registry |
| <img src="https://img.shields.io/badge/POST-009688?style=flat-square" alt="POST"/> | `/v1/embeddings` | OpenAI-compatible | Vertex |
| <img src="https://img.shields.io/badge/POST-009688?style=flat-square" alt="POST"/> | `/v1/chat/completions` | OpenAI-compatible | Vertex / Ollama |
| <img src="https://img.shields.io/badge/POST-009688?style=flat-square" alt="POST"/> | `/v1/rerank` | Cohere / LocalAI-compatible | Vertex |
| <img src="https://img.shields.io/badge/GET-2563EB?style=flat-square" alt="GET"/> | `/admin/cost/status` | Private admin | Cost ledger |
| <img src="https://img.shields.io/badge/GET-2563EB?style=flat-square" alt="GET"/> | `/admin/cost/events` | Private admin | Cost ledger |
| <img src="https://img.shields.io/badge/GET-2563EB?style=flat-square" alt="GET"/> | `/admin/cost/reconciliation` | Private admin | Billing reconciliation |

<br/>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" width="100%" alt="footer"/>
</div>

<div align="center">
  <a href="#-시스템-아키텍처"><img src="https://img.shields.io/badge/🏛️%20아키텍처-555555?style=for-the-badge" alt="아키텍처"/></a> &nbsp;|&nbsp;
  <a href="#-빠른-시작"><img src="https://img.shields.io/badge/🚀%20빠른%20시작-555555?style=for-the-badge" alt="빠른 시작"/></a> &nbsp;|&nbsp;
  <a href="#top"><img src="https://img.shields.io/badge/⬆️%20맨%20위로-555555?style=for-the-badge" alt="맨 위로"/></a>
</div>
