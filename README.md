<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=vertex-ai-api-wrapper&fontSize=40" width="100%"/>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white"/>
</div>

<br/>
<div align="center">
  <h3>Google Cloud Vertex AI (Google Agent Platform API) 임베딩 및 Rerank API를 OpenAI/LocalAI 규격 클라이언트에서 사용할 수 있게 해주는 <b>로컬/사내망 전용 프록시 서버</b>입니다.</h3>
</div>
<br/>

> [!WARNING]
> **배포 위치 주의**: 이 래퍼 서버는 구글 클라우드 강력 권한을 가진 서비스 계정 키(JSON)를 구동 환경에 두어야 작동합니다. 보안(데이터 유출 및 과금 폭탄 방지)을 위해 **퍼블릭망에 웹 서버로 띄우는 것을 엄격히 금지**합니다. 타겟 AI 클라이언트가 구동 중인 **동일한 로컬 PC나 사내 폐쇄망(Docker network) 내부에서 프라이빗 프록시로만 구동**하십시오.

<br/>

<div align="center">
  <a href="#-시스템-아키텍처"><img src="https://img.shields.io/badge/🏛️%20시스템%20아키텍처-555555?style=for-the-badge"/></a> &nbsp;|&nbsp;
  <a href="#-빠른-시작"><img src="https://img.shields.io/badge/🚀%20빠른%20시작-555555?style=for-the-badge"/></a> &nbsp;|&nbsp;
  <a href="#-환경-변수-설정"><img src="https://img.shields.io/badge/⚙️%20환경%20변수%20설정-555555?style=for-the-badge"/></a> &nbsp;|&nbsp;
  <a href="#-api-참조"><img src="https://img.shields.io/badge/📡%20API%20참조-555555?style=for-the-badge"/></a>
</div>

---

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>


## 💡 개발 배경 (Why?)

Google Agent Platform API(Vertex AI)의 우수한 모델들을 외부 오픈소스 생태계에 직접 연결하기에는 다음과 같은 까다로운 허들이 존재하여, 이를 중간에서 해결해주는 래퍼를 개발했습니다.

* **엔터프라이즈 보안 및 인증(Auth) 장벽**: 구글 AI Studio(개인/프로토타이핑용)는 연동이 쉬운 단순 API Key를 지원하지만 무료 티어 등에서 입력 데이터가 학습에 쓰일 위험이 있습니다. 반면, **데이터 프라이버시가 완벽히 보장되는 상용 엔터프라이즈용 Vertex AI**는 API Key 사용을 원천 차단하고 1시간마다 만료되는 임시 토큰(ADC/OAuth)을 강제합니다. 범용 오픈소스들은 고정 API Key만 지원하므로 백그라운드 갱신이 필요합니다.
* **배치 처리(Batching) 한계**: 외부 서비스는 한 번에 여러 데이터를 묶어 보내지만, 구글의 일부 모델(예: `gemini-embedding-001`)은 한 번에 1개씩의 입력만 허용하므로 중간에서 요청을 쪼개는 자동 분할(Auto-Batching)이 필수적입니다.
* **입출력 규격(Payload) 불일치**: 업계 표준인 OpenAI API 페이로드와 구글 Vertex 전용 페이로드 구조가 완전히 달라 실시간 통역기가 필요합니다.

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>

## 🏛️ 시스템 아키텍처

<div align="center">
  <img src="./assets/architecture.svg" width="100%"/>
</div>


### 🎨 핵심 설계 포인트

<table width="100%">
  <tr>
    <td width="50%" valign="top">

#### 🟦 Drop-in Replacement
<p>기존 OpenAI/LocalAI 생태계 코드 변경 없이 Vertex AI를 그대로 사용 가능합니다.</p>
    </td>
    <td width="50%" valign="top">

#### 🟩 Native Reranking
<p><code>LocalAI</code> provider 규격을 통해 Vertex AI Search Ranking API를 완벽하게 연결합니다.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">

#### 🟪 Auto-Batching
<p>기존 클라이언트의 대용량 배치 요청을 Vertex AI의 모델별 한도(1~5개)에 맞춰 자동 분할 및 병렬 처리합니다.</p>
    </td>
    <td width="50%" valign="top">

#### 🟧 Auth Abstraction
<p>리프레시 토큰 관리 없이 ADC(Application Default Credentials) 서비스 계정을 통해 자동으로 OAuth2 토큰을 발급받습니다.</p>
    </td>
  </tr>
</table>

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>

## ⚙️ 환경 변수 설정 (Configuration)

`.env` 파일을 생성하거나 컨테이너 환경 변수로 다음 값을 주입합니다.

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | `""` | GCP 서비스 계정 JSON 키 경로 (필요 권한: `roles/aiplatform.user`). |
| `VERTEX_PROJECT` | *(Required)* | GCP 프로젝트 ID. |
| `VERTEX_LOCATION` | `us-central1` | Vertex API를 호출할 GCP 리전. |
| `WRAPPER_API_KEY` | `""` | 래퍼 서버를 보호하기 위한 선택적 API 키 (Bearer Token). |
| `VERTEX_TASK_TYPE_DEFAULT` | `RETRIEVAL_DOCUMENT` | 텍스트 임베딩을 위한 기본 Task Type. |
| `VERTEX_AUTO_TRUNCATE` | `true` | 토큰 제한 초과 시 400 에러 대신 자동으로 입력 텍스트를 자를지 여부. |
| `MAX_CONCURRENCY` | `8` | Vertex API에 대한 최대 동시 HTTP 요청 수. |
| `HTTP_TIMEOUT_SECONDS` | `60` | Vertex API HTTP 요청 타임아웃. |
| `TOKEN_REFRESH_SKEW_SECONDS` | `300` | Google OAuth 토큰 만료 전 사전 갱신 시간 (초). |
| `EXTRA_MODELS` | `""` | 콤마(,)로 구분된 추가 지원 모델 목록. |
| `MODEL_REGISTRY_JSON` | `""` | 복잡한 모델 라우팅을 위한 JSON 설정. |
| `DEFAULT_MAX_INSTANCES` | `1` | 알 수 없는 모델에 대한 병렬 호출 시 기본 청크 크기. |
| `COST_TRACKING_ENABLED` | `false` | 비용 추적과 hard budget gate 활성화 여부. |
| `COST_LEDGER_PATH` | `""` | 비용 원장 SQLite 파일 경로. Docker에서는 `/data/cost-ledger.db` 권장. |
| `COST_PRICING_JSON` | `""` | 모델/엔드포인트별 가격 JSON. `COST_PRICING_PATH`와 둘 중 하나를 사용. |
| `COST_PRICING_PATH` | `""` | 가격 JSON 파일 경로. |
| `COST_SHORT_WINDOW_SECONDS` | `""` | 단기 budget window 길이(초). 비용 추적 활성화 시 필수. |
| `COST_SHORT_WINDOW_LIMIT_USD` | `""` | 단기 window hard limit. 비용 추적 활성화 시 필수. |
| `COST_DAILY_LIMIT_USD` | `""` | 일 단위 hard limit. 비용 추적 활성화 시 필수. |
| `COST_ADMIN_ENABLED` | `false` | private cost admin API 활성화 여부. |
| `COST_ADMIN_API_KEY` | `""` | cost admin 전용 Bearer token. `WRAPPER_API_KEY`와 별도 값이어야 함. |
| `COST_RECONCILIATION_ENABLED` | `false` | Cloud Billing BigQuery reconciliation 활성화 여부. |
| `COST_RETENTION_REQUEST_DAYS` | `90` | request-level 비용 원장 보존 기간. |
| `COST_RETENTION_AGGREGATE_MONTHS` | `13` | aggregate/reconciliation 보존 기간. |

<details>
<summary><b>💡 복잡한 모델 라우팅 추가 방법</b></summary>
<p>새로운 모델은 <code>EXTRA_MODELS</code> 환경 변수에 콤마로 구분하여 추가하거나, <code>MODEL_REGISTRY_JSON</code>을 통해 구체적인 API 타입, 리전, max_instances 등을 제어할 수 있습니다.</p>
</details>

### 비용 추적과 hard budget gate

`COST_TRACKING_ENABLED=true`이면 모든 billable request는 Vertex 호출 전에 SQLite 원장에 forecast 비용을 예약한다. 단기 window 또는 일 단위 limit을 넘는 요청은 upstream 호출 없이 HTTP 429 `budget_exceeded`로 차단된다. 성공 응답의 OpenAI-compatible shape에는 비용 필드를 추가하지 않는다.

가격 설정은 코드에 내장하지 않고 `COST_PRICING_JSON` 또는 `COST_PRICING_PATH`로 주입한다.

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

Docker Compose에서는 비용 원장 디렉터리를 `/data`에 mount한다. 컨테이너를 재생성해도 `COST_LEDGER_DIR`가 유지되면 `/data/cost-ledger.db`가 그대로 남는다.

```bash
mkdir -p ./data
docker compose up -d --build
```

운영 확인용 private endpoint는 `COST_ADMIN_ENABLED=true`와 `COST_ADMIN_API_KEY`가 모두 설정된 경우에만 열린다.

| Method | Endpoint | 용도 |
|---|---|---|
| `GET` | `/admin/cost/status` | 현재 spend, limit, reset time, health, reconciliation 상태 |
| `GET` | `/admin/cost/events` | allowlist 기반 최근 비용 이벤트 |
| `GET` | `/admin/cost/reconciliation` | Cloud Billing export 대조 상태 |

Cloud Billing BigQuery reconciliation은 request path를 막지 않는다. export 미설정은 `unavailable`, 최근 billing row 지연은 `pending`, 권한/쿼리 오류는 `error`로 admin API에 노출된다.

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>



## 🎯 범용 AI 클라이언트 연동 가이드

래퍼 서버를 로컬/사내망에 띄웠다면, OpenAI API 규격을 지원하는 어떤 툴(LangChain, Dify 등)이든 설정창에 아래 값을 입력하여 즉시 연동할 수 있습니다.

| 설정 항목 | 입력할 값 | 비고 |
|---|---|---|
| **API Provider** | `OpenAI` 또는 `LocalAI` | 호환 가능한 커스텀 API 제공자 선택 |
| **Base URL** | `http://127.0.0.1:8000` | 서버를 띄운 주소. 뒤에 `/v1`이 붙지 않도록 주의 |
| **API Key** | `sk-dummy` | 내부 로컬 통신이므로 아무 문자열이나 입력 |
| **Model Name** | `gemini-embedding-001` 등 | 구글 Vertex AI의 실제 모델명 입력 |

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>

## 🚀 빠른 시작

### ⚡ 요구사항
- uv
- Docker 및 Docker Compose
- GCP 서비스 계정 JSON 키 파일 (`roles/aiplatform.user`)

### 🧪 로컬 환경 (uv 사용)

```bash
# 의존성 설치 및 백엔드 서버 실행
uv run uvicorn app:app --reload --port 8000
```

### 🐳 Docker Compose 환경

```bash
# 백그라운드 컨테이너 빌드 및 실행
docker compose up -d --build
```

<br/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=2" width="100%"/>

## 📡 API 참조

이 래퍼는 아래와 같은 호환 엔드포인트를 제공합니다.

| Method | Endpoint | 호환 규격 | 반환 형식 |
|---|---|---|---|
| <img src="https://img.shields.io/badge/POST-009688?style=flat-square"/> | `/v1/embeddings` | OpenAI 호환 | `encoding_format` 지원 |
| <img src="https://img.shields.io/badge/POST-009688?style=flat-square"/> | `/v1/chat/completions` | OpenAI 호환 | SSE Stream 지원 |
| <img src="https://img.shields.io/badge/POST-009688?style=flat-square"/> | `/v1/rerank` | Cohere / LocalAI 호환 | `results[{index, relevance_score}]` |

<br/>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer"/>
</div>

<div align="center">
  <a href="#-시스템-아키텍처"><img src="https://img.shields.io/badge/🏛️%20아키텍처-555555?style=for-the-badge"/></a> &nbsp;|&nbsp;
  <a href="#-빠른-시작"><img src="https://img.shields.io/badge/🚀%20빠른%20시작-555555?style=for-the-badge"/></a> &nbsp;|&nbsp;
  <a href="#top"><img src="https://img.shields.io/badge/⬆️%20맨%20위로-555555?style=for-the-badge"/></a>
</div>
