# RAGFlow 0.25.6용 Vertex AI 임베딩 OpenAI 호환 래퍼 설계서

## 전제와 출처 기준

이 문서는 질문에 포함된 **이미 확정된 제약**을 재검증하지 않고 그대로 전제로 사용했다. 따라서 RAGFlow 0.25.6의 OpenAI-API-Compatible 임베딩 경로가 `openai` Python SDK를 사용하고, `base_url`에 `/v1`가 자동으로 붙지 않으며, 실제 호출이 `POST {BASE_URL}/embeddings`가 된다는 점은 **사용자 제공 사실**로 취급했다. 아래의 **확인된 사실**은 가능한 한 Google Cloud 공식 문서로 인용했고, 비교 대상으로 요청된 **Gemini Developer API / OpenAI Embeddings API / RAGFlow 동작**은 각각 Google AI for Developers, OpenAI 공식 개발자 문서, RAGFlow 공식 GitHub 이슈 로그를 별도로 인용했다.

## Vertex AI 임베딩 REST 스펙

**확인된 사실**: Google Cloud의 Vertex AI 텍스트 임베딩 REST 엔드포인트는 지역 리전 엔드포인트를 쓰는 `predict` 형태이며, 요청은 `instances[]`와 `parameters`를 가진다. Google 공식 텍스트 임베딩 API 문서는 이 경로를 `POST https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{MODEL}:predict`로 설명하고, `instances[].content`, `instances[].task_type`, `instances[].title`, `parameters.outputDimensionality`, `parameters.autoTruncate`를 문서화한다. 공식 Gemini Enterprise Agent Platform의 `embedContent` 참조도 동일한 임베딩 작업 유형 열거형과 `title`, `taskType`, `autoTruncate`, `outputDimensionality` 필드를 문서화한다. 응답 쪽은 `predictions[].embeddings.values`와 `predictions[].embeddings.statistics.token_count`, `truncated` 정보를 반환하는 예시가 공식 문서에 실려 있다. citeturn2view0turn10view0turn9search0

**확인된 사실**: Google 공식 문서에 나오는 임베딩 작업 유형은 `RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, `QUESTION_ANSWERING`, `FACT_VERIFICATION`, `CODE_RETRIEVAL_QUERY`이며, `UNSPECIFIED`도 enum 상 존재한다. 또한 `title`은 텍스트 전용 임베딩 모델에만 적용되며, 텍스트 임베딩 문서에서는 `RETRIEVAL_DOCUMENT`와 함께 쓰는 필드로 안내된다. citeturn10view0turn10view1turn10view2turn10view3turn10view4turn2view0

### 현재 문서상 지원 모델과 제약

아래 표는 **Vertex AI 텍스트 임베딩 `:predict` 계열** 기준으로 정리했다. 마지막 행의 `gemini-embedding-2`는 **2026-06-06 현재 Google Cloud가 별도 문서로 제공하는 최신 멀티모달 임베딩 모델**이지만, `text-embeddings-api`의 레거시 `:predict` 페이지가 아니라 `embedContent` 계열로 문서화돼 있다는 점을 구분해야 한다. citeturn2view0turn12search16turn16search3turn12search5

| 모델 | 최대 입력 | 요청당 최대 instance | 출력 차원 | 권장 용도 / 비고 |
|---|---|---:|---|---|
| `gemini-embedding-001` | 입력 텍스트 최대 2,048 tokens citeturn2view0turn13view5 | **1개**. Google Cloud 문서는 이 모델이 한 번에 하나의 입력을 받는다고 안내한다. 배치 추론 문서도 안정(stable) 텍스트 임베딩 중 예외로 `gemini-embedding-001`을 지목한다. citeturn2view0turn16search7 | 기본 3,072차원. Gemini API 모델 카드는 **128–3072**의 유연한 출력 차원과 권장값 768/1536/3072를 명시한다. 다만 **Vertex `:predict` 페이지 자체는 하한 128을 명시적으로 적지 않으므로**, 래퍼에서 하한값을 강제하기보다 서버 검증에 맡기는 편이 안전하다. citeturn2view0turn13view5 | 고품질 텍스트 임베딩. Vertex 쪽에서는 지역별 quota, `predict` API 추가 quota가 따로 있다. citeturn17view0 |
| `text-embedding-005` | 텍스트 최대 2,048 tokens citeturn2view0 | 최대 5개 텍스트를 한 요청에 포함 가능 citeturn2view0 | 768차원 기본. 공식 문서는 차원 단축을 지원한다고 안내하며, Google 문서 스니펫은 Matryoshka Representation Learning 기반으로 차원 축소가 가능하다고 설명한다. citeturn2view0turn0search1 | 영어/다국어 일반 텍스트 임베딩. 운영 안정 버전. citeturn2view0 |
| `text-multilingual-embedding-002` | 텍스트 최대 2,048 tokens citeturn2view0 | 최대 5개 텍스트를 한 요청에 포함 가능 citeturn2view0 | 768차원 기본. 공식 문서는 차원 단축을 지원한다고 안내한다. citeturn2view0turn0search1 | 다국어 검색·유사도용에 적합하다. citeturn2view0 |
| `gemini-embedding-2` | 별도 멀티모달 임베딩 문서 기준 모델 citeturn16search3 | `embedContent` 계열 사용. `:predict` 레거시 텍스트 임베딩 페이지와는 별도 API 패밀리다. citeturn12search5turn16search3 | 3,072차원 멀티모달 벡터 citeturn16search3 | **2026년 최신 Google Cloud 임베딩 모델**이지만, 본 래퍼 설계 범위의 `:predict` 텍스트 임베딩과는 API가 다르다. citeturn16search3turn17view0 |

### Vertex AI와 AI Studio의 `gemini-embedding-001` 차이

| 항목 | Vertex AI | AI Studio / Gemini Developer API |
|---|---|---|
| 기본 엔드포인트 | `https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{MODEL}:predict` 또는 최신 모델의 `embedContent` 변형 citeturn2view0turn12search5 | `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent` 및 `:batchEmbedContents` citeturn14view0 |
| 요청 형식 | `instances[]` + `parameters` citeturn2view0 | `content` / `contents` + `embedContentConfig`, 또는 `requests[]` 배치 citeturn14view0turn14view2turn14view3 |
| 응답 형식 | `predictions[].embeddings.values`, `predictions[].embeddings.statistics.token_count`, `truncated` citeturn9search0turn2view0 | 단건은 `embedding`, 다건은 `embeddings[]`, 그리고 `usageMetadata` citeturn14view4 |
| 인증 | OAuth Bearer 토큰. Google Cloud auth/ADC 사용이 기본이며, Google 문서는 OpenAI 호환 사용 시에도 short-lived access token을 쓰라고 안내한다. citeturn21view0turn21view2 | API key가 기본이며 `x-goog-api-key` 헤더 사용 예제가 공식 문서에 있다. citeturn13view2turn14view0 |
| 할당량 | Vertex 쪽은 **리전별**·프로젝트별 quota가 적용되고, `gemini-embedding-001`은 `embed_content_input_tokens_per_minute_per_base_model`와 `predict` API용 online prediction quota가 따로 문서화돼 있다. citeturn17view0 | AI Studio는 **프로젝트 usage tier** 기준의 RPM / TPM / RPD 체계를 사용한다. citeturn13view3turn18view2turn18view3 |

**추정/설계 판단**: RAGFlow용 OpenAI 호환 래퍼는 **Vertex `predict` 텍스트 임베딩**에 맞추는 편이 가장 단순하다. 이유는 질문에서 요구한 경로가 정확히 `:predict`이고, 사용 대상도 `gemini-embedding-001`, `text-embedding-005`, `text-multilingual-embedding-002`이기 때문이다. `gemini-embedding-2`까지 흡수하려면 같은 `/v1/embeddings` 외부 인터페이스 뒤에서 내부 Google 호출을 `predict`와 `embedContent` 두 계열로 분기해야 한다. 이 문서의 코드는 질문 범위에 맞춰 **`predict` 기반 텍스트 임베딩**에 집중했다. citeturn2view0turn16search3turn12search5

## Vertex 인증과 토큰 관리

**확인된 사실**: Application Default Credentials는 인증 라이브러리가 현재 환경에 맞는 자격증명을 자동으로 찾는 전략이며, 검색 순서는 `GOOGLE_APPLICATION_CREDENTIALS` 환경변수, `gcloud auth application-default login`으로 만든 로컬 ADC 파일, 그리고 메타데이터 서버가 제공하는 attached service account 순서다. `GOOGLE_APPLICATION_CREDENTIALS`에는 서비스 계정 키 JSON을 둘 수 있지만, Google 문서는 서비스 계정 키가 보안 위험을 만들기 때문에 권장하지 않는다고 명시한다. Google Cloud 상의 프로덕션 환경에서는 attached service account 사용이 선호된다. citeturn22view0turn22view1turn22view2turn22view3

**확인된 사실**: Vertex AI 인증 문서는 REST와 client library 양쪽 모두에서 ADC 사용을 안내하고, OpenAI 호환 사용 문서의 Python 예시는 `google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])`로 credential을 얻은 뒤 `creds.refresh(Request())`로 필요 시 자동 갱신하는 패턴을 보여 준다. 또한 같은 문서는 서비스 계정 access token이 기본적으로 **1시간** 동안 유효하다고 설명한다. Google Cloud의 토큰 타입 문서와 서비스 계정 credential 문서도 서비스 계정 access token이 기본 1시간 만료라고 확인한다. citeturn21view0turn21view1turn21view2turn23search1turn23search3

**확인된 사실**: Google Cloud client libraries 문서는 ADC를 쓰면 애플리케이션이 토큰을 직접 관리할 필요가 없고, 인증 라이브러리가 이를 자동으로 처리한다고 설명한다. 또한 서비스 계정 impersonation으로 만든 short-lived access token에는 refresh token이 없으며, 만료되면 다시 access token을 생성해야 한다고 Google 문서가 명시한다. 따라서 서버 자동화에서 일반적인 패턴은 **refresh token을 직접 저장·조작하는 것**이 아니라, `google-auth`가 access token을 재발급하도록 맡기고 필요 시 credential을 refresh하는 것이다. citeturn21view1turn23search9turn21view0

**추정/설계 판단**: 공식 문서는 멀티스레드 잠금 전략까지 규정하지 않는다. 하지만 custom raw-HTTP 래퍼에서는 credential 객체가 갱신 가능한 mutable 상태를 가지므로, **만료 직전 선갱신**, **동시 refresh 직렬화**, **async 엔드포인트에서 blocking refresh를 별도 thread로 실행**하는 구현이 실무적으로 안전하다. 아래 코드에서 이 부분을 `threading.Lock`과 `asyncio.to_thread()`로 처리했다.

## OpenAI Embeddings 대조 스펙과 필드 매핑

**확인된 사실**: OpenAI 공식 Embeddings API는 `POST /embeddings`이며, 요청 body의 핵심 필드는 `input`, `model`, optional `dimensions`, optional `encoding_format`, optional `user`다. `input`은 문자열 또는 문자열 배열이 될 수 있고, 응답은 `object:"list"`, `data:[{object:"embedding", index, embedding:[float]}]`, `model`, `usage:{prompt_tokens,total_tokens}` 구조를 가진다. OpenAI 가이드 문서는 임베딩 응답이 기본적으로 float 벡터를 포함하며, `text-embedding-3-small` 기본 길이가 1536, `text-embedding-3-large` 기본 길이가 3072라고 설명한다. citeturn32view0turn32view1

**확인된 사실**: OpenAI Python SDK의 `extra_body`는 **문서화되지 않은 추가 JSON body 필드**를 보내기 위한 request option이다. 즉 `extra_body={"drop_params": True}`는 OpenAI Embeddings API의 표준 필드가 아니라, SDK가 body에 임의 필드를 추가로 실어 보낼 수 있게 해 주는 메커니즘이다. 별도로 LiteLLM 문서는 `drop_params=True`를 “지원되지 않는 OpenAI 파라미터를 자동으로 제거하는 LiteLLM 옵션”으로 설명한다. LiteLLM 문서는 이 옵션이 **지원되지 않는 OpenAI 파라미터 드롭용**이며, provider-specific 파라미터는 body kwargs로 전달된다고도 적고 있다. citeturn35view0turn36search1turn36search3

**결론**: `drop_params`는 **OpenAI Embeddings 표준도 아니고 Vertex AI `predict` 표준도 아니다**. 따라서 이 래퍼는 이를 **무시해도 된다**. 더 정확히 말하면, 래퍼가 `drop_params`를 이해할 필요가 없고, body에 들어와도 **허용하지만 의미 없이 버리는 것**이 가장 안전하다. 이 판단은 OpenAI SDK의 `extra_body` 성격과 LiteLLM의 `drop_params` 정의에 부합한다. citeturn35view0turn36search1turn36search3

### OpenAI ↔ Vertex 필드 매핑 표

| OpenAI 호환 요청/응답 | Vertex AI `predict` 요청/응답 | 구현 메모 |
|---|---|---|
| `POST /v1/embeddings` | `POST .../publishers/google/models/{MODEL}:predict` | 모델명은 path로 들어간다. citeturn32view0turn2view0 |
| `input: "str"` | `instances: [{"content":"str"}]` | 1건 단일 입력 매핑. citeturn32view0turn2view0 |
| `input: ["a","b"]` | `instances: [{"content":"a"},{"content":"b"}]` 또는 모델 한도에 맞춰 분할 | `gemini-embedding-001`은 1건씩 분할, `text-embedding-*`는 최대 5건까지 한 요청에 실을 수 있다. citeturn2view0turn16search7 |
| `model` | URL path의 `{MODEL}` | body가 아니라 경로에 주입. citeturn32view0turn2view0 |
| `dimensions` | `parameters.outputDimensionality` | 지원 모델에서만 사용. citeturn32view0turn10view0 |
| `encoding_format="float"` | 별도 Vertex 요청 필드 없음 | Vertex는 float 리스트를 반환하므로 래퍼에서 그대로 OpenAI 형식으로 재포장한다. `base64`는 이 설계서 코드에서 지원하지 않는다. citeturn32view0turn9search0 |
| `user` | 직접 대응 없음 | Vertex `predict` 표준 필드에 없다. 래퍼에서 무시 가능. citeturn32view0turn2view0 |
| `drop_params` | 직접 대응 없음 | LiteLLM 전용 의미이므로 래퍼에서 드롭. citeturn35view0turn36search1turn36search3 |
| `data[i].embedding` | `predictions[i].embeddings.values` | 가장 핵심적인 응답 매핑. citeturn32view0turn9search0turn2view0 |
| `usage.prompt_tokens` / `usage.total_tokens` | `sum(predictions[].embeddings.statistics.token_count)` | 임베딩은 출력 토큰 개념이 없으므로 prompt/total을 동일하게 채우는 것이 OpenAI 형태에 가장 가깝다. citeturn32view0turn9search0 |

## 구현 설계와 실행 코드

**확인된 사실**: RAGFlow의 모델 추가 흐름에서 백엔드 로그는 `POST /v1/llm/add_llm` 직전에 실제 `/embeddings` 요청을 보내고, 실패 시 `Fail to access embedding model(...)`를 기록한다. 즉, 등록/검증 흐름이 **실제 임베딩 요청 성공 여부**를 게이트로 쓰는 것은 로그 수준에서 확인된다. 다만 이 로그만으로 **UI의 Verify 버튼이 내부적으로 정확히 어떤 백엔드 함수를 호출하는지**까지는 완전히 단정할 수 없어서, 아래의 “Verify 버튼 동작”은 확인된 사실과 추정을 분리해 적었다. citeturn40view0

**추정/설계 판단**: 아래 코드는 질문에서 요구한 범위에 맞춰 다음을 충족하도록 작성했다. OpenAI 형식의 `POST /v1/embeddings`, `google-auth` 기반 short-lived access token 발급·캐시·갱신, `gemini-embedding-001`의 1-instance 제약 처리, `outputDimensionality` 매핑, 기본 `task_type=RETRIEVAL_DOCUMENT`, `usage.total_tokens`를 Vertex `token_count` 합계로 채우는 동작, 그리고 Vertex 4xx/5xx를 OpenAI 유사 에러 객체로 변환하는 동작이다. `drop_params` 같은 비표준 extra field는 허용하되 무시한다. 공식 스펙상 근거가 있는 부분은 위 문단들의 인용을 따랐고, 에러 타입 문자열·선갱신 skew·query/document 분기 한계는 **호환성 설계 선택**이다. citeturn21view0turn23search1turn32view0turn9search0turn35view0turn36search1

### `app.py`

```python
from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

import google.auth
import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from pydantic import BaseModel, ConfigDict


CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
SUPPORTED_TASK_TYPES = {
    "UNSPECIFIED",
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
}

KNOWN_MAX_INSTANCES = {
    "gemini-embedding-001": 1,
    "text-embedding-005": 5,
    "text-multilingual-embedding-002": 5,
}

VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_TASK_TYPE_DEFAULT = os.getenv("VERTEX_TASK_TYPE_DEFAULT", "RETRIEVAL_DOCUMENT")
VERTEX_AUTO_TRUNCATE = os.getenv("VERTEX_AUTO_TRUNCATE", "true").lower() in {"1", "true", "yes", "on"}
WRAPPER_API_KEY = os.getenv("WRAPPER_API_KEY")
TOKEN_REFRESH_SKEW_SECONDS = int(os.getenv("TOKEN_REFRESH_SKEW_SECONDS", "300"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))


class OpenAIEmbeddingsRequest(BaseModel):
    # extra_body={"drop_params": True} 같은 필드는 top-level JSON으로 들어올 수 있으므로 허용
    model_config = ConfigDict(extra="allow")

    input: str | list[str] | list[int] | list[list[int]]
    model: str
    encoding_format: str | None = "float"
    dimensions: int | None = None
    user: str | None = None


class VertexAPIError(Exception):
    def __init__(self, status_code: int, message: str, code: str | None = None, raw: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.raw = raw


def openai_error_response(
    *,
    message: str,
    status_code: int,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def map_vertex_status_to_openai_type(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "api_error"
    return "invalid_request_error"


def coerce_string_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    # OpenAI 스펙은 token array도 허용하지만, 이 래퍼는 Vertex text-embedding predict의 text content만 지원
    raise ValueError("This wrapper supports only string or array[string] inputs.")


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class GoogleAccessTokenProvider:
    def __init__(self) -> None:
        creds, detected_project = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
        self._creds = creds
        self.project_id = VERTEX_PROJECT or detected_project
        if not self.project_id:
            raise RuntimeError(
                "Unable to determine Google Cloud project. Set VERTEX_PROJECT or configure ADC project."
            )
        self._lock = threading.Lock()
        self._request = GoogleAuthRequest()

    def _valid_with_skew(self) -> bool:
        token = getattr(self._creds, "token", None)
        expiry = getattr(self._creds, "expiry", None)

        if not token:
            return False
        if expiry is None:
            return bool(self._creds.valid)

        now = datetime.now(timezone.utc)
        remaining = (expiry - now).total_seconds()
        return bool(self._creds.valid) and remaining > TOKEN_REFRESH_SKEW_SECONDS

    def _get_token_sync(self) -> str:
        with self._lock:
            if not self._valid_with_skew():
                self._creds.refresh(self._request)

            token = getattr(self._creds, "token", None)
            if not token:
                raise RuntimeError("Failed to obtain Google access token.")
            return token

    async def get_token(self) -> str:
        return await asyncio.to_thread(self._get_token_sync)


class VertexEmbeddingClient:
    def __init__(self, token_provider: GoogleAccessTokenProvider) -> None:
        self.token_provider = token_provider
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))

    async def close(self) -> None:
        await self.http.aclose()

    async def predict(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        title: str | None,
        auto_truncate: bool,
    ) -> list[dict[str, Any]]:
        token = await self.token_provider.get_token()
        url = (
            f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/"
            f"v1/projects/{self.token_provider.project_id}/locations/{VERTEX_LOCATION}/"
            f"publishers/google/models/{model}:predict"
        )

        instances: list[dict[str, Any]] = []
        for text in texts:
            item: dict[str, Any] = {"content": text}
            if task_type:
                item["task_type"] = task_type
            if title:
                item["title"] = title
            instances.append(item)

        parameters: dict[str, Any] = {"autoTruncate": auto_truncate}
        if dimensions is not None:
            parameters["outputDimensionality"] = dimensions

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            resp = await self.http.post(
                url,
                headers=headers,
                json={
                    "instances": instances,
                    "parameters": parameters,
                },
            )
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": {"message": resp.text}}

            err = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = err.get("message") or resp.text or "Vertex AI request failed"
            code = err.get("status") or err.get("code") or str(resp.status_code)
            raise VertexAPIError(resp.status_code, message=message, code=str(code), raw=payload)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        predictions = data.get("predictions")
        if not isinstance(predictions, list):
            raise VertexAPIError(502, "Malformed Vertex AI response: missing predictions[]", code="bad_gateway")

        return predictions


app = FastAPI(title="Vertex AI OpenAI-Compatible Embeddings Wrapper", version="0.1.0")

token_provider = GoogleAccessTokenProvider()
vertex_client = VertexEmbeddingClient(token_provider)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await vertex_client.close()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/embeddings")
async def create_embeddings(
    payload: OpenAIEmbeddingsRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_vertex_task_type: str | None = Header(default=None, alias="X-Vertex-Task-Type"),
    x_vertex_title: str | None = Header(default=None, alias="X-Vertex-Title"),
) -> JSONResponse | dict[str, Any]:
    # 선택적 wrapper API key 검증
    if WRAPPER_API_KEY:
        expected = f"Bearer {WRAPPER_API_KEY}"
        if authorization != expected:
            return openai_error_response(
                message="Invalid wrapper API key.",
                status_code=401,
                error_type="authentication_error",
                code="invalid_api_key",
            )

    # OpenAI embeddings 표준과 RAGFlow 현재 호출 형태를 기준으로 float만 지원
    if payload.encoding_format not in (None, "float"):
        return openai_error_response(
            message="This wrapper supports only encoding_format='float'.",
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_encoding_format",
            param="encoding_format",
        )

    if payload.dimensions is not None and payload.dimensions < 1:
        return openai_error_response(
            message="dimensions must be >= 1.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_dimensions",
            param="dimensions",
        )

    try:
        texts = coerce_string_inputs(payload.input)
    except ValueError as exc:
        return openai_error_response(
            message=str(exc),
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_input_shape",
            param="input",
        )

    if not texts:
        return openai_error_response(
            message="input must not be empty.",
            status_code=400,
            error_type="invalid_request_error",
            code="empty_input",
            param="input",
        )

    task_type = x_vertex_task_type or VERTEX_TASK_TYPE_DEFAULT
    if task_type not in SUPPORTED_TASK_TYPES:
        return openai_error_response(
            message=f"Unsupported task type: {task_type}",
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_task_type",
            param="X-Vertex-Task-Type",
        )

    batch_size = KNOWN_MAX_INSTANCES.get(payload.model, 1)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def one_chunk(chunk: list[str]) -> list[dict[str, Any]]:
        async with semaphore:
            return await vertex_client.predict(
                model=payload.model,
                texts=chunk,
                dimensions=payload.dimensions,
                task_type=task_type,
                title=x_vertex_title,
                auto_truncate=VERTEX_AUTO_TRUNCATE,
            )

    try:
        chunk_results = await asyncio.gather(
            *(one_chunk(chunk) for chunk in chunked(texts, batch_size))
        )
    except VertexAPIError as exc:
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )

    data: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    output_index = 0

    for predictions in chunk_results:
        for pred in predictions:
            emb = pred.get("embeddings", {})
            values = emb.get("values")
            stats = emb.get("statistics", {}) if isinstance(emb, dict) else {}

            if not isinstance(values, list):
                return openai_error_response(
                    message="Malformed Vertex AI response: embeddings.values missing.",
                    status_code=502,
                    error_type="api_error",
                    code="bad_gateway",
                )

            token_count = stats.get("token_count", 0)
            try:
                total_prompt_tokens += int(token_count)
            except Exception:
                pass

            data.append(
                {
                    "object": "embedding",
                    "index": output_index,
                    "embedding": values,
                }
            )
            output_index += 1

    return {
        "object": "list",
        "data": data,
        "model": payload.model,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "total_tokens": total_prompt_tokens,
        },
    }
```

### `pyproject.toml`

```toml
[project]
name = "vertex-openai-embeddings-wrapper"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.116.0",
  "google-auth>=2.40.0",
  "httpx>=0.28.0",
  "pydantic>=2.11.0",
  "uvicorn[standard]>=0.35.0",
]

[tool.uv]
package = false
```

### `Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml
RUN uv sync --no-dev

COPY app.py /app/app.py

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 실행 예시

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/app/sa.json
export VERTEX_PROJECT=your-gcp-project
export VERTEX_LOCATION=us-central1
export WRAPPER_API_KEY=sk-ragflow-local
export VERTEX_TASK_TYPE_DEFAULT=RETRIEVAL_DOCUMENT
export VERTEX_AUTO_TRUNCATE=true

uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8080
```

## RAGFlow 등록값과 운영 체크리스트

**확인된 사실**: RAGFlow의 모델 추가/검증 흐름은 실제 `/embeddings` 호출을 시도하고, 실패하면 `Fail to access embedding model(...)`를 반환한다. 로그에는 `/embeddings`에 대한 retry도 보이므로, 네트워크 접근성·모델명·base URL·인증 실패가 모두 이 단계에서 바로 표면화된다. citeturn40view0

**추정/설계 판단**: 따라서 RAGFlow 등록값은 다음처럼 맞추는 것이 가장 안전하다. `Model type=Embedding`, `Model name=<Vertex 모델명>` 예를 들어 `gemini-embedding-001` 또는 `text-embedding-005`, `Base url=http://<wrapper-host>:<port>/v1`, `API-Key=<WRAPPER_API_KEY 또는 비워둠>`, `Max tokens=2048`. 마지막 값은 Google 공식 문서상 세 모델 모두 2,048 입력 토큰 한도와 맞춘 값이다. `Verify` 버튼은 **사실상 임베딩 1건 테스트 호출**이 성공해야 통과하는 것으로 보는 것이 맞고, 정확히는 **모델 추가/검증 경로에서 실제 `/embeddings` 요청이 성공해야** 한다. `Base url`에 `/v1`를 빠뜨리면 사용자 전제대로 실제 호출이 `/embeddings`로 가기 때문에 즉시 실패한다. `encoding_format="float"`는 RAGFlow가 현재 그렇게 보내므로 래퍼도 float 출력만 지원하면 충분하다. Google 모델 차원이 바뀌면 기존 지식베이스의 벡터 차원과 충돌할 수 있으므로, **지식베이스를 만든 뒤에는 `dimensions` 값을 고정**하는 것이 안전하다. 앞서 적었듯 `gemini-embedding-001`은 1건씩 분할해야 하고, 토큰 만료·동시 refresh·리전 지연 시간도 운영 포인트다. citeturn2view0turn13view5turn40view0

### 실무 체크리스트

| 항목 | 상태 구분 | 설명 |
|---|---|---|
| `Base url`에 `/v1` 포함 | 사용자 전제 + 설계 필수 | 누락 시 실제 호출이 잘못된 경로로 나간다. |
| `encoding_format="float"` | 확인된 사실 + 구현 반영 | RAGFlow 호출 형태와 맞다. OpenAI·Vertex 모두 float 벡터 표현과 양립한다. citeturn32view0turn9search0 |
| `gemini-embedding-001` 배치 분할 | 확인된 사실 | 요청당 1 instance 제약이 있다. citeturn2view0turn16search7 |
| `dimensions` 고정 | 추정/설계 판단 | KB 생성 후 차원을 바꾸면 기존 벡터와 불일치할 수 있다. |
| 토큰 만료 전 갱신 | 확인된 사실 + 설계 판단 | 서비스 계정 access token 기본 수명은 1시간이며, 선갱신 캐시가 안전하다. citeturn23search1turn21view0 |
| 동시성 refresh 보호 | 추정/설계 판단 | custom raw-HTTP wrapper에서는 refresh stampede 방지가 유리하다. |
| 리전 선택 | 확인된 사실 + 실무 판단 | Vertex 쿼터는 지역 기준이므로, RAGFlow와 가까운 리전을 쓰는 것이 latency·quota 운영에 유리하다. citeturn17view0turn21view2 |

### Docker 네트워크와 배포 포인트

**확인된 사실**: 의존성은 `fastapi`, `uvicorn`, `google-auth`, `httpx`만으로 충분하고, Google 공식 권장 인증 경로는 ADC다. 서비스 계정 키 파일을 쓴다면 `GOOGLE_APPLICATION_CREDENTIALS`로 경로를 주고, Google Cloud 상이라면 attached service account가 권장된다. 최소 권한 쪽은 Google Cloud 문서가 `roles/aiplatform.user`를 실제 Gemini Enterprise Agent Platform foundation model 사용 역할로 예시한다. 또한 IAM 문서는 필요 시 `aiplatform.endpoints.predict`만 담은 custom role을 만들 수 있음을 설명한다. citeturn21view1turn22view1turn22view3turn27search1turn28view0

**추정/설계 판단**: RAGFlow 컨테이너와 래퍼 컨테이너는 **같은 Docker network**에 두고, `Base url`은 `http://wrapper:8080/v1`처럼 컨테이너 이름으로 주는 것이 가장 단순하다. 래퍼가 호스트에서 돌고 RAGFlow가 Docker 안이라면 `host.docker.internal` 사용 여부는 플랫폼별 차이가 있으니, Linux에서는 별도 `extra_hosts` 설정이 더 안전할 수 있다. 이 부분은 Docker 배치 방식에 따라 달라지는 운영 판단이다.

## 미해결점과 한계

**확인된 사실**: Google 문서는 `RETRIEVAL_DOCUMENT`와 `RETRIEVAL_QUERY`를 अलग task type으로 구분한다. citeturn10view0turn10view1

**추정/설계 판단**: 그러나 **순수 OpenAI Embeddings 호환 인터페이스**에는 “지금 이 호출이 문서 임베딩인지, 쿼리 임베딩인지”를 표현하는 표준 필드가 없다. 따라서 RAGFlow를 수정하지 않고 이 래퍼만 끼우는 경우, 질문에서 요청한 기본값 `RETRIEVAL_DOCUMENT`는 **문서와 쿼리 모두에 같은 task type이 적용되는 한계**가 있다. 이를 엄밀히 해결하려면 RAGFlow에서 쿼리 시점과 문서 색인 시점에 서로 다른 헤더를 보내게 패치하거나, 별도 래퍼 인스턴스를 두 개 두고 호출 경로를 분리해야 한다.

**확인된 사실**: `gemini-embedding-2`는 2026-06 현재 Google Cloud에 문서화된 최신 임베딩 모델이지만, `embedContent` API 계열을 사용한다. citeturn16search3turn12search5turn17view0

**추정/설계 판단**: 따라서 본 답변의 코드는 질문 범위인 `:predict` 텍스트 임베딩에 초점을 맞췄고, `gemini-embedding-2`까지 같은 래퍼에서 지원하려면 내부 Google 호출 경로를 모델별로 더 분기하는 추가 작업이 필요하다.

**확인된 사실**: RAGFlow 모델 추가 흐름이 `/embeddings` 테스트 호출을 한다는 로그는 확보했지만, UI의 **Verify 버튼이 정확히 동일한 백엔드 경로를 호출하는지**는 여기서 직접 소스코드까지 추적하지 못했다. 다만 적어도 등록/검증 단계의 성공 조건이 “실제 임베딩 요청 성공”이라는 점은 로그로 확인됐다. citeturn40view0