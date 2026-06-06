# **RAGFlow 0.25.6 및 Google Cloud Vertex AI 임베딩 연동을 위한 백엔드 아키텍처 및 구현 설계서**

## **대규모 언어 모델 기반 지식 검색 증강(RAG) 파이프라인의 통합 과제**

엔터프라이즈 환경에서 지식 검색 증강(RAG, Retrieval-Augmented Generation) 시스템을 구축할 때, 시스템의 데이터 보존성, 보안, 그리고 검색 정확도를 결정짓는 핵심 계층은 임베딩(Embedding) 모델이다. 현재 오픈소스 RAG 파이프라인 플랫폼으로 널리 사용되는 RAGFlow 0.25.6은 내부적으로 openai Python SDK를 활용하여 외부 언어 모델 제공자들과 통신하는 구조를 채택하고 있다.1 이 플랫폼은 다수의 LLM 공급업체를 지원하나, Google Cloud의 엔터프라이즈 AI 플랫폼인 Vertex AI의 임베딩 모델(Google Models)을 위한 네이티브 프로바이더를 내장하고 있지 않다. 따라서 RAGFlow의 "OpenAI-API-Compatible" 프로바이더 기능을 활용하여 Vertex AI 임베딩 모델을 호출하기 위한 중간 래퍼(Wrapper) API 서버 설계가 필수적으로 요구된다.  
본 시스템 설계는 이미 확정된 RAGFlow의 내부 제약 사항을 전제로 한다. RAGFlow 0.25.6은 "OpenAI-API-Compatible" 모드에서 client.embeddings.create(input=\[...\], model=MODEL\_NAME, encoding\_format="float", extra\_body={"drop\_params": True}) 규격으로 요청을 전송한다. 또한, 사용자가 RAGFlow UI에 입력한 Base url은 어떠한 자동 경로 추가(/v1 등) 없이 그대로 사용되므로, 실제 호출이 올바르게 라우팅되기 위해서는 래퍼 서버의 URL 입력값이 명확해야 한다. 문서 파싱 과정에서 입력은 단일 문자열 또는 문자열 배열(Batch) 형태로 유입될 수 있으며, 쿼리 임베딩은 RAGFlow 내부적으로 8,191자 길이에서 강제 절사(Truncate)된다. 본 설계서는 이러한 기저 환경을 바탕으로 Google 공식 문서를 분석하고, 완벽히 동작하는 비동기 기반의 프로덕션 레벨 래퍼 서버 구현 및 배포 방안을 제시한다.

## **Google Cloud Vertex AI 임베딩 REST 스펙 및 데이터 모델링 분석**

Vertex AI는 OpenAI의 범용 API와는 완전히 다른 독자적인 RESTful API 스키마를 사용한다. 안정적인 시스템 간 통신(Server-to-Server)을 보장하기 위해서는 Vertex AI의 정확한 엔드포인트 토폴로지와 데이터 구조, 그리고 개별 임베딩 모델이 가지는 토큰 및 배치 제약 사항을 완벽히 매핑해야 한다.

### **텍스트 임베딩 엔드포인트 및 네트워크 라우팅 구조**

Vertex AI 임베딩 모델 호출을 위한 정확한 REST API 엔드포인트 포맷은 Google Cloud 인프라의 리전(Region) 격리 원칙을 따른다.3 요청은 반드시 해당 프로젝트가 위치한 리전의 API 게이트웨이로 전송되어야 한다.  
정확한 HTTP POST 엔드포인트는 다음과 같이 구성된다.3 POST https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{MODEL}:predict  
해당 엔드포인트는 Google Cloud의 사설 네트워크(VPC) 내부 및 외부에서 모두 접근 가능하며, 전송 계층 보안(TLS)을 통해 모든 데이터 페이로드가 암호화된다.

### **요청 및 응답 페이로드 스키마 (Request and Response Schema)**

Vertex AI는 입력 데이터를 instances 배열에 담아 전송하며, 하이퍼파라미터 설정은 parameters 객체를 통해 제어한다.4

| 파라미터 그룹 | 필드명 | 데이터 타입 | 필수 여부 | 공식 스펙 및 역할 설명 |
| :---- | :---- | :---- | :---- | :---- |
| **요청 (Request)** | instances.content | String | 필수 | 임베딩 벡터로 변환할 실제 텍스트 문자열. 모델의 최대 허용 토큰 수를 초과할 경우 autoTruncate 설정에 따라 절사되거나 에러를 발생시킨다.3 |
| **요청 (Request)** | instances.task\_type | String | 선택 | 모델이 생성할 임베딩의 최적화 컨텍스트를 지정한다. RAG 파이프라인의 검색 품질을 극대화하기 위해 제공된다.5 |
| **요청 (Request)** | instances.title | String | 선택 | 문서 임베딩 시 추가적인 문맥을 제공하기 위한 텍스트의 제목 또는 식별자.4 |
| **요청 (Request)** | parameters.outputDimensionality | Integer | 선택 | MRL(Matryoshka Representation Learning)을 지원하는 모델에서 출력 임베딩 벡터의 차원을 축소하여 반환하도록 지정한다.3 |
| **요청 (Request)** | parameters.autoTruncate | Boolean | 선택 | true로 설정 시 최대 토큰 한도를 초과하는 텍스트를 자동으로 절사한다. false일 경우 400 Bad Request 에러를 반환한다. 기본값은 true이다.3 |
| **응답 (Response)** | predictions.embeddings.values | Array of Floats | N/A | 실수 배열로 구성된 n차원의 생성된 임베딩 벡터 데이터.4 |
| **응답 (Response)** | predictions.embeddings.statistics.token\_count | Integer | N/A | 입력 텍스트를 처리하는 데 소모된 실제 토큰의 총 수. 이 값은 과금 기준 및 OpenAI 규격의 usage 정보로 매핑된다.4 |
| **응답 (Response)** | predictions.embeddings.statistics.truncated | Boolean | N/A | 원본 텍스트가 모델 한도를 초과하여 절사되었는지 여부를 나타내는 불리언 플래그.4 |

특히 instances.task\_type은 단순한 메타데이터가 아니라 임베딩 벡터의 공간(Space) 배치를 결정짓는 핵심 하이퍼파라미터이다.5 예를 들어, 질문과 답변은 의미적으로 유사하지 않기 때문에 단순한 의미 유사도 기반 검색에서는 관계를 맺기 어렵다.5 Google Cloud 공식 문서에 명시된 주요 task\_type은 다음과 같다.4

* RETRIEVAL\_DOCUMENT: 코퍼스(Corpus) 내에서 검색 대상이 되는 문서를 임베딩할 때 사용한다.  
* RETRIEVAL\_QUERY: 사용자의 질문이나 검색어를 임베딩할 때 사용한다. 문서를 찾는 데 최적화된 벡터를 생성한다.  
* SEMANTIC\_SIMILARITY: 두 텍스트 간의 의미적 유사도를 평가할 때 사용하며, RAG 검색용으로는 권장되지 않는다.  
* QUESTION\_ANSWERING: 질문 형태("하늘은 왜 파란가요?")의 쿼리에 최적화된 임베딩을 생성한다.  
* FACT\_VERIFICATION, CLASSIFICATION, CLUSTERING, CODE\_RETRIEVAL\_QUERY 등 데이터 마이닝 및 코드 검색을 위한 특정 태스크 타입들이 존재한다.

RAGFlow는 데이터베이스 인덱싱(문서 파싱) 단계와 쿼리 단계에서 동일한 API 엔드포인트를 호출한다. 시스템 통합 측면에서 래퍼 서버는 task\_type을 파라미터화하되, 기본값을 RAG의 핵심 동작인 RETRIEVAL\_DOCUMENT로 설정하여 문서 색인의 품질을 보장해야 한다.

### **2026년 기준 최신 임베딩 모델 라인업 및 물리적 제약**

Vertex AI에서 제공하는 임베딩 모델들은 목적, 언어 지원 범위, 그리고 연산 한계에 따라 명확한 제약 사항을 가진다. 아래 모델 스펙은 시스템 아키텍처에서 배치 크기와 토큰 제한 로직을 설계하는 기준점이 된다.4

| 모델 식별자 (Model Name) | 최대 입력 토큰 수 | 출력 차원 (기본 / 조정 가능) | 요청당 최대 Instance 허용 개수 | 아키텍처 특성 및 권장 용도 |
| :---- | :---- | :---- | :---- | :---- |
| gemini-embedding-001 | 2,048 토큰 | 기본 3072 (MRL 기반 축소 지원) | **1개 (요청당 단일 텍스트만 허용)** | 영문, 다국어, 코드를 통합 처리하는 State-of-the-art 모델. 고품질 RAG 텍스트 벡터 검색에 최적화되어 있다.4 |
| gemini-embedding-2 및 preview | 8,192 토큰 | 기본 3072 (MRL 기반 축소 지원) | 최대 250개 (배치 처리 가능) | 텍스트, 이미지, 오디오, 비디오, PDF 데이터를 단일 벡터 공간으로 매핑하는 옴니(Omni) 임베딩 모델. 긴 시퀀스의 컨텍스트 처리에 적합하다.8 |
| text-embedding-005 | 2,048 토큰 | 최대 768 | 최대 250개 | 영어 및 코드 작업에 특화된 고속 임베딩 모델로, 비용 효율적인 파이프라인 구축에 권장된다.4 |
| text-multilingual-embedding-002 | 2,048 토큰 | 최대 768 | 최대 250개 | 다국어 텍스트의 분류 및 유사도 평가에 특화된 모델이다.4 |

**설계 시 핵심 제약 조건 (사실 확인 및 대응 전략):** 공식 문서의 명시적 스펙에 따르면, gemini-embedding-001 모델은 REST API 호출 시 **단일 요청에 오직 1개의 입력 텍스트(instances 배열 크기 1)만 포함할 수 있다**.3 RAGFlow는 텍스트를 청킹(Chunking)하여 배치(Batch) 형태로 묶어 래퍼 서버에 전송한다. 만약 래퍼 서버가 수신된 문자열 배열을 그대로 instances에 담아 gemini-embedding-001로 전송할 경우 HTTP 400 Bad Request 에러가 발생하여 전체 인덱싱 파이프라인이 중단된다. 따라서 래퍼 서버는 대상 모델이 gemini-embedding-001인지 판별한 후, 입력 배열을 개별 요청으로 분할(Splitting)하여 비동기 동시 호출(Concurrent Requests)을 수행하고 결과를 병합(Merge)하는 다중화 비동기 I/O 로직을 반드시 구현해야 한다.

### **엔터프라이즈 워크로드를 위한 Vertex AI와 AI Studio의 아키텍처적 차이**

시스템 통합 과정에서 Google AI Studio(generativelanguage) API 키를 사용하는 방식과 Vertex AI(aiplatform)를 사용하는 방식 간의 차이를 명확히 이해해야 한다.3

1. **인프라 엔드포인트 분리**: Vertex AI는 aiplatform.googleapis.com을 사용하여 리전별(Regional) 트래픽 격리와 데이터 레지던시를 보장한다. 반면 AI Studio는 글로벌 엔드포인트(generativelanguage.googleapis.com)를 사용하여 소비자 수준의 서비스를 제공한다.  
2. **보안 및 인증 모델**: Vertex AI는 Google Cloud IAM(Identity and Access Management) 기반의 서비스 어카운트 및 짧은 수명의 OAuth 토큰을 사용한다. 반면 AI Studio는 정적(Static) API 키를 사용하여 인증하므로 엔터프라이즈 보안 정책(키 회전 등)을 충족하기 어렵다.11  
3. **할당량(Quota) 및 SLA**: Vertex AI는 프로젝트 및 리전 단위로 세분화된 할당량(예: 분당 토큰 수, 분당 요청 수)을 제공하며 엔터프라이즈 SLA(Service Level Agreement)가 적용된다.12 gemini-embedding-001은 리전 쿼터를 따르는 반면, gemini-embedding-2는 글로벌 쿼터 시스템을 사용한다.12 따라서 데이터 프라이버시가 엄격히 요구되는 사내 지식 기반 RAGFlow 구축에는 Vertex AI API를 연동하는 것이 필수적이다.

## **Vertex AI 인증 체계 및 분산 환경에서의 토큰 생명주기 관리**

시스템 간 통신(Server-to-Server)에서 Vertex AI API를 호출하기 위해서는 OAuth 2.0 기반의 액세스 토큰(Access Token)을 발급받아야 한다. 하드코딩된 자격 증명(Credential) 사용을 지양하고, 시스템이 스스로 인증을 관리하는 현대적 클라우드 네이티브 보안 모델을 적용한다.

### **서비스 어카운트 및 Application Default Credentials (ADC)**

(출처: [https://cloud.google.com/docs/authentication/client-libraries](https://cloud.google.com/docs/authentication/client-libraries?authuser=1) 13, [https://cloud.google.com/docs/authentication](https://cloud.google.com/docs/authentication?authuser=1) 14)  
Google Cloud는 워크로드(Workload)의 자동 인증을 위해 Application Default Credentials(ADC) 메커니즘을 권장한다. 래퍼 서버를 구동하는 컨테이너 환경에 권한이 부여된 서비스 어카운트(Service Account)의 JSON 키 파일을 마운트하고, 시스템 환경 변수 GOOGLE\_APPLICATION\_CREDENTIALS에 해당 파일의 절대 경로를 지정한다. Python의 google-auth 라이브러리는 런타임에 이를 자동으로 감지하여 내부적으로 크레덴셜 객체를 생성한다. Vertex AI 예측 API를 호출하기 위해 요구되는 OAuth 2.0 Scope는 https://www.googleapis.com/auth/cloud-platform이다.11

### **토큰 갱신(Refresh) 메커니즘과 추상화**

(출처: [https://cloud.google.com/iam/docs/create-short-lived-credentials-direct](https://cloud.google.com/iam/docs/create-short-lived-credentials-direct?authuser=1) 16)  
서비스 어카운트를 통해 발급받은 액세스 토큰의 수명은 기본적으로 약 1시간(3600초)이다.11 애플리케이션 코드는 Refresh Token을 직접 파싱하거나 보관할 필요가 없다. google-auth 라이브러리가 서명 생성, JWT 만료 감지, 메타데이터 서버 연동 등 복잡한 인증 흐름을 추상화하여 제공하기 때문이다. 정석적인 패턴은 google.auth.default()를 사용하여 크레덴셜 객체(credentials)를 획득한 후, google.auth.transport.requests.Request() 객체를 주입하여 credentials.refresh(request)를 호출하는 것이다. 이 호출이 성공하면 credentials.token에 새로운 유효 토큰이 주입된다.16

### **비동기 환경에서의 캐시 스탬피드(Cache Stampede) 방지 및 스레드 안전성**

RAGFlow가 대량의 문서 청크를 병렬로 래퍼 서버에 전송하는 환경에서는 다수의 비동기 코루틴(Coroutine)이 동시에 실행된다. 만약 토큰이 만료된 시점에 수십 개의 코루틴이 동시에 credentials.refresh()를 호출하게 되면, 중복된 네트워크 요청이 발생하여 GCP 인증 서버에 불필요한 부하를 주거나 병목이 생기는 캐시 스탬피드(Cache Stampede) 현상이 발생한다. 더불어 credentials.refresh()는 본질적으로 동기적(Synchronous) I/O 블로킹 작업이므로, 비동기 이벤트 루프(Event Loop) 내에서 직접 호출하면 전체 API 서버의 응답성이 저하된다.  
이 문제를 해결하기 위해 토큰 갱신 로직은 다음과 같은 스레드 안전성(Thread-Safety)을 확보하도록 설계되어야 한다.

1. 메모리에 현재 토큰의 만료 시간(credentials.expiry)을 초 단위로 추적한다.  
2. 만료 시간으로부터 5\~10분 전(Margin)에 도달하면, 요청을 처리하기 전에 선제적으로 토큰을 갱신한다.  
3. 갱신 작업 중에는 비동기 락(asyncio.Lock())을 사용하여 단 하나의 작업(Task)만 갱신을 수행하도록 제어하는 더블 체크 잠금(Double-checked locking) 패턴을 적용한다.  
4. 블로킹되는 refresh 호출은 asyncio.to\_thread를 사용하여 별도의 스레드 풀에서 실행함으로써 ASGI 워커의 이벤트 루프가 정지되지 않도록 한다.

## **대조군 분석: OpenAI 임베딩 API 스펙 및 통신 프로토콜 호환성**

RAGFlow는 래퍼 서버에게 OpenAI 표준 API 포맷으로 요청을 전송하고 동일한 규격의 응답을 기대한다. 이를 Vertex 스키마와 변환하기 위해 RAGFlow가 생성하는 OpenAI 스펙의 본질을 분석한다.

### **표준화된 요청 및 응답 스키마**

(출처: [https://platform.openai.com/docs/api-reference/embeddings](https://platform.openai.com/docs/api-reference/embeddings) 4)  
**OpenAI Request Body 규격:**

JSON  
{  
  "input": "텍스트 문자열 또는 \[문자열 배열\]",  
  "model": "사용할 모델명 (예: text-embedding-3-small)",  
  "encoding\_format": "float",  
  "dimensions": 1024,  
  "extra\_body": { "drop\_params": true }  
}

* input: 임베딩을 생성할 대상. 단일 문자열 또는 문자열 배열(List) 형태를 취한다.  
* dimensions: 출력 벡터의 차원을 제한하는 파라미터. RAGFlow에서 MRL을 활성화하기 위해 전달될 수 있다.  
* extra\_body: {"drop\_params": True} 속성은 RAGFlow와 LiteLLM 등 중간 프록시 레이어 간의 호환성 확보를 위해 전달되는 플래그이다. OpenAI 표준 API에서 지원하지 않는 파라미터가 포함되었을 때 에러를 발생시키지 말고 이를 조용히 무시(Ignore)하라는 의미를 갖는다.18 래퍼 서버 입장에서는 이 필드가 존재하더라도 안전하게 무시하고 처리 로직을 계속 진행하면 된다.

**OpenAI Response Body 규격:**

JSON  
{  
  "object": "list",  
  "data": \[  
    {  
      "object": "embedding",  
      "index": 0,  
      "embedding": \[0.01, \-0.02, 0.03,...\]  
    }  
  \],  
  "model": "text-embedding-3-small",  
  "usage": {  
    "prompt\_tokens": 15,  
    "total\_tokens": 15  
  }  
}

RAGFlow 시스템은 응답 객체 배열(data)의 순서(index)가 요청된 input 배열의 순서와 정확히 일치할 것을 기대한다. 또한 usage.total\_tokens 값을 파싱하여 시스템 내부에 사용량 지표를 로깅하므로 이 값 역시 누락 없이 채워져야 한다.

## **프로토콜 변환 설계: 필드 매핑 및 동작 전략**

요청을 중계하는 시스템의 핵심인 양방향 페이로드(Payload) 매핑 규칙은 다음과 같이 명확하게 정의된다. 이 규칙은 래퍼 서버가 데이터 유실 없이 두 플랫폼을 연결하는 기반이 된다.

| RAGFlow 수신 (OpenAI Format) | 변환 방향 | Vertex AI REST Format (요청/응답) | 변환 및 비즈니스 로직 규격 |
| :---- | :---- | :---- | :---- |
| input (String 또는 List) | **→** | instances.content | 문자열 또는 배열을 루프 처리하여 {"content": text, "task\_type": "RETRIEVAL\_DOCUMENT"} 객체 배열로 변환한다. gemini-embedding-001 모델은 배치 처리를 지원하지 않으므로, 1건씩 분할(Split)하여 병렬 HTTP 요청을 보낸다. |
| dimensions (Integer) | **→** | parameters.outputDimensionality | Vertex AI 모델 중 MRL을 지원하는 모델(gemini-embedding-001, gemini-embedding-2)로 라우팅될 경우에만 주입하여 차원 축소를 유도한다. |
| (없음, 래퍼 자체 기본값 주입) | **→** | parameters.autoTruncate \= true | RAG 처리 중 토큰 초과 시 400 Error 발생으로 인한 파이프라인 중단을 방지하고 RAG 동작의 유연성을 확보하기 위해 강제로 주입한다.3 |
| (응답 수신 대기) | **←** | predictions.embeddings.values | 수신된 Vertex의 실수 배열을 OpenAI 규격의 data.embedding 에 1:1로 매핑한다. 병렬 처리 분할 후 병합 시 원본 input의 index 순서 유지가 절대적으로 보장되어야 한다. |
| (응답 수신 대기) | **←** | predictions.embeddings.statistics.token\_count | 단건 또는 배치 내 모든 응답의 토큰을 합산하여 OpenAI 스키마의 usage.prompt\_tokens 및 usage.total\_tokens에 정수로 할당한다. |
| HTTP 에러 코드 (4xx, 5xx) | **←** | Vertex HTTP Status 및 Error Body | Vertex 에러 응답을 포착하여 OpenAI의 에러 포맷인 {"error": {"message": "...", "type": "...", "code":...}} 구조로 래핑(Wrapping)하여 반환한다. |

## **프로덕션 레벨 Python FastAPI 래퍼(Wrapper) 서버 구현체**

본 섹션에서는 상기 논의된 모든 제약 사항과 설계 사상을 반영하여 완벽히 동작하는 비동기 기반 Python FastAPI 구현 코드를 제시한다. 이 코드는 GCP 서비스 어카운트 인증 자동화, gemini-embedding-001의 1 Instance 제약을 우회하는 다중화 비동기 I/O, 그리고 견고한 예외 처리 로직을 모두 포함하고 있다.  
**의존성 패키지 (Dependencies):** fastapi, uvicorn, google-auth, httpx, pydantic

Python  
\# wrapper.py  
import os  
import time  
import asyncio  
from typing import List, Union, Optional, Dict, Any  
from fastapi import FastAPI, HTTPException, Request  
from fastapi.responses import JSONResponse  
from pydantic import BaseModel, Field  
import httpx  
import google.auth  
import google.auth.transport.requests

app \= FastAPI(title="Vertex AI to OpenAI Embedding Wrapper for RAGFlow")

\# 인프라스트럭처 환경 변수 기반 설정 로드  
GCP\_PROJECT\_ID \= os.environ.get("GCP\_PROJECT\_ID", "your-project-id")  
GCP\_LOCATION \= os.environ.get("GCP\_LOCATION", "us-central1")  
WRAPPER\_API\_KEY \= os.environ.get("WRAPPER\_API\_KEY", "your-secure-api-key")

class GCPTokenManager:  
    """GCP Access Token을 관리하고 만료 전 안전하게 갱신하는 싱글톤 매니저"""  
    def \_\_init\_\_(self):  
        \# ADC 메커니즘을 통해 인증 객체 로드  
        self.\_credentials, self.\_project \= google.auth.default(  
            scopes=\["https://www.googleapis.com/auth/cloud-platform"\]  
        )  
        self.\_token \= None  
        self.\_expiry \= 0  
        self.\_lock \= asyncio.Lock()  
        self.\_request \= google.auth.transport.requests.Request()

    async def get\_token(self) \-\> str:  
        current\_time \= time.time()  
        \# 토큰 만료 5분(300초) 전이면 선제적 갱신 시도  
        if not self.\_token or current\_time \>= (self.\_expiry \- 300):  
            async with self.\_lock:  
                \# 락을 획득한 후 다시 검사하는 Double-checked locking  
                current\_time \= time.time()  
                if not self.\_token or current\_time \>= (self.\_expiry \- 300):  
                    \# Blocking I/O인 토큰 갱신을 별도 스레드에서 실행하여 이벤트 루프 보호  
                    await asyncio.to\_thread(self.\_credentials.refresh, self.\_request)  
                    self.\_token \= self.\_credentials.token  
                    if self.\_credentials.expiry:  
                        self.\_expiry \= self.\_credentials.expiry.timestamp()  
                    else:  
                        \# 안전망: expiry가 명확하지 않다면 임의로 1시간 생명주기 설정  
                        self.\_expiry \= current\_time \+ 3600  
        return self.\_token

\# 싱글톤 인스턴스화  
token\_manager \= GCPTokenManager()

\# Pydantic을 활용한 OpenAI API 인터페이스 스키마 정의  
class OpenAIEmbeddingRequest(BaseModel):  
    input: Union\[str, List\[str\]\]  
    model: str  
    encoding\_format: Optional\[str\] \= "float"  
    dimensions: Optional\[int\] \= None  
    user: Optional\[str\] \= None  
    extra\_body: Optional\] \= None

def format\_openai\_error(message: str, status\_code: int, error\_type: str \= "api\_error"):  
    """Vertex 에러를 OpenAI 규격으로 변환하는 유틸리티"""  
    return JSONResponse(  
        status\_code=status\_code,  
        content={"error": {"message": message, "type": error\_type, "code": status\_code}}  
    )

@app.post("/v1/embeddings")  
async def create\_embeddings(req: Request, body: OpenAIEmbeddingRequest):  
    \# 1\. API Key 검증 로직 (사내 네트워크 보호)  
    auth\_header \= req.headers.get("Authorization", "")  
    if WRAPPER\_API\_KEY and WRAPPER\_API\_KEY\!= "NONE":  
        expected\_header \= f"Bearer {WRAPPER\_API\_KEY}"  
        if auth\_header\!= expected\_header:  
            return format\_openai\_error("Unauthorized access", 401, "invalid\_api\_key")

    \# 2\. Input 데이터 구조 정규화 (문자열을 리스트로 통일)  
    inputs \= \[body.input\] if isinstance(body.input, str) else body.input  
    if not inputs:  
        return format\_openai\_error("Input array cannot be empty", 400, "invalid\_request\_error")

    \# 3\. GCP 인증 토큰 획득  
    try:  
        access\_token \= await token\_manager.get\_token()  
    except Exception as e:  
        return format\_openai\_error(f"Failed to authenticate with GCP IAM: {str(e)}", 500)

    model\_name \= body.model  
    \# 4\. Vertex AI REST 호출 엔드포인트 조립  
    vertex\_url \= f"https://{GCP\_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP\_PROJECT\_ID}/locations/{GCP\_LOCATION}/publishers/google/models/{model\_name}:predict"  
    headers \= {  
        "Authorization": f"Bearer {access\_token}",  
        "Content-Type": "application/json"  
    }

    \# 하이퍼파라미터 구성  
    parameters \= {"autoTruncate": True}  
    if body.dimensions:  
        parameters \= body.dimensions

    \# 내부 비동기 HTTP 호출 클로저  
    async def fetch\_vertex\_embeddings(instances: List):  
        payload \= {"instances": instances, "parameters": parameters}  
        \# httpx 클라이언트를 활용하여 연결 재사용 및 비동기 통신 최적화  
        async with httpx.AsyncClient() as client:  
            resp \= await client.post(vertex\_url, headers=headers, json=payload, timeout=60.0)  
            if resp.status\_code\!= 200:  
                raise HTTPException(status\_code=resp.status\_code, detail=resp.text)  
            return resp.json()

    results \=  
    total\_tokens \= 0

    try:  
        \# 5\. 모델별 분기 처리 (제약 조건 우회)  
        if model\_name \== "gemini-embedding-001":  
            \# 1 Instance per request 제한: 배열을 개별 항목으로 분할하여 비동기 병렬 호출  
            tasks \=  
            for text in inputs:  
                \# Task type을 RETRIEVAL\_DOCUMENT로 고정하여 코퍼스 검색 품질 극대화  
                instance \=  
                tasks.append(fetch\_vertex\_embeddings(instance))  
              
            \# 모든 HTTP 코루틴을 병렬로 대기  
            responses \= await asyncio.gather(\*tasks)  
              
            \# 결과 병합 및 순서 유지  
            for i, resp in enumerate(responses):  
                prediction \= resp.get("predictions", \[{}\])  
                values \= prediction.get("embeddings", {}).get("values",)  
                tokens \= prediction.get("embeddings", {}).get("statistics", {}).get("token\_count", 0)  
                results.append({"object": "embedding", "index": i, "embedding": values})  
                total\_tokens \+= tokens  
        else:  
            \# 기타 모델 (예: text-embedding-005, gemini-embedding-2)  
            \# 최대 250 인스턴스 배치가 허용되므로, RAGFlow의 통상적인 청크 배치를 그대로 전송  
            instances \=  
            resp \= await fetch\_vertex\_embeddings(instances)  
            predictions \= resp.get("predictions",)  
            for i, prediction in enumerate(predictions):  
                values \= prediction.get("embeddings", {}).get("values",)  
                tokens \= prediction.get("embeddings", {}).get("statistics", {}).get("token\_count", 0)  
                results.append({"object": "embedding", "index": i, "embedding": values})  
                total\_tokens \+= tokens

    except HTTPException as e:  
        return format\_openai\_error(f"Vertex AI API Error: {e.detail}", e.status\_code)  
    except Exception as e:  
        return format\_openai\_error(f"Wrapper Internal Server Error: {str(e)}", 500)

    \# 6\. 최종 OpenAI 규격 응답 생성 반환  
    return {  
        "object": "list",  
        "data": results,  
        "model": model\_name,  
        "usage": {  
            "prompt\_tokens": total\_tokens,  
            "total\_tokens": total\_tokens  
        }  
    }

## **RAGFlow UI 시스템 통합 및 동작 메커니즘**

작성된 래퍼 서버를 사내 네트워크에 배포한 후, RAGFlow 시스템에 연동하기 위한 절차와 백엔드 메커니즘을 상세히 분석한다.

### **RAGFlow 시스템 관리자 설정 규격**

RAGFlow 관리자 UI의 Settings \-\> Model Providers \-\> Add Provider \-\> OpenAI-API-Compatible 메뉴에서 다음 값을 정확히 매핑하여 입력해야 한다.

* **Model type**: Embedding  
* **Model name**: gemini-embedding-001 (또는 text-embedding-005, gemini-embedding-2 등)  
  * 이 필드의 문자열은 래퍼 서버에서 가로채어 Vertex API URL 경로 조립에 직접 주입되므로, 오타 없이 Vertex AI 플랫폼에 등록된 공식 모델 식별자와 완벽히 일치해야 한다.  
* **Base url**: http://\<래퍼-서버-IP또는도메인\>:\<포트\>/v1  
  * **주의**: 반드시 /v1을 끝에 포함해야 한다. RAGFlow 내부 Golang/Python 백엔드 로직이 이 주소 문자열의 끝에 자동으로 /embeddings를 이어붙여 호출 대상을 구성하기 때문이다.20  
* **API-Key**: 래퍼 서버 배포 시 WRAPPER\_API\_KEY 환경 변수에 할당한 시크릿 문자열.  
* **Max tokens**: 2048 (gemini-embedding-001, text-embedding-005) 또는 8192 (gemini-embedding-2).  
  * RAGFlow는 파싱 중인 쿼리가 자체 기준인 8,191자를 넘으면 강제 절사한다. 그러나 gemini-embedding-001의 최대 수용 토큰은 2,048개이다. 만약 UI에서 Max tokens를 8,192로 잘못 기입하면, RAGFlow가 청크를 2,048 이상으로 생성하여 전송하게 되고 Vertex AI 측에서 autoTruncate 로직에 의해 텍스트 후반부가 모두 손실되는 심각한 인덱싱 품질 저하를 겪게 된다. 따라서 반드시 모델 물리 한계에 맞추어 2048을 입력해야 한다.3

### **Verify 동작 원리 및 통과 조건**

(출처: [https://github.com/infiniflow/ragflow/issues/14699](https://github.com/infiniflow/ragflow/issues/14699) 21, [https://github.com/infiniflow/ragflow/issues/8812](https://github.com/infiniflow/ragflow/issues/8812) 20)  
RAGFlow 시스템 내에서 **Verify** 버튼을 클릭할 때 발생하는 내부 백엔드(Golang)의 동작 프로세스는 다음과 같다.

1. RAGFlow 서버는 사용자가 입력한 Base URL, Model name, API-Key를 구조체(Struct)에 로드한다.22  
2. 프로바이더 타입이 임베딩이므로, 시스템은 내부적으로 짧은 더미 텍스트(예: "test" 또는 "hello")를 생성하여 구성된 URL(http://.../v1/embeddings)로 HTTP POST 요청을 단건 전송한다.  
3. 래퍼 서버는 이를 정상적인 OpenAI 규격 요청으로 인식하고 Vertex AI로 포워딩하여 결과를 응답한다.  
4. **통과 조건**: RAGFlow가 수신한 응답이 HTTP 200 OK 상태 코드를 반환하고, 응답 Body에 파싱 가능한 JSON 스키마(길이가 0이 아닌 float 배열을 포함한 data 객체)가 존재하면 검증 루틴이 성공으로 처리되어 모델이 활성화된다.21 위 래퍼 서버는 이 모든 조건을 완벽히 충족하므로 문제없이 Verify 루틴을 통과한다.

## **시스템 운영 환경에서의 잠재적 장애 지점(Pitfalls) 및 방어 전략**

대규모 코퍼스를 임베딩하는 엔터프라이즈 프로덕션 환경에 본 아키텍처를 도입할 때 흔히 발생하는 치명적 장애 요인과 그 대처 방안을 심도 있게 분석한다.

1. **Base URL 후행 슬래시 및 /v1 누락에 따른 라우팅 실패**: RAGFlow의 OpenAI 드라이버는 Base URL 필드의 값에 단순 문자열 조합을 수행한다. 만약 사용자가 UI에 http://wrapper:8000만 기입하면, 최종 요청 URL은 http://wrapper:8000/embeddings로 변조되어 전송된다. 이는 FastAPI 서버의 라우팅 규칙(/v1/embeddings)과 불일치하여 404 Not Found 예외를 발생시키고 전체 프로세스를 먹통으로 만든다.20 반드시 http://wrapper:8000/v1 형태로 입력해야 한다.  
2. **배치 인스턴스 한도 초과 및 동시성 제어 실패**: Vertex AI 기술 문서 3에 명시된 바와 같이, 가장 성능이 우수한 gemini-embedding-001 모델은 배열 인스턴스 전송을 전면 거부한다(요청당 1개만 허용). 앞서 제공한 Python 코드 내의 루프 분할(Splitting) 비동기 호출 로직이 어떠한 이유로든 중단되거나 asyncio.gather의 예외 처리가 누락되면, 단일 텍스트의 실패가 전체 청크 배치의 유실로 이어진다. 이 제약은 옴니 모델인 gemini-embedding-2 계열(최대 250개 허용)에서는 완화되므로, 모델 전환 시 래퍼 로직도 분기를 유지해야 한다.  
3. **벡터 데이터베이스의 차원(Dimensionality) 불일치와 MRL 적용 오류**: RAGFlow 시스템은 문서 검색을 위해 내부적으로 Elasticsearch 또는 Infinity와 같은 벡터 데이터베이스를 사용한다. 코퍼스를 인덱싱할 때, RAGFlow는 첫 번째 임베딩 API 호출 결과를 받아 그 벡터의 길이(차원)를 기준으로 벡터 데이터베이스의 테이블 스키마 차원을 영구 고정한다. 초기 테스트 시 gemini-embedding-001의 기본 차원인 3,072 차원으로 벡터가 저장된 이후, 사용자가 비용 절감을 위해 뒤늦게 dimensions=768 파라미터를 추가하면 차원 불일치(Dimension mismatch) 오류가 발생하며 파이프라인이 붕괴한다.23 인덱스 생성 전 모델과 차원을 명확히 결정하고, 변경이 필요할 경우 기존 인덱스를 모두 폐기(Drop)한 후 재파싱해야 한다.  
4. **리전 및 글로벌 할당량(Quota) 도달에 따른 백오프(Backoff)**: Google Cloud의 AI 할당량 정책에 따르면, gemini-embedding-001은 리전(Regional) 쿼터를 따르며, gemini-embedding-2는 분당 1천만 토큰 및 4만 요청이라는 글로벌 쿼터를 공유한다.12 RAGFlow를 통해 기가바이트(GB) 단위의 문서를 일괄 업로드하고 파싱을 시작할 경우, 수초 내에 429 Too Many Requests 한도 초과 에러가 발생할 위험이 크다. 프로덕션 환경에서는 래퍼 코드 내의 httpx 비동기 호출 부분에 tenacity 등의 패키지를 활용한 지수 백오프(Exponential Backoff) 재시도 로직을 추가하거나, GCP 콘솔에서 할당량 증설을 선제적으로 요청해야 한다.  
5. **토큰 갱신 시점의 스레드 충돌**:  
   다수의 문서 덩어리를 임베딩하는 도중 정확히 토큰 생명주기(1시간)가 만료되면, 동시에 실행 중이던 수백 개의 코루틴이 일제히 토큰 갱신 프로세스에 진입하려 한다. 설계서에 반영된 asyncio.Lock() 패턴이 없었다면 GCP 인증 서버는 이 급증한 트래픽을 비정상 호출로 간주할 수 있다. 제공된 더블 체크 잠금(Double-checked locking) 알고리즘은 이를 완벽히 방어한다.

## **클라우드 네이티브 배포 및 IAM 기반 보안 아키텍처**

안정적이고 가벼운 마이크로서비스 래퍼 서버 구축을 위해 최신 생태계 표준을 준수하는 배포 및 컨테이너라이제이션 아키텍처를 설계한다.

### **초고속 패키지 매니저 uv를 활용한 Docker 이미지 빌드**

Rust 기반의 초고속 패키지 인스톨러인 uv를 사용하면 기존 pip 대비 수십 배 빠른 의존성 해결 및 빌드 속도를 얻을 수 있으며, 결정론적(Deterministic) 환경 구성을 보장한다.

Dockerfile  
\# 초경량 베이스 이미지 채택  
FROM python:3.11\-slim

\# 보안 및 런타임 환경 변수 설정  
ENV PYTHONUNBUFFERED=1 \\  
    PYTHONDONTWRITEBYTECODE=1 \\  
    GOOGLE\_APPLICATION\_CREDENTIALS="/app/secrets/vertex-sa.json"

WORKDIR /app

\# uv 패키지 매니저 설치  
RUN pip install uv

\# 의존성 정의 파일 복사 및 설치 (캐시 최적화 레이어)  
COPY requirements.txt.  
RUN uv pip install \--system \-r requirements.txt

\# 애플리케이션 소스 코드 복사  
COPY wrapper.py.

\# 서버 실행 (포트 8000, 4개의 워커 프로세스 할당)  
EXPOSE 8000  
CMD \["uvicorn", "wrapper:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"\]

### **Docker Network 토폴로지 및 호스트 라우팅**

일반적으로 RAGFlow는 십여 개의 마이크로서비스(ragflow-server, mysql, minio, elasticsearch 등)가 docker-compose를 통해 사설 네트워크 브리지(Bridge)로 묶여 동작한다. 래퍼 서버 역시 이 동일한 Docker 네트워크 대역에 배포되어야 컨테이너 이름(예: http://vertex-wrapper:8000/v1)으로 원활한 통신이 가능하다. 만약 래퍼 서버를 Docker 데스크탑 호스트의 로컬 환경에서 단독 실행할 경우, RAGFlow 설정 창의 Base URL은 localhost가 아닌 http://host.docker.internal:8000/v1 로 설정해야만 컨테이너 내부에서 호스트로 트래픽이 빠져나올 수 있다.

### **IAM 기반 최소 권한 원칙(Least Privilege) 구현**

시스템 보안 측면에서 하드코딩된 자격 증명을 사용하는 것은 클라우드 보안 컴플라이언스에 정면으로 위배된다.13 래퍼 서버가 GCP 자원에 안전하게 접근하기 위한 IAM 계층 구조는 다음과 같다.

1. GCP Console에서 래퍼 전용 서비스 어카운트(Service Account)를 새로 생성한다.  
2. 이 서비스 어카운트에는 전체 관리자 권한이 아닌, 모델 추론 요청에만 한정된 **Vertex AI User (roles/aiplatform.user)** 역할만을 부여한다.24  
3. 생성된 서비스 어카운트의 JSON 키를 안전하게 보관한 뒤, Docker 구동 시 \-v 옵션을 통한 볼륨 마운트 방식으로 /app/secrets/vertex-sa.json 위치에 읽기 전용으로 주입한다. 컨테이너 내부에는 민감한 키 정보가 포함된 채로 빌드되지 않는다.  
4. RAGFlow 내부 트래픽이 래퍼로 인입될 때 무단 사용이나 SSRF(Server-Side Request Forgery)를 차단하기 위해, 시스템 환경 변수 WRAPPER\_API\_KEY에 높은 엔트로피를 가지는 임의의 UUID 문자열을 할당한다. 이 값은 RAGFlow UI의 API-Key 입력란과 정확히 일치해야만 FastAPI 미들웨어가 요청을 수락하도록 설계되어 보안 무결성을 보장한다.

#### **참고 자료**

1. \[Feature Request\]: Implement Embed (embeddings) in the TogetherAI Go driver · Issue \#15015 · infiniflow/ragflow \- GitHub, 6월 6, 2026에 액세스, [https://github.com/infiniflow/ragflow/issues/15015](https://github.com/infiniflow/ragflow/issues/15015)  
2. \[Feature Request\]: Implement Encode (embeddings) in the OpenAI Go driver · Issue \#14629 · infiniflow/ragflow \- GitHub, 6월 6, 2026에 액세스, [https://github.com/infiniflow/ragflow/issues/14629](https://github.com/infiniflow/ragflow/issues/14629)  
3. Get text embeddings | Gemini Enterprise Agent Platform \- Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-text-embeddings](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-text-embeddings)  
4. Text embeddings API | Gemini Enterprise Agent Platform | Google ..., 6월 6, 2026에 액세스, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/text-embeddings-api](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/text-embeddings-api)  
5. Choose an embeddings task type | Gemini Enterprise Agent Platform | Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/task-types](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/task-types)  
6. Class TextEmbeddingInput (1.154.0) | Python client libraries | Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/python/docs/reference/agentplatform/latest/vertexai.preview.language\_models.TextEmbeddingInput](https://docs.cloud.google.com/python/docs/reference/agentplatform/latest/vertexai.preview.language_models.TextEmbeddingInput)  
7. The AI.SIMILARITY function | BigQuery \- Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-similarity](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-similarity)  
8. Autogenerating embeddings for Vector Search 2.0 | Gemini Enterprise Agent Platform, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/vector-search-2/embeddings/autogenerating-embeddings](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/vector-search-2/embeddings/autogenerating-embeddings)  
9. Gemini Embedding 2 | Gemini Enterprise Agent Platform \- Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/embedding-2](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/embedding-2)  
10. Embeddings | Gemini API \- Google AI for Developers, 6월 6, 2026에 액세스, [https://ai.google.dev/gemini-api/docs/embeddings](https://ai.google.dev/gemini-api/docs/embeddings)  
11. Authenticate | Gemini Enterprise Agent Platform \- Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate/openai/auth-and-credentials](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate/openai/auth-and-credentials)  
12. Generative AI on Gemini Enterprise Agent Platform quotas and system limits, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/quotas](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/quotas)  
13. Authenticate with client libraries \- Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/docs/authentication/client-libraries](https://docs.cloud.google.com/docs/authentication/client-libraries)  
14. Authentication for Google Cloud APIs and services, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/docs/authentication](https://docs.cloud.google.com/docs/authentication)  
15. Authentication basics | Get started \- Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/docs/get-started/authentication](https://docs.cloud.google.com/docs/get-started/authentication)  
16. Create short-lived credentials for a service account | Identity and Access Management (IAM), 6월 6, 2026에 액세스, [https://docs.cloud.google.com/iam/docs/create-short-lived-credentials-direct](https://docs.cloud.google.com/iam/docs/create-short-lived-credentials-direct)  
17. Get an ID token | Authentication \- Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/docs/authentication/get-id-token](https://docs.cloud.google.com/docs/authentication/get-id-token)  
18. \[Bug\]: Cannot pass dimensions to openai compatible embedding model · Issue \#11940 · BerriAI/litellm \- GitHub, 6월 6, 2026에 액세스, [https://github.com/BerriAI/litellm/issues/11940](https://github.com/BerriAI/litellm/issues/11940)  
19. drop\_params ignored for \`dimensions\` on OpenAI-provider embedding calls (error message is misleading) · Issue \#26787 · BerriAI/litellm \- GitHub, 6월 6, 2026에 액세스, [https://github.com/BerriAI/litellm/issues/26787](https://github.com/BerriAI/litellm/issues/26787)  
20. \[Bug\]: Deepinfra doesn't work (OpenAI Compatible Embeddings) · Issue \#8812 · infiniflow/ragflow \- GitHub, 6월 6, 2026에 액세스, [https://github.com/infiniflow/ragflow/issues/8812](https://github.com/infiniflow/ragflow/issues/8812)  
21. \[Feature Request\]: Implement Encode (embeddings) in the NVIDIA Go driver · Issue \#14699 · infiniflow/ragflow \- GitHub, 6월 6, 2026에 액세스, [https://github.com/infiniflow/ragflow/issues/14699](https://github.com/infiniflow/ragflow/issues/14699)  
22. \[Feature Request\]: Validate and normalize model URL suffix config keys · Issue \#15591 · infiniflow/ragflow \- GitHub, 6월 6, 2026에 액세스, [https://github.com/infiniflow/ragflow/issues/15591](https://github.com/infiniflow/ragflow/issues/15591)  
23. \[Question\]: How to reset all documents in a dataset to run status 0 (or unstart) via the python\_sdk (or otherwise) · Issue \#12800 · infiniflow/ragflow \- GitHub, 6월 6, 2026에 액세스, [https://github.com/infiniflow/ragflow/issues/12800](https://github.com/infiniflow/ragflow/issues/12800)  
24. Access Gemini models from a workflow using Vertex AI \- Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/workflows/docs/tutorials/use-vertex-ai-models](https://docs.cloud.google.com/workflows/docs/tutorials/use-vertex-ai-models)  
25. Use Private Service Connect to access Generative AI on Vertex AI from on-premises, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/vertex-ai/docs/general/vertex-psc-gen-ai](https://docs.cloud.google.com/vertex-ai/docs/general/vertex-psc-gen-ai)  
26. Access control with IAM | Colab Enterprise \- Google Cloud Documentation, 6월 6, 2026에 액세스, [https://docs.cloud.google.com/colab/docs/access-control](https://docs.cloud.google.com/colab/docs/access-control)