"""래퍼의 순수 변환 로직 단위 테스트 (Vertex/GCP 호출 없이).

GoogleAccessTokenProvider는 import 시점이 아니라 lifespan에서 생성되므로,
ADC 자격증명 없이도 app 모듈을 import하고 순수 함수/응답 매핑을 검증할 수 있다.
실제 Vertex 호출 경로는 VertexEmbeddingClient.predict를 가짜로 주입해 테스트한다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as wrapper


# ---- 순수 함수 ----

def test_coerce_string_inputs_str():
    assert wrapper.coerce_string_inputs("hi") == ["hi"]


def test_coerce_string_inputs_list():
    assert wrapper.coerce_string_inputs(["a", "b"]) == ["a", "b"]


def test_coerce_string_inputs_rejects_tokens():
    with pytest.raises(ValueError):
        wrapper.coerce_string_inputs([[1, 2, 3]])


def test_chunked_splits_by_size():
    assert list(wrapper.chunked(["a", "b", "c", "d", "e"], 2)) == [["a", "b"], ["c", "d"], ["e"]]


def test_chunked_size_floor_is_one():
    # 한도 0이 들어와도 1로 보정
    assert list(wrapper.chunked(["a", "b"], 0)) == [["a"], ["b"]]


def test_gemini_001_max_instances_is_one():
    assert wrapper.KNOWN_MAX_INSTANCES["gemini-embedding-001"] == 1


def test_text_005_max_instances_is_five():
    assert wrapper.KNOWN_MAX_INSTANCES["text-embedding-005"] == 5


def test_status_mapping():
    assert wrapper.map_vertex_status_to_openai_type(429) == "rate_limit_error"
    assert wrapper.map_vertex_status_to_openai_type(503) == "api_error"
    assert wrapper.map_vertex_status_to_openai_type(400) == "invalid_request_error"


# ---- 엔드포인트 (Vertex 호출은 가짜로 대체) ----

class _FakeVertex:
    """predict()를 흉내. 각 instance(text)당 한 개의 prediction을 돌려준다."""

    def __init__(self):
        self.calls: list[list[str]] = []

    async def predict(self, *, model, texts, dimensions, task_type, title, auto_truncate):
        self.calls.append(list(texts))
        return [
            {"embeddings": {"values": [0.1, 0.2, 0.3], "statistics": {"token_count": 2}}}
            for _ in texts
        ]

    async def close(self):
        pass


@pytest.fixture
def client_with_fake(monkeypatch):
    fake = _FakeVertex()

    # lifespan이 GoogleAccessTokenProvider/VertexEmbeddingClient를 만들지 않도록 우회:
    # 테스트에서는 lifespan을 끄고 app.state를 직접 채운다.
    test_app = wrapper.app
    test_app.router.lifespan_context = _noop_lifespan
    with TestClient(test_app) as c:
        test_app.state.vertex_client = fake
        yield c, fake


from contextlib import asynccontextmanager


@asynccontextmanager
async def _noop_lifespan(app):
    yield


def test_embeddings_order_and_usage(client_with_fake):
    client, fake = client_with_fake
    r = client.post("/v1/embeddings", json={"model": "text-embedding-005", "input": ["a", "b", "c"]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    assert all(d["embedding"] == [0.1, 0.2, 0.3] for d in body["data"])
    assert body["usage"]["total_tokens"] == 6  # 3 텍스트 x 2 토큰


def test_gemini_001_splits_into_single_instance_calls(client_with_fake):
    client, fake = client_with_fake
    client.post("/v1/embeddings", json={"model": "gemini-embedding-001", "input": ["x", "y", "z"]})
    # 한도 1 -> 3번의 1개짜리 호출로 쪼개져야 함
    assert sorted(len(c) for c in fake.calls) == [1, 1, 1]


def test_text_005_splits_by_five(client_with_fake):
    client, fake = client_with_fake
    client.post("/v1/embeddings", json={"model": "text-embedding-005", "input": [str(i) for i in range(12)]})
    # 한도 5 -> 5,5,2
    assert sorted((len(c) for c in fake.calls), reverse=True) == [5, 5, 2]


def test_rejects_non_float_encoding(client_with_fake):
    client, _ = client_with_fake
    r = client.post("/v1/embeddings", json={"model": "text-embedding-005", "input": "a", "encoding_format": "base64"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_encoding_format"
