from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError


class _FakeProvider:
    async def close(self) -> None:
        pass


class _FakeFoundry(_FakeProvider):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "text": "hello from Foundry",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }

    async def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        yield {"delta_text": "hello", "finish_reason": None, "usage": None}
        yield {
            "delta_text": " from Foundry",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }


def _register_foundry_alias() -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY["foundry:gpt-6-astra"] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": "gpt-6-astra",
    }
    return old_registry


def _restore_registry(old_registry: dict[str, dict]) -> None:
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old_registry)


def test_foundry_build_request_maps_max_tokens_to_max_completion_tokens():
    body: dict = {}
    headers: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal body, headers
        body = json.loads(request.content)
        headers = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.test/chat", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = asyncio.run(
            client.generate(
                model="gpt-6-astra",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=17,
            )
        )
    finally:
        asyncio.run(client.close())

    assert body["model"] == "gpt-6-astra"
    assert body["max_completion_tokens"] == 17
    assert "max_tokens" not in body
    assert headers["authorization"] == "Bearer test-token"
    assert result["text"] == "ok"
    assert result["usage"]["total_tokens"] == 5


def test_foundry_stream_parses_sse_usage_and_eof_without_done_marker():
    stream_body = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":"hel"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n'
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_body)

    client = FoundryChatClient(base_url="https://foundry.test/chat", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def collect():
        events = []
        async for event in client.stream_chat(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8,
        ):
            events.append(event)
        await client.close()
        return events

    events = asyncio.run(collect())
    assert [event["delta_text"] for event in events] == ["hel", "lo"]
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["usage"]["total_tokens"] == 5


def test_foundry_error_unwraps_palantir_openai_error():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "errorCode": "CUSTOM_CLIENT",
                "errorName": "LanguageModelService:LlmHttpClientError",
                "parameters": {
                    "responseBody": (
                        'Optional[{"error":{"message":"Unsupported parameter: max_tokens",'
                        '"type":"invalid_request_error","code":"unsupported_parameter"}}]'
                    ),
                    "errorCode": "Optional[unsupported_parameter]",
                },
            },
        )

    client = FoundryChatClient(base_url="https://foundry.test/chat", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(VertexAPIError) as excinfo:
            asyncio.run(
                client.generate(
                    model="gpt-6-astra",
                    messages=[{"role": "user", "content": "hello"}],
                )
            )
    finally:
        asyncio.run(client.close())

    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "unsupported_parameter"
    assert "max_tokens" in excinfo.value.message


def test_foundry_alias_routes_non_stream_and_native_max_completion_tokens():
    fake_foundry = _FakeFoundry()
    old_registry = _register_foundry_alias()
    try:
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_FakeProvider,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=_FakeProvider,
            foundry_chat_client_factory=lambda: fake_foundry,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": 23,
                },
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello from Foundry"
        assert fake_foundry.calls[0]["model"] == "gpt-6-astra"
        assert fake_foundry.calls[0]["max_tokens"] == 23
    finally:
        _restore_registry(old_registry)


def test_foundry_alias_streams_openai_sse():
    fake_foundry = _FakeFoundry()
    old_registry = _register_foundry_alias()
    try:
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_FakeProvider,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=_FakeProvider,
            foundry_chat_client_factory=lambda: fake_foundry,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ) as response:
                body = response.read().decode()

        assert response.status_code == 200
        assert '"model": "foundry:gpt-6-astra"' in body
        assert '"content": "hello"' in body
        assert "data: [DONE]" in body
    finally:
        _restore_registry(old_registry)


def test_model_registry_accepts_foundry_provider(monkeypatch):
    custom = json.dumps(
        {
            "foundry:gpt-6-astra": {
                "provider": "foundry",
                "kind": "chat",
                "provider_model": "gpt-6-astra",
            }
        }
    )
    monkeypatch.setenv("MODEL_REGISTRY_JSON", custom)
    registry = vertex._build_registry()
    cfg = registry.get("foundry:gpt-6-astra")
    assert cfg is not None
    assert cfg["provider"] == "foundry"
    assert cfg["provider_model"] == "gpt-6-astra"
