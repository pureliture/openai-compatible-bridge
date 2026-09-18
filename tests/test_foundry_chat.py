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


def _register_foundry_alias(
    *,
    protocol: str = "openai_chat_completions",
    provider_model: str = "gpt-6-astra",
) -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY["foundry:gpt-6-astra"] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": provider_model,
        "protocol": protocol,
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
                temperature=0,
            )
        )
    finally:
        asyncio.run(client.close())

    assert body["model"] == "gpt-6-astra"
    assert body["max_completion_tokens"] == 17
    assert "max_tokens" not in body
    assert "temperature" not in body
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
        assert fake_foundry.calls[0]["resolved_config"]["protocol"] == "openai_chat_completions"
    finally:
        _restore_registry(old_registry)


def test_foundry_alias_stream_passes_provider_specific_protocol():
    fake_foundry = _FakeFoundry()
    old_registry = _register_foundry_alias(
        protocol="anthropic_messages",
        provider_model="claude-sonnet-5",
    )
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
                response.read()

        assert response.status_code == 200
        assert fake_foundry.calls[0]["model"] == "claude-sonnet-5"
        assert fake_foundry.calls[0]["resolved_config"]["protocol"] == "anthropic_messages"
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
    assert cfg["protocol"] == "openai_chat_completions"


def test_foundry_anthropic_and_xai_non_stream_normalization():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/anthropic/v1/messages"):
            return httpx.Response(
                200,
                json={
                    "type": "message",
                    "model": "claude-sonnet-5",
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": "claude ok"},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 4, "output_tokens": 3},
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "grok-4.6",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "grok ok"}],
                    },
                ],
                "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
            },
        )

    client = FoundryChatClient(
        base_url="https://foundry.test/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def collect():
        anthropic = await client.generate(
            model="claude-sonnet-5",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            max_tokens=17,
            temperature=0.2,
            resolved_config={"protocol": "anthropic_messages"},
        )
        xai = await client.generate(
            model="grok-4.6",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=19,
            reasoning_effort="low",
            resolved_config={"protocol": "xai_responses"},
        )
        await client.close()
        return anthropic, xai

    anthropic, xai = asyncio.run(collect())
    assert anthropic["text"] == "claude ok"
    assert anthropic["finish_reason"] == "stop"
    assert anthropic["usage"]["total_tokens"] == 7
    assert xai["text"] == "grok ok"
    assert xai["finish_reason"] == "stop"
    assert xai["usage"]["total_tokens"] == 9
    assert requests[0].url.path.endswith("/anthropic/v1/messages")
    assert json.loads(requests[0].content)["max_tokens"] == 17
    assert "temperature" not in json.loads(requests[0].content)
    assert requests[0].headers["anthropic-version"] == "2023-06-01"
    assert requests[1].url.path.endswith("/xai/v1/responses")
    assert json.loads(requests[1].content)["max_output_tokens"] == 19
    assert json.loads(requests[1].content)["reasoning"] == {"effort": "low"}


def test_foundry_anthropic_and_xai_stream_normalization():
    anthropic_stream = (
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":4,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"claude"}}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
    )
    xai_stream = (
        b'data: {"type":"response.output_text.delta","delta":"grok"}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":4,"total_tokens":9}}}\n\n'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = request.url.path.endswith("/anthropic/v1/messages")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=anthropic_stream if body else xai_stream,
        )

    client = FoundryChatClient(
        base_url="https://foundry.test/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def collect(protocol: str, model: str):
        events = []
        async for event in client.stream_chat(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=12,
            resolved_config={"protocol": protocol},
        ):
            events.append(event)
        return events

    async def run():
        anthropic = await collect("anthropic_messages", "claude-sonnet-5")
        xai = await collect("xai_responses", "grok-4.6")
        await client.close()
        return anthropic, xai

    anthropic, xai = asyncio.run(run())
    assert [event["delta_text"] for event in anthropic if event["delta_text"]] == ["claude"]
    assert anthropic[-1]["finish_reason"] == "stop"
    assert anthropic[-1]["usage"]["total_tokens"] == 7
    assert [event["delta_text"] for event in xai if event["delta_text"]] == ["grok"]
    assert xai[-1]["finish_reason"] == "stop"
    assert xai[-1]["usage"]["total_tokens"] == 9


def test_foundry_registry_rejects_unknown_protocol(monkeypatch):
    monkeypatch.setenv(
        "MODEL_REGISTRY_JSON",
        json.dumps(
            {
                "foundry:bad": {
                    "provider": "foundry",
                    "kind": "chat",
                    "provider_model": "bad",
                    "protocol": "arbitrary_url",
                }
            }
        ),
    )
    with pytest.raises(ValueError, match="invalid Foundry protocol"):
        vertex._build_registry()
