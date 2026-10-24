"""Anthropic ingress (POST /v1/messages) + Vertex function calling tests."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.main as bridge_main
from openai_compatible_bridge.anthropic_ingress import (
    AnthropicBadRequest,
    AnthropicStreamState,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    new_anthropic_message_id,
    openai_result_to_anthropic,
    parse_anthropic_chat_params,
)
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.vertex import VertexChatClient


class FakeTokens:
    def __init__(self) -> None:
        self.project_id = "test-project"

    async def get_token(self) -> str:
        return "fake-token"


def _vertex_client(handler) -> VertexChatClient:
    client = VertexChatClient(token_provider=FakeTokens())  # type: ignore[arg-type]
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _app_with(chat_client):
    class FakeEmbedding:
        async def close(self): ...

    class FakeRerank:
        async def close(self): ...

    return create_app(
        embedding_client_factory=lambda: FakeEmbedding(),
        chat_client_factory=lambda: chat_client,
        rerank_client_factory=lambda: FakeRerank(),
        ollama_chat_client_factory=lambda: chat_client,
        foundry_chat_client_factory=lambda: chat_client,
        cost_accounting_factory=lambda: None,
    )


# ---------------------------------------------------------------------------
# translation unit tests
# ---------------------------------------------------------------------------


def test_parse_basic_text_request():
    params = parse_anthropic_chat_params(
        {
            "model": "gemini-2.5-flash",
            "max_tokens": 64,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert params["model"] == "gemini-2.5-flash"
    assert params["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert params["messages"][1] == {"role": "user", "content": "hi"}
    assert params["stream"] is False
    assert params["tools"] is None


def test_parse_tool_use_and_tool_result_roundtrip():
    params = parse_anthropic_chat_params(
        {
            "model": "m",
            "max_tokens": 16,
            "tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Seoul"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "sunny"}],
                },
            ],
        }
    )
    assistant = params["messages"][1]
    assert assistant["tool_calls"][0]["id"] == "tu_1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Seoul"}
    tool_msg = params["messages"][2]
    assert tool_msg == {"role": "tool", "tool_call_id": "tu_1", "content": "sunny"}
    assert params["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert params["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_parse_rejects_media_and_bad_shapes():
    with pytest.raises(AnthropicBadRequest):
        parse_anthropic_chat_params({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    with pytest.raises(AnthropicBadRequest):
        parse_anthropic_chat_params(
            {
                "model": "m",
                "max_tokens": 8,
                "messages": [
                    {"role": "user", "content": [{"type": "image", "source": {"type": "base64"}}]}
                ],
            }
        )
    with pytest.raises(AnthropicBadRequest):
        anthropic_tools_to_openai([{"name": "", "input_schema": {}}])
    with pytest.raises(AnthropicBadRequest):
        anthropic_tool_choice_to_openai({"type": "mystery"})


def test_tool_choice_variants():
    assert anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"
    assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"
    assert anthropic_tool_choice_to_openai({"type": "none"}) == "none"
    assert anthropic_tool_choice_to_openai(None) is None


def test_openai_result_to_anthropic_maps_tools_and_stop():
    body = openai_result_to_anthropic(
        {
            "text": None,
            "tool_calls": [
                {"id": "call_x_0", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city":"Seoul"}'}},
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        },
        model="m",
        message_id=new_anthropic_message_id(),
    )
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0]["type"] == "tool_use"
    assert body["content"][0]["input"] == {"city": "Seoul"}
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 5}


def test_stream_state_text_and_tool_buffering():
    state = AnthropicStreamState(message_id="msg_1", model="m")
    chunks = state.feed({"delta_text": "hel", "finish_reason": None, "usage": None})
    assert any("message_start" in c for c in chunks)
    chunks = state.feed(
        {
            "delta_text": "",
            "delta_tool_calls": [{"index": 0, "id": "call_x_0",
                                  "function": {"name": "w", "arguments": '{"a":1}'}}],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }
    )
    tail = state.finish(prompt_tokens=3, completion_text="hel")
    joined = "".join(chunks + tail)
    assert "content_block_start" in joined
    assert "input_json_delta" in joined
    assert '"stop_reason": "tool_use"' in joined


# ---------------------------------------------------------------------------
# vertex function calling unit tests
# ---------------------------------------------------------------------------


def test_vertex_body_maps_tools_and_tool_messages():
    body = VertexChatClient._build_request_body(
        messages=[
            {"role": "user", "content": "weather in Seoul?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Seoul"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        max_tokens=32,
        temperature=None,
        top_p=None,
        stop=None,
        tools=[{"type": "function", "function": {"name": "get_weather", "description": "d",
                                                 "parameters": {"type": "object"}}}],
        tool_choice="auto",
    )
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    assert "toolConfig" not in body
    model_turns = [c for c in body["contents"] if c["role"] == "model"]
    assert model_turns[0]["parts"][0] == {"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}
    tool_turns = [c for c in body["contents"] if any("functionResponse" in p for p in c["parts"])]
    assert tool_turns[0]["parts"][0]["functionResponse"]["name"] == "get_weather"


def test_vertex_generate_parses_function_call():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5, "totalTokenCount": 8},
            },
        )

    async def run():
        client = _vertex_client(handler)
        try:
            return await client.generate(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "weather?"}],
                max_tokens=32,
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
                resolved_config={"api": "generateContent", "location": "us-central1"},
            )
        finally:
            await client.close()

    result = asyncio.run(run())
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert result["tool_calls"][0]["id"].startswith("call_get_weather_")
    assert result["text"] is None


# ---------------------------------------------------------------------------
# route tests
# ---------------------------------------------------------------------------


def test_messages_route_non_stream_text_and_auth():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
            },
        )

    app = _app_with(_vertex_client(handler))
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "x"},
            json={"model": "gemini-2.5-flash", "max_tokens": 16,
                   "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "hello"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 2, "output_tokens": 3}


def test_messages_route_tools_end_to_end_and_rejects():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert payload["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {}}}]},
                     "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5, "totalTokenCount": 8},
            },
        )

    app = _app_with(_vertex_client(handler))
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "gemini-2.5-flash",
                "max_tokens": 16,
                "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
                "messages": [{"role": "user", "content": "weather?"}],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0]["name"] == "get_weather"

    with TestClient(app) as client:
        bad = client.post("/v1/messages", json={"model": "nope", "max_tokens": 4,
                                                "messages": [{"role": "user", "content": "x"}]})
        assert bad.status_code == 404
        assert bad.json()["type"] == "error"
        missing = client.post("/v1/messages", json={"model": "gemini-2.5-flash",
                                                     "messages": [{"role": "user", "content": "x"}]})
        assert missing.status_code == 400


def test_messages_route_stream_sse_shape():
    wire = (
        b'data: {"candidates": [{"content": {"parts": [{"text": "hel"}]}}]}\n\n'
        b'data: {"candidates": [{"content": {"parts": [{"text": "lo"}]}, "finishReason": "STOP"}], '
        b'"usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire)

    app = _app_with(_vertex_client(handler))
    with TestClient(app) as client:
        with client.stream(
            "POST", "/v1/messages",
            json={"model": "gemini-2.5-flash", "max_tokens": 16, "stream": True,
                   "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            assert response.status_code == 200
            text = response.read().decode()
    assert '"type": "message_start"' in text
    assert '"type": "text_delta"' in text
    assert '"type": "message_stop"' in text
    assert '"stop_reason": "end_turn"' in text
    assert "[DONE]" not in text


def test_messages_route_ollama_tools_rejected_and_models_metadata(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 2,
                   "done": True, "done_reason": "stop"},
        )

    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    ollama = OllamaChatClient(base_url="http://ollama.test")
    ollama.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = _app_with(ollama)
    monkeypatch.setattr(bridge_main, "BRIDGE_API_KEY", "secret")
    with TestClient(app) as client:
        ok = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer secret"},
            json={"model": "ollama:llama3.1", "max_tokens": 8,
                   "messages": [{"role": "user", "content": "hi"}]},
        )
        assert ok.status_code == 200
        denied = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer wrong"},
            json={"model": "ollama:llama3.1", "max_tokens": 8,
                   "messages": [{"role": "user", "content": "hi"}]},
        )
        assert denied.status_code == 401
        tools = client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer secret"},
            json={"model": "ollama:llama3.1", "max_tokens": 8,
                   "tools": [{"name": "w", "input_schema": {"type": "object"}}],
                   "messages": [{"role": "user", "content": "hi"}]},
        )
        assert tools.status_code == 400
        models = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        entry = next(m for m in models.json()["data"] if m["id"] == "gemini-2.5-flash")
        assert entry["context_length"] >= 1
        assert entry["capabilities"]["streaming"] is True
    asyncio.run(ollama.close())
