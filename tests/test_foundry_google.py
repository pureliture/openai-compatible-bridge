from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.providers.foundry import (
    FOUNDRY_GOOGLE_GENERATE_CONTENT_PROTOCOL,
    FoundryChatClient,
)
from openai_compatible_bridge.providers.vertex import VertexAPIError


BASE_URL = "https://foundry.test/api/v2/llm/proxy/openai/v1/chat/completions"
GOOGLE_CONFIG = {"protocol": FOUNDRY_GOOGLE_GENERATE_CONTENT_PROTOCOL}


def _client(handler: Any) -> FoundryChatClient:
    client = FoundryChatClient(base_url=BASE_URL, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def test_google_generate_content_maps_messages_parameters_and_tools():
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 4,
                    "totalTokenCount": 9,
                },
            },
        )

    client = _client(handler)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            },
        }
    ]

    async def run():
        result = await client.generate(
            model="gemini-3.8-flash",
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
            ],
            max_tokens=17,
            temperature=0.2,
            top_p=0.8,
            stop=["END"],
            response_format={"type": "json_object"},
            tools=tools,
            tool_choice="auto",
            resolved_config=GOOGLE_CONFIG,
        )
        await client.close()
        return result

    result = asyncio.run(run())
    body = captured["body"]
    assert captured["url"] == (
        "https://foundry.test/api/v2/llm/proxy/google/v1/models/"
        "gemini-3.8-flash:generateContent"
    )
    assert captured["headers"]["authorization"] == "Bearer test-token"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert body["systemInstruction"] == {"parts": [{"text": "Be concise."}]}
    assert body["generationConfig"] == {
        "maxOutputTokens": 17,
        "temperature": 0.2,
        "topP": 0.8,
        "stopSequences": ["END"],
        "responseMimeType": "application/json",
    }
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                }
            ]
        }
    ]
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    assert "thinkingConfig" not in body.get("generationConfig", {})
    assert result == {
        "text": "ok",
        "tool_calls": None,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 9},
    }


def test_google_generate_content_normalizes_function_call_and_tool_history():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "I will look it up."},
                                {"functionCall": {"name": "lookup", "args": {"key": "x"}}},
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 6, "totalTokenCount": 11},
            },
        )

    client = _client(handler)
    captured: dict[str, Any] = {}

    async def run():
        async def capture(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return await handler(request)

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(capture))
        result = await client.generate(
            model="gemini-3.8-flash",
            messages=[
                {"role": "user", "content": "find x"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-prev",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"old"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-prev", "content": '{"value": 1}'},
            ],
            resolved_config=GOOGLE_CONFIG,
        )
        await client.close()
        return result

    result = asyncio.run(run())
    body = captured["body"]
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "find x"}]},
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "lookup", "args": {"key": "old"}}}],
        },
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "lookup", "response": {"value": 1}}}],
        },
    ]
    assert result["text"] == "I will look it up."
    assert result["tool_calls"] == [
        {
            "id": "call_lookup_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"key": "x"}'},
        }
    ]
    assert result["finish_reason"] == "tool_calls"


def test_google_stream_generate_content_parses_google_sse():
    stream_body = (
        b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]}}]}\n\n'
        b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}],'
        b'"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}}\n\n'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(
            "/api/v2/llm/proxy/google/v1/models/gemini-3.8-flash:streamGenerateContent"
        )
        assert request.url.query == b"alt=sse"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_body)

    client = _client(handler)

    async def run():
        events = []
        async for event in client.stream_chat(
            model="gemini-3.8-flash",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8,
            resolved_config=GOOGLE_CONFIG,
        ):
            events.append(event)
        await client.close()
        return events

    events = asyncio.run(run())
    assert [event["delta_text"] for event in events] == ["hel", "lo"]
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["usage"]["total_tokens"] == 5


def test_google_protocol_rejects_reasoning_override_explicitly():
    async def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("reasoning override must be rejected before the upstream request")
        raise AssertionError("unreachable")

    client = _client(handler)
    try:
        with pytest.raises(VertexAPIError, match="reasoning controls"):
            asyncio.run(
                client.generate(
                    model="gemini-3.8-flash",
                    messages=[{"role": "user", "content": "hello"}],
                    reasoning_effort="high",
                    resolved_config=GOOGLE_CONFIG,
                )
            )
    finally:
        asyncio.run(client.close())


def test_foundry_registry_accepts_google_generate_content_protocol(monkeypatch):
    monkeypatch.setenv(
        "MODEL_REGISTRY_JSON",
        json.dumps(
            {
                "foundry:gemini-3.8-flash": {
                    "provider": "foundry",
                    "kind": "chat",
                    "provider_model": "gemini-3.8-flash",
                    "protocol": FOUNDRY_GOOGLE_GENERATE_CONTENT_PROTOCOL,
                }
            }
        ),
    )
    registry = vertex._build_registry()
    assert registry["foundry:gemini-3.8-flash"]["protocol"] == FOUNDRY_GOOGLE_GENERATE_CONTENT_PROTOCOL
