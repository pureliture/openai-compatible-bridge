"""Registry and bridge contracts; MockTransport never calls real Foundry."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.foundry import FoundryChatClient

REGISTRY = Path(__file__).resolve().parents[1] / "examples/foundry-sonnet55-sol61.json"
CASES = (
    ("foundry:claude-sonnet-5.5", "ri.language-model-service..language-model.anthropic-claude-5-5-sonnet", "anthropic_messages", "/api/v2/llm/proxy/anthropic/v1/messages"),
    ("foundry:gpt-6.1-sol", "ri.language-model-service..language-model.gpt-6-1-sol", "openai_responses", "/api/v2/llm/proxy/openai/v1/responses"),
)
BASE = "https://foundry.test/api/v2/llm/proxy/openai/v1/chat/completions"
TOOL = {"type": "function", "function": {"name": "echo", "parameters": {
    "type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"],
}}}


class NoopClient:
    async def close(self):
        pass


def fragment():
    assert REGISTRY.is_file(), "new model registry fragment is missing"
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_registry_adds_only_requested_models_and_preserves_existing(monkeypatch):
    additions = fragment()
    assert set(additions) == {case[0] for case in CASES}
    existing = {"foundry:gpt-6-sol": {
        "provider": "foundry", "kind": "chat", "protocol": "openai_responses",
        "provider_model": "ri.language-model-service..language-model.gpt-6-sol",
    }}
    monkeypatch.setenv("MODEL_REGISTRY_JSON", json.dumps({**existing, **additions}))
    registry = vertex._build_registry()
    assert registry["foundry:gpt-6-sol"] == existing["foundry:gpt-6-sol"]
    for alias, rid, protocol, _ in CASES:
        assert registry[alias] == {"provider": "foundry", "kind": "chat", "protocol": protocol, "provider_model": rid}
    for alias, config in vertex._BUILTIN_REGISTRY.items():
        assert all(registry[alias][key] == value for key, value in config.items())


def upstream_response(protocol, stream, tool):
    if protocol == "anthropic_messages":
        content = {"type": "tool_use", "id": "call_1", "name": "echo", "input": {"value": "ok"}} if tool else {"type": "text", "text": "ok"}
        if not stream:
            return httpx.Response(200, json={"content": [content], "stop_reason": "tool_use" if tool else "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2}})
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 3, "output_tokens": 0}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        ]
    else:
        output = {"type": "function_call", "call_id": "call_1", "name": "echo", "arguments": '{"value":"ok"}'} if tool else {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
        if not stream:
            return httpx.Response(200, json={"status": "completed", "output": [output], "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}})
        events = [
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}},
        ]
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join(f"data: {json.dumps(event)}\n\n" for event in events))


def app_with_transport(monkeypatch, handler):
    monkeypatch.setenv("MODEL_REGISTRY_JSON", json.dumps(fragment()))
    monkeypatch.setattr(vertex, "MODEL_REGISTRY", vertex._build_registry())
    foundry = FoundryChatClient(base_url=BASE, token="synthetic-test-token")
    # Close the original unused client so the mocked transport owns all requests.
    import asyncio
    asyncio.run(foundry.http.aclose())
    foundry.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return create_app(
        embedding_client_factory=NoopClient, chat_client_factory=NoopClient,
        rerank_client_factory=NoopClient, ollama_chat_client_factory=NoopClient,
        foundry_chat_client_factory=lambda: foundry, cost_accounting_factory=lambda: None,
    )


@pytest.mark.parametrize("alias,rid,protocol,path", CASES)
@pytest.mark.parametrize("stream", (False, True), ids=("non-stream", "stream"))
def test_new_alias_routes_through_bridge(monkeypatch, alias, rid, protocol, path, stream):
    captured = []

    async def handler(request):
        captured.append(request)
        return upstream_response(protocol, stream, False)

    with TestClient(app_with_transport(monkeypatch, handler)) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert alias in {item["id"] for item in models.json()["data"]}
        response = client.post("/v1/chat/completions", json={
            "model": alias, "messages": [{"role": "user", "content": "say ok"}],
            "max_completion_tokens": 64, "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
        })
    assert response.status_code == 200
    if stream:
        frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        assert frames[-1] == "[DONE]"
        events = [json.loads(frame) for frame in frames[:-1]]
        assert any(event.get("choices") and event["choices"][0]["delta"].get("content") == "ok" for event in events)
        assert any(event.get("usage", {}).get("total_tokens") == 5 for event in events)
        assert all(event["model"] == alias for event in events)
    else:
        payload = response.json()
        assert payload["model"] == alias
        assert payload["choices"][0]["message"]["content"] == "ok"
        assert payload["usage"]["total_tokens"] == 5
    assert len(captured) == 1
    request = captured[0]
    assert request.url.path == path
    assert request.headers["authorization"] == "Bearer synthetic-test-token"
    body = json.loads(request.content)
    assert body["model"] == rid
    assert body["stream"] is stream
    assert body["max_tokens" if protocol == "anthropic_messages" else "max_output_tokens"] == 64


@pytest.mark.parametrize("alias,rid,protocol,path", CASES)
def test_new_alias_preserves_function_tool_call(monkeypatch, alias, rid, protocol, path):
    captured = []

    async def handler(request):
        captured.append(request)
        return upstream_response(protocol, False, True)

    with TestClient(app_with_transport(monkeypatch, handler)) as client:
        response = client.post("/v1/chat/completions", json={
            "model": alias, "messages": [{"role": "user", "content": "echo ok"}],
            "max_completion_tokens": 64, "tools": [TOOL], "tool_choice": "auto",
        })
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "echo"
    assert json.loads(call["function"]["arguments"]) == {"value": "ok"}
    assert len(captured) == 1
    assert captured[0].url.path == path
    body = json.loads(captured[0].content)
    assert body["model"] == rid
    assert body["tools"][0]["name"] == "echo"
