import json
from typing import Any
import pytest
from fastapi.testclient import TestClient

from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.vertex import VertexAPIError


class _FakeStreamChatClient:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(self, **kwargs: Any):
        self.calls.append(kwargs)
        for event in self._events:
            yield event

    async def close(self) -> None:
        pass


class _FakeProvider:
    async def close(self) -> None:
        pass


def _parse_sse_chunks(raw_body: str) -> list[dict[str, Any]]:
    chunks = []
    for line in raw_body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        chunks.append(json.loads(data))
    return chunks


def test_stream_without_stream_options_emits_no_usage_chunk_by_default():
    fake_client = _FakeStreamChatClient([
        {"delta_text": "Hello", "finish_reason": None, "usage": None},
        {"delta_text": " world", "finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
    ])
    app = create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=lambda: fake_client,
        rerank_client_factory=_FakeProvider,
        ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider,
        cost_accounting_factory=lambda: None,
    )
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()

    chunks = _parse_sse_chunks(body)
    assert any(c.get("choices") and c["choices"][0].get("finish_reason") == "stop" for c in chunks)
    # By default without stream_options.include_usage, no usage chunk with choices: []
    usage_chunks = [c for c in chunks if c.get("choices") == [] and "usage" in c]
    assert len(usage_chunks) == 0


def test_stream_with_include_usage_emits_standard_usage_chunk():
    fake_client = _FakeStreamChatClient([
        {"delta_text": "Hello", "finish_reason": None, "usage": None},
        {"delta_text": " world", "finish_reason": "stop", "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20}},
    ])
    app = create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=lambda: fake_client,
        rerank_client_factory=_FakeProvider,
        ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider,
        cost_accounting_factory=lambda: None,
    )
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()

    chunks = _parse_sse_chunks(body)
    usage_chunks = [c for c in chunks if c.get("choices") == [] and "usage" in c]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"] == {
        "prompt_tokens": 15,
        "completion_tokens": 5,
        "total_tokens": 20,
    }
    assert usage_chunks[0]["model"] == "gemini-2.5-flash"
    assert "data: [DONE]" in body


def test_stream_with_include_usage_fallback_estimate_when_upstream_omits_usage():
    fake_client = _FakeStreamChatClient([
        {"delta_text": "Alpha ", "finish_reason": None, "usage": None},
        {"delta_text": "Beta", "finish_reason": "stop", "usage": None},
    ])
    app = create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=lambda: fake_client,
        rerank_client_factory=_FakeProvider,
        ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider,
        cost_accounting_factory=lambda: None,
    )
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hello world test"}],
            },
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()

    chunks = _parse_sse_chunks(body)
    usage_chunks = [c for c in chunks if c.get("choices") == [] and "usage" in c]
    assert len(usage_chunks) == 1
    u = usage_chunks[0]["usage"]
    assert u["prompt_tokens"] > 0
    assert u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_stream_include_usage_default_env(monkeypatch):
    monkeypatch.setenv("STREAM_INCLUDE_USAGE_DEFAULT", "true")
    # Re-evaluate STREAM_INCLUDE_USAGE_DEFAULT
    from openai_compatible_bridge import main as main_mod
    monkeypatch.setattr(main_mod, "STREAM_INCLUDE_USAGE_DEFAULT", True)

    fake_client = _FakeStreamChatClient([
        {"delta_text": "Hi", "finish_reason": "stop", "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}},
    ])
    app = create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=lambda: fake_client,
        rerank_client_factory=_FakeProvider,
        ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider,
        cost_accounting_factory=lambda: None,
    )
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-flash",
                "stream": True,
                # No stream_options provided
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()

    chunks = _parse_sse_chunks(body)
    usage_chunks = [c for c in chunks if c.get("choices") == [] and "usage" in c]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"]["total_tokens"] == 10


def test_foundry_stream_with_include_usage():
    import openai_compatible_bridge.providers.vertex as vertex_mod

    old_registry = vertex_mod.MODEL_REGISTRY.copy()
    vertex_mod.MODEL_REGISTRY["foundry:gpt-6-astra"] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": "gpt-6-astra",
        "protocol": "openai_chat_completions",
    }

    fake_foundry = _FakeStreamChatClient([
        {"delta_text": "Foundry answer", "finish_reason": "stop", "usage": {"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50}},
    ])
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
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hello foundry"}],
                },
            ) as response:
                assert response.status_code == 200
                body = response.read().decode()

        chunks = _parse_sse_chunks(body)
        usage_chunks = [c for c in chunks if c.get("choices") == [] and "usage" in c]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"] == {
            "prompt_tokens": 42,
            "completion_tokens": 8,
            "total_tokens": 50,
        }
        assert usage_chunks[0]["model"] == "foundry:gpt-6-astra"
    finally:
        vertex_mod.MODEL_REGISTRY.clear()
        vertex_mod.MODEL_REGISTRY.update(old_registry)
