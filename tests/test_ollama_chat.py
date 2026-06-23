from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

import openai_compatible_bridge.providers.vertex as vertex
import openai_compatible_bridge.main as bridge_main
from openai_compatible_bridge.core.cost_tracking import CostLedger
from openai_compatible_bridge.main import create_app


class _FakeProvider:
    async def close(self) -> None:
        pass


class _UnexpectedVertexChat(_FakeProvider):
    async def generate(self, **_kwargs):
        raise AssertionError("Vertex chat provider should not handle Ollama aliases")


class _FakeOllamaChat(_FakeProvider):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "text": "hello from ollama",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }

    async def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        yield {"delta_text": "hel", "finish_reason": None, "usage": None}
        yield {
            "delta_text": "lo",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }


class _FailingStreamOllamaChat(_FakeProvider):
    async def generate(self, **_kwargs):
        raise AssertionError("streaming test should not call generate")

    async def stream_chat(self, **_kwargs):
        from openai_compatible_bridge.providers.vertex import VertexAPIError

        raise VertexAPIError(502, "ollama stream failed", code="connection_error")
        yield {}


def _register_ollama_alias():
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY["llama-local"] = {
        "provider": "ollama",
        "kind": "chat",
        "provider_model": "llama3.1",
    }
    return old_registry


def _restore_registry(old_registry):
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old_registry)


def _ollama_pricing_json() -> str:
    return """
    {
      "source": "unit-test",
      "version": "2026-06-22",
      "currency": "USD",
      "models": {
        "llama-local": {
          "chat": {
            "input_per_million": "0.10",
            "output_per_million": "0.20"
          }
        }
      }
    }
    """


def _enable_ollama_cost_tracking(monkeypatch, tmp_path, *, pricing_json: str | None = None):
    ledger_path = tmp_path / "ollama-cost.db"
    monkeypatch.setenv("COST_TRACKING_ENABLED", "true")
    monkeypatch.setenv("COST_LEDGER_PATH", str(ledger_path))
    monkeypatch.setenv("COST_PRICING_JSON", pricing_json or _ollama_pricing_json())
    monkeypatch.setenv("COST_SHORT_WINDOW_SECONDS", "60")
    monkeypatch.setenv("COST_SHORT_WINDOW_LIMIT_USD", "1.00")
    monkeypatch.setenv("COST_DAILY_LIMIT_USD", "10.00")
    return ledger_path


def test_ollama_response_format_json_object_maps_to_json():
    from openai_compatible_bridge.providers.ollama import _ollama_format_from_response_format

    assert _ollama_format_from_response_format({"type": "json_object"}) == "json"


def test_ollama_response_format_json_schema_maps_to_schema_only():
    from openai_compatible_bridge.providers.ollama import _ollama_format_from_response_format

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    result = _ollama_format_from_response_format(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "ExtractedEntities",
                "schema": schema,
                "strict": True,
            },
        }
    )

    assert result == schema
    assert result is schema
    assert "name" not in result
    assert "strict" not in result


def test_ollama_response_format_json_schema_missing_schema_errors():
    from openai_compatible_bridge.providers.ollama import _ollama_format_from_response_format
    from openai_compatible_bridge.providers.vertex import VertexAPIError

    with pytest.raises(VertexAPIError) as excinfo:
        _ollama_format_from_response_format(
            {
                "type": "json_schema",
                "json_schema": {"name": "ExtractedEntities", "strict": True},
            }
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "invalid_request"


def test_ollama_chat_alias_routes_to_ollama_provider():
    fake_ollama = _FakeOllamaChat()
    old_registry = _register_ollama_alias()
    try:
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_UnexpectedVertexChat,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=lambda: fake_ollama,
            cost_accounting_factory=lambda: None,
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-local",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "llama-local"
        assert body["choices"][0]["message"]["content"] == "hello from ollama"
        assert fake_ollama.calls[0]["model"] == "llama3.1"
    finally:
        _restore_registry(old_registry)


def test_dynamic_ollama_chat_model_routes_without_registry_alias():
    fake_ollama = _FakeOllamaChat()
    app = create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=_UnexpectedVertexChat,
        rerank_client_factory=_FakeProvider,
        ollama_chat_client_factory=lambda: fake_ollama,
        cost_accounting_factory=lambda: None,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama:minimax-m3:cloud",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "ollama:minimax-m3:cloud"
    assert body["choices"][0]["message"]["content"] == "hello from ollama"
    assert fake_ollama.calls[0]["model"] == "minimax-m3:cloud"


def test_dynamic_ollama_chat_model_requires_native_model():
    fake_ollama = _FakeOllamaChat()
    app = create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=_UnexpectedVertexChat,
        rerank_client_factory=_FakeProvider,
        ollama_chat_client_factory=lambda: fake_ollama,
        cost_accounting_factory=lambda: None,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama:",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_model"
    assert error["param"] == "model"
    assert fake_ollama.calls == []


def test_dynamic_ollama_models_are_not_listed():
    app = create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=_UnexpectedVertexChat,
        rerank_client_factory=_FakeProvider,
        ollama_chat_client_factory=_FakeProvider,
        cost_accounting_factory=lambda: None,
    )

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}
    assert "ollama:minimax-m3:cloud" not in model_ids


def test_ollama_chat_alias_streams_openai_sse():
    fake_ollama = _FakeOllamaChat()
    old_registry = _register_ollama_alias()
    try:
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_UnexpectedVertexChat,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=lambda: fake_ollama,
            cost_accounting_factory=lambda: None,
        )

        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "llama-local",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ) as response:
                body = response.read().decode()

        assert response.status_code == 200
        assert '"model": "llama-local"' in body
        assert '"content": "hel"' in body
        assert '"content": "lo"' in body
        assert "data: [DONE]" in body
        assert fake_ollama.calls[0]["model"] == "llama3.1"
    finally:
        _restore_registry(old_registry)


def test_ollama_stream_error_with_cost_disabled_returns_sse_error():
    old_registry = _register_ollama_alias()
    try:
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_UnexpectedVertexChat,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=_FailingStreamOllamaChat,
            cost_accounting_factory=lambda: None,
        )

        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "llama-local",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ) as response:
                body = response.read().decode()

        assert response.status_code == 200
        assert '"code": "connection_error"' in body
        assert "data: [DONE]" in body
    finally:
        _restore_registry(old_registry)


def test_ollama_chat_auth_rejects_before_provider_dispatch(monkeypatch):
    fake_ollama = _FakeOllamaChat()
    old_registry = _register_ollama_alias()
    old_key = bridge_main.BRIDGE_API_KEY
    try:
        monkeypatch.setattr(bridge_main, "BRIDGE_API_KEY", "secret-key")
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_UnexpectedVertexChat,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=lambda: fake_ollama,
            cost_accounting_factory=lambda: None,
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-local",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 401
        assert fake_ollama.calls == []
    finally:
        monkeypatch.setattr(bridge_main, "BRIDGE_API_KEY", old_key)
        _restore_registry(old_registry)


def test_ollama_chat_invalid_json_schema_missing_schema_returns_400(monkeypatch):
    import httpx
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    posted: list[dict] = []

    class NoNetworkAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def post(self, url, *, json=None, headers=None):
            posted.append({"url": url, "json": json, "headers": headers})
            raise AssertionError("invalid schema should be rejected before Ollama HTTP")

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", NoNetworkAsyncClient)
    old_registry = _register_ollama_alias()
    try:
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_UnexpectedVertexChat,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=lambda: OllamaChatClient(base_url="http://ollama.test"),
            cost_accounting_factory=lambda: None,
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-local",
                    "messages": [{"role": "user", "content": "hello"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "ExtractedEntities", "strict": True},
                    },
                },
            )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_request"
        assert posted == []
    finally:
        _restore_registry(old_registry)


def test_ollama_embeddings_and_rerank_are_rejected_before_provider_call():
    fake_ollama = _FakeOllamaChat()
    old_registry = _register_ollama_alias()
    try:
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_UnexpectedVertexChat,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=lambda: fake_ollama,
            cost_accounting_factory=lambda: None,
        )

        with TestClient(app) as client:
            embeddings = client.post(
                "/v1/embeddings",
                json={"model": "llama-local", "input": "hello"},
            )
            rerank = client.post(
                "/v1/rerank",
                json={"model": "llama-local", "query": "hello", "documents": ["a"]},
            )

        assert embeddings.status_code == 400
        assert embeddings.json()["error"]["code"] == "invalid_model"
        assert rerank.status_code == 400
        assert rerank.json()["error"]["code"] == "invalid_model"
        assert fake_ollama.calls == []
    finally:
        _restore_registry(old_registry)


def test_ollama_chat_cost_success_records_usage(monkeypatch, tmp_path):
    ledger_path = _enable_ollama_cost_tracking(monkeypatch, tmp_path)
    fake_ollama = _FakeOllamaChat()
    old_registry = _register_ollama_alias()
    try:
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_UnexpectedVertexChat,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=lambda: fake_ollama,
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-local",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 200
        event = CostLedger(ledger_path).fetch_events()[0]
        assert event["endpoint"] == "chat"
        assert event["model"] == "llama-local"
        assert event["status"] == "finalized"
        assert event["prompt_tokens"] == 3
        assert event["completion_tokens"] == 4
    finally:
        _restore_registry(old_registry)


def test_dynamic_ollama_chat_cost_uses_explicit_wildcard_pricing(monkeypatch, tmp_path):
    ledger_path = _enable_ollama_cost_tracking(
        monkeypatch,
        tmp_path,
        pricing_json="""
        {
          "source": "unit-test",
          "version": "2026-06-23",
          "currency": "USD",
          "models": {
            "ollama:*": {
              "chat": {
                "input_per_million": "0.10",
                "output_per_million": "0.20"
              }
            }
          }
        }
        """,
    )
    fake_ollama = _FakeOllamaChat()
    app = create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=_UnexpectedVertexChat,
        rerank_client_factory=_FakeProvider,
        ollama_chat_client_factory=lambda: fake_ollama,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama:qwen3.5:cloud",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert fake_ollama.calls[0]["model"] == "qwen3.5:cloud"
    event = CostLedger(ledger_path).fetch_events()[0]
    assert event["model"] == "ollama:qwen3.5:cloud"
    assert event["status"] == "finalized"
    assert event["pricing_source"] == "unit-test"


def test_ollama_chat_missing_pricing_fails_closed(monkeypatch, tmp_path):
    _enable_ollama_cost_tracking(
        monkeypatch,
        tmp_path,
        pricing_json='{"source":"unit-test","version":"2026-06-22","currency":"USD","models":{"other-model":{"chat":{"input_per_million":"1","output_per_million":"1"}}}}',
    )
    fake_ollama = _FakeOllamaChat()
    old_registry = _register_ollama_alias()
    try:
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_UnexpectedVertexChat,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=lambda: fake_ollama,
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-local",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "cost_config_error"
        assert fake_ollama.calls == []
    finally:
        _restore_registry(old_registry)


def test_create_app_uses_ollama_chat_client_by_default(monkeypatch):
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    created: list[object] = []

    class TrackingOllamaClient(OllamaChatClient):
        def __init__(self) -> None:
            created.append(self)

        async def close(self) -> None:
            pass

    monkeypatch.setattr("openai_compatible_bridge.main.OllamaChatClient", TrackingOllamaClient)
    app = create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=_FakeProvider,
        rerank_client_factory=_FakeProvider,
        cost_accounting_factory=lambda: None,
    )

    with TestClient(app):
        pass

    assert created


@pytest.mark.anyio
async def test_ollama_chat_client_generate_normalizes_response(monkeypatch):
    import httpx
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    posted: list[dict] = []

    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "message": {"role": "assistant", "content": "local answer"},
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 7,
            }

        @property
        def text(self):
            return "{}"

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def post(self, url, *, json=None, headers=None):
            posted.append({"url": url, "json": json, "headers": headers})
            return MockResponse()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    client = OllamaChatClient(base_url="http://ollama.test")
    result = await client.generate(
        model="llama3.1",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=12,
        temperature=0.2,
        top_p=0.9,
        stop=["done"],
    )

    assert posted[0]["url"] == "http://ollama.test/api/chat"
    assert posted[0]["json"]["model"] == "llama3.1"
    assert posted[0]["json"]["stream"] is False
    assert result == {
        "text": "local answer",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


@pytest.mark.anyio
async def test_ollama_chat_client_generate_strips_think_block(monkeypatch):
    import httpx
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "message": {"role": "assistant", "content": "<think>hidden reasoning</think>visible answer"},
                "done_reason": "stop",
            }

        @property
        def text(self):
            return "{}"

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def post(self, url, *, json=None, headers=None):
            return MockResponse()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    client = OllamaChatClient(base_url="http://ollama.test")
    result = await client.generate(
        model="minimax-m3:cloud",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result["text"] == "visible answer"


@pytest.mark.anyio
async def test_ollama_chat_client_generate_json_object_uses_json_format(monkeypatch):
    import httpx
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    posted: list[dict] = []

    class MockResponse:
        status_code = 200

        def json(self):
            return {"message": {"content": "{}"}, "done_reason": "stop"}

        @property
        def text(self):
            return "{}"

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def post(self, url, *, json=None, headers=None):
            posted.append({"url": url, "json": json, "headers": headers})
            return MockResponse()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    client = OllamaChatClient(base_url="http://ollama.test")
    await client.generate(
        model="llama3.1",
        messages=[{"role": "user", "content": "hello"}],
        response_format={"type": "json_object"},
    )

    assert posted[0]["json"]["format"] == "json"


@pytest.mark.anyio
async def test_ollama_chat_client_generate_json_schema_uses_schema_format(monkeypatch):
    import httpx
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    posted: list[dict] = []
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    class MockResponse:
        status_code = 200

        def json(self):
            return {"message": {"content": '{"name":"Ada"}'}, "done_reason": "stop"}

        @property
        def text(self):
            return "{}"

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def post(self, url, *, json=None, headers=None):
            posted.append({"url": url, "json": json, "headers": headers})
            return MockResponse()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    client = OllamaChatClient(base_url="http://ollama.test")
    await client.generate(
        model="llama3.1",
        messages=[{"role": "user", "content": "hello"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ExtractedEntities", "schema": schema, "strict": True},
        },
    )

    assert posted[0]["json"]["format"] == schema
    assert "name" not in posted[0]["json"]["format"]
    assert "strict" not in posted[0]["json"]["format"]


@pytest.mark.anyio
async def test_ollama_chat_client_stream_chat_normalizes_events(monkeypatch):
    import json
    import httpx
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    class MockStreamResponse:
        status_code = 200

        async def aiter_lines(self):
            yield json.dumps({"message": {"content": "hel"}, "done": False})
            yield json.dumps({"message": {"content": "lo"}, "done": False})
            yield json.dumps({
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 2,
                "eval_count": 3,
            })

        async def aread(self):
            return b""

    class MockStreamContext:
        async def __aenter__(self):
            return MockStreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def stream(self, method, url, *, json=None, headers=None):
            return MockStreamContext()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    client = OllamaChatClient(base_url="http://ollama.test")
    events = [
        event
        async for event in client.stream_chat(
            model="llama3.1",
            messages=[{"role": "user", "content": "hello"}],
        )
    ]

    assert events == [
        {"delta_text": "hel", "finish_reason": None, "usage": None},
        {"delta_text": "lo", "finish_reason": None, "usage": None},
        {
            "delta_text": "",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
    ]


@pytest.mark.anyio
async def test_ollama_chat_client_stream_chat_strips_split_think_block(monkeypatch):
    import json
    import httpx
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    class MockStreamResponse:
        status_code = 200

        async def aiter_lines(self):
            yield json.dumps({"message": {"content": "A <thi"}, "done": False})
            yield json.dumps({"message": {"content": "nk>hidden reasoning</thi"}, "done": False})
            yield json.dumps({"message": {"content": "nk> B"}, "done": False})
            yield json.dumps({
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 2,
                "eval_count": 3,
            })

        async def aread(self):
            return b""

    class MockStreamContext:
        async def __aenter__(self):
            return MockStreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def stream(self, method, url, *, json=None, headers=None):
            return MockStreamContext()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    client = OllamaChatClient(base_url="http://ollama.test")
    events = [
        event
        async for event in client.stream_chat(
            model="minimax-m3:cloud",
            messages=[{"role": "user", "content": "hello"}],
        )
    ]

    assert [event["delta_text"] for event in events] == ["A ", "", " B", ""]
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


@pytest.mark.anyio
async def test_ollama_chat_client_stream_chat_json_schema_uses_schema_format(monkeypatch):
    import json
    import httpx
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    streamed: list[dict] = []
    schema = {
        "type": "object",
        "properties": {"episode_indices": {"type": "array", "items": {"type": "integer"}}},
        "required": ["episode_indices"],
    }

    class MockStreamResponse:
        status_code = 200

        async def aiter_lines(self):
            yield json.dumps({
                "done": True,
                "done_reason": "stop",
                "message": {"content": '{"episode_indices":[0]}'},
            })

        async def aread(self):
            return b""

    class MockStreamContext:
        async def __aenter__(self):
            return MockStreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def stream(self, method, url, *, json=None, headers=None):
            streamed.append({"method": method, "url": url, "json": json, "headers": headers})
            return MockStreamContext()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    client = OllamaChatClient(base_url="http://ollama.test")
    events = [
        event
        async for event in client.stream_chat(
            model="llama3.1",
            messages=[{"role": "user", "content": "hello"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ExtractedEdges", "schema": schema, "strict": True},
            },
        )
    ]

    assert events[-1]["finish_reason"] == "stop"
    assert streamed[0]["json"]["format"] == schema
    assert "name" not in streamed[0]["json"]["format"]
    assert "strict" not in streamed[0]["json"]["format"]


@pytest.mark.anyio
async def test_ollama_chat_client_connection_error_maps_to_provider_error(monkeypatch):
    import httpx
    from openai_compatible_bridge.providers.ollama import OllamaChatClient
    from openai_compatible_bridge.providers.vertex import VertexAPIError

    class FailingAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def post(self, url, *, json=None, headers=None):
            raise httpx.ConnectError("offline")

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(httpx, "AsyncClient", FailingAsyncClient)

    client = OllamaChatClient(base_url="http://ollama.test")
    with pytest.raises(VertexAPIError) as excinfo:
        await client.generate(model="llama3.1", messages=[{"role": "user", "content": "hello"}])

    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "connection_error"
