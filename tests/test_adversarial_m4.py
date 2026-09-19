"""Adversarial stress-testing suite for Foundry xAI Responses protocol (Milestone 4 Challenger).

Empirically tests edge cases and failure modes:
1. Multi-function calls assembly (3, 4, 5 parallel function_call items) in non-streaming and streaming.
2. `function_call_output` serialization safety (Korean, emojis, deeply nested JSON, quotes, slashes, newlines).
3. xAI SSE stream interleaving of text and tool calls with accurate 0-based tool index mapping and terminal finish_reason="tool_calls".
4. Exhaustive `tool_choice` variants ("none", "required", "auto", OpenAI named dict, Anthropic named dict).
5. Error handling and unwrapping for xAI upstream failures (`response.failed`, 4xx/5xx HTTP errors, malformed responses).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError

FOUNDRY_TEST_BASE_URL = "https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions"

TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

TOOL_CALC = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform calculation",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    },
}

TOOL_TRANSLATE = {
    "type": "function",
    "function": {
        "name": "translate",
        "description": "Translate text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
    },
}

TOOL_DATABASE = {
    "type": "function",
    "function": {
        "name": "db_lookup",
        "description": "Lookup database records",
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string"}, "record_id": {"type": "integer"}},
            "required": ["table", "record_id"],
        },
    },
}

TOOL_NOTIFY = {
    "type": "function",
    "function": {
        "name": "send_notification",
        "description": "Send alert notification",
        "parameters": {
            "type": "object",
            "properties": {"channel": {"type": "string"}, "message": {"type": "string"}},
            "required": ["channel", "message"],
        },
    },
}


class _DummyProvider:
    async def close(self) -> None:
        pass


def _register_foundry_xai_alias(
    model_alias: str = "foundry:grok-4.6",
    provider_model: str = "grok-4.6",
) -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY[model_alias] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": provider_model,
        "protocol": "xai_responses",
    }
    return old_registry


def _restore_registry(old_registry: dict[str, dict]) -> None:
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old_registry)


# ==============================================================================
# 1. Multi-Function Calls Assembly (3, 4, 5 parallel calls)
# ==============================================================================


def test_adversarial_xai_multi_function_calls_non_stream_3_4_5():
    """Verify non-streaming handling when xAI returns 3, 4, or 5 parallel function_call items."""
    for n in (3, 4, 5):
        captured_requests: list[dict[str, Any]] = []

        mock_output = [
            {
                "type": "function_call",
                "id": f"call_id_{i}",
                "call_id": f"call_id_{i}",
                "name": f"tool_func_{i}",
                "arguments": json.dumps({"arg_key": f"val_{i}", "index": i}),
            }
            for i in range(n)
        ]

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/xai/v1/responses")
            captured_requests.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "id": f"resp_{n}_calls",
                    "status": "completed",
                    "output": mock_output,
                    "usage": {"input_tokens": 10 * n, "output_tokens": 20 * n, "total_tokens": 30 * n},
                },
            )

        client = FoundryChatClient(
            base_url=FOUNDRY_TEST_BASE_URL,
            token="test-token",
        )
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        async def run_client():
            return await client.generate(
                model="grok-4.6",
                messages=[{"role": "user", "content": f"Execute {n} parallel operations"}],
                resolved_config={"protocol": "xai_responses"},
            )

        res = asyncio.run(run_client())
        asyncio.run(client.close())

        assert res["text"] is None
        assert res["finish_reason"] == "tool_calls"
        assert res["tool_calls"] is not None
        assert len(res["tool_calls"]) == n
        for i, tc in enumerate(res["tool_calls"]):
            assert tc["id"] == f"call_id_{i}"
            assert tc["type"] == "function"
            assert tc["function"]["name"] == f"tool_func_{i}"
            parsed_args = json.loads(tc["function"]["arguments"])
            assert parsed_args == {"arg_key": f"val_{i}", "index": i}
        assert res["usage"]["total_tokens"] == 30 * n


def test_adversarial_xai_multi_function_calls_stream_interleaved_indices():
    """Verify streaming when 4 parallel function_calls are initialized and arguments arrive interleaved across calls."""
    # 4 functions: search (idx 0), calc (idx 1), translate (idx 2), notify (idx 3)
    stream_lines = [
        # Output items added
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_0","call_id":"call_0","name":"web_search","arguments":""}}\n\n',
        b'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"call_1","call_id":"call_1","name":"calculator","arguments":""}}\n\n',
        b'data: {"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"call_2","call_id":"call_2","name":"translate","arguments":""}}\n\n',
        b'data: {"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","id":"call_3","call_id":"call_3","name":"send_notification","arguments":""}}\n\n',
        # Interleaved argument chunks across different calls
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_2","delta":"{\\"text\\": "}\n\n',
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_0","delta":"{\\"query\\": "}\n\n',
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_1","delta":"{\\"expr\\": "}\n\n',
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_3","delta":"{\\"channel\\": \\"slack\\", \\"message\\": \\"alert!\\"}"}\n\n',
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_0","delta":"\\"quantum computing\\"}"}\n\n',
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_1","delta":"\\"42 * 100\\"}"}\n\n',
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_2","delta":"\\"hello\\", \\"target_lang\\": \\"ko\\"}"}\n\n',
        # Dones and completion
        b'data: {"type":"response.output_item.done","output_index":0}\n\n',
        b'data: {"type":"response.output_item.done","output_index":1}\n\n',
        b'data: {"type":"response.output_item.done","output_index":2}\n\n',
        b'data: {"type":"response.output_item.done","output_index":3}\n\n',
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":50,"output_tokens":80,"total_tokens":130}}}\n\n',
        b"data: [DONE]\n\n",
    ]
    stream_payload = b"".join(stream_lines)

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_xai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "Run 4 tools concurrently"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            lines = resp.text.split("\n")
            json_chunks = []
            for line in lines:
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    s = line[len("data:") :].strip()
                    if s:
                        json_chunks.append(json.loads(s))

            # Assemble arguments per tool index
            tool_args: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
            tool_names: dict[int, str] = {}
            tool_ids: dict[int, str] = {}

            for c in json_chunks:
                choice = c["choices"][0]
                delta = choice.get("delta", {})
                tc_list = delta.get("tool_calls")
                if tc_list:
                    for tc in tc_list:
                        idx = tc["index"]
                        if tc.get("id"):
                            tool_ids[idx] = tc["id"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_names[idx] = fn["name"]
                        if fn.get("arguments"):
                            tool_args[idx].append(fn["arguments"])

            # Verify names and IDs for all 4 tools
            assert tool_names == {0: "web_search", 1: "calculator", 2: "translate", 3: "send_notification"}
            assert tool_ids == {0: "call_0", 1: "call_1", 2: "call_2", 3: "call_3"}

            # Verify reassembled JSON arguments per index
            assert json.loads("".join(tool_args[0])) == {"query": "quantum computing"}
            assert json.loads("".join(tool_args[1])) == {"expr": "42 * 100"}
            assert json.loads("".join(tool_args[2])) == {"text": "hello", "target_lang": "ko"}
            assert json.loads("".join(tool_args[3])) == {"channel": "slack", "message": "alert!"}

            # Terminal finish_reason must be tool_calls
            assert json_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
            assert [l for l in lines if l.strip()][-1] == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# 2. `function_call_output` Serialization Safety (Korean, Emojis, Nested JSON)
# ==============================================================================


def test_adversarial_xai_function_call_output_unicode_and_nested_json():
    """Verify input items mapping when tool outputs contain Korean, emojis, complex nested JSON, and special symbols."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_unicode_ok",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "모든 도구 결과가 완벽하게 처리되었습니다! 🎉"}],
                    }
                ],
                "usage": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
            },
        )

    client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    complex_nested_output = {
        "status": "success",
        "location": "대한민국 서울특별시 종로구 청와대로 1",
        "weather": {
            "temperature_celsius": 21.5,
            "condition": "맑음 ☀️",
            "air_quality": {"pm25": 12, "grade": "좋음 ✨"},
        },
        "tags": ["서울", "날씨", "가을 🍂", "<script>alert('safe')</script>"],
        "quotes": 'Double "quotes" and single \'quotes\' and \\ backslashes \n and newlines \t tabs',
        "metadata": {
            "source": "KMA_API",
            "nested_null": None,
            "nested_bool": True,
            "nested_list": [[1, 2], [3, {"deep": "심층 값 🚀"}]],
        },
    }

    messages = [
        {"role": "user", "content": "서울 날씨와 환율 조회해줘"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather_kr",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "서울"}'},
                },
                {
                    "id": "call_exchange_kr",
                    "type": "function",
                    "function": {"name": "get_exchange", "arguments": '{"pair": "USD/KRW"}'},
                },
            ],
        },
        # Tool 1: Raw dict with Korean, emoji, deeply nested data
        {
            "role": "tool",
            "tool_call_id": "call_weather_kr",
            "content": complex_nested_output,
        },
        # Tool 2: Raw string with Korean, emoji, and special characters
        {
            "role": "tool",
            "tool_call_id": "call_exchange_kr",
            "content": "현재 원/달러 환율: 1,385.50원 (전일 대비 +3.20원 상승 📈, 변동성 주의!)",
        },
    ]

    async def run():
        return await client.generate(
            model="grok-4.6",
            messages=messages,
            resolved_config={"protocol": "xai_responses"},
        )

    res = asyncio.run(run())
    asyncio.run(client.close())

    assert res["finish_reason"] == "stop"
    assert "모든 도구 결과가 완벽하게 처리되었습니다! 🎉" in (res["text"] or "")

    # Inspect captured upstream body
    req = captured_requests[0]
    input_items = req["input"]

    # Verify input items structure
    assert len(input_items) == 5
    assert input_items[0]["role"] == "user"
    assert input_items[1]["type"] == "function_call"
    assert input_items[1]["call_id"] == "call_weather_kr"
    assert input_items[2]["type"] == "function_call"
    assert input_items[2]["call_id"] == "call_exchange_kr"

    # Item 3: function_call_output 1
    fc_out_1 = input_items[3]
    assert fc_out_1["type"] == "function_call_output"
    assert fc_out_1["call_id"] == "call_weather_kr"
    parsed_out_1 = json.loads(fc_out_1["output"])
    assert parsed_out_1["location"] == "대한민국 서울특별시 종로구 청와대로 1"
    assert parsed_out_1["weather"]["condition"] == "맑음 ☀️"
    assert parsed_out_1["metadata"]["nested_list"][1][1]["deep"] == "심층 값 🚀"
    assert "<script>alert('safe')</script>" in parsed_out_1["tags"]

    # Item 4: function_call_output 2
    fc_out_2 = input_items[4]
    assert fc_out_2["type"] == "function_call_output"
    assert fc_out_2["call_id"] == "call_exchange_kr"
    assert "1,385.50원" in fc_out_2["output"]
    assert "📈" in fc_out_2["output"]


# ==============================================================================
# 3. Interleaving Text and Tool Call Chunks in SSE Stream
# ==============================================================================


def test_adversarial_xai_stream_interleaving_text_and_tools():
    """Verify SSE stream correctly splits interleaved text and tool calls, yielding terminal finish_reason='tool_calls'."""
    stream_lines = [
        # 1. Grok begins by streaming thought / preliminary text
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant","content":[]}}\n\n',
        b'data: {"type":"response.output_text.delta","output_index":0,"delta":"I am searching for "}\n\n',
        b'data: {"type":"response.output_text.delta","output_index":0,"delta":"the latest market rates..."}\n\n',
        b'data: {"type":"response.output_item.done","output_index":0}\n\n',
        # 2. Tool 1 is called
        b'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"call_search_01","call_id":"call_search_01","name":"web_search","arguments":""}}\n\n',
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_search_01","delta":"{\\"query\\": "}\n\n',
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_search_01","delta":"\\"Fed rate decision\\"}"}\n\n',
        b'data: {"type":"response.output_item.done","output_index":1}\n\n',
        # 3. Interleaved additional text between tool calls
        b'data: {"type":"response.output_item.added","output_index":2,"item":{"type":"message","role":"assistant","content":[]}}\n\n',
        b'data: {"type":"response.output_text.delta","output_index":2,"delta":" and calculating bond yields:"}\n\n',
        b'data: {"type":"response.output_item.done","output_index":2}\n\n',
        # 4. Tool 2 is called
        b'data: {"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","id":"call_calc_01","call_id":"call_calc_01","name":"calculator","arguments":""}}\n\n',
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_calc_01","delta":"{\\"expr\\": \\"5.25 - 0.25\\"}"}\n\n',
        b'data: {"type":"response.output_item.done","output_index":3}\n\n',
        # 5. Completed
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":30,"output_tokens":45,"total_tokens":75}}}\n\n',
        b"data: [DONE]\n\n",
    ]
    stream_payload = b"".join(stream_lines)

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_xai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "Analyze Fed rates"}],
                    "tools": [TOOL_SEARCH, TOOL_CALC],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            lines = resp.text.split("\n")
            json_chunks = []
            for line in lines:
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    s = line[len("data:") :].strip()
                    if s:
                        json_chunks.append(json.loads(s))

            # Collect text parts and tool call parts
            text_deltas = []
            tool_calls_deltas = []

            for c in json_chunks:
                delta = c["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    text_deltas.append(delta["content"])
                if "tool_calls" in delta and delta["tool_calls"]:
                    tool_calls_deltas.extend(delta["tool_calls"])

            # Verify text is preserved across chunks
            full_text = "".join(text_deltas)
            assert "I am searching for the latest market rates..." in full_text
            assert " and calculating bond yields:" in full_text

            # Verify tool calls are separated and mapped with 0-based indices
            tool_0_args = "".join(
                tc["function"]["arguments"]
                for tc in tool_calls_deltas
                if tc["index"] == 0 and "arguments" in tc.get("function", {})
            )
            tool_1_args = "".join(
                tc["function"]["arguments"]
                for tc in tool_calls_deltas
                if tc["index"] == 1 and "arguments" in tc.get("function", {})
            )

            assert json.loads(tool_0_args) == {"query": "Fed rate decision"}
            assert json.loads(tool_1_args) == {"expr": "5.25 - 0.25"}

            # Terminal finish_reason must be strictly tool_calls
            assert json_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
            assert [l for l in lines if l.strip()][-1] == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# 4. Exhaustive `tool_choice` Variants
# ==============================================================================


def test_adversarial_xai_tool_choice_variants_exhaustive():
    """Verify tool_choice handling across all specifications: 'none', 'required', 'auto', OpenAI dict, Anthropic dict."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp_tc_var",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Understood."}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        )

    client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    test_cases = [
        # (tool_choice input, expected upstream tool_choice)
        ("none", "none"),
        ("auto", "auto"),
        ("required", "required"),
        (
            {"type": "function", "function": {"name": "calculator"}},
            {"type": "function", "name": "calculator"},
        ),
        (
            {"type": "tool", "name": "web_search"},
            {"type": "function", "name": "web_search"},
        ),
        (
            {"name": "translate"},
            {"type": "function", "name": "translate"},
        ),
    ]

    async def run_variants():
        for tc_input, expected_tc in test_cases:
            captured_requests.clear()
            await client.generate(
                model="grok-4.6",
                messages=[{"role": "user", "content": "Test tool_choice"}],
                tools=[TOOL_SEARCH, TOOL_CALC, TOOL_TRANSLATE],
                tool_choice=tc_input,
                resolved_config={"protocol": "xai_responses"},
            )
            req = captured_requests[0]
            assert req["tool_choice"] == expected_tc
            assert len(req["tools"]) == 3

    asyncio.run(run_variants())
    asyncio.run(client.close())


# ==============================================================================
# 5. Upstream Errors, Malformed Payloads, and Robust Recovery
# ==============================================================================


def test_adversarial_xai_upstream_errors_and_malformed_responses():
    """Verify bridge raises standard VertexAPIError with appropriate HTTP status on upstream failure."""
    # Sub-case 1: xAI returns 400 Bad Request
    async def handler_400(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Invalid parameter in xAI input", "code": "invalid_parameter"}},
        )

    client1 = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="tok")
    client1.http = httpx.AsyncClient(transport=httpx.MockTransport(handler_400))
    with pytest.raises(VertexAPIError) as exc_info:
        asyncio.run(
            client1.generate(
                model="grok-4.6",
                messages=[{"role": "user", "content": "hi"}],
                resolved_config={"protocol": "xai_responses"},
            )
        )
    assert exc_info.value.status_code == 400
    assert "Invalid parameter in xAI input" in exc_info.value.message
    asyncio.run(client1.close())

    # Sub-case 2: xAI returns 200 with missing output array
    async def handler_missing_output(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "bad_resp", "status": "completed"})

    client2 = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="tok")
    client2.http = httpx.AsyncClient(transport=httpx.MockTransport(handler_missing_output))
    with pytest.raises(VertexAPIError) as exc_info:
        asyncio.run(
            client2.generate(
                model="grok-4.6",
                messages=[{"role": "user", "content": "hi"}],
                resolved_config={"protocol": "xai_responses"},
            )
        )
    assert exc_info.value.status_code == 502
    assert "missing output[]" in exc_info.value.message
    asyncio.run(client2.close())

    # Sub-case 3: xAI streaming emits response.failed
    stream_err = (
        b'data: {"type":"response.failed","error":{"message":"Grok context overflow","code":"context_length_exceeded"}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def handler_stream_failed(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_err)

    client3 = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="tok")
    client3.http = httpx.AsyncClient(transport=httpx.MockTransport(handler_stream_failed))
    with pytest.raises(VertexAPIError) as exc_info:
        async def run_stream():
            async for _ in client3.stream_chat(
                model="grok-4.6",
                messages=[{"role": "user", "content": "hi"}],
                resolved_config={"protocol": "xai_responses"},
            ):
                pass

        asyncio.run(run_stream())
    assert exc_info.value.status_code == 502
    assert "Grok context overflow" in exc_info.value.message
    asyncio.run(client3.close())
