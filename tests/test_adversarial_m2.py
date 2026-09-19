"""Adversarial stress-testing suite for Foundry OpenAI protocol (Milestone 2 Challenger).

Empirically tests edge cases and failure modes:
1. Malformed / empty / raw / whitespace arguments handling (non-stream & stream).
2. 3+ parallel tool calls with interleaved chunks and index-based client assembly.
3. Multibyte Unicode (Korean, Japanese, Chinese, Arabic, emojis) & special characters chunked streaming safety.
4. Missing / omitted finish_reason fallback behavior (non-stream & stream, tool_calls vs text).
5. Tool choice variants, parallel_tool_calls flags, and Foundry OpenAI specific parameter enforcement.
6. Upstream errors during non-streaming and streaming sessions.
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


class _DummyProvider:
    async def close(self) -> None:
        pass


def _register_foundry_openai_alias(
    model_alias: str = "foundry:gpt-6-astra",
    provider_model: str = "gpt-6-astra",
) -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY[model_alias] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": provider_model,
        "protocol": "openai_chat_completions",
    }
    return old_registry


def _restore_registry(old_registry: dict[str, dict]) -> None:
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old_registry)


# ---------------------------------------------------------------------------
# Test 1: Arguments handling (Empty string, empty object, malformed, whitespace)
# ---------------------------------------------------------------------------


def test_adversarial_arguments_handling_non_stream():
    """Verify arguments with empty string, empty json, raw non-json, and whitespace."""
    test_cases = [
        ("call_empty_str", "empty_fn", ""),
        ("call_empty_obj", "empty_obj_fn", "{}"),
        ("call_raw_str", "raw_fn", "not a valid json {{{"),
        ("call_ws_str", "ws_fn", "   \n\t  "),
        ("call_nested", "nested_fn", '{\n  "a": [1, null, true, false, "hello"]\n}'),
    ]

    for call_id, fn_name, arg_str in test_cases:
        async def mock_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {"name": fn_name, "arguments": arg_str},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
            )

        client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        old_reg = _register_foundry_openai_alias()
        try:
            # 1. Unit client test
            res = asyncio.run(
                client.generate(
                    model="gpt-6-astra",
                    messages=[{"role": "user", "content": "run"}],
                )
            )
            assert res["finish_reason"] == "tool_calls"
            assert res["tool_calls"] is not None
            assert len(res["tool_calls"]) == 1
            tc = res["tool_calls"][0]
            assert tc["id"] == call_id
            assert tc["function"]["name"] == fn_name
            assert tc["function"]["arguments"] == arg_str

            # 2. Public E2E Test
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
                        "model": "foundry:gpt-6-astra",
                        "messages": [{"role": "user", "content": "run"}],
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                choice = data["choices"][0]
                assert choice["finish_reason"] == "tool_calls"
                assert choice["message"]["role"] == "assistant"
                assert choice["message"]["content"] is None
                assert choice["message"]["tool_calls"][0]["function"]["arguments"] == arg_str
        finally:
            _restore_registry(old_reg)
            asyncio.run(client.close())


def test_adversarial_arguments_missing_in_upstream_message():
    """Verify behavior when upstream function object has arguments missing or None."""
    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "thinking",
                            "tool_calls": [
                                {
                                    "id": "call_missing_args",
                                    "type": "function",
                                    "function": {"name": "no_args_func"},  # arguments omitted
                                },
                                {
                                    "id": "call_none_args",
                                    "type": "function",
                                    "function": {"name": "none_args_func", "arguments": None},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_reg = _register_foundry_openai_alias()
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
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "run"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            tool_calls = data["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 2
            assert tool_calls[0]["function"]["name"] == "no_args_func"
            assert tool_calls[1]["function"]["name"] == "none_args_func"
    finally:
        _restore_registry(old_reg)
        asyncio.run(client.close())


# ---------------------------------------------------------------------------
# Test 2: 3+ Parallel Tool Calls with Interleaved Chunks and Assembly
# ---------------------------------------------------------------------------


def test_adversarial_parallel_tool_calls_interleaved_streaming():
    """Verify 4 parallel tool calls streamed with interleaved index chunks and assembled by client."""
    # We will simulate 4 functions:
    # 0: get_weather({"city": "Seoul", "unit": "celsius"})
    # 1: get_stock({"ticker": "AAPL", "currency": "USD"})
    # 2: translate({"text": "Hello world", "target_lang": "ko"})
    # 3: calculate({"expression": "42 * 100 + 7"})

    stream_payload = (
        # Chunk 1: Function declarations (indices 0 and 1)
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":['
        b'{"index":0,"id":"call_000","type":"function","function":{"name":"get_weather","arguments":""}},'
        b'{"index":1,"id":"call_111","type":"function","function":{"name":"get_stock","arguments":""}}'
        b']},"finish_reason":null}]}\n\n'
        # Chunk 2: Function declarations (indices 2 and 3)
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":['
        b'{"index":2,"id":"call_222","type":"function","function":{"name":"translate","arguments":""}},'
        b'{"index":3,"id":"call_333","type":"function","function":{"name":"calculate","arguments":""}}'
        b']},"finish_reason":null}]}\n\n'
        # Chunk 3: Arguments for index 0 and 2 (interleaved)
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":['
        b'{"index":0,"function":{"arguments":"{\\"city\\": "}},'
        b'{"index":2,"function":{"arguments":"{\\"text\\": \\"Hello world\\", "}}'
        b']},"finish_reason":null}]}\n\n'
        # Chunk 4: Arguments for index 1 and 3
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":['
        b'{"index":1,"function":{"arguments":"{\\"ticker\\": \\"AAPL\\", "}},'
        b'{"index":3,"function":{"arguments":"{\\"expression\\": "}}'
        b']},"finish_reason":null}]}\n\n'
        # Chunk 5: Arguments for index 0 (continuation)
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":['
        b'{"index":0,"function":{"arguments":"\\"Seoul\\", \\"unit\\": \\"celsius\\"}"}}'
        b']},"finish_reason":null}]}\n\n'
        # Chunk 6: Arguments for index 2 (continuation)
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":['
        b'{"index":2,"function":{"arguments":"\\"target_lang\\": \\"ko\\"}"}}'
        b']},"finish_reason":null}]}\n\n'
        # Chunk 7: Arguments for index 1 and 3 (finish arguments)
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":['
        b'{"index":1,"function":{"arguments":"\\"currency\\": \\"USD\\"}"}},'
        b'{"index":3,"function":{"arguments":"\\"42 * 100 + 7\\"}"}}'
        b']},"finish_reason":null}]}\n\n'
        # Chunk 8: finish_reason="tool_calls" and usage
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":50,"completion_tokens":80,"total_tokens":130}}\n\n'
        b"data: [DONE]\n\n"
    )

    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_reg = _register_foundry_openai_alias()
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Execute 4 tasks"}],
                    "parallel_tool_calls": True,
                },
            ) as response:
                assert response.status_code == 200
                lines = list(response.iter_lines())

        # OpenAI standard client tool-calls assembler simulation
        assembled_tools: dict[int, dict[str, Any]] = {}
        terminal_finish_reason: str | None = None

        for line in lines:
            line_str = line.strip()
            if not line_str.startswith("data:"):
                continue
            if line_str == "data: [DONE]":
                break
            chunk_data = json.loads(line_str[len("data:") :])
            choice = chunk_data["choices"][0]
            if choice.get("finish_reason"):
                terminal_finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            if "tool_calls" in delta:
                for tc_delta in delta["tool_calls"]:
                    idx = tc_delta["index"]
                    if idx not in assembled_tools:
                        assembled_tools[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.get("id"):
                        assembled_tools[idx]["id"] += tc_delta["id"]
                    fn_delta = tc_delta.get("function", {})
                    if fn_delta.get("name"):
                        assembled_tools[idx]["function"]["name"] += fn_delta["name"]
                    if fn_delta.get("arguments"):
                        assembled_tools[idx]["function"]["arguments"] += fn_delta["arguments"]

        assert terminal_finish_reason == "tool_calls"
        assert len(assembled_tools) == 4
        assert set(assembled_tools.keys()) == {0, 1, 2, 3}

        # Verify each assembled tool
        t0 = assembled_tools[0]
        assert t0["id"] == "call_000"
        assert t0["function"]["name"] == "get_weather"
        assert json.loads(t0["function"]["arguments"]) == {"city": "Seoul", "unit": "celsius"}

        t1 = assembled_tools[1]
        assert t1["id"] == "call_111"
        assert t1["function"]["name"] == "get_stock"
        assert json.loads(t1["function"]["arguments"]) == {"ticker": "AAPL", "currency": "USD"}

        t2 = assembled_tools[2]
        assert t2["id"] == "call_222"
        assert t2["function"]["name"] == "translate"
        assert json.loads(t2["function"]["arguments"]) == {"text": "Hello world", "target_lang": "ko"}

        t3 = assembled_tools[3]
        assert t3["id"] == "call_333"
        assert t3["function"]["name"] == "calculate"
        assert json.loads(t3["function"]["arguments"]) == {"expression": "42 * 100 + 7"}

    finally:
        _restore_registry(old_reg)
        asyncio.run(client.close())


# ---------------------------------------------------------------------------
# Test 3: Multibyte Unicode & Special Characters Streaming Safety
# ---------------------------------------------------------------------------


def test_adversarial_multibyte_unicode_and_special_chars_streaming():
    """Verify complex Multibyte Unicode (Korean, Japanese, Chinese, Arabic, emojis) and special characters streaming."""
    complex_data = {
        "korean": "안녕하세요! 멀티바이트 테스트입니다. 가나다라마바사 123 !@#$%^&*()_+",
        "japanese": "こんにちは！マルチバイトテストです。カタカナ、漢字、ひらがな",
        "chinese": "你好！这是一个多字节测试。汉字测试。",
        "arabic": "مرحبا بالعالم",
        "emojis": "🚀🔥🎉🤖⚡️👍✨🇰🇷🇯🇵🇺🇸",
        "escaped_json": 'He said, "It\'s a \\"quote\\" inside quotes!"\nAnd line breaks \r\n and tabs \t',
        "symbols": "<>&'\"/\\~`|{}[]",
    }
    raw_args_json = json.dumps(complex_data, ensure_ascii=False)

    # Slice raw_args_json into tiny 2-character pieces to maximally stress-test chunk boundaries
    chunk_size = 3
    arg_pieces = [raw_args_json[i : i + chunk_size] for i in range(0, len(raw_args_json), chunk_size)]

    # Build SSE payload
    events: list[bytes] = []
    # Initial header
    init_event = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_unicode_99",
                            "type": "function",
                            "function": {"name": "process_multilingual", "arguments": ""},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ]
    }
    events.append(f"data: {json.dumps(init_event, ensure_ascii=False)}\n\n".encode("utf-8"))

    # Argument chunks
    for piece in arg_pieces:
        ev = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": piece}}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        events.append(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8"))

    # Terminal chunk
    term_event = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
    }
    events.append(f"data: {json.dumps(term_event, ensure_ascii=False)}\n\n".encode("utf-8"))
    events.append(b"data: [DONE]\n\n")

    stream_content = b"".join(events)

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"}, content=stream_content)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_reg = _register_foundry_openai_alias()
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "multilingual request"}],
                },
            ) as response:
                assert response.status_code == 200
                lines = list(response.iter_lines())

        accumulated_args = ""
        finish_reason = None
        for line in lines:
            line_str = line.strip()
            if not line_str.startswith("data:") or line_str == "data: [DONE]":
                continue
            payload = json.loads(line_str[len("data:") :])
            choice = payload["choices"][0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    accumulated_args += tc["function"].get("arguments", "")

        assert finish_reason == "tool_calls"
        # Verify complete uncorrupted round-trip recovery
        parsed_result = json.loads(accumulated_args)
        assert parsed_result == complex_data
        assert parsed_result["korean"] == complex_data["korean"]
        assert parsed_result["japanese"] == complex_data["japanese"]
        assert parsed_result["chinese"] == complex_data["chinese"]
        assert parsed_result["arabic"] == complex_data["arabic"]
        assert parsed_result["emojis"] == complex_data["emojis"]
        assert parsed_result["escaped_json"] == complex_data["escaped_json"]

    finally:
        _restore_registry(old_reg)
        asyncio.run(client.close())


# ---------------------------------------------------------------------------
# Test 4: Missing / Omitted Finish Reason Fallback Behavior
# ---------------------------------------------------------------------------


def test_adversarial_missing_finish_reason_non_stream():
    """Verify non-streaming fallback when upstream omits finish_reason entirely."""
    # Case A: tool_calls present -> finish_reason defaults to "tool_calls"
    async def mock_handler_tools(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_fallback",
                                    "type": "function",
                                    "function": {"name": "action", "arguments": "{}"},
                                }
                            ],
                        },
                        # finish_reason key omitted!
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    # Case B: normal text present -> finish_reason defaults to "stop"
    async def mock_handler_text(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello without finish_reason",
                        },
                        # finish_reason key omitted!
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    client_tools = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client_tools.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler_tools))

    client_text = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client_text.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler_text))

    old_reg = _register_foundry_openai_alias()
    try:
        # Case A: tools fallback
        res_tools = asyncio.run(
            client_tools.generate(
                model="gpt-6-astra",
                messages=[{"role": "user", "content": "call tool"}],
            )
        )
        assert res_tools["finish_reason"] == "tool_calls"

        # Case B: text fallback
        res_text = asyncio.run(
            client_text.generate(
                model="gpt-6-astra",
                messages=[{"role": "user", "content": "say hello"}],
            )
        )
        assert res_text["finish_reason"] == "stop"
    finally:
        _restore_registry(old_reg)
        asyncio.run(client_tools.close())
        asyncio.run(client_text.close())


def test_adversarial_missing_finish_reason_stream_tool_calls():
    """Verify streaming fallback when upstream emits tool_calls chunks but NEVER sends finish_reason before [DONE]."""
    stream_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"search","arguments":"{\\"q\\": "}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"python\\"}"}}]},"finish_reason":null}]}\n\n'
        # Notice: No chunk with finish_reason! Immediately followed by usage, then [DONE]
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_reg = _register_foundry_openai_alias()
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "search"}],
                },
            ) as response:
                assert response.status_code == 200
                lines = list(response.iter_lines())

        parsed_events = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("data:") and line_str != "data: [DONE]":
                parsed_events.append(json.loads(line_str[len("data:") :]))

        # The terminal chunk must have finish_reason="tool_calls" thanks to the fallback!
        terminal_frs = [ev["choices"][0].get("finish_reason") for ev in parsed_events if ev.get("choices") and ev["choices"][0].get("finish_reason") is not None]
        assert "tool_calls" in terminal_frs
        assert terminal_frs[-1] == "tool_calls"
    finally:
        _restore_registry(old_reg)
        asyncio.run(client.close())


def test_adversarial_missing_finish_reason_stream_text():
    """Verify streaming fallback when upstream emits text chunks but NEVER sends finish_reason before [DONE]."""
    stream_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello world!"},"finish_reason":null}]}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_reg = _register_foundry_openai_alias()
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "say hello"}],
                },
            ) as response:
                assert response.status_code == 200
                lines = list(response.iter_lines())

        parsed_events = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("data:") and line_str != "data: [DONE]":
                parsed_events.append(json.loads(line_str[len("data:") :]))

        # The terminal chunk must fallback to finish_reason="stop"
        terminal_frs = [ev["choices"][0].get("finish_reason") for ev in parsed_events if ev.get("choices") and ev["choices"][0].get("finish_reason") is not None]
        assert "stop" in terminal_frs
        assert terminal_frs[-1] == "stop"
    finally:
        _restore_registry(old_reg)
        asyncio.run(client.close())


# ---------------------------------------------------------------------------
# Test 5: Foundry OpenAI Specific Parameter Enforcement
# ---------------------------------------------------------------------------


def test_adversarial_foundry_openai_parameter_filtering_and_passthrough():
    """Verify Foundry OpenAI backend specific parameter mappings:
    - max_tokens -> max_completion_tokens
    - temperature == 1.0 forwarded, non-1.0 omitted
    - tool_choice ('none', 'required', named) forwarded
    - parallel_tool_calls (True, False) forwarded
    """
    captured_request: dict[str, Any] = {}

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    # Scenario 1: temperature 0.7 (non-1.0 should be omitted)
    asyncio.run(
        client.generate(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=500,
            temperature=0.7,
            tool_choice="none",
            parallel_tool_calls=False,
        )
    )
    assert "max_tokens" not in captured_request
    assert captured_request["max_completion_tokens"] == 500
    assert "temperature" not in captured_request  # non-1.0 filtered out!
    assert captured_request["tool_choice"] == "none"
    assert captured_request["parallel_tool_calls"] is False

    # Scenario 2: temperature 1.0 (should be preserved) and tool_choice="required"
    asyncio.run(
        client.generate(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "test"}],
            temperature=1.0,
            tool_choice="required",
            parallel_tool_calls=True,
        )
    )
    assert captured_request["temperature"] == 1.0
    assert captured_request["tool_choice"] == "required"
    assert captured_request["parallel_tool_calls"] is True

    asyncio.run(client.close())


# ---------------------------------------------------------------------------
# Test 6: Upstream Error Responses During Streaming and Non-Streaming
# ---------------------------------------------------------------------------


def test_adversarial_upstream_error_handling():
    """Verify HTTP 400 and stream error payload handling."""
    # 1. Non-streaming upstream 400 error
    async def mock_400_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid tool specification: missing properties",
                    "type": "invalid_request_error",
                    "code": "bad_tool",
                }
            },
        )

    client_400 = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client_400.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_400_handler))

    old_reg = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client_400,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "invalid tool test"}],
                },
            )
            assert resp.status_code == 400
            err = resp.json().get("error", {})
            assert "Invalid tool specification" in err.get("message", "")
            assert err.get("code") == "bad_tool"
    finally:
        _restore_registry(old_reg)
        asyncio.run(client_400.close())

    # 2. Streaming mid-stream error
    stream_error_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Starting..."},"finish_reason":null}]}\n\n'
        b'data: {"error":{"message":"Foundry upstream model rate limited or disconnected","code":"rate_limit_exceeded"}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_stream_err_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_error_payload)

    client_stream_err = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client_stream_err.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_stream_err_handler))

    old_reg2 = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client_stream_err,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "stream error test"}],
                },
            ) as stream_resp:
                assert stream_resp.status_code == 200
                stream_lines = list(stream_resp.iter_lines())

        parsed_stream_items = []
        for line in stream_lines:
            line_str = line.strip()
            if line_str.startswith("data:") and line_str != "data: [DONE]":
                parsed_stream_items.append(json.loads(line_str[len("data:") :]))

        # Should contain error object emitted before termination
        err_events = [it for it in parsed_stream_items if "error" in it]
        assert len(err_events) == 1
        assert "rate limited or disconnected" in err_events[0]["error"]["message"]
        assert err_events[0]["error"]["code"] == "rate_limit_exceeded"
        # And stream terminates with [DONE]
        assert [l.strip() for l in stream_lines if l.strip()][-1] == "data: [DONE]"
    finally:
        _restore_registry(old_reg2)
        asyncio.run(client_stream_err.close())


# ---------------------------------------------------------------------------
# Test 7: Multi-Turn History with 3+ Parallel Tool Calls & Tool Results
# ---------------------------------------------------------------------------


def test_adversarial_parallel_3plus_tool_calls_multi_turn_cycle():
    """Verify 3 parallel tool calls followed by 3 role: 'tool' responses in multi-turn history."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        if len(captured_requests) == 1:
            # Turn 1: model returns 3 parallel tool calls
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_p1",
                                        "type": "function",
                                        "function": {"name": "func_a", "arguments": '{"x": 1}'},
                                    },
                                    {
                                        "id": "call_p2",
                                        "type": "function",
                                        "function": {"name": "func_b", "arguments": '{"y": 2}'},
                                    },
                                    {
                                        "id": "call_p3",
                                        "type": "function",
                                        "function": {"name": "func_c", "arguments": '{"z": 3}'},
                                    },
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
                },
            )

        # Turn 2: final synthesis text
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "All 3 tasks succeeded: a=10, b=20, c=30.",
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_reg = _register_foundry_openai_alias()
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
            # Turn 1
            r1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "run 3 parallel functions"}],
                },
            )
            assert r1.status_code == 200
            d1 = r1.json()
            tool_calls = d1["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 3

            # Turn 2: Provide 3 tool responses
            messages_turn2 = [
                {"role": "user", "content": "run 3 parallel functions"},
                {"role": "assistant", "content": None, "tool_calls": tool_calls},
                {"role": "tool", "tool_call_id": "call_p1", "name": "func_a", "content": '{"result": 10}'},
                {"role": "tool", "tool_call_id": "call_p2", "name": "func_b", "content": '{"result": 20}'},
                {"role": "tool", "tool_call_id": "call_p3", "name": "func_c", "content": '{"result": 30}'},
            ]

            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": messages_turn2,
                },
            )
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["choices"][0]["finish_reason"] == "stop"
            assert "All 3 tasks succeeded" in d2["choices"][0]["message"]["content"]

            # Verify that upstream received Turn 2 messages completely intact
            upstream_msgs = captured_requests[1]["messages"]
            assert len(upstream_msgs) == 5
            # Assistant with 3 tool calls
            assert upstream_msgs[1]["role"] == "assistant"
            assert len(upstream_msgs[1]["tool_calls"]) == 3
            assert [tc["id"] for tc in upstream_msgs[1]["tool_calls"]] == ["call_p1", "call_p2", "call_p3"]
            # 3 tool messages
            assert upstream_msgs[2]["role"] == "tool"
            assert upstream_msgs[2]["tool_call_id"] == "call_p1"
            assert upstream_msgs[3]["role"] == "tool"
            assert upstream_msgs[3]["tool_call_id"] == "call_p2"
            assert upstream_msgs[4]["role"] == "tool"
            assert upstream_msgs[4]["tool_call_id"] == "call_p3"
    finally:
        _restore_registry(old_reg)
        asyncio.run(client.close())


# ---------------------------------------------------------------------------
# Test 8: Large Payload (100KB+) Streaming Assembly
# ---------------------------------------------------------------------------


def test_adversarial_large_arguments_payload_streaming():
    """Stress test with 100KB+ JSON payload split across many SSE chunks."""
    # Generate ~100KB of repetitive structured data
    items = [{"id": i, "name": f"item_{i}", "desc": "x" * 200} for i in range(500)]
    large_dict = {"total": len(items), "items": items}
    large_args_str = json.dumps(large_dict)
    assert len(large_args_str) > 100_000

    # Chunk into 4KB pieces
    chunk_sz = 4096
    pieces = [large_args_str[i : i + chunk_sz] for i in range(0, len(large_args_str), chunk_sz)]

    events: list[bytes] = []
    # Start chunk
    ev_start = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_big_data",
                            "type": "function",
                            "function": {"name": "save_big_data", "arguments": ""},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ]
    }
    events.append(f"data: {json.dumps(ev_start)}\n\n".encode("utf-8"))

    # Argument chunks
    for p in pieces:
        ev = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": p}}]
                    },
                    "finish_reason": None,
                }
            ]
        }
        events.append(f"data: {json.dumps(ev)}\n\n".encode("utf-8"))

    # End chunk
    ev_end = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 25000, "total_tokens": 25500},
    }
    events.append(f"data: {json.dumps(ev_end)}\n\n".encode("utf-8"))
    events.append(b"data: [DONE]\n\n")

    stream_content = b"".join(events)

    async def mock_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_content)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_reg = _register_foundry_openai_alias()
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "big data"}],
                },
            ) as response:
                assert response.status_code == 200
                lines = list(response.iter_lines())

        accumulated = ""
        for line in lines:
            line_str = line.strip()
            if not line_str.startswith("data:") or line_str == "data: [DONE]":
                continue
            p = json.loads(line_str[len("data:") :])
            delta = p["choices"][0].get("delta", {})
            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    accumulated += tc["function"].get("arguments", "")

        assert len(accumulated) == len(large_args_str)
        recovered = json.loads(accumulated)
        assert recovered["total"] == 500
        assert len(recovered["items"]) == 500
        assert recovered["items"][499]["name"] == "item_499"
    finally:
        _restore_registry(old_reg)
        asyncio.run(client.close())


# ---------------------------------------------------------------------------
# Test 9: Robustness against Malformed / Empty SSE Chunks
# ---------------------------------------------------------------------------


def test_adversarial_malformed_and_empty_sse_chunks_in_stream():
    """Verify stream client handles malformed JSON lines, empty delta, empty choices list without crashing."""
    stream_payload = (
        b": keepalive comment\n\n"
        b"data: invalid-json-not-an-object\n\n"
        b'data: {"choices": []}\n\n'
        b'data: {"choices": [{"index": 0, "delta": {"tool_calls": []}}]}\n\n'  # empty tool_calls list
        b'data: {"choices": [{"index": 0, "delta": {"tool_calls": "not-a-list"}}]}\n\n'  # malformed tool_calls type
        b'data: {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Recovered"}}]}\n\n'
        b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_reg = _register_foundry_openai_alias()
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "test malformed"}],
                },
            ) as response:
                assert response.status_code == 200
                lines = list(response.iter_lines())

        parsed_events = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("data:") and line_str != "data: [DONE]":
                parsed_events.append(json.loads(line_str[len("data:") :]))

        # We must observe that "Recovered" was yielded and finish_reason is "stop"
        contents = [ev["choices"][0]["delta"].get("content") for ev in parsed_events if "content" in ev["choices"][0].get("delta", {})]
        assert "Recovered" in contents
        assert parsed_events[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        _restore_registry(old_reg)
        asyncio.run(client.close())

