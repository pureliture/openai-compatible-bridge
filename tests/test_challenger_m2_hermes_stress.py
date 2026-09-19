"""Adversarial and stress test harness for Foundry OpenAI Protocol (Milestone 2).

Conducted by Challenger 2 (teamwork_preview_challenger_m2_2).
Validates:
1. Hermes Agent continuous multi-cycle tool calling:
   Turn 1 (User) -> Turn 2 (Assistant tool_call 1) -> Turn 3 (Client tool_result 1) ->
   Turn 4 (Assistant tool_call 2) -> Turn 5 (Client tool_result 2) -> Turn 6 (Assistant text)
   Both non-streaming and streaming.
2. tool_choice upstream payload mapping:
   "none", "required", "auto", and specific function object.
3. Parallel multi-tool calling multi-turn cycle and interleaved streaming chunks.
4. Upstream error propagation and edge-case tool content formats.
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


TOOL_STOCK: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_stock_quote",
        "description": "Fetch current stock quote",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
}

TOOL_CALC_PE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculate_pe",
        "description": "Calculate price-to-earnings ratio",
        "parameters": {
            "type": "object",
            "properties": {
                "price": {"type": "number"},
                "eps": {"type": "number"},
            },
            "required": ["price", "eps"],
        },
    },
}

ALL_TOOLS = [TOOL_STOCK, TOOL_CALC_PE]


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


def test_hermes_continuous_multi_cycle_tool_calling_non_stream():
    """Stress test: 2 sequential tool calling cycles before final text synthesis (Non-streaming).

    User Query -> Tool Call 1 (get_stock_quote) -> Tool Result 1 ->
    Tool Call 2 (calculate_pe) -> Tool Result 2 -> Final Synthesis Text.
    """
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        call_idx = len(captured_requests)
        if call_idx == 1:
            # First turn response: Tool Call 1
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
                                        "id": "call_stock_001",
                                        "type": "function",
                                        "function": {
                                            "name": "get_stock_quote",
                                            "arguments": '{"ticker": "AAPL"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
                },
            )
        elif call_idx == 2:
            # Second turn response: Tool Call 2 (model uses stock price to call calculate_pe)
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
                                        "id": "call_pe_002",
                                        "type": "function",
                                        "function": {
                                            "name": "calculate_pe",
                                            "arguments": '{"price": 225.0, "eps": 6.5}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 55, "completion_tokens": 20, "total_tokens": 75},
                },
            )
        else:
            # Third turn response: Final synthesis text
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "AAPL's current stock price is $225.00, resulting in a P/E ratio of 34.61 with an EPS of $6.50.",
                                "tool_calls": None,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 90, "completion_tokens": 30, "total_tokens": 120},
                },
            )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
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
            # Step 1: Client issues initial prompt
            conversation = [
                {"role": "user", "content": "What is AAPL's P/E ratio if EPS is 6.50?"}
            ]
            resp1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": conversation,
                    "tools": ALL_TOOLS,
                    "tool_choice": "auto",
                },
            )
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["choices"][0]["finish_reason"] == "tool_calls"
            msg1 = data1["choices"][0]["message"]
            assert msg1["role"] == "assistant"
            assert msg1["content"] is None
            assert len(msg1["tool_calls"]) == 1
            tool_call_1 = msg1["tool_calls"][0]
            assert tool_call_1["id"] == "call_stock_001"
            assert tool_call_1["function"]["name"] == "get_stock_quote"

            # Step 2: Hermes client executes Tool 1 and appends to conversation
            conversation.append(msg1)
            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call_1["id"],
                "name": tool_call_1["function"]["name"],
                "content": json.dumps({"ticker": "AAPL", "price": 225.0, "currency": "USD"}),
            })

            # Send Turn 2 request
            resp2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": conversation,
                    "tools": ALL_TOOLS,
                    "tool_choice": "auto",
                },
            )
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["choices"][0]["finish_reason"] == "tool_calls"
            msg2 = data2["choices"][0]["message"]
            assert msg2["role"] == "assistant"
            assert msg2["content"] is None
            assert len(msg2["tool_calls"]) == 1
            tool_call_2 = msg2["tool_calls"][0]
            assert tool_call_2["id"] == "call_pe_002"
            assert tool_call_2["function"]["name"] == "calculate_pe"

            # Verify that upstream request for Turn 2 contains full history
            req2 = captured_requests[1]
            assert len(req2["messages"]) == 3
            assert req2["messages"][0]["role"] == "user"
            assert req2["messages"][1]["role"] == "assistant"
            assert req2["messages"][1]["tool_calls"][0]["id"] == "call_stock_001"
            assert req2["messages"][2]["role"] == "tool"
            assert req2["messages"][2]["tool_call_id"] == "call_stock_001"

            # Step 3: Hermes client executes Tool 2 and appends to conversation
            conversation.append(msg2)
            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call_2["id"],
                "name": tool_call_2["function"]["name"],
                "content": json.dumps({"price": 225.0, "eps": 6.5, "pe_ratio": 34.61}),
            })

            # Send Turn 3 request (Final synthesis)
            resp3 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": conversation,
                    "tools": ALL_TOOLS,
                },
            )
            assert resp3.status_code == 200
            data3 = resp3.json()
            assert data3["choices"][0]["finish_reason"] == "stop"
            msg3 = data3["choices"][0]["message"]
            assert msg3["role"] == "assistant"
            assert "34.61" in msg3["content"]
            assert msg3.get("tool_calls") is None

            # Verify upstream request for Turn 3 contains all 5 prior messages
            req3 = captured_requests[2]
            assert len(req3["messages"]) == 5
            assert [m["role"] for m in req3["messages"]] == ["user", "assistant", "tool", "assistant", "tool"]
            assert req3["messages"][1]["tool_calls"][0]["id"] == "call_stock_001"
            assert req3["messages"][2]["tool_call_id"] == "call_stock_001"
            assert req3["messages"][3]["tool_calls"][0]["id"] == "call_pe_002"
            assert req3["messages"][4]["tool_call_id"] == "call_pe_002"

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_hermes_continuous_multi_cycle_tool_calling_streaming():
    """Stress test: 2 sequential tool calling cycles before final text synthesis (Streaming SSE).

    All turns use stream=True.
    """
    captured_requests: list[dict[str, Any]] = []

    stream_cycle_1 = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_stock_s1","type":"function","function":{"name":"get_stock_quote","arguments":""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"ticker\\": "}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"AAPL\\"}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":25,"completion_tokens":15,"total_tokens":40}}\n\n'
        b"data: [DONE]\n\n"
    )

    stream_cycle_2 = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_pe_s2","type":"function","function":{"name":"calculate_pe","arguments":""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"price\\": 225.0, "}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"eps\\": 6.5}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":60,"completion_tokens":20,"total_tokens":80}}\n\n'
        b"data: [DONE]\n\n"
    )

    stream_cycle_3 = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"AAPL P/E "},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"ratio is 34.61."},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":95,"completion_tokens":15,"total_tokens":110}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        call_idx = len(captured_requests)
        if call_idx == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_cycle_1)
        elif call_idx == 2:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_cycle_2)
        else:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_cycle_3)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
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
            conversation = [
                {"role": "user", "content": "What is AAPL P/E ratio?"}
            ]

            # Stream Turn 1
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": conversation,
                    "tools": ALL_TOOLS,
                },
            ) as resp1:
                lines1 = list(resp1.iter_lines())

            assert resp1.status_code == 200
            events1 = [json.loads(l[6:]) for l in lines1 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            assert events1[-1]["choices"][0]["finish_reason"] == "tool_calls"
            # Reconstruct tool call 1
            tc1_id = events1[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
            tc1_name = events1[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            tc1_args = "".join(
                ev["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                for ev in events1
                if "tool_calls" in ev["choices"][0]["delta"] and "arguments" in ev["choices"][0]["delta"]["tool_calls"][0]["function"]
            )
            assert tc1_id == "call_stock_s1"
            assert tc1_name == "get_stock_quote"
            assert json.loads(tc1_args) == {"ticker": "AAPL"}

            # Build history for Turn 2
            conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc1_id,
                        "type": "function",
                        "function": {"name": tc1_name, "arguments": tc1_args},
                    }
                ],
            })
            conversation.append({
                "role": "tool",
                "tool_call_id": tc1_id,
                "name": tc1_name,
                "content": json.dumps({"ticker": "AAPL", "price": 225.0}),
            })

            # Stream Turn 2
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": conversation,
                    "tools": ALL_TOOLS,
                },
            ) as resp2:
                lines2 = list(resp2.iter_lines())

            assert resp2.status_code == 200
            events2 = [json.loads(l[6:]) for l in lines2 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            assert events2[-1]["choices"][0]["finish_reason"] == "tool_calls"
            tc2_id = events2[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
            tc2_name = events2[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            tc2_args = "".join(
                ev["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                for ev in events2
                if "tool_calls" in ev["choices"][0]["delta"] and "arguments" in ev["choices"][0]["delta"]["tool_calls"][0]["function"]
            )
            assert tc2_id == "call_pe_s2"
            assert tc2_name == "calculate_pe"
            assert json.loads(tc2_args) == {"price": 225.0, "eps": 6.5}

            # Build history for Turn 3
            conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc2_id,
                        "type": "function",
                        "function": {"name": tc2_name, "arguments": tc2_args},
                    }
                ],
            })
            conversation.append({
                "role": "tool",
                "tool_call_id": tc2_id,
                "name": tc2_name,
                "content": json.dumps({"price": 225.0, "eps": 6.5, "pe_ratio": 34.61}),
            })

            # Stream Turn 3 (Final text)
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": conversation,
                    "tools": ALL_TOOLS,
                },
            ) as resp3:
                lines3 = list(resp3.iter_lines())

            assert resp3.status_code == 200
            events3 = [json.loads(l[6:]) for l in lines3 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            assert events3[-1]["choices"][0]["finish_reason"] == "stop"
            final_text = "".join(
                ev["choices"][0]["delta"].get("content", "")
                for ev in events3
                if "content" in ev["choices"][0]["delta"]
            )
            assert final_text == "AAPL P/E ratio is 34.61."

            # Verify upstream requests received stream=True and preserved messages
            assert len(captured_requests) == 3
            for r in captured_requests:
                assert r["stream"] is True
            assert len(captured_requests[2]["messages"]) == 5

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_tool_choice_payload_mapping_variants():
    """Verify tool_choice upstream payload mapping: 'none', 'required', 'auto', and dict."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        tc = body.get("tool_choice")
        if tc == "none":
            # Model does not call tool, returns text
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "I am not allowed to use tools."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 10},
                },
            )
        else:
            # Model calls tool
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
                                        "id": "call_req_1",
                                        "type": "function",
                                        "function": {"name": "get_stock_quote", "arguments": '{"ticker": "NVDA"}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"total_tokens": 20},
                },
            )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
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
            # 1. tool_choice: "none"
            resp_none = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Tell me a joke without tools"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "none",
                },
            )
            assert resp_none.status_code == 200
            data_none = resp_none.json()
            assert data_none["choices"][0]["finish_reason"] == "stop"
            assert data_none["choices"][0]["message"]["content"] == "I am not allowed to use tools."
            assert data_none["choices"][0]["message"].get("tool_calls") is None

            assert captured_requests[0]["tool_choice"] == "none"
            assert captured_requests[0]["tools"] == ALL_TOOLS

            # 2. tool_choice: "required"
            resp_req = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "You must call a tool"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "required",
                },
            )
            assert resp_req.status_code == 200
            data_req = resp_req.json()
            assert data_req["choices"][0]["finish_reason"] == "tool_calls"
            assert captured_requests[1]["tool_choice"] == "required"

            # 3. tool_choice: "auto"
            resp_auto = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Auto tool choice"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "auto",
                },
            )
            assert resp_auto.status_code == 200
            assert captured_requests[2]["tool_choice"] == "auto"

            # 4. tool_choice: named function object
            named_choice = {"type": "function", "function": {"name": "get_stock_quote"}}
            resp_named = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Get stock"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": named_choice,
                },
            )
            assert resp_named.status_code == 200
            assert captured_requests[3]["tool_choice"] == named_choice

            # 5. parallel_tool_calls: False and True mapping
            http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Test"}],
                    "tools": ALL_TOOLS,
                    "parallel_tool_calls": False,
                },
            )
            assert captured_requests[4]["parallel_tool_calls"] is False

            http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Test"}],
                    "tools": ALL_TOOLS,
                    "parallel_tool_calls": True,
                },
            )
            assert captured_requests[5]["parallel_tool_calls"] is True

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_hermes_parallel_tool_calling_multi_turn_cycle():
    """Verify Hermes pattern with 2 parallel tool calls in Turn 1, and 2 tool responses in Turn 2."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        if len(captured_requests) == 1:
            # Parallel tool calls for NVDA and MSFT
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
                                        "id": "call_nvda",
                                        "type": "function",
                                        "function": {"name": "get_stock_quote", "arguments": '{"ticker": "NVDA"}'},
                                    },
                                    {
                                        "id": "call_msft",
                                        "type": "function",
                                        "function": {"name": "get_stock_quote", "arguments": '{"ticker": "MSFT"}'},
                                    },
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"total_tokens": 40},
                },
            )
        else:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "NVDA is $128 and MSFT is $430.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 80},
                },
            )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
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
            resp1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Compare NVDA and MSFT"}],
                    "tools": [TOOL_STOCK],
                    "parallel_tool_calls": True,
                },
            )
            assert resp1.status_code == 200
            data1 = resp1.json()
            tool_calls = data1["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 2
            assert tool_calls[0]["id"] == "call_nvda"
            assert tool_calls[1]["id"] == "call_msft"

            # Turn 2: Hermes client provides 2 tool results
            turn2_messages = [
                {"role": "user", "content": "Compare NVDA and MSFT"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_nvda",
                    "name": "get_stock_quote",
                    "content": '{"price": 128.0}',
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_msft",
                    "name": "get_stock_quote",
                    "content": '{"price": 430.0}',
                },
            ]

            resp2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": turn2_messages,
                    "tools": [TOOL_STOCK],
                },
            )
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["choices"][0]["finish_reason"] == "stop"
            assert "NVDA is $128 and MSFT is $430." in data2["choices"][0]["message"]["content"]

            # Verify upstream request body
            req2 = captured_requests[1]
            assert len(req2["messages"]) == 4
            assert req2["messages"][1]["role"] == "assistant"
            assert len(req2["messages"][1]["tool_calls"]) == 2
            assert req2["messages"][2]["tool_call_id"] == "call_nvda"
            assert req2["messages"][3]["tool_call_id"] == "call_msft"

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_streaming_interleaved_parallel_tool_calls():
    """Verify SSE streaming with interleaved chunk deltas for parallel tool calls."""
    stream_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_stock_quote","arguments":""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_stock_quote","arguments":""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"ticker\\": "}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"ticker\\": "}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"NVDA\\"}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\\"MSFT\\"}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"total_tokens":50}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
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
                    "messages": [{"role": "user", "content": "Parallel stream"}],
                    "tools": [TOOL_STOCK],
                },
            ) as resp:
                lines = list(resp.iter_lines())

            assert resp.status_code == 200
            events = [json.loads(l[6:]) for l in lines if l.startswith("data: ") and l.strip() != "data: [DONE]"]

            # Reconstruct tool calls by index
            reconstructed: dict[int, dict[str, Any]] = {}
            for ev in events:
                delta = ev["choices"][0]["delta"]
                for tc in delta.get("tool_calls", []):
                    idx = tc["index"]
                    if idx not in reconstructed:
                        reconstructed[idx] = {"id": tc.get("id"), "name": tc.get("function", {}).get("name"), "arguments": ""}
                    if tc.get("id"):
                        reconstructed[idx]["id"] = tc["id"]
                    if tc.get("function", {}).get("name"):
                        reconstructed[idx]["name"] = tc["function"]["name"]
                    if tc.get("function", {}).get("arguments"):
                        reconstructed[idx]["arguments"] += tc["function"]["arguments"]

            assert len(reconstructed) == 2
            assert reconstructed[0]["id"] == "call_1"
            assert json.loads(reconstructed[0]["arguments"]) == {"ticker": "NVDA"}
            assert reconstructed[1]["id"] == "call_2"
            assert json.loads(reconstructed[1]["arguments"]) == {"ticker": "MSFT"}

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_edge_case_tool_message_contents():
    """Verify tool message containing multi-byte Unicode, empty strings, and special characters."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Processed tool result successfully."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 30},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
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
            # Multi-byte Korean & emojis
            korean_content = '{"status": "완료", "result": "서울특별시 종로구 날씨 맑음 ☀️ \n\t 온도: 24°C"}'
            messages = [
                {"role": "user", "content": "날씨 어때?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather_kr",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "서울"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather_kr",
                    "name": "get_weather",
                    "content": korean_content,
                },
            ]

            resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": messages,
                },
            )
            assert resp.status_code == 200
            assert captured_requests[0]["messages"][2]["content"] == korean_content

            # Empty string tool result content
            messages_empty = [
                {"role": "user", "content": "Run action"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_action_0",
                            "type": "function",
                            "function": {"name": "ping", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_action_0",
                    "name": "ping",
                    "content": "",
                },
            ]

            resp_empty = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": messages_empty,
                },
            )
            assert resp_empty.status_code == 200
            assert captured_requests[1]["messages"][2]["content"] == ""

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_tool_choice_none_and_required_streaming():
    """Verify tool_choice 'none' and 'required' forwarding and SSE chunk formatting in streaming mode."""
    captured_requests: list[dict[str, Any]] = []

    stream_text_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"No tools used."},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":15}}\n\n'
        b"data: [DONE]\n\n"
    )

    stream_tool_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_req_s1","type":"function","function":{"name":"get_stock_quote","arguments":"{\\"ticker\\": \\"TSLA\\"}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"total_tokens":25}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        if body.get("tool_choice") == "none":
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_text_payload)
        else:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_tool_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
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
            # 1. tool_choice: "none" in stream
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "none",
                },
            ) as resp1:
                lines1 = list(resp1.iter_lines())

            assert resp1.status_code == 200
            assert captured_requests[0]["tool_choice"] == "none"
            assert captured_requests[0]["stream"] is True
            events1 = [json.loads(l[6:]) for l in lines1 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            assert events1[-1]["choices"][0]["finish_reason"] == "stop"
            assert any(ev["choices"][0]["delta"].get("content") == "No tools used." for ev in events1)

            # 2. tool_choice: "required" in stream
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Quote TSLA"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "required",
                },
            ) as resp2:
                lines2 = list(resp2.iter_lines())

            assert resp2.status_code == 200
            assert captured_requests[1]["tool_choice"] == "required"
            assert captured_requests[1]["stream"] is True
            events2 = [json.loads(l[6:]) for l in lines2 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            assert events2[-1]["choices"][0]["finish_reason"] == "tool_calls"
            assert events2[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_req_s1"

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_upstream_error_propagation_for_tool_calling():
    """Verify that 400/500 errors from Foundry upstream are mapped to OpenAI error response format."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid schema for tool get_weather: missing type",
                    "type": "invalid_request_error",
                    "code": "invalid_tool_schema",
                }
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
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
            # Non-streaming 400 error
            resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "broken tool test"}],
                    "tools": [{"type": "function", "function": {"name": "broken"}}],
                },
            )
            assert resp.status_code == 400
            err = resp.json()["error"]
            assert "Invalid schema for tool get_weather" in err["message"]
            assert err["code"] == "invalid_tool_schema"

            # Streaming 400 error
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "broken tool test"}],
                    "tools": [{"type": "function", "function": {"name": "broken"}}],
                },
            ) as stream_resp:
                lines = list(stream_resp.iter_lines())

            assert stream_resp.status_code == 200
            # Error event sent via SSE
            err_line = [l for l in lines if "error" in l][0]
            err_event = json.loads(err_line[6:])
            assert "Invalid schema for tool get_weather" in err_event["error"]["message"]
            assert [l for l in lines if l.strip()][-1] == "data: [DONE]"

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_hermes_user_intervention_in_multi_turn_cycle():
    """Verify conversation pattern where user intervenes mid-tool-cycle:
    User -> Tool Call -> Tool Result -> User comment -> Tool Call -> Tool Result -> Synthesis.
    """
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Final synthesis after intervention.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 100},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
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
            intervened_history = [
                {"role": "user", "content": "What is the price of GOOG?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "get_stock_quote", "arguments": '{"ticker": "GOOG"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "get_stock_quote", "content": '{"price": 180.0}'},
                {"role": "user", "content": "Actually, please check GOOGL instead of GOOG."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c2", "type": "function", "function": {"name": "get_stock_quote", "arguments": '{"ticker": "GOOGL"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c2", "name": "get_stock_quote", "content": '{"price": 182.0}'},
            ]

            resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": intervened_history,
                    "tools": [TOOL_STOCK],
                },
            )
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "Final synthesis after intervention."

            # Verify upstream request body has all 6 messages preserved in order
            req = captured_requests[0]
            assert len(req["messages"]) == 6
            roles = [m["role"] for m in req["messages"]]
            assert roles == ["user", "assistant", "tool", "user", "assistant", "tool"]
            assert req["messages"][1]["tool_calls"][0]["id"] == "c1"
            assert req["messages"][2]["tool_call_id"] == "c1"
            assert req["messages"][4]["tool_calls"][0]["id"] == "c2"
            assert req["messages"][5]["tool_call_id"] == "c2"

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())

