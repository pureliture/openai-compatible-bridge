"""Hermes Agent Adversarial and Stress Test Harness for Foundry xAI Responses Protocol (Milestone 4).

Conducted by Challenger 2 (teamwork_preview_challenger_m4_2).
Validates:
1. Hermes Agent continuous multi-cycle tool calling:
   Turn 1 (User) -> Turn 2 (Assistant function_call 1) -> Turn 3 (Client function_call_output 1) ->
   Turn 4 (Assistant function_call 2) -> Turn 5 (Client function_call_output 2) -> Turn 6 (Assistant text)
   Both non-streaming and streaming.
2. xAI Responses API unique `input` items sequence integrity and strict ordering:
   - System message preservation
   - Interleaved assistant text thoughts + function_call items
   - function_call_output items with exact call_id matching
   - Interspersed multi-turn user/assistant/tool messages
3. Parallel multi-tool calling multi-turn cycles (non-streaming and interleaved streaming).
4. Edge-case and hostile tool execution payloads:
   - Nested JSON with Korean and emoji (ensure_ascii=False preservation)
   - Python dict, list, int, bool, None, empty string
   - Large (>10KB) tool output payloads
5. Streaming chunk fragmentation & interleaved argument deltas across multiple function calls.
6. Exhaustive tool_choice upstream mapping ("auto", "none", "required", named dict variations).
7. Stream fallback finish_reason and error propagation (502 / response.failed).
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


class _DummyProvider:
    async def close(self) -> None:
        pass


TOOL_STOCK: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_stock_quote",
        "description": "Fetch current stock quote and financials",
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

TOOL_MARKET_CAP: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_market_cap",
        "description": "Fetch company market capitalization",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
}

ALL_TOOLS = [TOOL_STOCK, TOOL_CALC_PE, TOOL_MARKET_CAP]


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
# Test 1: Continuous Multi-Cycle Tool Calling (Non-Streaming)
# ==============================================================================


def test_hermes_xai_continuous_multi_cycle_non_stream():
    """Stress test: 2 sequential tool calling cycles before final text synthesis (Non-streaming).

    Turn 1 (User Query) -> Tool Call 1 (get_stock_quote)
    Turn 2 (Tool Result 1) -> Tool Call 2 (calculate_pe)
    Turn 3 (Tool Result 2) -> Final Synthesis Text
    """
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/xai/v1/responses")
        assert request.headers.get("authorization") == "Bearer test-token"
        body = json.loads(request.content)
        captured_requests.append(body)

        call_idx = len(captured_requests)
        if call_idx == 1:
            # Cycle 1 response: assistant invokes get_stock_quote
            return httpx.Response(
                200,
                json={
                    "id": "resp_cycle_1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_stock_101",
                            "name": "get_stock_quote",
                            "arguments": '{"ticker": "NVDA"}',
                        }
                    ],
                    "usage": {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45},
                },
            )
        elif call_idx == 2:
            # Cycle 2 response: assistant invokes calculate_pe
            return httpx.Response(
                200,
                json={
                    "id": "resp_cycle_2",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_pe_202",
                            "name": "calculate_pe",
                            "arguments": '{"price": 130.0, "eps": 2.5}',
                        }
                    ],
                    "usage": {"input_tokens": 65, "output_tokens": 20, "total_tokens": 85},
                },
            )
        else:
            # Final synthesis response: assistant returns final answer
            return httpx.Response(
                200,
                json={
                    "id": "resp_cycle_3",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "NVDA is trading at $130.00 with EPS of 2.5, resulting in a P/E ratio of 52.0.",
                                }
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 120, "output_tokens": 35, "total_tokens": 155},
                },
            )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            # Turn 1: User asks for valuation
            r1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "What is the P/E ratio for NVDA?"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "auto",
                },
            )
            assert r1.status_code == 200
            d1 = r1.json()
            assert d1["choices"][0]["finish_reason"] == "tool_calls"
            assert d1["choices"][0]["message"]["content"] is None
            tc1 = d1["choices"][0]["message"]["tool_calls"]
            assert len(tc1) == 1
            assert tc1[0]["id"] == "call_stock_101"
            assert tc1[0]["function"]["name"] == "get_stock_quote"
            assert json.loads(tc1[0]["function"]["arguments"]) == {"ticker": "NVDA"}

            # Validate upstream request 1
            inp1 = captured_requests[0]["input"]
            assert len(inp1) == 1
            assert inp1[0] == {"role": "user", "content": "What is the P/E ratio for NVDA?"}

            # Turn 2: Hermes Client provides Tool Result 1
            messages_turn2 = [
                {"role": "user", "content": "What is the P/E ratio for NVDA?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tc1,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_stock_101",
                    "content": json.dumps({"ticker": "NVDA", "price": 130.0, "eps": 2.5}),
                },
            ]
            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": messages_turn2,
                    "tools": ALL_TOOLS,
                },
            )
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["choices"][0]["finish_reason"] == "tool_calls"
            tc2 = d2["choices"][0]["message"]["tool_calls"]
            assert len(tc2) == 1
            assert tc2[0]["id"] == "call_pe_202"
            assert tc2[0]["function"]["name"] == "calculate_pe"
            assert json.loads(tc2[0]["function"]["arguments"]) == {"price": 130.0, "eps": 2.5}

            # Validate upstream request 2 input sequence integrity
            inp2 = captured_requests[1]["input"]
            assert len(inp2) == 3
            assert inp2[0] == {"role": "user", "content": "What is the P/E ratio for NVDA?"}
            assert inp2[1] == {
                "type": "function_call",
                "call_id": "call_stock_101",
                "name": "get_stock_quote",
                "arguments": '{"ticker": "NVDA"}',
            }
            assert inp2[2] == {
                "type": "function_call_output",
                "call_id": "call_stock_101",
                "output": '{"ticker": "NVDA", "price": 130.0, "eps": 2.5}',
            }

            # Turn 3: Hermes Client provides Tool Result 2
            messages_turn3 = messages_turn2 + [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tc2,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_pe_202",
                    "content": json.dumps({"pe_ratio": 52.0}),
                },
            ]
            r3 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": messages_turn3,
                    "tools": ALL_TOOLS,
                },
            )
            assert r3.status_code == 200
            d3 = r3.json()
            assert d3["choices"][0]["finish_reason"] == "stop"
            assert d3["choices"][0]["message"].get("tool_calls") is None
            assert "P/E ratio of 52.0" in d3["choices"][0]["message"]["content"]

            # Validate upstream request 3 input sequence integrity (exact 5 items)
            inp3 = captured_requests[2]["input"]
            assert len(inp3) == 5
            assert inp3[0] == {"role": "user", "content": "What is the P/E ratio for NVDA?"}
            assert inp3[1] == {
                "type": "function_call",
                "call_id": "call_stock_101",
                "name": "get_stock_quote",
                "arguments": '{"ticker": "NVDA"}',
            }
            assert inp3[2] == {
                "type": "function_call_output",
                "call_id": "call_stock_101",
                "output": '{"ticker": "NVDA", "price": 130.0, "eps": 2.5}',
            }
            assert inp3[3] == {
                "type": "function_call",
                "call_id": "call_pe_202",
                "name": "calculate_pe",
                "arguments": '{"price": 130.0, "eps": 2.5}',
            }
            assert inp3[4] == {
                "type": "function_call_output",
                "call_id": "call_pe_202",
                "output": '{"pe_ratio": 52.0}',
            }
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 2: Continuous Multi-Cycle Tool Calling (SSE Streaming)
# ==============================================================================


def test_hermes_xai_continuous_multi_cycle_stream():
    """Stress test: 2 sequential tool calling cycles with SSE streaming."""
    captured_requests: list[dict[str, Any]] = []

    # Stream 1: invokes get_stock_quote
    stream_1 = (
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_s1","call_id":"call_s1","name":"get_stock_quote","arguments":""}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_s1","delta":"{\\"ticker\\": "}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_s1","delta":"\\"AAPL\\"}"}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":30,"output_tokens":20,"total_tokens":50}}}\n\n'
        b"data: [DONE]\n\n"
    )

    # Stream 2: invokes calculate_pe
    stream_2 = (
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_p2","call_id":"call_p2","name":"calculate_pe","arguments":""}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_p2","delta":"{\\"price\\": 220.0, "}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_p2","delta":"\\"eps\\": 6.5}"}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":70,"output_tokens":25,"total_tokens":95}}}\n\n'
        b"data: [DONE]\n\n"
    )

    # Stream 3: text synthesis
    stream_3 = (
        b'data: {"type":"response.output_text.delta","delta":"AAPL valuation: "}\n\n'
        b'data: {"type":"response.output_text.delta","delta":"P/E ratio is 33.85."}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":125,"output_tokens":30,"total_tokens":155}}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        call_idx = len(captured_requests)
        payload = stream_1 if call_idx == 1 else (stream_2 if call_idx == 2 else stream_3)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            # Stream Cycle 1
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "Evaluate AAPL"}],
                    "tools": ALL_TOOLS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                chunks_1 = []
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks_1.append(json.loads(line[len("data: ") :]))

            # Accumulate tool call from stream 1
            args_accum = ""
            fn_name = ""
            call_id = ""
            saw_tool_calls_finish = False
            for c in chunks_1:
                choice = c["choices"][0]
                if choice.get("finish_reason") == "tool_calls":
                    saw_tool_calls_finish = True
                delta = choice.get("delta", {})
                if "tool_calls" in delta:
                    tc = delta["tool_calls"][0]
                    if tc.get("id"):
                        call_id = tc["id"]
                    if tc.get("function", {}).get("name"):
                        fn_name = tc["function"]["name"]
                    args_accum += tc.get("function", {}).get("arguments", "")

            assert saw_tool_calls_finish is True
            assert call_id == "call_s1"
            assert fn_name == "get_stock_quote"
            assert json.loads(args_accum) == {"ticker": "AAPL"}

            # Stream Cycle 2
            messages_turn2 = [
                {"role": "user", "content": "Evaluate AAPL"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": args_accum},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps({"ticker": "AAPL", "price": 220.0, "eps": 6.5}),
                },
            ]
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": messages_turn2,
                    "tools": ALL_TOOLS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                chunks_2 = []
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks_2.append(json.loads(line[len("data: ") :]))

            args_accum_2 = ""
            fn_name_2 = ""
            call_id_2 = ""
            saw_tool_calls_finish_2 = False
            for c in chunks_2:
                choice = c["choices"][0]
                if choice.get("finish_reason") == "tool_calls":
                    saw_tool_calls_finish_2 = True
                delta = choice.get("delta", {})
                if "tool_calls" in delta:
                    tc = delta["tool_calls"][0]
                    if tc.get("id"):
                        call_id_2 = tc["id"]
                    if tc.get("function", {}).get("name"):
                        fn_name_2 = tc["function"]["name"]
                    args_accum_2 += tc.get("function", {}).get("arguments", "")

            assert saw_tool_calls_finish_2 is True
            assert call_id_2 == "call_p2"
            assert fn_name_2 == "calculate_pe"
            assert json.loads(args_accum_2) == {"price": 220.0, "eps": 6.5}

            # Stream Cycle 3: Final synthesis
            messages_turn3 = messages_turn2 + [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id_2,
                            "type": "function",
                            "function": {"name": fn_name_2, "arguments": args_accum_2},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id_2,
                    "content": json.dumps({"pe": 33.85}),
                },
            ]
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": messages_turn3,
                    "tools": ALL_TOOLS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                text_accum = ""
                finish_reason_3 = None
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        c = json.loads(line[len("data: ") :])
                        choice = c["choices"][0]
                        text_accum += choice.get("delta", {}).get("content", "")
                        if choice.get("finish_reason"):
                            finish_reason_3 = choice["finish_reason"]

            assert finish_reason_3 == "stop"
            assert "AAPL valuation: P/E ratio is 33.85." in text_accum

            # Validate upstream request 3 input sequence
            inp3 = captured_requests[2]["input"]
            assert len(inp3) == 5
            assert inp3[0]["role"] == "user"
            assert inp3[1]["type"] == "function_call"
            assert inp3[1]["call_id"] == "call_s1"
            assert inp3[2]["type"] == "function_call_output"
            assert inp3[2]["call_id"] == "call_s1"
            assert inp3[3]["type"] == "function_call"
            assert inp3[3]["call_id"] == "call_p2"
            assert inp3[4]["type"] == "function_call_output"
            assert inp3[4]["call_id"] == "call_p2"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 3: Parallel Tool Calling Multi-Turn Cycle
# ==============================================================================


def test_hermes_xai_parallel_multi_cycle():
    """Stress test: Turn 1 generates 2 parallel tool calls; Turn 2 generates 1 follow-up call."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        idx = len(captured_requests)

        if idx == 1:
            # Parallel tool calls: quote for MSFT and AAPL
            return httpx.Response(
                200,
                json={
                    "id": "resp_p1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_msft",
                            "name": "get_stock_quote",
                            "arguments": '{"ticker": "MSFT"}',
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_aapl",
                            "name": "get_stock_quote",
                            "arguments": '{"ticker": "AAPL"}',
                        },
                    ],
                    "usage": {"input_tokens": 40, "output_tokens": 30, "total_tokens": 70},
                },
            )
        elif idx == 2:
            # Follow-up tool call: calculate market cap comparison
            return httpx.Response(
                200,
                json={
                    "id": "resp_p2",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_mcap",
                            "name": "get_market_cap",
                            "arguments": '{"ticker": "MSFT"}',
                        }
                    ],
                    "usage": {"input_tokens": 90, "output_tokens": 20, "total_tokens": 110},
                },
            )
        else:
            return httpx.Response(
                200,
                json={
                    "id": "resp_p3",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Comparison complete."}],
                        }
                    ],
                    "usage": {"input_tokens": 140, "output_tokens": 10, "total_tokens": 150},
                },
            )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            # Turn 1
            r1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "Compare MSFT and AAPL"}],
                    "tools": ALL_TOOLS,
                },
            )
            assert r1.status_code == 200
            d1 = r1.json()
            tcs1 = d1["choices"][0]["message"]["tool_calls"]
            assert len(tcs1) == 2
            assert tcs1[0]["id"] == "call_msft"
            assert tcs1[1]["id"] == "call_aapl"

            # Turn 2: Hermes Client executes both tools and sends both tool results
            messages_turn2 = [
                {"role": "user", "content": "Compare MSFT and AAPL"},
                {"role": "assistant", "content": None, "tool_calls": tcs1},
                {
                    "role": "tool",
                    "tool_call_id": "call_msft",
                    "content": json.dumps({"ticker": "MSFT", "price": 420.0}),
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_aapl",
                    "content": json.dumps({"ticker": "AAPL", "price": 220.0}),
                },
            ]
            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": messages_turn2,
                    "tools": ALL_TOOLS,
                },
            )
            assert r2.status_code == 200
            d2 = r2.json()
            tcs2 = d2["choices"][0]["message"]["tool_calls"]
            assert len(tcs2) == 1
            assert tcs2[0]["id"] == "call_mcap"

            # Validate input sequence for Turn 2:
            # 1 user + 2 function_calls + 2 function_call_outputs
            inp2 = captured_requests[1]["input"]
            assert len(inp2) == 5
            assert inp2[0]["role"] == "user"
            assert inp2[1]["type"] == "function_call" and inp2[1]["call_id"] == "call_msft"
            assert inp2[2]["type"] == "function_call" and inp2[2]["call_id"] == "call_aapl"
            assert inp2[3]["type"] == "function_call_output" and inp2[3]["call_id"] == "call_msft"
            assert inp2[4]["type"] == "function_call_output" and inp2[4]["call_id"] == "call_aapl"

            # Turn 3: Hermes Client sends market cap result
            messages_turn3 = messages_turn2 + [
                {"role": "assistant", "content": None, "tool_calls": tcs2},
                {
                    "role": "tool",
                    "tool_call_id": "call_mcap",
                    "content": json.dumps({"ticker": "MSFT", "mcap": "3.1T"}),
                },
            ]
            r3 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": messages_turn3,
                    "tools": ALL_TOOLS,
                },
            )
            assert r3.status_code == 200
            assert r3.json()["choices"][0]["finish_reason"] == "stop"

            # Validate input sequence for Turn 3 (7 items total)
            inp3 = captured_requests[2]["input"]
            assert len(inp3) == 7
            assert inp3[5]["type"] == "function_call" and inp3[5]["call_id"] == "call_mcap"
            assert inp3[6]["type"] == "function_call_output" and inp3[6]["call_id"] == "call_mcap"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 4: Streaming Interleaved Parallel Function Calls with Fragmented Arguments
# ==============================================================================


def test_hermes_xai_parallel_stream_interleaved_fragmented():
    """Stress test: Streaming 2 parallel function calls with interleaved argument deltas across indices."""
    stream_payload = (
        # Call 1 added (output_index 0)
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_p1","call_id":"call_p1","name":"get_stock_quote","arguments":""}}\n\n'
        # Call 2 added (output_index 1)
        b'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"call_p2","call_id":"call_p2","name":"get_market_cap","arguments":""}}\n\n'
        # Fragmented delta for Call 1
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_p1","delta":"{\\"tic"}\n\n'
        # Fragmented delta for Call 2 (using item_id instead of call_id)
        b'data: {"type":"response.function_call_arguments.delta","item_id":"call_p2","delta":"{\\"tic"}\n\n'
        # More deltas for Call 1
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_p1","delta":"ker\\": \\"GOOG\\"}"}\n\n'
        # More deltas for Call 2 (using output_index)
        b'data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"ker\\": \\"GOOG\\"}"}\n\n'
        # Done events
        b'data: {"type":"response.output_item.done","output_index":0}\n\n'
        b'data: {"type":"response.output_item.done","output_index":1}\n\n'
        # Completed
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":50,"output_tokens":40,"total_tokens":90}}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "Fetch quote and cap for GOOG"}],
                    "tools": ALL_TOOLS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                tool_calls_map: dict[int, dict[str, Any]] = {}
                saw_finish_tool_calls = False

                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[len("data: ") :])
                        choice = chunk["choices"][0]
                        if choice.get("finish_reason") == "tool_calls":
                            saw_finish_tool_calls = True
                        delta = choice.get("delta", {})
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc["index"]
                                if idx not in tool_calls_map:
                                    tool_calls_map[idx] = {
                                        "id": "",
                                        "name": "",
                                        "arguments": "",
                                    }
                                if tc.get("id"):
                                    tool_calls_map[idx]["id"] = tc["id"]
                                if tc.get("function", {}).get("name"):
                                    tool_calls_map[idx]["name"] = tc["function"]["name"]
                                tool_calls_map[idx]["arguments"] += tc.get("function", {}).get("arguments", "")

                assert saw_finish_tool_calls is True
                assert len(tool_calls_map) == 2

                # Check index 0: call_p1, get_stock_quote, GOOG
                assert tool_calls_map[0]["id"] == "call_p1"
                assert tool_calls_map[0]["name"] == "get_stock_quote"
                assert json.loads(tool_calls_map[0]["arguments"]) == {"ticker": "GOOG"}

                # Check index 1: call_p2, get_market_cap, GOOG
                assert tool_calls_map[1]["id"] == "call_p2"
                assert tool_calls_map[1]["name"] == "get_market_cap"
                assert json.loads(tool_calls_map[1]["arguments"]) == {"ticker": "GOOG"}
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 5: Assistant Reasoning Thoughts + Tool Calls (Interleaved Text & Tools)
# ==============================================================================


def test_hermes_xai_interleaved_assistant_thought_and_tool_call():
    """Stress test: Assistant emits reasoning text followed by function_call in history and stream."""
    captured_requests: list[dict[str, Any]] = []

    # Stream returns reasoning output_text delta, then function_call
    stream_payload = (
        b'data: {"type":"response.output_text.delta","delta":"I should check AAPL quote first."}\n\n'
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_th1","call_id":"call_th1","name":"get_stock_quote","arguments":""}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_th1","delta":"{\\"ticker\\": \\"AAPL\\"}"}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":40,"output_tokens":25,"total_tokens":65}}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            # Turn 1: Assistant emits text thought + tool call via stream
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "What is AAPL?"}],
                    "tools": ALL_TOOLS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                accum_text = ""
                tool_calls: list[dict[str, Any]] = []
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[len("data: ") :])
                        delta = chunk["choices"][0]["delta"]
                        accum_text += delta.get("content", "")
                        if "tool_calls" in delta:
                            tool_calls.extend(delta["tool_calls"])

                assert "I should check AAPL quote first." in accum_text
                assert len(tool_calls) >= 1

            # Turn 2: Hermes Client passes assistant message WITH text thought AND tool_calls
            messages_turn2 = [
                {"role": "user", "content": "What is AAPL?"},
                {
                    "role": "assistant",
                    "content": "I should check AAPL quote first.",
                    "tool_calls": [
                        {
                            "id": "call_th1",
                            "type": "function",
                            "function": {"name": "get_stock_quote", "arguments": '{"ticker": "AAPL"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_th1",
                    "content": '{"price": 220.0}',
                },
            ]
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": messages_turn2,
                    "tools": ALL_TOOLS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200

            # Validate upstream request 2:
            # Must preserve assistant text message followed by function_call item followed by function_call_output!
            inp2 = captured_requests[1]["input"]
            assert len(inp2) == 4
            assert inp2[0] == {"role": "user", "content": "What is AAPL?"}
            assert inp2[1] == {"role": "assistant", "content": "I should check AAPL quote first."}
            assert inp2[2] == {
                "type": "function_call",
                "call_id": "call_th1",
                "name": "get_stock_quote",
                "arguments": '{"ticker": "AAPL"}',
            }
            assert inp2[3] == {
                "type": "function_call_output",
                "call_id": "call_th1",
                "output": '{"price": 220.0}',
            }
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 6: Hostile / Edge-Case Tool Result Data Types, Unicode & Large Payloads
# ==============================================================================


def test_hermes_xai_tool_result_data_types_and_unicode():
    """Stress test: function_call_output handling with Korean, emoji, dicts, lists, numbers, booleans, None, and >10KB payload."""
    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")

    # 1. Direct unit-level stress test for _xai_input_messages with native Python types
    native_test_cases = [
        # Native Dict with Korean & Emoji
        ({"종목명": "삼성전자", "현재가": 75000, "상태": "정상🚀"}, '{"종목명": "삼성전자", "현재가": 75000, "상태": "정상🚀"}'),
        # Native List
        ([1, "two", {"three": 3}], '[1, "two", {"three": 3}]'),
        # Integer
        (42, "42"),
        # Float
        (3.14159, "3.14159"),
        # Boolean
        (True, "True"),
        # None
        (None, ""),
        # Empty string
        ("", ""),
        # Large payload (>10KB)
        ({"data": "X" * 15000}, json.dumps({"data": "X" * 15000})),
    ]

    for idx, (raw_val, expected_str) in enumerate(native_test_cases):
        raw_messages = [
            {"role": "user", "content": "run tool"},
            {"role": "assistant", "tool_calls": [{"id": f"c_{idx}", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c_{idx}", "content": raw_val},
        ]
        body = client._build_xai_request_body(
            model="grok-4.6",
            messages=raw_messages,
            max_tokens=1000,
            temperature=None,
            top_p=None,
            stop=None,
            response_format=None,
            reasoning_effort=None,
            stream=False,
        )
        inp = body["input"]
        assert len(inp) == 3
        out_item = inp[2]
        assert out_item["type"] == "function_call_output"
        assert out_item["call_id"] == f"c_{idx}"
        assert out_item["output"] == expected_str

    # 2. HTTP E2E TestClient test
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "resp_types",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "All results processed."}],
                    }
                ],
                "usage": {"input_tokens": 500, "output_tokens": 10, "total_tokens": 510},
            },
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
            large_str = json.dumps({"items": [{"id": i, "val": f"data_{i}" * 50} for i in range(100)]})
            korean_str = json.dumps({"종목명": "삼성전자", "거래가": 75000, "상태": "정상 거래중 🚀", "알림": "확인 완료!"}, ensure_ascii=False)

            messages = [
                {"role": "user", "content": "Execute multiple tools"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
                        {"id": "c2", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
                        {"id": "c3", "type": "function", "function": {"name": "f3", "arguments": "{}"}},
                        {"id": "c4", "type": "function", "function": {"name": "f4", "arguments": "{}"}},
                        {"id": "c5", "type": "function", "function": {"name": "f5", "arguments": "{}"}},
                    ],
                },
                # 1. JSON String with Korean & Emoji
                {"role": "tool", "tool_call_id": "c1", "content": korean_str},
                # 2. Plain text
                {"role": "tool", "tool_call_id": "c2", "content": "Simple output text"},
                # 3. None / null
                {"role": "tool", "tool_call_id": "c3", "content": None},
                # 4. Empty string
                {"role": "tool", "tool_call_id": "c4", "content": ""},
                # 5. Large JSON string (>10KB)
                {"role": "tool", "tool_call_id": "c5", "content": large_str},
            ]

            resp = http_client.post(
                "/v1/chat/completions",
                json={"model": "foundry:grok-4.6", "messages": messages, "tools": ALL_TOOLS},
            )
            assert resp.status_code == 200

            inp = captured_requests[0]["input"]
            # 1 user + 5 function_calls + 5 function_call_outputs = 11 items
            assert len(inp) == 11

            # Korean & Emoji preserved without ascii escaping
            out1 = inp[6]
            assert out1["call_id"] == "c1"
            assert "삼성전자" in out1["output"]
            assert "🚀" in out1["output"]

            # Large payload (>10KB)
            out5 = inp[10]
            assert out5["call_id"] == "c5"
            assert len(out5["output"]) > 10000
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 7: Exhaustive Tool Choice Mapping Variants
# ==============================================================================


def test_hermes_xai_tool_choice_exhaustive_mapping():
    """Verify tool_choice upstream mapping: 'auto', 'none', 'required', and dict formats."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "resp_tc",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Acknowledged"}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            # 1. tool_choice = "auto"
            http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "test"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "auto",
                },
            )
            assert captured_requests[0]["tool_choice"] == "auto"

            # 2. tool_choice = "none"
            http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "test"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "none",
                },
            )
            assert captured_requests[1]["tool_choice"] == "none"

            # 3. tool_choice = "required"
            http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "test"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "required",
                },
            )
            assert captured_requests[2]["tool_choice"] == "required"

            # 4. tool_choice = {"type": "function", "function": {"name": "get_stock_quote"}}
            http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "test"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": {"type": "function", "function": {"name": "get_stock_quote"}},
                },
            )
            assert captured_requests[3]["tool_choice"] == {"type": "function", "name": "get_stock_quote"}

            # 5. tool_choice = {"type": "function", "name": "calculate_pe"}
            http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "test"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": {"type": "function", "name": "calculate_pe"},
                },
            )
            assert captured_requests[4]["tool_choice"] == {"type": "function", "name": "calculate_pe"}

            # Check tools flattening: xAI tools must have top-level name, description, parameters
            for req in captured_requests:
                assert "tools" in req
                assert len(req["tools"]) == 3
                assert req["tools"][0]["name"] == "get_stock_quote"
                assert "parameters" in req["tools"][0]
                assert "function" not in req["tools"][0]
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 8: Complex Multi-User Turns with Interspersed System & Client Interventions
# ==============================================================================


def test_hermes_xai_system_messages_and_interspersed_turns():
    """Stress test: System message + initial User + tool cycle + User follow-up + second tool cycle."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "resp_complex",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Final synthesis after intervention."}],
                    }
                ],
                "usage": {"input_tokens": 180, "output_tokens": 20, "total_tokens": 200},
            },
        )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            messages = [
                {"role": "system", "content": "You are Hermes Finance Agent."},
                {"role": "user", "content": "Look up NVDA."},
                {
                    "role": "assistant",
                    "content": "Looking up NVDA...",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_stock_quote", "arguments": '{"ticker": "NVDA"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"price": 130.0}',
                },
                # User intervenes with extra requirement
                {"role": "user", "content": "Also check market cap before calculating anything."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "get_market_cap", "arguments": '{"ticker": "NVDA"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": '{"mcap": "3.2T"}',
                },
            ]

            resp = http_client.post(
                "/v1/chat/completions",
                json={"model": "foundry:grok-4.6", "messages": messages, "tools": ALL_TOOLS},
            )
            assert resp.status_code == 200
            assert "Final synthesis after intervention." in resp.json()["choices"][0]["message"]["content"]

            inp = captured_requests[0]["input"]
            # Expected sequence:
            # 0: system
            # 1: user "Look up NVDA."
            # 2: assistant "Looking up NVDA..."
            # 3: function_call call_1
            # 4: function_call_output call_1
            # 5: user "Also check market cap..."
            # 6: function_call call_2
            # 7: function_call_output call_2
            assert len(inp) == 8
            assert inp[0] == {"role": "system", "content": "You are Hermes Finance Agent."}
            assert inp[1] == {"role": "user", "content": "Look up NVDA."}
            assert inp[2] == {"role": "assistant", "content": "Looking up NVDA..."}
            assert inp[3]["type"] == "function_call" and inp[3]["call_id"] == "call_1"
            assert inp[4]["type"] == "function_call_output" and inp[4]["call_id"] == "call_1"
            assert inp[5] == {"role": "user", "content": "Also check market cap before calculating anything."}
            assert inp[6]["type"] == "function_call" and inp[6]["call_id"] == "call_2"
            assert inp[7]["type"] == "function_call_output" and inp[7]["call_id"] == "call_2"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 9: Stream Abrupt Disconnect Fallback finish_reason
# ==============================================================================


def test_hermes_xai_stream_abrupt_disconnect_fallback():
    """Verify fallback finish_reason='tool_calls' when xAI stream abruptly ends with [DONE] after function_call."""
    stream_payload = (
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_abrupt","call_id":"call_abrupt","name":"get_stock_quote","arguments":""}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_abrupt","delta":"{\\"ticker\\": \\"TSLA\\"}"}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "TSLA"}],
                    "tools": ALL_TOOLS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                chunks = []
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks.append(json.loads(line[len("data: ") :]))

                finish_reasons = [c["choices"][0].get("finish_reason") for c in chunks if c["choices"][0].get("finish_reason")]
                assert "tool_calls" in finish_reasons
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 10: Stream response.failed Error Propagation
# ==============================================================================


def test_hermes_xai_stream_response_failed_error():
    """Verify xAI stream properly propagates response.failed error as 502 OpenAI-format error."""
    stream_payload = (
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_err","call_id":"call_err","name":"test","arguments":""}}\n\n'
        b'data: {"type":"response.failed","error":{"code":"rate_limit_exceeded","message":"xAI upstream quota exhausted"}}\n\n'
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "fail"}],
                    "tools": ALL_TOOLS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                saw_error_chunk = False
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[len("data: ") :])
                        if "error" in chunk:
                            saw_error_chunk = True
                            assert "xAI upstream quota exhausted" in chunk["error"]["message"]
                assert saw_error_chunk is True
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())
