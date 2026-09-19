"""Adversarial and stress test harness for Foundry Anthropic Protocol (Milestone 3).

Conducted by Challenger 2 (teamwork_preview_challenger_m3_2).
Validates:
1. Hermes Agent continuous multi-cycle tool calling:
   Turn 1 (User) -> Turn 2 (Assistant tool_use 1) -> Turn 3 (Client tool_result 1) ->
   Turn 4 (Assistant tool_use 2) -> Turn 5 (Client tool_result 2) -> Turn 6 (Assistant text)
   Both non-streaming and streaming.
2. Anthropic Messages API strict alternation under edge-case and hostile message histories:
   - Consecutive user messages merging
   - Consecutive assistant messages merging (text + tool_calls, tool_calls + text)
   - Consecutive tool messages merging into single user turn
   - Interspersed system messages merging into top-level system parameter
   - Tool result followed by user message in same turn
   - User message followed by tool result in same turn
   - Empty content messages & empty message lists
3. Parallel multi-tool calling multi-turn cycle.
4. Streaming interleaved text and parallel tool_use blocks with zero-based tool indexing.
5. Exhaustive tool_choice upstream payload mapping.
6. Upstream error propagation and edge-case tool content serialization (Korean, emoji, dict, list, large payload).
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


def _register_foundry_anthropic_alias(
    model_alias: str = "foundry:claude-sonnet-5",
    provider_model: str = "claude-sonnet-5",
) -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY[model_alias] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": provider_model,
        "protocol": "anthropic_messages",
    }
    return old_registry


def _restore_registry(old_registry: dict[str, dict]) -> None:
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old_registry)


def _assert_strict_alternation(messages: list[dict[str, Any]]) -> None:
    """Validate that messages follow Anthropic's strict turn alternation rule.

    Roles must alternate: user, assistant, user, assistant, ...
    No two consecutive messages may share the same role.
    """
    assert len(messages) > 0, "Anthropic messages list must not be empty"
    for i in range(len(messages)):
        role = messages[i]["role"]
        assert role in {"user", "assistant"}, f"Unexpected role {role} at index {i}"
        if i > 0:
            prev_role = messages[i - 1]["role"]
            assert role != prev_role, (
                f"Strict alternation violation at index {i}: consecutive '{role}' turns"
            )


# ==============================================================================
# Test 1: Continuous Multi-Cycle Tool Calling (Non-Streaming)
# ==============================================================================


def test_hermes_anthropic_continuous_multi_cycle_non_stream():
    """Stress test: 2 sequential tool calling cycles before final text synthesis (Non-streaming).

    Turn 1 (User Query) -> Tool Call 1 (get_stock_quote)
    Turn 2 (Tool Result 1) -> Tool Call 2 (calculate_pe)
    Turn 3 (Tool Result 2) -> Final Synthesis Text
    """
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/anthropic/v1/messages")
        assert request.headers.get("anthropic-version") == "2023-06-01"
        body = json.loads(request.content)
        captured_requests.append(body)

        call_idx = len(captured_requests)
        if call_idx == 1:
            # Cycle 1 response: assistant invokes get_stock_quote
            return httpx.Response(
                200,
                json={
                    "id": "msg_cycle_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_stock_101",
                            "name": "get_stock_quote",
                            "input": {"ticker": "AAPL"},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 25, "output_tokens": 15},
                },
            )
        elif call_idx == 2:
            # Cycle 2 response: assistant invokes calculate_pe
            return httpx.Response(
                200,
                json={
                    "id": "msg_cycle_2",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_pe_202",
                            "name": "calculate_pe",
                            "input": {"price": 180.0, "eps": 6.0},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 60, "output_tokens": 20},
                },
            )
        else:
            # Final synthesis response: assistant returns final answer
            return httpx.Response(
                200,
                json={
                    "id": "msg_cycle_3",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "AAPL is trading at $180.00 with EPS of 6.0, resulting in a P/E ratio of 30.0.",
                        }
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 110, "output_tokens": 30},
                },
            )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_anthropic_alias()
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
                    "model": "foundry:claude-sonnet-5",
                    "messages": [{"role": "user", "content": "What is the P/E ratio for AAPL?"}],
                    "tools": ALL_TOOLS,
                    "tool_choice": "auto",
                },
            )
            assert r1.status_code == 200
            d1 = r1.json()
            assert d1["choices"][0]["finish_reason"] == "tool_calls"
            tc1 = d1["choices"][0]["message"]["tool_calls"]
            assert len(tc1) == 1
            assert tc1[0]["id"] == "toolu_stock_101"
            assert tc1[0]["function"]["name"] == "get_stock_quote"
            assert json.loads(tc1[0]["function"]["arguments"]) == {"ticker": "AAPL"}

            # Validate upstream request 1
            _assert_strict_alternation(captured_requests[0]["messages"])
            assert len(captured_requests[0]["messages"]) == 1
            assert captured_requests[0]["messages"][0]["role"] == "user"

            # Turn 2: Hermes Client provides Tool Result 1
            messages_turn2 = [
                {"role": "user", "content": "What is the P/E ratio for AAPL?"},
                {"role": "assistant", "content": None, "tool_calls": tc1},
                {
                    "role": "tool",
                    "tool_call_id": "toolu_stock_101",
                    "content": '{"ticker": "AAPL", "price": 180.0, "eps": 6.0}',
                },
            ]
            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": messages_turn2,
                    "tools": ALL_TOOLS,
                },
            )
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["choices"][0]["finish_reason"] == "tool_calls"
            tc2 = d2["choices"][0]["message"]["tool_calls"]
            assert len(tc2) == 1
            assert tc2[0]["id"] == "toolu_pe_202"
            assert tc2[0]["function"]["name"] == "calculate_pe"
            assert json.loads(tc2[0]["function"]["arguments"]) == {"price": 180.0, "eps": 6.0}

            # Validate upstream request 2: strict alternation [user, assistant, user]
            _assert_strict_alternation(captured_requests[1]["messages"])
            assert len(captured_requests[1]["messages"]) == 3
            assert captured_requests[1]["messages"][0]["role"] == "user"
            assert captured_requests[1]["messages"][1]["role"] == "assistant"
            assert captured_requests[1]["messages"][1]["content"][0]["type"] == "tool_use"
            assert captured_requests[1]["messages"][2]["role"] == "user"
            assert captured_requests[1]["messages"][2]["content"][0]["type"] == "tool_result"
            assert captured_requests[1]["messages"][2]["content"][0]["tool_use_id"] == "toolu_stock_101"

            # Turn 3: Hermes Client provides Tool Result 2
            messages_turn3 = messages_turn2 + [
                {"role": "assistant", "content": None, "tool_calls": tc2},
                {
                    "role": "tool",
                    "tool_call_id": "toolu_pe_202",
                    "content": '{"pe_ratio": 30.0}',
                },
            ]
            r3 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": messages_turn3,
                    "tools": ALL_TOOLS,
                },
            )
            assert r3.status_code == 200
            d3 = r3.json()
            assert d3["choices"][0]["finish_reason"] == "stop"
            assert d3["choices"][0]["message"].get("tool_calls") is None
            assert "30.0" in d3["choices"][0]["message"]["content"]

            # Validate upstream request 3: strict alternation [user, assistant, user, assistant, user]
            _assert_strict_alternation(captured_requests[2]["messages"])
            assert len(captured_requests[2]["messages"]) == 5
            roles = [m["role"] for m in captured_requests[2]["messages"]]
            assert roles == ["user", "assistant", "user", "assistant", "user"]
            assert captured_requests[2]["messages"][4]["content"][0]["tool_use_id"] == "toolu_pe_202"

    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 2: Continuous Multi-Cycle Tool Calling (Streaming SSE)
# ==============================================================================


def test_hermes_anthropic_continuous_multi_cycle_stream():
    """Stress test: 2 sequential tool calling cycles before final text synthesis (Streaming SSE).

    All 3 turns executed via stream: True and parsed chunk-by-chunk.
    """
    captured_requests: list[dict[str, Any]] = []

    stream_cycle_1 = (
        b'data: {"type":"message_start","message":{"id":"msg_s1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":20,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_s1","name":"get_stock_quote","input":{}}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"ticker\\": \\"AA"}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"PL\\"}"}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":15}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
        b"data: [DONE]\n\n"
    )

    stream_cycle_2 = (
        b'data: {"type":"message_start","message":{"id":"msg_s2","type":"message","role":"assistant","content":[],"usage":{"input_tokens":50,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_s2","name":"calculate_pe","input":{}}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"price\\": 180.0, "}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"eps\\": 6.0}"}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
        b"data: [DONE]\n\n"
    )

    stream_cycle_3 = (
        b'data: {"type":"message_start","message":{"id":"msg_s3","type":"message","role":"assistant","content":[],"usage":{"input_tokens":90,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The calculated P/E ratio "}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"for AAPL is 30.0."}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":25}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        assert body.get("stream") is True

        call_idx = len(captured_requests)
        payload = stream_cycle_1 if call_idx == 1 else (stream_cycle_2 if call_idx == 2 else stream_cycle_3)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_anthropic_alias()
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
            # Turn 1: Streaming initial tool call
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "stream": True,
                    "messages": [{"role": "user", "content": "What is AAPL P/E ratio?"}],
                    "tools": ALL_TOOLS,
                },
            ) as resp1:
                lines1 = list(resp1.iter_lines())

            assert resp1.status_code == 200
            events1 = [
                json.loads(l[len("data: ") :])
                for l in lines1
                if l.startswith("data: ") and l.strip() != "data: [DONE]"
            ]
            assert events1[-1]["choices"][0]["finish_reason"] == "tool_calls"
            # Reconstruct tool call 1
            tc1_id = events1[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
            tc1_name = events1[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            tc1_args = "".join(
                ev["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
                for ev in events1
                if "tool_calls" in ev["choices"][0]["delta"]
            )
            assert tc1_id == "toolu_s1"
            assert tc1_name == "get_stock_quote"
            assert json.loads(tc1_args) == {"ticker": "AAPL"}

            # Turn 2: Streaming tool result 1 -> tool call 2
            messages_turn2 = [
                {"role": "user", "content": "What is AAPL P/E ratio?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc1_id,
                            "type": "function",
                            "function": {"name": tc1_name, "arguments": tc1_args},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc1_id,
                    "content": '{"ticker": "AAPL", "price": 180.0, "eps": 6.0}',
                },
            ]
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "stream": True,
                    "messages": messages_turn2,
                    "tools": ALL_TOOLS,
                },
            ) as resp2:
                lines2 = list(resp2.iter_lines())

            assert resp2.status_code == 200
            events2 = [
                json.loads(l[len("data: ") :])
                for l in lines2
                if l.startswith("data: ") and l.strip() != "data: [DONE]"
            ]
            assert events2[-1]["choices"][0]["finish_reason"] == "tool_calls"
            tc2_id = events2[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
            tc2_name = events2[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            tc2_args = "".join(
                ev["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
                for ev in events2
                if "tool_calls" in ev["choices"][0]["delta"]
            )
            assert tc2_id == "toolu_s2"
            assert tc2_name == "calculate_pe"
            assert json.loads(tc2_args) == {"price": 180.0, "eps": 6.0}

            # Check upstream strict alternation on turn 2
            _assert_strict_alternation(captured_requests[1]["messages"])
            assert len(captured_requests[1]["messages"]) == 3

            # Turn 3: Streaming tool result 2 -> final synthesis text
            messages_turn3 = messages_turn2 + [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc2_id,
                            "type": "function",
                            "function": {"name": tc2_name, "arguments": tc2_args},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc2_id,
                    "content": '{"pe_ratio": 30.0}',
                },
            ]
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "stream": True,
                    "messages": messages_turn3,
                    "tools": ALL_TOOLS,
                },
            ) as resp3:
                lines3 = list(resp3.iter_lines())

            assert resp3.status_code == 200
            events3 = [
                json.loads(l[len("data: ") :])
                for l in lines3
                if l.startswith("data: ") and l.strip() != "data: [DONE]"
            ]
            assert events3[-1]["choices"][0]["finish_reason"] == "stop"
            final_text = "".join(
                ev["choices"][0]["delta"].get("content", "")
                for ev in events3
                if "content" in ev["choices"][0]["delta"]
            )
            assert "30.0" in final_text

            # Check upstream strict alternation on turn 3
            _assert_strict_alternation(captured_requests[2]["messages"])
            assert len(captured_requests[2]["messages"]) == 5
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 3: Parallel Multi-Tool Calling Multi-Turn Cycle
# ==============================================================================


def test_hermes_anthropic_parallel_tools_in_multiturn_cycle():
    """Verify parallel tool calls: assistant emits 2 tools, client returns 2 tool results.

    Tests that consecutive role: "tool" messages are merged into a SINGLE user turn,
    preserving strict alternation.
    """
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        call_idx = len(captured_requests)
        if call_idx == 1:
            # Emit parallel tool calls: quote and market cap
            return httpx.Response(
                200,
                json={
                    "id": "msg_par_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_p1", "name": "get_stock_quote", "input": {"ticker": "MSFT"}},
                        {"type": "tool_use", "id": "call_p2", "name": "get_market_cap", "input": {"ticker": "MSFT"}},
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 20, "output_tokens": 30},
                },
            )
        else:
            # Emit final answer
            return httpx.Response(
                200,
                json={
                    "id": "msg_par_2",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "MSFT is at $420 with a $3.1T market cap."},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 80, "output_tokens": 20},
                },
            )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_anthropic_alias()
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
                    "model": "foundry:claude-sonnet-5",
                    "messages": [{"role": "user", "content": "Analyze MSFT stock and market cap"}],
                    "tools": ALL_TOOLS,
                },
            )
            d1 = r1.json()
            tool_calls = d1["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 2

            # Turn 2: inject 2 consecutive role: "tool" messages
            messages_turn2 = [
                {"role": "user", "content": "Analyze MSFT stock and market cap"},
                {"role": "assistant", "content": None, "tool_calls": tool_calls},
                {"role": "tool", "tool_call_id": "call_p1", "content": '{"price": 420.0}'},
                {"role": "tool", "tool_call_id": "call_p2", "content": '{"market_cap": "3.1T"}'},
            ]
            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": messages_turn2,
                    "tools": ALL_TOOLS,
                },
            )
            assert r2.status_code == 200
            assert "3.1T" in r2.json()["choices"][0]["message"]["content"]

            # CRITICAL VERIFICATION: Upstream request 2 must have exactly 3 turns
            req2_msgs = captured_requests[1]["messages"]
            _assert_strict_alternation(req2_msgs)
            assert len(req2_msgs) == 3
            assert req2_msgs[0]["role"] == "user"
            assert req2_msgs[1]["role"] == "assistant"
            assert req2_msgs[2]["role"] == "user"

            # Check that Turn 2 user content contains BOTH tool_result blocks
            turn2_blocks = req2_msgs[2]["content"]
            assert len(turn2_blocks) == 2
            assert turn2_blocks[0]["type"] == "tool_result"
            assert turn2_blocks[0]["tool_use_id"] == "call_p1"
            assert turn2_blocks[1]["type"] == "tool_result"
            assert turn2_blocks[1]["tool_use_id"] == "call_p2"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Test 4: Anthropic Strict Turn Alternation Adversarial Matrix
# ==============================================================================


def test_anthropic_strict_alternation_adversarial_matrix():
    """Adversarial stress test on _append_anthropic_turn and _build_anthropic_request_body.

    Tests edge cases:
    a. Consecutive user messages [user, user, user]
    b. Consecutive assistant messages [asst_text, asst_tools]
    c. Consecutive assistant messages [asst_tools, asst_text]
    d. Interspersed multiple system messages
    e. Tool result followed by user message in same turn
    f. User message followed by tool result in same turn
    g. 4 consecutive tool messages with mixed is_error
    h. Empty messages list
    """
    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")

    # Case A: Consecutive user messages
    body_a = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[
            {"role": "user", "content": "Question part 1."},
            {"role": "user", "content": "Question part 2."},
            {"role": "user", "content": "Question part 3."},
        ],
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    _assert_strict_alternation(body_a["messages"])
    assert len(body_a["messages"]) == 1
    assert "Question part 1.\n\nQuestion part 2.\n\nQuestion part 3." == body_a["messages"][0]["content"]

    # Case B: Consecutive assistant messages (text then tools)
    body_b = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[
            {"role": "user", "content": "Run tools"},
            {"role": "assistant", "content": "I will execute the tool now."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f1", "arguments": "{}"}}],
            },
        ],
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    _assert_strict_alternation(body_b["messages"])
    assert len(body_b["messages"]) == 2
    assert body_b["messages"][0]["role"] == "user"
    assert body_b["messages"][1]["role"] == "assistant"
    asst_content = body_b["messages"][1]["content"]
    assert isinstance(asst_content, list)
    assert len(asst_content) == 2
    assert asst_content[0]["type"] == "text"
    assert asst_content[0]["text"] == "I will execute the tool now."
    assert asst_content[1]["type"] == "tool_use"
    assert asst_content[1]["id"] == "c1"

    # Case C: Consecutive assistant messages (tools then text)
    body_c = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[
            {"role": "user", "content": "Run tools"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f1", "arguments": "{}"}}],
            },
            {"role": "assistant", "content": "Tool dispatched."},
        ],
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    _assert_strict_alternation(body_c["messages"])
    assert len(body_c["messages"]) == 2
    asst_content_c = body_c["messages"][1]["content"]
    assert len(asst_content_c) == 2
    assert asst_content_c[0]["type"] == "tool_use"
    assert asst_content_c[1]["type"] == "text"
    assert asst_content_c[1]["text"] == "Tool dispatched."

    # Case D: Interspersed multiple system messages
    body_d = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[
            {"role": "system", "content": "System directive 1."},
            {"role": "user", "content": "User prompt 1."},
            {"role": "system", "content": "System directive 2."},
            {"role": "user", "content": "User prompt 2."},
            {"role": "assistant", "content": "Assistant reply 1."},
            {"role": "system", "content": "System directive 3."},
            {"role": "user", "content": "User prompt 3."},
        ],
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    assert body_d["system"] == "System directive 1.\n\nSystem directive 2.\n\nSystem directive 3."
    _assert_strict_alternation(body_d["messages"])
    assert len(body_d["messages"]) == 3
    assert body_d["messages"][0]["role"] == "user"
    assert body_d["messages"][0]["content"] == "User prompt 1.\n\nUser prompt 2."
    assert body_d["messages"][1]["role"] == "assistant"
    assert body_d["messages"][1]["content"] == "Assistant reply 1."
    assert body_d["messages"][2]["role"] == "user"
    assert body_d["messages"][2]["content"] == "User prompt 3."

    # Case E: Tool result followed by user message in same turn
    body_e = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[
            {"role": "user", "content": "Query"},
            {"role": "assistant", "tool_calls": [{"id": "call_x", "type": "function", "function": {"name": "fx", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_x", "content": "tool output"},
            {"role": "user", "content": "User extra instructions"},
        ],
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    _assert_strict_alternation(body_e["messages"])
    assert len(body_e["messages"]) == 3
    turn2_blocks_e = body_e["messages"][2]["content"]
    assert len(turn2_blocks_e) == 2
    assert turn2_blocks_e[0]["type"] == "tool_result"
    assert turn2_blocks_e[1]["type"] == "text"
    assert turn2_blocks_e[1]["text"] == "User extra instructions"

    # Case F: User message followed by tool result in same turn
    body_f = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[
            {"role": "user", "content": "Query"},
            {"role": "assistant", "tool_calls": [{"id": "call_y", "type": "function", "function": {"name": "fy", "arguments": "{}"}}]},
            {"role": "user", "content": "Here is additional context before result:"},
            {"role": "tool", "tool_call_id": "call_y", "content": "tool output y"},
        ],
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    _assert_strict_alternation(body_f["messages"])
    assert len(body_f["messages"]) == 3
    turn2_blocks_f = body_f["messages"][2]["content"]
    assert len(turn2_blocks_f) == 2
    assert turn2_blocks_f[0]["type"] == "text"
    assert turn2_blocks_f[0]["text"] == "Here is additional context before result:"
    assert turn2_blocks_f[1]["type"] == "tool_result"
    assert turn2_blocks_f[1]["tool_use_id"] == "call_y"

    # Case G: 4 consecutive tool messages with mixed is_error
    body_g = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[
            {"role": "user", "content": "Run 4 tools"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
                    {"id": "c3", "type": "function", "function": {"name": "t3", "arguments": "{}"}},
                    {"id": "c4", "type": "function", "function": {"name": "t4", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "OK 1"},
            {"role": "tool", "tool_call_id": "c2", "content": "Error in 2", "is_error": True},
            {"role": "tool", "tool_call_id": "c3", "content": "OK 3"},
            {"role": "tool", "tool_call_id": "c4", "content": "Timeout in 4", "is_error": True},
        ],
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    _assert_strict_alternation(body_g["messages"])
    assert len(body_g["messages"]) == 3
    res_blocks_g = body_g["messages"][2]["content"]
    assert len(res_blocks_g) == 4
    assert res_blocks_g[0]["content"] == "OK 1"
    assert "is_error" not in res_blocks_g[0]
    assert res_blocks_g[1]["content"] == "Error in 2"
    assert res_blocks_g[1]["is_error"] is True
    assert res_blocks_g[2]["content"] == "OK 3"
    assert "is_error" not in res_blocks_g[2]
    assert res_blocks_g[3]["content"] == "Timeout in 4"
    assert res_blocks_g[3]["is_error"] is True

    # Case H: Empty messages list
    body_h = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[],
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    _assert_strict_alternation(body_h["messages"])
    assert len(body_h["messages"]) == 1
    assert body_h["messages"][0]["role"] == "user"
    assert body_h["messages"][0]["content"] == ""


# ==============================================================================
# Test 5: Tool Result Content Serialization & Encoding Stress
# ==============================================================================


def test_hermes_anthropic_tool_result_content_types_stress():
    """Verify serialization of diverse data types in role: 'tool' content."""
    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")

    # Dict and List objects (client-side structured result before stringification)
    test_cases = [
        # Native Dict
        ({"status": "success", "count": 10}, '{"status": "success", "count": 10}'),
        # Native List
        ([1, "two", {"three": 3}], '[1, "two", {"three": 3}]'),
        # Korean & Emoji
        ("주가 분석 완료 🚀 최고가 달성!", "주가 분석 완료 🚀 최고가 달성!"),
        # Empty string
        ("", ""),
        # None
        (None, ""),
        # Large text (50KB)
        ("A" * 50000, "A" * 50000),
    ]

    for idx, (raw_val, expected_str) in enumerate(test_cases):
        body = client._build_anthropic_request_body(
            model="claude-sonnet-5",
            messages=[
                {"role": "user", "content": "ping"},
                {"role": "assistant", "tool_calls": [{"id": f"c_{idx}", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"c_{idx}", "content": raw_val},
            ],
            max_tokens=1000,
            stop=None,
            stream=False,
        )
        tool_block = body["messages"][1]["content"][0] if body["messages"][1]["role"] == "user" else body["messages"][2]["content"][0]
        assert tool_block["type"] == "tool_result"
        assert tool_block["content"] == expected_str


# ==============================================================================
# Test 6: Streaming Interleaved Text and Parallel Tool Use Blocks
# ==============================================================================


def test_hermes_anthropic_streaming_interleaved_text_and_parallel_tools():
    """Verify Anthropic SSE with text block followed by 2 parallel tool_use blocks.

    Ensures zero-based tool indexing (tool 0 and tool 1) even when text block 0 is present.
    """
    stream_payload = (
        b'data: {"type":"message_start","message":{"id":"msg_interleaved","type":"message","role":"assistant","content":[],"usage":{"input_tokens":30,"output_tokens":0}}}\n\n'
        # Block 0: Text explanation
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking both metrics now."}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        # Block 1: Tool call 1 (get_stock_quote)
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call_tok1","name":"get_stock_quote","input":{}}}\n\n'
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"ticker\\": \\"N"}}\n\n'
        # Block 2: Tool call 2 (get_market_cap) - Interleaved start!
        b'data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call_tok2","name":"get_market_cap","input":{}}}\n\n'
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"VDA\\"}"}}\n\n'
        b'data: {"type":"content_block_stop","index":1}\n\n'
        b'data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"ticker\\": \\"NVDA\\"}"}}\n\n'
        b'data: {"type":"content_block_stop","index":2}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":35}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_test():
        events = []
        async for ev in client.stream_chat(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Analyze NVDA"}],
            tools=ALL_TOOLS,
            resolved_config={"protocol": "anthropic_messages"},
        ):
            events.append(ev)
        return events

    events = asyncio.run(run_test())
    asyncio.run(client.close())

    # Verify text delta was emitted
    text_events = [e for e in events if e.get("delta_text")]
    assert len(text_events) >= 1
    assert "Checking both metrics" in text_events[0]["delta_text"]

    # Verify tool call 0 (block index 1 mapped to tool_idx 0)
    tc0_events = [
        e["delta_tool_calls"][0]
        for e in events
        if e.get("delta_tool_calls") and e["delta_tool_calls"][0]["index"] == 0
    ]
    assert tc0_events[0]["id"] == "call_tok1"
    assert tc0_events[0]["function"]["name"] == "get_stock_quote"
    tc0_args = "".join(tc["function"].get("arguments", "") for tc in tc0_events)
    assert json.loads(tc0_args) == {"ticker": "NVDA"}

    # Verify tool call 1 (block index 2 mapped to tool_idx 1)
    tc1_events = [
        e["delta_tool_calls"][0]
        for e in events
        if e.get("delta_tool_calls") and e["delta_tool_calls"][0]["index"] == 1
    ]
    assert tc1_events[0]["id"] == "call_tok2"
    assert tc1_events[0]["function"]["name"] == "get_market_cap"
    tc1_args = "".join(tc["function"].get("arguments", "") for tc in tc1_events)
    assert json.loads(tc1_args) == {"ticker": "NVDA"}

    # Verify terminal finish_reason
    finish_events = [e for e in events if e.get("finish_reason")]
    assert finish_events[-1]["finish_reason"] == "tool_calls"


# ==============================================================================
# Test 7: Tool Choice Variants Full Matrix
# ==============================================================================


def test_hermes_anthropic_tool_choice_full_matrix():
    """Verify tool_choice upstream mapping: 'auto', 'none', 'required', and named forms."""
    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")

    # 1. "auto" -> {"type": "auto"}
    b1 = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "test"}],
        tools=ALL_TOOLS,
        tool_choice="auto",
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    assert b1["tool_choice"] == {"type": "auto"}
    assert len(b1["tools"]) == 3

    # 2. "none" -> tools and tool_choice omitted
    b2 = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "test"}],
        tools=ALL_TOOLS,
        tool_choice="none",
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    assert "tools" not in b2
    assert "tool_choice" not in b2

    # 3. "required" -> {"type": "any"}
    b3 = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "test"}],
        tools=ALL_TOOLS,
        tool_choice="required",
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    assert b3["tool_choice"] == {"type": "any"}

    # 4. Named function OpenAI format: {"type": "function", "function": {"name": "calculate_pe"}}
    b4 = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "test"}],
        tools=ALL_TOOLS,
        tool_choice={"type": "function", "function": {"name": "calculate_pe"}},
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    assert b4["tool_choice"] == {"type": "tool", "name": "calculate_pe"}

    # 5. Named tool Anthropic format: {"type": "tool", "name": "calculate_pe"}
    b5 = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "test"}],
        tools=ALL_TOOLS,
        tool_choice={"type": "tool", "name": "calculate_pe"},
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    assert b5["tool_choice"] == {"type": "tool", "name": "calculate_pe"}

    # 6. Simple name dict: {"name": "calculate_pe"}
    b6 = client._build_anthropic_request_body(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "test"}],
        tools=ALL_TOOLS,
        tool_choice={"name": "calculate_pe"},
        max_tokens=1000,
        stop=None,
        stream=False,
    )
    assert b6["tool_choice"] == {"type": "tool", "name": "calculate_pe"}


# ==============================================================================
# Test 8: Upstream Error Propagation & HTTP Status Handling
# ==============================================================================


def test_hermes_anthropic_upstream_error_propagation():
    """Verify upstream Anthropic error payloads (400, 401, 429, 500) and streaming errors."""
    error_cases = [
        (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "Field 'tools' is invalid"}}, 400),
        (401, {"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}}, 401),
        (429, {"type": "error", "error": {"type": "rate_limit_error", "message": "Tokens per minute exceeded"}}, 429),
        (500, {"type": "error", "error": {"type": "api_error", "message": "Anthropic internal error"}}, 500),
    ]

    for status_code, err_body, expected_status in error_cases:
        async def mock_err_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, json=err_body)

        client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_err_handler))

        with pytest.raises(VertexAPIError) as exc_info:
            asyncio.run(
                client.generate(
                    model="claude-sonnet-5",
                    messages=[{"role": "user", "content": "trigger"}],
                    resolved_config={"protocol": "anthropic_messages"},
                )
            )
        assert exc_info.value.status_code == expected_status
        assert err_body["error"]["message"] in exc_info.value.message
        asyncio.run(client.close())

    # Streaming mid-flight error event
    stream_err_payload = (
        b'data: {"type":"message_start","message":{"id":"msg_err","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
        b'data: {"type":"error","error":{"type":"overloaded_error","message":"Anthropic servers are overloaded"}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_stream_err_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_err_payload)

    client2 = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
    client2.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_stream_err_handler))

    with pytest.raises(VertexAPIError) as exc_info:
        async def run_stream_err():
            async for _ in client2.stream_chat(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "stream err"}],
                resolved_config={"protocol": "anthropic_messages"},
            ):
                pass

        asyncio.run(run_stream_err())
    assert exc_info.value.status_code == 502
    assert "overloaded" in exc_info.value.message
    asyncio.run(client2.close())
