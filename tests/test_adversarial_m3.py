"""Adversarial stress-testing suite for Foundry Anthropic protocol (Milestone 3 Challenger).

Empirically tests edge cases and failure modes:
1. Sequential 3+ `role: "tool"` messages merging into a single `user` turn with strict alternation.
2. `tool_result` content encoding and serialization safety (Korean, emojis, nested dict/list, raw JSON, huge text, empty).
3. Anthropic SSE stream interleaving of text and tool_use blocks with 0-based tool call index consistency.
4. Exhaustive `tool_choice` variants: "none" (complete omission), "required" ("any"), and various named dict shapes.
5. Error handling and robust fallbacks for upstream failures and malformed payloads.
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


# ---------------------------------------------------------------------------
# Test 1: 3+ sequential tool messages merge into single user turn
# ---------------------------------------------------------------------------


def test_adversarial_anthropic_multi_tool_merge_sequential_3_plus():
    """Verify that 3, 4, and 5 sequential tool messages merge into a single user turn."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "msg_m3_test1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "All tools processed successfully."}],
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": 15},
            },
        )

    client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    # Test with 4 sequential tool results
    messages = [
        {"role": "user", "content": "Fetch data from multiple sources"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "fn1", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "fn2", "arguments": "{}"}},
                {"id": "call_3", "type": "function", "function": {"name": "fn3", "arguments": "{}"}},
                {"id": "call_4", "type": "function", "function": {"name": "fn4", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Result 1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "Result 2"},
        {"role": "tool", "tool_call_id": "call_3", "content": "Result 3", "is_error": True},
        {"role": "tool", "tool_call_id": "call_4", "content": "Result 4"},
    ]

    res = asyncio.run(
        client.generate(
            model="claude-sonnet-5",
            messages=messages,
            resolved_config={"protocol": "anthropic_messages"},
        )
    )
    assert res["text"] == "All tools processed successfully."
    assert res["finish_reason"] == "stop"

    assert len(captured_requests) == 1
    sent_messages = captured_requests[0]["messages"]

    # Verify strict turn alternation: User -> Assistant -> User (exactly 3 turns, NOT 7 turns!)
    assert len(sent_messages) == 3, f"Expected 3 alternating turns, got {len(sent_messages)}"
    assert sent_messages[0]["role"] == "user"
    assert sent_messages[1]["role"] == "assistant"
    assert sent_messages[2]["role"] == "user"

    # Verify assistant turn contains 4 tool_use blocks
    assistant_blocks = sent_messages[1]["content"]
    assert len(assistant_blocks) == 4
    assert [b["id"] for b in assistant_blocks] == ["call_1", "call_2", "call_3", "call_4"]

    # Verify merged user turn contains exactly 4 tool_result blocks
    merged_user_content = sent_messages[2]["content"]
    assert isinstance(merged_user_content, list)
    assert len(merged_user_content) == 4
    for idx, block in enumerate(merged_user_content, 1):
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == f"call_{idx}"
        assert block["content"] == f"Result {idx}"
        if idx == 3:
            assert block["is_error"] is True
        else:
            assert "is_error" not in block or block["is_error"] is False

    # Now append an extra user prompt after the tool results:
    # [user, assistant, tool1, tool2, tool3, tool4, user("Synthesize now")]
    captured_requests.clear()
    messages_with_followup = list(messages) + [
        {"role": "user", "content": "Please synthesize everything now."}
    ]

    res2 = asyncio.run(
        client.generate(
            model="claude-sonnet-5",
            messages=messages_with_followup,
            resolved_config={"protocol": "anthropic_messages"},
        )
    )
    assert res2["text"] == "All tools processed successfully."
    assert len(captured_requests) == 1
    sent_messages2 = captured_requests[0]["messages"]

    # Still exactly 3 turns, because user prompt is merged into the same user turn!
    assert len(sent_messages2) == 3
    final_turn_blocks = sent_messages2[2]["content"]
    assert len(final_turn_blocks) == 5  # 4 tool_results + 1 text block
    assert final_turn_blocks[-1] == {"type": "text", "text": "Please synthesize everything now."}


# ---------------------------------------------------------------------------
# Test 2: tool_result encoding and serialization safety
# ---------------------------------------------------------------------------


def test_adversarial_anthropic_tool_result_content_encoding_and_serialization():
    """Verify tool_result serialization for Korean, emojis, nested dict/list, raw JSON, huge text, and None."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "msg_m3_test2",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Encoding test passed"}],
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        )

    client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    korean_text = "서울 강남구 역삼동 현재 날씨: 맑음, 기온 23.5℃, 습도 55%"
    emoji_text = "🌤️🌡️🎉🚀✨🔥 [OK] 정상 완료!"
    nested_data = {
        "status": "success",
        "records": [
            {"id": 101, "name": "홍길동", "active": True, "scores": [95, 88.5, 100]},
            {"id": 102, "name": "Jane Doe", "tags": ["admin", "developer", "<safe>"]},
        ],
        "meta": {"timestamp": "2026-09-19T00:00:00Z", "null_field": None},
    }
    special_json_str = '{"raw_key": "contains \\"quotes\\" and \\nnewlines\\t and <xml>tags</xml>"}'
    huge_text = "대용량 한글 텍스트 블록 " * 5000  # ~125KB text

    messages = [
        {"role": "user", "content": "Execute multiple tools with various result types"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_kor", "type": "function", "function": {"name": "fn_kor", "arguments": "{}"}},
                {"id": "call_emo", "type": "function", "function": {"name": "fn_emo", "arguments": "{}"}},
                {"id": "call_nest", "type": "function", "function": {"name": "fn_nest", "arguments": "{}"}},
                {"id": "call_spec", "type": "function", "function": {"name": "fn_spec", "arguments": "{}"}},
                {"id": "call_huge", "type": "function", "function": {"name": "fn_huge", "arguments": "{}"}},
                {"id": "call_empty", "type": "function", "function": {"name": "fn_empty", "arguments": "{}"}},
                {"id": "call_none", "type": "function", "function": {"name": "fn_none", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_kor", "content": korean_text},
        {"role": "tool", "tool_call_id": "call_emo", "content": emoji_text},
        {"role": "tool", "tool_call_id": "call_nest", "content": nested_data},  # dict passed directly
        {"role": "tool", "tool_call_id": "call_spec", "content": special_json_str},
        {"role": "tool", "tool_call_id": "call_huge", "content": huge_text},
        {"role": "tool", "tool_call_id": "call_empty", "content": ""},
        {"role": "tool", "tool_call_id": "call_none", "content": None},
    ]

    res = asyncio.run(
        client.generate(
            model="claude-sonnet-5",
            messages=messages,
            resolved_config={"protocol": "anthropic_messages"},
        )
    )
    assert res["text"] == "Encoding test passed"

    sent_messages = captured_requests[0]["messages"]
    user_blocks = sent_messages[-1]["content"]
    assert len(user_blocks) == 7

    # 1. Korean text preservation
    assert user_blocks[0]["content"] == korean_text
    assert "서울" in user_blocks[0]["content"]

    # 2. Emoji text preservation
    assert user_blocks[1]["content"] == emoji_text
    assert "🌤️" in user_blocks[1]["content"]

    # 3. Nested dict serialized with ensure_ascii=False (Korean characters unescaped)
    serialized_nest = user_blocks[2]["content"]
    assert "홍길동" in serialized_nest
    deserialized = json.loads(serialized_nest)
    assert deserialized["records"][0]["name"] == "홍길동"
    assert deserialized["records"][0]["scores"] == [95, 88.5, 100]

    # 4. Special JSON string
    assert user_blocks[3]["content"] == special_json_str

    # 5. Huge text
    assert len(user_blocks[4]["content"]) == len(huge_text)

    # 6. Empty string
    assert user_blocks[5]["content"] == ""

    # 7. None value becomes empty string
    assert user_blocks[6]["content"] == ""


# ---------------------------------------------------------------------------
# Test 3: SSE stream interleaving and 0-based tool call indexing
# ---------------------------------------------------------------------------


def test_adversarial_anthropic_sse_interleaved_blocks_and_zero_based_tool_indices():
    """Verify that interleaved text and tool_use blocks produce consistent 0-based OpenAI tool call indices."""
    sse_events = [
        {"event": "message_start", "data": {"type": "message_start", "message": {"id": "msg_stream_adv", "type": "message", "role": "assistant", "model": "claude-sonnet-5", "usage": {"input_tokens": 40, "output_tokens": 1}}}},
        # Block 0: Text preamble
        {"event": "content_block_start", "data": {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}},
        {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "I will execute three tools: "}}},
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
        # Block 1: Tool 1 (should become OpenAI tool_call index 0)
        {"event": "content_block_start", "data": {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tool_call_101", "name": "search_db", "input": {}}}},
        {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"query": '}}},
        # Block 2: Interleaved Text (thought / chain-of-thought)
        {"event": "content_block_start", "data": {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}}},
        {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Next checking weather: "}}},
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 2}},
        # Block 3: Tool 2 (should become OpenAI tool_call index 1)
        {"event": "content_block_start", "data": {"type": "content_block_start", "index": 3, "content_block": {"type": "tool_use", "id": "tool_call_102", "name": "get_weather", "input": {}}}},
        {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 3, "delta": {"type": "input_json_delta", "partial_json": '{"city": "Seoul"'}}},
        # Interleaved JSON deltas for Tool 1 and Tool 2
        {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"database_query"}'}}},
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 1}},
        {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 3, "delta": {"type": "input_json_delta", "partial_json": ', "unit": "metric"}'}}},
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 3}},
        # Block 4: Tool 3 (should become OpenAI tool_call index 2)
        {"event": "content_block_start", "data": {"type": "content_block_start", "index": 4, "content_block": {"type": "tool_use", "id": "tool_call_103", "name": "send_notification", "input": {}}}},
        {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 4, "delta": {"type": "input_json_delta", "partial_json": '{"msg": "done"}'}}},
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 4}},
        # Finish
        {"event": "message_delta", "data": {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 60}}},
        {"event": "message_stop", "data": {"type": "message_stop"}},
    ]

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        lines = []
        for ev in sse_events:
            lines.append(f"event: {ev['event']}\n")
            lines.append(f"data: {json.dumps(ev['data'])}\n\n")
        body = "".join(lines).encode("utf-8")
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_stream():
        collected_text = ""
        tool_calls_assembled: dict[int, dict[str, Any]] = {}
        final_finish_reason = None
        final_usage = None

        async for chunk in client.stream_chat(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Execute multi tools"}],
            resolved_config={"protocol": "anthropic_messages"},
        ):
            if "delta_text" in chunk:
                collected_text += chunk["delta_text"]
            if "delta_tool_calls" in chunk:
                for tc_delta in chunk["delta_tool_calls"]:
                    idx = tc_delta["index"]
                    if idx not in tool_calls_assembled:
                        tool_calls_assembled[idx] = {
                            "id": tc_delta.get("id"),
                            "name": tc_delta.get("function", {}).get("name"),
                            "arguments": "",
                        }
                    raw_args = tc_delta.get("function", {}).get("arguments", "")
                    tool_calls_assembled[idx]["arguments"] += raw_args
            if "finish_reason" in chunk:
                final_finish_reason = chunk["finish_reason"]
            if "usage" in chunk:
                final_usage = chunk["usage"]

        return collected_text, tool_calls_assembled, final_finish_reason, final_usage

    text, tool_calls, finish_reason, usage = asyncio.run(run_stream())

    # Verify text deltas from block 0 and block 2 were aggregated
    assert text == "I will execute three tools: Next checking weather: "

    # Verify tool call indices are exactly [0, 1, 2] with no gaps
    assert sorted(tool_calls.keys()) == [0, 1, 2]

    # Tool 0: search_db
    assert tool_calls[0]["id"] == "tool_call_101"
    assert tool_calls[0]["name"] == "search_db"
    assert json.loads(tool_calls[0]["arguments"]) == {"query": "database_query"}

    # Tool 1: get_weather
    assert tool_calls[1]["id"] == "tool_call_102"
    assert tool_calls[1]["name"] == "get_weather"
    assert json.loads(tool_calls[1]["arguments"]) == {"city": "Seoul", "unit": "metric"}

    # Tool 2: send_notification
    assert tool_calls[2]["id"] == "tool_call_103"
    assert tool_calls[2]["name"] == "send_notification"
    assert json.loads(tool_calls[2]["arguments"]) == {"msg": "done"}

    assert finish_reason == "tool_calls"
    assert usage == {"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100}


# ---------------------------------------------------------------------------
# Test 4: Exhaustive tool_choice variants and safety
# ---------------------------------------------------------------------------


def test_adversarial_anthropic_tool_choice_variants_exhaustive():
    """Verify all tool_choice variants: 'none', 'required', named dict shapes, and 'auto'."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "msg_choice",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "choice handled"}],
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 5},
            },
        )

    client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    sample_tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Calculate expression",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup entity",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
            },
        },
    ]

    test_matrix = [
        # 1. tool_choice="none" -> tools MUST be omitted entirely from Anthropic body
        ("none", None, False, False),
        # 2. tool_choice="required" -> Anthropic {"type": "any"}
        ("required", {"type": "any"}, True, True),
        # 3. tool_choice="auto" -> Anthropic {"type": "auto"}
        ("auto", {"type": "auto"}, True, True),
        # 4. tool_choice named (OpenAI standard shape)
        ({"type": "function", "function": {"name": "calc"}}, {"type": "tool", "name": "calc"}, True, True),
        # 5. tool_choice named (flat shape)
        ({"type": "tool", "name": "lookup"}, {"type": "tool", "name": "lookup"}, True, True),
        # 6. tool_choice named (minimal name dict)
        ({"name": "calc"}, {"type": "tool", "name": "calc"}, True, True),
        # 7. tool_choice=None -> tools present, tool_choice omitted (default auto)
        (None, None, True, False),
    ]

    for tc_input, expected_tc, expect_tools, expect_tc_field in test_matrix:
        captured_requests.clear()
        res = asyncio.run(
            client.generate(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "hi"}],
                tools=sample_tools,
                tool_choice=tc_input,
                resolved_config={"protocol": "anthropic_messages"},
            )
        )
        assert res["text"] == "choice handled"
        assert len(captured_requests) == 1
        req = captured_requests[0]

        if expect_tools:
            assert "tools" in req
            assert len(req["tools"]) == 2
            assert req["tools"][0]["name"] == "calc"
            assert req["tools"][0]["input_schema"] == {"type": "object", "properties": {"expr": {"type": "string"}}}
        else:
            assert "tools" not in req, f"Expected tools to be omitted for tool_choice={tc_input}"

        if expect_tc_field:
            assert "tool_choice" in req
            assert req["tool_choice"] == expected_tc
        else:
            assert "tool_choice" not in req


# ---------------------------------------------------------------------------
# Test 5: FastAPI E2E testing for all edge cases
# ---------------------------------------------------------------------------


def test_adversarial_anthropic_fastapi_e2e_full_cycle():
    """End-to-end test via FastAPI TestClient validating multi-turn tool calling and streaming."""
    old_registry = _register_foundry_anthropic_alias()
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "msg_e2e",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Calculated value is 42."},
                    {"type": "tool_use", "id": "call_e2e_next", "name": "log_result", "input": {"val": 42}},
                ],
                "model": "claude-sonnet-5",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 150, "output_tokens": 25},
            },
        )

    foundry_client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    foundry_client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    app = create_app(
        embedding_client_factory=_DummyProvider,
        chat_client_factory=_DummyProvider,
        rerank_client_factory=_DummyProvider,
        ollama_chat_client_factory=_DummyProvider,
        foundry_chat_client_factory=lambda: foundry_client,
        cost_accounting_factory=lambda: None,
    )

    try:
        with TestClient(app) as test_client:
            # Non-streaming multi-tool history request with tool_choice="required"
            resp = test_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": [
                        {"role": "user", "content": "Compute 6 * 7"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": '{"a": 6, "b": 7}'}},
                                {"id": "c2", "type": "function", "function": {"name": "verify", "arguments": '{"expected": 42}'}},
                            ],
                        },
                        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"result": 42, "status": "ok"})},
                        {"role": "tool", "tool_call_id": "c2", "content": "verified successfully"},
                    ],
                    "tools": [
                        {"type": "function", "function": {"name": "log_result", "parameters": {"type": "object"}}}
                    ],
                    "tool_choice": "required",
                },
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["choices"][0]["finish_reason"] == "tool_calls"
            # Assistant returned both text and a tool call
            assert data["choices"][0]["message"]["content"] == "Calculated value is 42."
            assert len(data["choices"][0]["message"]["tool_calls"]) == 1
            assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "log_result"
            assert json.loads(data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {"val": 42}

            # Verify upstream request got properly merged into 3 turns
            assert len(captured_requests) == 1
            upstream_msgs = captured_requests[0]["messages"]
            assert len(upstream_msgs) == 3
            assert captured_requests[0]["tool_choice"] == {"type": "any"}
            assert len(upstream_msgs[2]["content"]) == 2
            assert upstream_msgs[2]["content"][0]["type"] == "tool_result"
            assert json.loads(upstream_msgs[2]["content"][0]["content"]) == {"result": 42, "status": "ok"}
            assert upstream_msgs[2]["content"][1]["content"] == "verified successfully"

    finally:
        _restore_registry(old_registry)


# ---------------------------------------------------------------------------
# Test 6: Upstream error propagation and edge inputs
# ---------------------------------------------------------------------------


def test_adversarial_anthropic_error_propagation_and_edge_inputs():
    """Verify upstream Anthropic 400/429/500 and malformed responses are translated cleanly."""
    error_cases = [
        (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens too large"}}, 400),
        (429, {"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}}, 429),
        (500, {"type": "error", "error": {"type": "api_error", "message": "Anthropic internal error"}}, 500),
    ]

    for status_code, err_body, expected_bridge_status in error_cases:
        async def mock_err_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, json=err_body)

        client = FoundryChatClient(
            base_url=FOUNDRY_TEST_BASE_URL,
            token="test-token",
        )
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_err_handler))

        with pytest.raises(VertexAPIError) as exc_info:
            asyncio.run(
                client.generate(
                    model="claude-sonnet-5",
                    messages=[{"role": "user", "content": "trigger error"}],
                    resolved_config={"protocol": "anthropic_messages"},
                )
            )
        assert exc_info.value.status_code == expected_bridge_status
        assert err_body["error"]["message"] in exc_info.value.message


def test_adversarial_anthropic_malformed_response_missing_content():
    """Verify VertexAPIError 502 when Anthropic returns invalid structure without content[]."""
    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_bad", "type": "message", "role": "assistant"})

    client = FoundryChatClient(
        base_url=FOUNDRY_TEST_BASE_URL,
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    with pytest.raises(VertexAPIError) as exc_info:
        asyncio.run(
            client.generate(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "test"}],
                resolved_config={"protocol": "anthropic_messages"},
            )
        )
    assert exc_info.value.status_code == 502
    assert "missing content[]" in exc_info.value.message


def test_adversarial_anthropic_sse_stream_error_event():
    """Verify that an error event in Anthropic SSE stream raises VertexAPIError with 502."""
    sse_events = [
        {"event": "message_start", "data": {"type": "message_start", "message": {"id": "msg_err", "type": "message", "role": "assistant"}}},
        {"event": "error", "data": {"type": "error", "error": {"type": "overloaded_error", "message": "Anthropic is overloaded."}}},
    ]

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        lines = [f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n" for ev in sse_events]
        return httpx.Response(200, content="".join(lines).encode("utf-8"), headers={"content-type": "text/event-stream"})

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def consume_stream():
        async for _ in client.stream_chat(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hello"}],
            resolved_config={"protocol": "anthropic_messages"},
        ):
            pass

    with pytest.raises(VertexAPIError) as exc_info:
        asyncio.run(consume_stream())

    assert exc_info.value.status_code == 502
    assert "Anthropic is overloaded." in exc_info.value.message
    assert exc_info.value.code == "overloaded_error"


def test_adversarial_anthropic_sse_stream_max_tokens_length_finish_reason():
    """Verify that stop_reason='max_tokens' in stream maps to finish_reason='length'."""
    sse_events = [
        {"event": "message_start", "data": {"type": "message_start", "message": {"id": "msg_len", "type": "message", "role": "assistant", "usage": {"input_tokens": 10}}}},
        {"event": "content_block_start", "data": {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}},
        {"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Truncated..."}}},
        {"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
        {"event": "message_delta", "data": {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 100}}},
        {"event": "message_stop", "data": {"type": "message_stop"}},
    ]

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        lines = [f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n" for ev in sse_events]
        return httpx.Response(200, content="".join(lines).encode("utf-8"), headers={"content-type": "text/event-stream"})

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def consume_stream():
        reasons = []
        async for chunk in client.stream_chat(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hello"}],
            resolved_config={"protocol": "anthropic_messages"},
        ):
            if "finish_reason" in chunk:
                reasons.append(chunk["finish_reason"])
        return reasons

    finish_reasons = asyncio.run(consume_stream())
    assert "length" in finish_reasons


def test_adversarial_anthropic_complex_multi_turn_multi_cycle_history():
    """Verify a multi-turn conversation with two tool-calling rounds strictly alternates turns."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "msg_cycle",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Final synthesis complete."}],
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 20},
            },
        )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    # Round 1: User -> Assistant (2 calls) -> 2 Tool results
    # Round 2: Assistant (3 calls) -> 3 Tool results
    # Final: User prompt -> Next Assistant turn
    messages = [
        {"role": "user", "content": "Round 1 question"},
        {
            "role": "assistant",
            "content": "Let me call tools in round 1",
            "tool_calls": [
                {"id": "r1_c1", "type": "function", "function": {"name": "fetch_a", "arguments": "{}"}},
                {"id": "r1_c2", "type": "function", "function": {"name": "fetch_b", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "r1_c1", "content": "Data A"},
        {"role": "tool", "tool_call_id": "r1_c2", "content": "Data B"},
        {
            "role": "assistant",
            "content": "Now round 2 tools",
            "tool_calls": [
                {"id": "r2_c1", "type": "function", "function": {"name": "fetch_c", "arguments": "{}"}},
                {"id": "r2_c2", "type": "function", "function": {"name": "fetch_d", "arguments": "{}"}},
                {"id": "r2_c3", "type": "function", "function": {"name": "fetch_e", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "r2_c1", "content": "Data C"},
        {"role": "tool", "tool_call_id": "r2_c2", "content": "Data D"},
        {"role": "tool", "tool_call_id": "r2_c3", "content": "Data E"},
        {"role": "user", "content": "Now summarize both rounds."},
    ]

    res = asyncio.run(
        client.generate(
            model="claude-sonnet-5",
            messages=messages,
            resolved_config={"protocol": "anthropic_messages"},
        )
    )
    assert res["text"] == "Final synthesis complete."
    assert len(captured_requests) == 1

    sent_messages = captured_requests[0]["messages"]
    # Turn sequence must be: [user, assistant, user, assistant, user] (exactly 5 turns!)
    roles = [m["role"] for m in sent_messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"], f"Unexpected roles: {roles}"

    # Verify each turn's content structure
    # Turn 0: user text
    assert sent_messages[0]["content"] == "Round 1 question"

    # Turn 1: assistant text + 2 tool_use blocks
    assert sent_messages[1]["content"][0] == {"type": "text", "text": "Let me call tools in round 1"}
    assert [b["id"] for b in sent_messages[1]["content"][1:]] == ["r1_c1", "r1_c2"]

    # Turn 2: user 2 tool_result blocks
    assert len(sent_messages[2]["content"]) == 2
    assert [b["tool_use_id"] for b in sent_messages[2]["content"]] == ["r1_c1", "r1_c2"]

    # Turn 3: assistant text + 3 tool_use blocks
    assert sent_messages[3]["content"][0] == {"type": "text", "text": "Now round 2 tools"}
    assert [b["id"] for b in sent_messages[3]["content"][1:]] == ["r2_c1", "r2_c2", "r2_c3"]

    # Turn 4: user 3 tool_result blocks + 1 text prompt
    assert len(sent_messages[4]["content"]) == 4
    assert [b["tool_use_id"] for b in sent_messages[4]["content"][:3]] == ["r2_c1", "r2_c2", "r2_c3"]
    assert sent_messages[4]["content"][3] == {"type": "text", "text": "Now summarize both rounds."}


def test_adversarial_anthropic_tool_definition_fallbacks_and_empty_parameters():
    """Verify tool definition fallbacks when parameters is None, empty dict, or missing."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "msg_tool_def",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Tools received"}],
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    tools_with_fallbacks = [
        # Tool 1: parameters is None
        {"type": "function", "function": {"name": "no_params", "parameters": None}},
        # Tool 2: parameters key completely missing
        {"type": "function", "function": {"name": "missing_params", "description": "desc"}},
        # Tool 3: flat tool format (without function wrapper)
        {"name": "flat_tool", "description": "flat", "parameters": {"type": "object"}},
    ]

    res = asyncio.run(
        client.generate(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools_with_fallbacks,
            resolved_config={"protocol": "anthropic_messages"},
        )
    )
    assert res["text"] == "Tools received"

    req_tools = captured_requests[0]["tools"]
    assert len(req_tools) == 3

    # Fallback schema must be {"type": "object", "properties": {}}
    assert req_tools[0]["name"] == "no_params"
    assert req_tools[0]["input_schema"] == {"type": "object", "properties": {}}

    assert req_tools[1]["name"] == "missing_params"
    assert req_tools[1]["input_schema"] == {"type": "object", "properties": {}}

    assert req_tools[2]["name"] == "flat_tool"
    assert req_tools[2]["input_schema"] == {"type": "object"}

