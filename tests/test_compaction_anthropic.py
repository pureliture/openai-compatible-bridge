"""Native Anthropic HTTP fixtures; no paid provider or real local LFM calls."""
from __future__ import annotations

import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.context_compaction import AFFINITY_HEADER, HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from test_compaction_foundry_protocols import ARGS, MESSAGES, SOURCE, SUMMARY, TOOL, SyntheticLFM, Unused

ALIAS = "foundry:synthetic-anthropic"
AFFINITY = "synthetic-anthropic-conversation"


def native_call(name, args, call_id="private-call"):
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


def native_response(*calls, text=None, usage=None):
    content = list(calls)
    if text is not None:
        content.append({"type": "text", "text": text})
    return {"id": "msg-synthetic", "type": "message", "role": "assistant", "content": content,
            "stop_reason": "tool_use" if calls else "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2} if usage is None else usage}


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_LFM_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_LAYA_ENABLED", "false")
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", "")
    monkeypatch.setitem(vertex.MODEL_REGISTRY, ALIAS, {"provider": "foundry", "kind": "chat",
                        "provider_model": "synthetic-anthropic", "protocol": "anthropic_messages"})
    bodies, replies = [], []

    async def handler(request):
        assert request.url.path == "/api/v2/llm/proxy/anthropic/v1/messages"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body["stream"] is False
        bodies.append(body)
        assert replies, "unexpected native HTTP request"
        reply = replies.pop(0)
        return httpx.Response(200, json=reply(body) if callable(reply) else reply)

    provider = FoundryChatClient(base_url="https://foundry.example/api/v2/llm/proxy/openai/v1/chat/completions", token="synthetic")
    asyncio.run(provider.http.aclose())
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    lfm = SyntheticLFM()
    app = create_app(embedding_client_factory=Unused, chat_client_factory=Unused,
                     rerank_client_factory=Unused, ollama_chat_client_factory=lambda: lfm,
                     foundry_chat_client_factory=lambda: provider, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        yield client, app, bodies, replies, lfm
    assert not replies, "not all scripted native replies consumed"


def post(client, *, messages=None, headers=None, **kwargs):
    return client.post("/v1/chat/completions", headers={AFFINITY_HEADER: AFFINITY} if headers is None else headers,
                       json={"model": ALIAS, "messages": copy.deepcopy(MESSAGES if messages is None else messages),
                             "tools": [TOOL], **kwargs})


def blocks(body, kind):
    return [block for message in body["messages"] if isinstance(message["content"], list)
            for block in message["content"] if block.get("type") == kind]


def output(body, call_id="original-call"):
    return next(block["content"] for block in blocks(body, "tool_result") if block["tool_use_id"] == call_id)


def private_output(body, call_id="private-call"):
    return json.loads(output(body, call_id))


def assert_original_call(body):
    assert native_call("terminal", json.loads(ARGS), "original-call") in blocks(body, "tool_use")
    assert body["messages"][0] == MESSAGES[0]


def test_anthropic_native_hide_roundtrip_is_private_and_result_only(bridge):
    client, app, bodies, replies, lfm = bridge
    original = copy.deepcopy(MESSAGES)
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})),
                    native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "finished"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in data["choices"][0]["message"]
    assert all(name not in response.text for name in (HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL, "private-call"))
    assert data["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    assert len(bodies) == 2
    assert {tool["name"] for tool in bodies[0]["tools"]} == {"terminal", HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}
    assert bodies[0]["model"] == "synthetic-anthropic"
    assert bodies[0]["tools"][0]["input_schema"] == TOOL["function"]["parameters"]
    assert output(bodies[0]) == SOURCE
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    assert item.original == SOURCE and item.compaction_source == "lfm"
    assert SUMMARY in item.compacted and output(bodies[1]) == item.compacted != SOURCE
    for body in bodies:
        assert_original_call(body)
    assert native_call(HIDE_TOOL, {"tool_call_id": "original-call"}) in blocks(bodies[1], "tool_use")
    assert bodies[1]["messages"][-1]["role"] == "user"
    assert private_output(bodies[1])["ok"] is True
    assert MESSAGES == original
    assert len(lfm.calls) == 1
    packet = json.loads(lfm.calls[0]["messages"][1]["content"].split("\n\n", 1)[1])
    assert set(packet) == {"result", "required_evidence"}
    assert ARGS not in json.dumps(lfm.calls) and MESSAGES[0]["content"] not in json.dumps(lfm.calls)
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.lfm_applied and measurement.lfm_calls == 1
    assert measurement.provider_calls == 2 and measurement.lfm_fallback_reason is None


def test_anthropic_native_cache_usage_is_aggregated_across_private_continuation(bridge):
    client, app, bodies, replies, lfm = bridge
    replies.extend([
        native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"}), usage={
            "input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 4}),
        native_response(text="finished", usage={
            "input_tokens": 20, "output_tokens": 5, "cache_read_input_tokens": 6, "cache_creation_input_tokens": 8}),
    ])
    response = post(client)
    assert response.status_code == 200
    assert response.json()["usage"] == {
        "prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37,
        "cache_read_input_tokens": 9, "cache_creation_input_tokens": 12,
    }
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.provider_calls == 2
    assert measurement.cache_read_tokens == 9 and measurement.cache_write_tokens == 12


def test_anthropic_after_list_exact_unhide_and_affinity_isolation(bridge):
    client, app, bodies, replies, lfm = bridge
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="hidden")])
    assert post(client).status_code == 200
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    replies.append(native_response(text="after"))
    assert post(client).status_code == 200
    assert output(bodies[-1]) == item.compacted
    for policy in ({"tools": None}, {"tool_choice": "none"}):
        replies.append(native_response(text="visibility only"))
        assert post(client, **policy).status_code == 200
        assert output(bodies[-1]) == item.compacted
        assert not bodies[-1].get("tools")
    replies.append(native_response(text="isolated"))
    assert post(client, headers={AFFINITY_HEADER: "other"}).status_code == 200
    assert output(bodies[-1]) == SOURCE
    replies.extend([native_response(native_call(UNHIDE_TOOL, {"item_id": item.item_id}, "cross-unhide")), native_response(text="isolated restore")])
    assert post(client, headers={AFFINITY_HEADER: "other"}).status_code == 200
    assert private_output(bodies[-1], "cross-unhide") == {"ok": False, "error": "not_found"}
    assert app.state.context_compaction_store.items(AFFINITY)[0].visibility == "compacted"
    replies.extend([native_response(native_call(LIST_TOOL, {}, "list-call")), native_response(text="listed")])
    listing = post(client)
    assert listing.status_code == 200 and LIST_TOOL not in listing.text
    listed = private_output(bodies[-1], "list-call")
    assert listed["ok"] and len(listed["items"]) == 1
    assert listed["items"][0]["item_id"] == item.item_id
    assert listed["items"][0]["tool_call_id"] == "original-call"
    assert listed["items"][0]["original_available"] is True
    replies.extend([native_response(native_call(UNHIDE_TOOL, {"item_id": item.item_id}, "unhide-call")), native_response(text="restored")])
    rendered = copy.deepcopy(MESSAGES)
    rendered[2]["content"] = item.compacted
    restored = post(client, messages=rendered)
    assert restored.status_code == 200 and UNHIDE_TOOL not in restored.text
    assert output(bodies[-2]) == item.compacted and output(bodies[-1]) == SOURCE
    assert private_output(bodies[-1], "unhide-call")["visibility"] == "visible"
    replies.append(native_response(text="still restored"))
    assert post(client, messages=rendered).status_code == 200
    assert output(bodies[-1]) == SOURCE
    saved = app.state.context_compaction_store.get(AFFINITY, item.item_id)
    assert saved is not None and saved.visibility == "original" and saved.original == SOURCE
    replies.extend([native_response(native_call(LIST_TOOL, {}, "list-after")), native_response(text="empty list")])
    assert post(client).status_code == 200
    assert private_output(bodies[-1], "list-after")["items"] == []
    for body in bodies:
        assert_original_call(body)
    assert rendered[2]["content"] == item.compacted and len(lfm.calls) == 1


@pytest.mark.parametrize("mixed", [False, True])
def test_anthropic_external_calls_preserve_id_and_input_without_executing_private_mixed(bridge, mixed):
    client, app, bodies, replies, lfm = bridge
    external = native_call("terminal", {"command": "next synthetic command", "timeout": 37}, "external-call")
    calls = [external]
    if mixed:
        calls.insert(0, native_call(HIDE_TOOL, {"tool_call_id": "original-call"}))
    replies.append(native_response(*calls, text="public explanation"))
    response = post(client)
    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]
    assert message["content"] == "public explanation"
    assert message["tool_calls"] == [{"id": "external-call", "type": "function", "function": {
        "name": "terminal", "arguments": json.dumps(external["input"], ensure_ascii=False)}}]
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["usage"]["total_tokens"] == 12
    assert HIDE_TOOL not in response.text and "private-call" not in response.text
    assert len(bodies) == 1 and not lfm.calls
    assert app.state.context_compaction_store.items(AFFINITY) == ()


@pytest.mark.parametrize("skip", ["missing_header", "disabled", "forced", "collision", "none", "no_tools"])
def test_anthropic_unsafe_or_visibility_only_requests_do_not_inject_private_tools(bridge, monkeypatch, skip):
    client, app, bodies, replies, lfm = bridge
    kwargs = {}
    if skip == "missing_header":
        kwargs["headers"] = {}
    elif skip == "disabled":
        monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "false")
    elif skip == "forced":
        kwargs["tool_choice"] = {"type": "function", "function": {"name": "terminal"}}
    elif skip == "collision":
        kwargs["tools"] = [TOOL, {"type": "function", "function": {"name": HIDE_TOOL, "parameters": {"type": "object"}}}]
    elif skip == "none":
        kwargs["tool_choice"] = "none"
    elif skip == "no_tools":
        kwargs["tools"] = None
    replies.append(native_response(text="plain"))
    response = post(client, **kwargs)
    assert response.status_code == 200 and response.json()["choices"][0]["message"]["content"] == "plain"
    names = {tool["name"] for tool in bodies[0].get("tools", [])}
    expected = {"terminal", HIDE_TOOL} if skip == "collision" else set() if skip in {"none", "no_tools"} else {"terminal"}
    assert names == expected
    if skip == "forced":
        assert bodies[0]["tool_choice"] == {"type": "tool", "name": "terminal"}
    assert output(bodies[0]) == SOURCE and not lfm.calls


@pytest.mark.parametrize("choice,native", [("auto", {"type": "auto"}), ("required", {"type": "any"})])
def test_anthropic_injected_tools_keep_native_choice_policy(bridge, choice, native):
    client, app, bodies, replies, lfm = bridge
    replies.append(native_response(native_call("terminal", {"command": "next"}, "external-call")))
    assert post(client, tool_choice=choice).status_code == 200
    assert bodies[0]["tool_choice"] == native
    assert {tool["name"] for tool in bodies[0]["tools"]} == {"terminal", HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}


def test_anthropic_external_call_after_private_hide_keeps_public_id_input_and_usage(bridge):
    client, app, bodies, replies, lfm = bridge
    external = native_call("terminal", {"command": "next command", "timeout": 37}, "external-after-hide")
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})),
                    native_response(external, text="public final explanation")])
    response = post(client)
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "public final explanation"
    assert choice["message"]["tool_calls"] == [{"id": "external-after-hide", "type": "function", "function": {
        "name": "terminal", "arguments": json.dumps(external["input"], ensure_ascii=False)}}]
    assert data["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    assert HIDE_TOOL not in response.text and "private-call" not in response.text
    assert len(bodies) == 2 and len(lfm.calls) == 1
    assert output(bodies[-1]) == app.state.context_compaction_store.items(AFFINITY)[0].compacted


def test_anthropic_multiple_private_tool_results_merge_without_losing_ids(bridge):
    client, app, bodies, replies, lfm = bridge
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"}, "hide-call"),
                                    native_call(LIST_TOOL, {}, "list-call")), native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200 and response.json()["choices"][0]["message"]["content"] == "finished"
    assert all(name not in response.text for name in (HIDE_TOOL, LIST_TOOL, "hide-call", "list-call"))
    final_messages = bodies[-1]["messages"]
    assert [message["role"] for message in final_messages] == ["user", "assistant", "user", "assistant", "user"]
    assert [block["id"] for block in final_messages[-2]["content"]] == ["hide-call", "list-call"]
    assert [block["tool_use_id"] for block in final_messages[-1]["content"]] == ["hide-call", "list-call"]
    hidden = private_output(bodies[-1], "hide-call")
    assert hidden["ok"] and private_output(bodies[-1], "list-call")["items"][0]["item_id"] == hidden["item_id"]
    assert len(lfm.calls) == 1


def test_anthropic_adjacent_original_results_and_user_text_keep_native_order(bridge):
    client, app, bodies, replies, lfm = bridge
    messages = copy.deepcopy(MESSAGES)
    second = {"id": "second-call", "type": "function", "function": {"name": "terminal", "arguments": '{ "command": "inspect second" }'}}
    messages[1]["tool_calls"].append(second)
    messages.extend([{"role": "tool", "tool_call_id": "second-call", "content": "second result untouched"},
                     {"role": "user", "content": "retain adjacent user text"}])
    before = copy.deepcopy(messages)
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="finished")])
    assert post(client, messages=messages).status_code == 200
    assert messages == before
    for body in bodies:
        assert native_call("terminal", {"command": "inspect second"}, "second-call") in blocks(body, "tool_use")
        assert body["messages"][2]["role"] == "user"
        merged = body["messages"][2]["content"]
        assert [block["type"] for block in merged] == ["tool_result", "tool_result", "text"]
        assert [block["tool_use_id"] for block in merged[:2]] == ["original-call", "second-call"]
        assert merged[1]["content"] == "second result untouched"
        assert merged[2]["text"] == "retain adjacent user text"
    assert output(bodies[0]) == SOURCE and output(bodies[1]) != SOURCE


def test_anthropic_invalid_lfm_output_is_identifiable_rule_fallback(bridge):
    client, app, bodies, replies, lfm = bridge
    lfm.text = "not JSON"
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="fallback")])
    assert post(client).status_code == 200
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    assert item.compaction_source == "rule" and SUMMARY not in item.compacted
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.lfm_calls == 1 and not measurement.lfm_applied
    assert measurement.lfm_fallback_reason == "invalid_json"
    assert output(bodies[-1]) == item.compacted


@pytest.mark.parametrize("missing", ["cache_read_input_tokens", "cache_creation_input_tokens"])
def test_anthropic_missing_cache_counter_is_not_reported_as_measured_zero(bridge, missing):
    client, app, bodies, replies, lfm = bridge
    first = {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 4}
    second = {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 4}
    del second[missing]
    replies.extend([native_response(native_call(LIST_TOOL, {}, "list-call"), usage=first), native_response(text="finished", usage=second)])
    response = post(client)
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert missing not in usage
    present = "cache_creation_input_tokens" if missing == "cache_read_input_tokens" else "cache_read_input_tokens"
    assert usage[present] == (8 if present == "cache_creation_input_tokens" else 0)
