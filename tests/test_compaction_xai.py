"""Native xAI MockTransport fixtures, not paid-provider or real LFM evidence."""
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
from test_compaction_foundry_protocols import (
    ARGS, MESSAGES, SOURCE, SUMMARY, TOOL, SyntheticLFM, Unused,
    native_call, original_output, private_output,
)

ALIAS = "foundry:synthetic-xai"
AFFINITY = "synthetic-xai-conversation"


def native_response(*calls, text=None, usage=None):
    output = list(calls)
    if text is not None:
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    return {"id": "resp-synthetic-xai", "status": "completed", "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12} if usage is None else usage}


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_LFM_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_LAYA_ENABLED", "false")
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", "")
    monkeypatch.setitem(vertex.MODEL_REGISTRY, ALIAS, {"provider": "foundry", "kind": "chat",
                        "provider_model": "synthetic-xai", "protocol": "xai_responses"})
    bodies, replies = [], []

    async def handler(request):
        assert request.url.path == "/api/v2/llm/proxy/xai/v1/responses"
        assert request.url.query == b""
        body = json.loads(request.content)
        assert body["stream"] is False and "messages" not in body
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


def assert_original_call(body):
    assert {"type": "function_call", "call_id": "original-call", "name": "terminal", "arguments": ARGS} in body["input"]
    assert body["input"][0] == MESSAGES[0]


def test_xai_native_hide_roundtrip_is_private_and_result_only(bridge):
    client, app, bodies, replies, lfm = bridge
    original = copy.deepcopy(MESSAGES)
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"}), text="private intermediate"),
                    native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "finished"}
    assert data["choices"][0]["finish_reason"] == "stop"
    assert all(value not in response.text for value in (HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL, "private-call", "fc-distinct-item", "private intermediate"))
    assert data["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    assert len(bodies) == 2 and not replies
    assert bodies[0]["model"] == "synthetic-xai"
    assert {t["name"] for t in bodies[0]["tools"]} == {"terminal", HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}
    assert bodies[0]["tools"][0]["parameters"] == TOOL["function"]["parameters"]
    assert original_output(bodies[0]) == SOURCE
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    assert item.original == SOURCE and item.compaction_source == "lfm"
    assert SUMMARY in item.compacted and original_output(bodies[1]) == item.compacted != SOURCE
    for body in bodies:
        assert_original_call(body)
    private = next(i for i in bodies[1]["input"] if i.get("name") == HIDE_TOOL)
    assert private["call_id"] == "private-call"  # Not the distinct native item id.
    assert private["arguments"] == json.dumps({"tool_call_id": "original-call"})
    assert private_output(bodies[1], "private-call")["ok"] is True
    assert MESSAGES == original and len(lfm.calls) == 1
    packet = json.loads(lfm.calls[0]["messages"][1]["content"].split("\n\n", 1)[1])
    assert set(packet) == {"result", "required_evidence"}
    assert ARGS not in json.dumps(lfm.calls) and MESSAGES[0]["content"] not in json.dumps(lfm.calls)
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.lfm_applied and measurement.lfm_calls == 1
    assert measurement.provider_calls == 2 and measurement.lfm_fallback_reason is None


def test_xai_native_cache_usage_survives_private_continuation(bridge):
    client, app, bodies, replies, lfm = bridge
    usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 16,
             "input_tokens_details": {"cached_tokens": 3}, "output_tokens_details": {"reasoning_tokens": 4}}
    replies.extend([native_response(native_call(LIST_TOOL, {}), usage=usage),
                    native_response(text="finished", usage=usage)])
    response = post(client)
    assert response.status_code == 200
    assert response.json()["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 32,
                                       "prompt_tokens_details": {"cached_tokens": 6}}
    assert app.state.context_compaction_store.last_measurement.cache_read_tokens == 6
    assert not lfm.calls and not replies


def test_xai_after_list_exact_restore_and_affinity_isolation(bridge):
    client, app, bodies, replies, lfm = bridge
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="hidden")])
    assert post(client).status_code == 200
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    replies.append(native_response(text="after"))
    assert post(client).status_code == 200 and original_output(bodies[-1]) == item.compacted
    for policy in ({"tools": None}, {"tool_choice": "none"}):
        replies.append(native_response(text="visibility only"))
        assert post(client, **policy).status_code == 200
        assert original_output(bodies[-1]) == item.compacted
        assert {t["name"] for t in bodies[-1].get("tools", [])} == ({"terminal"} if policy.get("tool_choice") == "none" else set())
    replies.append(native_response(text="isolated"))
    assert post(client, headers={AFFINITY_HEADER: "other"}).status_code == 200
    assert original_output(bodies[-1]) == SOURCE
    replies.extend([native_response(native_call(UNHIDE_TOOL, {"item_id": item.item_id}, "cross-unhide", "fc-cross")), native_response(text="isolated restore")])
    assert post(client, headers={AFFINITY_HEADER: "other"}).status_code == 200
    assert private_output(bodies[-1], "cross-unhide") == {"ok": False, "error": "not_found"}
    assert app.state.context_compaction_store.items(AFFINITY)[0].visibility == "compacted"
    replies.extend([native_response(native_call(LIST_TOOL, {}, "list-call", "fc-list")), native_response(text="listed")])
    listing = post(client)
    assert listing.status_code == 200 and LIST_TOOL not in listing.text
    listed = private_output(bodies[-1], "list-call")
    assert listed["ok"] and len(listed["items"]) == 1
    assert listed["items"][0]["item_id"] == item.item_id
    assert listed["items"][0]["tool_call_id"] == "original-call"
    assert listed["items"][0]["original_available"] is True
    rendered = copy.deepcopy(MESSAGES)
    rendered[2]["content"] = item.compacted
    replies.extend([native_response(native_call(UNHIDE_TOOL, {"item_id": item.item_id}, "unhide-call", "fc-unhide")), native_response(text="restored")])
    restored = post(client, messages=rendered)
    assert restored.status_code == 200 and UNHIDE_TOOL not in restored.text
    assert original_output(bodies[-2]) == item.compacted and original_output(bodies[-1]) == SOURCE
    assert private_output(bodies[-1], "unhide-call")["visibility"] == "visible"
    replies.append(native_response(text="still restored"))
    assert post(client, messages=rendered).status_code == 200
    assert original_output(bodies[-1]) == SOURCE and rendered[2]["content"] == item.compacted
    saved = app.state.context_compaction_store.get(AFFINITY, item.item_id)
    assert saved.original == SOURCE and saved.visibility == "original"
    replies.extend([native_response(native_call(LIST_TOOL, {}, "list-after", "fc-list-after")), native_response(text="empty list")])
    assert post(client).status_code == 200 and private_output(bodies[-1], "list-after")["items"] == []
    for body in bodies:
        assert_original_call(body)
    assert len(lfm.calls) == 1 and not replies


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("has_call_id", [False, True])
def test_xai_external_mixed_native_ids_and_exact_args_are_public(bridge, mixed, has_call_id):
    client, app, bodies, replies, lfm = bridge
    external = native_call("terminal", {}, "external-call", "fc-external")
    external["arguments"] = '{ "command": "next synthetic command", "timeout":37 }'
    if not has_call_id:
        del external["call_id"]
    calls = [external]
    if mixed:
        calls.insert(0, native_call(HIDE_TOOL, {"tool_call_id": "original-call"}))
    replies.append(native_response(*calls, text="public explanation"))
    response = post(client)
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "public explanation" and choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [{"id": "external-call" if has_call_id else "fc-external", "type": "function",
        "function": {"name": "terminal", "arguments": external["arguments"]}}]
    assert HIDE_TOOL not in response.text and "private-call" not in response.text
    assert response.json()["usage"]["total_tokens"] == 12
    assert len(bodies) == 1 and not lfm.calls and not replies
    assert app.state.context_compaction_store.items(AFFINITY) == ()


@pytest.mark.parametrize("skip", ["missing_header", "disabled", "forced", "collision", "none", "no_tools"])
def test_xai_excluded_requests_do_not_inject_private_tools(bridge, monkeypatch, skip):
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
    assert response.status_code == 200
    names = {t["name"] for t in bodies[0].get("tools", [])}
    assert names == ({"terminal", HIDE_TOOL} if skip == "collision" else set() if skip == "no_tools" else {"terminal"})
    if skip == "forced":
        assert bodies[0]["tool_choice"] == {"type": "function", "name": "terminal"}
    if skip == "none":
        assert bodies[0]["tool_choice"] == "none"
    assert original_output(bodies[0]) == SOURCE and not lfm.calls and not replies


@pytest.mark.parametrize("choice", ["auto", "required"])
def test_xai_injected_tools_preserve_choice_policy(bridge, choice):
    client, app, bodies, replies, lfm = bridge
    replies.append(native_response(native_call("terminal", {"command": "next"}, "external", "fc-external")))
    assert post(client, tool_choice=choice).status_code == 200
    assert bodies[0]["tool_choice"] == choice
    assert {t["name"] for t in bodies[0]["tools"]} == {"terminal", HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}
    assert not lfm.calls and not replies


def test_xai_external_after_private_hide_preserves_id_arguments_and_usage(bridge):
    client, app, bodies, replies, lfm = bridge
    external = native_call("terminal", {}, "external-after-hide", "fc-after-hide")
    external["arguments"] = '{ "command": "next", "background":false }'
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})),
                    native_response(external, text="public final")])
    response = post(client)
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["tool_calls"] == [{"id": "external-after-hide", "type": "function",
        "function": {"name": "terminal", "arguments": external["arguments"]}}]
    assert choice["message"]["content"] == "public final" and choice["finish_reason"] == "tool_calls"
    assert response.json()["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    assert HIDE_TOOL not in response.text and "fc-after-hide" not in response.text
    assert original_output(bodies[-1]) == app.state.context_compaction_store.items(AFFINITY)[0].compacted
    assert len(lfm.calls) == 1 and not replies


def test_xai_parallel_private_results_pair_by_call_id_not_item_id(bridge):
    client, app, bodies, replies, lfm = bridge
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"}, "hide-call", "fc-hide"),
                                    native_call(LIST_TOOL, {}, "list-call", "fc-list")), native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200
    assert all(value not in response.text for value in (HIDE_TOOL, LIST_TOOL, "hide-call", "list-call", "fc-hide", "fc-list"))
    final = bodies[-1]["input"]
    assert [i["call_id"] for i in final if i.get("type") == "function_call"] == ["original-call", "hide-call", "list-call"]
    assert [i["call_id"] for i in final if i.get("type") == "function_call_output"] == ["original-call", "hide-call", "list-call"]
    hidden = private_output(bodies[-1], "hide-call")
    assert hidden["ok"] and private_output(bodies[-1], "list-call")["items"][0]["item_id"] == hidden["item_id"]
    assert len(lfm.calls) == 1 and not replies


def test_xai_adjacent_original_calls_results_and_user_text_remain_exact(bridge):
    client, app, bodies, replies, lfm = bridge
    messages = copy.deepcopy(MESSAGES)
    second_args = '{ "command": "inspect second", "timeout":37 }'
    messages[1]["tool_calls"].append({"id": "second-call", "type": "function", "function": {"name": "terminal", "arguments": second_args}})
    messages.extend([{"role": "tool", "tool_call_id": "second-call", "content": "second result untouched"},
                     {"role": "user", "content": "retain adjacent user text"}])
    before = copy.deepcopy(messages)
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="finished")])
    assert post(client, messages=messages).status_code == 200 and messages == before
    for body in bodies:
        assert_original_call(body)
        assert body["input"][2] == {"type": "function_call", "call_id": "second-call", "name": "terminal", "arguments": second_args}
        assert body["input"][4] == {"type": "function_call_output", "call_id": "second-call", "output": "second result untouched"}
        assert body["input"][5] == {"role": "user", "content": "retain adjacent user text"}
    assert original_output(bodies[0]) == SOURCE and original_output(bodies[1]) != SOURCE
    assert len(lfm.calls) == 1 and not replies


def test_xai_repeated_private_rounds_preserve_native_continuation_order(bridge):
    client, app, bodies, replies, lfm = bridge
    replies.extend([native_response(native_call(LIST_TOOL, {}, "list-one", "fc-one")),
                    native_response(native_call(LIST_TOOL, {"query": "again"}, "list-two", "fc-two")),
                    native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200 and LIST_TOOL not in response.text
    final = bodies[-1]["input"]
    assert [i["call_id"] for i in final if i.get("type") == "function_call"] == ["original-call", "list-one", "list-two"]
    assert [i["call_id"] for i in final if i.get("type") == "function_call_output"] == ["original-call", "list-one", "list-two"]
    empty_listing = {"ok": True, "items": [], "hidden_count": 0, "saved_bytes": 0}
    assert private_output(bodies[-1], "list-one") == private_output(bodies[-1], "list-two") == empty_listing
    assert response.json()["usage"] == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}
    assert not lfm.calls and not replies


def test_xai_multiple_hides_keep_lfm_at_one_attempt_per_turn(bridge):
    client, app, bodies, replies, lfm = bridge
    messages = copy.deepcopy(MESSAGES)
    messages[1]["tool_calls"].append({"id": "second-call", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"second"}'}})
    messages.append({"role": "tool", "tool_call_id": "second-call", "content": SOURCE})
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"}, "hide-one", "fc-one"),
                                    native_call(HIDE_TOOL, {"tool_call_id": "second-call"}, "hide-two", "fc-two")),
                    native_response(text="finished")])
    response = post(client, messages=messages)
    assert response.status_code == 200 and HIDE_TOOL not in response.text
    assert len(lfm.calls) == 1
    items = app.state.context_compaction_store.items(AFFINITY)
    assert len(items) == 2 and {item.compaction_source for item in items} == {"lfm", "rule"}
    assert [private_output(bodies[-1], ident)["tool_call_id"] for ident in ("hide-one", "hide-two")] == ["original-call", "second-call"]
    assert not replies


def test_xai_invalid_lfm_output_is_rule_fallback_not_lfm_success(bridge):
    client, app, bodies, replies, lfm = bridge
    lfm.text = "not JSON"
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="fallback")])
    assert post(client).status_code == 200
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    assert item.compaction_source == "rule" and SUMMARY not in item.compacted
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.lfm_calls == 1 and not measurement.lfm_applied and measurement.lfm_fallback_reason == "invalid_json"
    assert original_output(bodies[-1]) == item.compacted and not replies


@pytest.mark.parametrize("cached", [None, 0, True, -1, "3"])
def test_xai_absent_or_invalid_native_cache_is_not_fabricated(bridge, cached):
    client, app, bodies, replies, lfm = bridge
    first = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "input_tokens_details": {"cached_tokens": 3}}
    second: dict[str, object] = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    if cached is not None:
        second["input_tokens_details"] = {"cached_tokens": cached}
    replies.extend([native_response(native_call(LIST_TOOL, {}), usage=first), native_response(text="finished", usage=second)])
    response = post(client)
    assert response.status_code == 200
    if cached == 0 and not isinstance(cached, bool):
        assert response.json()["usage"]["prompt_tokens_details"] == {"cached_tokens": 3}
        assert app.state.context_compaction_store.last_measurement.cache_read_tokens == 3
    else:
        assert "prompt_tokens_details" not in response.json()["usage"]
        assert app.state.context_compaction_store.last_measurement.cache_read_tokens is None
    assert not lfm.calls and not replies
