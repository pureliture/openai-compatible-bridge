"""Native Google MockTransport fixtures, not paid-provider or real LFM evidence."""
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

ALIAS = "foundry:synthetic-google"
AFFINITY = "synthetic-google-conversation"


def native_call(name, args, call_id=None, signature=None):
    call = {"name": name, "args": args}
    if call_id is not None:
        call["id"] = call_id
    part = {"functionCall": call}
    if signature is not None:
        part["thoughtSignature"] = signature
    return part


def native_response(*calls, text=None, usage=None):
    parts = list(calls)
    if text is not None:
        parts.append({"text": text})
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12}
            if usage is None else usage}


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_LFM_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_LAYA_ENABLED", "false")
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", "")
    monkeypatch.setitem(vertex.MODEL_REGISTRY, ALIAS, {"provider": "foundry", "kind": "chat",
                        "provider_model": "synthetic-google", "protocol": "google_generate_content"})
    bodies, replies = [], []

    async def handler(request):
        assert request.url.path == "/api/v2/llm/proxy/google/v1/models/synthetic-google:generateContent"
        assert request.url.query == b""
        body = json.loads(request.content)
        assert "stream" not in body and "messages" not in body
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


def parts(body, kind):
    return [part for message in body["contents"] for part in message["parts"] if kind in part]


def output(body, name="terminal", index=0):
    return [part["functionResponse"]["response"] for part in parts(body, "functionResponse")
            if part["functionResponse"]["name"] == name][index]


def original_output(body):
    return output(body)["content"]


def declarations(body):
    return [tool for group in body.get("tools", []) for tool in group["functionDeclarations"]]


def assert_original_call(body):
    assert native_call("terminal", json.loads(ARGS)) in parts(body, "functionCall")
    assert body["contents"][0] == {"role": "user", "parts": [{"text": MESSAGES[0]["content"]}]}


def test_google_native_name_only_hide_is_private_and_result_only(bridge):
    client, app, bodies, replies, lfm = bridge
    original = copy.deepcopy(MESSAGES)
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})),
                    native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "finished"}
    assert data["choices"][0]["finish_reason"] == "stop"
    assert all(name not in response.text for name in (HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL))
    assert data["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    assert len(bodies) == 2
    assert {tool["name"] for tool in declarations(bodies[0])} == {"terminal", HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}
    assert declarations(bodies[0])[0]["parameters"] == TOOL["function"]["parameters"]
    assert original_output(bodies[0]) == SOURCE
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    assert item.original == SOURCE and item.compaction_source == "lfm"
    assert SUMMARY in item.compacted and original_output(bodies[1]) == item.compacted != SOURCE
    assert output(bodies[1], HIDE_TOOL)["ok"] is True
    assert native_call(HIDE_TOOL, {"tool_call_id": "original-call"}) in parts(bodies[1], "functionCall")
    for body in bodies:
        assert_original_call(body)
    assert MESSAGES == original and len(lfm.calls) == 1
    packet = json.loads(lfm.calls[0]["messages"][1]["content"].split("\n\n", 1)[1])
    assert set(packet) == {"result", "required_evidence"}
    assert ARGS not in json.dumps(lfm.calls) and MESSAGES[0]["content"] not in json.dumps(lfm.calls)
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.lfm_applied and measurement.lfm_calls == 1
    assert measurement.provider_calls == 2 and measurement.lfm_fallback_reason is None


def test_google_private_continuation_preserves_native_id_and_thought_signature(bridge):
    client, app, bodies, replies, lfm = bridge
    private = native_call(HIDE_TOOL, {"tool_call_id": "original-call"}, "native-private-id", "synthetic-signature")
    replies.extend([native_response(private), native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200
    assert private in parts(bodies[-1], "functionCall")
    result = output(bodies[-1], HIDE_TOOL)
    assert result["ok"] and result["tool_call_id"] == "original-call"
    assert all(value not in response.text for value in ("native-private-id", "synthetic-signature", HIDE_TOOL))
    assert next(part["functionResponse"] for part in parts(bodies[-1], "functionResponse")
                if part["functionResponse"]["name"] == HIDE_TOOL)["id"] == "native-private-id"


def test_google_multiple_name_only_private_results_share_one_native_user_turn(bridge):
    client, app, bodies, replies, lfm = bridge
    replies.extend([native_response(native_call(LIST_TOOL, {}), native_call(LIST_TOOL, {"query": "second"})),
                    native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200
    assert bodies[-1]["contents"][-1]["role"] == "user"
    assert len(bodies[-1]["contents"][-1]["parts"]) == 2
    assert [part["functionCall"]["args"] for part in bodies[-1]["contents"][-2]["parts"]] == [{}, {"query": "second"}]
    empty_listing = {"ok": True, "items": [], "hidden_count": 0, "saved_bytes": 0}
    assert output(bodies[-1], LIST_TOOL, 0) == empty_listing
    assert output(bodies[-1], LIST_TOOL, 1) == empty_listing
    assert LIST_TOOL not in response.text and not lfm.calls


def test_google_cache_and_total_usage_survive_private_continuation(bridge):
    client, app, bodies, replies, lfm = bridge
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 16,
             "cachedContentTokenCount": 3, "thoughtsTokenCount": 4}
    replies.extend([native_response(native_call(LIST_TOOL, {}), usage=usage),
                    native_response(text="finished", usage=usage)])
    response = post(client)
    assert response.status_code == 200
    assert response.json()["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 32,
        "prompt_tokens_details": {"cached_tokens": 6}}
    assert app.state.context_compaction_store.last_measurement.cache_read_tokens == 6


def test_google_name_only_external_ids_do_not_collide_with_prior_same_name_calls(bridge):
    client, app, bodies, replies, lfm = bridge
    messages = copy.deepcopy(MESSAGES)
    messages[1]["tool_calls"][0]["id"] = "call_terminal_0"
    messages[2]["tool_call_id"] = "call_terminal_0"
    replies.append(native_response(native_call("terminal", {"command": "next"})))
    response = post(client, messages=messages)
    assert response.status_code == 200
    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["id"] != "call_terminal_0"
    continued = copy.deepcopy(messages)
    continued.extend([{"role": "assistant", "content": None, "tool_calls": [call]},
                      {"role": "tool", "tool_call_id": call["id"], "content": SOURCE}])
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": call["id"]})),
                    native_response(text="hidden second")])
    assert post(client, messages=continued).status_code == 200
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    assert item.tool_call_id == call["id"]
    assert output(bodies[-1], "terminal", 0)["content"] == SOURCE
    assert output(bodies[-1], "terminal", 1)["content"] == item.compacted
    assert len(lfm.calls) == 1


def test_google_after_list_exact_restore_and_affinity_isolation(bridge):
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
        assert {tool["name"] for tool in declarations(bodies[-1])} == ({"terminal"} if policy.get("tool_choice") == "none" else set())
    replies.append(native_response(text="other affinity"))
    assert post(client, headers={AFFINITY_HEADER: "other"}).status_code == 200
    assert original_output(bodies[-1]) == SOURCE
    replies.extend([native_response(native_call(UNHIDE_TOOL, {"item_id": item.item_id})), native_response(text="other restore")])
    assert post(client, headers={AFFINITY_HEADER: "other"}).status_code == 200
    assert output(bodies[-1], UNHIDE_TOOL) == {"ok": False, "error": "not_found"}
    assert app.state.context_compaction_store.items(AFFINITY)[0].visibility == "compacted"
    replies.extend([native_response(native_call(LIST_TOOL, {})), native_response(text="listed")])
    response = post(client)
    assert response.status_code == 200 and LIST_TOOL not in response.text
    listed = output(bodies[-1], LIST_TOOL)
    assert listed["ok"] and len(listed["items"]) == 1
    assert listed["items"][0]["item_id"] == item.item_id
    assert listed["items"][0]["tool_call_id"] == "original-call"
    assert listed["items"][0]["original_available"] is True
    rendered = copy.deepcopy(MESSAGES)
    rendered[2]["content"] = item.compacted
    replies.extend([native_response(native_call(UNHIDE_TOOL, {"item_id": item.item_id})), native_response(text="restored")])
    restored = post(client, messages=rendered)
    assert restored.status_code == 200 and UNHIDE_TOOL not in restored.text
    assert original_output(bodies[-2]) == item.compacted and original_output(bodies[-1]) == SOURCE
    assert output(bodies[-1], UNHIDE_TOOL)["visibility"] == "visible"
    replies.append(native_response(text="still restored"))
    assert post(client, messages=rendered).status_code == 200
    assert original_output(bodies[-1]) == SOURCE and rendered[2]["content"] == item.compacted
    saved = app.state.context_compaction_store.get(AFFINITY, item.item_id)
    assert saved.original == SOURCE and saved.visibility == "original"
    replies.extend([native_response(native_call(LIST_TOOL, {})), native_response(text="empty list")])
    assert post(client).status_code == 200 and output(bodies[-1], LIST_TOOL)["items"] == []
    for body in bodies:
        assert_original_call(body)
    assert len(lfm.calls) == 1


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("native_id", [None, "external-native-id"])
def test_google_external_and_mixed_calls_preserve_public_ids_and_args(bridge, mixed, native_id):
    client, app, bodies, replies, lfm = bridge
    args = {"command": "next synthetic command", "timeout": 37}
    calls = [native_call("terminal", args, native_id)]
    if mixed:
        calls.insert(0, native_call(HIDE_TOOL, {"tool_call_id": "original-call"}))
    replies.append(native_response(*calls, text="public explanation"))
    response = post(client)
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "public explanation" and choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [{"id": native_id or f"call_terminal_{int(mixed)}", "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps(args, ensure_ascii=False)}}]
    assert HIDE_TOOL not in response.text and "_google" not in response.text
    assert len(bodies) == 1 and not lfm.calls and app.state.context_compaction_store.items(AFFINITY) == ()


@pytest.mark.parametrize("skip", ["missing_header", "disabled", "forced", "collision", "none", "no_tools"])
def test_google_excluded_requests_do_not_inject_private_tools(bridge, monkeypatch, skip):
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
    names = {tool["name"] for tool in declarations(bodies[0])}
    expected = {"terminal", HIDE_TOOL} if skip == "collision" else set() if skip == "no_tools" else {"terminal"}
    assert names == expected and original_output(bodies[0]) == SOURCE and not lfm.calls
    if skip == "forced":
        assert bodies[0]["toolConfig"] == {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["terminal"]}}
    if skip == "none":
        assert bodies[0]["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}


@pytest.mark.parametrize("choice,mode", [("auto", "AUTO"), ("required", "ANY")])
def test_google_injected_tools_preserve_choice_policy(bridge, choice, mode):
    client, app, bodies, replies, lfm = bridge
    replies.append(native_response(native_call("terminal", {"command": "next"})))
    assert post(client, tool_choice=choice).status_code == 200
    assert bodies[0]["toolConfig"] == {"functionCallingConfig": {"mode": mode}}
    assert {tool["name"] for tool in declarations(bodies[0])} == {"terminal", HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}


def test_google_external_call_after_hide_retains_native_id_and_aggregate_usage(bridge):
    client, app, bodies, replies, lfm = bridge
    args = {"command": "next command", "timeout": 37}
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})),
                    native_response(native_call("terminal", args, "external-after-hide", "external-signature"), text="public final")])
    response = post(client)
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["tool_calls"] == [{"id": "external-after-hide", "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps(args)}}]
    assert choice["message"]["content"] == "public final" and choice["finish_reason"] == "tool_calls"
    assert response.json()["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    assert all(value not in response.text for value in (HIDE_TOOL, "external-signature", "_google"))
    assert original_output(bodies[-1]) == app.state.context_compaction_store.items(AFFINITY)[0].compacted
    assert len(lfm.calls) == 1


def test_google_repeated_name_only_private_rounds_remain_ordered(bridge):
    client, app, bodies, replies, lfm = bridge
    replies.extend([native_response(native_call(LIST_TOOL, {})),
                    native_response(native_call(LIST_TOOL, {"query": "again"})),
                    native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200 and LIST_TOOL not in response.text
    final = bodies[-1]["contents"]
    assert [message["role"] for message in final] == ["user", "model", "user", "model", "user", "model", "user"]
    assert [part["functionCall"]["args"] for part in parts(bodies[-1], "functionCall") if part["functionCall"]["name"] == LIST_TOOL] == [{}, {"query": "again"}]
    assert [output(bodies[-1], LIST_TOOL, i) for i in range(2)] == [
        {"ok": True, "items": [], "hidden_count": 0, "saved_bytes": 0},
    ] * 2
    assert response.json()["usage"] == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}
    assert not lfm.calls


def test_google_multiple_hides_keep_lfm_at_one_attempt_per_turn(bridge):
    client, app, bodies, replies, lfm = bridge
    messages = copy.deepcopy(MESSAGES)
    messages[1]["tool_calls"].append({"id": "second-call", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"second"}'}})
    messages.append({"role": "tool", "tool_call_id": "second-call", "content": SOURCE})
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"}), native_call(HIDE_TOOL, {"tool_call_id": "second-call"})),
                    native_response(text="finished")])
    assert post(client, messages=messages).status_code == 200
    assert len(lfm.calls) == 1
    items = app.state.context_compaction_store.items(AFFINITY)
    assert len(items) == 2 and {item.compaction_source for item in items} == {"lfm", "rule"}
    assert [output(bodies[-1], HIDE_TOOL, i)["tool_call_id"] for i in range(2)] == ["original-call", "second-call"]


def test_google_original_repeated_name_calls_and_results_keep_order(bridge):
    client, app, bodies, replies, lfm = bridge
    messages = copy.deepcopy(MESSAGES)
    second = {"id": "second-call", "type": "function", "function": {"name": "terminal", "arguments": '{ "command": "inspect second" }'}}
    messages[1]["tool_calls"].append(second)
    messages.extend([{"role": "tool", "tool_call_id": "second-call", "content": "second result untouched"},
                     {"role": "user", "content": "retain adjacent user text"}])
    before = copy.deepcopy(messages)
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="finished")])
    assert post(client, messages=messages).status_code == 200 and messages == before
    for body in bodies:
        calls = parts(body, "functionCall")
        assert calls[:2] == [native_call("terminal", json.loads(ARGS)), native_call("terminal", {"command": "inspect second"})]
        assert [p["functionResponse"]["name"] for p in body["contents"][2]["parts"]] == ["terminal", "terminal"]
        assert output(body, "terminal", 1) == {"content": "second result untouched"}
        assert body["contents"][3] == {"role": "user", "parts": [{"text": "retain adjacent user text"}]}
    assert original_output(bodies[0]) == SOURCE and original_output(bodies[1]) != SOURCE


def test_google_invalid_lfm_output_is_rule_fallback_not_lfm_success(bridge):
    client, app, bodies, replies, lfm = bridge
    lfm.text = "not JSON"
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="fallback")])
    assert post(client).status_code == 200
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    assert item.compaction_source == "rule" and SUMMARY not in item.compacted
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.lfm_calls == 1 and not measurement.lfm_applied and measurement.lfm_fallback_reason == "invalid_json"
    assert original_output(bodies[-1]) == item.compacted


@pytest.mark.parametrize("cached", [None, 0, True, -1, "3"])
def test_google_absent_or_invalid_cache_count_is_not_fabricated(bridge, cached):
    client, app, bodies, replies, lfm = bridge
    first = {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12, "cachedContentTokenCount": 3}
    second = {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12}
    if cached is not None:
        second["cachedContentTokenCount"] = cached
    replies.extend([native_response(native_call(LIST_TOOL, {}), usage=first), native_response(text="finished", usage=second)])
    response = post(client)
    assert response.status_code == 200
    if cached == 0 and not isinstance(cached, bool):
        assert response.json()["usage"]["prompt_tokens_details"] == {"cached_tokens": 3}
        assert app.state.context_compaction_store.last_measurement.cache_read_tokens == 3
    else:
        assert "prompt_tokens_details" not in response.json()["usage"]
        assert app.state.context_compaction_store.last_measurement.cache_read_tokens is None

