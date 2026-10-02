"""Native HTTP protocol fixtures, not paid Foundry/provider canaries."""
from __future__ import annotations

import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.context_compaction import AFFINITY_HEADER, HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.foundry import FoundryChatClient

ALIAS = "foundry:synthetic-responses"
SOURCE = "\n".join(["Synthetic catalog contains ordinary component metadata." for _ in range(60)])
SUMMARY = "The synthetic catalog contains ordinary component metadata."
ARGS = '{"command":"inspect synthetic catalog","timeout":37,"background":false}'
TOOL = {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}
MESSAGES = [
    {"role": "user", "content": "Inspect the synthetic catalog; preserve this original request."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "original-call", "type": "function", "function": {"name": "terminal", "arguments": ARGS}},
    ]},
    {"role": "tool", "tool_call_id": "original-call", "content": SOURCE},
]


def native_call(name, args, call_id="private-call", item_id="fc-distinct-item"):
    return {"type": "function_call", "id": item_id, "call_id": call_id,
            "name": name, "arguments": json.dumps(args)}


def native_response(*calls, text=None):
    output = list(calls)
    if text is not None:
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    return {"id": "resp-synthetic", "status": "completed", "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                      "input_tokens_details": {"cached_tokens": 3}}}


class SyntheticLFM:
    """Exercise real LFMSummarizer validation with deterministic synthetic output."""
    def __init__(self):
        self.calls = []
        self.text = json.dumps({"summary": SUMMARY})

    async def generate(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return {"text": self.text, "finish_reason": "stop", "tool_calls": None,
                "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}}

    async def close(self):
        pass


class Unused:
    async def close(self):
        pass


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_LFM_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_LAYA_ENABLED", "false")
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", "")
    monkeypatch.setitem(vertex.MODEL_REGISTRY, ALIAS, {"provider": "foundry", "kind": "chat",
                        "provider_model": "synthetic-responses", "protocol": "openai_responses"})
    bodies, replies = [], []

    async def handler(request):
        assert request.url.path.endswith("/openai/v1/responses")
        body = json.loads(request.content)
        assert body["stream"] is False
        assert "messages" not in body
        bodies.append(body)
        assert replies, "unexpected native HTTP request"
        reply = replies.pop(0)
        return httpx.Response(200, json=reply(body) if callable(reply) else reply)

    # Construct with the HTTP client the provider owns; no real network or credentials.
    provider = FoundryChatClient(base_url="https://foundry.example/api/v2/llm/proxy/openai/v1/chat/completions", token="synthetic")
    original_http = provider.http
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    lfm = SyntheticLFM()
    app = create_app(embedding_client_factory=Unused, chat_client_factory=Unused,
                     rerank_client_factory=Unused, ollama_chat_client_factory=lambda: lfm,
                     foundry_chat_client_factory=lambda: provider, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        yield client, app, bodies, replies, lfm
    import asyncio
    asyncio.run(original_http.aclose())


def post(client, *, messages=None, headers=None, **kwargs):
    return client.post("/v1/chat/completions", headers={AFFINITY_HEADER: "synthetic-conversation"} if headers is None else headers,
                       json={"model": ALIAS, "messages": copy.deepcopy(MESSAGES if messages is None else messages),
                             "tools": [TOOL], **kwargs})


def original_output(body):
    return next(item["output"] for item in body["input"]
                if item.get("type") == "function_call_output" and item["call_id"] == "original-call")


def private_output(body, call_id):
    return json.loads(next(item["output"] for item in body["input"]
                           if item.get("type") == "function_call_output" and item["call_id"] == call_id))


def test_responses_native_hide_roundtrip_is_private_and_result_only(bridge):
    client, app, bodies, replies, lfm = bridge
    original = copy.deepcopy(MESSAGES)
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})),
                    native_response(text="finished")])
    response = post(client)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "finished"
    assert "tool_calls" not in data["choices"][0]["message"]
    assert all(name not in json.dumps(data) for name in (HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL, "private-call", "fc-distinct-item"))
    assert data["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24,
                             "prompt_tokens_details": {"cached_tokens": 6}}
    assert len(bodies) == 2
    assert {tool["name"] for tool in bodies[0]["tools"]} == {"terminal", HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}
    assert bodies[0]["model"] == "synthetic-responses"
    assert original_output(bodies[0]) == SOURCE
    item = app.state.context_compaction_store.items("synthetic-conversation")[0]
    assert item.compaction_source == "lfm"
    assert SUMMARY in item.compacted
    assert original_output(bodies[1]) == item.compacted != SOURCE
    expected_call = {"type": "function_call", "call_id": "original-call", "name": "terminal", "arguments": ARGS}
    assert all(expected_call in body["input"] for body in bodies)
    assert all(body["input"][0] == MESSAGES[0] for body in bodies)
    # Native item id differs from call_id. Continuation must pair by call_id.
    private = next(i for i in bodies[1]["input"] if i.get("name") == HIDE_TOOL)
    assert private["call_id"] == "private-call"
    assert private["arguments"] == json.dumps({"tool_call_id": "original-call"})
    assert private_output(bodies[1], "private-call")["ok"] is True
    assert MESSAGES == original
    assert len(lfm.calls) == 1
    packet = json.loads(lfm.calls[0]["messages"][1]["content"].split("\n\n", 1)[1])
    assert set(packet) == {"result", "required_evidence"}
    assert "invocation" not in packet and "context" not in packet
    assert ARGS not in json.dumps(lfm.calls) and MESSAGES[0]["content"] not in json.dumps(lfm.calls)
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.lfm_applied and measurement.lfm_calls == 1
    assert measurement.provider_calls == 2 and measurement.lfm_fallback_reason is None


def test_responses_after_list_exact_unhide_and_affinity_isolation(bridge):
    client, app, bodies, replies, lfm = bridge
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="hidden")])
    assert post(client).status_code == 200
    item = app.state.context_compaction_store.items("synthetic-conversation")[0]
    replies.append(native_response(text="after"))
    assert post(client).status_code == 200
    assert original_output(bodies[-1]) == item.compacted
    replies.append(native_response(text="visibility only"))
    assert post(client, tools=None).status_code == 200
    assert original_output(bodies[-1]) == item.compacted
    assert not bodies[-1].get("tools")
    replies.append(native_response(text="isolated"))
    assert post(client, headers={AFFINITY_HEADER: "other"}).status_code == 200
    assert original_output(bodies[-1]) == SOURCE
    replies.extend([native_response(native_call(LIST_TOOL, {}, "list-call", "fc-list")), native_response(text="listed")])
    listing = post(client)
    assert listing.status_code == 200 and LIST_TOOL not in listing.text
    listed = private_output(bodies[-1], "list-call")
    assert listed["ok"] and len(listed["items"]) == 1
    assert listed["items"][0]["item_id"] == item.item_id
    assert listed["items"][0]["tool_call_id"] == "original-call"
    assert listed["items"][0]["original_available"] is True
    replies.extend([native_response(native_call(UNHIDE_TOOL, {"item_id": item.item_id}, "unhide-call", "fc-unhide")), native_response(text="restored")])
    # Restore from an already rendered summary, not only client-resubmitted original.
    rendered = copy.deepcopy(MESSAGES)
    rendered[2]["content"] = item.compacted
    restored = post(client, messages=rendered)
    assert restored.status_code == 200 and UNHIDE_TOOL not in restored.text
    assert original_output(bodies[-2]) == item.compacted
    assert original_output(bodies[-1]) == SOURCE
    assert private_output(bodies[-1], "unhide-call")["visibility"] == "visible"
    replies.append(native_response(text="still restored"))
    assert post(client, messages=rendered).status_code == 200
    assert original_output(bodies[-1]) == SOURCE
    saved = app.state.context_compaction_store.get("synthetic-conversation", item.item_id)
    assert saved is not None and saved.visibility == "original" and saved.original == SOURCE
    replies.extend([native_response(native_call(LIST_TOOL, {}, "list-after", "fc-list-after")), native_response(text="empty list")])
    assert post(client).status_code == 200
    assert private_output(bodies[-1], "list-after")["items"] == []
    expected_call = {"type": "function_call", "call_id": "original-call", "name": "terminal", "arguments": ARGS}
    assert all(expected_call in body["input"] for body in bodies)
    assert rendered[2]["content"] == item.compacted
    assert len(lfm.calls) == 1 and not replies


@pytest.mark.parametrize("mixed", [False, True])
def test_responses_external_calls_preserve_native_call_id_without_executing_private_mixed(bridge, mixed):
    client, app, bodies, replies, lfm = bridge
    external = native_call("terminal", {"command": "next synthetic command"}, "external-call", "fc-external")
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
        "name": "terminal", "arguments": external["arguments"]}}]
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["usage"]["total_tokens"] == 12
    assert HIDE_TOOL not in response.text and "fc-external" not in response.text
    assert len(bodies) == 1 and not lfm.calls
    assert app.state.context_compaction_store.items("synthetic-conversation") == ()


@pytest.mark.parametrize("skip", ["missing_header", "disabled", "forced", "collision", "none", "no_tools"])
def test_responses_unsafe_or_visibility_only_requests_do_not_inject_private_tools(bridge, monkeypatch, skip):
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
    names = {t["name"] for t in bodies[0].get("tools", [])}
    assert names == ({"terminal", HIDE_TOOL} if skip == "collision" else set() if skip == "no_tools" else {"terminal"})
    if skip == "forced":
        assert bodies[0]["tool_choice"] == {"type": "function", "name": "terminal"}
    assert original_output(bodies[0]) == SOURCE and not lfm.calls


def test_responses_invalid_lfm_output_is_identifiable_rule_fallback(bridge):
    client, app, bodies, replies, lfm = bridge
    lfm.text = "not JSON"
    replies.extend([native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})), native_response(text="fallback")])
    assert post(client).status_code == 200
    item = app.state.context_compaction_store.items("synthetic-conversation")[0]
    assert item.compaction_source == "rule" and SUMMARY not in item.compacted
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.lfm_calls == 1 and not measurement.lfm_applied
    assert measurement.lfm_fallback_reason == "invalid_json"
    assert original_output(bodies[-1]) == item.compacted


@pytest.mark.parametrize("protocol,stream,provider,reason", [
    ("openai_responses", True, "foundry", "streaming"),
    ("openai_responses", False, "vertex", "unsupported_protocol"),
])
def test_only_responses_nonstream_foundry_slice_is_enabled(protocol, stream, provider, reason):
    from openai_compatible_bridge.context_compaction import CompactionSettings, plan_request
    plan, actual = plan_request(settings=CompactionSettings(enabled=True), headers={AFFINITY_HEADER: "synthetic"},
                                tools=[TOOL], tool_choice=None, provider=provider, protocol=protocol, stream=stream)
    assert plan is None and actual == reason
