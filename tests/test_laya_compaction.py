"""Synthetic M2/M3 opt-in checks; no real transcripts, hosted calls or credentials."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from openai_compatible_bridge.context_compaction import (
    COMPACT_TOOL,
    CompactionSettings,
    MemoryContextStore,
    _list_call,
    load_settings,
    plan_request,
    run_turn,
)
from openai_compatible_bridge.laya_http import LayaClient, LayaUnavailable
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers import vertex


class FakeLaya:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    async def choose(self, state, questions, question_id):
        self.calls.append((state, questions, question_id))
        if self.fail:
            raise LayaUnavailable("http_503")
        choices = questions[question_id]["criteria"]
        return next((key for key, value in choices.items() if "target_line" in value or "item_target" in value), "skip")


def _text(with_error=False):
    lines = [f"ordinary section {i:02d} " + "x" * 62 for i in range(18)]
    lines[10] = "target_line " + "a" * 70
    if with_error:
        lines[9] = "ERROR: test failed"
    return "\n".join(lines)


def _messages(original):
    return [
        {"role": "user", "content": "target_line 을 확인해 줘"},
        {"role": "assistant", "tool_calls": [{"id": "external", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "external", "content": original},
    ]


def _internal(name, args, ident="internal"):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeProvider:
    def __init__(self, first, second):
        self.responses = [first, second]
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _response(tool_call=None):
    return {"text": "done" if tool_call is None else None,
            "tool_calls": None if tool_call is None else [tool_call],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}


def _settings(*, validated=True):
    return CompactionSettings(enabled=True, laya_enabled=True,
                              laya_validated=validated, laya_base_url="http://laya.example:8000")


def _run(model, messages, store, settings, laya):
    plan, reason = plan_request(settings=settings, headers={"x-hermes-conversation": "conv"},
                                tools=[{"type": "function", "function": {"name": "terminal"}}],
                                tool_choice=None, provider="foundry", protocol="openai_chat_completions", stream=False)
    assert reason == "apply"
    return asyncio.run(run_turn(generate=model.generate, base_kwargs={}, messages=messages,
                                plan=plan, store=store, settings=settings, laya_client=laya))


def test_unvalidated_laya_never_receives_tool_or_user_text():
    settings = load_settings({"CONTEXT_COMPACTION_ENABLED": "true", "CONTEXT_COMPACTION_LAYA_ENABLED": "true", "LAYA_BASE_URL": "http://laya.example:8000"})
    assert not settings.laya_active
    laya = FakeLaya()
    original = _text()
    store = MemoryContextStore()
    model = FakeProvider(_response(_internal(COMPACT_TOOL, {"tool_call_id": "external"})), _response())
    outcome = _run(model, _messages(original), store, settings, laya)
    assert outcome.result["text"] == "done"
    assert laya.calls == []
    assert store.items("conv")[0].compacted == model.calls[1]["messages"][2]["content"]


def test_m2_laya_selects_exact_middle_line_and_freezes_once():
    settings = _settings()
    original = _text()
    messages = _messages(original)
    store = MemoryContextStore()
    laya = FakeLaya()
    model = FakeProvider(_response(_internal(COMPACT_TOOL, {"tool_call_id": "external"})), _response())
    outcome = _run(model, messages, store, settings, laya)
    assert outcome.result["text"] == "done"
    assert len(laya.calls) > 0
    assert all(question["relevance"]["type"] == "choice" for _, question, _ in laya.calls)
    item = store.items("conv")[0]
    assert "target_line" in item.compacted
    assert all(line in original.splitlines() for line in item.excerpt_lines)
    assert model.calls[0]["messages"][2]["content"] == original
    assert model.calls[1]["messages"][2]["content"] == item.compacted
    assert messages == _messages(original)
    assert store.compact(affinity="conv", tool_call_id="external", original=original, tool_name="terminal").item.compacted == item.compacted
    assert outcome.measurement.laya_calls == len(laya.calls)


def test_m2_full_wire_contract_with_mock_http_transport():
    payloads = []
    def respond(request):
        body = json.loads(request.content)
        payloads.append(body)
        options = body["questions"]["relevance"]["criteria"]
        choice = next((key for key, text in options.items() if "target_line" in text), "skip")
        return httpx.Response(200, json={
            "model": "laya-rl-agent", "routing": {"model": "multilingual"},
            "answers": {"relevance": {"type": "choice", "choice": choice,
                                      "probabilities": {choice: 0.91}, "answer_confidence": 0.91}},
            "usage": {"input_tokens": 42, "output_tokens": 0},
        })
    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    laya = LayaClient("http://laya.example:8000", http=http)
    model = FakeProvider(_response(_internal(COMPACT_TOOL, {"tool_call_id": "external"})), _response())
    outcome = _run(model, _messages(_text()), MemoryContextStore(), _settings(), laya)
    assert outcome.result is not None and outcome.measurement is not None
    assert outcome.result["text"] == "done"
    assert "target_line" in model.calls[1]["messages"][2]["content"]
    assert all(body["model"] == "multilingual" for body in payloads)
    assert outcome.measurement.laya_calls == len(payloads)
    asyncio.run(laya.close())


@pytest.mark.parametrize("protected", ["ERROR: test failed", "Duplicate invoice INV-EXAMPLE-42 was issued."])
def test_m2_protected_output_never_calls_laya(protected):
    original = _text().replace("target_line " + "a" * 70, protected)
    store = MemoryContextStore()
    laya = FakeLaya()
    model = FakeProvider(_response(_internal(COMPACT_TOOL, {"tool_call_id": "external"})), _response())
    outcome = _run(model, _messages(original), store, _settings(), laya)
    assert outcome.result is not None
    assert outcome.result["text"] == "done"
    assert model.calls[1]["messages"][2]["content"] == original
    assert store.items("conv") == ()
    assert laya.calls == []


def test_m2_laya_failure_and_error_output_fall_back_without_data_loss():
    for fail, error in [(True, False), (False, True)]:
        original = _text(with_error=error)
        store = MemoryContextStore()
        laya = FakeLaya(fail=fail)
        model = FakeProvider(_response(_internal(COMPACT_TOOL, {"tool_call_id": "external"})), _response())
        outcome = _run(model, _messages(original), store, _settings(), laya)
        assert outcome.result["text"] == "done"
        assert original == model.calls[1]["messages"][2]["content"] if error else True
        if error:
            assert laya.calls == []
            assert store.items("conv") == ()
        else:
            assert laya.calls
            assert store.items("conv")[0].compacted == model.calls[1]["messages"][2]["content"]
            assert "target_line" not in model.calls[1]["messages"][2]["content"]


def test_m2_oversized_candidate_set_uses_rule_without_remote_call():
    original = "\n".join(f"plain line {i:03d} " + "x" * 62 for i in range(70))
    laya = FakeLaya()
    model = FakeProvider(_response(_internal(COMPACT_TOOL, {"tool_call_id": "external"})), _response())
    outcome = _run(model, _messages(original), MemoryContextStore(), _settings(), laya)
    assert outcome.result["text"] == "done"
    assert laya.calls == []


def test_m3_laya_reorders_all_items_but_only_llm_may_unhide():
    store = MemoryContextStore()
    originals = (_text(), _text().replace("target_line", "item_target"))
    items = [store.compact(affinity="conv", tool_call_id=f"c{i}", original=original, tool_name="terminal").item
             for i, original in enumerate(originals)]
    assert all(items)
    expected = _list_call({"query": ""}, "conv", store)["items"]
    # Force Laya to select whichever is last in the rule ranking.
    target = expected[-1]["item_id"]
    class RerankLaya(FakeLaya):
        async def choose(self, state, questions, question_id):
            self.calls.append((state, questions, question_id))
            return target
    laya = RerankLaya()
    model = FakeProvider(_response(_internal("list_context_items", {"query": "target"})), _response())
    outcome = _run(model, _messages(originals[0]), store, _settings(), laya)
    listing = json.loads(model.calls[1]["messages"][-1]["content"])["items"]
    assert listing[0]["item_id"] == target
    assert {row["item_id"] for row in listing} == {item.item_id for item in items}
    assert all(item.visibility == "compacted" for item in store.items("conv"))
    assert outcome.measurement.laya_calls == 1


def test_m3_laya_failure_keeps_full_rule_list():
    store = MemoryContextStore()
    store.compact(affinity="conv", tool_call_id="c1", original=_text(), tool_name="terminal")
    laya = FakeLaya(fail=True)
    model = FakeProvider(_response(_internal("list_context_items", {"query": "target"})), _response())
    _run(model, _messages(_text()), store, _settings(), laya)
    listed = json.loads(model.calls[1]["messages"][-1]["content"])["items"]
    assert len(listed) == 1
    assert listed[0]["item_id"] == store.items("conv")[0].item_id


def test_http_app_opt_in_passes_laya_to_turn_and_closes_it(monkeypatch):
    for key, value in {
        "CONTEXT_COMPACTION_ENABLED": "true",
        "CONTEXT_COMPACTION_LAYA_ENABLED": "true",
        "CONTEXT_COMPACTION_LAYA_VALIDATED": "true",
        "LAYA_BASE_URL": "http://laya.example:8000",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", None)
    original_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY["foundry:synthetic-test-model"] = {
        "provider": "foundry", "kind": "chat", "provider_model": "synthetic-test-model",
        "protocol": "openai_chat_completions",
    }

    class Provider(FakeProvider):
        async def close(self):
            return None

    class Unused:
        async def close(self):
            return None

    class ClosingLaya(FakeLaya):
        closed = False
        async def close(self):
            self.closed = True

    laya = ClosingLaya()
    provider = Provider(_response(_internal(COMPACT_TOOL, {"tool_call_id": "external"})), _response())
    try:
        app = create_app(
            embedding_client_factory=Unused,
            chat_client_factory=Unused,
            rerank_client_factory=Unused,
            foundry_chat_client_factory=lambda: provider,
            laya_client_factory=lambda settings: laya,
        )
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", headers={"x-hermes-conversation": "conv"},
                                   json={"model": "foundry:synthetic-test-model", "messages": _messages(_text()),
                                         "tools": [{"type": "function", "function": {"name": "terminal"}}]})
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "done"
            assert "target_line" in provider.calls[1]["messages"][2]["content"]
            assert laya.calls
            assert app.state.context_compaction_store.last_measurement.laya_calls == len(laya.calls)
        assert laya.closed
    finally:
        vertex.MODEL_REGISTRY.clear()
        vertex.MODEL_REGISTRY.update(original_registry)
