from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.context_compaction import (
    CompactionUpstreamError,
    AFFINITY_HEADER,
    HIDE_TOOL,
    CompactionSettings,
    ContextItem,
    LIST_TOOL,
    MemoryContextStore,
    RuleSpanSelector,
    SpanChoice,
    aggregate_usage,
    apply_visibility,
    compare_span_selectors,
    compare_unhide_rankers,
    load_settings,
    internal_tool_definitions,
    _hide_call,
    _unhide_call,
    plan_request,
    rank_compacted_items,
    run_turn,
)
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.vertex import VertexAPIError, _coerce_openai_usage


CLIENT_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Run a command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
}


class _UnusedVertexClient:
    """Vertex 슬롯을 채우기 위한 no-op 클라이언트.

    이 테스트들은 Foundry 경로만 검증하므로 GCP 인증이 필요 없어야 한다.
    lifespan이 시작/종료 시 close()를 호출하므로 반드시 정의한다.
    """

    async def close(self) -> None:
        return None


def _foundry_only_app(**overrides: Any) -> Any:
    """Vertex 팩토리를 no-op로 대체한 create_app.

    기본 create_app()은 lifespan에서 Vertex 클라이언트를 만들어 google.auth.default()를
    호출하므로, GCP 자격증명이 없는 환경(CI)에서는 TestClient 기동이 실패한다.
    """
    kwargs: dict[str, Any] = {
        "embedding_client_factory": _UnusedVertexClient,
        "chat_client_factory": _UnusedVertexClient,
        "rerank_client_factory": _UnusedVertexClient,
    }
    kwargs.update(overrides)
    return create_app(**kwargs)


def _listing() -> str:
    # Exercise successful compaction with bounded evidence. Large path lists
    # must now stay original (covered separately by protection regressions).
    lines = [f"ordinary output segment {index:03d} without special evidence" for index in range(60)]
    for index in (0, 1, 58, 59):
        lines[index] = f"src/module_{index:03d}.py"
    lines.insert(30, "id: 123e4567-e89b-12d3-a456-426614174000")
    return "\n".join(lines)


def _messages(content: str | None = None) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "목록을 확인해 줘"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_17",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{\"command\":\"rg --files src\"}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_17", "content": content if content is not None else _listing()},
    ]


def _usage(prompt: int, completion: int, cached: int | None = None) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def _hide_tool_call() -> dict[str, Any]:
    return {
        "id": "call_hide",
        "type": "function",
        "function": {"name": HIDE_TOOL, "arguments": json.dumps({"tool_call_id": "call_17"})},
    }


def _settings(**overrides: Any) -> CompactionSettings:
    values = {
        "enabled": True,
        "header_name": AFFINITY_HEADER,
        "ttl_seconds": 60,
        "max_bytes": 1024 * 1024,
        "max_internal_rounds": 3,
        "min_chars": 800,
        "laya_enabled": False,
    }
    values.update(overrides)
    return CompactionSettings(**values)


def _plan(headers: dict[str, str] | None = None, **overrides: Any):
    settings = _settings(**{key: overrides.pop(key) for key in list(overrides) if key in CompactionSettings.__dataclass_fields__})
    return plan_request(
        settings=settings,
        headers=headers if headers is not None else {AFFINITY_HEADER: "conv-a"},
        tools=overrides.pop("tools", [CLIENT_TOOL]),
        tool_choice=overrides.pop("tool_choice", None),
        provider=overrides.pop("provider", "foundry"),
        protocol=overrides.pop("protocol", "openai_chat_completions"),
        stream=overrides.pop("stream", False),
    )


class _ScriptedModel:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected provider call")
        return self.responses.pop(0)

    async def close(self) -> None:
        return None


def _register_alias() -> dict[str, dict[str, Any]]:
    old = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY["foundry:gpt-6-astra"] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": "gpt-6-astra",
        "protocol": "openai_chat_completions",
    }
    return old


def _restore_alias(old: dict[str, dict[str, Any]]) -> None:
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old)


def test_affinity_header_is_the_only_key_and_missing_header_skips():
    plan, reason = _plan(headers={})
    assert plan is None
    assert reason == "missing_affinity"
    plan, reason = _plan(headers={"X-Hermes-Conversation": " conv-a "})
    assert plan is not None
    assert plan.affinity_key == " conv-a "
    assert reason == "apply"
    disabled, reason = plan_request(
        settings=load_settings({}),
        headers={AFFINITY_HEADER: "conv-a"},
        tools=[CLIENT_TOOL],
        tool_choice=None,
        provider="foundry",
        protocol="openai_chat_completions",
        stream=False,
    )
    assert disabled is None
    assert reason == "disabled"
    skipped, reason = _plan(provider="vertex")
    assert skipped is None
    assert reason == "unsupported_protocol"
    skipped, reason = _plan(stream=True)
    assert skipped is None
    assert reason == "streaming"


@pytest.mark.parametrize("protocol", ["openai_chat_completions", "openai_responses", "anthropic_messages"])
def test_verified_foundry_nonstream_protocols_share_compaction_gate(protocol):
    plan, reason = _plan(protocol=protocol)
    assert plan is not None and reason == "apply" and plan.tools is not None
    assert {tool["function"]["name"] for tool in plan.tools} == {
        "terminal", HIDE_TOOL, LIST_TOOL, "unhide_context",
    }
    skipped, reason = _plan(protocol=protocol, stream=True)
    assert skipped is None and reason == "streaming"
    skipped, reason = _plan(protocol=protocol, provider="vertex")
    assert skipped is None and reason == "unsupported_protocol"


@pytest.mark.parametrize("protocol", ["google_generate_content", "xai_responses"])
def test_unverified_foundry_protocol_slices_remain_disabled(protocol):
    plan, reason = _plan(protocol=protocol)
    assert plan is None and reason == "unsupported_protocol"


def test_forced_tool_choice_and_name_collision_skip_feature():
    plan, reason = _plan(tool_choice={"type": "function", "function": {"name": "terminal"}})
    assert plan is None
    assert reason == "forced_tool_choice"
    colliding = [*([CLIENT_TOOL]), {"type": "function", "function": {"name": HIDE_TOOL, "parameters": {}}}]
    plan, reason = _plan(tools=colliding)
    assert plan is None
    assert reason == "tool_name_collision"


def test_bridge_exposes_hide_unhide_list_tools_and_requires_tool_call_id_for_hide():
    definitions = internal_tool_definitions()
    by_name = {tool["function"]["name"]: tool["function"] for tool in definitions}
    assert set(by_name) == {HIDE_TOOL, LIST_TOOL, "unhide_context"}
    assert "compact_context" not in by_name

    hide = by_name[HIDE_TOOL]
    assert hide["parameters"]["required"] == ["tool_call_id"]
    assert set(hide["parameters"]["properties"]) == {"tool_call_id", "context"}
    assert hide["parameters"]["additionalProperties"] is False
    assert "hide" in hide["description"].lower()
    assert "Does not rerun" in hide["description"]

    unhide = by_name["unhide_context"]
    assert unhide["parameters"]["required"] == ["item_id"]
    assert set(unhide["parameters"]["properties"]) == {"item_id"}

    store = MemoryContextStore(min_chars=800)
    messages = _messages()
    missing = _hide_call({}, messages, "conv-a", store)
    unexpected = _hide_call(
        {"tool_call_id": "call_17", "item_id": "item-synthetic"}, messages, "conv-a", store,
    )
    unknown = _hide_call({"tool_call_id": "not-a-result"}, messages, "conv-a", store)
    duplicated = _messages()
    duplicated.append({"role": "tool", "tool_call_id": "call_17", "content": _listing()})
    ambiguous = _hide_call({"tool_call_id": "call_17"}, duplicated, "conv-a", store)
    assert missing == {"ok": False, "error": "missing_tool_call_id"}
    assert unexpected == {"ok": False, "error": "invalid_arguments"}
    assert unknown == {"ok": False, "error": "not_found"}
    assert ambiguous == {"ok": False, "error": "ambiguous"}
    assert store.items("conv-a") == ()

    hidden = _hide_call({"tool_call_id": "call_17"}, messages, "conv-a", store)
    assert hidden["ok"] is True
    assert hidden["visibility"] == "hidden"
    hidden_item = store.get("conv-a", hidden["item_id"])
    assert hidden_item is not None
    assert store.visible_content("conv-a", "call_17", messages[2]["content"]) == hidden_item.compacted
    assert messages[1]["tool_calls"][0]["id"] == "call_17"
    restored = _unhide_call({"item_id": hidden["item_id"]}, "conv-a", store)
    assert restored["ok"] is True
    assert restored["visibility"] == "visible"
    assert store.visible_content("conv-a", "call_17", messages[2]["content"]) == messages[2]["content"]


def test_legacy_compact_context_is_not_injected_or_consumed_as_bridge_alias():
    legacy_tool = {
        "type": "function",
        "function": {"name": "compact_context", "parameters": {"type": "object"}},
    }
    plan, reason = _plan(tools=[CLIENT_TOOL, legacy_tool])
    assert plan is not None and reason == "apply"
    store = MemoryContextStore(min_chars=800)
    model = _ScriptedModel([{
        "text": None,
        "tool_calls": [{
            "id": "legacy-client-call",
            "type": "function",
            "function": {"name": "compact_context", "arguments": "{}"},
        }],
        "finish_reason": "tool_calls",
        "usage": _usage(2, 1),
    }])
    outcome = asyncio.run(run_turn(
        generate=model.generate,
        base_kwargs={"tools": [CLIENT_TOOL, legacy_tool]},
        messages=_messages(),
        plan=plan,
        store=store,
        settings=_settings(),
    ))
    assert outcome.result is not None
    assert outcome.result["tool_calls"][0]["function"]["name"] == "compact_context"
    assert HIDE_TOOL in {tool["function"]["name"] for tool in model.calls[0]["tools"]}
    assert store.items("conv-a") == ()


def test_rule_compaction_keeps_exact_excerpts_and_is_idempotent():
    store = MemoryContextStore(ttl_seconds=60, min_chars=800, clock=lambda: 100.0)
    original = _listing()
    messages = _messages(original)
    first = store.compact(affinity="conv-a", tool_call_id="call_17", original=original, tool_name="terminal", now=100)
    second = store.compact(affinity="conv-a", tool_call_id="call_17", original=original, tool_name="terminal", now=101)
    assert first.ok and second.ok
    assert first.item is not None and second.item is not None
    assert first.item.compacted == second.item.compacted
    assert second.item.version == 1
    for line in first.item.excerpt_lines:
        assert line in original.splitlines()
    assert "123e4567-e89b-12d3-a456-426614174000" in first.item.compacted
    assert len(first.item.compacted) < len(original)
    visible = apply_visibility(messages, affinity="conv-a", store=store, now=101)
    again = apply_visibility(messages, affinity="conv-a", store=store, now=101)
    assert visible == again
    assert messages[2]["content"] == original
    assert visible[1]["tool_calls"] == messages[1]["tool_calls"]
    assert visible[2]["tool_call_id"] == "call_17"
    assert visible[2]["content"] == first.item.compacted
    assert visible[0]["content"] == messages[0]["content"]


def test_error_short_and_unverifiable_results_stay_original():
    store = MemoryContextStore(min_chars=800)
    short = store.compact(affinity="conv-a", tool_call_id="call_short", original="ok", tool_name="terminal")
    assert short.error == "not_long"
    error = store.compact(
        affinity="conv-a",
        tool_call_id="call_err",
        original="ERROR: boom\n" + ("x" * 900),
        tool_name="terminal",
    )
    assert error.error == "protected_error"
    giant = store.compact(affinity="conv-a", tool_call_id="call_giant", original="x" * 900, tool_name="terminal")
    assert giant.error == "verification_failed"
    assert store.items("conv-a") == ()


def test_affinity_isolation_expiry_and_concurrent_compact():
    clock = {"now": 0.0}
    store = MemoryContextStore(ttl_seconds=10, min_chars=800, clock=lambda: clock["now"])
    original = _listing()
    other = _listing() + "\nextra.py"
    first = store.compact(affinity="conv-a", tool_call_id="call_17", original=original, tool_name="terminal")
    assert first.ok
    full = store.compact(affinity="conv-a", tool_call_id="call_18", original=other, tool_name="terminal")
    assert full.ok  # No per-header item-count quota.
    isolated = store.compact(affinity="conv-b", tool_call_id="call_17", original=original, tool_name="terminal")
    assert isolated.ok
    assert store.visible_content("conv-b", "call_17", original) == isolated.item.compacted
    assert store.visible_content("conv-a", "call_17", other) == other
    clock["now"] = 11
    assert store.visible_content("conv-a", "call_17", original, now=11) == original

    shared = MemoryContextStore(min_chars=800)
    barrier = threading.Barrier(2)
    results: list[str] = []

    def compact() -> None:
        barrier.wait()
        result = shared.compact(affinity="conv-a", tool_call_id="call_17", original=original, tool_name="terminal")
        assert result.item is not None
        results.append(result.item.compacted)

    threads = [threading.Thread(target=compact), threading.Thread(target=compact)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results[0] == results[1]
    assert len(shared.items("conv-a")) == 1


def test_unhide_restores_exact_original_and_recompact_reuses_bytes():
    store = MemoryContextStore(min_chars=800, clock=lambda: 5)
    original = _listing()
    compacted = store.compact(affinity="conv-a", tool_call_id="call_17", original=original, tool_name="terminal")
    assert compacted.item is not None
    restored = store.unhide("conv-a", compacted.item.item_id)
    assert restored.item is not None
    assert restored.item.visibility == "original"
    assert store.visible_content("conv-a", "call_17", original) == original
    again = store.compact(affinity="conv-a", tool_call_id="call_17", original=original, tool_name="terminal")
    assert again.item is not None
    assert again.item.compacted == compacted.item.compacted
    assert again.item.version == 3
    missing = store.unhide("conv-b", compacted.item.item_id)
    assert missing.error == "not_found"


def test_internal_loop_hides_tools_and_sums_cache_usage():
    store = MemoryContextStore(min_chars=800)
    plan, reason = _plan()
    assert plan is not None and reason == "apply"
    model = _ScriptedModel(
        [
            {
                "text": None,
                "tool_calls": [_hide_tool_call()],
                "finish_reason": "tool_calls",
                "usage": _usage(10, 2, 4),
            },
            {
                "text": "완료",
                "tool_calls": None,
                "finish_reason": "stop",
                "usage": _usage(6, 1, 5),
            },
        ]
    )
    messages = _messages()
    outcome = asyncio.run(
        run_turn(
            generate=model.generate,
            base_kwargs={"model": "gpt-6-astra", "messages": messages, "tools": [CLIENT_TOOL]},
            messages=messages,
            plan=plan,
            store=store,
            settings=_settings(),
        )
    )
    assert outcome.result is not None
    assert outcome.result["text"] == "완료"
    assert outcome.result["tool_calls"] is None
    assert outcome.result["usage"]["prompt_tokens"] == 16
    assert outcome.result["usage"]["prompt_tokens_details"]["cached_tokens"] == 9
    assert model.calls[0]["messages"][2]["content"] == _listing()
    assert model.calls[1]["messages"][2]["content"] != _listing()
    assert "123e4567-e89b-12d3-a456-426614174000" in model.calls[1]["messages"][2]["content"]
    assert model.calls[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == "{\"command\":\"rg --files src\"}"
    assert all(call["function"]["name"] != HIDE_TOOL for call in model.calls[0]["messages"][1]["tool_calls"])
    assert HIDE_TOOL in {tool["function"]["name"] for tool in model.calls[0]["tools"]}
    assert outcome.measurement is not None
    assert outcome.measurement.provider_calls == 2
    assert outcome.measurement.laya_calls == 0
    partial = aggregate_usage([_usage(3, 1, 2), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}])
    assert "prompt_tokens_details" not in partial


def test_mixed_calls_do_not_change_state_or_leak_internal_tools():
    store = MemoryContextStore(min_chars=800)
    plan, _reason = _plan()
    assert plan is not None
    external = {
        "id": "call_ext",
        "type": "function",
        "function": {"name": "terminal", "arguments": "{\"command\":\"pwd\"}"},
    }
    model = _ScriptedModel(
        [
            {
                "text": None,
                "tool_calls": [_hide_tool_call(), external],
                "finish_reason": "tool_calls",
                "usage": _usage(4, 1),
            }
        ]
    )
    outcome = asyncio.run(
        run_turn(
            generate=model.generate,
            base_kwargs={},
            messages=_messages(),
            plan=plan,
            store=store,
            settings=_settings(),
        )
    )
    assert outcome.result is not None
    assert [call["function"]["name"] for call in outcome.result["tool_calls"]] == ["terminal"]
    assert store.items("conv-a") == ()


def test_store_failure_before_provider_call_skips():
    class BrokenStore(MemoryContextStore):
        def visible_content(self, affinity: str, tool_call_id: str, content: str, *, now: float | None = None) -> str:
            raise RuntimeError("disk failed")

    plan, _reason = _plan()
    assert plan is not None
    model = _ScriptedModel([])
    outcome = asyncio.run(
        run_turn(
            generate=model.generate,
            base_kwargs={},
            messages=_messages(),
            plan=plan,
            store=BrokenStore(),
            settings=_settings(),
        )
    )
    assert outcome.skipped is True
    assert model.calls == []


def test_loop_limit_does_not_return_internal_tool_call():
    plan, _reason = _plan()
    assert plan is not None
    model = _ScriptedModel(
        [
            {"text": None, "tool_calls": [_hide_tool_call()], "finish_reason": "tool_calls", "usage": _usage(2, 1)},
            {"text": None, "tool_calls": [_hide_tool_call()], "finish_reason": "tool_calls", "usage": _usage(2, 1)},
        ]
    )
    outcome = asyncio.run(
        run_turn(
            generate=model.generate,
            base_kwargs={},
            messages=_messages(),
            plan=plan,
            store=MemoryContextStore(min_chars=800),
            settings=_settings(max_internal_rounds=1),
        )
    )
    assert outcome.error is not None
    assert outcome.error[2] == "context_compaction_loop_limit"
    assert outcome.result is None
    dumped = json.dumps(outcome.error)
    assert HIDE_TOOL not in dumped
    assert "src/module" not in dumped


def test_laya_is_not_accepted_without_a_better_validated_result():
    original = _listing()
    cases = [{"text": original, "must_keep": ("src/module_000.py",)}]
    rule = compare_span_selectors(cases, RuleSpanSelector(), validated=False)
    assert rule.accepted is True
    assert rule.missing_required == 0
    assert rule.invented_text == 0

    class Dropping:
        source = "laya-stand-in"

        def select(self, text: str) -> SpanChoice:
            return SpanChoice((), self.source)

    dropped = compare_span_selectors(cases, Dropping(), validated=True)
    assert dropped.accepted is False
    assert dropped.missing_required == 1
    settings = load_settings({"CONTEXT_COMPACTION_LAYA_ENABLED": "true"})
    assert settings.laya_enabled is True
    assert settings.laya_active is False

    item = ContextItem(
        item_id="item_relevant",
        tool_call_id="call_17",
        content_sha256="abc",
        original="src/api.py",
        compacted="원문 발췌: src/api.py",
        excerpt_lines=("src/api.py",),
        visibility="compacted",
        version=1,
        expires_at=10,
        tool_name="terminal",
    )
    other = ContextItem(
        item_id="item_other",
        tool_call_id="call_18",
        content_sha256="def",
        original="notes",
        compacted="원문 발췌: notes",
        excerpt_lines=("notes",),
        visibility="compacted",
        version=1,
        expires_at=10,
        tool_name="search",
    )
    ranking = compare_unhide_rankers(
        [{"query": "api", "items": (item, other), "relevant": ("item_relevant",), "irrelevant": ("item_other",)}],
        ranker_name="rule",
        rank=rank_compacted_items,
        validated=False,
    )
    assert ranking.missed_relevant == 0
    assert rank_compacted_items("api", (item, other))[0].item_id == "item_relevant"


def test_usage_coercion_preserves_only_returned_cache_fields():
    plain = _coerce_openai_usage({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})
    assert plain == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    detailed = _coerce_openai_usage(
        {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 5, "audio_tokens": "nope"},
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 1,
        }
    )
    assert detailed["prompt_tokens_details"] == {"cached_tokens": 5}
    assert detailed["cache_read_input_tokens"] == 5
    assert detailed["cache_creation_input_tokens"] == 1


def _post(client: TestClient, headers: dict[str, str], *, user: str | None = None, tool_choice: Any = None, tools: list[dict[str, Any]] | None = None):
    payload: dict[str, Any] = {
        "model": "foundry:gpt-6-astra",
        "messages": _messages(),
        "tools": tools if tools is not None else [CLIENT_TOOL],
    }
    if user is not None:
        payload["user"] = user
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return client.post("/v1/chat/completions", headers=headers, json=payload)


def test_http_compaction_is_opt_in_and_keyed_only_by_header(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "true")
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", "shared-bridge-key")
    monkeypatch.setenv("COST_TRACKING_ENABLED", "true")
    monkeypatch.setenv("COST_LEDGER_PATH", str(tmp_path / "cost.db"))
    monkeypatch.setenv(
        "COST_PRICING_JSON",
        json.dumps(
            {
                "source": "unit-test",
                "version": "2026-09-27",
                "currency": "USD",
                "models": {
                    "foundry:gpt-6-astra": {
                        "chat": {"input_per_million": "1", "output_per_million": "1"}
                    }
                },
            }
        ),
    )
    monkeypatch.setenv("COST_SHORT_WINDOW_SECONDS", "60")
    monkeypatch.setenv("COST_SHORT_WINDOW_LIMIT_USD", "unlimited")
    monkeypatch.setenv("COST_DAILY_LIMIT_USD", "unlimited")
    old = _register_alias()
    model = _ScriptedModel(
        [
            {"text": None, "tool_calls": [_hide_tool_call()], "finish_reason": "tool_calls", "usage": _usage(10, 2, 3)},
            {"text": "완료", "tool_calls": None, "finish_reason": "stop", "usage": _usage(7, 1, 6)},
            {"text": "다음", "tool_calls": None, "finish_reason": "stop", "usage": _usage(7, 1, 6)},
            {"text": "다른 대화", "tool_calls": None, "finish_reason": "stop", "usage": _usage(9, 1)},
            {"text": "헤더 없음", "tool_calls": None, "finish_reason": "stop", "usage": _usage(4, 1)},
        ]
    )
    # This scripted model has no HTTP transport. Keep explicit legacy accounting
    # coverage here; test_compaction_cost_integration exercises runtime HTTP accounting.
    import os
    from openai_compatible_bridge.core.cost_tracking import build_cost_accounting_from_env

    try:
        app = _foundry_only_app(
            foundry_chat_client_factory=lambda: model,
            cost_accounting_factory=lambda: build_cost_accounting_from_env(os.environ),
        )
        with TestClient(app) as client:
            auth = {"Authorization": "Bearer shared-bridge-key", "X-Hermes-Conversation": "conv-a"}
            first = _post(client, auth, user="atlas")
            assert first.status_code == 200
            body = first.json()
            assert body["choices"][0]["message"]["content"] == "완료"
            assert "tool_calls" not in body["choices"][0]["message"]
            assert HIDE_TOOL not in json.dumps(body["choices"])
            assert body["usage"]["prompt_tokens"] == 17
            assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 9
            compacted = model.calls[1]["messages"][2]["content"]
            assert compacted != _listing()
            second = _post(client, auth, user="tyche")
            assert second.status_code == 200
            assert model.calls[2]["messages"][2]["content"] == compacted
            other = _post(client, {"Authorization": "Bearer shared-bridge-key", "X-Hermes-Conversation": "conv-b"}, user="atlas")
            assert other.status_code == 200
            assert model.calls[3]["messages"][2]["content"] == _listing()
            missing = _post(client, {"Authorization": "Bearer shared-bridge-key"})
            assert missing.status_code == 200
            assert model.calls[4]["messages"][2]["content"] == _listing()
            assert HIDE_TOOL not in {tool["function"]["name"] for tool in model.calls[4]["tools"]}
        with sqlite3.connect(tmp_path / "cost.db") as conn:
            row = conn.execute("select sum(prompt_tokens), sum(completion_tokens) from cost_events").fetchone()
        assert row == (17 + 7 + 9 + 4, 3 + 1 + 1 + 1)
    finally:
        _restore_alias(old)


def test_http_disabled_and_unsafe_requests_pass_through(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CONTEXT_COMPACTION_ENABLED", raising=False)
    old = _register_alias()
    model = _ScriptedModel(
        [
            {"text": "그대로", "tool_calls": None, "finish_reason": "stop", "usage": _usage(1, 1)},
            {"text": "강제", "tool_calls": None, "finish_reason": "stop", "usage": _usage(1, 1)},
        ]
    )
    try:
        app = _foundry_only_app(foundry_chat_client_factory=lambda: model)
        with TestClient(app) as client:
            disabled = _post(client, {AFFINITY_HEADER: "conv-a"})
            assert disabled.status_code == 200
            assert model.calls[0]["messages"][2]["content"] == _listing()
            monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "true")
            forced = _post(
                client,
                {AFFINITY_HEADER: "conv-a"},
                tool_choice={"type": "function", "function": {"name": "terminal"}},
            )
            assert forced.status_code == 200
            assert model.calls[1]["messages"][2]["content"] == _listing()
            assert HIDE_TOOL not in {tool["function"]["name"] for tool in model.calls[1]["tools"]}
    finally:
        _restore_alias(old)


def test_upstream_failure_after_internal_call_preserves_cause():
    plan, _reason = _plan()
    assert plan is not None

    class FailingSecond(_ScriptedModel):
        async def generate(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {"text": None, "tool_calls": [_hide_tool_call()], "finish_reason": "tool_calls", "usage": _usage(3, 1)}
            raise VertexAPIError(503, "upstream unavailable", code="unavailable")

    model = FailingSecond([])
    with pytest.raises(CompactionUpstreamError) as raised:
        asyncio.run(
            run_turn(
                generate=model.generate,
                base_kwargs={},
                messages=_messages(),
                plan=plan,
                store=MemoryContextStore(min_chars=800),
                settings=_settings(),
            )
        )
    assert isinstance(raised.value.__cause__, VertexAPIError)
    assert raised.value.usages[0]["prompt_tokens"] == 3


def test_raw_headers_remain_distinct_and_blank_headers_skip():
    a, _ = _plan(headers={AFFINITY_HEADER: "conv-a"})
    b, _ = _plan(headers={AFFINITY_HEADER: " conv-a "})
    assert a is not None and b is not None
    assert a.affinity_key != b.affinity_key
    for value in ("", "   ", "\r\n"):
        assert _plan(headers={AFFINITY_HEADER: value})[0] is None


def test_shared_header_list_restore_and_reused_call_id():
    from openai_compatible_bridge.context_compaction import _list_call

    store = MemoryContextStore()
    first = _listing()
    second = first.replace("module_", "changed_")
    a = store.compact(affinity="shared", tool_call_id="call_17", original=first, tool_name="terminal").item
    b = store.compact(affinity="shared", tool_call_id="call_17", original=second, tool_name="terminal").item
    assert a is not None and b is not None and a.item_id != b.item_id
    assert {i["item_id"] for i in _list_call({}, "shared", store)["items"]} == {a.item_id, b.item_id}
    assert _list_call({}, "other", store)["items"] == []
    assert store.get("other", a.item_id) is None
    assert store.unhide("other", a.item_id).error == "not_found"
    assert store.visible_content("other", "call_17", first) == first
    assert store.visible_content("shared", "call_17", first) == a.compacted
    assert store.visible_content("shared", "call_17", second) == b.compacted
    assert store.visible_content("shared", "call_17", "unrelated") == "unrelated"
    assert store.unhide("shared", a.item_id).ok
    # Restore the stored original, not just an original supplied by the caller.
    assert store.visible_content("shared", "call_17", a.compacted) == first
    assert store.visible_content("shared", "call_17", second) == b.compacted
    assert [i["item_id"] for i in _list_call({}, "shared", store)["items"]] == [b.item_id]
    restored = apply_visibility(_messages(a.compacted), affinity="shared", store=store)
    assert restored[2]["content"] == first
    assert restored[1] == _messages()[1]


def test_global_byte_budget_preserves_existing_originals_until_expiry():
    clock = {"now": 0.0}
    probe = MemoryContextStore(clock=lambda: clock["now"])
    original = _listing()
    a = probe.compact(affinity="a", tool_call_id="call", original=original, tool_name="terminal")
    assert a.ok
    budget = probe.used_bytes
    assert budget >= len(original.encode())
    store = MemoryContextStore(max_bytes=budget, ttl_seconds=10, clock=lambda: clock["now"])
    first = store.compact(affinity="a", tool_call_id="call", original=original, tool_name="terminal").item
    assert first is not None
    # One global budget, not a new allowance for each header. No LRU eviction.
    for header in ("a", "b", "c"):
        refused = store.compact(affinity=header, tool_call_id="another", original=original, tool_name="terminal")
        assert refused.error == "store_full"
    assert store.get("a", first.item_id) == first
    assert store.used_bytes == budget
    clock["now"] = 8
    assert store.unhide("a", first.item_id).ok
    assert store.compact(affinity="a", tool_call_id="call", original=original, tool_name="terminal").ok
    current = store.get("a", first.item_id)
    assert current is not None and current.expires_at == 10
    clock["now"] = 10
    # Global expiry does not depend on revisiting header a.
    assert store.expire() == 1
    assert store.used_bytes == 0
    assert store.unhide("a", first.item_id).error == "not_found"
    assert store.compact(affinity="b", tool_call_id="call", original=original, tool_name="terminal").ok


def test_capacity_is_atomic_across_headers_and_utf8_payloads():
    from concurrent.futures import ThreadPoolExecutor
    original = _listing().replace("src", "한글")
    probe = MemoryContextStore()
    probe.compact(affinity="a", tool_call_id="call", original=original, tool_name="terminal")
    budget = probe.used_bytes
    store = MemoryContextStore(max_bytes=budget)
    barrier = threading.Barrier(8)
    def save(index):
        barrier.wait()
        return store.compact(affinity=str(index), tool_call_id="call", original=original, tool_name="terminal")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(save, range(8)))
    assert sum(result.ok for result in results) == 1
    assert store.used_bytes == budget
    empty = MemoryContextStore(max_bytes=1)
    for i in range(50):
        assert empty.compact(affinity=str(i), tool_call_id="call", original=original, tool_name="terminal").error == "store_full"
    assert empty.used_bytes == 0
    assert len(empty._items) == 0  # No empty header groups retained on refusals.


def test_http_capacity_skip_keeps_original_and_returns_answer(monkeypatch):
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_MAX_BYTES", "1")
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", None)
    old = _register_alias()
    model = _ScriptedModel([
        {"text": None, "tool_calls": [_hide_tool_call()], "usage": _usage(1, 1)},
        {"text": "answer", "tool_calls": None, "usage": _usage(1, 1)},
    ])
    try:
        with TestClient(_foundry_only_app(foundry_chat_client_factory=lambda: model)) as client:
            response = _post(client, {AFFINITY_HEADER: "a"})
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "answer"
        assert model.calls[1]["messages"][2]["content"] == _listing()
        assert json.loads(model.calls[1]["messages"][-1]["content"])["error"] == "store_full"
    finally:
        _restore_alias(old)


def test_periodic_expiry_releases_idle_items(monkeypatch):
    import openai_compatible_bridge.context_compaction as module
    clock = {"now": 0.0}
    store = MemoryContextStore(ttl_seconds=1, clock=lambda: clock["now"])
    store.compact(affinity="idle", tool_call_id="call", original=_listing(), tool_name="terminal")
    calls = []
    real_expire = store.expire
    def observed_expire():
        calls.append(real_expire())
    async def fake_sleep(interval):
        assert interval == 60
        if calls:
            raise asyncio.CancelledError
        clock["now"] = 2
    monkeypatch.setattr(store, "expire", observed_expire)
    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(module.expire_context_periodically(store))
    assert calls == [1]
    assert store.used_bytes == 0


def test_app_owns_and_cancels_expiry_task():
    model = _ScriptedModel([])
    app = _foundry_only_app(foundry_chat_client_factory=lambda: model)
    with TestClient(app):
        task = app.state.context_compaction_expiry_task
        assert not task.done()
    assert task.cancelled()


def test_internal_list_and_unhide_roundtrip_uses_stored_original():
    store = MemoryContextStore()
    original = _listing()
    item = store.compact(affinity="conv-a", tool_call_id="call_17", original=original, tool_name="terminal").item
    assert item is not None
    def call(name, args, ident):
        return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    model = _ScriptedModel([
        {"tool_calls": [call("list_context_items", {}, "list")], "usage": _usage(1, 1)},
        {"tool_calls": [call("unhide_context", {"item_id": item.item_id}, "restore")], "usage": _usage(1, 1)},
        {"text": "done", "tool_calls": None, "usage": _usage(1, 1)},
    ])
    plan, _ = _plan()
    assert plan is not None
    messages = _messages()
    outcome = asyncio.run(run_turn(generate=model.generate, base_kwargs={}, messages=messages,
                                  plan=plan, store=store, settings=_settings()))
    assert outcome.result is not None
    assert outcome.result["text"] == "done"
    assert model.calls[0]["messages"][2]["content"] == item.compacted
    listed = json.loads(model.calls[1]["messages"][-1]["content"])["items"]
    assert [row["item_id"] for row in listed] == [item.item_id]
    assert model.calls[2]["messages"][2]["content"] == original
    assert messages == _messages()
    assert outcome.result["tool_calls"] is None


def test_global_budget_setting_and_invalid_values():
    default = load_settings({}).max_bytes
    assert default > 0
    assert load_settings({"CONTEXT_COMPACTION_MAX_BYTES": "12345"}).max_bytes == 12345
    for value in ("0", "-1", "bad", ""):
        assert load_settings({"CONTEXT_COMPACTION_MAX_BYTES": value}).max_bytes == default
    with pytest.raises(ValueError):
        MemoryContextStore(max_bytes=0)
    with pytest.raises(ValueError):
        MemoryContextStore(ttl_seconds=0)
