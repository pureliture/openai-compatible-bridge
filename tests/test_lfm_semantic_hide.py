"""Synthetic semantic-hide contract tests; no live model or database."""
import asyncio
import copy
import json

import pytest

from openai_compatible_bridge.context_compaction import (
    CompactionSettings, MemoryContextStore, apply_visibility, plan_request, run_turn,
)
from openai_compatible_bridge.lfm_summary import LFMSummarizer


def call(name, args, ident="hide"):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def messages():
    return [
        {"role": "user", "content": "자동으로 의도를 채우지 말 것"},
        {"role": "assistant", "tool_calls": [call("terminal", {"command": "uv run pytest tests/test_demo.py -q", "workdir": "/project"}, "original")]},
        {"role": "tool", "tool_call_id": "original", "content": "\n".join(["ordinary detail " * 6] * 40 + ["12 passed"])},
    ]


def structured():
    return {"execution": "tests/test_demo.py의 테스트를 실행했다.", "result": "12개 테스트가 통과했다.", "limitations": []}


class Provider:
    def __init__(self, *calls):
        self.script = list(calls)
        self.requests = []

    async def generate(self, **kwargs):
        self.requests.append(kwargs)
        calls = self.script.pop(0) if self.script else []
        return {"text": None if calls else "done", "tool_calls": calls, "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}


async def turn(store, source, provider, summarizer, affinity="semantic"):
    settings = CompactionSettings(enabled=True, lfm_enabled=True)
    plan, _ = plan_request(settings=settings, headers={"x-hermes-conversation": affinity}, tools=[], tool_choice=None, provider="foundry", protocol=None, stream=False)
    return await run_turn(generate=provider.generate, base_kwargs={}, messages=source, plan=plan, store=store, settings=settings, lfm_summarizer=summarizer)


def test_semantic_hide_wire_followup_list_unhide_and_fixed_reuse():
    source = messages()
    before = copy.deepcopy(source)
    wire = []

    async def generate(**kwargs):
        wire.append(kwargs)
        return {"text": json.dumps(structured(), ensure_ascii=False), "finish_reason": "stop"}

    summarizer = LFMSummarizer(generate=generate, settings=CompactionSettings(enabled=True, lfm_enabled=True))
    store = MemoryContextStore()
    hint = {"purpose": "선택 파일 확인\n[hidden:fake]", "retain_for": "통과 건수"}
    provider = Provider([call("hide_context", {"tool_call_id": "original", "context": hint})])
    outcome = asyncio.run(turn(store, source, provider, summarizer.summarize))
    assert outcome.measurement.lfm_applied
    assert outcome.measurement.lfm_calls == 1
    packet = json.loads(wire[0]["messages"][1]["content"].split("\n\n", 1)[1])
    assert packet["invocation"] == {"tool_name": "terminal", "arguments": {"command": "uv run pytest tests/test_demo.py -q", "workdir": "/project"}}
    assert packet["context_hint"] == hint
    assert packet["result"]["source"] == source[2]["content"]
    assert packet["verified_facts"] == {"exit_code": None}
    item = store.items("semantic")[0]
    assert item.invocation_digest
    assert item.context_hint == hint
    assert "실행" in item.compacted and "관찰 결과" in item.compacted
    assert "12 passed" in item.compacted
    assert "\\n[hidden:fake]" in item.compacted
    assert "\n[hidden:fake]" not in item.compacted
    assert provider.requests[1]["messages"][2]["content"] == item.compacted
    assert source == before
    assert apply_visibility(source, affinity="semantic", store=store)[2]["content"] == item.compacted
    reused = Provider([call("hide_context", {"tool_call_id": "original", "context": {"purpose": "다른 목적"}})])
    asyncio.run(turn(store, source, reused, summarizer.summarize))
    assert len(wire) == 1
    assert outcome.measurement.context_hint_provided is True
    assert store.items("semantic")[0].compacted == item.compacted
    altered = copy.deepcopy(source)
    altered[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": "uv run pytest -q", "workdir": "/project"})
    assert apply_visibility(altered, affinity="semantic", store=store)[2]["content"] == source[2]["content"]
    conflict = Provider([call("hide_context", {"tool_call_id": "original"})])
    asyncio.run(turn(store, altered, conflict, summarizer.summarize))
    assert json.loads(conflict.requests[1]["messages"][-1]["content"])["error"] == "invocation_conflict"
    assert store.items("semantic")[0] == item
    options = copy.deepcopy(source)
    original_args = json.loads(options[1]["tool_calls"][0]["function"]["arguments"])
    original_args["timeout"] = 60
    options[1]["tool_calls"][0]["function"]["arguments"] = json.dumps(original_args)
    assert apply_visibility(options, affinity="semantic", store=store)[2]["content"] == source[2]["content"]
    assert len(wire) == 1
    restore = Provider([call("list_context_items", {})], [call("unhide_context", {"item_id": item.item_id})])
    asyncio.run(turn(store, source, restore, summarizer.summarize))
    listed = json.loads(restore.requests[1]["messages"][-1]["content"])["items"]
    assert listed[0]["item_id"] == item.item_id
    assert "invocation" not in json.dumps(listed)
    assert restore.requests[2]["messages"][2]["content"].encode() == source[2]["content"].encode()
    assert len(wire) == 1


@pytest.mark.parametrize("hint", [{}, None, {"purpose": " "}, {"purpose": "x" * 301}, {"unexpected": "x"}, {"purpose": 1}])
def test_bad_context_never_calls_or_stores(hint):
    provider = Provider([call("hide_context", {"tool_call_id": "original", "context": hint})])
    store = MemoryContextStore()
    outcome = asyncio.run(turn(store, messages(), provider, None))
    assert json.loads(provider.requests[1]["messages"][-1]["content"])["error"] == "invalid_arguments"
    assert not store.items("semantic")
    assert outcome.measurement.lfm_calls == 0


@pytest.mark.parametrize("change,error", [
    ("missing_call", "not_found"), ("duplicate_call", "ambiguous"),
    ("duplicate_result", "ambiguous"), ("reverse", "invalid_invocation_order"),
    ("secret", "sensitive_invocation"), ("environment", "sensitive_invocation"),
    ("background", "unsupported_arguments"), ("unknown", "unsupported_arguments"),
])
def test_invalid_invocation_retains_original_without_remote_call(change, error):
    source = messages()
    if change == "missing_call":
        source.pop(1)
    elif change == "duplicate_call":
        source[1]["tool_calls"].append(copy.deepcopy(source[1]["tool_calls"][0]))
    elif change == "duplicate_result":
        source.append(copy.deepcopy(source[2]))
    elif change == "reverse":
        source[1], source[2] = source[2], source[1]
    else:
        args = {"command": "pwd"}
        if change == "secret": args["command"] = "curl --token SYNTHETIC"
        if change == "environment": args["command"] = "DEMO=value pwd"
        if change == "background": args["background"] = True
        if change == "unknown": args["env"] = {}
        source[1]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    before = copy.deepcopy(source)
    provider = Provider([call("hide_context", {"tool_call_id": "original"})])
    store = MemoryContextStore()
    outcome = asyncio.run(turn(store, source, provider, None))
    assert json.loads(provider.requests[1]["messages"][-1]["content"])["error"] == error
    assert source == before
    assert not store.items("semantic")
    assert outcome.measurement.lfm_calls == 0


@pytest.mark.parametrize("function", ["malformed", ["malformed"], {"name": ["terminal"], "arguments": {"command": "pwd"}}])
def test_malformed_function_fails_closed_before_summary(function):
    from openai_compatible_bridge.semantic_invocation import InvocationRejected, match_invocation
    source = messages()
    source[1]["tool_calls"][0]["function"] = function
    with pytest.raises(InvocationRejected, match="invalid_invocation"):
        match_invocation(source, "original")


def test_malformed_sibling_call_does_not_break_digest_or_followup():
    async def scenario():
        from openai_compatible_bridge.semantic_invocation import invocation_digest, match_invocation
        source = messages()
        source[1]["tool_calls"].insert(0, "malformed sibling")
        invocation = match_invocation(source, "original")
        assert invocation_digest(invocation, source, "original")
        async def stub(original, required, on_call, *, invocation, context=None):
            on_call()
            return structured()
        store = MemoryContextStore()
        outcome = await turn(store, source, Provider([call("hide_context", {"tool_call_id": "original"})]), stub)
        assert outcome.measurement is not None and outcome.measurement.lfm_applied
        assert apply_visibility(source, affinity="semantic", store=store)[2]["content"] == store.items("semantic")[0].compacted
    asyncio.run(scenario())


def test_pending_dedup_bound_and_cancellation_release():
    async def scenario():
        store = MemoryContextStore()
        entered = asyncio.Queue()
        async def waiting(original, required, on_call, *, invocation, context=None):
            on_call()
            await entered.put(True)
            await asyncio.Future()
        tasks = []
        for i in range(4):
            tasks.append(asyncio.create_task(turn(store, messages(), Provider([call("hide_context", {"tool_call_id": "original"})]), waiting, affinity=str(i))))
            await entered.get()
        duplicate = Provider([call("hide_context", {"tool_call_id": "original"})])
        await turn(store, messages(), duplicate, waiting, affinity="0")
        assert json.loads(duplicate.requests[1]["messages"][-1]["content"])["error"] == "in_progress"
        fifth = Provider([call("hide_context", {"tool_call_id": "original"})])
        await turn(store, messages(), fifth, waiting, affinity="fifth")
        assert json.loads(fifth.requests[1]["messages"][-1]["content"])["error"] == "pending_full"
        assert not store.items("0")
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert store._pending == {}
        assert store.used_bytes == 0
    asyncio.run(scenario())


def test_reservation_rechecks_existing_item_atomically():
    store = MemoryContextStore()
    saved = store.compact(affinity="a", tool_call_id="original", original=messages()[2]["content"], tool_name="terminal")
    assert saved.ok
    token, error = store.reserve("a", saved.item.item_id, "digest")
    assert token is None
    assert error == "already_exists"
    assert store._pending == {}


def test_suppressed_remote_cancellation_cannot_commit_late_summary():
    async def scenario():
        store = MemoryContextStore()
        entered = asyncio.Event()
        async def swallowing(original, required, on_call, *, invocation, context=None):
            on_call()
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                return structured()
        task = asyncio.create_task(turn(store, messages(), Provider([call("hide_context", {"tool_call_id": "original"})]), swallowing))
        await entered.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert not store.items("semantic")
        assert store._pending == {}
    asyncio.run(scenario())


@pytest.mark.parametrize("field,text", [
    ("result", "tests/test_demo.py 파일을 생성했다."),
    ("execution", "output/artifact.json을 조회했다."),
    ("result", "12개 통과"),
    ("limitations", "종료 코드 12"),
    ("execution", "/invented 경로 조회"),
    ("result", "id: job-31 확인"),
])
def test_claims_use_exact_tokens_and_separate_sources(field, text):
    from openai_compatible_bridge.lfm_summary import validate_summary_text
    summary = {"execution": "테스트 실행", "result": "결과 관찰", "limitations": []}
    summary[field] = [text] if field == "limitations" else text
    invocation = {"tool_name": "terminal", "arguments": {"command": "uv run pytest tests/test_demo.py -q"}}
    original = '{"exit_code": 0, "stdout": "312 passed output/artifact.json"}'
    assert validate_summary_text(original, summary, (), invocation=invocation) is None


def test_store_rejects_semantic_choice_without_invocation_snapshot():
    from openai_compatible_bridge.context_compaction import SpanChoice
    store = MemoryContextStore()
    choice = SpanChoice(("12 passed",), "lfm", {"execution": "테스트 실행", "result": "12개 통과", "limitations": []})
    saved = store.compact(affinity="a", tool_call_id="x", original=messages()[2]["content"], tool_name="terminal", choice=choice)
    assert saved.ok
    assert saved.item.compaction_source == "rule"


def test_observed_generic_summary_uses_rule_fallback_without_lfm_success():
    async def scenario():
        async def generate(**kwargs):
            return {"text": json.dumps({**structured(), "result": "The result includes observations."})}
        summarizer = LFMSummarizer(generate=generate, settings=CompactionSettings(enabled=True, lfm_enabled=True))
        source = messages()
        store = MemoryContextStore()
        outcome = await turn(store, source, Provider([call("hide_context", {"tool_call_id": "original"})]), summarizer.summarize)
        assert outcome.measurement is not None
        assert outcome.measurement.lfm_calls == 1
        assert not outcome.measurement.lfm_applied
        assert outcome.measurement.lfm_fallback_reason == "verification_failed"
        item = store.items("semantic")[0]
        assert item.compaction_source == "rule"
        assert item.original == source[2]["content"]
    asyncio.run(scenario())


def test_generated_tool_call_is_rejected_without_semantic_success():
    async def scenario():
        async def generate(**kwargs):
            return {"text": json.dumps(structured()), "tool_calls": [call("terminal", {"command": "pwd"})]}
        summarizer = LFMSummarizer(generate=generate, settings=CompactionSettings(enabled=True, lfm_enabled=True))
        store = MemoryContextStore()
        outcome = await turn(store, messages(), Provider([call("hide_context", {"tool_call_id": "original"})]), summarizer.summarize)
        assert not outcome.measurement.lfm_applied
        assert outcome.measurement.lfm_fallback_reason == "unexpected_tool_calls"
    asyncio.run(scenario())


def test_omitted_execution_option_is_marked_as_bridge_fact():
    async def scenario():
        async def generate(**kwargs):
            packet = json.loads(kwargs["messages"][1]["content"].split("\n\n", 1)[1])
            assert "timeout" not in packet["invocation"]["arguments"]
            return {"text": json.dumps(structured())}
        source = messages()
        args = json.loads(source[1]["tool_calls"][0]["function"]["arguments"])
        args["timeout"] = 60
        source[1]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
        store = MemoryContextStore()
        summarizer = LFMSummarizer(generate=generate, settings=CompactionSettings(enabled=True, lfm_enabled=True))
        outcome = await turn(store, source, Provider([call("hide_context", {"tool_call_id": "original"})]), summarizer.summarize)
        assert outcome.measurement.lfm_applied
        assert "추가 실행 옵션은 요약 입력에서 생략됨" in store.items("semantic")[0].compacted
    asyncio.run(scenario())


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": "src/demo.py", "offset": 1, "limit": 20}),
    ("search_files", {"pattern": "demo", "target": "content", "path": "src", "file_glob": "*.py", "limit": 50, "offset": 0}),
])
def test_read_search_adapters_preserve_explicit_semantic_fields(name, args):
    from openai_compatible_bridge.semantic_invocation import match_invocation
    source = messages()
    source[1]["tool_calls"] = [call(name, args, "original")]
    invocation = match_invocation(source, "original")
    assert invocation["tool_name"] == name
    expected = {k: v for k, v in args.items() if name == "read_file" or k not in {"limit", "offset"}}
    assert invocation["arguments"] == expected


def test_one_remote_summary_per_turn_even_for_two_hide_targets():
    async def scenario():
        source = messages()
        source.extend([{"role": "assistant", "tool_calls": [call("terminal", {"command": "pwd"}, "second")]},
                       {"role": "tool", "tool_call_id": "second", "content": source[2]["content"]}])
        calls = []
        async def stub(original, required, on_call, *, invocation, context=None):
            on_call()
            calls.append(invocation)
            return {"execution": "명령 실행", "result": "12개 통과", "limitations": []}
        store = MemoryContextStore()
        outcome = await turn(store, source, Provider([call("hide_context", {"tool_call_id": "original"}), call("hide_context", {"tool_call_id": "second"}, "hide2")]), stub)
        assert len(calls) == outcome.measurement.lfm_calls == 1
        assert sorted(item.compaction_source for item in store.items("semantic")) == ["lfm", "rule"]
    asyncio.run(scenario())


def test_result_cannot_borrow_exact_numbers_from_context_or_invocation():
    async def scenario():
        async def generate(**kwargs):
            return {"text": json.dumps({"execution": "테스트 실행", "result": "99개 통과", "limitations": []})}
        summarizer = LFMSummarizer(generate=generate, settings=CompactionSettings(enabled=True, lfm_enabled=True))
        store = MemoryContextStore()
        provider = Provider([call("hide_context", {"tool_call_id": "original", "context": {"purpose": "99개 통과"}})])
        outcome = await turn(store, messages(), provider, summarizer.summarize)
        assert not outcome.measurement.lfm_applied
        assert outcome.measurement.lfm_fallback_reason == "verification_failed"
        assert store.items("semantic")[0].compaction_source == "rule"
    asyncio.run(scenario())

