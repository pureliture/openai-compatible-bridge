"""Result-only contract; all examples are newly authored synthetic data."""
from tests.test_lfm_summary import result_packet

import asyncio
import copy
import json
import os

from openai_compatible_bridge.context_compaction import (
    CompactionSettings, MemoryContextStore, apply_visibility, internal_tool_definitions,
)
import pytest

from openai_compatible_bridge.lfm_summary import LFMSummarizer, validate_summary_text
from tests.test_lfm_semantic_hide import Provider, call, messages, turn


def test_result_only_hide_ignores_compatibility_context_through_restore():
    source = messages()
    before = copy.deepcopy(source)
    requests = []
    hint = {"purpose": "HINT_ONLY_99", "retain_for": "RETAIN_ONLY"}

    async def generate(**kwargs):
        requests.append(kwargs)
        return {"text": json.dumps({"summary": "12 tests passed."})}

    summarizer = LFMSummarizer(generate=generate, settings=CompactionSettings(enabled=True, lfm_enabled=True))
    store = MemoryContextStore()
    provider = Provider([call("hide_context", {"tool_call_id": "original", "context": hint})])
    outcome = asyncio.run(turn(store, source, provider, summarizer.summarize))
    assert outcome.measurement.lfm_applied
    packet = result_packet(requests[0]["messages"][1]["content"])
    assert set(packet) == {"result", "required_evidence"}
    assert packet["result"] == {"source": source[2]["content"]}
    assert packet["required_evidence"] == ["12 passed"]
    wire = json.dumps(requests)
    assert "HINT_ONLY" not in wire and "RETAIN_ONLY" not in wire
    assert "uv run pytest" not in wire and "context_hint" not in wire
    item = store.items("semantic")[0]
    assert item.context_hint is None
    assert not outcome.measurement.context_hint_provided
    assert "HINT_ONLY" not in item.compacted and "참고 의도" not in item.compacted
    assert provider.requests[1]["messages"][1] == before[1]
    assert source == before
    assert apply_visibility(source, affinity="semantic", store=store)[2]["content"] == item.compacted
    restored = Provider([call("unhide_context", {"item_id": item.item_id})])
    asyncio.run(turn(store, source, restored, summarizer.summarize))
    assert restored.requests[1]["messages"][2]["content"].encode() == before[2]["content"].encode()
    assert restored.requests[1]["messages"][1] == before[1]
    assert len(requests) == 1
    definition = internal_tool_definitions()[0]["function"]
    assert "context" in definition["parameters"]["properties"]
    assert "purpose/retain_for" not in definition["description"]
    assert "ignored" in definition["description"]


@pytest.mark.parametrize("claim", ["Exit code 0.", "exit_code: 1", "종료 코드 0", "Exit code 0; exit code 1."])
@pytest.mark.parametrize("source", [
    '{"exit_code": null, "stdout": "row 0 has 1 entry"}',
    '{"exit_code": "unknown", "stdout": "row 0 has 1 entry"}',
    '{"exit_code": false, "stdout": "row 0 has 1 entry"}',
    'row 0 has 1 entry',
])
def test_unknown_exit_code_cannot_borrow_unrelated_digits(source, claim):
    assert validate_summary_text(source, {"summary": claim}, ()) is None


@pytest.mark.parametrize("claim", [
    'exit code is 0', 'exit code was 0', 'exit code: "0"',
    'return code 0', 'return_code=0', 'exit status 0',
    'Exit code 0; exit code is 1',
])
def test_every_exit_claim_is_checked_even_if_digits_exist(claim):
    assert validate_summary_text('{"exit_code": null, "stdout": "row 0 has 1 entry"}', {"summary": claim}, ()) is None


@pytest.mark.parametrize("claim", ['Exit code 0.', 'exit code is 0', 'return code 0'])
def test_explicit_source_exit_code_can_be_summarized(claim):
    source = '2 tests passed; exit code 0.'
    assert validate_summary_text(source, {"summary": claim}, ()) == {"summary": claim}


def test_all_exit_claims_checked_against_authoritative_metadata():
    source = '{"exit_code": 0, "stdout": "row 1; 2 passed"}'
    assert validate_summary_text(source, {"summary": "Exit code 0; exit code is 1."}, ()) is None


def test_legacy_context_item_can_be_reused_and_restored_without_regeneration():
    from dataclasses import replace
    source = messages()
    store = MemoryContextStore()
    async def stub(original, required, on_call, *, invocation, context=None):
        on_call()
        return {'summary': '12 tests passed.'}
    asyncio.run(turn(store, source, Provider([call('hide_context', {'tool_call_id': 'original'})]), stub))
    item = store.items('semantic')[0]
    legacy = replace(item, context_hint={'purpose': 'legacy purpose'})
    store._items[('semantic', item.item_id)] = legacy
    async def never(*args, **kwargs):
        raise AssertionError('existing item must not regenerate')
    provider = Provider([call('hide_context', {'tool_call_id': 'original', 'context': {'purpose': 'ignored new purpose'}})])
    outcome = asyncio.run(turn(store, source, provider, never))
    assert outcome.measurement.lfm_calls == 0
    assert store.items('semantic')[0] == legacy
    restored = Provider([call('unhide_context', {'item_id': item.item_id})])
    asyncio.run(turn(store, source, restored, never))
    assert restored.requests[1]['messages'][2]['content'] == source[2]['content']


def test_ordinary_annotations_cannot_be_reclassified_as_source_warnings():
    source = ('The orbit-widget catalog uses exact-name matching.\n'
              'Warning: optional descriptions are omitted.\n'
              + 'Repeated ordinary catalog annotation.\n' * 55)
    bad = {'summary': 'The orbit-widget catalog uses exact-name matching, with warnings about omitted optional descriptions and repeated annotations.'}
    assert validate_summary_text(source, bad, ()) is None
    noted = {'summary': 'The orbit-widget catalog uses exact-name matching; descriptions omitted; repeated annotations noted as warnings.'}
    assert validate_summary_text(source, noted, ()) is None
    good = {'summary': 'The orbit-widget catalog uses exact-name matching; optional descriptions are omitted.'}
    assert validate_summary_text(source, good, ()) == good
    separate = {'summary': 'The orbit-widget catalog uses exact-name matching. Warning: optional descriptions are omitted. Repeated annotations are ordinary.'}
    assert validate_summary_text(source, separate, ()) == separate
    genuine_source = source + 'Warning: repeated annotations are invalid.\n'
    assert validate_summary_text(genuine_source, bad, ()) == bad


@pytest.mark.parametrize('source', [
    'Warning: optional descriptions are omitted.',
    '{"stdout": "Warning: optional descriptions are omitted."}',
])
def test_explicit_optional_omission_warning_cannot_be_generalized(source):
    unqualified = {'summary': 'Warning: descriptions are omitted.'}
    assert validate_summary_text(source, unqualified, ()) is None
    qualified = {'summary': 'Warning: optional descriptions are omitted.'}
    assert validate_summary_text(source, qualified, ()) == qualified
    # Unqualified source warnings need no invented qualifier.
    ordinary_source = 'Warning: descriptions are omitted.'
    assert validate_summary_text(ordinary_source, unqualified, ()) == unqualified


# Frozen synthetic facts, authored before the first real generation.
SYNTHETIC_CASES = (
    ("catalog", "The orbit-widget catalog uses exact-name matching.\nWarning: optional descriptions are omitted.",
     ("orbit-widget", "exact", "descriptions", "omitted")),
    ("checks", "2 synthetic checks passed; exit code 0.", ("2", "passed", "exit", "0")),
    ("unknown-exit", "Row 0 lists the amber-widget component.\nThe amber-widget catalog uses exact-name matching.",
     ("amber-widget", "exact")),
)


@pytest.mark.skipif(os.getenv('RUN_LFM_RESULT_ONLY_SMOKE') != '1', reason='opt-in synthetic local LFM smoke')
@pytest.mark.parametrize('repeat', range(3))
@pytest.mark.parametrize('case,facts,required_terms', SYNTHETIC_CASES)
def test_actual_result_only_lfm_content_apply_restore(case, facts, required_terms, repeat):
    from openai_compatible_bridge.context_compaction import RuleSpanSelector, _required_evidence_indexes
    from openai_compatible_bridge.providers.ollama import OllamaChatClient
    source = facts + '\n' + '\n'.join(['Repeated ordinary catalog annotation.'] * 55)
    invocation = {'tool_name': 'terminal', 'arguments': {'command': 'python synthetic_probe.py', 'timeout': 60}}
    transcript = [
        {'role': 'user', 'content': 'USER_ONLY_NOT_FOR_LFM'},
        {'role': 'assistant', 'tool_calls': [call('terminal', invocation['arguments'], 'synthetic')]},
        {'role': 'tool', 'tool_call_id': 'synthetic', 'content': source},
    ]
    before = copy.deepcopy(transcript)
    lines = source.splitlines()
    evidence = tuple(lines[i] for i in _required_evidence_indexes(lines))
    assert RuleSpanSelector().select(source) is not None
    settings = CompactionSettings(enabled=True, lfm_enabled=True)
    generated = []
    async def exercise():
        client = OllamaChatClient(base_url='http://127.0.0.1:11434')
        async def capture(**kwargs):
            assert kwargs['model'] == 'lfm2.5-thinking:latest'
            assert kwargs['max_tokens'] == 384 and kwargs['timeout_seconds'] == 60
            assert len(''.join(m['content'] for m in kwargs['messages']).encode()) <= 12288
            packet = result_packet(kwargs['messages'][1]['content'])
            assert set(packet) == {'result', 'required_evidence'}
            wire = json.dumps(kwargs['messages'])
            assert not any(private in wire for private in (
                'USER_ONLY_NOT_FOR_LFM', 'HINT_ONLY_99', 'synthetic_probe.py',
            ))
            result = await client.generate(**kwargs)
            generated.append(result)
            print('ACTUAL_LFM_RESULT', json.dumps({'case': case, 'repeat': repeat, 'response': result}, ensure_ascii=False))
            return result
        try:
            summarizer = LFMSummarizer(generate=capture, settings=settings)
            store = MemoryContextStore()
            provider = Provider([call('hide_context', {'tool_call_id': 'synthetic', 'context': {'purpose': 'HINT_ONLY_99'}})])
            outcome = await turn(store, transcript, provider, summarizer.summarize)
            assert outcome.measurement.lfm_applied, outcome.measurement.lfm_fallback_reason
            assert len(generated) == 1
            result = json.loads(generated[0]['text'])
            text = result['summary'].lower()
            assert all(term in text for term in required_terms), text
            assert not any(term in text for term in ('next step', 'you should', 'evaluation passed', '99', 'synthetic_probe.py'))
            if case == 'catalog':
                import re
                # Independent relationship check, beyond the frozen keyword anchors.
                assert re.search(r'\boptional\s+descriptions\b', text), text
                assert not any(phrase in text for phrase in (
                    'main finding', 'no additional details', 'no further details',
                )), text
                assert not re.search(r'\bwarnings?\b[^.;!?]*\bannotations?\b', text), text
                assert not re.search(r'\bannotations?\b[^.;!?]*\b(?:are|as)\s+warnings?\b', text), text
            if case == 'unknown-exit':
                assert not any(term in text for term in ('exit', 'return code', 'success', 'failed'))
            assert outcome.measurement.lfm_calls == 1
            item = store.items('semantic')[0]
            assert item.compaction_source == 'lfm' and item.context_hint is None
            assert len(item.compacted.encode()) < len(source.encode()) * .8
            assert provider.requests[1]['messages'][1] == before[1]
            assert provider.requests[1]['messages'][2]['content'] == item.compacted
            restore = Provider([call('unhide_context', {'item_id': item.item_id})])
            await turn(store, transcript, restore, summarizer.summarize)
            assert len(generated) == 1
            assert restore.requests[1]['messages'][2]['content'].encode() == source.encode()
            assert transcript == before
            print('ACTUAL_LFM_CONTENT_PASS', case, repeat, json.dumps(result), 'exact_restore=true')
        finally:
            await client.close()
    asyncio.run(exercise())
