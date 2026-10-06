"""Synthetic native HTTP final-review regressions; no paid traffic or transcripts."""
import asyncio
import copy
import json
import re
import time
from dataclasses import replace
from typing import Any

import httpx
import pytest

from openai_compatible_bridge.core.cost_tracking import DisabledCostAccounting, NormalizedUsage
from openai_compatible_bridge.main import _collect_compaction_stream, _log_compaction_upstream_failure
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError
from test_compaction_chat_stream import wire
from test_metered_http import Accounting, ByteStream, wrap
from fastapi.testclient import TestClient
import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.context_compaction import (
    AFFINITY_HEADER,
    HIDE_TOOL,
    CompactionUpstreamError,
    ContextItem,
    MemoryContextStore,
    Scope,
    bridge_scope,
)
from openai_compatible_bridge.main import create_app
from test_compaction_foundry_protocols import MESSAGES, TOOL, SyntheticLFM, Unused
from test_compaction_chat_stream import call as chat_call, sse
from test_compaction_responses_stream import wire as responses_wire
from test_compaction_foundry_protocols import native_call as responses_call
from test_compaction_anthropic_stream import wire as anthropic_wire
from test_compaction_anthropic import native_call as anthropic_call
from test_compaction_google_stream import wire as google_wire
from test_compaction_google import native_call as google_call, native_response as google_response


class _SyntheticCompactionStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aiter__(self):
        for event in self.events:
            yield event

    async def aclose(self):
        self.closed = True


class _SyntheticCompactionClient:
    def __init__(self, stream):
        self.stream = stream

    def stream_chat(self, **kwargs):
        assert kwargs.pop("_require_complete") is True
        return self.stream


def _collect_synthetic_stream(events, *, max_tokens=1):
    stream = _SyntheticCompactionStream(events)
    client = _SyntheticCompactionClient(stream)

    async def run():
        ctx = DisabledCostAccounting().reservation(
            endpoint="chat", model="synthetic", provider="foundry", forecast_usage=NormalizedUsage(),
        )
        async with ctx:
            try:
                result = await _collect_compaction_stream(
                    client, ctx, model="synthetic", messages=[], max_tokens=max_tokens,
                )
                return result, None
            except Exception as error:
                return None, error

    result, error = asyncio.run(run())
    return result, error, stream


def _scope_item():
    return ContextItem(
        item_id="item_scope_review",
        tool_call_id="call_scope_review",
        content_sha256="synthetic-digest",
        original="synthetic original",
        compacted="synthetic replacement",
        excerpt_lines=("synthetic excerpt",),
        visibility="compacted",
        version=1,
        expires_at=time.monotonic() + 120,
    )


@pytest.mark.parametrize('complete', [False, True])
def test_chat_terminal_validation_controls_http_usage(complete):
    async def run():
        accounting = Accounting()
        data = wire(content='synthetic final')
        if not complete:
            data = data.removesuffix(b'data: [DONE]\n\n')
        stream = ByteStream(data)
        client = await wrap(FoundryChatClient(base_url='https://foundry.example/openai/v1/chat/completions',
                                             token='synthetic'), accounting,
                            lambda request: httpx.Response(200, stream=stream), 'foundry')
        ctx = DisabledCostAccounting().reservation(endpoint='chat', model='synthetic', provider='foundry',
                                                   forecast_usage=NormalizedUsage())
        try:
            async with ctx:
                kwargs = dict(model='synthetic', messages=[{'role': 'user', 'content': 'synthetic'}],
                              max_tokens=1, temperature=None, top_p=None, stop=None, response_format=None)
                if complete:
                    result = await _collect_compaction_stream(client, ctx, **kwargs)
                    assert result['text'] == 'synthetic final'
                else:
                    with pytest.raises(VertexAPIError) as caught:
                        await _collect_compaction_stream(client, ctx, **kwargs)
                    assert caught.value.code == 'incomplete_stream'
            assert len(accounting.records) == 1
            assert accounting.records[0][1] == (NormalizedUsage(prompt_tokens=10, completion_tokens=2,
                                                               total_tokens=12) if complete else None)
            assert stream.closed
        finally:
            await client.close()
    asyncio.run(run())

PROTOCOLS = ['openai_chat_completions', 'openai_responses', 'anthropic_messages',
             'google_generate_content', 'xai_responses']
PRIVATE_MESSAGE = 'hide_context arguments: PRIVATE_MARKER'


@pytest.mark.parametrize(('stage', 'expected'), [
    ('provider_read', 'provider_read'),
    ('PRIVATE_MARKER', 'unknown'),
])
def test_private_stream_stage_diagnostic_is_safe_allowlisted(caplog, stage, expected):
    error = VertexAPIError(504, 'PRIVATE_MARKER', code='timeout', stage=stage)
    failure = CompactionUpstreamError([])
    with caplog.at_level('WARNING', logger='context_compaction'):
        try:
            raise failure from error
        except CompactionUpstreamError as caught:
            _log_compaction_upstream_failure(
                caught, provider='foundry', protocol='openai_responses', stream=True,
                started_at=time.monotonic(),
            )
    diagnostic = next(record.getMessage() for record in caplog.records
                      if record.name == 'context_compaction' and 'upstream_failed' in record.getMessage())
    assert f'stage={expected}' in diagnostic
    assert 'PRIVATE_MARKER' not in diagnostic


@pytest.fixture
def review_bridge(monkeypatch):
    monkeypatch.setenv('CONTEXT_COMPACTION_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LFM_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LAYA_ENABLED', 'false')
    monkeypatch.setattr('openai_compatible_bridge.main.BRIDGE_API_KEY', '')
    alias = 'foundry:synthetic-review'
    config = {'provider': 'foundry', 'kind': 'chat', 'provider_model': 'synthetic',
              'protocol': 'openai_chat_completions'}
    monkeypatch.setitem(vertex.MODEL_REGISTRY, alias, config)
    replies, requests, streams = [], [], []
    def handler(request):
        requests.append(json.loads(request.content))
        reply = replies.pop(0)
        if isinstance(reply, tuple):
            status, payload = reply
            return httpx.Response(status, json=payload)
        if isinstance(reply, dict):
            return httpx.Response(200, json=reply)
        stream = reply if isinstance(reply, httpx.AsyncByteStream) else ByteStream(reply)
        streams.append(stream)
        return httpx.Response(200, stream=stream, headers={'content-type': 'text/event-stream'})
    provider = FoundryChatClient(base_url='https://foundry.example/api/v2/llm/proxy/openai/v1/chat/completions',
                                token='synthetic')
    asyncio.run(provider.http.aclose())
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    lfm = SyntheticLFM()
    app = create_app(embedding_client_factory=Unused, chat_client_factory=Unused,
                     rerank_client_factory=Unused, ollama_chat_client_factory=lambda: lfm,
                     foundry_chat_client_factory=lambda: provider, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        yield client, config, replies, requests, streams, lfm, app, alias


def review_post(bridge, *, stream=True, active=True, **kwargs):
    client, _, _, _, _, _, _, alias = bridge
    return client.post('/v1/chat/completions', headers={AFFINITY_HEADER: 'synthetic-review'} if active else {},
                       json={'model': alias, 'messages': copy.deepcopy(MESSAGES), 'tools': [TOOL],
                             'stream': stream, **kwargs})


def private_wire(protocol):
    args = {'tool_call_id': 'original-call'}
    if protocol == 'openai_chat_completions':
        return wire(chat_call(HIDE_TOOL, args))
    if protocol == 'anthropic_messages':
        return anthropic_wire(anthropic_call(HIDE_TOOL, args))
    if protocol == 'google_generate_content':
        return google_wire(google_call(HIDE_TOOL, args))
    return responses_wire(responses_call(HIDE_TOOL, args))


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('safe_code', [False, True])
def test_private_native_stream_error_is_redacted(review_bridge, protocol, safe_code, caplog):
    _, config, replies, requests, streams, lfm, app, _ = review_bridge
    config['protocol'] = protocol
    code = 'timeout' if safe_code else 'PRIVATE_MARKER'
    replies.extend([private_wire(protocol), sse({'type': 'error',
                    'error': {'message': PRIVATE_MESSAGE, 'code': code, 'type': code}})])
    response = review_post(review_bridge)
    assert 'PRIVATE_MARKER' not in response.text and HIDE_TOOL not in response.text
    error = json.loads(response.text.split('data: ', 1)[1].split('\n\n', 1)[0])['error']
    assert error['code'] == ('timeout' if safe_code else 'upstream_error')
    assert response.text.endswith('data: [DONE]\n\n')
    assert len(requests) == 2 and len(lfm.calls) == 1 and all(s.closed for s in streams)
    assert app.state.context_compaction_store.items('synthetic-review')[0].original == MESSAGES[2]['content']
    diagnostic = [record.getMessage() for record in caplog.records
                  if record.name == 'context_compaction' and 'upstream_failed' in record.getMessage()]
    assert len(diagnostic) == 1
    line = diagnostic[0]
    assert f'protocol={protocol}' in line and 'provider=foundry' in line
    assert 'stream=true' in line and 'round=2' in line
    assert f'code={"timeout" if safe_code else "upstream_error"}' in line
    assert re.search(r'status=\d+(?:\s|$)', line)
    assert re.search(r'correlation_id=[0-9a-f]{16}(?:\s|$)', line)
    assert re.search(r'elapsed_ms=\d+(?:\s|$)', line)
    assert 'PRIVATE_MARKER' not in line and PRIVATE_MESSAGE not in line


def test_private_read_timeout_logs_only_safe_stage(review_bridge, caplog):
    _, config, replies, requests, streams, lfm, app, _ = review_bridge
    config['protocol'] = 'openai_responses'

    class ReadTimeoutStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            raise httpx.ReadTimeout('SYNTHETIC_PRIVATE_READ_DETAIL')
            yield b''  # pragma: no cover - makes this an async generator

        async def aclose(self):
            self.closed = True

    replies.extend([private_wire('openai_responses'), ReadTimeoutStream()])
    with caplog.at_level('WARNING', logger='context_compaction'):
        response = review_post(review_bridge)

    assert response.status_code == 200
    assert 'SYNTHETIC_PRIVATE_READ_DETAIL' not in response.text
    assert PRIVATE_MESSAGE not in response.text and HIDE_TOOL not in response.text
    assert len(requests) == 2 and len(lfm.calls) == 1 and len(streams) == 2
    assert all(stream.closed for stream in streams)
    assert app.state.context_compaction_store.items('synthetic-review')[0].original == MESSAGES[2]['content']
    diagnostics = [record.getMessage() for record in caplog.records
                   if record.name == 'context_compaction' and 'upstream_failed' in record.getMessage()]
    assert len(diagnostics) == 1
    assert 'status=504' in diagnostics[0] and 'code=timeout' in diagnostics[0]
    assert 'stage=provider_read' in diagnostics[0]
    assert 'SYNTHETIC_PRIVATE_READ_DETAIL' not in diagnostics[0]


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('active', [False, True])
def test_private_http_error_preserves_status_without_payload(review_bridge, protocol, stream, active):
    _, config, replies, _, _, _, _, _ = review_bridge
    config['protocol'] = protocol
    replies.append((429, {'error': {'message': PRIVATE_MESSAGE, 'code': 'rate_limit_exceeded'}}))
    response = review_post(review_bridge, stream=stream, active=active)
    if active:
        assert 'PRIVATE_MARKER' not in response.text and HIDE_TOOL not in response.text
    else:
        assert PRIVATE_MESSAGE in response.text
    if stream:
        error = json.loads(response.text.split('data: ', 1)[1].split('\n\n', 1)[0])['error']
    else:
        assert response.status_code == 429
        error = response.json()['error']
    assert error['code'] == 'rate_limit_exceeded'
    assert error['type'] == 'rate_limit_error'


@pytest.mark.parametrize('case', ['signature', 'aggregate', 'utf8', 'boundary', 'args', 'valid', 'larger_budget'])
def test_google_nonstream_native_parts_bounded_before_continuation(review_bridge, case):
    _, config, replies, requests, _, lfm, app, _ = review_bridge
    config['protocol'] = 'google_generate_content'
    count = 2 if case == 'aggregate' else 1
    signature = ('한' * 23000 if case == 'utf8' else 'x' * (34000 if count == 2 else
                 66000 if case in ('signature', 'larger_budget') else 60000))
    parts: list[dict[str, Any]] = [google_call(HIDE_TOOL, {'tool_call_id': 'original-call'}, f'native-{i}', signature)
             for i in range(count)]
    if case == 'boundary':
        parts[0]['thoughtSignature'] = ''
        parts[0]['thoughtSignature'] = 'x' * (65536 - len(json.dumps(parts[0], ensure_ascii=False).encode('utf-8')))
    if case == 'args':
        parts[0]['functionCall']['args']['synthetic_extra'] = 'x' * 66000
        parts[0]['thoughtSignature'] = 'small'
    replies.append(google_response(*parts))
    overflow = case in ('signature', 'aggregate', 'utf8', 'boundary', 'args')
    replies.append(google_response(text='synthetic final'))
    response = review_post(review_bridge, stream=False, max_tokens=2048 if case == 'larger_budget' else 1)
    if overflow:
        assert response.status_code == 502
        assert response.json()['error']['code'] == 'context_compaction_native_limit'
        assert len(requests) == 1 and not lfm.calls
        assert app.state.context_compaction_store.items('synthetic-review') == ()
        assert signature not in response.text and HIDE_TOOL not in response.text
    else:
        assert response.status_code == 200 and len(requests) == 2 and len(lfm.calls) == 1
        assert parts == [part for message in requests[1]['contents'] for part in message['parts']
                         if part.get('functionCall', {}).get('name') == HIDE_TOOL]
        assert response.json()['choices'][0]['message']['content'] == 'synthetic final'


def test_compaction_stream_rejects_large_tool_call_id_and_closes_upstream():
    _, error, stream = _collect_synthetic_stream([
        {"delta_tool_calls": [{"index": 0, "id": "x" * 70000, "function": {}}]},
    ])

    assert isinstance(error, VertexAPIError)
    assert error.code == "context_compaction_stream_limit"
    assert stream.closed


def test_compaction_stream_bounds_many_empty_call_indexes():
    empty_calls = [
        {"index": index, "function": {"name": "", "arguments": ""}}
        for index in range(512)
    ]
    _, error, stream = _collect_synthetic_stream([{"delta_tool_calls": empty_calls}])

    assert isinstance(error, VertexAPIError)
    assert error.code == "context_compaction_stream_limit"
    assert stream.closed


def test_compaction_stream_bounds_many_empty_deltas():
    empty_deltas = [
        {"index": 0, "function": {"name": "", "arguments": ""}}
        for _ in range(5000)
    ]
    _, error, stream = _collect_synthetic_stream([{"delta_tool_calls": empty_deltas}])

    assert isinstance(error, VertexAPIError)
    assert error.code == "context_compaction_stream_limit"
    assert stream.closed


def test_compaction_stream_bounds_many_empty_events():
    _, error, stream = _collect_synthetic_stream(({} for _ in range(10000)))

    assert isinstance(error, VertexAPIError)
    assert error.code == "context_compaction_stream_limit"
    assert stream.closed


def test_compaction_stream_bounds_nested_google_native_part_objects():
    native_parts = {"call": {"parts": [{} for _ in range(2000)]}}
    _, error, stream = _collect_synthetic_stream([{"_google_call_parts": native_parts}])

    assert isinstance(error, VertexAPIError)
    assert error.code == "context_compaction_stream_limit"
    assert stream.closed


def test_compaction_stream_bounds_nested_usage_objects():
    usage = {"details": [{} for _ in range(2000)]}
    _, error, stream = _collect_synthetic_stream([{"usage": usage}])

    assert isinstance(error, VertexAPIError)
    assert error.code == "context_compaction_stream_limit"
    assert stream.closed


def test_compaction_stream_has_a_hard_cap_even_for_large_token_budgets():
    _, error, stream = _collect_synthetic_stream(
        [{"delta_text": "x" * (1024 * 1024 + 1)}], max_tokens=10**9,
    )

    assert isinstance(error, VertexAPIError)
    assert error.code == "context_compaction_stream_limit"
    assert stream.closed


@pytest.mark.parametrize(
    ("scope_field", "different_value"),
    [
        ("adapter_id", "other-adapter"),
        ("host_profile", "other-profile"),
        ("branch_scope", "other-branch"),
    ],
)
def test_scope_mismatch_cannot_fall_back_to_raw_session_key(scope_field, different_value):
    affinity = "same-session-id"
    canonical_scope = bridge_scope(affinity)
    foreign_scope = replace(canonical_scope, **{scope_field: different_value})
    store = MemoryContextStore()
    item = _scope_item()
    assert store.put(affinity, item).ok

    assert store.get(foreign_scope, item.item_id) is None
    assert store.items(foreign_scope) == ()
    result = store.unhide(foreign_scope, item.item_id)
    assert not result.ok and result.error == "not_found"
    assert store.get(affinity, item.item_id) == item


def test_exact_canonical_scope_can_still_read_raw_affinity_item():
    affinity = "raw-to-canonical"
    canonical_scope = bridge_scope(affinity)
    store = MemoryContextStore()
    item = _scope_item()
    assert store.put(affinity, item).ok

    assert store.get(canonical_scope, item.item_id) == item
    assert store.items(canonical_scope) == (item,)
    result = store.unhide(canonical_scope, item.item_id)
    assert result.ok and result.item is not None
    assert result.item.visibility == "original"


def test_raw_affinity_can_still_read_exact_canonical_scope_item():
    affinity = "canonical-to-raw"
    canonical_scope = bridge_scope(affinity)
    store = MemoryContextStore()
    item = _scope_item()
    assert store.put(canonical_scope, item).ok

    assert store.get(affinity, item.item_id) == item
    assert store.items(affinity) == (item,)
    result = store.unhide(affinity, item.item_id)
    assert result.ok and result.item is not None
    assert result.item.visibility == "original"


def test_missing_foreign_scope_item_keeps_empty_and_not_found_results():
    scope = replace(bridge_scope("empty-session"), host_profile="other-profile")
    store = MemoryContextStore()

    assert store.get(scope, "missing-item") is None
    assert store.items(scope) == ()
    result = store.unhide(scope, "missing-item")
    assert not result.ok and result.error == "not_found"


def test_bridge_store_rehide_after_unhide_succeeds_at_exact_item_budget():
    from context_hide.engine import ContextHideEngine
    from context_hide.model import ToolResultRecord, compute_invocation_digest, compute_sha256

    affinity = "bridge-rehide-budget"
    scope = bridge_scope(affinity)
    content = "\n".join(f"ordinary synthetic output line {index:03d}" for index in range(60))
    invocation = {"tool_name": "terminal", "arguments": {"command": "synthetic"}}
    record = ToolResultRecord(
        call_id="call-bridge-rehide",
        content=content,
        content_sha256=compute_sha256(content),
        invocation=invocation,
        invocation_digest=compute_invocation_digest(invocation),
    )
    probe = ContextHideEngine().hide_sync(scope, record)
    assert probe.ok and probe.item is not None

    store = MemoryContextStore(max_bytes=MemoryContextStore._item_size(probe.item))
    engine = store.engine
    hidden = engine.hide_sync(scope, record)
    assert hidden.ok and hidden.item is not None
    assert engine.unhide(scope, hidden.item.item_id).ok

    rehidden = engine.hide_sync(scope, record)

    assert rehidden.ok and rehidden.item is not None
    assert rehidden.item.visibility == "compacted"
