"""Native chunked Responses SSE through the public bridge; no paid traffic."""
import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from openai_compatible_bridge.core.cost_tracking import DisabledCostAccounting, NormalizedUsage
import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.context_compaction import AFFINITY_HEADER, HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL
from openai_compatible_bridge.main import _COMPACTION_STREAM_MAX_EVENTS, _collect_compaction_stream, create_app
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError
from test_compaction_foundry_protocols import (
    ALIAS, ARGS, MESSAGES, SOURCE, SUMMARY, TOOL, SyntheticLFM, Unused,
    native_call, original_output, private_output,
)


def sse(event):
    return ('data: ' + json.dumps(event) + '\n\n').encode()


def wire(*calls, text=None, usage=True, cached=3, finish='completed'):
    events = []
    if text is not None:
        events += [sse({'type': 'response.output_text.delta', 'delta': text[:3]}),
                   sse({'type': 'response.output_text.delta', 'delta': text[3:]})]
    for index, call in enumerate(calls):
        events.append(sse({'type': 'response.output_item.added', 'output_index': index,
                           'item': {**call, 'arguments': ''}}))
        args = call['arguments']
        for delta in (args[:5], args[5:]):
            events.append(sse({'type': 'response.function_call_arguments.delta',
                               'item_id': call['id'], 'output_index': index, 'delta': delta}))
        events.append(sse({'type': 'response.output_item.done', 'output_index': index, 'item': call}))
    response = {'status': finish, 'id': 'resp-private-never-public'}
    if usage:
        response['usage'] = {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}
        if cached is not None:
            response['usage']['input_tokens_details'] = {'cached_tokens': cached}
    events += [sse({'type': 'response.completed', 'response': response}), b'data: [DONE]\n\n']
    return b''.join(events)


class Chunked(httpx.AsyncByteStream):
    def __init__(self, payload, fail=False):
        self.payload, self.fail, self.closed = payload, fail, False

    async def __aiter__(self):
        for i in range(0, len(self.payload), 11):
            yield self.payload[i:i + 11]
        if self.fail:
            raise httpx.ReadError('synthetic interrupted upstream')

    async def aclose(self):
        self.closed = True


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv('CONTEXT_COMPACTION_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LFM_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LAYA_ENABLED', 'false')
    monkeypatch.setattr('openai_compatible_bridge.main.BRIDGE_API_KEY', '')
    monkeypatch.setitem(vertex.MODEL_REGISTRY, ALIAS, {'provider': 'foundry', 'kind': 'chat',
                        'provider_model': 'synthetic-responses', 'protocol': 'openai_responses'})
    bodies, replies, streams = [], [], []

    async def handler(request):
        assert request.url.path.endswith('/openai/v1/responses')
        body = json.loads(request.content)
        assert body['stream'] is True, 'must use actual upstream streaming'
        bodies.append(body)
        assert replies, 'unexpected upstream call'
        reply = replies.pop(0)
        stream = reply if isinstance(reply, Chunked) else Chunked(reply(body) if callable(reply) else reply)
        streams.append(stream)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)

    provider = FoundryChatClient(base_url='https://foundry.example/api/v2/llm/proxy/openai/v1/chat/completions', token='synthetic')
    asyncio.run(provider.http.aclose())
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    lfm = SyntheticLFM()
    app = create_app(embedding_client_factory=Unused, chat_client_factory=Unused,
                     rerank_client_factory=Unused, ollama_chat_client_factory=lambda: lfm,
                     foundry_chat_client_factory=lambda: provider, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        yield client, app, bodies, replies, lfm, streams


def post(client, *, messages=None, headers=None, **kwargs):
    return client.post('/v1/chat/completions', headers={AFFINITY_HEADER: 'synthetic-conversation'} if headers is None else headers,
                       json={'model': ALIAS, 'messages': copy.deepcopy(MESSAGES if messages is None else messages),
                             'tools': [TOOL], 'stream': True, 'stream_options': {'include_usage': True}, **kwargs})


def public(response):
    assert response.status_code == 200, response.text
    rows = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
    assert rows[-1] == '[DONE]'
    return [json.loads(row) for row in rows[:-1]]


def text(rows):
    return ''.join(c['delta'].get('content', '') for row in rows for c in row.get('choices', []))


def test_native_stream_hide_buffers_private_and_emits_final_usage(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    replies.extend([wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'}), text='PRIVATE intermediate answer'),
                    wire(text='final public answer')])
    response = post(client, max_tokens=123, reasoning_effort='high')
    rows = public(response)
    assert text(rows) == 'final public answer'
    assert all(secret not in response.text for secret in (HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL, 'private-call', 'fc-distinct-item', 'PRIVATE', 'resp-private'))
    assert rows[-2]['choices'][0]['finish_reason'] == 'stop'
    assert rows[-1]['usage'] == {'prompt_tokens': 20, 'completion_tokens': 4, 'total_tokens': 24,
                                 'prompt_tokens_details': {'cached_tokens': 6}}
    assert len(bodies) == 2 and len(lfm.calls) == 1
    assert bodies[0]['max_output_tokens'] == 123
    assert {t['name'] for t in bodies[0]['tools']} == {'terminal', HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}
    item = app.state.context_compaction_store.items('synthetic-conversation')[0]
    assert item.compaction_source == 'lfm' and SUMMARY in item.compacted
    assert original_output(bodies[0]) == SOURCE
    assert original_output(bodies[1]) == item.compacted
    assert private_output(bodies[1], 'private-call')['ok'] is True
    assert next(i for i in bodies[1]['input'] if i.get('name') == HIDE_TOOL)['call_id'] == 'private-call'
    original_call = {'type': 'function_call', 'call_id': 'original-call', 'name': 'terminal', 'arguments': ARGS}
    assert all(original_call in body['input'] for body in bodies)
    assert all(s.closed for s in streams)
    assert app.state.context_compaction_store.last_measurement.provider_calls == 2


def test_active_private_followup_stream_can_exceed_idle_timeout(bridge, monkeypatch):
    client, app, bodies, replies, lfm, streams = bridge
    monkeypatch.setattr('openai_compatible_bridge.providers.foundry.HTTP_TIMEOUT_SECONDS', 0.04)

    class Active(Chunked):
        async def __aiter__(self):
            for frame in self.payload.split(b'\n\n'):
                if frame:
                    await asyncio.sleep(0.015)
                    yield frame + b'\n\n'

    replies.extend([
        wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})),
        Active(wire(text='synthetic follow-up completed')),
    ])

    response = post(client)

    assert text(public(response)) == 'synthetic follow-up completed'
    assert len(bodies) == 2 and len(streams) == 2 and len(lfm.calls) == 1
    assert all(stream.closed for stream in streams)
    assert original_output(bodies[1]) != SOURCE
    assert app.state.context_compaction_store.items('synthetic-conversation')[0].original == SOURCE


def test_raw_responses_activity_keeps_private_collector_alive_without_public_leak(bridge, monkeypatch):
    client, app, bodies, replies, lfm, streams = bridge
    monkeypatch.setattr('openai_compatible_bridge.providers.foundry.HTTP_TIMEOUT_SECONDS', 0.15)

    class ProgressOnly(Chunked):
        async def __aiter__(self):
            for frame in self.payload.split(b'\n\n'):
                if frame:
                    await asyncio.sleep(0.03)
                    yield frame + b'\n\n'

    progress = b''.join([
        b': synthetic-heartbeat\n\n',
        sse({'type': 'response.created', 'response': {'status': 'in_progress'}}),
        sse({'type': 'response.reasoning_summary_text.delta', 'delta': 'SYNTHETIC_PRIVATE_REASONING'}),
        sse({'type': 'response.in_progress', 'response': {'status': 'in_progress'}}),
        b': synthetic-heartbeat\n\n',
        sse({'type': 'response.queued'}),
        sse({'type': 'response.reasoning_summary_text.delta', 'delta': 'SYNTHETIC_PRIVATE_PROGRESS'}),
    ])
    replies.extend([
        wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})),
        ProgressOnly(progress + wire(text='synthetic follow-up completed')),
    ])

    response = post(client)

    assert text(public(response)) == 'synthetic follow-up completed'
    assert 'SYNTHETIC_PRIVATE_REASONING' not in response.text
    assert 'SYNTHETIC_PRIVATE_PROGRESS' not in response.text
    assert len(bodies) == 2 and len(streams) == 2 and len(lfm.calls) == 1
    assert all(stream.closed for stream in streams)
    assert app.state.context_compaction_store.items('synthetic-conversation')[0].original == SOURCE


def test_raw_private_activity_has_event_limit_and_closes_upstream(bridge):
    client, app, bodies, replies, lfm, streams = bridge

    class Heartbeats(Chunked):
        async def __aiter__(self):
            for _ in range(_COMPACTION_STREAM_MAX_EVENTS + 1):
                yield b': synthetic-heartbeat\n\n'

    replies.append(Heartbeats(b''))

    async def run():
        ctx = DisabledCostAccounting().reservation(
            endpoint='chat', model=ALIAS, provider='foundry', forecast_usage=NormalizedUsage(),
        )
        async with ctx:
            try:
                await _collect_compaction_stream(
                    app.state.foundry_chat_client, ctx, model='synthetic-responses', messages=MESSAGES,
                    resolved_config={'protocol': 'openai_responses'},
                )
            except VertexAPIError as error:
                return error
        raise AssertionError('heartbeat event limit was not enforced')

    error = client.portal.call(run)
    assert error.code == 'context_compaction_stream_limit'
    assert len(streams) == 1 and streams[0].closed


def test_two_hides_in_one_stream_turn_never_generate_lfm_twice(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    messages = copy.deepcopy(MESSAGES)
    messages.extend([{'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'second-call', 'type': 'function', 'function': {'name': 'terminal', 'arguments': ARGS}}]},
        {'role': 'tool', 'tool_call_id': 'second-call', 'content': SOURCE + '\nsecond synthetic result'}])
    replies.extend([wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'}, 'hide-one', 'fc-one'),
                         native_call(HIDE_TOOL, {'tool_call_id': 'second-call'}, 'hide-two', 'fc-two')), wire(text='done')])
    assert text(public(post(client, messages=messages))) == 'done'
    assert len(lfm.calls) == 1
    assert private_output(bodies[-1], 'hide-one')['ok'] and private_output(bodies[-1], 'hide-two')['ok']
    assert app.state.context_compaction_store.last_measurement.lfm_calls == 1


@pytest.mark.parametrize('protocol', ['unverified_protocol'])
def test_unverified_stream_protocols_remain_gated(protocol):
    from openai_compatible_bridge.context_compaction import CompactionSettings, plan_request
    plan, reason = plan_request(settings=CompactionSettings(enabled=True), headers={AFFINITY_HEADER: 'synthetic'},
                                tools=[TOOL], tool_choice=None, provider='foundry', protocol=protocol, stream=True)
    assert plan is None and reason == 'streaming'


def test_stream_after_list_and_exact_restore(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    replies.extend([wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})), wire(text='hidden')])
    assert text(public(post(client))) == 'hidden'
    item = app.state.context_compaction_store.items('synthetic-conversation')[0]
    replies.append(wire(text='after'))
    assert text(public(post(client, tools=None))) == 'after'
    assert original_output(bodies[-1]) == item.compacted
    replies.extend([wire(native_call(LIST_TOOL, {}, 'list-call', 'fc-list')), wire(text='listed')])
    assert text(public(post(client))) == 'listed'
    assert private_output(bodies[-1], 'list-call')['items'][0]['item_id'] == item.item_id
    rendered = copy.deepcopy(MESSAGES)
    rendered[2]['content'] = item.compacted
    replies.extend([wire(native_call(UNHIDE_TOOL, {'item_id': item.item_id}, 'restore-call', 'fc-restore')), wire(text='restored')])
    assert text(public(post(client, messages=rendered))) == 'restored'
    assert original_output(bodies[-1]) == SOURCE
    replies.append(wire(text='still restored'))
    public(post(client, messages=rendered))
    assert original_output(bodies[-1]) == SOURCE and len(lfm.calls) == 1
    assert all(s.closed for s in streams)


@pytest.mark.parametrize('mixed', [False, True])
def test_external_calls_are_public_and_private_mixed_calls_not_executed(bridge, mixed):
    client, app, bodies, replies, lfm, streams = bridge
    external = native_call('terminal', {'command': 'public command'}, 'public-call', 'fc-external')
    calls = ([native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})] if mixed else []) + [external]
    replies.append(wire(*calls, text='public explanation'))
    response = post(client)
    rows = public(response)
    assert text(rows) == 'public explanation'
    tool_calls = [call for row in rows for choice in row.get('choices', []) for call in choice['delta'].get('tool_calls', [])]
    assert tool_calls == [{'index': 0, 'id': 'public-call', 'type': 'function', 'function': {'name': 'terminal', 'arguments': external['arguments']}}]
    assert rows[-2]['choices'][0]['finish_reason'] == 'tool_calls'
    assert HIDE_TOOL not in response.text and 'private-call' not in response.text
    assert not lfm.calls and len(bodies) == 1


@pytest.mark.parametrize('skip', ['disabled', 'header', 'collision', 'forced'])
def test_excluded_stream_uses_unchanged_default_path(bridge, monkeypatch, skip):
    client, app, bodies, replies, lfm, streams = bridge
    kwargs = {}
    if skip == 'disabled':
        monkeypatch.setenv('CONTEXT_COMPACTION_ENABLED', 'false')
    elif skip == 'header':
        kwargs['headers'] = {}
    elif skip == 'collision':
        kwargs['tools'] = [TOOL, {'type': 'function', 'function': {'name': HIDE_TOOL}}]
    else:
        kwargs['tool_choice'] = {'type': 'function', 'function': {'name': 'terminal'}}
    replies.append(wire(text='default text'))
    rows = public(post(client, **kwargs))
    # Two upstream text fragments remain separate, unlike the buffered path.
    assert [c['delta']['content'] for r in rows for c in r.get('choices', []) if 'content' in c['delta']] == ['def', 'ault text']
    names = {t['name'] for t in bodies[0]['tools']}
    assert names == ({'terminal', HIDE_TOOL} if skip == 'collision' else {'terminal'})
    assert original_output(bodies[0]) == SOURCE and not lfm.calls


@pytest.mark.parametrize('failure', ['read', 'eof', 'event'])
def test_upstream_partial_failure_emits_error_not_success_and_closes(bridge, failure):
    client, app, bodies, replies, lfm, streams = bridge
    replies.append(wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'}), text='PRIVATE first round'))
    partial = sse({'type': 'response.output_text.delta', 'delta': 'PRIVATE partial final'})
    if failure == 'read':
        partial = Chunked(partial, fail=True)
    elif failure == 'event':
        partial += sse({'type': 'response.failed', 'response': {'error': {'message': 'synthetic failure', 'code': 'synthetic_failure'}}})
    replies.append(partial)
    response = post(client)
    rows = public(response)
    assert rows[0].get('error'), response.text
    assert all('choices' not in r and 'usage' not in r for r in rows)
    assert 'PRIVATE' not in response.text and HIDE_TOOL not in response.text
    assert all(s.closed for s in streams)
    assert app.state.context_compaction_store.items('synthetic-conversation')[0].original == SOURCE


@pytest.mark.parametrize('missing', ['all_usage', 'one_round_cache', 'invalid_cache'])
def test_missing_usage_is_not_fabricated(bridge, missing):
    client, app, bodies, replies, lfm, streams = bridge
    first = wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'}))
    final = wire(text='done', usage=missing != 'all_usage', cached=None if missing == 'one_round_cache' else True if missing == 'invalid_cache' else 3)
    replies.extend([first, final])
    rows = public(post(client))
    usages = [row['usage'] for row in rows if 'usage' in row]
    if missing == 'all_usage':
        assert usages == []
    else:
        assert usages == [{'prompt_tokens': 20, 'completion_tokens': 4, 'total_tokens': 24}]


def test_argument_fragments_identified_only_by_item_id_keep_distinct_calls(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    calls = [native_call('terminal', {'command': 'a'}, 'public-a', 'fc-a'), native_call('terminal', {'command': 'b'}, 'public-b', 'fc-b')]
    payload = wire(*calls)
    # Real Responses argument events may identify the native item without output_index.
    rows = [json.loads(line[6:]) for line in payload.decode().splitlines() if line.startswith('data: ') and line[6:] != '[DONE]']
    # Initialize both calls before the first argument delta: last-slot fallback is unsafe.
    rows = ([r for r in rows if r['type'] == 'response.output_item.added'] +
            [r for r in rows if r['type'] != 'response.output_item.added'])
    for row in rows:
        if row['type'] == 'response.function_call_arguments.delta':
            row.pop('output_index')
    replies.append(b''.join(sse(row) for row in rows) + b'data: [DONE]\n\n')
    rows = public(post(client))
    actual = [call for row in rows for c in row.get('choices', []) for call in c['delta'].get('tool_calls', [])]
    assert [(c['id'], json.loads(c['function']['arguments'])) for c in actual] == [('public-a', {'command': 'a'}), ('public-b', {'command': 'b'})]


def test_round_limit_returns_error_and_lfm_once(bridge, monkeypatch):
    client, app, bodies, replies, lfm, streams = bridge
    monkeypatch.setenv('CONTEXT_COMPACTION_MAX_INTERNAL_ROUNDS', '1')
    replies.extend([wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})) for _ in range(2)])
    rows = public(post(client))
    assert rows == [{'error': {'message': 'Context compaction could not complete local processing.',
                              'type': 'api_error', 'param': None, 'code': 'context_compaction_loop_limit'}}]
    assert len(lfm.calls) == 1 and len(bodies) == 2 and all(s.closed for s in streams)


def test_cancellation_closes_native_upstream_generator(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    from openai_compatible_bridge.main import _collect_compaction_stream
    from openai_compatible_bridge.core.cost_tracking import DisabledCostAccounting, NormalizedUsage

    async def run():
        started = asyncio.Event()
        closed = asyncio.Event()

        class Waiting(Chunked):
            async def __aiter__(self):
                yield sse({'type': 'response.output_text.delta', 'delta': 'PRIVATE waiting'})
                started.set()
                await asyncio.Event().wait()

            async def aclose(self):
                await super().aclose()
                closed.set()

        provider = app.state.foundry_chat_client
        replies.append(Waiting(b''))
        ctx = DisabledCostAccounting().reservation(endpoint='chat', model=ALIAS, forecast_usage=NormalizedUsage(), provider='foundry')
        task = asyncio.create_task(_collect_compaction_stream(provider, ctx, model='synthetic-responses', messages=MESSAGES,
                                  resolved_config={'protocol': 'openai_responses'}))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set() and streams[-1].closed

    client.portal.call(run)


@pytest.mark.parametrize('failure', [None, 'second_admission', 'partial', 'cancel'])
def test_native_attempt_metering_and_lfm_context_restoration(bridge, monkeypatch, failure):
    from decimal import Decimal
    from openai_compatible_bridge.core.async_cost import AsyncCostAccounting, _ACTIVE
    from openai_compatible_bridge.core.cost_tracking import CostConfigError, CostReservation, NormalizedUsage
    from openai_compatible_bridge.core.metered_http import MeteredHTTPClient
    from openai_compatible_bridge.providers.ollama import OllamaChatClient

    client, app, bodies, replies, lfm, streams = bridge
    admitted, recorded = [], []
    accounting = AsyncCostAccounting(config=None, billing={'foundry': 'metered', 'ollama': 'metered'})

    async def admit(provider):
        ctx = _ACTIVE.get()
        admitted.append((provider, ctx.model))
        if failure == 'second_admission' and len(admitted) == 3:
            raise CostConfigError('synthetic admission denied')
        return CostReservation(str(len(admitted)), 'synthetic', provider, 'chat', ctx.model,
                               Decimal(0), 'USD', 'synthetic', 'synthetic', 'synthetic', 'synthetic')

    monkeypatch.setattr(accounting, 'before_attempt', admit)
    monkeypatch.setattr(accounting, 'record_attempt', lambda reservation, usage: recorded.append((reservation.provider, usage)))
    app.state.cost_accounting = accounting
    provider = app.state.foundry_chat_client
    provider.http = MeteredHTTPClient(provider.http, accounting, provider='foundry')
    ollama = OllamaChatClient(base_url='https://synthetic.invalid')
    asyncio.run(ollama.http.aclose())

    def respond(request):
        assert json.loads(request.content)['stream'] is False
        return httpx.Response(200, json={'model': 'synthetic-lfm', 'message': {'role': 'assistant', 'content': json.dumps({'summary': SUMMARY})},
                                        'prompt_eval_count': 7, 'eval_count': 1, 'done': True})

    ollama.http = MeteredHTTPClient(httpx.AsyncClient(transport=httpx.MockTransport(respond)), accounting, provider='ollama')
    app.state.ollama_chat_client = ollama
    replies.append(wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})))
    if failure == 'partial':
        replies.append(Chunked(sse({'type': 'response.output_text.delta', 'delta': 'PRIVATE'}), fail=True))
    else:
        replies.append(wire(text='metered final'))
    if failure != 'cancel':
        rows = public(post(client))
        assert bool(rows[0].get('error')) == (failure is not None)
        assert admitted == [('foundry', ALIAS), ('ollama', 'ollama:lfm2.5-thinking:latest'), ('foundry', ALIAS)]
        assert [p for p, u in recorded] == (['foundry', 'ollama'] if failure == 'second_admission' else ['foundry', 'ollama', 'foundry'])
        assert recorded[0][1].total_tokens == 12 and recorded[1][1].total_tokens == 8
        if failure == 'partial':
            assert recorded[-1][1] is None
        elif failure is None:
            assert recorded[-1][1].total_tokens == 12
    else:
        from openai_compatible_bridge.main import _collect_compaction_stream

        async def cancel():
            entered = asyncio.Event()

            class Waiting(Chunked):
                async def __aiter__(self):
                    yield sse({'type': 'response.output_text.delta', 'delta': 'PRIVATE'})
                    entered.set()
                    await asyncio.Event().wait()

            replies.clear()
            replies.append(Waiting(b''))
            async with accounting.reservation(endpoint='chat', model=ALIAS, provider='foundry', forecast_usage=NormalizedUsage()) as ctx:
                task = asyncio.create_task(_collect_compaction_stream(provider, ctx, model='synthetic-responses', messages=MESSAGES,
                                          resolved_config={'protocol': 'openai_responses'}))
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert recorded == [('foundry', None)] and streams[-1].closed

        client.portal.call(cancel)
    client.portal.call(ollama.close)
    client.portal.call(accounting.aclose)


@pytest.mark.parametrize('case', ['partial_usage', 'done_arguments', 'length'])
def test_terminal_native_edge_fields(bridge, case):
    client, app, bodies, replies, lfm, streams = bridge
    if case == 'partial_usage':
        payload = sse({'type': 'response.output_text.delta', 'delta': 'done'}) + sse({'type': 'response.completed', 'response': {
            'status': 'completed', 'usage': {'input_tokens': 10}}}) + b'data: [DONE]\n\n'
    elif case == 'done_arguments':
        call = native_call('terminal', {'command': 'done-only'}, 'external-call', 'fc-external')
        payload = sse({'type': 'response.output_item.added', 'output_index': 0, 'item': {**call, 'arguments': ''}})
        payload += sse({'type': 'response.output_item.done', 'output_index': 0, 'item': call})
        payload += sse({'type': 'response.completed', 'response': {'status': 'completed'}}) + b'data: [DONE]\n\n'
    else:
        payload = sse({'type': 'response.output_text.delta', 'delta': 'limited'}) + sse({'type': 'response.incomplete', 'response': {
            'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
            'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}}}) + b'data: [DONE]\n\n'
    replies.append(payload)
    rows = public(post(client))
    if case == 'partial_usage':
        assert not any('usage' in row for row in rows)
    elif case == 'done_arguments':
        call = next(call for row in rows for c in row.get('choices', []) for call in c['delta'].get('tool_calls', []))
        assert json.loads(call['function']['arguments']) == {'command': 'done-only'}
    else:
        assert rows[-2]['choices'][0]['finish_reason'] == 'length'


@pytest.mark.parametrize('case', ['buffer', 'timeout'])
def test_collection_bounds_close_native_stream_without_success(bridge, monkeypatch, case):
    client, app, bodies, replies, lfm, streams = bridge
    if case == 'buffer':
        replies.append(wire(text='x' * 70000))
    else:
        monkeypatch.setattr('openai_compatible_bridge.providers.foundry.HTTP_TIMEOUT_SECONDS', 0.01)

        class Slow(Chunked):
            async def __aiter__(self):
                yield sse({'type': 'response.output_text.delta', 'delta': 'PRIVATE'})
                await asyncio.Event().wait()

        replies.append(Slow(b''))
    if case == 'timeout':
        # Test watchdog is longer than the production whole-round deadline.
        from openai_compatible_bridge.main import _collect_compaction_stream
        from openai_compatible_bridge.core.cost_tracking import DisabledCostAccounting, NormalizedUsage
        from openai_compatible_bridge.providers.vertex import VertexAPIError

        async def run():
            ctx = DisabledCostAccounting().reservation(endpoint='chat', model=ALIAS, provider='foundry', forecast_usage=NormalizedUsage())
            with pytest.raises(VertexAPIError, match='timed out') as caught:
                await asyncio.wait_for(_collect_compaction_stream(app.state.foundry_chat_client, ctx, model='synthetic-responses',
                                       messages=MESSAGES, resolved_config={'protocol': 'openai_responses'}), 0.1)
            return caught.value
        error = client.portal.call(run)
        assert error.stage == 'collector_idle'
    else:
        rows = public(post(client, max_tokens=1))
        assert rows[0]['error']['code'] == 'context_compaction_stream_limit'
    assert all(s.closed for s in streams)
