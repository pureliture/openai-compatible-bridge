"""Native chunked Anthropic SSE through the bridge; synthetic HTTP only."""
import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.context_compaction import AFFINITY_HEADER, HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL
from openai_compatible_bridge.main import create_app, _collect_compaction_stream
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from test_compaction_foundry_protocols import ARGS, MESSAGES, SOURCE, SUMMARY, TOOL, SyntheticLFM, Unused
from test_compaction_anthropic import ALIAS, AFFINITY, native_call, output, private_output, blocks, assert_original_call
from test_compaction_responses_stream import Chunked, public, text


def sse(event):
    return ('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n').encode()


def wire(*calls, text=None, start_usage=None, end_usage=None, stop=None, complete=True):
    start_usage = {'input_tokens': 10, 'output_tokens': 0, 'cache_read_input_tokens': 3,
                   'cache_creation_input_tokens': 4} if start_usage is None else start_usage
    end_usage = {'output_tokens': 2} if end_usage is None else end_usage
    events = [sse({'type': 'message_start', 'message': {'id': 'msg-private', 'usage': start_usage}})]
    if text is not None:
        events.append(sse({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}))
        for part in (text[:3], text[3:]):
            events.append(sse({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': part}}))
        events.append(sse({'type': 'content_block_stop', 'index': 0}))
    for index, call in enumerate(calls, 1):
        events.append(sse({'type': 'content_block_start', 'index': index, 'content_block': {**call, 'input': {}}}))
        args = json.dumps(call['input'])
        for part in (args[:5], args[5:]):
            events.append(sse({'type': 'content_block_delta', 'index': index,
                               'delta': {'type': 'input_json_delta', 'partial_json': part}}))
        events.append(sse({'type': 'content_block_stop', 'index': index}))
    events.append(sse({'type': 'message_delta', 'delta': {'stop_reason': stop or ('tool_use' if calls else 'end_turn')}, 'usage': end_usage}))
    if complete:
        events.append(sse({'type': 'message_stop'}))
    return b''.join(events)


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv('CONTEXT_COMPACTION_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LFM_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LAYA_ENABLED', 'false')
    monkeypatch.setattr('openai_compatible_bridge.main.BRIDGE_API_KEY', '')
    monkeypatch.setitem(vertex.MODEL_REGISTRY, ALIAS, {'provider': 'foundry', 'kind': 'chat',
                        'provider_model': 'synthetic-anthropic', 'protocol': 'anthropic_messages'})
    bodies, replies, streams = [], [], []

    async def handler(request):
        assert request.url.path == '/api/v2/llm/proxy/anthropic/v1/messages'
        assert request.headers['anthropic-version'] == '2023-06-01'
        body = json.loads(request.content)
        assert body['stream'] is True, 'generate fallback is forbidden'
        bodies.append(body)
        assert replies, 'unexpected upstream call'
        reply = replies.pop(0)
        stream = reply if isinstance(reply, Chunked) else Chunked(reply)
        streams.append(stream)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)

    provider = FoundryChatClient(base_url='https://foundry.example/api/v2/llm/proxy/openai/v1/chat/completions', token='synthetic')
    asyncio.run(provider.http.aclose())
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def forbidden_generate(**kwargs):
        pytest.fail('upstream generate must not be used')

    monkeypatch.setattr(provider, 'generate', forbidden_generate)
    lfm = SyntheticLFM()
    app = create_app(embedding_client_factory=Unused, chat_client_factory=Unused,
                     rerank_client_factory=Unused, ollama_chat_client_factory=lambda: lfm,
                     foundry_chat_client_factory=lambda: provider, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        yield client, app, bodies, replies, lfm, streams


def post(client, *, messages=None, headers=None, **kwargs):
    return client.post('/v1/chat/completions', headers={AFFINITY_HEADER: AFFINITY} if headers is None else headers,
                       json={'model': ALIAS, 'messages': copy.deepcopy(MESSAGES if messages is None else messages),
                             'tools': [TOOL], 'stream': True, 'stream_options': {'include_usage': True}, **kwargs})


def test_native_hide_final_external_call_usage_and_private_buffer(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    external = native_call('terminal', {'command': 'public next', 'timeout': 37}, 'public-call')
    replies.extend([wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'}), text='PRIVATE intermediate'),
                    wire(external, text='final public answer')])
    response = post(client, max_tokens=123)
    rows = public(response)
    assert text(rows) == 'final public answer'
    assert all(secret not in response.text for secret in (HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL, 'private-call', 'PRIVATE', 'msg-private'))
    calls = [call for row in rows for c in row.get('choices', []) for call in c['delta'].get('tool_calls', [])]
    assert calls == [{'index': 0, 'id': 'public-call', 'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps(external['input'])}}]
    assert rows[-2]['choices'][0]['finish_reason'] == 'tool_calls'
    assert rows[-1]['usage'] == {'prompt_tokens': 20, 'completion_tokens': 4, 'total_tokens': 24,
                                 'cache_read_input_tokens': 6, 'cache_creation_input_tokens': 8}
    assert len(bodies) == 2 and len(lfm.calls) == 1
    assert bodies[0]['max_tokens'] == 123
    assert {t['name'] for t in bodies[0]['tools']} == {'terminal', HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    assert item.compaction_source == 'lfm' and SUMMARY in item.compacted
    assert output(bodies[0]) == SOURCE and output(bodies[1]) == item.compacted
    assert private_output(bodies[1])['ok'] is True
    assert native_call(HIDE_TOOL, {'tool_call_id': 'original-call'}) in blocks(bodies[1], 'tool_use')
    for body in bodies:
        assert_original_call(body)
    measurement = app.state.context_compaction_store.last_measurement
    assert measurement.provider_calls == 2 and measurement.cache_read_tokens == 6 and measurement.cache_write_tokens == 8
    assert all(s.closed for s in streams)


def test_hide_after_list_exact_restore(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    replies.extend([wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})), wire(text='hidden')])
    assert text(public(post(client))) == 'hidden'
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    replies.append(wire(text='after'))
    assert text(public(post(client, tools=None))) == 'after'
    assert output(bodies[-1]) == item.compacted
    replies.extend([wire(native_call(LIST_TOOL, {}, 'list-call')), wire(text='listed')])
    assert text(public(post(client))) == 'listed'
    assert private_output(bodies[-1], 'list-call')['items'][0]['item_id'] == item.item_id
    rendered = copy.deepcopy(MESSAGES)
    rendered[2]['content'] = item.compacted
    replies.extend([wire(native_call(UNHIDE_TOOL, {'item_id': item.item_id}, 'restore-call')), wire(text='restored')])
    assert text(public(post(client, messages=rendered))) == 'restored'
    assert output(bodies[-1]) == SOURCE
    replies.append(wire(text='still restored'))
    public(post(client, messages=rendered))
    assert output(bodies[-1]) == SOURCE and len(lfm.calls) == 1
    assert all(s.closed for s in streams)


def test_multiple_private_calls_adjacent_original_results_and_native_continuation(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    messages = copy.deepcopy(MESSAGES)
    messages[1]['tool_calls'].append({'id': 'second-call', 'type': 'function', 'function': {'name': 'terminal', 'arguments': ARGS}})
    messages.extend([{'role': 'tool', 'tool_call_id': 'second-call', 'content': SOURCE + '\nsecond result'},
                     {'role': 'user', 'content': 'retain adjacent text'}])
    original = copy.deepcopy(messages)
    replies.extend([wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'}, 'hide-one'),
                         native_call(HIDE_TOOL, {'tool_call_id': 'second-call'}, 'hide-two'),
                         native_call(LIST_TOOL, {}, 'list-call')), wire(text='done')])
    assert text(public(post(client, messages=messages))) == 'done'
    assert len(lfm.calls) == 1 and messages == original
    assert [b['id'] for b in bodies[-1]['messages'][-2]['content']] == ['hide-one', 'hide-two', 'list-call']
    assert [b['tool_use_id'] for b in bodies[-1]['messages'][-1]['content']] == ['hide-one', 'hide-two', 'list-call']
    assert private_output(bodies[-1], 'hide-one')['ok'] and private_output(bodies[-1], 'hide-two')['ok']
    for body in bodies:
        assert [b['tool_use_id'] for b in body['messages'][2]['content'][:2]] == ['original-call', 'second-call']
        assert body['messages'][2]['content'][-1] == {'type': 'text', 'text': 'retain adjacent text'}
    assert app.state.context_compaction_store.last_measurement.lfm_calls == 1


@pytest.mark.parametrize('mixed', [False, True])
def test_parallel_external_calls_and_mixed_private_not_executed(bridge, mixed):
    client, app, bodies, replies, lfm, streams = bridge
    calls = [native_call('terminal', {'command': command}, 'public-' + command) for command in ('a', 'b')]
    replies.append(wire(*([native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})] if mixed else []), *calls, text='explanation'))
    response = post(client)
    rows = public(response)
    actual = [call for row in rows for c in row.get('choices', []) for call in c['delta'].get('tool_calls', [])]
    assert [(c['id'], json.loads(c['function']['arguments'])) for c in actual] == [('public-a', {'command': 'a'}), ('public-b', {'command': 'b'})]
    assert text(rows) == 'explanation' and rows[-2]['choices'][0]['finish_reason'] == 'tool_calls'
    assert HIDE_TOOL not in response.text and 'private-call' not in response.text
    assert not lfm.calls and len(bodies) == 1


@pytest.mark.parametrize('case', ['missing_input', 'missing_output', 'missing_all', 'invalid_input', 'invalid_output', 'zero', 'missing_cache', 'invalid_cache', 'length', 'wrong_input_event'])
def test_native_usage_missing_zero_cache_and_finish(bridge, case):
    client, app, bodies, replies, lfm, streams = bridge
    start = {'input_tokens': 10, 'output_tokens': 0, 'cache_read_input_tokens': 3, 'cache_creation_input_tokens': 4}
    end = {'output_tokens': 2}
    if case == 'missing_input':
        start.pop('input_tokens')
    elif case == 'missing_output':
        end = {}
    elif case == 'missing_all':
        start, end = {}, {}
    elif case == 'wrong_input_event':
        start.pop('input_tokens')
        end['input_tokens'] = 10
    elif case == 'invalid_input':
        start['input_tokens'] = True
    elif case == 'invalid_output':
        end['output_tokens'] = '2'
    elif case == 'zero':
        start['input_tokens'], end['output_tokens'] = 0, 0
    elif case == 'missing_cache':
        start.pop('cache_read_input_tokens')
    elif case == 'invalid_cache':
        start['cache_read_input_tokens'] = True
    replies.append(wire(text='done', start_usage=start, end_usage=end, stop='max_tokens' if case == 'length' else None))
    rows = public(post(client))
    usages = [row['usage'] for row in rows if 'usage' in row]
    if case in {'missing_input', 'missing_output', 'missing_all', 'invalid_input', 'invalid_output', 'wrong_input_event'}:
        assert usages == []
    else:
        expected = {'prompt_tokens': 0 if case == 'zero' else 10, 'completion_tokens': 0 if case == 'zero' else 2,
                    'total_tokens': 0 if case == 'zero' else 12, 'cache_creation_input_tokens': 4}
        if case not in {'missing_cache', 'invalid_cache'}:
            expected['cache_read_input_tokens'] = 3
        assert usages == [expected]
    assert rows[-(2 if usages else 1)]['choices'][0]['finish_reason'] == ('length' if case == 'length' else 'stop')


@pytest.mark.parametrize('failure', ['read', 'eof', 'delta_eof', 'event'])
def test_partial_native_failure_does_not_leak_or_succeed(bridge, failure):
    client, app, bodies, replies, lfm, streams = bridge
    replies.append(wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})))
    partial = sse({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'PRIVATE partial'}})
    if failure == 'read':
        partial = Chunked(partial, fail=True)
    elif failure == 'delta_eof':
        partial = wire(text='PRIVATE partial', complete=False)
    elif failure == 'event':
        partial += sse({'type': 'error', 'error': {'type': 'synthetic_failure', 'message': 'synthetic failure'}})
    replies.append(partial)
    response = post(client)
    rows = public(response)
    assert rows[0].get('error'), response.text
    if failure in {'eof', 'delta_eof'}:
        assert rows[0]['error']['code'] == 'incomplete_stream'
    assert len(bodies) == 2
    assert all('choices' not in r and 'usage' not in r for r in rows)
    assert 'PRIVATE' not in response.text and HIDE_TOOL not in response.text
    assert all(s.closed for s in streams)
    assert app.state.context_compaction_store.items(AFFINITY)[0].original == SOURCE


@pytest.mark.parametrize('skip', ['disabled', 'header', 'collision', 'forced'])
def test_excluded_native_stream_path_unchanged(bridge, monkeypatch, skip):
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
    assert [c['delta']['content'] for r in rows for c in r.get('choices', []) if 'content' in c['delta']] == ['def', 'ault text']
    assert {t['name'] for t in bodies[0]['tools']} == ({'terminal', HIDE_TOOL} if skip == 'collision' else {'terminal'})
    assert not lfm.calls and output(bodies[0]) == SOURCE


@pytest.mark.parametrize('case', ['buffer', 'timeout', 'cancel'])
def test_bounds_and_cancellation_close_upstream(bridge, monkeypatch, case):
    from openai_compatible_bridge.core.cost_tracking import DisabledCostAccounting, NormalizedUsage
    from openai_compatible_bridge.providers.vertex import VertexAPIError
    client, app, bodies, replies, lfm, streams = bridge
    if case == 'buffer':
        replies.append(wire(text='x' * 70000))
        rows = public(post(client, max_tokens=1))
        assert rows[0]['error']['code'] == 'context_compaction_stream_limit'
    else:
        async def run():
            entered = asyncio.Event()

            class Waiting(Chunked):
                async def __aiter__(self):
                    yield sse({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'PRIVATE'}})
                    entered.set()
                    await asyncio.Event().wait()

            replies.append(Waiting(b''))
            ctx = DisabledCostAccounting().reservation(endpoint='chat', model=ALIAS, provider='foundry', forecast_usage=NormalizedUsage())
            kwargs = dict(model='synthetic-anthropic', messages=MESSAGES, resolved_config={'protocol': 'anthropic_messages'})
            if case == 'timeout':
                monkeypatch.setattr('openai_compatible_bridge.providers.foundry.HTTP_TIMEOUT_SECONDS', 0.01)
                with pytest.raises(VertexAPIError, match='timed out'):
                    await asyncio.wait_for(_collect_compaction_stream(app.state.foundry_chat_client, ctx, **kwargs), 0.1)
            else:
                task = asyncio.create_task(_collect_compaction_stream(app.state.foundry_chat_client, ctx, **kwargs))
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        client.portal.call(run)
    assert all(s.closed for s in streams)


@pytest.mark.parametrize('failure', [None, 'second_admission', 'partial', 'cancel'])
def test_native_http_attempt_metering_and_lfm_context_restoration(bridge, monkeypatch, failure):
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
    partial = sse({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'PRIVATE'}})
    replies.append(Chunked(partial, fail=True) if failure == 'partial' else wire(text='metered final'))
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
        async def cancel():
            entered = asyncio.Event()

            class Waiting(Chunked):
                async def __aiter__(self):
                    yield partial
                    entered.set()
                    await asyncio.Event().wait()

            replies.clear()
            replies.append(Waiting(b''))
            async with accounting.reservation(endpoint='chat', model=ALIAS, provider='foundry', forecast_usage=NormalizedUsage()) as ctx:
                task = asyncio.create_task(_collect_compaction_stream(provider, ctx, model='synthetic-anthropic', messages=MESSAGES,
                                          resolved_config={'protocol': 'anthropic_messages'}))
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert recorded == [('foundry', None)] and streams[-1].closed
        client.portal.call(cancel)
    assert all(s.closed for s in streams)
    client.portal.call(ollama.close)
    client.portal.call(accounting.aclose)

