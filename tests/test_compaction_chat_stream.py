"""Native OpenAI SSE over MockTransport, through the public bridge; no paid traffic."""
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
from test_compaction_foundry_protocols import ALIAS, ARGS, MESSAGES, SOURCE, SUMMARY, TOOL, SyntheticLFM, Unused
from test_compaction_xai_stream import Chunked, public, text


def sse(event):
    return ('data: ' + json.dumps(event) + '\n\n').encode()


def delta(content):
    return {'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]}


def call(name, args, identity='private-call'):
    return {'id': identity, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


def wire(*calls, content=None, usage=True, cached=3, finish=None):
    events = [sse({'id': 'private-upstream-id', 'choices': [
        {'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})]
    if content is not None:
        events += [sse(delta(content[:3])), sse(delta(content[3:]))]
    for index, c in enumerate(calls):
        events.append(sse({'choices': [{'index': 0, 'delta': {'tool_calls': [
            {'index': index, 'id': c['id'], 'type': 'function',
             'function': {'name': c['function']['name'][:4], 'arguments': ''}}]}, 'finish_reason': None}]}))
    # Interleave index-only fragments in reverse order after all identities exist.
    for index, c in reversed(list(enumerate(calls))):
        fn = c['function']
        for name, args in [(fn['name'][4:], fn['arguments'][:5]), ('', fn['arguments'][5:])]:
            events.append(sse({'choices': [{'index': 0, 'delta': {'tool_calls': [
                {'index': index, 'function': {'name': name, 'arguments': args}}]}, 'finish_reason': None}]}))
    events.append(sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish or ('tool_calls' if calls else 'stop')}]}))
    if usage:
        counts = {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}
        if cached is not None:
            counts['prompt_tokens_details'] = {'cached_tokens': cached}
        events.append(sse({'choices': [], 'usage': counts}))
    return b''.join(events) + b'data: [DONE]\n\n'


def output(body, identity='original-call'):
    return next(m['content'] for m in body['messages'] if m.get('tool_call_id') == identity)


@pytest.fixture(params=['openai_chat_completions', None])
def bridge(monkeypatch, request):
    monkeypatch.setenv('CONTEXT_COMPACTION_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LFM_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LAYA_ENABLED', 'false')
    monkeypatch.setattr('openai_compatible_bridge.main.BRIDGE_API_KEY', '')
    monkeypatch.setitem(vertex.MODEL_REGISTRY, ALIAS, {'provider': 'foundry', 'kind': 'chat',
                        'provider_model': 'synthetic-chat', 'protocol': request.param})
    bodies, replies, streams = [], [], []

    async def handler(request):
        assert request.url.path.endswith('/openai/v1/chat/completions')
        body = json.loads(request.content)
        assert body['stream'] is True
        bodies.append(body)
        assert replies, 'unexpected upstream request'
        reply = replies.pop(0)
        stream = reply if isinstance(reply, Chunked) else Chunked(reply)
        streams.append(stream)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)

    provider = FoundryChatClient(base_url='https://foundry.example/api/v2/llm/proxy/openai/v1/chat/completions', token='synthetic')
    asyncio.run(provider.http.aclose())
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def forbidden(**kwargs):
        pytest.fail('native streaming must not call generate')

    monkeypatch.setattr(provider, 'generate', forbidden)
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


def test_native_hide_result_only_private_and_final_usage(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    replies.extend([wire(call(HIDE_TOOL, {'tool_call_id': 'original-call'}), content='PRIVATE answer'),
                    wire(content='final public')])
    response = post(client, max_tokens=123)
    rows = public(response)
    assert text(rows) == 'final public'
    assert not any(s in response.text for s in [HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL, 'private-call', 'private-upstream-id', 'PRIVATE'])
    assert rows[0]['choices'][0]['delta']['role'] == 'assistant'
    assert rows[-2]['choices'][0]['finish_reason'] == 'stop'
    assert rows[-1]['usage'] == {'prompt_tokens': 20, 'completion_tokens': 4, 'total_tokens': 24,
                                 'prompt_tokens_details': {'cached_tokens': 6}}
    assert len(bodies) == 2 and len(lfm.calls) == 1 and all(s.closed for s in streams)
    assert bodies[0]['max_completion_tokens'] == 123
    assert all(b['stream_options'] == {'include_usage': True} for b in bodies)
    assert {t['function']['name'] for t in bodies[0]['tools']} == {'terminal', HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}
    item = app.state.context_compaction_store.items('synthetic-conversation')[0]
    assert item.compaction_source == 'lfm' and SUMMARY in item.compacted
    assert output(bodies[0]) == SOURCE and output(bodies[1]) == item.compacted
    assert json.loads(output(bodies[1], 'private-call'))['ok']
    assert all(b['messages'][:2] == MESSAGES[:2] for b in bodies)
    assert ARGS not in json.dumps(lfm.calls) and MESSAGES[0]['content'] not in json.dumps(lfm.calls)
    packet = json.loads(lfm.calls[0]['messages'][1]['content'].split('\n\n', 1)[1])
    assert set(packet) == {'result', 'required_evidence'}
    assert app.state.context_compaction_store.last_measurement.provider_calls == 2


@pytest.mark.parametrize('case', ['parallel', 'round_limit'])
def test_private_loop_limits_lfm_once(bridge, monkeypatch, case):
    client, app, bodies, replies, lfm, streams = bridge
    if case == 'round_limit':
        monkeypatch.setenv('CONTEXT_COMPACTION_MAX_INTERNAL_ROUNDS', '1')
        replies.extend([wire(call(HIDE_TOOL, {'tool_call_id': 'original-call'})) for _ in range(2)])
        assert public(post(client))[0]['error']['code'] == 'context_compaction_loop_limit'
    else:
        messages = copy.deepcopy(MESSAGES)
        messages.extend([{'role': 'assistant', 'content': None, 'tool_calls': [call('terminal', json.loads(ARGS), 'second-call')]},
                         {'role': 'tool', 'tool_call_id': 'second-call', 'content': SOURCE + '\nsecond result'}])
        replies.extend([wire(call(HIDE_TOOL, {'tool_call_id': 'original-call'}, 'hide-one'),
                             call(HIDE_TOOL, {'tool_call_id': 'second-call'}, 'hide-two')), wire(content='done')])
        assert text(public(post(client, messages=messages))) == 'done'
        assert json.loads(output(bodies[-1], 'hide-one'))['ok'] and json.loads(output(bodies[-1], 'hide-two'))['ok']
    assert len(lfm.calls) == 1 and len(bodies) == 2 and all(s.closed for s in streams)


def test_after_list_exact_restore(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    replies.extend([wire(call(HIDE_TOOL, {'tool_call_id': 'original-call'})), wire(content='hidden')])
    public(post(client))
    item = app.state.context_compaction_store.items('synthetic-conversation')[0]
    replies.append(wire(content='after'))
    public(post(client, tools=None))
    assert output(bodies[-1]) == item.compacted
    replies.extend([wire(call(LIST_TOOL, {}, 'list-call')), wire(content='listed')])
    public(post(client))
    assert json.loads(output(bodies[-1], 'list-call'))['items'][0]['item_id'] == item.item_id
    rendered = copy.deepcopy(MESSAGES)
    rendered[2]['content'] = item.compacted
    replies.extend([wire(call(UNHIDE_TOOL, {'item_id': item.item_id}, 'restore-call')), wire(content='restored')])
    public(post(client, messages=rendered))
    assert output(bodies[-1]) == SOURCE
    replies.append(wire(content='still restored'))
    public(post(client, messages=rendered))
    assert output(bodies[-1]) == SOURCE and len(lfm.calls) == 1


@pytest.mark.parametrize('mixed', [False, True])
def test_fragmented_parallel_external_calls_and_private_mixed_suppression(bridge, mixed):
    client, app, bodies, replies, lfm, streams = bridge
    a, b = call('terminal', {'command': 'a'}, 'public-a'), call('terminal', {'command': 'b'}, 'public-b')
    a['function']['arguments'] = '{ "command" : "a" }'
    calls = ([call(HIDE_TOOL, {'tool_call_id': 'original-call'})] if mixed else []) + [a, b]
    replies.append(wire(*calls, content='public explanation'))
    response = post(client)
    rows = public(response)
    actual = [c for r in rows for ch in r.get('choices', []) for c in ch['delta'].get('tool_calls', [])]
    assert actual == [{'index': i, **c} for i, c in enumerate([a, b])]
    assert rows[-2]['choices'][0]['finish_reason'] == 'tool_calls'
    assert text(rows) == 'public explanation' and HIDE_TOOL not in response.text and 'private-call' not in response.text
    assert not lfm.calls and len(bodies) == 1


@pytest.mark.parametrize('skip', ['disabled', 'header', 'collision', 'forced'])
def test_exclusions_keep_default_fragment_stream(bridge, monkeypatch, skip):
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
    replies.append(wire(content='default text'))
    rows = public(post(client, **kwargs))
    assert [ch['delta']['content'] for r in rows for ch in r.get('choices', []) if 'content' in ch['delta']] == ['def', 'ault text']
    names = {t['function']['name'] for t in bodies[0]['tools']}
    assert names == ({'terminal', HIDE_TOOL} if skip == 'collision' else {'terminal'})
    assert output(bodies[0]) == SOURCE and not lfm.calls
    assert 'stream_options' not in bodies[0]


@pytest.mark.parametrize('failure', ['read', 'eof', 'error', 'done_no_finish', 'finish_no_done', 'invalid_finish'])
def test_partial_failure_no_success_no_private_leak(bridge, failure):
    client, app, bodies, replies, lfm, streams = bridge
    replies.append(wire(call(HIDE_TOOL, {'tool_call_id': 'original-call'})))
    payload = sse(delta('PRIVATE partial'))
    if failure == 'read':
        payload = Chunked(payload, fail=True)
    elif failure == 'error':
        payload += sse({'error': {'message': 'synthetic rejection', 'code': 'synthetic_failure'}})
    elif failure == 'done_no_finish':
        payload += b'data: [DONE]\n\n'
    elif failure == 'finish_no_done':
        payload = wire(content='PRIVATE partial')[:-len(b'data: [DONE]\n\n')]
    elif failure == 'invalid_finish':
        payload = wire(content='PRIVATE partial').replace(b'"finish_reason": "stop"', b'"finish_reason": ""')
    replies.append(payload)
    response = post(client)
    rows = public(response)
    assert rows[0].get('error') and all('choices' not in r and 'usage' not in r for r in rows)
    assert 'PRIVATE' not in response.text and HIDE_TOOL not in response.text
    assert all(s.closed for s in streams)
    assert app.state.context_compaction_store.items('synthetic-conversation')[0].original == SOURCE


@pytest.mark.parametrize('usage', [None, {}, {'prompt_tokens': 10}, {'prompt_tokens': True, 'completion_tokens': 2},
                                  {'prompt_tokens': 10, 'completion_tokens': -1}, {'prompt_tokens': 10, 'completion_tokens': '2'},
                                  {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': True},
                                  {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': -1},
                                  {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': '12'},
                                  {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'prompt_tokens_details': {'cached_tokens': 0}}])
def test_usage_unknown_not_zero_and_exact_zero_preserved(bridge, usage):
    client, app, bodies, replies, lfm, streams = bridge
    payload = wire(content='done', usage=False)[:-len(b'data: [DONE]\n\n')]
    if usage is not None:
        payload += sse({'choices': [], 'usage': usage})
    replies.append(payload + b'data: [DONE]\n\n')
    rows = public(post(client))
    usages = [r['usage'] for r in rows if 'usage' in r]
    assert usages == ([usage] if usage and usage.get('prompt_tokens') == 0 else [])


@pytest.mark.parametrize('case', ['missing_round', 'missing_cache', 'invalid_cache', 'opt_out', 'length'])
def test_usage_round_aggregation_cache_opt_out_and_finish(bridge, case):
    client, app, bodies, replies, lfm, streams = bridge
    replies.extend([wire(call(HIDE_TOOL, {'tool_call_id': 'original-call'})),
                    wire(content='done', usage=case != 'missing_round', cached=None if case == 'missing_cache' else True if case == 'invalid_cache' else 3,
                         finish='length' if case == 'length' else 'stop')])
    rows = public(post(client, stream_options={'include_usage': case != 'opt_out'}))
    usages = [r['usage'] for r in rows if 'usage' in r]
    expected = {'prompt_tokens': 20, 'completion_tokens': 4, 'total_tokens': 24}
    if case == 'length':
        expected['prompt_tokens_details'] = {'cached_tokens': 6}
    assert usages == ([] if case in ['missing_round', 'opt_out'] else [expected])
    assert next(ch['finish_reason'] for r in reversed(rows) for ch in r.get('choices', []) if ch['finish_reason']) == ('length' if case == 'length' else 'stop')


@pytest.mark.parametrize('case', ['buffer', 'timeout', 'cancel'])
def test_bounds_and_cancellation_close_native_stream(bridge, monkeypatch, case):
    from openai_compatible_bridge.core.cost_tracking import DisabledCostAccounting, NormalizedUsage
    from openai_compatible_bridge.providers.vertex import VertexAPIError
    client, app, bodies, replies, lfm, streams = bridge
    if case == 'buffer':
        replies.append(wire(content='x' * 70000))
        assert public(post(client, max_tokens=1))[0]['error']['code'] == 'context_compaction_stream_limit'
    else:
        async def run():
            started = asyncio.Event()
            class Waiting(Chunked):
                async def __aiter__(self):
                    yield sse(delta('PRIVATE waiting'))
                    started.set()
                    await asyncio.Event().wait()
            replies.append(Waiting(b''))
            ctx = DisabledCostAccounting().reservation(endpoint='chat', model=ALIAS, provider='foundry', forecast_usage=NormalizedUsage())
            kwargs = dict(model='synthetic-chat', messages=MESSAGES, resolved_config={'protocol': None})
            if case == 'timeout':
                monkeypatch.setattr('openai_compatible_bridge.providers.foundry.HTTP_TIMEOUT_SECONDS', 0.01)
                with pytest.raises(VertexAPIError, match='timed out'):
                    await asyncio.wait_for(_collect_compaction_stream(app.state.foundry_chat_client, ctx, **kwargs), 0.1)
            else:
                task = asyncio.create_task(_collect_compaction_stream(app.state.foundry_chat_client, ctx, **kwargs))
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        client.portal.call(run)
    assert all(s.closed for s in streams)


@pytest.mark.parametrize('failure', [None, 'second_admission', 'partial', 'cancel'])
def test_attempt_metering_and_lfm_model_restoration(bridge, monkeypatch, failure):
    from decimal import Decimal
    from openai_compatible_bridge.core.async_cost import AsyncCostAccounting, _ACTIVE
    from openai_compatible_bridge.core.cost_tracking import CostConfigError, CostReservation, NormalizedUsage
    from openai_compatible_bridge.core.metered_http import MeteredHTTPClient
    from openai_compatible_bridge.providers.ollama import OllamaChatClient
    client, app, bodies, replies, lfm, streams = bridge
    admitted, recorded = [], []
    accounting = AsyncCostAccounting(config=None, billing={'foundry': 'metered', 'ollama': 'metered'})
    async def admit(provider):
        admitted.append((provider, _ACTIVE.get().model))
        if failure == 'second_admission' and len(admitted) == 3:
            raise CostConfigError('synthetic denied')
        return CostReservation(str(len(admitted)), 'synthetic', provider, 'chat', _ACTIVE.get().model,
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
        return httpx.Response(200, json={'message': {'role': 'assistant', 'content': json.dumps({'summary': SUMMARY})},
                                        'prompt_eval_count': 7, 'eval_count': 1, 'done': True})
    ollama.http = MeteredHTTPClient(httpx.AsyncClient(transport=httpx.MockTransport(respond)), accounting, provider='ollama')
    app.state.ollama_chat_client = ollama
    if failure != 'cancel':
        replies.extend([wire(call(HIDE_TOOL, {'tool_call_id': 'original-call'})),
                        Chunked(sse(delta('PRIVATE')), fail=True) if failure == 'partial' else wire(content='metered final')])
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
        async def run():
            started = asyncio.Event()
            class Waiting(Chunked):
                async def __aiter__(self):
                    yield sse(delta('PRIVATE'))
                    started.set()
                    await asyncio.Event().wait()
            replies.append(Waiting(b''))
            async with accounting.reservation(endpoint='chat', model=ALIAS, provider='foundry', forecast_usage=NormalizedUsage()) as ctx:
                task = asyncio.create_task(_collect_compaction_stream(provider, ctx, model='synthetic-chat', messages=MESSAGES, resolved_config={'protocol': None}))
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert recorded == [('foundry', None)] and streams[-1].closed
        client.portal.call(run)
    client.portal.call(ollama.close)
    client.portal.call(accounting.aclose)
