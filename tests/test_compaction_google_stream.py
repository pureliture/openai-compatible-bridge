"""Google native chunked SSE through real Foundry stream_chat; no paid traffic."""
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
from test_compaction_google import ALIAS, AFFINITY, native_call, parts, output, original_output, declarations, assert_original_call
from test_compaction_responses_stream import Chunked, public, text, sse


def event(*calls, text=None, finish=None, usage=None):
    candidate = {'content': {'role': 'model', 'parts': list(calls) + ([{'text': text}] if text is not None else [])}}
    if finish is not None:
        candidate['finishReason'] = finish
    row = {'candidates': [candidate]}
    if usage is not None:
        row['usageMetadata'] = usage
    return sse(row)


def wire(*calls, text=None, usage=None, finish='STOP'):
    usage = {'promptTokenCount': 10, 'candidatesTokenCount': 2, 'totalTokenCount': 12,
             'cachedContentTokenCount': 3} if usage is None else usage
    return (event(text=text[:3]) + event(text=text[3:]) if text is not None else b'') + event(*calls, finish=finish, usage=usage)


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv('CONTEXT_COMPACTION_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LFM_ENABLED', 'true')
    monkeypatch.setenv('CONTEXT_COMPACTION_LAYA_ENABLED', 'false')
    monkeypatch.setattr('openai_compatible_bridge.main.BRIDGE_API_KEY', '')
    monkeypatch.setitem(vertex.MODEL_REGISTRY, ALIAS, {'provider': 'foundry', 'kind': 'chat',
                        'provider_model': 'synthetic-google', 'protocol': 'google_generate_content'})
    bodies, replies, streams = [], [], []
    async def handler(request):
        assert request.url.path == '/api/v2/llm/proxy/google/v1/models/synthetic-google:streamGenerateContent'
        assert request.url.query == b'alt=sse'
        body = json.loads(request.content)
        assert 'stream' not in body and 'messages' not in body
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
        pytest.fail('generate fallback forbidden')
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


def test_native_hide_metadata_final_external_usage_private_nonleak(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    private = native_call(HIDE_TOOL, {'tool_call_id': 'original-call'}, 'private-native', 'private-signature')
    external = native_call('terminal', {'command': 'public next', 'timeout': 37}, 'public-id', 'public-signature')
    replies.extend([wire(private, text='PRIVATE intermediate'), wire(external, text='final public')])
    response = post(client, max_tokens=123)
    rows = public(response)
    assert text(rows) == 'final public'
    assert all(secret not in response.text for secret in (HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL, 'PRIVATE', 'private-native', 'private-signature', '_google', 'public-signature'))
    calls = [call for row in rows for c in row.get('choices', []) for call in c['delta'].get('tool_calls', [])]
    assert calls == [{'index': 0, 'id': 'public-id', 'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps(external['functionCall']['args'])}}]
    assert rows[-2]['choices'][0]['finish_reason'] == 'tool_calls'
    assert rows[-1]['usage'] == {'prompt_tokens': 20, 'completion_tokens': 4, 'total_tokens': 24, 'prompt_tokens_details': {'cached_tokens': 6}}
    assert len(bodies) == 2 and len(lfm.calls) == 1
    assert bodies[0]['generationConfig']['maxOutputTokens'] == 123
    assert {t['name'] for t in declarations(bodies[0])} == {'terminal', HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL}
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    assert item.compaction_source == 'lfm' and SUMMARY in item.compacted
    assert original_output(bodies[0]) == SOURCE and original_output(bodies[1]) == item.compacted
    assert private in parts(bodies[1], 'functionCall')
    result = next(p['functionResponse'] for p in parts(bodies[1], 'functionResponse') if p['functionResponse']['name'] == HIDE_TOOL)
    assert result['id'] == 'private-native' and result['response']['ok']
    for body in bodies:
        assert_original_call(body)
    assert app.state.context_compaction_store.last_measurement.provider_calls == 2
    assert all(s.closed for s in streams)


def test_private_round_missing_cache_omits_aggregate_cache(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    replies.extend([wire(native_call(LIST_TOOL, {})), wire(text='done', usage={'promptTokenCount': 10, 'candidatesTokenCount': 2, 'totalTokenCount': 12})])
    rows = public(post(client))
    assert rows[-1]['usage'] == {'prompt_tokens': 20, 'completion_tokens': 4, 'total_tokens': 24}


def test_native_metadata_buffer_bound_closes_without_public_success(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    replies.append(event(native_call(LIST_TOOL, {}, 'private', 'x' * 70000)))
    response = post(client, max_tokens=1)
    rows = public(response)
    assert rows[0]['error']['code'] == 'context_compaction_stream_limit'
    assert LIST_TOOL not in response.text and not lfm.calls
    assert all(s.closed for s in streams)


@pytest.mark.parametrize('terminal_usage', [None, {'promptTokenCount': 10}])
def test_provisional_usage_followed_by_missing_terminal_usage_not_public(bridge, terminal_usage):
    client, app, bodies, replies, lfm, streams = bridge
    replies.append(event(text='done', usage={'promptTokenCount': 10, 'candidatesTokenCount': 0, 'totalTokenCount': 10}) + event(finish='STOP', usage=terminal_usage))
    rows = public(post(client))
    assert not any('usage' in row for row in rows)


def test_two_same_name_noid_hides_keep_parallel_results_and_lfm_once(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    messages = copy.deepcopy(MESSAGES)
    messages[1]['tool_calls'].append({'id': 'second-call', 'type': 'function', 'function': {'name': 'terminal', 'arguments': ARGS}})
    messages.append({'role': 'tool', 'tool_call_id': 'second-call', 'content': SOURCE + '\nsecond result'})
    calls = [native_call(HIDE_TOOL, {'tool_call_id': call_id}, signature='signature-' + call_id) for call_id in ('original-call', 'second-call')]
    replies.extend([event(*calls) + wire(*calls), wire(text='done')])
    assert text(public(post(client, messages=messages))) == 'done'
    assert len(lfm.calls) == 1
    assert bodies[-1]['contents'][-2]['parts'] == calls
    results = bodies[-1]['contents'][-1]['parts']
    assert len(results) == 2
    assert [r['functionResponse']['response']['tool_call_id'] for r in results] == ['original-call', 'second-call']
    assert all('id' not in r['functionResponse'] for r in results)


def test_repeated_terminal_snapshot_does_not_duplicate_call_arguments(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    call = native_call('terminal', {'command': 'once'}, 'one')
    replies.append(wire(call) + wire(call))
    rows = public(post(client))
    actual = [call for row in rows for c in row.get('choices', []) for call in c['delta'].get('tool_calls', [])]
    assert len(actual) == 1
    assert actual[0]['function']['name'] == 'terminal'
    assert json.loads(actual[0]['function']['arguments']) == {'command': 'once'}


def test_hide_after_list_exact_unhide(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    original = copy.deepcopy(MESSAGES)
    replies.extend([wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})), wire(text='hidden')])
    assert text(public(post(client))) == 'hidden'
    item = app.state.context_compaction_store.items(AFFINITY)[0]
    replies.append(wire(text='after'))
    assert text(public(post(client, tools=None))) == 'after'
    assert original_output(bodies[-1]) == item.compacted
    replies.extend([wire(native_call(LIST_TOOL, {}, 'list-id', 'list-signature')), wire(text='listed')])
    assert text(public(post(client))) == 'listed'
    assert output(bodies[-1], LIST_TOOL)['items'][0]['item_id'] == item.item_id
    rendered = copy.deepcopy(MESSAGES)
    rendered[2]['content'] = item.compacted
    replies.extend([wire(native_call(UNHIDE_TOOL, {'item_id': item.item_id}, 'restore-id')), wire(text='restored')])
    assert text(public(post(client, messages=rendered))) == 'restored'
    assert original_output(bodies[-1]) == SOURCE
    replies.append(wire(text='still restored'))
    public(post(client, messages=rendered))
    assert original_output(bodies[-1]) == SOURCE and len(lfm.calls) == 1
    assert MESSAGES == original and rendered[2]['content'] == item.compacted
    for body in bodies:
        assert_original_call(body)
    assert all(s.closed for s in streams)


@pytest.mark.parametrize('native_ids', [False, True])
def test_parallel_repeated_partial_snapshots_and_noid_collision(bridge, native_ids):
    client, app, bodies, replies, lfm, streams = bridge
    messages = copy.deepcopy(MESSAGES)
    messages[1]['tool_calls'][0]['id'] = 'call_terminal_0'
    messages[2]['tool_call_id'] = 'call_terminal_0'
    first = [native_call('terminal', {'command': command}, 'native-' + command if native_ids else None) for command in ('a', 'b')]
    complete = [native_call('terminal', {'command': command, 'timeout': 37}, 'native-' + command if native_ids else None, 'signature-' + command) for command in ('a', 'b')]
    replies.append(event(*first) + event(*first) + event(*complete) + wire(*complete))
    rows = public(post(client, messages=messages))
    actual = [call for row in rows for c in row.get('choices', []) for call in c['delta'].get('tool_calls', [])]
    assert len(actual) == 2 and len({c['id'] for c in actual}) == 2
    assert all(c['id'] != 'call_terminal_0' for c in actual)
    assert [json.loads(c['function']['arguments']) for c in actual] == [{'command': 'a', 'timeout': 37}, {'command': 'b', 'timeout': 37}]
    assert rows[-2]['choices'][0]['finish_reason'] == 'tool_calls'


def test_native_id_order_repeated_same_name_private_results(bridge):
    client, app, bodies, replies, lfm, streams = bridge
    a = native_call(LIST_TOOL, {}, 'a', 'signature-a')
    b = native_call(LIST_TOOL, {'query': 'second'}, 'b', 'signature-b')
    replies.extend([event(a, b) + event(b, a) + wire(), wire(text='done')])
    assert text(public(post(client))) == 'done'
    assert bodies[-1]['contents'][-2]['parts'] == [a, b]
    responses = bodies[-1]['contents'][-1]['parts']
    assert [r['functionResponse']['id'] for r in responses] == ['a', 'b']
    assert all(r['functionResponse']['response'] == {
        'ok': True, 'items': [], 'hidden_count': 0, 'saved_bytes': 0,
    } for r in responses)
    assert not lfm.calls


@pytest.mark.parametrize('mixed', [False, True])
def test_mixed_external_no_private_execution(bridge, mixed):
    client, app, bodies, replies, lfm, streams = bridge
    calls = ([native_call(HIDE_TOOL, {'tool_call_id': 'original-call'}, 'private')] if mixed else []) + [native_call('terminal', {'command': 'a'}, 'a'), native_call('terminal', {'command': 'b'}, 'b')]
    replies.append(wire(*calls, text='explanation'))
    response = post(client)
    rows = public(response)
    actual = [call for row in rows for c in row.get('choices', []) for call in c['delta'].get('tool_calls', [])]
    assert [(c['id'], json.loads(c['function']['arguments'])) for c in actual] == [('a', {'command': 'a'}), ('b', {'command': 'b'})]
    assert HIDE_TOOL not in response.text and not lfm.calls and len(bodies) == 1


@pytest.mark.parametrize('case', ['missing', 'partial', 'invalid', 'zero', 'missing_cache', 'invalid_cache', 'length', 'thoughts'])
def test_usage_no_fabrication_finish_cache(bridge, case):
    client, app, bodies, replies, lfm, streams = bridge
    usage = {'promptTokenCount': 10, 'candidatesTokenCount': 2, 'totalTokenCount': 12, 'cachedContentTokenCount': 3}
    if case == 'missing': usage = {}
    if case == 'partial': usage.pop('candidatesTokenCount')
    if case == 'invalid': usage['promptTokenCount'] = True
    if case == 'zero': usage = dict.fromkeys(usage, 0)
    if case == 'missing_cache': usage.pop('cachedContentTokenCount')
    if case == 'invalid_cache': usage['cachedContentTokenCount'] = True
    if case == 'thoughts': usage.update(totalTokenCount=16, thoughtsTokenCount=4)
    replies.append(wire(text='done', usage=usage, finish='MAX_TOKENS' if case == 'length' else 'STOP'))
    rows = public(post(client))
    usages = [r['usage'] for r in rows if 'usage' in r]
    if case in {'missing', 'partial', 'invalid'}:
        assert usages == []
    else:
        expected = {'prompt_tokens': 0 if case == 'zero' else 10, 'completion_tokens': 0 if case == 'zero' else 2, 'total_tokens': 0 if case == 'zero' else 16 if case == 'thoughts' else 12}
        if case not in {'missing_cache', 'invalid_cache'}: expected['prompt_tokens_details'] = {'cached_tokens': 0 if case == 'zero' else 3}
        assert usages == [expected]
    assert rows[-(2 if usages else 1)]['choices'][0]['finish_reason'] == ('length' if case == 'length' else 'stop')


@pytest.mark.parametrize('failure', ['read', 'eof', 'blocked', 'event'])
def test_incomplete_block_error_no_private_leak(bridge, failure):
    client, app, bodies, replies, lfm, streams = bridge
    replies.append(wire(native_call(HIDE_TOOL, {'tool_call_id': 'original-call'})))
    partial = event(text='PRIVATE partial')
    if failure == 'read': partial = Chunked(partial, fail=True)
    if failure == 'blocked': partial += sse({'promptFeedback': {'blockReason': 'SAFETY'}})
    if failure == 'event': partial += sse({'error': {'message': 'synthetic failure', 'code': 'synthetic_error'}})
    replies.append(partial)
    response = post(client)
    rows = public(response)
    assert rows[0].get('error')
    assert all('choices' not in r and 'usage' not in r for r in rows)
    assert 'PRIVATE' not in response.text and HIDE_TOOL not in response.text
    assert all(s.closed for s in streams)
    assert app.state.context_compaction_store.items(AFFINITY)[0].original == SOURCE


@pytest.mark.parametrize('skip', ['disabled', 'header', 'collision', 'forced'])
def test_excluded_stream_path_unchanged(bridge, monkeypatch, skip):
    client, app, bodies, replies, lfm, streams = bridge
    kwargs = {}
    if skip == 'disabled': monkeypatch.setenv('CONTEXT_COMPACTION_ENABLED', 'false')
    elif skip == 'header': kwargs['headers'] = {}
    elif skip == 'collision': kwargs['tools'] = [TOOL, {'type': 'function', 'function': {'name': HIDE_TOOL}}]
    else: kwargs['tool_choice'] = {'type': 'function', 'function': {'name': 'terminal'}}
    replies.append(wire(text='default text'))
    rows = public(post(client, **kwargs))
    assert [c['delta']['content'] for r in rows for c in r.get('choices', []) if 'content' in c['delta']] == ['def', 'ault text']
    assert {t['name'] for t in declarations(bodies[0])} == ({'terminal', HIDE_TOOL} if skip == 'collision' else {'terminal'})
    assert not lfm.calls and original_output(bodies[0]) == SOURCE


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
                    yield event(text='PRIVATE')
                    entered.set()
                    await asyncio.Event().wait()

            replies.append(Waiting(b''))
            ctx = DisabledCostAccounting().reservation(endpoint='chat', model=ALIAS, provider='foundry', forecast_usage=NormalizedUsage())
            kwargs = dict(model='synthetic-google', messages=MESSAGES, resolved_config={'protocol': 'google_generate_content'})
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
    partial = event(text='PRIVATE')
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
                task = asyncio.create_task(_collect_compaction_stream(provider, ctx, model='synthetic-google', messages=MESSAGES,
                                          resolved_config={'protocol': 'google_generate_content'}))
                await asyncio.wait_for(entered.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert recorded == [('foundry', None)] and streams[-1].closed
        client.portal.call(cancel)
    assert all(s.closed for s in streams)
    client.portal.call(ollama.close)
    client.portal.call(accounting.aclose)