"""Synthetic native HTTP final-review regressions; no paid traffic or transcripts."""
import asyncio
import copy
import json
from typing import Any

import httpx
import pytest

from openai_compatible_bridge.core.cost_tracking import DisabledCostAccounting, NormalizedUsage
from openai_compatible_bridge.main import _collect_compaction_stream
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError
from test_compaction_chat_stream import wire
from test_metered_http import Accounting, ByteStream, wrap
from fastapi.testclient import TestClient
import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.context_compaction import AFFINITY_HEADER, HIDE_TOOL
from openai_compatible_bridge.main import create_app
from test_compaction_foundry_protocols import MESSAGES, TOOL, SyntheticLFM, Unused
from test_compaction_chat_stream import call as chat_call, sse
from test_compaction_responses_stream import wire as responses_wire
from test_compaction_foundry_protocols import native_call as responses_call
from test_compaction_anthropic_stream import wire as anthropic_wire
from test_compaction_anthropic import native_call as anthropic_call
from test_compaction_google_stream import wire as google_wire
from test_compaction_google import native_call as google_call, native_response as google_response


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
        stream = ByteStream(reply)
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
def test_private_native_stream_error_is_redacted(review_bridge, protocol, safe_code):
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

