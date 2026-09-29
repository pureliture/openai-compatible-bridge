"""Laya v0.3.21 wire contract, using synthetic data and MockTransport only."""

import asyncio
import json

import httpx
import pytest

from openai_compatible_bridge.laya_http import LayaClient, LayaUnavailable


def _response(choice="line_2", *, routing="multilingual", probability=0.91):
    return {
        "model": "laya-rl-agent",
        "answers": {
            "relevance": {
                "type": "choice",
                "choice": choice,
                "probabilities": {"line_1": 0.09, choice: probability},
                "answer_confidence": probability,
                "confidence": 0.5,
            }
        },
        "routing": {"model": routing},
        "usage": {"input_tokens": 17, "output_tokens": 0},
    }


def test_laya_uses_systemone_multilingual_and_parses_choice():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=_response())

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = LayaClient("http://laya.example:8000", http=http)
    question = {"relevance": {"type": "choice", "instructions": "Choose the relevant line", "criteria": {"line_1": "a", "line_2": "b"}}}
    choice = asyncio.run(client.choose("current task", question, "relevance"))
    assert choice == "line_2"
    assert str(requests[0].url) == "http://laya.example:8000/v1/systemone"
    assert requests[0].headers["content-type"] == "application/json"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "multilingual"
    assert payload["state"] == "current task"
    assert payload["questions"] == question
    assert payload["max_len"] <= 8192
    assert "Authorization" not in requests[0].headers
    asyncio.run(client.close())


@pytest.mark.parametrize("response", [
    _response(routing="english"),
    _response(choice="invented"),
    _response(probability=0.50),
    {"model": "multilingual", "answers": {}, "routing": {"model": "multilingual"}},
    {"answers": {"relevance": {"choice": "line_2", "probabilities": {"line_2": True}}}, "routing": {"model": "multilingual"}},
])
def test_laya_rejects_unverified_and_ambiguous_answers(response):
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)))
    client = LayaClient("http://laya.example:8000", http=http)
    questions = {"relevance": {"type": "choice", "instructions": "select", "criteria": {"line_1": "a", "line_2": "b"}}}
    with pytest.raises(LayaUnavailable):
        asyncio.run(client.choose("goal", questions, "relevance"))
    asyncio.run(client.close())


@pytest.mark.parametrize("status", [400, 401, 413, 422, 500, 503])
def test_laya_http_errors_do_not_expose_server_detail(status):
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status, json={"detail": "private error"})))
    client = LayaClient("http://laya.example:8000", http=http)
    questions = {"relevance": {"type": "choice", "instructions": "select", "criteria": {"line_1": "a"}}}
    with pytest.raises(LayaUnavailable) as exc:
        asyncio.run(client.choose("goal", questions, "relevance"))
    assert "private error" not in str(exc.value)
    assert str(status) in str(exc.value)
    asyncio.run(client.close())


def test_laya_timeout_and_bad_url_fail_closed():
    async def timeout(_):
        raise httpx.ReadTimeout("private request body")

    http = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    client = LayaClient("http://laya.example:8000", http=http)
    questions = {"relevance": {"type": "choice", "instructions": "select", "criteria": {"line_1": "a"}}}
    with pytest.raises(LayaUnavailable) as exc:
        asyncio.run(client.choose("goal", questions, "relevance"))
    assert "private request body" not in str(exc.value)
    asyncio.run(client.close())
    for base in ("", "http://", "http://user:password@laya.example:8000", "http://laya.example:8000/v1/systemone", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            LayaClient(base)
