"""Origin boundary regressions use MockTransport; never contact a server."""

import asyncio

import httpx
import pytest

from openai_compatible_bridge.context_compaction import load_settings
from openai_compatible_bridge.laya_http import LayaClient, LayaUnavailable

ORIGIN = "http://100.64.0.10:8000"  # synthetic address, never used on the network
QUESTIONS = {"relevance": {"type": "choice", "criteria": {"a": "alpha", "b": "beta"}}}


@pytest.mark.parametrize("approved", ["", ORIGIN])
def test_settings_require_separate_approved_origin(approved):
    settings = load_settings({
        "CONTEXT_COMPACTION_ENABLED": "true",
        "CONTEXT_COMPACTION_LAYA_ENABLED": "true",
        "CONTEXT_COMPACTION_LAYA_VALIDATED": "true",
        "LAYA_BASE_URL": ORIGIN,
        "LAYA_APPROVED_ORIGIN": approved,
    })
    assert settings.laya_approved_origin == approved
    assert settings.laya_active is bool(approved)


@pytest.mark.parametrize("base", ["http://public.example:8000", ORIGIN])
def test_unapproved_origin_is_rejected_before_any_request(base):
    requests = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: (requests.append(req), httpx.Response(200))[1],
    ))
    try:
        with pytest.raises(ValueError):
            LayaClient(base, http=http)
        assert requests == []
    finally:
        asyncio.run(http.aclose())


@pytest.mark.parametrize("operation", ["choose", "health"])
def test_redirect_cannot_forward_request_to_another_origin(operation):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "100.64.0.10":
            return httpx.Response(307, headers={"location": "http://elsewhere.example:8000/target"})
        return httpx.Response(200, json={})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True)
    client = LayaClient(ORIGIN, approved_origin=ORIGIN, http=http)
    try:
        if operation == "choose":
            with pytest.raises(LayaUnavailable):
                asyncio.run(client.choose("synthetic goal", QUESTIONS, "relevance"))
        else:
            assert not asyncio.run(client.health())
        assert len(requests) == 1
    finally:
        asyncio.run(client.close())


@pytest.mark.parametrize("base,approved", [
    ("http://public.example:8000", "http://public.example:8000"),
    ("http://8.8.8.8:8000", "http://8.8.8.8:8000"),
    ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
    ("http://10.0.0.10:8000", "http://10.0.0.10:8000"),
    ("http://laya.internal:8000", "http://laya.internal:8000"),
    ("http://100.64.0.11:8000", ORIGIN),
    ("https://100.64.0.10:8000", ORIGIN),
    ("http://100.64.0.10:8001", ORIGIN),
    ("http://100.64.0.10:0", "http://100.64.0.10"),
    ("http://100.64.0.10:99999", ORIGIN),
    ("http://100.64.0.10:8000/private", ORIGIN),
    ("http://100.64.0.10:8000?q=x", ORIGIN),
    ("http://user:synthetic@100.64.0.10:8000", ORIGIN),
])
def test_approval_is_exact_and_cannot_allow_public_dns_or_other_private_networks(base, approved):
    requests = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: (requests.append(req), httpx.Response(200))[1],
    ))
    try:
        with pytest.raises(ValueError):
            LayaClient(base, approved_origin=approved, http=http)
        assert requests == []
    finally:
        asyncio.run(http.aclose())
