import asyncio

import httpx
import pytest

from scripts.check_laya_remote import probe


def test_remote_probe_checks_health_checkpoint_and_routing_without_raw_body():
    requests = []
    def handle(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "loaded": ["multilingual"],
                                             "device": "cpu", "private": "secret"})
        return httpx.Response(200, json={"model": "laya-rl-agent",
                                         "routing": {"model": "multilingual"},
                                         "answers": {"kind": {"type": "choice", "choice": "billing"}},
                                         "usage": {"input_tokens": 7, "output_tokens": 0}})
    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    result = asyncio.run(probe("http://laya.example:8000", http))
    assert result["status"] == "ok"
    assert result["health_ok"] and result["multilingual_loaded"] and result["inference_ok"]
    assert result["inference_seconds"] is not None
    assert [req.url.path for req in requests] == ["/health", "/v1/systemone"]
    assert "private" not in str(result)
    asyncio.run(http.aclose())


@pytest.mark.parametrize("loaded, routing, expected_calls", [(["english"], "english", 1),
                                                         (["multilingual"], "english", 2)])
def test_remote_probe_refuses_unloaded_or_wrong_checkpoint(loaded, routing, expected_calls):
    calls = []
    def handle(request):
        calls.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "loaded": loaded, "device": "cpu"})
        return httpx.Response(200, json={"routing": {"model": routing},
                                         "answers": {"kind": {"choice": "billing"}}})
    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    result = asyncio.run(probe("http://laya.example:8000", http))
    assert not result["inference_ok"]
    assert len(calls) == expected_calls
    asyncio.run(http.aclose())
