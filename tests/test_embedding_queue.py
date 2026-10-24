from __future__ import annotations

import asyncio
import json
import socket
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from openai_compatible_bridge.embedding_limits import EmbeddingRequestLimits
from openai_compatible_bridge.main import create_app


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("embedding queue tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket.socket, "connect_ex", fail)


class _SlowApp:
    """외부 호출 없이 처리 수와 실제 동시 실행 수를 기록한다."""

    def __init__(self, delay=0.15):
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.messages = [
            {"type": "http.response.start", "status": 200,
             "headers": [(b"content-type", b"application/json"), (b"x-test", b"unchanged")]},
            {"type": "http.response.body", "body": b'{"object":"list","data":[],"usage":{"total_tokens":0}}'},
        ]

    async def __call__(self, scope, receive, send):
        await receive()
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            for message in self.messages:
                await send(message)
        finally:
            self.active -= 1


class _Request:
    def __init__(self, body=b"{}", *, headers=None, more_body=False):
        self.scope = {
            "type": "http", "method": "POST", "path": "/v1/embeddings",
            "headers": [(b"content-length", str(len(body)).encode())] if headers is None else headers,
        }
        self.events = asyncio.Queue()
        self.events.put_nowait({"type": "http.request", "body": body, "more_body": more_body})
        self.messages = []
        self.response_at = None

    @property
    def status(self):
        return next((message["status"] for message in self.messages
                     if message["type"] == "http.response.start"), None)

    async def send(self, message):
        if message["type"] == "http.response.start":
            self.response_at = time.monotonic()
        self.messages.append(message)

    async def run(self, app):
        await app(self.scope, self.events.get, self.send)
        return self.status


async def _wait_for(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


def _idle(limiter):
    assert limiter.inflight == limiter.queue_depth == 0


def test_saturated_requests_wait_instead_of_429(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "2")
    app = _SlowApp(delay=0.15)
    limiter = EmbeddingRequestLimits(app, max_inflight=1)

    async def run():
        requests = [_Request(), _Request()]
        results = await asyncio.gather(*(request.run(limiter) for request in requests))
        assert results == [200, 200]
        assert app.calls == 2
        assert app.peak == 1
        assert all(request.messages == app.messages for request in requests)
        _idle(limiter)
        print(f"queue_success: statuses={results}, calls={app.calls}, inflight=0, queue_depth=0")

    asyncio.run(run())


def test_queue_timeout_still_returns_429(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "0.05")
    app = _SlowApp(delay=0.4)
    limiter = EmbeddingRequestLimits(app, max_inflight=1)

    async def run():
        requests = [_Request(), _Request()]
        started = time.monotonic()
        results = await asyncio.gather(*(request.run(limiter) for request in requests))
        assert sorted(results) == [200, 429]
        assert app.calls == 1
        rejected = next(request for request in requests if request.status == 429)
        assert rejected.response_at - started >= 0.05
        assert (b"retry-after", b"2") in rejected.messages[0]["headers"]
        assert json.loads(rejected.messages[1]["body"])["error"]["code"] == "embedding_concurrency_limit"
        _idle(limiter)
        print(f"queue_timeout: statuses={sorted(results)}, calls={app.calls}, inflight=0, queue_depth=0")

    asyncio.run(run())


@pytest.mark.parametrize("headers,chunks", [
    ([(b"content-length", b"3")], [b"abc"]),
    ([], [b"a", b"bc"]),
])
def test_oversized_body_returns_413(monkeypatch, headers, chunks):
    monkeypatch.setenv("EMBEDDINGS_MAX_REQUEST_BYTES", "2")
    app = _SlowApp()
    limiter = EmbeddingRequestLimits(app)

    async def run():
        request = _Request(chunks[0], headers=headers, more_body=len(chunks) > 1)
        for chunk in chunks[1:]:
            request.events.put_nowait({"type": "http.request", "body": chunk, "more_body": False})
        assert await request.run(limiter) == 413
        assert app.calls == 0
        _idle(limiter)
        print("oversized_body: status=413, calls=0, inflight=0, queue_depth=0")

    asyncio.run(run())


@pytest.mark.parametrize("lengths", [
    [b"1"], [b"3"], [b"invalid"], [b"-1"], [b"1.5"], [b""], [b"2", b"2"],
])
def test_invalid_content_length_returns_400(lengths):
    app = _SlowApp()
    limiter = EmbeddingRequestLimits(app)

    async def run():
        request = _Request(headers=[(b"content-length", length) for length in lengths])
        assert await request.run(limiter) == 400
        assert app.calls == 0
        _idle(limiter)
        print("invalid_content_length: status=400, calls=0, inflight=0, queue_depth=0")

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["body", "queued", "upstream"])
def test_disconnect_returns_capacity(monkeypatch, phase):
    monkeypatch.setenv("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "2")
    app = _SlowApp(delay=0.15)
    limiter = EmbeddingRequestLimits(app, max_inflight=1)

    async def run():
        active = None
        if phase == "queued":
            active = asyncio.create_task(_Request().run(limiter))
            await _wait_for(lambda: app.calls == 1)
        request = _Request(headers=[], more_body=phase == "body")
        task = asyncio.create_task(request.run(limiter))
        if phase == "queued":
            await _wait_for(lambda: limiter.queue_depth == 1)
        else:
            await _wait_for(lambda: limiter.inflight == 1 and (phase == "body" or app.calls == 1))
        request.events.put_nowait({"type": "http.disconnect"})
        assert await asyncio.wait_for(task, 0.1) is None
        assert limiter.queue_depth == 0
        assert limiter.inflight == (1 if active else 0)
        if active:
            assert await active == 200
        _idle(limiter)
        assert await _Request().run(limiter) == 200
        _idle(limiter)
        print(f"disconnect_{phase}: inflight=0, queue_depth=0, replacement=200")

    asyncio.run(run())


@pytest.mark.parametrize("queued", [False, True])
def test_task_cancellation_returns_capacity(monkeypatch, queued):
    monkeypatch.setenv("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "2")
    app = _SlowApp(delay=0.15)
    limiter = EmbeddingRequestLimits(app, max_inflight=1)

    async def run():
        active = None
        if queued:
            active = asyncio.create_task(_Request().run(limiter))
            await _wait_for(lambda: app.calls == 1)
        task = asyncio.create_task(_Request().run(limiter))
        await _wait_for(lambda: limiter.queue_depth == 1 if queued else app.calls == 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert limiter.queue_depth == 0
        if active:
            assert await active == 200
        _idle(limiter)

    asyncio.run(run())


def test_downstream_exception_returns_capacity():
    async def fail(scope, receive, send):
        await receive()
        raise RuntimeError("simulated failure")

    limiter = EmbeddingRequestLimits(fail, max_inflight=1)

    async def run():
        with pytest.raises(RuntimeError, match="simulated failure"):
            await _Request().run(limiter)
        _idle(limiter)

    asyncio.run(run())


def test_burst_never_exceeds_capacity(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "2")
    app = _SlowApp(delay=0.005)
    limiter = EmbeddingRequestLimits(app, max_inflight=3)

    async def run():
        results = await asyncio.gather(*(_Request().run(limiter) for _ in range(50)))
        assert results == [200] * 50
        assert app.calls == 50
        assert app.peak == 3
        _idle(limiter)
        print("burst: 50x200, calls=50, peak=3, capacity=3, inflight=0, queue_depth=0")

    asyncio.run(run())


@pytest.mark.parametrize("name,value", [
    ("EMBEDDINGS_MAX_INFLIGHT", "0"), ("EMBEDDINGS_MAX_INFLIGHT", "-1"),
    ("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "-0.1"),
    ("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "nan"),
    ("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "inf"),
    ("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "invalid"),
])
def test_invalid_limits_fail_at_startup(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        with TestClient(create_app()):
            pass


def test_default_queue_timeout(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", raising=False)
    assert EmbeddingRequestLimits(_SlowApp()).queue_timeout == 30


def test_readyz_reports_live_counters_and_preserves_fields(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_MAX_INFLIGHT", "1")
    monkeypatch.setenv("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "2")

    async def run():
        app = create_app()
        # ASGITransport로 실제 provider lifespan 없이 상태 엔드포인트를 검사한다.
        app.middleware_stack = app.build_middleware_stack()
        limiter = app.state.embedding_limits
        downstream = limiter.app
        slow = _SlowApp(delay=0.15)

        async def route(scope, receive, send):
            if scope.get("path") == "/v1/embeddings":
                return await slow(scope, receive, send)
            return await downstream(scope, receive, send)

        limiter.app = route
        first = asyncio.create_task(_Request().run(limiter))
        await _wait_for(lambda: limiter.inflight == 1)
        second = asyncio.create_task(_Request().run(limiter))
        await _wait_for(lambda: limiter.queue_depth == 1)
        # embedding 처리 앱만 지연시킨 뒤 실제 상태 라우터는 그대로 호출한다.
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/readyz")
            assert response.status_code == 503
            assert response.json() == {
                "status": "unavailable",
                "cost_tracking": {"enabled": None, "database_available": False, "healthy": False},
                "embeddings": {"inflight": 1, "queue_depth": 1, "capacity": 1},
            }
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-content-type-options"] == "nosniff"
            assert await asyncio.gather(first, second) == [200, 200]
            assert (await client.get("/readyz")).json()["embeddings"] == {
                "inflight": 0, "queue_depth": 0, "capacity": 1,
            }
        _idle(limiter)
        print("readyz: active={inflight:1, queue_depth:1, capacity:1}, idle={inflight:0, queue_depth:0, capacity:1}")

    asyncio.run(run())
