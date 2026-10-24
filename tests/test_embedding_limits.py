from __future__ import annotations

import asyncio

from openai_compatible_bridge.embedding_limits import EmbeddingRequestLimits


def _scope(body: bytes):
    return {
        "type": "http", "method": "POST", "path": "/v1/embeddings",
        "headers": [(b"content-length", str(len(body)).encode())],
    }


def test_default_embedding_concurrency_is_32(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_MAX_INFLIGHT", raising=False)
    middleware = EmbeddingRequestLimits(lambda *_args: None)
    assert middleware.max_inflight == 32


def test_zero_queue_timeout_restores_immediate_rejection(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_MAX_INFLIGHT", raising=False)
    monkeypatch.setenv("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "0")
    body = b'{"input":"test"}'
    started_count = 0

    async def run():
        nonlocal started_count
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def downstream(scope, receive, send):
            nonlocal started_count
            await receive()
            started_count += 1
            if started_count == 32:
                all_started.set()
            await release.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = EmbeddingRequestLimits(downstream, max_bytes=1024)

        async def request():
            messages = []
            received = False

            async def receive():
                nonlocal received
                if received:
                    await asyncio.Future()
                received = True
                return {"type": "http.request", "body": body, "more_body": False}

            async def send(message):
                messages.append(message)

            await middleware(_scope(body), receive, send)
            response = next(message for message in messages if message["type"] == "http.response.start")
            return response

        active = [asyncio.create_task(request()) for _ in range(32)]
        try:
            await asyncio.wait_for(all_started.wait(), 1)
            rejected = await request()
            assert rejected["status"] == 429
            assert (b"retry-after", b"2") in rejected["headers"]
            release.set()
            responses = await asyncio.gather(*active)
            assert [response["status"] for response in responses] == [200] * 32
            assert middleware.inflight == middleware.queue_depth == 0
            print("rollback_timeout_zero: 32x200 + 1x429, inflight=0, queue_depth=0")
        finally:
            release.set()
            await asyncio.gather(*active, return_exceptions=True)

    asyncio.run(run())
