"""임베딩 본문 크기를 제한하고 포화된 요청을 FIFO 순서로 슬롯이 빌 때까지 대기시킨다."""

from __future__ import annotations

import asyncio
import collections
import json
import math
import os

from .core.embedding_resilience import embedding_max_inflight


class EmbeddingRequestLimits:
    def __init__(self, app, *, max_bytes: int | None = None,
                 max_inflight: int | None = None, state=None) -> None:
        self.app = app
        self.max_bytes = int(os.environ.get("EMBEDDINGS_MAX_REQUEST_BYTES", "1048576")) if max_bytes is None else max_bytes
        self.max_inflight = embedding_max_inflight() if max_inflight is None else max_inflight
        if self.max_bytes < 1 or self.max_inflight < 1:
            raise ValueError("embedding request limits must be positive")
        self.queue_timeout = float(os.environ.get("EMBEDDINGS_QUEUE_TIMEOUT_SECONDS", "30"))
        if not math.isfinite(self.queue_timeout) or self.queue_timeout < 0:
            raise ValueError("embedding queue timeout must be finite and non-negative")
        self.inflight = 0
        self.queue_depth = 0
        # 대기자는 도착 순서(FIFO)대로 슬롯을 직접 인계받는다.
        self._waiters: collections.deque[asyncio.Future[None]] = collections.deque()
        if state is not None:
            state.embedding_limits = self

    def metrics(self) -> dict[str, int]:
        return {"inflight": self.inflight, "queue_depth": self.queue_depth,
                "capacity": self.max_inflight}

    async def _acquire_slot(self) -> bool:
        # 대기자가 있으면 새 요청은 빈 슬롯이 있어도 끼어들지 않고 줄 뒤에 선다.
        # 슬롯 확인과 증가 사이에는 await가 없어 단일 이벤트 루프에서 경쟁이 없다.
        if self.inflight < self.max_inflight and not self._waiters:
            self.inflight += 1
            return True
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        self.queue_depth += 1
        try:
            async with asyncio.timeout(self.queue_timeout):
                await waiter
        except BaseException as exc:
            if waiter.done() and not waiter.cancelled():
                # 타임아웃/취소와 인계가 동시에 일어난 경우: 인계받은 슬롯을 반환한다.
                self._release_slot()
            else:
                # 인계받지 못했다면 줄에서 빠지고 대기 카운터를 정리한다.
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
                self.queue_depth -= 1
            if isinstance(exc, TimeoutError):
                return False
            raise
        # 인계 시점에 _release_slot이 이미 queue_depth를 줄였고 inflight는 유지된다.
        return True

    def _release_slot(self) -> None:
        # 슬롯을 비우지 않고 가장 오래 기다린 살아 있는 대기자에게 직접 넘긴다.
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self.queue_depth -= 1
                waiter.set_result(None)
                return
        self.inflight -= 1

    async def _reject(self, send, status: int, code: str, message: str):
        body = json.dumps({"error": {"message": message, "type": "invalid_request_error",
                                    "param": "input", "code": code}}).encode()
        headers = [(b"content-type", b"application/json"),
                   (b"content-length", str(len(body)).encode()), (b"connection", b"close")]
        if status == 429:
            headers.append((b"retry-after", b"2"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path", "").rstrip("/") != "/v1/embeddings"):
            return await self.app(scope, receive, send)
        lengths = [value for name, value in scope.get("headers", []) if name.lower() == b"content-length"]
        try:
            if len(lengths) > 1 or (lengths and not lengths[0].isdigit()):
                raise ValueError()
            declared = int(lengths[0]) if lengths else None
        except ValueError:
            return await self._reject(send, 400, "invalid_content_length", "Invalid Content-Length.")
        if declared is not None and declared > self.max_bytes:
            return await self._reject(send, 413, "request_too_large", "Embedding request exceeds the configured byte limit.")
        body_ready = asyncio.get_running_loop().create_future()
        disconnected = asyncio.Event()
        request_task = asyncio.create_task(self._serve(scope, body_ready, disconnected, send))
        receive_task = asyncio.create_task(self._receive_body(receive, declared, body_ready, disconnected))
        try:
            done, _ = await asyncio.wait(
                (request_task, receive_task), return_when=asyncio.FIRST_COMPLETED,
            )
            if request_task in done:
                return await request_task
            error = receive_task.result()
            # 중단 또는 잘못된 본문은 큐/슬롯을 먼저 반환한 뒤 종료한다.
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
            if error is not None:
                return await self._reject(send, *error)
        finally:
            for task in (request_task, receive_task):
                task.cancel()
            await asyncio.gather(request_task, receive_task, return_exceptions=True)

    async def _receive_body(self, receive, declared, body_ready, disconnected):
        # receive 소유자를 하나로 유지한다. 큐에서도 본문을 제한된 크기로만
        # 버퍼링하며, 본문 수신 후에도 클라이언트 중단을 계속 감지한다.
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                disconnected.set()
                return
            if event["type"] != "http.request":
                continue
            data = event.get("body", b"")
            if len(body) + len(data) > self.max_bytes:
                return (413, "request_too_large", "Embedding request exceeds the configured byte limit.")
            body.extend(data)
            if not event.get("more_body", False):
                break
        if declared is not None and declared != len(body):
            return (400, "invalid_content_length", "Content-Length does not match the request body.")
        body_ready.set_result(bytes(body))
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                disconnected.set()
                return

    async def _serve(self, scope, body_ready, disconnected, send):
        if not await self._acquire_slot():
            return await self._reject(send, 429, "embedding_concurrency_limit", "Embedding capacity is busy; retry later.")
        try:
            body = await body_ready
            replayed = False

            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await disconnected.wait()
                return {"type": "http.disconnect"}

            return await self.app(scope, replay, send)
        finally:
            self._release_slot()
