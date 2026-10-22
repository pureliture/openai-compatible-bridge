"""Reject oversized embedding requests before JSON decoding or upstream work."""

from __future__ import annotations

import json
import os


class EmbeddingRequestLimits:
    def __init__(self, app, *, max_bytes: int | None = None,
                 max_inflight: int | None = None) -> None:
        self.app = app
        self.max_bytes = int(os.environ.get("EMBEDDINGS_MAX_REQUEST_BYTES", "1048576")) if max_bytes is None else max_bytes
        self.max_inflight = int(os.environ.get("EMBEDDINGS_MAX_INFLIGHT", "4")) if max_inflight is None else max_inflight
        if self.max_bytes < 1 or self.max_inflight < 1:
            raise ValueError("embedding request limits must be positive")
        self.inflight = 0

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
        # Reserve capacity before reading. These operations contain no await, so
        # tasks on this ASGI event loop cannot race the check/increment.
        if self.inflight >= self.max_inflight:
            return await self._reject(send, 429, "embedding_concurrency_limit", "Embedding capacity is busy; retry later.")
        self.inflight += 1
        try:
            body = bytearray()
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                if event["type"] != "http.request":
                    continue
                data = event.get("body", b"")
                if len(body) + len(data) > self.max_bytes:
                    return await self._reject(send, 413, "request_too_large", "Embedding request exceeds the configured byte limit.")
                body.extend(data)
                if not event.get("more_body", False):
                    break
            if declared is not None and declared != len(body):
                return await self._reject(send, 400, "invalid_content_length", "Content-Length does not match the request body.")
            replayed = False

            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            return await self.app(scope, replay, send)
        finally:
            self.inflight -= 1
