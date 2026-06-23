from __future__ import annotations

import os
import json
from typing import Any

import httpx

from openai_compatible_bridge.providers.vertex import VertexAPIError

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))


def _matching_suffix_prefix_len(text: str, token: str) -> int:
    max_len = min(len(text), len(token) - 1)
    for size in range(max_len, 0, -1):
        if text[-size:] == token[:size]:
            return size
    return 0


class _ThinkBlockStripper:
    _START = "<think>"
    _END = "</think>"

    def __init__(self) -> None:
        self._inside_think = False
        self._pending = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""

        data = self._pending + text
        self._pending = ""
        visible: list[str] = []
        cursor = 0

        while cursor < len(data):
            if self._inside_think:
                end = data.find(self._END, cursor)
                if end == -1:
                    pending_len = _matching_suffix_prefix_len(data[cursor:], self._END)
                    if pending_len:
                        self._pending = data[-pending_len:]
                    return "".join(visible)
                cursor = end + len(self._END)
                self._inside_think = False
                continue

            start = data.find(self._START, cursor)
            if start == -1:
                pending_len = _matching_suffix_prefix_len(data[cursor:], self._START)
                emit_end = len(data) - pending_len
                if emit_end > cursor:
                    visible.append(data[cursor:emit_end])
                if pending_len:
                    self._pending = data[-pending_len:]
                return "".join(visible)

            visible.append(data[cursor:start])
            cursor = start + len(self._START)
            self._inside_think = True

        return "".join(visible)

    def finish(self) -> str:
        if self._inside_think:
            self._pending = ""
            return ""
        pending = self._pending
        self._pending = ""
        return pending


def _strip_think_blocks(text: str) -> str:
    stripper = _ThinkBlockStripper()
    return stripper.feed(text) + stripper.finish()


def _ollama_format_from_response_format(response_format: dict[str, Any] | None) -> str | dict[str, Any] | None:
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise VertexAPIError(
            400,
            "response_format must be an object.",
            code="invalid_request",
        )

    fmt_type = response_format.get("type")
    if fmt_type == "text":
        return None
    if fmt_type == "json_object":
        return "json"
    if fmt_type == "json_schema":
        json_schema_obj = response_format.get("json_schema")
        raw_schema = json_schema_obj.get("schema") if isinstance(json_schema_obj, dict) else None
        if not isinstance(raw_schema, dict) or not raw_schema:
            raise VertexAPIError(
                400,
                "response_format.json_schema.schema must be a non-empty JSON schema object.",
                code="invalid_request",
            )
        return raw_schema

    raise VertexAPIError(
        400,
        f"Unsupported response_format.type: {fmt_type!r}.",
        code="invalid_request",
    )


class OllamaChatClient:
    def __init__(self, *, base_url: str | None = None) -> None:
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))

    async def close(self) -> None:
        await self.http.aclose()

    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        options: dict[str, Any] = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if stop is not None:
            options["stop"] = [stop] if isinstance(stop, str) else list(stop)
        if options:
            body["options"] = options
        ollama_format = _ollama_format_from_response_format(response_format)
        if ollama_format is not None:
            body["format"] = ollama_format

        try:
            resp = await self.http.post(f"{self.base_url}/api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Ollama request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Ollama connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            raise VertexAPIError(resp.status_code, resp.text or "Ollama request failed", code=str(resp.status_code))

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Ollama: {exc}", code="bad_gateway") from exc

        message = data.get("message", {}) if isinstance(data, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        return {
            "text": _strip_think_blocks(content),
            "finish_reason": data.get("done_reason") or "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
    ):
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        options: dict[str, Any] = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if stop is not None:
            options["stop"] = [stop] if isinstance(stop, str) else list(stop)
        if options:
            body["options"] = options
        ollama_format = _ollama_format_from_response_format(response_format)
        if ollama_format is not None:
            body["format"] = ollama_format

        stream_ctx = self.http.stream("POST", f"{self.base_url}/api/chat", json=body)
        try:
            try:
                resp = await stream_ctx.__aenter__()
            except httpx.TimeoutException as exc:
                raise VertexAPIError(504, f"Ollama request timed out: {exc}", code="timeout") from exc
            except httpx.RequestError as exc:
                raise VertexAPIError(502, f"Ollama connection error: {exc}", code="connection_error") from exc

            if resp.status_code >= 400:
                try:
                    await resp.aread()
                except Exception:
                    pass
                raise VertexAPIError(resp.status_code, "Ollama request failed", code=str(resp.status_code))

            think_stripper = _ThinkBlockStripper()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except (TypeError, ValueError):
                    continue

                message = data.get("message", {}) if isinstance(data, dict) else {}
                delta_text = message.get("content", "") if isinstance(message, dict) else ""
                finish_reason = data.get("done_reason") if data.get("done") else None
                usage = None
                if data.get("done"):
                    delta_text = think_stripper.feed(delta_text) + think_stripper.finish()
                    prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
                    completion_tokens = int(data.get("eval_count", 0) or 0)
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                else:
                    delta_text = think_stripper.feed(delta_text)
                yield {
                    "delta_text": delta_text,
                    "finish_reason": finish_reason,
                    "usage": usage,
                }
        finally:
            await stream_ctx.__aexit__(None, None, None)
