from __future__ import annotations

import os
import json
from typing import Any

import httpx

from openai_compatible_bridge.providers.vertex import VertexAPIError

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
HTTP_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_HTTP_TIMEOUT_SECONDS", os.getenv("HTTP_TIMEOUT_SECONDS", "60")))


def _ollama_think_from_env(value: str | None) -> bool | str | None:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"", "omit", "none"}:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"low", "medium", "high"}:
        return normalized
    raise ValueError("OLLAMA_THINK must be true, false, low, medium, high, omit, or none")


OLLAMA_THINK = _ollama_think_from_env(os.getenv("OLLAMA_THINK", "true"))


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


def _content_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _raise_if_empty_after_reasoning_normalization(
    *,
    raw_content_chars: int,
    normalized_content_chars: int,
    thinking_chars: int,
    completion_tokens: int,
) -> None:
    if normalized_content_chars > 0:
        return
    if raw_content_chars == 0 and thinking_chars == 0 and completion_tokens == 0:
        return
    raise VertexAPIError(
        502,
        "Ollama response contained no visible content after reasoning normalization "
        f"(raw_content_chars={raw_content_chars}, "
        f"normalized_content_chars={normalized_content_chars}, "
        f"thinking_chars={thinking_chars}, "
        f"completion_tokens={completion_tokens}).",
        code="empty_content_after_think_strip",
    )


def _normalize_ollama_message_content(
    message: dict[str, Any],
    *,
    completion_tokens: int = 0,
) -> str:
    content = _content_text(message.get("content"))
    thinking = _content_text(message.get("thinking"))
    normalized = _strip_think_blocks(content)
    _raise_if_empty_after_reasoning_normalization(
        raw_content_chars=len(content),
        normalized_content_chars=len(normalized),
        thinking_chars=len(thinking),
        completion_tokens=completion_tokens,
    )
    return normalized


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
        if OLLAMA_THINK is not None:
            body["think"] = OLLAMA_THINK
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
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        content = _normalize_ollama_message_content(
            message if isinstance(message, dict) else {},
            completion_tokens=completion_tokens,
        )
        return {
            "text": content,
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
        if OLLAMA_THINK is not None:
            body["think"] = OLLAMA_THINK
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
            raw_content_chars = 0
            normalized_content_chars = 0
            thinking_chars = 0
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except (TypeError, ValueError):
                    continue

                message = data.get("message", {}) if isinstance(data, dict) else {}
                delta_text = message.get("content", "") if isinstance(message, dict) else ""
                delta_text = delta_text if isinstance(delta_text, str) else ""
                delta_thinking = message.get("thinking", "") if isinstance(message, dict) else ""
                delta_thinking = delta_thinking if isinstance(delta_thinking, str) else ""
                raw_content_chars += len(delta_text)
                thinking_chars += len(delta_thinking)
                finish_reason = data.get("done_reason") if data.get("done") else None
                usage = None
                if data.get("done"):
                    delta_text = think_stripper.feed(delta_text) + think_stripper.finish()
                    prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
                    completion_tokens = int(data.get("eval_count", 0) or 0)
                    normalized_content_chars += len(delta_text)
                    _raise_if_empty_after_reasoning_normalization(
                        raw_content_chars=raw_content_chars,
                        normalized_content_chars=normalized_content_chars,
                        thinking_chars=thinking_chars,
                        completion_tokens=completion_tokens,
                    )
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                else:
                    delta_text = think_stripper.feed(delta_text)
                    normalized_content_chars += len(delta_text)
                yield {
                    "delta_text": delta_text,
                    "finish_reason": finish_reason,
                    "usage": usage,
                }
        finally:
            await stream_ctx.__aexit__(None, None, None)
