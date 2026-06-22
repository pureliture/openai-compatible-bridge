from __future__ import annotations

import os
import json
from typing import Any

import httpx

from openai_compatible_bridge.providers.vertex import VertexAPIError

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))


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
                    prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
                    completion_tokens = int(data.get("eval_count", 0) or 0)
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                yield {
                    "delta_text": delta_text,
                    "finish_reason": finish_reason,
                    "usage": usage,
                }
        finally:
            await stream_ctx.__aexit__(None, None, None)
