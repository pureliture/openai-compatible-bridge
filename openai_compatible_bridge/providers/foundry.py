"""Palantir Foundry OpenAI-compatible chat provider."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from openai_compatible_bridge.providers.vertex import (
    VertexAPIError,
    _coerce_openai_usage,
    _extract_message_text,
)

FOUNDRY_BASE_URL = os.getenv("FOUNDRY_BASE_URL", "").strip().rstrip("/")
FOUNDRY_TOKEN = os.getenv("FOUNDRY_TOKEN", "").strip()
HTTP_TIMEOUT_SECONDS = float(
    os.getenv("FOUNDRY_HTTP_TIMEOUT_SECONDS", os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
)


def _unwrap_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("Optional[") and text.endswith("]"):
        text = text[len("Optional[") : -1].strip()
    return text or None


def _openai_error_from_payload(payload: Any) -> tuple[str, str | None] | None:
    if not isinstance(payload, dict):
        return None

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        code = error.get("code") or error.get("type")
        if message:
            return str(message), str(code) if code is not None else None

    parameters = payload.get("parameters")
    if isinstance(parameters, dict):
        response_body = _unwrap_optional(parameters.get("responseBody"))
        if response_body:
            try:
                nested = json.loads(response_body)
            except (TypeError, ValueError):
                nested = None
            parsed = _openai_error_from_payload(nested)
            if parsed is not None:
                return parsed

        message = _unwrap_optional(parameters.get("errorMessage"))
        code = _unwrap_optional(parameters.get("errorCode"))
        if message:
            return message, code

    return None


def _parse_foundry_error(response: httpx.Response) -> VertexAPIError:
    try:
        payload = response.json()
    except Exception:
        payload = {"error": {"message": response.text}}

    parsed = _openai_error_from_payload(payload)
    if parsed is None:
        message = "Foundry request failed."
        code = str(response.status_code)
    else:
        message, code = parsed

    return VertexAPIError(
        response.status_code,
        message=message,
        code=code,
        raw=payload,
    )


class FoundryChatClient:
    """OpenAI Chat Completions client for a fixed Foundry endpoint.

    The endpoint is deployment configuration, not request input. This prevents a
    caller from redirecting the bridge's bearer token to an arbitrary URL.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else FOUNDRY_BASE_URL).strip().rstrip("/")
        self.token = token if token is not None else FOUNDRY_TOKEN
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))

    async def close(self) -> None:
        await self.http.aclose()

    def _headers(self) -> dict[str, str]:
        if not self.base_url:
            raise VertexAPIError(
                503,
                "FOUNDRY_BASE_URL is not configured.",
                code="provider_config_error",
            )
        if not self.token:
            raise VertexAPIError(
                503,
                "FOUNDRY_TOKEN is not configured.",
                code="provider_config_error",
            )
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _build_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        response_format: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        # Foundry's current OpenAI backend rejects max_tokens and requires this
        # newer field name. The bridge's public contract remains max_tokens.
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        # gpt-6-astra currently accepts only the default temperature. Omitting a
        # caller-supplied non-default is the compatible behavior for this backend;
        # forwarding it produces a deterministic upstream 400.
        if temperature is not None and temperature == 1.0:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if response_format is not None:
            body["response_format"] = response_format
        return body

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
        **_: Any,
    ) -> dict[str, Any]:
        body = self._build_request_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            stream=False,
        )

        try:
            response = await self.http.post(self.base_url, headers=self._headers(), json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Foundry request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Foundry connection error: {exc}", code="connection_error") from exc

        if response.status_code >= 400:
            raise _parse_foundry_error(response)

        try:
            payload = response.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Foundry: {exc}", code="bad_gateway") from exc

        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise VertexAPIError(502, "Malformed Foundry response: missing choices[]", code="bad_gateway")

        choice = choices[0] or {}
        message = choice.get("message", {}) or {}
        content = _extract_message_text(message.get("content", ""))
        return {
            "text": content,
            "finish_reason": choice.get("finish_reason") or "stop",
            "usage": _coerce_openai_usage(payload.get("usage") if isinstance(payload, dict) else None),
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
        **_: Any,
    ):
        body = self._build_request_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            stream=True,
        )
        stream_context = self.http.stream(
            "POST",
            self.base_url,
            headers=self._headers(),
            json=body,
        )
        entered = False
        try:
            try:
                response = await stream_context.__aenter__()
                entered = True
            except httpx.TimeoutException as exc:
                raise VertexAPIError(504, f"Foundry request timed out: {exc}", code="timeout") from exc
            except httpx.RequestError as exc:
                raise VertexAPIError(502, f"Foundry connection error: {exc}", code="connection_error") from exc

            if response.status_code >= 400:
                try:
                    await response.aread()
                except Exception:
                    pass
                raise _parse_foundry_error(response)

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:") :].strip()
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue

                stream_error = _openai_error_from_payload(event)
                if stream_error is not None and "choices" not in event:
                    message, code = stream_error
                    raise VertexAPIError(502, message, code=code, raw=event)

                choices = event.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else {}
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                delta = delta if isinstance(delta, dict) else {}
                delta_text = _extract_message_text(delta.get("content", ""))
                finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                usage = event.get("usage")
                normalized_usage = _coerce_openai_usage(usage) if isinstance(usage, dict) else None

                if delta_text or finish_reason is not None or normalized_usage is not None:
                    yield {
                        "delta_text": delta_text,
                        "finish_reason": finish_reason,
                        "usage": normalized_usage,
                    }
        finally:
            if entered:
                await stream_context.__aexit__(None, None, None)
