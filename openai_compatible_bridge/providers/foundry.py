"""Palantir Foundry OpenAI-compatible chat provider."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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

FOUNDRY_OPENAI_PROTOCOL = "openai_chat_completions"
FOUNDRY_ANTHROPIC_PROTOCOL = "anthropic_messages"
FOUNDRY_XAI_RESPONSES_PROTOCOL = "xai_responses"
FOUNDRY_OPENAI_PATH = "/api/v2/llm/proxy/openai/v1/chat/completions"
FOUNDRY_ANTHROPIC_PATH = "/api/v2/llm/proxy/anthropic/v1/messages"
FOUNDRY_XAI_RESPONSES_PATH = "/api/v2/llm/proxy/xai/v1/responses"


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


def _usage(prompt_tokens: Any, completion_tokens: Any, total_tokens: Any = None) -> dict[str, int]:
    def as_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    prompt = as_int(prompt_tokens)
    completion = as_int(completion_tokens)
    total = as_int(total_tokens) or prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _anthropic_usage(payload: Any) -> dict[str, int]:
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return _usage(usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens"))


def _xai_usage(payload: Any) -> dict[str, int]:
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return _usage(usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens"))


def _map_finish_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    value = str(reason)
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "completed": "stop",
        "max_tokens": "length",
        "max_output_tokens": "length",
        "length": "length",
    }.get(value, value)


def _xai_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "".join(parts)


def _openai_input_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the public OpenAI message shape for xAI Responses input."""
    return [
        {"role": message.get("role", "user"), "content": message.get("content", "")}
        for message in messages
    ]


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

    def _protocol(self, resolved_config: dict[str, Any] | None) -> str:
        protocol = (resolved_config or {}).get("protocol", FOUNDRY_OPENAI_PROTOCOL)
        if protocol not in {
            FOUNDRY_OPENAI_PROTOCOL,
            FOUNDRY_ANTHROPIC_PROTOCOL,
            FOUNDRY_XAI_RESPONSES_PROTOCOL,
        }:
            raise VertexAPIError(
                503,
                f"Unsupported Foundry protocol: {protocol}.",
                code="provider_config_error",
            )
        return protocol

    def _url_for_protocol(self, protocol: str) -> str:
        if not self.base_url:
            raise VertexAPIError(
                503,
                "FOUNDRY_BASE_URL is not configured.",
                code="provider_config_error",
            )
        if protocol == FOUNDRY_OPENAI_PROTOCOL:
            return self.base_url

        parsed = urlsplit(self.base_url)
        if parsed.query or parsed.fragment or not parsed.path.endswith(FOUNDRY_OPENAI_PATH):
            raise VertexAPIError(
                503,
                "FOUNDRY_BASE_URL must be the documented OpenAI chat proxy URL "
                "when using a non-OpenAI Foundry protocol.",
                code="provider_config_error",
            )
        root = parsed.path[: -len(FOUNDRY_OPENAI_PATH)]
        path = {
            FOUNDRY_ANTHROPIC_PROTOCOL: FOUNDRY_ANTHROPIC_PATH,
            FOUNDRY_XAI_RESPONSES_PROTOCOL: FOUNDRY_XAI_RESPONSES_PATH,
        }[protocol]
        return urlunsplit((parsed.scheme, parsed.netloc, root + path, "", ""))

    def _headers(self, protocol: str) -> dict[str, str]:
        if not self.token:
            raise VertexAPIError(
                503,
                "FOUNDRY_TOKEN is not configured.",
                code="provider_config_error",
            )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        if protocol == FOUNDRY_ANTHROPIC_PROTOCOL:
            headers["anthropic-version"] = "2023-06-01"
        return headers

    @staticmethod
    def _build_openai_request_body(
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

    @staticmethod
    def _build_anthropic_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        stop: str | list[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if role == "system":
                system_parts.append(_extract_message_text(content))
            elif role in {"user", "assistant"}:
                chat_messages.append({"role": role, "content": content})
            else:
                # Anthropic Messages does not accept arbitrary OpenAI roles.
                chat_messages.append({"role": "user", "content": _extract_message_text(content)})

        body: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
            "stream": stream,
        }
        if system_parts:
            body["system"] = "\\n\\n".join(part for part in system_parts if part)
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        # Sonnet 5 and Opus 5 reject non-default sampling fields; intentionally
        # do not forward temperature/top_p from the public OpenAI contract.
        return body

    @staticmethod
    def _build_xai_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "input": _openai_input_messages(messages),
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if response_format is not None:
            format_type = response_format.get("type")
            if format_type == "json_object":
                body["text"] = {"format": {"type": "json_object"}}
            elif format_type == "json_schema":
                body["text"] = {"format": response_format}
        return body

    def _build_request_body(
        self,
        *,
        protocol: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        if protocol == FOUNDRY_OPENAI_PROTOCOL:
            return self._build_openai_request_body(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                response_format=response_format,
                stream=stream,
            )
        if protocol == FOUNDRY_ANTHROPIC_PROTOCOL:
            return self._build_anthropic_request_body(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stop=stop,
                stream=stream,
            )
        return self._build_xai_request_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            stream=stream,
        )

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
        reasoning_effort: str | None = None,
        resolved_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        protocol = self._protocol(resolved_config)
        body = self._build_request_body(
            protocol=protocol,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            stream=False,
        )

        try:
            response = await self.http.post(
                self._url_for_protocol(protocol),
                headers=self._headers(protocol),
                json=body,
            )
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

        if protocol == FOUNDRY_OPENAI_PROTOCOL:
            choices = payload.get("choices") if isinstance(payload, dict) else None
            if not isinstance(choices, list) or not choices:
                raise VertexAPIError(502, "Malformed Foundry response: missing choices[]", code="bad_gateway")
            choice = choices[0] or {}
            message = choice.get("message", {}) or {}
            return {
                "text": _extract_message_text(message.get("content", "")),
                "finish_reason": choice.get("finish_reason") or "stop",
                "usage": _coerce_openai_usage(payload.get("usage") if isinstance(payload, dict) else None),
            }

        if protocol == FOUNDRY_ANTHROPIC_PROTOCOL:
            content = payload.get("content") if isinstance(payload, dict) else None
            if not isinstance(content, list):
                raise VertexAPIError(502, "Malformed Anthropic response: missing content[]", code="bad_gateway")
            text = "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            return {
                "text": text,
                "finish_reason": _map_finish_reason(payload.get("stop_reason")) or "stop",
                "usage": _anthropic_usage(payload),
            }

        text = _xai_text(payload)
        if not text and not isinstance(payload.get("output"), list):
            raise VertexAPIError(502, "Malformed xAI response: missing output[]", code="bad_gateway")
        status = payload.get("status") if isinstance(payload, dict) else None
        incomplete = payload.get("incomplete_details") if isinstance(payload, dict) else None
        finish_reason = "stop" if status == "completed" else None
        if isinstance(incomplete, dict):
            finish_reason = _map_finish_reason(incomplete.get("reason")) or finish_reason
        return {
            "text": text,
            "finish_reason": finish_reason or "stop",
            "usage": _xai_usage(payload),
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
        reasoning_effort: str | None = None,
        resolved_config: dict[str, Any] | None = None,
        **_: Any,
    ):
        protocol = self._protocol(resolved_config)
        body = self._build_request_body(
            protocol=protocol,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            stream=True,
        )
        stream_context = self.http.stream(
            "POST",
            self._url_for_protocol(protocol),
            headers=self._headers(protocol),
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

            stream_usage: dict[str, int] | None = None
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

                delta_text = ""
                finish_reason: str | None = None
                normalized_usage: dict[str, int] | None = None

                if protocol == FOUNDRY_OPENAI_PROTOCOL:
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

                elif protocol == FOUNDRY_ANTHROPIC_PROTOCOL:
                    event_type = event.get("type")
                    if event_type == "message_start":
                        message = event.get("message", {}) or {}
                        usage = message.get("usage", {}) or {}
                        stream_usage = _usage(
                            usage.get("input_tokens"),
                            usage.get("output_tokens"),
                            usage.get("total_tokens"),
                        )
                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {}) or {}
                        if delta.get("type") == "text_delta":
                            delta_text = str(delta.get("text", ""))
                    elif event_type == "message_delta":
                        delta = event.get("delta", {}) or {}
                        usage = event.get("usage", {}) or {}
                        if stream_usage is None:
                            stream_usage = _usage(0, usage.get("output_tokens"), usage.get("total_tokens"))
                        else:
                            stream_usage = _usage(
                                stream_usage.get("prompt_tokens"),
                                usage.get("output_tokens", stream_usage.get("completion_tokens")),
                                usage.get("total_tokens"),
                            )
                        finish_reason = _map_finish_reason(delta.get("stop_reason"))
                        normalized_usage = stream_usage

                else:
                    event_type = event.get("type")
                    if event_type == "response.output_text.delta":
                        delta_text = str(event.get("delta", ""))
                    elif event_type == "response.completed":
                        completed = event.get("response", {}) or {}
                        normalized_usage = _xai_usage(completed)
                        status = completed.get("status")
                        finish_reason = "stop" if status == "completed" else None
                        incomplete = completed.get("incomplete_details")
                        if isinstance(incomplete, dict):
                            finish_reason = _map_finish_reason(incomplete.get("reason")) or finish_reason
                    elif event_type in {"response.failed", "error"}:
                        error = event.get("error", {}) or {}
                        message = error.get("message") or "xAI Responses stream failed"
                        raise VertexAPIError(502, str(message), code=error.get("code"), raw=event)

                if delta_text or finish_reason is not None or normalized_usage is not None:
                    yield {
                        "delta_text": delta_text,
                        "finish_reason": finish_reason,
                        "usage": normalized_usage,
                    }
        finally:
            if entered:
                await stream_context.__aexit__(None, None, None)
