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
FOUNDRY_OPENAI_RESPONSES_PROTOCOL = "openai_responses"
FOUNDRY_OPENAI_PATH = "/api/v2/llm/proxy/openai/v1/chat/completions"
FOUNDRY_ANTHROPIC_PATH = "/api/v2/llm/proxy/anthropic/v1/messages"
FOUNDRY_XAI_RESPONSES_PATH = "/api/v2/llm/proxy/xai/v1/responses"
FOUNDRY_OPENAI_RESPONSES_PATH = "/api/v2/llm/proxy/openai/v1/responses"


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


_XAI_SUPPORTED_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def _clamp_xai_reasoning_effort(effort: Any) -> str | None:
    """Map Hermes-style effort values onto Foundry xAI's accepted set.

    Live probing on foundry:grok-4.6 shows minimal|low|medium|high|xhigh are
    accepted while 'none' and 'max'/'ultra' are rejected with HTTP 400. This
    clamps unsupported values instead of forwarding a deterministic 400.
    """
    if effort is None:
        return None
    value = str(effort).strip().lower()
    if not value:
        return None
    if value in _XAI_SUPPORTED_EFFORTS:
        return value
    if value in {"max", "ultra"}:
        return "xhigh"
    # 'none' (and unknown values) cannot be represented on this route: omitting
    # the field lets the server default apply instead of a rejected value.
    return None


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


def _xai_input_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transform message history into xAI Responses input items sequence."""
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        if role == "tool":
            content_val = content
            if isinstance(content_val, (dict, list)):
                out_str = json.dumps(content_val, ensure_ascii=False)
            else:
                out_str = str(content_val if content_val is not None else "")
            call_id = message.get("tool_call_id") or message.get("id") or ""
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": out_str,
                }
            )
        elif role == "assistant":
            if content:
                text = _extract_message_text(content)
                if text:
                    input_items.append({"role": "assistant", "content": text})
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    raw_args = fn.get("arguments")
                    if not raw_args:
                        args_str = "{}"
                    elif isinstance(raw_args, str):
                        args_str = raw_args
                    else:
                        args_str = json.dumps(raw_args, ensure_ascii=False)
                    call_id = tc.get("id") or tc.get("call_id") or ""
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": fn.get("name", ""),
                            "arguments": args_str,
                        }
                    )
            elif not content:
                input_items.append({"role": "assistant", "content": ""})
        elif role == "system":
            text = _extract_message_text(content) if content is not None else ""
            input_items.append({"role": "system", "content": text})
        else:
            text = _extract_message_text(content) if content is not None else ""
            input_items.append({"role": role, "content": text})
    return input_items


_openai_input_messages = _xai_input_messages



def _append_anthropic_turn(chat_messages: list[dict[str, Any]], role: str, content: Any) -> None:
    if not chat_messages:
        chat_messages.append({"role": role, "content": content})
        return

    if chat_messages[-1]["role"] != role:
        chat_messages.append({"role": role, "content": content})
        return

    prev_content = chat_messages[-1]["content"]
    if isinstance(prev_content, list) and isinstance(content, list):
        prev_content.extend(content)
    elif isinstance(prev_content, list) and not isinstance(content, list):
        text = _extract_message_text(content)
        if text:
            prev_content.append({"type": "text", "text": text})
    elif not isinstance(prev_content, list) and isinstance(content, list):
        prev_text = _extract_message_text(prev_content)
        new_blocks: list[dict[str, Any]] = []
        if prev_text:
            new_blocks.append({"type": "text", "text": prev_text})
        new_blocks.extend(content)
        chat_messages[-1]["content"] = new_blocks
    else:
        prev_text = _extract_message_text(prev_content)
        curr_text = _extract_message_text(content)
        if prev_text and curr_text:
            chat_messages[-1]["content"] = f"{prev_text}\n\n{curr_text}"
        elif curr_text:
            chat_messages[-1]["content"] = curr_text
        else:
            chat_messages[-1]["content"] = prev_text


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
            FOUNDRY_OPENAI_RESPONSES_PROTOCOL,
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
            FOUNDRY_OPENAI_RESPONSES_PROTOCOL: FOUNDRY_OPENAI_RESPONSES_PATH,
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
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
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            body["parallel_tool_calls"] = parallel_tool_calls
        return body

    @staticmethod
    def _build_anthropic_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        stop: str | list[str] | None,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content")
            if role == "system":
                if content:
                    system_parts.append(_extract_message_text(content))
            elif role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    blocks: list[dict[str, Any]] = []
                    if content:
                        text = _extract_message_text(content)
                        if text:
                            blocks.append({"type": "text", "text": text})
                    for tc in tool_calls:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        raw_args = fn.get("arguments", "{}")
                        if isinstance(raw_args, str):
                            try:
                                input_dict = json.loads(raw_args) if raw_args.strip() else {}
                            except Exception:
                                input_dict = {}
                        elif isinstance(raw_args, dict):
                            input_dict = raw_args
                        else:
                            input_dict = {}
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": fn.get("name", ""),
                                "input": input_dict,
                            }
                        )
                    _append_anthropic_turn(chat_messages, "assistant", blocks)
                else:
                    _append_anthropic_turn(chat_messages, "assistant", content if content is not None else "")
            elif role == "tool":
                content_val = message.get("content", "")
                if isinstance(content_val, (dict, list)):
                    result_content = json.dumps(content_val, ensure_ascii=False)
                else:
                    result_content = str(content_val if content_val is not None else "")
                tool_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id") or message.get("id") or "",
                    "content": result_content,
                }
                if message.get("is_error"):
                    tool_block["is_error"] = True
                _append_anthropic_turn(chat_messages, "user", [tool_block])
            else:
                _append_anthropic_turn(chat_messages, "user", content if content is not None else "")

        if not chat_messages:
            chat_messages.append({"role": "user", "content": ""})

        body: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
            "stream": stream,
        }
        if system_parts:
            body["system"] = "\n\n".join(part for part in system_parts if part)
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        if tools and tool_choice != "none":
            anthropic_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function", {}) if t.get("type") == "function" else t
                tool_def: dict[str, Any] = {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
                anthropic_tools.append(tool_def)
            body["tools"] = anthropic_tools

            if tool_choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict):
                name = (
                    tool_choice.get("function", {}).get("name")
                    if isinstance(tool_choice.get("function"), dict)
                    else tool_choice.get("name")
                )
                if name:
                    body["tool_choice"] = {"type": "tool", "name": name}

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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "input": _xai_input_messages(messages),
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
        clamped_effort = _clamp_xai_reasoning_effort(reasoning_effort)
        if clamped_effort is not None:
            body["reasoning"] = {"effort": clamped_effort}
        if response_format is not None:
            format_type = response_format.get("type")
            if format_type == "json_object":
                body["text"] = {"format": {"type": "json_object"}}
            elif format_type == "json_schema":
                body["text"] = {"format": response_format}

        if tools is not None:
            xai_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function", {}) if t.get("type") == "function" else t
                xai_tools.append(
                    {
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
            body["tools"] = xai_tools

        if tool_choice is not None:
            if isinstance(tool_choice, dict):
                fn_dict = tool_choice.get("function")
                name = fn_dict.get("name") if isinstance(fn_dict, dict) else tool_choice.get("name")
                if name:
                    body["tool_choice"] = {"type": "function", "name": name}
                else:
                    body["tool_choice"] = tool_choice
            elif isinstance(tool_choice, str):
                body["tool_choice"] = tool_choice

        return body

    @staticmethod
    def _build_openai_responses_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """OpenAI Responses proxy body for the gpt-6-astra family.

        These models reject function tools on /v1/chat/completions (Foundry
        returns HTTP 400 for tools with reasoning), while /v1/responses accepts
        tools with default reasoning. The verified working shape is
        model/input/tools/tool_choice/max_output_tokens; sampling and reasoning
        fields are intentionally omitted on this route.
        """
        body: dict[str, Any] = {
            "model": model,
            "input": _xai_input_messages(messages),
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens

        if tools is not None:
            responses_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function", {}) if t.get("type") == "function" else t
                responses_tools.append(
                    {
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
            body["tools"] = responses_tools

        if tool_choice is not None:
            if isinstance(tool_choice, dict):
                fn_dict = tool_choice.get("function")
                name = fn_dict.get("name") if isinstance(fn_dict, dict) else tool_choice.get("name")
                if name:
                    body["tool_choice"] = {"type": "function", "name": name}
                else:
                    body["tool_choice"] = tool_choice
            elif isinstance(tool_choice, str):
                body["tool_choice"] = tool_choice

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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
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
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                stream=stream,
            )
        if protocol == FOUNDRY_ANTHROPIC_PROTOCOL:
            return self._build_anthropic_request_body(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stop=stop,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
            )
        if protocol == FOUNDRY_OPENAI_RESPONSES_PROTOCOL:
            return self._build_openai_responses_request_body(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
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
            tools=tools,
            tool_choice=tool_choice,
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
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
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
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
            tool_calls = message.get("tool_calls")
            content = message.get("content")
            if tool_calls:
                extracted = _extract_message_text(content) if content is not None else ""
                text = extracted if extracted else None
                finish_reason = choice.get("finish_reason") or "tool_calls"
            else:
                text = _extract_message_text(content or "")
                finish_reason = choice.get("finish_reason") or "stop"
            return {
                "text": text,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": finish_reason,
                "usage": _coerce_openai_usage(payload.get("usage") if isinstance(payload, dict) else None),
            }

        if protocol == FOUNDRY_ANTHROPIC_PROTOCOL:
            content = payload.get("content") if isinstance(payload, dict) else None
            if not isinstance(content, list):
                raise VertexAPIError(502, "Malformed Anthropic response: missing content[]", code="bad_gateway")
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "tool_use":
                    raw_input = block.get("input")
                    if isinstance(raw_input, (dict, list)):
                        args_str = json.dumps(raw_input, ensure_ascii=False)
                    elif isinstance(raw_input, str):
                        args_str = raw_input
                    else:
                        args_str = "{}"
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": args_str,
                            },
                        }
                    )
            stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
            if stop_reason == "tool_use" or (tool_calls and stop_reason is None):
                finish_reason = "tool_calls"
            else:
                finish_reason = _map_finish_reason(stop_reason) or "stop"

            extracted_text = "".join(text_parts)
            if tool_calls:
                text = extracted_text if extracted_text else None
            else:
                text = extracted_text

            return {
                "text": text,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": finish_reason,
                "usage": _anthropic_usage(payload),
            }

        output_list = payload.get("output") if isinstance(payload, dict) else None
        if not isinstance(output_list, list):
            raise VertexAPIError(502, "Malformed xAI response: missing output[]", code="bad_gateway")

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in output_list:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "message":
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        text_parts.append(str(c.get("text", "")))
            elif itype == "function_call":
                raw_args = item.get("arguments")
                if not raw_args:
                    args_str = "{}"
                elif isinstance(raw_args, str):
                    args_str = raw_args
                elif isinstance(raw_args, (dict, list)):
                    args_str = json.dumps(raw_args, ensure_ascii=False)
                else:
                    args_str = "{}"
                call_id = item.get("call_id") or item.get("id") or ""
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": args_str,
                        },
                    }
                )

        extracted_text = "".join(text_parts)
        if tool_calls:
            text = extracted_text if extracted_text else None
            finish_reason = "tool_calls"
        else:
            text = extracted_text
            status = payload.get("status") if isinstance(payload, dict) else None
            incomplete = payload.get("incomplete_details") if isinstance(payload, dict) else None
            finish_reason = "stop" if status == "completed" else None
            if isinstance(incomplete, dict):
                finish_reason = _map_finish_reason(incomplete.get("reason")) or finish_reason
            finish_reason = finish_reason or "stop"

        return {
            "text": text,
            "tool_calls": tool_calls if tool_calls else None,
            "finish_reason": finish_reason,
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
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
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
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

            anthropic_tool_call_map: dict[int, int] = {}
            xai_tool_call_map: dict[str, int] = {}
            stream_usage: dict[str, int] | None = None
            stream_finish_reason: str | None = None
            saw_tool_calls: bool = False
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
                delta_tool_calls: list[dict[str, Any]] | None = None
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
                    raw_tool_calls = delta.get("tool_calls")
                    if isinstance(raw_tool_calls, list) and raw_tool_calls:
                        delta_tool_calls = raw_tool_calls
                        saw_tool_calls = True
                    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                    if finish_reason is not None:
                        stream_finish_reason = finish_reason
                    usage = event.get("usage")
                    normalized_usage = _coerce_openai_usage(usage) if isinstance(usage, dict) else None
                    if normalized_usage is not None and finish_reason is None and stream_finish_reason is not None:
                        finish_reason = stream_finish_reason

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
                    elif event_type == "content_block_start":
                        cb = event.get("content_block", {}) or {}
                        if cb.get("type") == "tool_use":
                            block_idx = event.get("index", 0)
                            tool_idx = len(anthropic_tool_call_map)
                            anthropic_tool_call_map[block_idx] = tool_idx
                            delta_tool_calls = [
                                {
                                    "index": tool_idx,
                                    "id": cb.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": cb.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                            saw_tool_calls = True
                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {}) or {}
                        delta_type = delta.get("type")
                        if delta_type == "text_delta":
                            delta_text = str(delta.get("text", ""))
                        elif delta_type == "input_json_delta":
                            block_idx = event.get("index", 0)
                            tool_idx = anthropic_tool_call_map.get(block_idx, 0)
                            partial_json = delta.get("partial_json", "")
                            delta_tool_calls = [
                                {
                                    "index": tool_idx,
                                    "function": {
                                        "arguments": partial_json,
                                    },
                                }
                            ]
                            saw_tool_calls = True
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
                        stop_reason = delta.get("stop_reason")
                        if stop_reason == "tool_use":
                            finish_reason = "tool_calls"
                        else:
                            finish_reason = _map_finish_reason(stop_reason)
                        if finish_reason is not None:
                            stream_finish_reason = finish_reason
                        normalized_usage = stream_usage
                    elif event_type in {"error"}:
                        error = event.get("error", {}) or {}
                        message = error.get("message") or "Anthropic stream failed"
                        raise VertexAPIError(502, str(message), code=error.get("type"), raw=event)

                else:
                    event_type = event.get("type")
                    if event_type == "response.output_item.added":
                        item = event.get("item", {}) or {}
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id") or item.get("id") or ""
                            tool_idx = len(set(xai_tool_call_map.values()))
                            if call_id:
                                xai_tool_call_map[call_id] = tool_idx
                            if "output_index" in event:
                                xai_tool_call_map[f"idx_{event['output_index']}"] = tool_idx
                            delta_tool_calls = [
                                {
                                    "index": tool_idx,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                            saw_tool_calls = True
                    elif event_type == "response.function_call_arguments.delta":
                        call_id = event.get("call_id") or event.get("id") or event.get("item_id")
                        tool_idx = None
                        if call_id and call_id in xai_tool_call_map:
                            tool_idx = xai_tool_call_map[call_id]
                        elif "output_index" in event and f"idx_{event['output_index']}" in xai_tool_call_map:
                            tool_idx = xai_tool_call_map[f"idx_{event['output_index']}"]
                        elif xai_tool_call_map:
                            tool_idx = list(xai_tool_call_map.values())[-1]
                        else:
                            tool_idx = 0
                        delta_arg = str(event.get("delta", ""))
                        delta_tool_calls = [
                            {
                                "index": tool_idx,
                                "function": {
                                    "arguments": delta_arg,
                                },
                            }
                        ]
                        saw_tool_calls = True
                    elif event_type == "response.output_item.done":
                        pass
                    elif event_type == "response.output_text.delta":
                        delta_text = str(event.get("delta", ""))
                    elif event_type == "response.completed":
                        completed = event.get("response", {}) or {}
                        normalized_usage = _xai_usage(completed)
                        stream_usage = normalized_usage
                        if saw_tool_calls:
                            finish_reason = "tool_calls"
                        else:
                            status = completed.get("status")
                            finish_reason = "stop" if status == "completed" else None
                            incomplete = completed.get("incomplete_details")
                            if isinstance(incomplete, dict):
                                finish_reason = _map_finish_reason(incomplete.get("reason")) or finish_reason
                            finish_reason = finish_reason or "stop"
                        stream_finish_reason = finish_reason
                    elif event_type in {"response.failed", "error"}:
                        error = event.get("error", {}) or {}
                        message = error.get("message") or "xAI Responses stream failed"
                        raise VertexAPIError(502, str(message), code=error.get("code"), raw=event)

                if delta_text or delta_tool_calls or finish_reason is not None or normalized_usage is not None:
                    event_dict: dict[str, Any] = {
                        "delta_text": delta_text,
                        "finish_reason": finish_reason,
                        "usage": normalized_usage,
                    }
                    if delta_tool_calls:
                        event_dict["delta_tool_calls"] = delta_tool_calls
                    yield event_dict

            if (
                protocol
                in {
                    FOUNDRY_OPENAI_PROTOCOL,
                    FOUNDRY_ANTHROPIC_PROTOCOL,
                    FOUNDRY_XAI_RESPONSES_PROTOCOL,
                    FOUNDRY_OPENAI_RESPONSES_PROTOCOL,
                }
                and saw_tool_calls
                and stream_finish_reason is None
            ):
                yield {
                    "delta_text": "",
                    "delta_tool_calls": None,
                    "finish_reason": "tool_calls",
                    "usage": stream_usage,
                }
        finally:
            if entered:
                await stream_context.__aexit__(None, None, None)
