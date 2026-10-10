"""Bounded, prompt-isolated summaries from the configured local Ollama model.

Unified with context_hide engine package (Single Source of Truth).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from context_hide.policy import _EXIT_CLAIM, verified_facts
from context_hide.summary import (
    _GENERIC_SUMMARY_PATTERNS,
    _IDENTIFIER_PATTERN,
    _INJECTION_OUTPUT_PATTERNS,
    _PLAIN_RESULT_PROMPT,
    _SYSTEM_PROMPT,
    _covers_critical_facts,
    _lossless_repeated_source,
    _result_source,
    is_lossless_encoded,
    prepare_result_source,
    restore_lossless_source,
    restore_repeated_source,
    validate_summary_text,
    verify_exit_claims,
    LFMSummarizer as _EngineLFMSummarizer,
    SummarizerConfig,
)
from context_hide.transport import LFMUnavailable, OnCall

Generate = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class LFMRequest:
    model: str
    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float
    response_format: dict[str, Any]
    reasoning: dict[str, str]
    timeout_seconds: int


class _BridgeLLMError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class LFMSummarizer(_EngineLFMSummarizer):
    """Prompt-isolated summary requester delegating to context_hide.summary.LFMSummarizer."""

    def __init__(
        self,
        *,
        generate: Callable[..., Awaitable[dict[str, Any]]],
        config: SummarizerConfig | None = None,
        settings: Any | None = None,
    ) -> None:
        async def intercepting_generate(**kwargs: Any) -> dict[str, Any]:
            res = await generate(**kwargs)
            if isinstance(res, dict):
                if res.get("tool_calls"):
                    raise _BridgeLLMError("unexpected_tool_calls")
                if res.get("finish_reason") == "length":
                    raise _BridgeLLMError("truncated_response")
            return res

        super().__init__(generate=intercepting_generate, config=config, settings=settings)


__all__ = [
    "LFMSummarizer",
    "LFMRequest",
    "LFMUnavailable",
    "OnCall",
    "Generate",
    "SummarizerConfig",
    "_PLAIN_RESULT_PROMPT",
    "_SYSTEM_PROMPT",
    "_covers_critical_facts",
    "_lossless_repeated_source",
    "_result_source",
    "is_lossless_encoded",
    "prepare_result_source",
    "restore_lossless_source",
    "restore_repeated_source",
    "validate_summary_text",
    "verified_facts",
    "verify_exit_claims",
]
