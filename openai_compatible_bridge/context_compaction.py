"""Opt-in compaction of already-read tool result bodies.

The correlation key is only the raw ``x-hermes-conversation`` header value
produced by Hermes' session-affinity code. This module does not authenticate
an owner, derive a branch id, or fall back to ``user``, a transcript
fingerprint, or the shared bridge API key. A missing header skips the feature.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from context_hide.engine import ContextHideEngine
from context_hide.model import (
    ContextItem,
    EngineConfig,
    ItemMetadata,
    MutationResult,
    ReplacementPlan,
    Scope,
    SpanChoice,
    ToolResultRecord,
    compute_invocation_digest,
    compute_sha256,
)
from context_hide.policy import (
    _BUSINESS_STATE_PATTERNS,
    _ERROR_PATTERNS,
    _EXIT_CLAIM,
    _PATH_EVIDENCE_PATTERN,
    _STRONG_EVIDENCE_PATTERN,
    HEAD_EXCERPTS,
    MAX_EVIDENCE_EXCERPTS,
    MAX_EXCERPT_LINE,
    TAIL_EXCERPTS,
    RuleSpanSelector,
    check_refusal_reason,
    excerpts_are_exact,
    is_business_state_critical,
    is_protected_error,
    refusal_reason,
    render_compaction,
    required_evidence_indexes,
    required_evidence_lines,
    validate_evidence_retention,
    verified_facts,
)
from context_hide.store import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MIN_CHARS,
    DEFAULT_TTL_SECONDS,
    _scope_key,
)
from context_hide.store import (
    MemoryContextStore as _BaseMemoryContextStore,
)
from context_hide.store import (
    _item_id as _base_item_id,
)
from context_hide.summary import (
    LFMSummarizer,
    SummarizerConfig,
    is_lossless_encoded,
    prepare_result_source,
    restore_lossless_source,
    restore_repeated_source,
    validate_summary_text,
    verify_exit_claims,
)
from context_hide.transport import (
    LFMUnavailable,
    OnCall,
    Summarizer,
    SummarizerError,
)

from openai_compatible_bridge.laya_http import LayaClient, LayaUnavailable
from openai_compatible_bridge.laya_selection import rank_items, select_extra_lines
from openai_compatible_bridge.semantic_invocation import (
    InvocationRejected,
    canonical,
    context_hint,
    invocation_digest,
    match_invocation,
)

logger = logging.getLogger("context_compaction")

AFFINITY_HEADER = "x-hermes-conversation"
FOUNDRY_OPENAI_PROTOCOL = "openai_chat_completions"
HIDE_TOOL = "hide_context"
LIST_TOOL = "list_context_items"
UNHIDE_TOOL = "unhide_context"
INTERNAL_TOOL_NAMES = (HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL)
MAX_AFFINITY_LENGTH = 256
DEFAULT_MAX_INTERNAL_ROUNDS = 3
DEFAULT_LFM_MODEL = "lfm2.5-thinking:latest"
DEFAULT_LFM_MAX_INPUT_CHARS = 50_000
DEFAULT_LFM_MAX_INPUT_BYTES = 12_288
DEFAULT_LFM_MAX_OUTPUT_TOKENS = 384
DEFAULT_LFM_TIMEOUT_SECONDS = 60
MAX_LFM_INPUT_CHARS = 100_000
MAX_LFM_INPUT_BYTES = 12_288
MAX_LFM_OUTPUT_TOKENS = 1_024
MAX_LFM_TIMEOUT_SECONDS = 300

_required_evidence_indexes = required_evidence_indexes


def bridge_scope(affinity: str) -> Scope:
    """Map Hermes session affinity key to host-neutral Scope."""
    return Scope(
        adapter_id="bridge",
        host_profile="hermes",
        session_id=affinity,
        branch_scope="default",
    )

Generate = Callable[..., Awaitable[dict[str, Any]]]
LFMSummarize = Callable[..., Awaitable[dict[str, Any]]]

__all__ = [
    "AFFINITY_HEADER",
    "DEFAULT_LFM_MAX_INPUT_BYTES",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MIN_CHARS",
    "DEFAULT_TTL_SECONDS",
    "HEAD_EXCERPTS",
    "HIDE_TOOL",
    "INTERNAL_TOOL_NAMES",
    "LIST_TOOL",
    "MAX_EVIDENCE_EXCERPTS",
    "MAX_EXCERPT_LINE",
    "TAIL_EXCERPTS",
    "UNHIDE_TOOL",
    "_BUSINESS_STATE_PATTERNS",
    "_ERROR_PATTERNS",
    "_EXIT_CLAIM",
    "_PATH_EVIDENCE_PATTERN",
    "_STRONG_EVIDENCE_PATTERN",
    "CompactionPlan",
    "CompactionSettings",
    "CompactionUpstreamError",
    "ContextHideEngine",
    "ContextItem",
    "EngineConfig",
    "ItemMetadata",
    "LFMSummarizer",
    "LFMUnavailable",
    "MemoryContextStore",
    "MutationResult",
    "OnCall",
    "RankingComparison",
    "ReplacementPlan",
    "RuleSpanSelector",
    "Scope",
    "SelectorComparison",
    "SpanChoice",
    "Summarizer",
    "SummarizerConfig",
    "SummarizerError",
    "ToolResultRecord",
    "TurnMeasurement",
    "TurnOutcome",
    "_excerpts_are_exact",
    "_hide_call",
    "_item_id",
    "_list_call",
    "_refusal_reason",
    "_replace",
    "_required_evidence_indexes",
    "_scope_key",
    "_sha256",
    "_unhide_call",
    "aggregate_usage",
    "apply_visibility",
    "bridge_scope",
    "canonical",
    "check_refusal_reason",
    "compare_span_selectors",
    "compare_unhide_rankers",
    "compute_invocation_digest",
    "compute_sha256",
    "excerpts_are_exact",
    "expire_context_periodically",
    "internal_tool_definitions",
    "is_business_state_critical",
    "is_lossless_encoded",
    "is_protected_error",
    "load_settings",
    "plan_request",
    "prepare_result_source",
    "rank_compacted_items",
    "refusal_reason",
    "render_compaction",
    "required_evidence_indexes",
    "required_evidence_lines",
    "restore_lossless_source",
    "restore_repeated_source",
    "run_turn",
    "validate_evidence_retention",
    "validate_summary_text",
    "verified_facts",
    "verify_exit_claims",
]


@dataclass(frozen=True)
class CompactionSettings:
    enabled: bool = False
    header_name: str = AFFINITY_HEADER
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    max_bytes: int = DEFAULT_MAX_BYTES
    max_internal_rounds: int = DEFAULT_MAX_INTERNAL_ROUNDS
    min_chars: int = DEFAULT_MIN_CHARS
    lfm_enabled: bool = False
    lfm_model: str = DEFAULT_LFM_MODEL
    lfm_max_input_chars: int = DEFAULT_LFM_MAX_INPUT_CHARS
    lfm_max_input_bytes: int = DEFAULT_LFM_MAX_INPUT_BYTES
    lfm_max_output_tokens: int = DEFAULT_LFM_MAX_OUTPUT_TOKENS
    lfm_timeout_seconds: int = DEFAULT_LFM_TIMEOUT_SECONDS
    laya_enabled: bool = False
    laya_validated: bool = False
    laya_base_url: str = ""
    laya_approved_origin: str = ""
    laya_timeout_seconds: int = 60

    @property
    def laya_available(self) -> bool:
        return bool(self.laya_base_url and self.laya_approved_origin)

    @property
    def laya_active(self) -> bool:
        return self.enabled and self.laya_enabled and self.laya_validated and self.laya_available

    @property
    def lfm_active(self) -> bool:
        return self.enabled and self.lfm_enabled


@dataclass
class TurnMeasurement:
    provider_calls: int = 0
    internal_rounds: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    laya_calls: int = 0
    lfm_calls: int = 0
    lfm_applied: bool = False
    context_hint_provided: bool = False
    lfm_fallback_reason: str | None = None
    applied: bool = False
    skipped_reason: str | None = None
    usages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TurnOutcome:
    skipped: bool
    result: dict[str, Any] | None = None
    error: tuple[int, str, str] | None = None
    measurement: TurnMeasurement | None = None
    prior_usages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CompactionPlan:
    affinity_key: str
    visibility_only: bool
    tools: list[dict[str, Any]] | None
    scope: Scope = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.scope is None:
            object.__setattr__(self, "scope", bridge_scope(self.affinity_key))


@dataclass(frozen=True)
class SelectorComparison:
    selector: str
    cases: int
    missing_required: int
    invented_text: int
    accepted: bool


@dataclass(frozen=True)
class RankingComparison:
    ranker: str
    cases: int
    missed_relevant: int
    irrelevant_ranked_first: int
    accepted: bool


def load_settings(environ: Mapping[str, str] | None = None) -> CompactionSettings:
    source = os.environ if environ is None else environ
    return CompactionSettings(
        enabled=_env_flag(source, "CONTEXT_COMPACTION_ENABLED"),
        header_name=_env_text(source, "CONTEXT_COMPACTION_AFFINITY_HEADER", AFFINITY_HEADER),
        ttl_seconds=_env_int(source, "CONTEXT_COMPACTION_TTL_SECONDS", DEFAULT_TTL_SECONDS),
        max_bytes=_env_int(source, "CONTEXT_COMPACTION_MAX_BYTES", DEFAULT_MAX_BYTES),
        max_internal_rounds=_env_int(
            source,
            "CONTEXT_COMPACTION_MAX_INTERNAL_ROUNDS",
            DEFAULT_MAX_INTERNAL_ROUNDS,
        ),
        min_chars=_env_int(source, "CONTEXT_COMPACTION_MIN_CHARS", DEFAULT_MIN_CHARS),
        lfm_enabled=_env_flag(source, "CONTEXT_COMPACTION_LFM_ENABLED"),
        lfm_model=_lfm_model(source),
        lfm_max_input_chars=_bounded_env_int(
            source, "CONTEXT_COMPACTION_LFM_MAX_INPUT_CHARS", DEFAULT_LFM_MAX_INPUT_CHARS, 256, MAX_LFM_INPUT_CHARS,
        ),
        lfm_max_input_bytes=_bounded_env_int(
            source, "CONTEXT_COMPACTION_LFM_MAX_INPUT_BYTES", DEFAULT_LFM_MAX_INPUT_BYTES,
            256, MAX_LFM_INPUT_BYTES,
        ),
        lfm_max_output_tokens=_bounded_env_int(
            source, "CONTEXT_COMPACTION_LFM_MAX_OUTPUT_TOKENS", DEFAULT_LFM_MAX_OUTPUT_TOKENS, 32, MAX_LFM_OUTPUT_TOKENS,
        ),
        lfm_timeout_seconds=_bounded_env_int(
            source, "CONTEXT_COMPACTION_LFM_TIMEOUT_SECONDS", DEFAULT_LFM_TIMEOUT_SECONDS, 1, MAX_LFM_TIMEOUT_SECONDS,
        ),
        laya_enabled=_env_flag(source, "CONTEXT_COMPACTION_LAYA_ENABLED"),
        laya_validated=_env_flag(source, "CONTEXT_COMPACTION_LAYA_VALIDATED"),
        laya_base_url=_env_text(source, "LAYA_BASE_URL", "") if source.get("LAYA_BASE_URL") else "",
        laya_approved_origin=_env_text(source, "LAYA_APPROVED_ORIGIN", "") if source.get("LAYA_APPROVED_ORIGIN") else "",
        laya_timeout_seconds=_env_int(source, "CONTEXT_COMPACTION_LAYA_TIMEOUT_SECONDS", 60),
    )


def affinity_from_headers(headers: Mapping[str, Any] | None, header_name: str) -> str | None:
    if headers is None or not header_name:
        return None
    target = header_name.lower()
    found: str | None = None
    for key, value in headers.items():
        if str(key).lower() != target:
            continue
        found = "" if value is None else str(value)
        break
    if found is None:
        return None
    if not found.strip() or len(found) > MAX_AFFINITY_LENGTH or "\n" in found or "\r" in found:
        return None
    return found


def plan_request(
    *,
    settings: CompactionSettings,
    headers: Mapping[str, Any] | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    provider: str,
    protocol: str | None,
    stream: bool,
) -> tuple[CompactionPlan | None, str]:
    if not settings.enabled:
        return None, "disabled"
    if stream and (protocol or FOUNDRY_OPENAI_PROTOCOL) not in {FOUNDRY_OPENAI_PROTOCOL, "openai_responses", "anthropic_messages", "google_generate_content", "xai_responses"}:
        return None, "streaming"
    # Only native protocols verified through the private-tool continuation loop.
    # Unrecognized protocols remain excluded from the private continuation loop.
    if provider != "foundry" or (protocol or FOUNDRY_OPENAI_PROTOCOL) not in {
        FOUNDRY_OPENAI_PROTOCOL, "openai_responses", "anthropic_messages", "google_generate_content", "xai_responses",
    }:
        return None, "unsupported_protocol"
    affinity = affinity_from_headers(headers, settings.header_name)
    if affinity is None:
        return None, "missing_affinity"
    names = tool_names(tools)
    if any(name in names for name in INTERNAL_TOOL_NAMES):
        return None, "tool_name_collision"
    if _forced_named_tool(tool_choice):
        return None, "forced_tool_choice"
    visibility_only = tools is None or tool_choice == "none"
    injected = None if visibility_only else _inject_tools(tools or [])
    scope = bridge_scope(affinity)
    return CompactionPlan(affinity, visibility_only, injected, scope=scope), "apply"


class MemoryContextStore(_BaseMemoryContextStore):
    """Single-user, process-local cache keyed by (raw header, item id).

    cachetools owns expiration and size accounting. All access is locked; admission
    is checked before insertion so its LRU policy never evicts a live original.
    The budget accounts UTF-8 payload plus a fixed metadata allowance, not RSS.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        min_chars: int = DEFAULT_MIN_CHARS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(
            ttl_seconds=ttl_seconds,
            max_bytes=max_bytes,
            min_chars=min_chars,
            clock=clock,
        )
        self.last_measurement: TurnMeasurement | None = None
        self._engine: ContextHideEngine | None = None

    @property
    def engine(self) -> ContextHideEngine:
        if self._engine is None:
            self._engine = ContextHideEngine(store=self)
        return self._engine

    def record_measurement(self, measurement: TurnMeasurement) -> None:
        self.last_measurement = measurement

    def get(self, scope: Scope | str, item_id: str, *, now: float | None = None) -> ContextItem | None:
        item = super().get(scope, item_id, now=now)
        if item is not None:
            return item
        if isinstance(scope, str):
            bridge_sc = bridge_scope(scope)
            return super().get(bridge_sc, item_id, now=now)
        if isinstance(scope, Scope) and scope == bridge_scope(scope.session_id):
            return super().get(scope.session_id, item_id, now=now)
        return None

    def items(self, scope: Scope | str, *, now: float | None = None) -> tuple[ContextItem, ...]:
        res = super().items(scope, now=now)
        if res:
            return res
        if isinstance(scope, str):
            bridge_sc = bridge_scope(scope)
            return super().items(bridge_sc, now=now)
        if isinstance(scope, Scope) and scope == bridge_scope(scope.session_id):
            return super().items(scope.session_id, now=now)
        return ()

    def unhide(self, scope: Scope | str, item_id: str, *, now: float | None = None) -> MutationResult:
        res = super().unhide(scope, item_id, now=now)
        if res.ok or res.error != "not_found":
            return res
        if isinstance(scope, str):
            bridge_sc = bridge_scope(scope)
            return super().unhide(bridge_sc, item_id, now=now)
        if isinstance(scope, Scope) and scope == bridge_scope(scope.session_id):
            return super().unhide(scope.session_id, item_id, now=now)
        return res


async def expire_context_periodically(store: MemoryContextStore) -> None:
    """App-owned maintenance task; no requests are needed to release expired data."""
    while True:
        await asyncio.sleep(60)
        store.expire()


def apply_visibility(
    messages: list[dict[str, Any]],
    *,
    affinity: str,
    store: MemoryContextStore,
    now: float | None = None,
) -> list[dict[str, Any]]:
    visible = copy.deepcopy(messages)
    for message in visible:
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        semantic_item = next((item for item in store.items(affinity, now=now)
                              if item.tool_call_id == tool_call_id
                              and message["content"] in (item.original, item.compacted)
                              and item.invocation_digest is not None), None)
        if semantic_item is not None:
            try:
                invocation = match_invocation(messages, tool_call_id)
                matches = invocation_digest(invocation, messages, tool_call_id) == semantic_item.invocation_digest
            except InvocationRejected:
                matches = False
            if not matches:
                message["content"] = semantic_item.original
                continue
        message["content"] = store.visible_content(
            affinity,
            tool_call_id,
            message["content"],
            now=now,
        )
    return visible


def rank_compacted_items(query: str, items: tuple[ContextItem, ...] | list[ContextItem]) -> list[ContextItem]:
    compacted = [item for item in items if item.visibility == "compacted"]
    tokens = [token.lower() for token in re.findall(r"[0-9A-Za-z가-힣_]{2,}", query or "")]

    def score(item: ContextItem) -> int:
        if not tokens:
            return 0
        haystack = " ".join(
            part for part in (item.tool_name, item.tool_call_id, item.compacted) if part
        ).lower()
        return sum(1 for token in tokens if token in haystack)

    return sorted(compacted, key=lambda item: (-score(item), item.item_id))


def internal_tool_definitions() -> list[dict[str, Any]]:
    return [
        _function_tool(
            HIDE_TOOL,
            "Hide one already-read prior tool result body from future upstream requests. "
            "Pass the required tool_call_id of a unique role=tool result in the current conversation. "
            "The bridge stores the original and replaces only that result body with exact source excerpts; "
            "when opt-in LFM is enabled it may add a separately marked untrusted summary. "
            "Does not rerun the tool or hide the original assistant tool call. "
            "Optional context is validated but ignored for compatibility. "
            "Do not invent result facts or rewrite the command.",
            {
                "type": "object",
                "properties": {
                    "tool_call_id": {"type": "string"},
                    "context": {"type": "object", "properties": {
                        "purpose": {"type": "string", "maxLength": 300},
                        "retain_for": {"type": "string", "maxLength": 300}},
                        "additionalProperties": False},
                },
                "required": ["tool_call_id"],
                "additionalProperties": False,
            },
        ),
        _function_tool(
            LIST_TOOL,
            "List hidden tool results for this conversation header. "
            "Returns ids and verified metadata only.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        _function_tool(
            UNHIDE_TOOL,
            "Restore the exact stored original tool result body on later upstream requests. "
            "Pass item_id returned by hide_context or list_context_items. Does not regenerate a summary or rerun the tool.",
            {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
                "additionalProperties": False,
            },
        ),
    ]


def aggregate_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = 0
    completion = 0
    total = 0
    for usage in usages:
        prompt += _int_or_zero(usage.get("prompt_tokens"))
        completion += _int_or_zero(usage.get("completion_tokens"))
        total += _int_or_zero(usage.get("total_tokens"))
    if total == 0:
        total = prompt + completion
    aggregated: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    for detail_key in ("prompt_tokens_details", "completion_tokens_details"):
        fields = _common_detail_fields(usages, detail_key)
        if fields:
            aggregated[detail_key] = fields
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        if usages and all(key in usage for usage in usages):
            aggregated[key] = sum(_int_or_zero(usage.get(key)) for usage in usages)
    return aggregated


def measurement_from_usages(
    usages: list[dict[str, Any]],
    *,
    applied: bool,
    rounds: int,
    skipped: str | None,
    laya_calls: int = 0,
    lfm_calls: int = 0,
    lfm_applied: bool = False,
    context_hint_provided: bool = False,
    lfm_fallback_reason: str | None = None,
) -> TurnMeasurement:
    usage = aggregate_usage(usages)
    cache_read = None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and "cached_tokens" in details:
        cache_read = details["cached_tokens"]
    elif "cache_read_input_tokens" in usage:
        cache_read = usage["cache_read_input_tokens"]
    cache_write = usage.get("cache_creation_input_tokens")
    return TurnMeasurement(
        provider_calls=len(usages),
        internal_rounds=rounds,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write if isinstance(cache_write, int) else None,
        laya_calls=laya_calls,
        lfm_calls=lfm_calls,
        lfm_applied=lfm_applied,
        context_hint_provided=context_hint_provided,
        lfm_fallback_reason=lfm_fallback_reason,
        applied=applied,
        skipped_reason=skipped,
        usages=list(usages),
    )


async def run_turn(
    *,
    generate: Generate,
    base_kwargs: dict[str, Any],
    messages: list[dict[str, Any]],
    plan: CompactionPlan,
    store: MemoryContextStore,
    settings: CompactionSettings,
    laya_client: LayaClient | None = None,
    lfm_summarizer: LFMSummarize | None = None,
    correlation_id: str | None = None,
    on_local_failure: Callable[[int, str, int], None] | None = None,
) -> TurnOutcome:
    hermes_messages = copy.deepcopy(messages)
    suffix: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    rounds = 0
    laya_calls = [0]
    lfm_state: dict[str, Any] = {"calls": 0, "applied": False, "fallback_reason": None}
    def count_laya_call() -> None:
        if laya_calls[0] >= 2:
            raise LayaUnavailable("turn_call_limit")
        laya_calls[0] += 1
    def count_lfm_call() -> None:
        if lfm_state["calls"] >= 1:
            raise RuntimeError("turn_call_limit")
        lfm_state["calls"] += 1
    def current_measurement(*, applied: bool, rounds: int, skipped: str | None) -> TurnMeasurement:
        return measurement_from_usages(
            usages,
            applied=applied,
            rounds=rounds,
            skipped=skipped,
            laya_calls=laya_calls[0],
            lfm_calls=lfm_state["calls"],
            lfm_applied=lfm_state["applied"],
            context_hint_provided=lfm_state.get("context_hint_provided", False),
            lfm_fallback_reason=lfm_state["fallback_reason"],
        )

    def report_store_unavailable() -> None:
        if on_local_failure is not None:
            on_local_failure(0, "context_compaction_store_unavailable", rounds)

    # Never send a whole transcript to the separate System-One server.
    goal = next((message["content"] for message in reversed(messages)
                 if message.get("role") == "user" and isinstance(message.get("content"), str)), "")
    selector = laya_client if settings.laya_active else None
    try:
        _ = apply_visibility(hermes_messages, affinity=plan.affinity_key, store=store)
    except Exception:  # noqa: BLE001
        report_store_unavailable()
        return TurnOutcome(
            skipped=True,
            measurement=measurement_from_usages([], applied=False, rounds=0, skipped="store_unavailable"),
        )

    while True:
        try:
            outbound = apply_visibility(hermes_messages, affinity=plan.affinity_key, store=store)
        except Exception:  # noqa: BLE001
            if usages:
                measurement = current_measurement(applied=True, rounds=rounds, skipped="store_unavailable")
                store.record_measurement(measurement)
                return TurnOutcome(
                    skipped=False,
                    error=(502, "Context compaction could not complete local processing.", "context_compaction_store_unavailable"),
                    measurement=measurement,
                    prior_usages=usages,
                )
            report_store_unavailable()
            return TurnOutcome(skipped=True)
        outbound.extend(copy.deepcopy(suffix))
        kwargs = dict(base_kwargs)
        kwargs["messages"] = outbound
        if plan.tools is not None:
            kwargs["tools"] = copy.deepcopy(plan.tools)
        try:
            result = await generate(**kwargs)
        except Exception as exc:
            if isinstance(exc, CompactionUpstreamError):
                raise
            if usages:
                measurement = current_measurement(applied=True, rounds=rounds, skipped="upstream_error")
                store.record_measurement(measurement)
                raise CompactionUpstreamError(
                    usages, phase="continuation", round_number=rounds + 1, correlation_id=correlation_id,
                ) from exc
            raise CompactionUpstreamError(
                [], phase="initial", round_number=1, correlation_id=correlation_id,
            ) from exc
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        usages.append(usage)
        tool_calls = result.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []
        internal = [call for call in tool_calls if _call_name(call) in INTERNAL_TOOL_NAMES]
        external = [call for call in tool_calls if _call_name(call) not in INTERNAL_TOOL_NAMES]
        if internal and external:
            measurement = current_measurement(applied=True, rounds=rounds, skipped=None)
            store.record_measurement(measurement)
            logger.info(
                "context_compaction calls=%s rounds=%s mixed=1 lfm_calls=%s lfm_applied=%s lfm_fallback=%s",
                len(usages), rounds, lfm_state["calls"], lfm_state["applied"],
                bool(lfm_state["fallback_reason"]),
            )
            return TurnOutcome(
                skipped=False,
                result=_public_result(result, external, usage=aggregate_usage(usages)),
                measurement=measurement,
            )
        if not internal:
            measurement = current_measurement(applied=True, rounds=rounds, skipped=None)
            store.record_measurement(measurement)
            logger.info(
                "context_compaction calls=%s rounds=%s lfm_calls=%s lfm_applied=%s lfm_fallback=%s",
                len(usages), rounds, lfm_state["calls"], lfm_state["applied"],
                bool(lfm_state["fallback_reason"]),
            )
            return TurnOutcome(
                skipped=False,
                result=_public_result(result, external, usage=aggregate_usage(usages)),
                measurement=measurement,
            )
        rounds += 1
        if rounds > settings.max_internal_rounds:
            measurement = current_measurement(applied=True, rounds=rounds, skipped="loop_limit")
            store.record_measurement(measurement)
            return TurnOutcome(
                skipped=False,
                error=(
                    502,
                    "Context compaction could not complete local processing.",
                    "context_compaction_loop_limit",
                ),
                measurement=measurement,
                prior_usages=usages,
            )
        suffix.append(_assistant_message(result, internal))
        for call in internal:
            payload = await _execute_internal(
                call, hermes_messages, plan, store, selector, goal, count_laya_call,
                settings=settings,
                lfm_summarizer=lfm_summarizer,
                count_lfm_call=count_lfm_call,
                lfm_state=lfm_state,
            )
            suffix.append(_tool_message(call, payload))
            if payload.get("error") == "store_unavailable":
                report_store_unavailable()

    raise AssertionError("unreachable")


class CompactionUpstreamError(Exception):
    def __init__(
        self,
        usages: list[dict[str, Any]],
        *,
        phase: str = "continuation",
        round_number: int | None = None,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__("context compaction upstream failed")
        self.usages = usages
        self.phase = phase
        self.round_number = round_number
        self.correlation_id = correlation_id


def compare_span_selectors(
    cases: list[dict[str, Any]],
    selector: RuleSpanSelector | Any,
    *,
    validated: bool,
) -> SelectorComparison:
    missing = 0
    invented = 0
    for case in cases:
        original = str(case["text"])
        choice = selector.select(original)
        lines = () if choice is None else choice.lines
        for required in case.get("must_keep", ()):
            if required not in lines:
                missing += 1
        for line in lines:
            if line not in original.splitlines():
                invented += 1
    source = getattr(selector, "source", "unknown")
    accepted = validated and missing == 0 and invented == 0 and source != "rule"
    if source == "rule":
        accepted = missing == 0 and invented == 0
    return SelectorComparison(source, len(cases), missing, invented, accepted)


def compare_unhide_rankers(
    cases: list[dict[str, Any]],
    *,
    ranker_name: str,
    rank,
    validated: bool,
) -> RankingComparison:
    missed = 0
    false_first = 0
    for case in cases:
        ranked = [item.item_id for item in rank(case["query"], case["items"])]
        relevant = list(case["relevant"])
        if any(item_id not in ranked for item_id in relevant):
            missed += 1
        if ranked and relevant and ranked[0] not in relevant and ranked[0] in case.get("irrelevant", ()):
            false_first += 1
    accepted = validated and missed == 0 and false_first == 0 and ranker_name != "rule"
    if ranker_name == "rule":
        accepted = missed == 0
    return RankingComparison(ranker_name, len(cases), missed, false_first, accepted)


def tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
        elif isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return names


async def _execute_internal(
    call: dict[str, Any],
    hermes_messages: list[dict[str, Any]],
    plan: CompactionPlan,
    store: MemoryContextStore,
    laya_client: LayaClient | None,
    goal: str,
    count_call: Callable[[], None],
    *,
    settings: CompactionSettings,
    lfm_summarizer: LFMSummarize | None,
    count_lfm_call: Callable[[], None],
    lfm_state: dict[str, Any],
) -> dict[str, Any]:
    name = _call_name(call)
    try:
        args = _call_arguments(call)
    except ValueError:
        return {"ok": False, "error": "invalid_arguments"}
    try:
        if name == HIDE_TOOL:
            choice = None
            tool_call_id = args.get("tool_call_id")
            if set(args) - {"tool_call_id", "context"} or not isinstance(tool_call_id, str) or not tool_call_id:
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    return {"ok": False, "error": "missing_tool_call_id"}
                return {"ok": False, "error": "invalid_arguments"}
            try:
                context_hint(args)
            except InvocationRejected as exc:
                return {"ok": False, "error": str(exc)}
            if settings.lfm_active:
                return await _semantic_hide(args, hermes_messages, plan.affinity_key, store,
                                            lfm_summarizer, count_lfm_call, lfm_state)
            if laya_client is not None and isinstance(tool_call_id, str):
                matches = _tool_messages(hermes_messages, tool_call_id)
                if len(matches) == 1 and isinstance(matches[0].get("content"), str):
                    original = matches[0]["content"]
                    content_sha = _sha256(original)
                    existing = store.get(plan.affinity_key, _item_id(plan.affinity_key, tool_call_id, content_sha))
                    # No remote call for already fixed items, protected errors, short
                    # outputs, evidence overflow, or outputs too large for storage.
                    if (existing is None and _refusal_reason(original, store.min_chars) is None
                            and len(original.encode("utf-8")) <= store.max_bytes - store.used_bytes):
                        baseline = RuleSpanSelector().select(original)
                        if baseline is not None:
                            extra = await select_extra_lines(original, goal, baseline.lines, laya_client, count_call)
                            # An abstention or failed remote decision cannot prove that
                            # a middle task-relevant line is safe to omit.
                            if extra is None:
                                return {"ok": False, "error": "verification_failed"}
                            choice = SpanChoice(extra, "laya")
            return _hide_call(args, hermes_messages, plan.affinity_key, store, choice=choice)
        if name == LIST_TOOL:
            if laya_client is None:
                return _list_call(args, plan.affinity_key, store)
            query = args.get("query")
            if not isinstance(query, str):
                query = ""
            ranked = rank_compacted_items(query, store.items(plan.affinity_key))
            reordered = await rank_items(query, ranked, laya_client, count_call)
            # A slow remote decision must not reintroduce expired or restored items.
            fresh = rank_compacted_items(query, store.items(plan.affinity_key))
            order = {item.item_id: index for index, item in enumerate(reordered)}
            return _list_call(args, plan.affinity_key, store,
                              ranked=sorted(fresh, key=lambda item: order.get(item.item_id, len(order))))
        if name == UNHIDE_TOOL:
            return _unhide_call(args, plan.affinity_key, store)
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": "store_unavailable"}
    return {"ok": False, "error": "unknown_tool"}


async def _semantic_hide(args, messages, affinity, store, summarizer, on_call, state):
    ident = args["tool_call_id"]
    context_hint(args)  # Validate legacy input, but do not use or store it.
    try:
        invocation = match_invocation(messages, ident)
    except InvocationRejected as exc:
        reason = str(exc)
        state["fallback_reason"] = reason
        if reason == "unsupported_tool":
            return _hide_call(args, messages, affinity, store)
        return {"ok": False, "error": reason}
    digest = invocation_digest(invocation, messages, ident)
    original_call = next(c for m in messages if m.get("role") == "assistant"
                         for c in (m.get("tool_calls") or []) if isinstance(c, dict) and c.get("id") == ident)
    options_omitted = bool(set(_call_arguments(original_call)) - set(invocation["arguments"]))
    original = _tool_messages(messages, ident)[0]["content"]
    # A caller may submit the already rendered body; recover the saved source.
    for item in store.items(affinity):
        if item.tool_call_id == ident and original == item.compacted:
            original = item.original
            break
    item_id = _item_id(affinity, ident, _sha256(original))
    existing = store.get(affinity, item_id)
    if existing is not None:
        saved = store.compact(affinity=affinity, tool_call_id=ident, original=original,
                              tool_name=invocation["tool_name"], invocation=invocation, input_digest=digest)
        if saved.ok and saved.item.compaction_source == "lfm":
            state["applied"] = True
        return _mutation_payload(saved)
    refusal = _refusal_reason(original, store.min_chars)
    if refusal:
        state["fallback_reason"] = refusal
        return {"ok": False, "error": refusal}
    if len(original.encode()) > store.max_bytes - store.used_bytes:
        state["fallback_reason"] = "store_full"
        return {"ok": False, "error": "store_full"}
    if RuleSpanSelector().select(original) is None:
        state["fallback_reason"] = "no_safe_excerpt"
        return {"ok": False, "error": "verification_failed"}
    token, error = store.reserve(affinity, item_id, digest)
    if error:
        return {"ok": False, "error": error}
    try:
        lines = original.splitlines()
        required = tuple(lines[i] for i in _required_evidence_indexes(lines))
        choice = None
        try:
            if summarizer is None:
                raise RuntimeError("summarizer_unavailable")
            summary = await summarizer(original, required, on_call, invocation=invocation)
            choice = SpanChoice(required, "lfm", summary)
        except Exception as exc:  # noqa: BLE001
            state["fallback_reason"] = getattr(exc, "reason", None) or (
                str(exc) if str(exc) in {"turn_call_limit", "summarizer_unavailable"} else "summary_failed")
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError
        saved = store.compact(affinity=affinity, tool_call_id=ident, original=original,
                              tool_name=invocation["tool_name"], choice=choice, invocation=invocation,
                              reservation=token, input_digest=digest, options_omitted=options_omitted)
        if saved.ok and saved.item.compaction_source == "lfm":
            state["applied"] = True
        elif state["fallback_reason"] is None:
            state["fallback_reason"] = saved.error or "rule_fallback"
        return _mutation_payload(saved)
    finally:
        store.release(affinity, item_id, token)


def _hide_call(
    args: Mapping[str, Any],
    messages: list[dict[str, Any]],
    affinity: str,
    store: MemoryContextStore,
    *,
    choice: SpanChoice | None = None,
) -> dict[str, Any]:
    tool_call_id = args.get("tool_call_id")
    if set(args) - {"tool_call_id", "context"} or not isinstance(tool_call_id, str) or not tool_call_id:
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return {"ok": False, "error": "missing_tool_call_id"}
        return {"ok": False, "error": "invalid_arguments"}
    matches = _tool_messages(messages, tool_call_id)
    if not matches:
        return {"ok": False, "error": "not_found"}
    if len(matches) > 1:
        return {"ok": False, "error": "ambiguous"}
    message = matches[0]
    content = message.get("content")
    if not isinstance(content, str):
        return {"ok": False, "error": "unsupported_content"}
    result = store.compact(
        affinity=affinity,
        tool_call_id=tool_call_id,
        original=content,
        tool_name=_tool_name_for(messages, tool_call_id),
        choice=choice,
    )
    return _mutation_payload(result)


def _list_call(args: Mapping[str, Any], affinity: str, store: MemoryContextStore, *, ranked: list[ContextItem] | None = None) -> dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str):
        query = ""
    if ranked is None:
        ranked = rank_compacted_items(query, store.items(affinity))
    return {
        "ok": True,
        "items": [
            {
                "item_id": item.item_id,
                "tool_call_id": item.tool_call_id,
                "tool_name": item.tool_name,
                "original_available": True,
                "visibility": "hidden",
                "original_bytes": len(item.original.encode("utf-8")),
                "compacted_bytes": len(item.compacted.encode("utf-8")),
            }
            for item in ranked
        ],
    }


def _unhide_call(args: Mapping[str, Any], affinity: str, store: MemoryContextStore) -> dict[str, Any]:
    item_id = args.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        return {"ok": False, "error": "missing_item_id"}
    return _mutation_payload(store.unhide(affinity, item_id))


def _mutation_payload(result: MutationResult) -> dict[str, Any]:
    if not result.ok or result.item is None:
        return {"ok": False, "error": result.error or "rejected"}
    return {
        "ok": True,
        "item_id": result.item.item_id,
        "tool_call_id": result.item.tool_call_id,
        "visibility": "hidden" if result.item.visibility == "compacted" else "visible",
        "original_available": True,
        "compaction_source": result.item.compaction_source,
    }


def _tool_messages(messages: list[dict[str, Any]], tool_call_id: str) -> list[dict[str, Any]]:
    return [
        message
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id") == tool_call_id
    ]


def _tool_name_for(messages: list[dict[str, Any]], tool_call_id: str) -> str | None:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id") == tool_call_id:
                name = _call_name(call)
                return name if isinstance(name, str) else None
    return None


def _public_result(result: dict[str, Any], tool_calls: list[dict[str, Any]], *, usage: dict[str, Any]) -> dict[str, Any]:
    public = {
        "text": result.get("text"),
        "tool_calls": tool_calls or None,
        "finish_reason": result.get("finish_reason") or ("tool_calls" if tool_calls else "stop"),
        "usage": usage,
    }
    if tool_calls and not public["finish_reason"]:
        public["finish_reason"] = "tool_calls"
    return public


def _assistant_message(result: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    text = result.get("text")
    calls = copy.deepcopy(tool_calls)
    native_parts = result.get("_google_call_parts") or {}
    for call in calls:
        if call.get("id") in native_parts:
            call["_google_part"] = copy.deepcopy(native_parts[call["id"]])
    return {
        "role": "assistant",
        "content": text if text else None,
        "tool_calls": calls,
    }


def _tool_message(call: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.get("id"),
        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def _call_name(call: Any) -> str | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def _call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    raw = function.get("arguments") if isinstance(function, dict) else None
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError("invalid arguments")  # noqa: TRY004
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("invalid arguments")  # noqa: TRY004
    return parsed


def _inject_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [*copy.deepcopy(tools), *internal_tool_definitions()]


def _function_tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _forced_named_tool(tool_choice: str | dict[str, Any] | None) -> bool:
    return isinstance(tool_choice, dict)


def _refusal_reason(original: str, min_chars: int) -> str | None:
    return check_refusal_reason(original, min_chars)


def _excerpts_are_exact(original: str, lines: tuple[str, ...] | list[str]) -> bool:
    return excerpts_are_exact(original, lines)


def _item_id(affinity: str, tool_call_id: str, content_sha: str) -> str:
    return _base_item_id(affinity, tool_call_id, content_sha)


def _sha256(value: str) -> str:
    return compute_sha256(value)


def _replace(item: ContextItem, **changes: Any) -> ContextItem:
    from dataclasses import replace
    return replace(item, **changes)


def _common_detail_fields(usages: list[dict[str, Any]], detail_key: str) -> dict[str, int]:
    if not usages:
        return {}
    keys: set[str] | None = None
    for usage in usages:
        detail = usage.get(detail_key)
        if not isinstance(detail, dict):
            return {}
        present = {key for key, value in detail.items() if isinstance(key, str) and _exact_int(value) is not None}
        keys = present if keys is None else keys & present
    if not keys:
        return {}
    summed: dict[str, int] = {}
    for key in sorted(keys):
        total = 0
        for usage in usages:
            parsed = _exact_int(usage[detail_key][key])
            if parsed is None:
                return {}
            total += parsed
        summed[key] = total
    return summed


def _int_or_zero(value: Any) -> int:
    parsed = _exact_int(value)
    return 0 if parsed is None else parsed


def _exact_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _env_flag(source: Mapping[str, str], name: str) -> bool:
    value = source.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(source: Mapping[str, str], name: str, default: str) -> str:
    value = source.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(source: Mapping[str, str], name: str, default: int) -> int:
    value = source.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _bounded_env_int(
    source: Mapping[str, str], name: str, default: int, minimum: int, maximum: int,
) -> int:
    parsed = _env_int(source, name, default)
    return parsed if minimum <= parsed <= maximum else default


def _lfm_model(source: Mapping[str, str]) -> str:
    value = _env_text(source, "CONTEXT_COMPACTION_LFM_MODEL", DEFAULT_LFM_MODEL)
    return value if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", value) else DEFAULT_LFM_MODEL
