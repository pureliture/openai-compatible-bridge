"""Opt-in compaction of already-read tool result bodies.

The correlation key is only the raw ``x-hermes-conversation`` header value
produced by Hermes' session-affinity code. This module does not authenticate
an owner, derive a branch id, or fall back to ``user``, a transcript
fingerprint, or the shared bridge API key. A missing header skips the feature.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("context_compaction")

AFFINITY_HEADER = "x-hermes-conversation"
FOUNDRY_OPENAI_PROTOCOL = "openai_chat_completions"
COMPACT_TOOL = "compact_context"
LIST_TOOL = "list_context_items"
UNHIDE_TOOL = "unhide_context"
INTERNAL_TOOL_NAMES = (COMPACT_TOOL, LIST_TOOL, UNHIDE_TOOL)
MAX_AFFINITY_LENGTH = 256
DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MAX_ITEMS = 200
DEFAULT_MAX_INTERNAL_ROUNDS = 3
DEFAULT_MIN_CHARS = 800
HEAD_EXCERPTS = 5
TAIL_EXCERPTS = 3
MAX_EVIDENCE_EXCERPTS = 12
MAX_EXCERPT_LINE = 240

_ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"(?m)^(ERROR|Error|FAILED|FAIL):"),
    re.compile(r"(?m)^exit[_ ]code[:=]\s*[1-9]\d*\b"),
    re.compile(r"(?i)\bcommand failed\b"),
    re.compile(r"(?i)\bnon-zero exit\b"),
)
_STRONG_EVIDENCE_PATTERN = re.compile(
    r"("
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    r"|\b(?:[Ii][Dd]|[Uu][Uu][Ii][Dd]|[Ss][Hh][Aa]256)\s*[:=]"
    r"|\|"
    r")"
)
_PATH_EVIDENCE_PATTERN = re.compile(r"(?:^|[\s\"'`])(?:[\w.-]+/)+[\w.-]+\.[\w.-]+")

Generate = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class CompactionSettings:
    enabled: bool = False
    header_name: str = AFFINITY_HEADER
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    max_items: int = DEFAULT_MAX_ITEMS
    max_internal_rounds: int = DEFAULT_MAX_INTERNAL_ROUNDS
    min_chars: int = DEFAULT_MIN_CHARS
    laya_enabled: bool = False

    @property
    def laya_available(self) -> bool:
        return False

    @property
    def laya_active(self) -> bool:
        return self.laya_enabled and self.laya_available and False


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    tool_call_id: str
    content_sha256: str
    original: str
    compacted: str
    excerpt_lines: tuple[str, ...]
    visibility: str
    version: int
    expires_at: float
    tool_name: str | None = None


@dataclass(frozen=True)
class MutationResult:
    ok: bool
    error: str | None = None
    item: ContextItem | None = None


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


@dataclass(frozen=True)
class SpanChoice:
    lines: tuple[str, ...]
    source: str


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
        max_items=_env_int(source, "CONTEXT_COMPACTION_MAX_ITEMS", DEFAULT_MAX_ITEMS),
        max_internal_rounds=_env_int(
            source,
            "CONTEXT_COMPACTION_MAX_INTERNAL_ROUNDS",
            DEFAULT_MAX_INTERNAL_ROUNDS,
        ),
        min_chars=_env_int(source, "CONTEXT_COMPACTION_MIN_CHARS", DEFAULT_MIN_CHARS),
        laya_enabled=_env_flag(source, "CONTEXT_COMPACTION_LAYA_ENABLED"),
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
    if stream:
        return None, "streaming"
    if provider != "foundry" or (protocol or FOUNDRY_OPENAI_PROTOCOL) != FOUNDRY_OPENAI_PROTOCOL:
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
    return CompactionPlan(affinity, visibility_only, injected), "apply"


class MemoryContextStore:
    """Process-local originals keyed only by the affinity header value."""

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_items: int = DEFAULT_MAX_ITEMS,
        min_chars: int = DEFAULT_MIN_CHARS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self.min_chars = min_chars
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, ContextItem]] = {}
        self.last_measurement: TurnMeasurement | None = None

    def now(self) -> float:
        return self._clock()

    def record_measurement(self, measurement: TurnMeasurement) -> None:
        self.last_measurement = measurement

    def get(self, affinity: str, item_id: str, *, now: float | None = None) -> ContextItem | None:
        with self._lock:
            self._purge(affinity, self._now(now))
            item = self._items.get(affinity, {}).get(item_id)
            return None if item is None else _copy_item(item)

    def items(self, affinity: str, *, now: float | None = None) -> tuple[ContextItem, ...]:
        with self._lock:
            self._purge(affinity, self._now(now))
            return tuple(_copy_item(item) for item in self._items.get(affinity, {}).values())

    def compact(
        self,
        *,
        affinity: str,
        tool_call_id: str,
        original: str,
        tool_name: str | None,
        now: float | None = None,
    ) -> MutationResult:
        moment = self._now(now)
        with self._lock:
            self._purge(affinity, moment)
            bucket = self._items.setdefault(affinity, {})
            content_sha = _sha256(original)
            item_id = _item_id(affinity, tool_call_id, content_sha)
            existing = bucket.get(item_id)
            if existing is not None:
                if existing.visibility != "compacted":
                    existing = _replace(existing, visibility="compacted", version=existing.version + 1)
                    bucket[item_id] = existing
                return MutationResult(ok=True, item=_copy_item(existing))
            if len(bucket) >= self.max_items:
                return MutationResult(ok=False, error="store_full")
            refusal = _refusal_reason(original, self.min_chars)
            if refusal is not None:
                return MutationResult(ok=False, error=refusal)
            choice = RuleSpanSelector().select(original)
            if choice is None:
                return MutationResult(ok=False, error="verification_failed")
            rendered = render_compaction(item_id, choice.lines)
            if not _excerpts_are_exact(original, choice.lines) or len(rendered) >= len(original):
                return MutationResult(ok=False, error="verification_failed")
            item = ContextItem(
                item_id=item_id,
                tool_call_id=tool_call_id,
                content_sha256=content_sha,
                original=original,
                compacted=rendered,
                excerpt_lines=choice.lines,
                visibility="compacted",
                version=1 if existing is None else existing.version + 1,
                expires_at=moment + self.ttl_seconds,
                tool_name=tool_name,
            )
            bucket[item_id] = item
            return MutationResult(ok=True, item=_copy_item(item))

    def unhide(self, affinity: str, item_id: str, *, now: float | None = None) -> MutationResult:
        with self._lock:
            self._purge(affinity, self._now(now))
            item = self._items.get(affinity, {}).get(item_id)
            if item is None:
                return MutationResult(ok=False, error="not_found")
            if item.visibility != "original":
                item = _replace(item, visibility="original", version=item.version + 1)
                self._items[affinity][item_id] = item
            return MutationResult(ok=True, item=_copy_item(item))

    def visible_content(self, affinity: str, tool_call_id: str, content: str, *, now: float | None = None) -> str:
        with self._lock:
            self._purge(affinity, self._now(now))
            for item in self._items.get(affinity, {}).values():
                if item.tool_call_id != tool_call_id:
                    continue
                if content == item.original or content == item.compacted:
                    return item.compacted if item.visibility == "compacted" else item.original
            return content

    def _now(self, now: float | None) -> float:
        return self.now() if now is None else now

    def _purge(self, affinity: str, now: float) -> None:
        bucket = self._items.get(affinity)
        if not bucket:
            return
        expired = [item_id for item_id, item in bucket.items() if item.expires_at <= now]
        for item_id in expired:
            del bucket[item_id]
        if not bucket:
            self._items.pop(affinity, None)


class RuleSpanSelector:
    source = "rule"

    def select(self, original: str) -> SpanChoice | None:
        lines = original.splitlines()
        eligible = [index for index, line in enumerate(lines) if 0 < len(line) <= MAX_EXCERPT_LINE]
        if not eligible:
            return None
        selected: list[int] = []
        evidence = 0
        for index in eligible:
            if not _STRONG_EVIDENCE_PATTERN.search(lines[index]):
                continue
            selected.append(index)
            evidence += 1
            if evidence >= MAX_EVIDENCE_EXCERPTS:
                break
        for index in eligible[:HEAD_EXCERPTS]:
            if index not in selected:
                selected.append(index)
        for index in eligible[-TAIL_EXCERPTS:]:
            if index not in selected:
                selected.append(index)
        for index in eligible:
            if evidence >= MAX_EVIDENCE_EXCERPTS or index in selected:
                continue
            if _PATH_EVIDENCE_PATTERN.search(lines[index]):
                selected.append(index)
                evidence += 1
        ordered = tuple(lines[index] for index in sorted(selected))
        if not ordered or not _excerpts_are_exact(original, ordered):
            return None
        return SpanChoice(ordered, self.source)


def render_compaction(item_id: str, lines: tuple[str, ...] | list[str]) -> str:
    excerpts = "\n".join(f"원문 발췌: {line}" for line in lines)
    return (
        f"[compact:{item_id}]\n"
        f"{excerpts}\n"
        f"나머지 결과는 보관됨. 필요하면 unhide_context({item_id})."
    )


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
            COMPACT_TOOL,
            "Replace one already-read tool result body with a fixed exact excerpt. "
            "Pass tool_call_id. Does not rerun the tool or hide the original tool call.",
            {
                "type": "object",
                "properties": {
                    "tool_call_id": {"type": "string"},
                    "item_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        ),
        _function_tool(
            LIST_TOOL,
            "List compacted tool results for this conversation header. "
            "Returns ids and verified metadata only.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        _function_tool(
            UNHIDE_TOOL,
            "Restore one stored tool result body on later model calls. Does not rerun the tool.",
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


def measurement_from_usages(usages: list[dict[str, Any]], *, applied: bool, rounds: int, skipped: str | None) -> TurnMeasurement:
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
        laya_calls=0,
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
) -> TurnOutcome:
    hermes_messages = copy.deepcopy(messages)
    suffix: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    rounds = 0
    try:
        _ = apply_visibility(hermes_messages, affinity=plan.affinity_key, store=store)
    except Exception:
        logger.warning("context_compaction skipped reason=store_unavailable")
        return TurnOutcome(skipped=True, measurement=measurement_from_usages([], applied=False, rounds=0, skipped="store_unavailable"))

    while True:
        try:
            outbound = apply_visibility(hermes_messages, affinity=plan.affinity_key, store=store)
        except Exception:
            logger.warning("context_compaction skipped reason=store_unavailable")
            if usages:
                measurement = measurement_from_usages(usages, applied=True, rounds=rounds, skipped="store_unavailable")
                store.record_measurement(measurement)
                return TurnOutcome(
                    skipped=False,
                    error=(502, "Context compaction storage failed before the response was completed.", "context_compaction_store_unavailable"),
                    measurement=measurement,
                    prior_usages=usages,
                )
            return TurnOutcome(skipped=True)
        outbound.extend(copy.deepcopy(suffix))
        kwargs = dict(base_kwargs)
        kwargs["messages"] = outbound
        if plan.tools is not None:
            kwargs["tools"] = copy.deepcopy(plan.tools)
        try:
            result = await generate(**kwargs)
        except Exception as exc:
            if usages:
                measurement = measurement_from_usages(usages, applied=True, rounds=rounds, skipped="upstream_error")
                store.record_measurement(measurement)
                raise CompactionUpstreamError(usages) from exc
            raise
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        usages.append(usage)
        tool_calls = result.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []
        internal = [call for call in tool_calls if _call_name(call) in INTERNAL_TOOL_NAMES]
        external = [call for call in tool_calls if _call_name(call) not in INTERNAL_TOOL_NAMES]
        if internal and external:
            measurement = measurement_from_usages(usages, applied=True, rounds=rounds, skipped=None)
            store.record_measurement(measurement)
            logger.info("context_compaction calls=%s rounds=%s mixed=1", len(usages), rounds)
            return TurnOutcome(
                skipped=False,
                result=_public_result(result, external, usage=aggregate_usage(usages)),
                measurement=measurement,
            )
        if not internal:
            measurement = measurement_from_usages(usages, applied=True, rounds=rounds, skipped=None)
            store.record_measurement(measurement)
            logger.info("context_compaction calls=%s rounds=%s", len(usages), rounds)
            return TurnOutcome(
                skipped=False,
                result=_public_result(result, external, usage=aggregate_usage(usages)),
                measurement=measurement,
            )
        rounds += 1
        if rounds > settings.max_internal_rounds:
            measurement = measurement_from_usages(usages, applied=True, rounds=rounds, skipped="loop_limit")
            store.record_measurement(measurement)
            return TurnOutcome(
                skipped=False,
                error=(
                    502,
                    "Context compaction stopped because the internal tool loop exceeded its limit.",
                    "context_compaction_loop_limit",
                ),
                measurement=measurement,
                prior_usages=usages,
            )
        suffix.append(_assistant_message(result, internal))
        for call in internal:
            suffix.append(_tool_message(call, _execute_internal(call, hermes_messages, plan, store)))

    raise AssertionError("unreachable")


class CompactionUpstreamError(Exception):
    def __init__(self, usages: list[dict[str, Any]]) -> None:
        super().__init__("context compaction upstream failed")
        self.usages = usages


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


def _execute_internal(
    call: dict[str, Any],
    hermes_messages: list[dict[str, Any]],
    plan: CompactionPlan,
    store: MemoryContextStore,
) -> dict[str, Any]:
    name = _call_name(call)
    try:
        args = _call_arguments(call)
    except ValueError:
        return {"ok": False, "error": "invalid_arguments"}
    try:
        if name == COMPACT_TOOL:
            return _compact_call(args, hermes_messages, plan.affinity_key, store)
        if name == LIST_TOOL:
            return _list_call(args, plan.affinity_key, store)
        if name == UNHIDE_TOOL:
            return _unhide_call(args, plan.affinity_key, store)
    except Exception:
        logger.warning("context_compaction tool_failed name=%s", name)
        return {"ok": False, "error": "store_unavailable"}
    return {"ok": False, "error": "unknown_tool"}


def _compact_call(
    args: Mapping[str, Any],
    messages: list[dict[str, Any]],
    affinity: str,
    store: MemoryContextStore,
) -> dict[str, Any]:
    tool_call_id = args.get("tool_call_id")
    item_id = args.get("item_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        if isinstance(item_id, str) and item_id:
            existing = store.get(affinity, item_id)
            if existing is None:
                return {"ok": False, "error": "not_found"}
            tool_call_id = existing.tool_call_id
        else:
            return {"ok": False, "error": "missing_tool_call_id"}
    matches = _tool_messages(messages, tool_call_id)
    if not matches:
        existing = store.get(affinity, str(item_id)) if isinstance(item_id, str) else None
        if existing is not None and existing.tool_call_id == tool_call_id:
            return _mutation_payload(MutationResult(ok=True, item=existing))
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
    )
    return _mutation_payload(result)


def _list_call(args: Mapping[str, Any], affinity: str, store: MemoryContextStore) -> dict[str, Any]:
    query = args.get("query") if isinstance(args.get("query"), str) else ""
    ranked = rank_compacted_items(query, store.items(affinity))
    return {
        "ok": True,
        "items": [
            {
                "item_id": item.item_id,
                "tool_call_id": item.tool_call_id,
                "tool_name": item.tool_name,
                "original_available": True,
                "visibility": item.visibility,
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
        "visibility": result.item.visibility,
        "original_available": True,
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
    return {
        "role": "assistant",
        "content": text if text else None,
        "tool_calls": copy.deepcopy(tool_calls),
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
        raise ValueError("invalid arguments")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("invalid arguments")
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
    if len(original) < min_chars:
        return "not_long"
    if any(pattern.search(original) for pattern in _ERROR_PATTERNS):
        return "protected_error"
    return None


def _excerpts_are_exact(original: str, lines: tuple[str, ...] | list[str]) -> bool:
    original_lines = original.splitlines()
    return all(line in original_lines and line in original for line in lines)


def _item_id(affinity: str, tool_call_id: str, content_sha: str) -> str:
    digest = hashlib.sha256(f"{affinity}\0{tool_call_id}\0{content_sha}".encode()).hexdigest()[:16]
    return f"item_{digest}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _copy_item(item: ContextItem) -> ContextItem:
    return ContextItem(**item.__dict__)


def _replace(item: ContextItem, **changes: Any) -> ContextItem:
    data = item.__dict__.copy()
    data.update(changes)
    return ContextItem(**data)


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
