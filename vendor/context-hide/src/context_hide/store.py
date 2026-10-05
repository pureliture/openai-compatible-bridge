"""Thread-safe, capacity-bounded memory context store with TTL expiration."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from cachetools import TLRUCache

from context_hide.model import (
    ContextItem,
    MutationResult,
    Scope,
    SpanChoice,
    canonical,
    compute_invocation_digest,
    compute_sha256,
)
from context_hide.policy import (
    RuleSpanSelector,
    check_refusal_reason,
    excerpts_are_exact,
    render_compaction,
    required_evidence_indexes,
    verified_facts,
)
from context_hide.summary import validate_summary_text

DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 hours
DEFAULT_MAX_BYTES = 64 * 1024 * 1024  # 64 MB
DEFAULT_MIN_CHARS = 800


def _scope_key(scope: Scope | str) -> str:
    """Extract string key from Scope or raw string identifier."""
    if isinstance(scope, Scope):
        return scope.scope_key()
    return str(scope)


def _item_id(scope_key: str, tool_call_id: str, content_sha: str) -> str:
    digest = hashlib.sha256(f"{scope_key}\0{tool_call_id}\0{content_sha}".encode()).hexdigest()[:16]
    return f"item_{digest}"


class MemoryContextStore:
    """Thread-safe process-local context store backed by cachetools.TLRUCache.

    Invariants:
    1. Keyed by (scope_key, item_id).
    2. Expiration via TLRUCache TTU based on item.expires_at.
    3. Admission check before insertion: guarantees LRU never evicts live originals.
    4. Size accounting: 2048 bytes metadata allowance + UTF-8 payload lengths.
    5. Max 4 concurrent pending reservations per process.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        min_chars: int = DEFAULT_MIN_CHARS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0 or max_bytes <= 0:
            raise ValueError("ttl_seconds and max_bytes must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self.min_chars = min_chars
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._items = TLRUCache(
            maxsize=max_bytes,
            ttu=lambda _key, item, _now: item.expires_at,
            timer=self._clock,
            getsizeof=self._item_size,
        )
        self._pending: dict[tuple[str, str], tuple[object, str]] = {}

    @staticmethod
    def _item_size(item: ContextItem) -> int:
        texts = (
            item.original,
            item.compacted,
            *item.excerpt_lines,
            item.item_id,
            item.tool_call_id,
            item.content_sha256,
            item.tool_name or "",
            item.compaction_source,
            item.invocation_digest or "",
            canonical(item.invocation_snapshot or {}),
            canonical(item.context_hint or {}),
        )
        return 2048 + sum(len(text.encode("utf-8")) for text in texts)

    def now(self) -> float:
        return self._clock()

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return int(self._items.currsize)

    @property
    def remaining_bytes(self) -> int:
        with self._lock:
            return int(self._items.maxsize - self._items.currsize)

    def expire(self, *, now: float | None = None) -> int:
        """Release expired items across all scopes."""
        with self._lock:
            moment = self.now() if now is None else now
            return len(list(self._items.expire(moment)))

    def get(self, scope: Scope | str, item_id: str, *, now: float | None = None) -> ContextItem | None:
        with self._lock:
            moment = self.now() if now is None else now
            self._items.expire(moment)
            return self._items.get((_scope_key(scope), item_id))

    def items(self, scope: Scope | str, *, now: float | None = None) -> tuple[ContextItem, ...]:
        with self._lock:
            moment = self.now() if now is None else now
            self._items.expire(moment)
            target_key = _scope_key(scope)
            return tuple(
                item for (item_scope, _), item in self._items.items()
                if item_scope == target_key
            )

    def reserve(
        self,
        scope: Scope | str,
        item_id: str,
        digest: str | None = None,
    ) -> tuple[object | None, str | None]:
        with self._lock:
            key = (_scope_key(scope), item_id)
            self._items.expire(self.now())
            if key in self._items:
                return None, "already_exists"
            if key in self._pending:
                return None, "in_progress"
            if len(self._pending) >= 4:
                return None, "pending_full"
            token = object()
            self._pending[key] = (token, digest or "")
            return token, None

    def release(self, scope: Scope | str, item_id: str, token: object) -> None:
        with self._lock:
            key = (_scope_key(scope), item_id)
            if key in self._pending and self._pending[key][0] is token:
                del self._pending[key]

    def put(
        self,
        scope: Scope | str,
        item: ContextItem,
        *,
        reservation: object | None = None,
    ) -> MutationResult:
        with self._lock:
            self._items.expire(self.now())
            key = (_scope_key(scope), item.item_id)
            if reservation is not None:
                pending_entry = self._pending.get(key)
                if pending_entry is None or pending_entry[0] is not reservation:
                    return MutationResult(ok=False, error="reservation_invalid")
            item_bytes = self._item_size(item)
            if item_bytes > self._items.maxsize - self._items.currsize:
                return MutationResult(ok=False, error="store_full")
            self._items[key] = item
            return MutationResult(ok=True, item=item)

    def unhide(self, scope: Scope | str, item_id: str, *, now: float | None = None) -> MutationResult:
        with self._lock:
            moment = self.now() if now is None else now
            self._items.expire(moment)
            key = (_scope_key(scope), item_id)
            item = self._items.get(key)
            if item is None:
                return MutationResult(ok=False, error="not_found")
            if item.visibility != "original":
                item = replace(item, visibility="original", version=item.version + 1)
                self._items[key] = item
            return MutationResult(ok=True, item=item)

    def visible_content(
        self,
        scope: Scope | str,
        tool_call_id: str,
        content: str,
        *,
        now: float | None = None,
    ) -> str:
        for item in self.items(scope, now=now):
            if item.tool_call_id == tool_call_id and content in (item.original, item.compacted):
                return item.compacted if item.visibility == "compacted" else item.original
        return content

    def compact(
        self,
        *,
        affinity: Scope | str,
        tool_call_id: str,
        original: str,
        tool_name: str | None = None,
        now: float | None = None,
        choice: SpanChoice | None = None,
        invocation: dict[str, Any] | None = None,
        context: dict[str, str] | None = None,
        reservation: object | None = None,
        input_digest: str | None = None,
        options_omitted: bool = False,
    ) -> MutationResult:
        """Compat compact method replicating bridge behavior for test migration."""
        with self._lock:
            moment = self.now() if now is None else now
            self._items.expire(moment)
            content_sha = compute_sha256(original)
            target_key = _scope_key(affinity)
            item_id = _item_id(target_key, tool_call_id, content_sha)
            key = (target_key, item_id)

            digest = input_digest or (compute_invocation_digest(invocation) if invocation is not None else None)
            if reservation is not None and self._pending.get(key) != (reservation, digest or ""):
                return MutationResult(ok=False, error="reservation_invalid")

            existing = self._items.get(key)
            if existing is not None:
                if existing.invocation_digest is not None and existing.invocation_digest != digest:
                    return MutationResult(ok=False, error="invocation_conflict")
                if existing.visibility != "compacted":
                    existing = replace(existing, visibility="compacted", version=existing.version + 1)
                    self._items[key] = existing
                return MutationResult(ok=True, item=existing)

            refusal = check_refusal_reason(original, self.min_chars)
            if refusal is not None:
                return MutationResult(ok=False, error=refusal)

            if len(original.encode("utf-8")) > self._items.maxsize - self._items.currsize:
                return MutationResult(ok=False, error="store_full")

            baseline = RuleSpanSelector().select(original)
            if baseline is None:
                return MutationResult(ok=False, error="verification_failed")

            if choice is not None and choice.source == "lfm":
                original_lines = original.splitlines()
                required = tuple(original_lines[index] for index in required_evidence_indexes(original_lines))
                summary = validate_summary_text(original, choice.summary_text, required, invocation=invocation)
                if (invocation is None or choice.summary_text is None or not set(required).issubset(choice.lines)
                        or not excerpts_are_exact(original, choice.lines) or summary is None):
                    choice = baseline
                else:
                    choice = SpanChoice(required, "lfm", summary)
            elif (choice is None or not set(baseline.lines).issubset(choice.lines)
                    or not excerpts_are_exact(original, choice.lines)):
                choice = baseline

            rendered = render_compaction(
                item_id,
                choice.lines,
                summary_text=choice.summary_text,
                context=context,
                facts=verified_facts(original),
                options_omitted=options_omitted,
            )

            if (choice.source == "lfm"
                    and len(rendered.encode("utf-8")) > len(original.encode("utf-8")) * 0.8):
                choice = baseline
                rendered = render_compaction(item_id, choice.lines)

            if len(rendered) >= len(original) and choice is not baseline:
                choice = baseline
                rendered = render_compaction(item_id, choice.lines)

            original_lines = original.splitlines()
            required_idx = required_evidence_indexes(original_lines)
            if (not excerpts_are_exact(original, choice.lines)
                    or any(original_lines[index] not in choice.lines for index in required_idx)
                    or len(rendered) >= len(original)):
                return MutationResult(ok=False, error="verification_failed")

            item = ContextItem(
                item_id=item_id,
                tool_call_id=tool_call_id,
                content_sha256=content_sha,
                original=original,
                compacted=rendered,
                excerpt_lines=choice.lines,
                visibility="compacted",
                version=1,
                expires_at=moment + self.ttl_seconds,
                tool_name=tool_name,
                compaction_source=choice.source,
                invocation_snapshot=copy.deepcopy(invocation) if choice.source == "lfm" else None,
                invocation_digest=digest if choice.source == "lfm" else None,
                context_hint=None,
                invocation_options_omitted=options_omitted if choice.source == "lfm" else False,
            )

            if self._item_size(item) > self._items.maxsize - self._items.currsize:
                return MutationResult(ok=False, error="store_full")

            self._items[key] = item
            return MutationResult(ok=True, item=item)
