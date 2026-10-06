"""Context hide common engine orchestrating store, policy, and summarization."""
from __future__ import annotations

import asyncio
import copy
from typing import Any

from context_hide.model import (
    ContextItem,
    EngineConfig,
    ItemMetadata,
    MutationResult,
    ReplacementPlan,
    Scope,
    SpanChoice,
    ToolResultRecord,
)
from context_hide.policy import (
    RuleSpanSelector,
    excerpts_are_exact,
    refusal_reason,
    render_compaction,
    required_evidence_lines,
    verified_facts,
)
from context_hide.store import MemoryContextStore, _item_id
from context_hide.summary import validate_summary_text
from context_hide.transport import Summarizer


class ContextHideEngine:
    """Host-neutral tool result summarization and context compaction engine."""

    def __init__(
        self,
        store: MemoryContextStore | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.store = store or MemoryContextStore(
            ttl_seconds=self.config.ttl_seconds,
            max_bytes=self.config.max_bytes,
            min_chars=self.config.min_chars,
        )

    async def hide(
        self,
        scope: Scope,
        record: ToolResultRecord,
        summarizer: Summarizer | None = None,
        *,
        options_omitted: bool = False,
    ) -> MutationResult:
        """Hide tool result content into store with optional summarizer and rule fallback.

        Guarantees: Fail-Open on any failure. Never crashes host loop.
        """
        # 1. Policy & input validation
        refusal = refusal_reason(record, self.config)
        if refusal is not None:
            return MutationResult(ok=False, error=refusal)

        item_id = _item_id(scope.scope_key(), record.call_id, record.content_sha256)

        # 2. Re-compaction / Cache check (Re-summary exclusion)
        existing = self.store.get(scope, item_id)
        if existing is not None:
            if existing.invocation_digest is not None and existing.invocation_digest != record.invocation_digest:
                return MutationResult(ok=False, error="invocation_conflict")
            if existing.visibility != "compacted":
                from dataclasses import replace
                updated = replace(existing, visibility="compacted", version=existing.version + 1)
                put_res = self.store.put(scope, updated)
                if not put_res.ok:
                    return put_res
                plan = ReplacementPlan(
                    item_id=updated.item_id,
                    expected_content_sha256=record.content_sha256,
                    replacement_text=updated.compacted,
                    visibility_version=updated.version,
                    host_handle=record.host_handle,
                )
                return MutationResult(ok=True, plan=plan, item=updated)
            plan = ReplacementPlan(
                item_id=existing.item_id,
                expected_content_sha256=record.content_sha256,
                replacement_text=existing.compacted,
                visibility_version=existing.version,
                host_handle=record.host_handle,
            )
            return MutationResult(ok=True, plan=plan, item=existing)

        # 3. Storage budget check
        if len(record.content.encode("utf-8")) > self.store.remaining_bytes:
            return MutationResult(ok=False, error="store_full")

        # 4. Baseline Rule Excerpt check
        baseline = RuleSpanSelector().select(record.content)
        if baseline is None:
            return MutationResult(ok=False, error="verification_failed")

        # 5. Pending reservation
        token, reserve_err = self.store.reserve(scope, item_id, record.invocation_digest)
        if reserve_err is not None:
            return MutationResult(ok=False, error=reserve_err)

        try:
            required = required_evidence_lines(record.content)
            choice: SpanChoice | None = None

            # 6. Summarization attempt with rule fallback
            if summarizer is not None:
                try:
                    summary_dict = await summarizer.summarize(
                        record.content,
                        required,
                        invocation=record.invocation,
                    )
                    choice = SpanChoice(required, "lfm", summary_dict)
                except Exception:
                    choice = None

            # Cancellation check
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError

            # 7. Rule Verification & Choice Resolution
            if choice is not None and choice.source == "lfm":
                summary = validate_summary_text(
                    record.content, choice.summary_text, required, invocation=record.invocation,
                )
                if (choice.summary_text is None
                        or not set(required).issubset(choice.lines)
                        or not excerpts_are_exact(record.content, choice.lines)
                        or summary is None):
                    choice = baseline
                else:
                    choice = SpanChoice(required, "lfm", summary)
            else:
                choice = baseline

            # 8. Compaction rendering & compression bounds
            rendered = render_compaction(
                item_id,
                choice.lines,
                summary_text=choice.summary_text,
                facts=verified_facts(record.content),
                options_omitted=options_omitted,
            )
            if choice.source == "lfm" and len(rendered.encode("utf-8")) > len(record.content.encode("utf-8")) * 0.8:
                choice = baseline
                rendered = render_compaction(item_id, choice.lines)
            if len(rendered) >= len(record.content) and choice is not baseline:
                choice = baseline
                rendered = render_compaction(item_id, choice.lines)

            # Final Fail-Open check
            if (not excerpts_are_exact(record.content, choice.lines)
                    or any(line not in choice.lines for line in required)
                    or len(rendered) >= len(record.content)):
                return MutationResult(ok=False, error="verification_failed")

            # 9. Item construction & Store commit
            item = ContextItem(
                item_id=item_id,
                tool_call_id=record.call_id,
                content_sha256=record.content_sha256,
                original=record.content,
                compacted=rendered,
                excerpt_lines=choice.lines,
                visibility="compacted",
                version=1,
                expires_at=self.store.now() + self.store.ttl_seconds,
                tool_name=record.invocation.get("tool_name"),
                compaction_source=choice.source,
                invocation_snapshot=copy.deepcopy(record.invocation) if choice.source == "lfm" else None,
                invocation_digest=record.invocation_digest if choice.source == "lfm" else None,
                context_hint=None,
                invocation_options_omitted=options_omitted if choice.source == "lfm" else False,
            )
            put_res = self.store.put(scope, item, reservation=token)
            if not put_res.ok:
                return put_res

            plan = ReplacementPlan(
                item_id=item.item_id,
                expected_content_sha256=record.content_sha256,
                replacement_text=rendered,
                visibility_version=item.version,
                host_handle=record.host_handle,
            )
            return MutationResult(ok=True, plan=plan, item=item)

        finally:
            self.store.release(scope, item_id, token)

    def hide_sync(
        self,
        scope: Scope,
        record: ToolResultRecord,
        *,
        options_omitted: bool = False,
    ) -> MutationResult:
        """Synchronous hide method using conservative RuleSpanSelector."""
        return asyncio.run(self.hide(scope, record, summarizer=None, options_omitted=options_omitted))

    def list_items(self, scope: Scope, query: str = "") -> list[ItemMetadata]:
        """List metadata of context items belonging to scope."""
        raw_items = self.store.items(scope)
        results: list[ItemMetadata] = []
        for item in raw_items:
            if query and query.lower() not in item.compacted.lower() and query.lower() not in item.item_id.lower():
                continue
            results.append(
                ItemMetadata(
                    item_id=item.item_id,
                    call_id=item.tool_call_id,
                    content_sha256=item.content_sha256,
                    visibility=item.visibility,
                    version=item.version,
                    expires_at=item.expires_at,
                    source=item.compaction_source,
                    bytes_size=item.bytes_size,
                    tool_name=item.tool_name,
                    excerpt_lines=item.excerpt_lines,
                    summary=item.compacted,
                )
            )
        return results

    def unhide(self, scope: Scope, item_id: str) -> MutationResult:
        """Unhide item, marking visibility as original without re-running tools."""
        return self.store.unhide(scope, item_id)

    def project(self, scope: Scope, records: list[ToolResultRecord]) -> list[ReplacementPlan]:
        """Generate replacement plans for host transcript mutation.

        Unhidden items (visibility == 'original') or absent items yield no replacement plan,
        leaving host transcript content untouched.
        """
        plans: list[ReplacementPlan] = []
        for record in records:
            item_id = _item_id(scope.scope_key(), record.call_id, record.content_sha256)
            item = self.store.get(scope, item_id)
            if item is not None and item.visibility == "compacted":
                plans.append(
                    ReplacementPlan(
                        item_id=item.item_id,
                        expected_content_sha256=record.content_sha256,
                        replacement_text=item.compacted,
                        visibility_version=item.version,
                        host_handle=record.host_handle,
                    )
                )
        return plans
