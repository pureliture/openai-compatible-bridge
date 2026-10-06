"""Host-neutral data models and contracts for context-hide engine."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def canonical(value: Any) -> str:
    """Produce deterministically sorted, compact JSON encoding without extra whitespace."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_sha256(value: str) -> str:
    """Compute standard hexadecimal SHA-256 digest of UTF-8 encoded text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compute_invocation_digest(
    invocation: dict[str, Any],
    omitted_options: dict[str, Any] | None = None,
) -> str:
    """Compute deterministically ordered canonical SHA-256 digest for tool invocation."""
    identity = dict(invocation)
    if omitted_options:
        identity["omitted_options"] = omitted_options
    return hashlib.sha256(canonical(identity).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Scope:
    """Namespace scope identifying an isolated session partition."""
    adapter_id: str
    host_profile: str
    session_id: str
    branch_scope: str = "default"

    def scope_key(self) -> str:
        """Produce canonical string representation of scope identity."""
        return f"{self.adapter_id}:{self.host_profile}:{self.session_id}:{self.branch_scope}"


@dataclass(frozen=True)
class ToolResultRecord:
    """Input record provided by host adapter representing an executed tool result."""
    call_id: str
    result_position: int = 0
    content: str = ""
    content_sha256: str = ""
    invocation: dict[str, Any] = field(default_factory=dict)
    invocation_digest: str = ""
    status: str = "ok"
    host_handle: Any = None

    def verify_sha256(self) -> bool:
        """Verify that content matches content_sha256."""
        return compute_sha256(self.content) == self.content_sha256


@dataclass(frozen=True)
class ReplacementPlan:
    """Instruction emitted by engine directing adapter to replace result content."""
    item_id: str
    expected_content_sha256: str
    replacement_text: str
    visibility_version: int
    host_handle: Any = None


@dataclass(frozen=True)
class ContextItem:
    """Stored representation of a tool result and its compaction state."""
    item_id: str
    tool_call_id: str
    content_sha256: str
    original: str
    compacted: str
    excerpt_lines: tuple[str, ...]
    visibility: str  # "compacted" | "original"
    version: int
    expires_at: float
    tool_name: str | None = None
    compaction_source: str = "rule"
    invocation_snapshot: dict[str, Any] | None = None
    invocation_digest: str | None = None
    context_hint: dict[str, str] | None = None
    invocation_options_omitted: bool = False

    @property
    def call_id(self) -> str:
        return self.tool_call_id

    @property
    def source(self) -> str:
        return self.compaction_source

    @property
    def bytes_size(self) -> int:
        texts = (
            self.original,
            self.compacted,
            *self.excerpt_lines,
            self.item_id,
            self.tool_call_id,
            self.content_sha256,
            self.tool_name or "",
            self.compaction_source,
            self.invocation_digest or "",
            canonical(self.invocation_snapshot or {}),
            canonical(self.context_hint or {}),
        )
        return 2048 + sum(len(text.encode("utf-8")) for text in texts)


@dataclass(frozen=True)
class ItemMetadata:
    """Read-only metadata descriptor for a stored context item."""
    item_id: str
    call_id: str
    content_sha256: str
    visibility: str  # "compacted" | "original"
    version: int
    expires_at: float
    source: str = "rule"  # "rule" | "summarizer" | "lfm"
    bytes_size: int = 0
    tool_name: str | None = None
    excerpt_lines: tuple[str, ...] = ()
    summary: str = ""

    @property
    def tool_call_id(self) -> str:
        return self.call_id

    @property
    def compaction_source(self) -> str:
        return self.source

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class MutationResult:
    """Result of hide/unhide mutation on engine store."""
    ok: bool
    error: str | None = None
    plan: ReplacementPlan | None = None
    item: ContextItem | None = None

    @property
    def replacement(self) -> ReplacementPlan | None:
        """Alias for plan to support bridge adapter conventions."""
        return self.plan


@dataclass(frozen=True)
class SpanChoice:
    """Selected excerpt lines and optional summary payload."""
    lines: tuple[str, ...]
    source: str
    summary_text: dict[str, Any] | None = None


@dataclass(frozen=True)
class EngineConfig:
    """Engine operating parameters and safety limits."""
    ttl_seconds: int = 86400
    max_bytes: int = 67108864  # 64 MB
    min_chars: int = 800
    max_pending: int = 4
    max_evidence_excerpts: int = 12
    max_excerpt_line_length: int = 240
    max_summary_length: int = 1600
    max_invocation_bytes: int = 2048
    lfm_max_input_bytes: int = 12288
