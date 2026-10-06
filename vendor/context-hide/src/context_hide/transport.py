"""Transport protocols and pluggable summarizer interfaces."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

OnCall = Callable[[], None]


class SummarizerError(Exception):
    """Raised when an external summarizer encounters a failure or validation error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LFMUnavailable(SummarizerError):
    """Compatibility alias for summarizer unavailability or validation rejection."""
    pass


@runtime_checkable
class Summarizer(Protocol):
    """Protocol for pluggable summary models (LFM, local LLM, or external APIs)."""

    async def summarize(
        self,
        original: str,
        required_evidence: tuple[str, ...],
        on_call: OnCall | None = None,
        *,
        invocation: dict[str, Any] | None = None,
        context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Request and return a validated summary dictionary containing {'summary': str}.

        Args:
            original: Full raw output of the tool result.
            required_evidence: Mandatory lines that must be retained or accurately reflected.
            on_call: Measurement hook invoked immediately before sending external request.
            invocation: Serialized tool call definition and arguments.
            context: Optional non-binding caller hints.

        Returns:
            dict containing at least {"summary": str}.

        Raises:
            SummarizerError: On network failure, schema rejection, or unverified facts.
        """
        ...
