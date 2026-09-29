"""Bounded, advisory Laya choices for M2 excerpts and M3 list ordering.

This module cannot change required evidence or restore items. The rules remain the
fallback and all chosen excerpts are exact lines of the original.
"""

from __future__ import annotations

from collections.abc import Callable

from openai_compatible_bridge.laya_http import LayaClient

MAX_CANDIDATES = 16
CHUNK_SIZE = 8
MAX_GOAL_CHARS = 300
MAX_QUERY_CHARS = 300
MAX_RANK_ITEMS = 8


async def select_extra_lines(
    original: str,
    goal: str,
    baseline: tuple[str, ...],
    client: LayaClient,
    on_call: Callable[[], None],
) -> tuple[str, ...] | None:
    """Select at most one extra line per chunk; None means use the rule baseline."""
    if not goal.strip():
        return None
    lines = original.splitlines()
    candidates = [
        (index, line)
        for index, line in enumerate(lines)
        if line and len(line) <= 180 and line not in baseline
    ]
    # Do not send a partial long document to the selector as though it were complete.
    if not candidates or len(candidates) > MAX_CANDIDATES:
        return None
    selected: list[str] = []
    for start in range(0, len(candidates), CHUNK_SIZE):
        chunk = candidates[start:start + CHUNK_SIZE]
        original_by_key = {f"line_{index}": line for index, line in chunk}
        # Head text is only a decision hint. The final excerpt comes from the
        # original_by_key full line, never from this truncated model input.
        criteria = {key: line[:120] for key, line in original_by_key.items()}
        criteria["skip"] = "No additional line is relevant"
        question = {"relevance": {
            "type": "choice",
            "instructions": "Select one exact line useful for the current task, or skip. Do not summarize.",
            "criteria": criteria,
        }}
        try:
            on_call()
            choice = await client.choose("Current task: " + goal[:MAX_GOAL_CHARS], question, "relevance")
        except Exception:  # noqa: BLE001 -- Laya failure must preserve the rule baseline
            return None
        if choice != "skip":
            if choice not in original_by_key or original_by_key[choice] not in lines:
                return None
            selected.append(original_by_key[choice])
    if not selected:
        return None
    wanted = set(baseline) | set(selected)
    return tuple(line for line in lines if line in wanted)


async def rank_items(
    query: str,
    ranked: list,
    client: LayaClient,
    on_call: Callable[[], None],
) -> list:
    """Reorder at most one chosen candidate; never drop items or auto-unhide."""
    if not query.strip() or len(ranked) < 2 or len(ranked) > MAX_RANK_ITEMS:
        return ranked
    criteria = {
        item.item_id: f"{item.tool_name or ''}: {item.compacted[:120]}"
        for item in ranked
    }
    criteria["skip"] = "No relevant item"
    question = {"relevance": {
        "type": "choice",
        "instructions": "Choose the most relevant hidden tool-result id for the query, or skip.",
        "criteria": criteria,
    }}
    try:
        on_call()
        choice = await client.choose("Current query: " + query[:MAX_QUERY_CHARS], question, "relevance")
    except Exception:  # noqa: BLE001 -- preserve the full rule-ranked list
        return ranked
    if choice == "skip" or choice not in {item.item_id for item in ranked}:
        return ranked
    return sorted(ranked, key=lambda item: item.item_id != choice)
