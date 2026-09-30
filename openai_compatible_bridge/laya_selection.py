"""Bounded, advisory Laya choices for M2 excerpts and M3 list ordering.

This module cannot change required evidence or restore items. The rules remain the
fallback and all chosen excerpts are exact lines of the original.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from openai_compatible_bridge.laya_http import LayaClient

MAX_GOAL_CHARS = 300
MAX_QUERY_CHARS = 300
MAX_RANK_ITEMS = 8
_GOAL_STOPWORDS = {
    "the", "for", "find", "locate", "where", "entry", "point", "in", "is", "are", "and",
    "을", "를", "의", "위치", "찾기", "확인해", "줘", "기본",
}


def _related_to_goal(line: str, goal: str) -> bool:
    """Only propose lines with an observable shared task term; not a safety classifier."""
    words = [word.lower() for word in re.findall(r"[\w가-힣]+", goal)
             if len(word) >= 3 and word.lower() not in _GOAL_STOPWORDS]
    text = line.lower()
    return bool(words) and any(word in text for word in words)


async def select_extra_lines(
    original: str,
    goal: str,
    baseline: tuple[str, ...],
    client: LayaClient,
    on_call: Callable[[], None],
) -> tuple[str, ...] | None:
    """Choose one directly related line, or refuse compaction when uncertain."""
    if not goal.strip():
        return None
    lines = original.splitlines()
    # A baseline excerpt may already contain another answer to the same goal.
    # Choosing only the remaining line would hide that competing evidence.
    if any(_related_to_goal(line, goal) for line in baseline):
        return None
    if any(_related_to_goal(line, goal) and len(line) > 180 for line in lines):
        return None
    candidates = [
        (index, line)
        for index, line in enumerate(lines)
        if line and len(line) <= 180 and line not in baseline and _related_to_goal(line, goal)
    ]
    # One candidate only: never invite a choice from an incomplete set of
    # potentially competing answers.
    if len(candidates) != 1:
        return None
    index, line = candidates[0]
    key = f"line_{index}"
    question = {"relevance": {
        "type": "choice",
        "instructions": "Choose the line that directly answers the current task. If none answers it, choose skip. Do not summarize or invent facts.",
        "criteria": {key: line[:120], "skip": "No additional line is relevant"},
    }}
    try:
        on_call()
        choice = await client.choose("Current task: " + goal[:MAX_GOAL_CHARS], question, "relevance")
    except Exception:  # noqa: BLE001 -- failed decision must retain the original
        return None
    if choice != key or line not in lines:
        return None
    wanted = set(baseline) | {line}
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
