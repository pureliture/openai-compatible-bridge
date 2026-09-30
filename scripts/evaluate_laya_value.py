"""Paired synthetic M2/M3 value evaluation; only sanitized metrics are printed.

Default: scripted negative control. Set LAYA_BASE_URL in the process environment
for actual Laya calls; never pass real transcripts, prompts or credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, cast

from openai_compatible_bridge.context_compaction import (
    ContextItem,
    MemoryContextStore,
    RuleSpanSelector,
    SpanChoice,
    rank_compacted_items,
)
from openai_compatible_bridge.laya_http import LayaClient
from openai_compatible_bridge.laya_selection import rank_items, select_extra_lines


@dataclass(frozen=True)
class M2Case:
    name: str
    text: str
    goal: str
    required: tuple[str, ...]


def _output(middle: str) -> str:
    rows = [f"sample component row {i:02d} alpha beta gamma " + "x" * 50 for i in range(18)]
    rows[10] = middle
    return "\n".join(rows)


# Non-sensitive, low-risk technical output; ground truth is fixed before any choice.
M2_CASES = (
    M2Case("file_ko", _output("widget 렌더링 정의는 widget_panel 모듈에 있습니다."),
           "widget 렌더링 정의 위치 찾기", ("widget 렌더링 정의는 widget_panel 모듈에 있습니다.",)),
    M2Case("file_en", _output("widget rendering entry is located in widget_panel module."),
           "Find the widget rendering entry", ("widget rendering entry is located in widget_panel module.",)),
    M2Case("module_ko", _output("파서의 기본 진입점은 parse_catalog 함수입니다."),
           "파서의 진입점 찾기", ("파서의 기본 진입점은 parse_catalog 함수입니다.",)),
    M2Case("module_en", _output("The parser entry point is parse_catalog in catalog_parser."),
           "Locate the parser entry point", ("The parser entry point is parse_catalog in catalog_parser.",)),
    M2Case("renderer_new", _output("The dashboard renderer lives in display_core."),
           "Find the dashboard renderer", ("The dashboard renderer lives in display_core.",)),
    M2Case("graph_new", _output("그래프 그리기 함수는 draw_edges 입니다."),
           "그래프 그리기 함수 찾기", ("그래프 그리기 함수는 draw_edges 입니다.",)),
)


@dataclass(frozen=True)
class M3Case:
    name: str
    query: str
    items: tuple[ContextItem, ...]
    correct_id: str


def _item(name: str, content: str) -> ContextItem:
    return ContextItem(name, name, name, content, content, (content,), "compacted", 1, 999999)


def _rank_case(name: str, query: str, decoy: str, answer: str) -> M3Case:
    return M3Case(name, query, (_item("item_0", decoy),
                               _item("item_1", answer),
                               _item("item_2", "sample plain index alpha")), "item_1")


M3_CASES = (
    _rank_case("widget", "widget widget widget: find rendering entry",
               "widget widget widget index for archived notes", "UI renderer is in panel_view module"),
    _rank_case("parser", "parser parser parser: locate catalog entry",
               "parser parser parser examples index", "parse_catalog is the catalog entry function"),
    _rank_case("graph", "graph graph graph: find drawing helper",
               "graph graph graph list of old sketches", "draw_nodes is the drawing helper"),
)


class ScriptedSkip:
    async def choose(self, state: str, questions: dict[str, Any], question_id: str) -> str:
        return "skip"


class ScriptedOracle:
    """Positive control for measurement plumbing, NOT a model result."""

    async def choose(self, state: str, questions: dict[str, Any], question_id: str) -> str:
        criteria = questions[question_id]["criteria"]
        if state.startswith("Current query:"):
            return "item_1"
        return next((key for key in criteria if key != "skip"), "skip")


async def evaluate(client: Any, *, conditions: str = "scripted_local") -> dict[str, Any]:
    m2 = []
    m3 = []
    for case in M2_CASES:
        baseline = RuleSpanSelector().select(case.text)
        assert baseline is not None
        start = time.perf_counter()
        rule_store = MemoryContextStore()
        rule = rule_store.compact(affinity="synthetic", tool_call_id=case.name,
                                  original=case.text, tool_name="terminal")
        rule_visible = rule_store.visible_content("synthetic", case.name, case.text)
        rule_ms = (time.perf_counter() - start) * 1000
        calls = []
        start = time.perf_counter()
        extra = await select_extra_lines(case.text, case.goal, baseline.lines, cast(LayaClient, client),
                                         lambda count=calls: count.append(1)) if rule.ok else None
        candidate_store = MemoryContextStore()
        # Match run_turn: no verified extra line means no compaction, not a
        # lossy rule excerpt presented as an accepted Laya result.
        if extra is None and rule.ok:
            visible = case.text
            candidate_original = True
        else:
            candidate = candidate_store.compact(
                affinity="synthetic", tool_call_id=case.name, original=case.text,
                tool_name="terminal", choice=SpanChoice(extra, "laya") if extra else None,
            )
            visible = candidate_store.visible_content("synthetic", case.name, case.text)
            candidate_original = not candidate.ok
            assert not candidate.ok or (candidate.item is not None and
                                        all(line in case.text.splitlines() for line in candidate.item.excerpt_lines))
        candidate_ms = (time.perf_counter() - start) * 1000
        m2.append({"case": case.name, "rule_missing": sum(line not in rule_visible for line in case.required),
                   "candidate_missing": sum(line not in visible for line in case.required),
                   "new_missing": sum(line in rule_visible and line not in visible for line in case.required),
                   "rule_original": not rule.ok, "candidate_original": candidate_original,
                   "rule_chars": len(rule_visible), "candidate_chars": len(visible),
                   "original_chars": len(case.text), "remote_calls": len(calls),
                   "rule_ms": round(rule_ms, 3), "candidate_ms": round(candidate_ms, 3)})
    for case in M3_CASES:
        start = time.perf_counter()
        baseline = rank_compacted_items(case.query, case.items)
        rule_ms = (time.perf_counter() - start) * 1000
        calls = []
        start = time.perf_counter()
        proposed = await rank_items(case.query, baseline, cast(LayaClient, client),
                                    lambda count=calls: count.append(1))
        candidate_ms = (time.perf_counter() - start) * 1000
        same_set = sorted(item.item_id for item in proposed) == sorted(item.item_id for item in baseline)
        assert same_set and all(item.visibility == "compacted" for item in proposed)
        m3.append({"case": case.name, "rule_first_correct": baseline[0].item_id == case.correct_id,
                   "candidate_first_correct": proposed[0].item_id == case.correct_id,
                   "all_items_retained": same_set, "automatic_unhide": False,
                   "remote_calls": len(calls), "rule_ms": round(rule_ms, 3),
                   "candidate_ms": round(candidate_ms, 3)})
    return {"conditions": conditions, "m2": m2, "m3": m3,
            "totals": {"m2_cases": len(m2), "rule_missing": sum(row["rule_missing"] for row in m2),
                       "candidate_missing": sum(row["candidate_missing"] for row in m2),
                       "new_missing": sum(row["new_missing"] for row in m2),
                       "m2_remote_calls": sum(row["remote_calls"] for row in m2),
                       "m3_cases": len(m3), "rule_first_hits": sum(row["rule_first_correct"] for row in m3),
                       "candidate_first_hits": sum(row["candidate_first_correct"] for row in m3),
                       "m3_remote_calls": sum(row["remote_calls"] for row in m3)}}


async def main() -> int:
    origin = os.environ.get("LAYA_BASE_URL")
    if origin:
        if os.getenv("LAYA_REMOTE_TEST_APPROVED") != "true":
            print("Remote test approval is required after log/retention verification.", file=sys.stderr)
            return 2
        try:
            client = LayaClient(origin, approved_origin=os.getenv("LAYA_APPROVED_ORIGIN", ""), timeout_seconds=10)
        except ValueError:
            print("Laya origin must match the approved tailnet destination.", file=sys.stderr)
            return 2
        try:
            result = await evaluate(client, conditions="remote_synthetic")
        finally:
            await client.close()
    else:
        result = {"controls_only": True,
                  "skip": await evaluate(ScriptedSkip(), conditions="scripted_skip"),
                  "oracle": await evaluate(ScriptedOracle(), conditions="scripted_oracle")}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
