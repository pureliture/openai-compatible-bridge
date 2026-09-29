"""Synthetic quality corpus; never include transcripts or credentials."""

import asyncio

from openai_compatible_bridge.context_compaction import (
    MemoryContextStore,
    rank_compacted_items,
)
from scripts.evaluate_laya_value import M2_CASES, M3_CASES, ScriptedOracle, evaluate


def test_m2_fixtures_are_actually_compactable_and_rule_misses_middle():
    for case in M2_CASES:
        store = MemoryContextStore()
        result = store.compact(affinity="synthetic", tool_call_id=case.name,
                               original=case.text, tool_name="terminal")
        assert result.ok and result.item is not None, case.name
        assert len(result.item.compacted) < len(case.text)
        assert any(line not in result.item.excerpt_lines for line in case.required), case.name


def test_m3_fixtures_have_unambiguous_answer_and_wrong_real_rule_order():
    for case in M3_CASES:
        ranked = rank_compacted_items(case.query, case.items)
        assert {item.item_id for item in ranked} == {item.item_id for item in case.items}
        assert ranked[0].item_id != case.correct_id, case.name
        assert sum(item.item_id == case.correct_id for item in case.items) == 1


def test_scripted_comparison_counts_new_omissions_and_bad_ranks():
    class Scripted:
        calls = 0

        async def choose(self, state, questions, question_id):
            self.calls += 1
            choices = questions[question_id]["criteria"]
            # Refuse all M2 suggestions; deliberately select an M3 wrong id.
            return "skip" if "Current task:" in state else next(key for key in choices if key != "skip")

    result = asyncio.run(evaluate(Scripted()))
    assert result["conditions"] == "scripted_local"
    assert all(row["rule_missing"] for row in result["m2"])
    assert all(row["candidate_missing"] for row in result["m2"])
    assert all(not row["candidate_first_correct"] for row in result["m3"])
    assert all(row["all_items_retained"] and not row["automatic_unhide"] for row in result["m3"])


def test_scripted_oracle_proves_evaluator_can_observe_improvement():
    result = asyncio.run(evaluate(ScriptedOracle(), conditions="scripted_oracle"))
    assert result["totals"]["rule_missing"] == len(M2_CASES)
    assert result["totals"]["candidate_missing"] == 0
    assert result["totals"]["rule_first_hits"] == 0
    assert result["totals"]["candidate_first_hits"] == len(M3_CASES)