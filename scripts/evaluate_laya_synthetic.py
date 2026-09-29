"""Offline paired M2/M3 evaluation. Synthetic data only; no network or transcripts.

Run: .venv/bin/python -m scripts.evaluate_laya_synthetic
The Laya choices below are scripted, NOT measurements of the hosted model.
"""

import asyncio
import json
import time
from typing import cast

from openai_compatible_bridge.context_compaction import (
    ContextItem,
    MemoryContextStore,
    RuleSpanSelector,
    SpanChoice,
    rank_compacted_items,
)
from openai_compatible_bridge.laya_http import LayaClient
from openai_compatible_bridge.laya_selection import rank_items, select_extra_lines


def output(middle):
    rows = [f"ordinary synthetic line {index:02d} " + "x" * 60 for index in range(18)]
    rows[10] = middle
    return "\n".join(rows)


# Ground truth is declared before making either choice. No actual user data.
M2 = [
    ("delivery_ko", "배송 지연 사유: 도로 통제로 배차가 늦어졌습니다."),
    ("delivery_en", "Delivery delayed because the carrier missed pickup."),
    ("maintenance_ko", "점검 시간: 14:30~16:00, 결제 조회가 제한됩니다."),
    ("maintenance_en", "Maintenance window 02:00-03:00 UTC; checkout unavailable."),
    ("payment_ko", "결제 상태 미확인: 승인 결과를 아직 받지 못했습니다."),
    ("payment_en", "Payment authorization is pending verification."),
    ("failure_ko", "처리 결과: 요청이 거절됐고 재시도 대기 중입니다."),
    ("failure_en", "Job ended with status unsuccessful, awaiting review."),
]


class ScriptedChoice:
    def __init__(self, choice):
        self.choice = choice
        self.calls = 0

    async def choose(self, state, questions, question_id):
        self.calls += 1
        return self.choice(questions[question_id]["criteria"])


def m2_case(label, critical):
    text = output(critical)
    baseline = RuleSpanSelector().select(text)
    assert baseline is not None
    base_start = time.perf_counter()
    base_store = MemoryContextStore()
    base = base_store.compact(affinity="fixture", tool_call_id="call", original=text, tool_name="terminal")
    base_visible = base_store.visible_content("fixture", "call", text)
    base_ms = (time.perf_counter() - base_start) * 1000
    client = ScriptedChoice(lambda criteria: next((key for key, value in criteria.items() if value.startswith(critical[:80])), "skip"))
    remote_calls = []

    async def candidate():
        extra = await select_extra_lines(text, "synthetic task", baseline.lines, cast(LayaClient, client), lambda: remote_calls.append(1)) if base.ok else None
        store = MemoryContextStore()
        result = store.compact(affinity="fixture", tool_call_id="call", original=text,
                               tool_name="terminal", choice=SpanChoice(extra, "laya") if extra else None)
        return store.visible_content("fixture", "call", text), result

    start = time.perf_counter()
    candidate_visible, candidate_result = asyncio.run(candidate())
    candidate_ms = (time.perf_counter() - start) * 1000
    assert len(remote_calls) == client.calls
    assert critical in text
    assert critical in base_visible and critical in candidate_visible
    return {"case": label, "rule_missing": int(critical not in base_visible),
            "candidate_missing": int(critical not in candidate_visible),
            "new_missing": int(critical in base_visible and critical not in candidate_visible),
            "rule_original": not base.ok, "candidate_original": not candidate_result.ok,
            "remote_calls": client.calls, "rule_ms": round(base_ms, 3),
            "candidate_ms": round(candidate_ms, 3)}


def m3_case(label, decision):
    def item(name, summary):
        return ContextItem(name, name, name, summary, summary, (summary,), "compacted", 1, 999999)
    items = [item("decoy", "receipt receipt receipt receipt unrelated record"),
             item("answer", "carrier could not collect the parcel"),
             item("another_decoy", "receipt receipt archived document")]
    query = "receipt receipt: why is my delivery late?"
    start = time.perf_counter()
    baseline = rank_compacted_items(query, items)
    base_ms = (time.perf_counter() - start) * 1000
    assert baseline[0].item_id != "answer"
    client = ScriptedChoice(lambda criteria: decision)
    calls = []
    start = time.perf_counter()
    proposed = asyncio.run(rank_items(query, baseline, cast(LayaClient, client), lambda: calls.append(1)))
    candidate_ms = (time.perf_counter() - start) * 1000
    assert len(calls) == client.calls == 1
    assert sorted(item.item_id for item in proposed) == sorted(item.item_id for item in baseline)
    assert all(item.visibility == "compacted" for item in proposed)
    return {"case": label, "rule_first_correct": baseline[0].item_id == "answer",
            "candidate_first_correct": proposed[0].item_id == "answer",
            "candidate_first": proposed[0].item_id, "all_items_retained": True,
            "automatic_unhide": False, "remote_calls": client.calls,
            "rule_ms": round(base_ms, 3), "candidate_ms": round(candidate_ms, 3)}


def main():
    m2 = [m2_case(label, critical) for label, critical in M2]
    m3 = [m3_case("improvement", "answer"), m3_case("wrong_choice", "another_decoy"),
          m3_case("no_improvement", "skip")]
    assert [case["candidate_first_correct"] for case in m3] == [True, False, False]
    print(json.dumps({"conditions": "offline scripted Laya choices; not remote model accuracy; wall time local only",
                      "m2": m2, "m3": m3,
                      "totals": {"m2_cases": len(m2), "rule_missing": sum(row["rule_missing"] for row in m2),
                                 "candidate_missing": sum(row["candidate_missing"] for row in m2),
                                 "new_missing": sum(row["new_missing"] for row in m2),
                                 "m2_calls": sum(row["remote_calls"] for row in m2),
                                 "m3_cases": len(m3), "rule_hits": sum(row["rule_first_correct"] for row in m3),
                                 "candidate_hits": sum(row["candidate_first_correct"] for row in m3),
                                 "m3_calls": sum(row["remote_calls"] for row in m3)}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
