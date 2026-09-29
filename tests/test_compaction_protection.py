"""Safety regressions use synthetic tool output, never real transcripts."""

import pytest

from openai_compatible_bridge.context_compaction import MemoryContextStore


def output_with(*evidence):
    lines = [
        f"ordinary output line {i:03d} without special evidence" for i in range(80)
    ]
    lines[40:40] = evidence
    return "\n".join(lines)


@pytest.mark.parametrize(
    "error",
    [
        "오류: 결제 기록 저장 실패. 아직 해결되지 않음",
        "처리 중 오류가 발생했습니다",
        "저장에 실패했습니다",
        "FAILED tests/test_payment.py::test_save - AssertionError",
        "  error: database unavailable",
        '{"exit_code": 1, "output": "not saved"}',
        "exit_code: -9",
        "Duplicate invoice INV-EXAMPLE-42 was issued.",
        "중복 청구가 발생했습니다. 확인되지 않은 청구입니다.",
    ],
)
def test_unresolved_errors_keep_entire_original(error):
    original = output_with(error)
    store = MemoryContextStore()
    result = store.compact(
        affinity="synthetic",
        tool_call_id="call",
        original=original,
        tool_name="terminal",
    )
    assert not result.ok
    assert result.error == "protected_error"
    assert store.items("synthetic") == ()
    assert store.visible_content("synthetic", "call", original) == original


@pytest.mark.parametrize(
    "evidence",
    [
        tuple(f"id: receipt-{i}" for i in range(14)),
        ("id: " + "x" * 300,),
        ("648 passed, 1 warning in 75.20s",),
        ("제약: 운영 배포 금지",),
        ("Created record receipt-123",),
    ],
)
def test_required_evidence_is_never_silently_dropped(evidence):
    original = output_with(*evidence)
    store = MemoryContextStore()
    result = store.compact(
        affinity="synthetic",
        tool_call_id="call",
        original=original,
        tool_name="terminal",
    )
    visible = store.visible_content("synthetic", "call", original)
    assert all(line in visible for line in evidence)
    if result.ok:
        assert result.item is not None
        assert all(line in result.item.excerpt_lines for line in evidence)
    else:
        assert visible == original


@pytest.mark.parametrize("critical", [
    "배송 지연 사유: 도로 통제로 배차가 늦어졌습니다.",
    "Delivery delayed because the carrier missed pickup.",
    "점검 시간: 14:30~16:00, 결제 조회가 제한됩니다.",
    "Scheduled maintenance window 02:00-03:00 UTC; checkout unavailable.",
    "결제 상태 미확인: 승인 결과를 아직 받지 못했습니다.",
    "Payment status is pending verification; do not charge again.",
    "처리 결과: 요청이 거절됐고 재시도 대기 중입니다.",
    "Job ended with status unsuccessful, awaiting review.",
    "배송 지연은 해결됨. 이전 이유는 도로 통제였습니다.",
    "Payment dispute resolved after manual review.",
    "점검 완료 여부 불명확; 확인 전 재개하지 마세요.",
])
def test_business_state_and_reason_are_not_compacted_or_sent_to_selector(critical):
    original = output_with(critical)
    store = MemoryContextStore()
    result = store.compact(affinity="synthetic", tool_call_id="call", original=original, tool_name="terminal")
    assert not result.ok
    assert result.error == "protected_error"
    assert store.visible_content("synthetic", "call", original) == original


def test_risky_line_too_long_for_excerpt_stays_original():
    critical = "배송 지연 사유: " + "도로 통제 " * 60
    original = output_with(critical)
    store = MemoryContextStore()
    result = store.compact(affinity="synthetic", tool_call_id="call", original=original, tool_name="terminal")
    assert not result.ok
    assert store.visible_content("synthetic", "call", original) == original
