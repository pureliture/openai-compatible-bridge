"""Safety regressions use synthetic tool output, never real transcripts."""

import pytest

from openai_compatible_bridge.context_compaction import MemoryContextStore


def output_with(*evidence):
    lines = [f"ordinary output line {i:03d} without special evidence" for i in range(80)]
    lines[40:40] = evidence
    return "\n".join(lines)


@pytest.mark.parametrize("error", [
    "오류: 결제 기록 저장 실패. 아직 해결되지 않음",
    "처리 중 오류가 발생했습니다",
    "저장에 실패했습니다",
    "FAILED tests/test_payment.py::test_save - AssertionError",
    "  error: database unavailable",
    '{"exit_code": 1, "output": "not saved"}',
    'exit_code: -9',
])
def test_unresolved_errors_keep_entire_original(error):
    original = output_with(error)
    store = MemoryContextStore()
    result = store.compact(affinity="synthetic", tool_call_id="call", original=original, tool_name="terminal")
    assert not result.ok
    assert result.error == "protected_error"
    assert store.items("synthetic") == ()
    assert store.visible_content("synthetic", "call", original) == original


@pytest.mark.parametrize("evidence", [
    tuple(f"id: receipt-{i}" for i in range(14)),
    ("id: " + "x" * 300,),
    ("648 passed, 1 warning in 75.20s",),
    ("제약: 운영 배포 금지",),
    ("Created record receipt-123",),
])
def test_required_evidence_is_never_silently_dropped(evidence):
    original = output_with(*evidence)
    store = MemoryContextStore()
    result = store.compact(affinity="synthetic", tool_call_id="call", original=original, tool_name="terminal")
    visible = store.visible_content("synthetic", "call", original)
    assert all(line in visible for line in evidence)
    if result.ok:
        assert result.item is not None
        assert all(line in result.item.excerpt_lines for line in evidence)
    else:
        assert visible == original
