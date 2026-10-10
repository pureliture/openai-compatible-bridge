"""Offline request-size regressions for UTF-8 LFM prompt serialization."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from openai_compatible_bridge.context_compaction import DEFAULT_LFM_MAX_INPUT_BYTES, load_settings
from openai_compatible_bridge.lfm_summary import LFMSummarizer, LFMUnavailable


def _capture_request(original: str, *, max_bytes: int = DEFAULT_LFM_MAX_INPUT_BYTES):
    calls: list[dict[str, Any]] = []
    on_call: list[bool] = []

    async def generate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"text": json.dumps({"summary": "synthetic result"})}

    summarizer = LFMSummarizer(
        generate=generate,
        settings=load_settings({"CONTEXT_COMPACTION_LFM_MAX_INPUT_BYTES": str(max_bytes)}),
    )
    try:
        asyncio.run(summarizer.summarize(original, (), lambda: on_call.append(True), invocation={}))
    except LFMUnavailable:
        # This helper checks request assembly; summary quality is covered elsewhere.
        pass
    return calls, on_call


def _prompt_bytes(request: dict[str, Any]) -> int:
    return sum(len(message["content"].encode("utf-8")) for message in request["messages"])


@pytest.mark.parametrize("structured", [False, True])
def test_non_ascii_prompt_uses_utf8_budget_for_plain_and_structured_sources(structured: bool):
    # Sized so BOTH engine prompt routes (plain _PLAIN_RESULT_PROMPT and
    # structured _SYSTEM_PROMPT) stay under DEFAULT_LFM_MAX_INPUT_BYTES;
    # 45 lines would exceed it after the SSoT delegation (2026-10-10).
    source = "\n".join("한글 결과 내용 데이터 분석 문서 " * 3 + str(index) for index in range(44)) + "\n🙂"
    original = json.dumps({"output": source, "exit_code": 0}, ensure_ascii=False) if structured else source

    calls, on_call = _capture_request(original)

    assert len(calls) == 1
    assert on_call == [True]
    user_content = calls[0]["messages"][1]["content"]
    # Engine routes differ: structured prefixes "Write factual findings as JSON."
    # before the `{"result": {"source": {...}}}` envelope; plain sends just the
    # `{"source": ...}` envelope. Strip the prefix, then decode either shape.
    if structured:
        head, _, body = user_content.partition("\n\n")
        assert head == "Write factual findings as JSON."
        packet = json.loads(body)
    else:
        packet = json.loads(user_content)
    if structured:
        payload = packet["result"]["source"]
    else:
        payload = packet["source"]
    assert isinstance(payload, (str, dict))
    blob = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    assert "한글" in blob and "🙂" in blob
    assert len(user_content.encode("utf-8")) <= DEFAULT_LFM_MAX_INPUT_BYTES
    assert _prompt_bytes(calls[0]) <= DEFAULT_LFM_MAX_INPUT_BYTES


def test_prompt_accepts_exact_utf8_byte_limit_and_rejects_one_byte_over():
    probe_calls, _ = _capture_request("x")
    assert len(probe_calls) == 1
    initial_size = _prompt_bytes(probe_calls[0])
    exact_source = "x" * (1 + DEFAULT_LFM_MAX_INPUT_BYTES - initial_size)

    exact_calls, exact_on_call = _capture_request(exact_source)
    assert len(exact_calls) == 1
    assert exact_on_call == [True]
    assert _prompt_bytes(exact_calls[0]) == DEFAULT_LFM_MAX_INPUT_BYTES

    over_calls, over_on_call = _capture_request(exact_source + "x")
    assert over_calls == []
    assert over_on_call == []


def test_ascii_prompt_keeps_existing_json_spacing_and_shape():
    original = "plain synthetic output"
    calls, _ = _capture_request(original)

    assert len(calls) == 1
    expected = json.dumps({"source": original, "required_evidence": []})
    assert calls[0]["messages"][1]["content"] == expected
