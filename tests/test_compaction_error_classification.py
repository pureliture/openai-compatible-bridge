"""Synthetic safe-classification regressions. All provider traffic is mocked."""
from __future__ import annotations

import copy
import json
import re
import time
import asyncio

import httpx
import pytest

import openai_compatible_bridge.context_compaction as context_compaction
from openai_compatible_bridge.context_compaction import CompactionUpstreamError, HIDE_TOOL
from openai_compatible_bridge.main import (
    _log_compaction_upstream_failure,
    _private_compaction_error,
)
from openai_compatible_bridge.providers.vertex import VertexAPIError
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from test_compaction_foundry_protocols import MESSAGES, native_call, native_response
from test_compaction_review_regressions import (
    PRIVATE_MESSAGE,
    review_bridge,
    review_post,
    private_wire,
    sse,
)


def _diagnostics(caplog, event: str) -> list[str]:
    return [record.getMessage() for record in caplog.records
            if record.name == "context_compaction" and event in record.getMessage()]


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_initial_provider_429_is_logged_once_and_has_upstream_message(review_bridge, stream, caplog):
    _, config, replies, requests, _, lfm, app, _ = review_bridge
    config["protocol"] = "openai_chat_completions"
    replies.append((429, {"error": {
        "message": "PRIVATE_UPSTREAM_BODY request quota tokens",
        "code": "rate_limit_exceeded",
    }}))

    with caplog.at_level("WARNING", logger="context_compaction"):
        response = review_post(review_bridge, stream=stream)

    if stream:
        error = json.loads(response.text.split("data: ", 1)[1].split("\n\n", 1)[0])["error"]
    else:
        assert response.status_code == 429
        error = response.json()["error"]
    assert error["message"] == "The upstream model rejected or limited the request."
    assert error["code"] == "rate_limit_exceeded"
    assert error["type"] == "rate_limit_error"
    assert len(requests) == 1 and not lfm.calls
    assert app.state.context_compaction_store.items("synthetic-review") == ()

    lines = _diagnostics(caplog, "upstream_failed")
    assert len(lines) == 1
    line = lines[0]
    assert "phase=initial" in line and "round=1" in line
    assert "status=429" in line and "upstream_status=429" in line
    assert "failure_category=upstream_rate_limited" in line
    assert "code=rate_limit_exceeded" in line
    assert re.search(r"correlation_id=[0-9a-f]{16}(?:\s|$)", line)
    assert all(marker not in line for marker in ("PRIVATE_UPSTREAM_BODY", "request quota", "tokens"))
    assert "request_quota" not in line and "token_quota" not in line
    assert "hide" not in error["message"].lower()


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_continuation_provider_429_keeps_phase_round_and_prior_hide(review_bridge, stream, caplog):
    _, config, replies, requests, _, _, app, _ = review_bridge
    config["protocol"] = "openai_responses"
    if stream:
        first = private_wire("openai_responses")
    else:
        first = native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"}))
    replies.extend([first, (429, {"error": {
        "message": "PRIVATE_CONTINUATION_BODY",
        "code": "rate_limit_error",
    }})])

    with caplog.at_level("WARNING", logger="context_compaction"):
        response = review_post(review_bridge, stream=stream)

    if stream:
        error = json.loads(response.text.split("data: ", 1)[1].split("\n\n", 1)[0])["error"]
    else:
        assert response.status_code == 429
        error = response.json()["error"]
    assert error["message"] == "The upstream model rejected or limited the request."
    assert error["code"] == "rate_limit_error"
    assert len(requests) == 2
    hidden = app.state.context_compaction_store.items("synthetic-review")
    assert len(hidden) == 1 and hidden[0].original
    assert app.state.context_compaction_store.last_measurement.provider_calls == 1

    lines = _diagnostics(caplog, "upstream_failed")
    assert len(lines) == 1
    line = lines[0]
    assert "phase=continuation" in line and "round=2" in line
    assert "status=429" in line and "upstream_status=429" in line
    assert "failure_category=upstream_rate_limited" in line
    assert "PRIVATE_CONTINUATION_BODY" not in line


@pytest.mark.parametrize(("status", "code", "stage", "category", "safe_code"), [
    (504, "timeout", "provider_connect", "upstream_transport", "timeout"),
    (504, "timeout", "provider_read", "upstream_transport", "timeout"),
    (503, "unavailable", "provider_http_status", "upstream_server_error", "upstream_error"),
    (429, "requests_per_minute", "provider_http_status", "upstream_rate_limited", "upstream_error"),
    (504, "timeout", "collector_idle", "local_compaction_idle_timeout", "timeout"),
    (502, "context_compaction_stream_limit", "collector_buffer", "local_compaction_buffer_limit",
     "context_compaction_stream_limit"),
    (502, "PRIVATE_MARKER", "PRIVATE_MARKER", "upstream_error", "upstream_error"),
])
def test_diagnostic_uses_allowlisted_classification_and_status_evidence(
    status, code, stage, category, safe_code, caplog,
):
    cause = VertexAPIError(status, "PRIVATE_MARKER", code=code, stage=stage)
    failure = CompactionUpstreamError(
        [{"prompt_tokens": 1}], phase="continuation", round_number=2,
        correlation_id="0123456789abcdef",
    )
    with caplog.at_level("WARNING", logger="context_compaction"):
        try:
            raise failure from cause
        except CompactionUpstreamError as caught:
            _log_compaction_upstream_failure(
                caught, provider="foundry", protocol="openai_responses", stream=True,
                started_at=time.monotonic(),
            )

    event = "processing_failed" if category.startswith("local_compaction_") else "upstream_failed"
    lines = _diagnostics(caplog, event)
    assert len(lines) == 1
    line = lines[0]
    assert ("phase=local" in line) is category.startswith("local_compaction_")
    assert f"failure_category={category}" in line
    assert f"code={safe_code}" in line
    expected_stage = stage if stage in {
        "provider_connect", "provider_read", "provider_http_status", "collector_idle", "collector_buffer",
    } else "unknown"
    assert f"stage={expected_stage}" in line
    assert "PRIVATE_MARKER" not in line
    if stage == "provider_http_status":
        assert f"upstream_status={status}" in line


def test_native_sse_error_records_bridge_and_upstream_status_separately(review_bridge, caplog):
    _, config, replies, requests, _, _, app, _ = review_bridge
    config["protocol"] = "openai_chat_completions"
    replies.extend([
        private_wire("openai_chat_completions"),
        sse({"type": "error", "error": {"message": PRIVATE_MESSAGE, "code": "PRIVATE_CODE"}}),
    ])

    with caplog.at_level("WARNING", logger="context_compaction"):
        response = review_post(review_bridge, stream=True)

    error = json.loads(response.text.split("data: ", 1)[1].split("\n\n", 1)[0])["error"]
    assert error["message"] == "The upstream model could not complete the request."
    assert error["code"] == "upstream_error"
    assert len(requests) == 2
    assert app.state.context_compaction_store.items("synthetic-review")[0].original

    lines = _diagnostics(caplog, "upstream_failed")
    assert len(lines) == 1
    line = lines[0]
    assert "status=502" in line and "upstream_status=200" in line
    assert "stage=provider_sse_error" in line
    assert "failure_category=upstream_response_error" in line
    assert PRIVATE_MESSAGE not in line and "PRIVATE_CODE" not in line


def test_hide_store_failure_preserves_tool_fallback_and_logs_once(
    review_bridge, monkeypatch, caplog,
):
    _, config, replies, requests, _, _, app, _ = review_bridge
    config["protocol"] = "openai_responses"
    monkeypatch.setenv("CONTEXT_COMPACTION_LFM_ENABLED", "false")
    replies.extend([
        native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"})),
        native_response(text="final after failed hide fallback"),
    ])
    original_messages = copy.deepcopy(MESSAGES)
    store = app.state.context_compaction_store

    def fail_store(*_args, **_kwargs):
        raise OSError("PRIVATE_STORE_DETAIL")

    monkeypatch.setattr(store, "compact", fail_store)
    with caplog.at_level("WARNING", logger="context_compaction"):
        response = review_post(review_bridge, stream=False)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "final after failed hide fallback"
    assert len(requests) == 2
    assert store.items("synthetic-review") == ()
    assert MESSAGES == original_messages
    assert "store_unavailable" in str(requests[1])

    lines = _diagnostics(caplog, "processing_failed")
    assert len(lines) == 1
    assert "failure_category=local_compaction_store_error" in lines[0]
    assert "PRIVATE_STORE_DETAIL" not in lines[0]


@pytest.mark.parametrize("fail_at", [1, 2], ids=["preflight", "initial-loop"])
def test_initial_visibility_failure_falls_back_and_logs_once(
    review_bridge, monkeypatch, caplog, fail_at,
):
    _, config, replies, requests, _, _, app, _ = review_bridge
    config["protocol"] = "openai_responses"
    replies.append(native_response(text="uncompacted fallback response"))
    original_messages = copy.deepcopy(MESSAGES)
    real_apply_visibility = context_compaction.apply_visibility
    call_count = 0

    def fail_one_visibility(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == fail_at:
            raise OSError("PRIVATE_VISIBILITY_DETAIL")
        return real_apply_visibility(*args, **kwargs)

    monkeypatch.setattr(context_compaction, "apply_visibility", fail_one_visibility)
    with caplog.at_level("WARNING", logger="context_compaction"):
        response = review_post(review_bridge, stream=False)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "uncompacted fallback response"
    assert len(requests) == 1
    assert app.state.context_compaction_store.items("synthetic-review") == ()
    assert MESSAGES == original_messages
    lines = _diagnostics(caplog, "processing_failed")
    assert len(lines) == 1
    assert "status=0" in lines[0]
    assert "phase=local" in lines[0]
    assert "failure_category=local_compaction_store_error" in lines[0]
    assert "code=context_compaction_store_unavailable" in lines[0]
    assert "PRIVATE_VISIBILITY_DETAIL" not in lines[0]


def test_private_buffer_error_has_local_user_message():
    error = _private_compaction_error(VertexAPIError(
        502, "PRIVATE_BUFFER_DETAIL", code="context_compaction_stream_limit", stage="collector_buffer",
    ))
    assert error.status_code == 502
    assert error.code == "context_compaction_stream_limit"
    assert error.message == "Context compaction could not complete local processing."
    assert "PRIVATE_BUFFER_DETAIL" not in error.message


def test_provider_wrapper_does_not_label_local_buffer_failure_as_upstream(caplog):
    cause = VertexAPIError(
        502, "PRIVATE_BUFFER_DETAIL", code="context_compaction_stream_limit", stage="collector_buffer",
    )
    failure = CompactionUpstreamError(
        [{"prompt_tokens": 1}], phase="continuation", round_number=2,
        correlation_id="0123456789abcdef",
    )
    with caplog.at_level("WARNING", logger="context_compaction"):
        try:
            raise failure from cause
        except CompactionUpstreamError as caught:
            _log_compaction_upstream_failure(
                caught, provider="foundry", protocol="openai_responses", stream=True,
                started_at=time.monotonic(),
            )

    lines = _diagnostics(caplog, "processing_failed")
    assert len(lines) == 1
    assert "phase=local" in lines[0]
    assert "failure_category=local_compaction_buffer_limit" in lines[0]
    assert "stage=collector_buffer" in lines[0]
    assert _diagnostics(caplog, "upstream_failed") == []


def test_native_sse_rate_limit_has_a_distinct_safe_user_message():
    error = _private_compaction_error(VertexAPIError(
        502, "PRIVATE_SSE_DETAIL", code="rate_limit_error", stage="provider_sse_error",
        upstream_status=200,
    ))
    assert error.status_code == 502
    assert error.code == "rate_limit_error"
    assert error.upstream_status == 200
    assert error.message == "The upstream model rejected or limited the request."
    assert "PRIVATE_SSE_DETAIL" not in error.message


def test_loop_limit_is_logged_as_local_compaction_failure(review_bridge, monkeypatch, caplog):
    _, config, replies, requests, _, _, app, _ = review_bridge
    config["protocol"] = "openai_responses"
    monkeypatch.setenv("CONTEXT_COMPACTION_LFM_ENABLED", "false")
    monkeypatch.setenv("CONTEXT_COMPACTION_MAX_INTERNAL_ROUNDS", "1")
    hide = native_response(native_call(HIDE_TOOL, {"tool_call_id": "original-call"}))
    replies.extend([hide, hide])

    with caplog.at_level("WARNING", logger="context_compaction"):
        response = review_post(review_bridge, stream=False)

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Context compaction could not complete local processing."
    assert response.json()["error"]["code"] == "context_compaction_loop_limit"
    assert len(requests) == 2
    lines = _diagnostics(caplog, "processing_failed")
    assert len(lines) == 1
    assert "failure_category=local_compaction_loop_limit" in lines[0]
    assert "phase=local" in lines[0] and "round=2" in lines[0]
    assert len(app.state.context_compaction_store.items("synthetic-review")) == 1


def test_foundry_connect_timeout_keeps_provider_connect_stage():
    async def run():
        provider = FoundryChatClient(
            base_url="https://foundry.example/api/v2/llm/proxy/openai/v1/chat/completions",
            token="synthetic",
        )
        await provider.http.aclose()

        async def handler(_request):
            raise httpx.ConnectTimeout("PRIVATE_CONNECT_DETAIL")

        provider.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(VertexAPIError) as caught:
                await provider.generate(
                    model="synthetic-model", messages=[], max_tokens=1,
                    resolved_config={"protocol": "openai_chat_completions"},
                )
            assert caught.value.stage == "provider_connect"
            assert caught.value.code == "timeout"
        finally:
            await provider.http.aclose()

    asyncio.run(run())
