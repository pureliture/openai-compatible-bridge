from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from test_cost_backend import app_for, cost_env
from openai_compatible_bridge.core.async_cost import (
    AsyncCostAccounting,
    build_async_cost_accounting,
)
from openai_compatible_bridge.core.cost_tracking import (
    CostConfigError,
    CostSubsystemUnhealthy,
    NormalizedUsage,
)


def _cost_env(tmp_path, workers=None):
    env = cost_env(tmp_path) | {
        "COST_PROVIDER_BILLING_JSON": '{"vertex":"metered"}',
    }
    if workers is not None:
        env["COST_ADMISSION_WORKERS"] = workers
    return env


@pytest.mark.parametrize(("workers", "expected"), [
    (None, 4), ("", 4), ("  ", 4), ("1", 1), ("7", 7), (" 8 ", 8), ("64", 64),
])
def test_admission_worker_env_default_and_override(tmp_path, workers, expected):
    accounting = build_async_cost_accounting(_cost_env(tmp_path, workers))
    try:
        assert isinstance(accounting, AsyncCostAccounting)
        assert accounting._admission_workers == expected
        assert accounting._admission_executor._max_workers == expected
        assert accounting.metrics()["admission_capacity"] == expected
    finally:
        asyncio.run(accounting.aclose())


@pytest.mark.parametrize("workers", ["invalid", "0", "-1", "65", "1.5", "nan", "inf"])
def test_invalid_admission_worker_env_fails_closed(tmp_path, workers):
    accounting = build_async_cost_accounting(_cost_env(tmp_path, workers))
    assert isinstance(accounting, AsyncCostAccounting)
    assert accounting.gate is None

    async def verify():
        try:
            async with accounting.reservation(
                endpoint="chat", model="gemini-2.5-flash", provider="vertex",
                forecast_usage=NormalizedUsage(prompt_tokens=1),
            ):
                with pytest.raises(CostConfigError):
                    await accounting.before_attempt("vertex")
        finally:
            await accounting.aclose()

    asyncio.run(verify())


async def _wait_for_thread_event(event: threading.Event):
    for _ in range(1000):
        if event.is_set():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("admission worker did not start")


def test_saturated_admission_waits_for_worker_then_runs():
    accounting = AsyncCostAccounting(
        config=None, billing={}, admission_workers=1, admission_timeout=1.0,
    )
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_operation():
        entered.set()
        if not release.wait(2):
            raise TimeoutError("test worker was not released")
        calls.append("first")
        return "first"

    async def run():
        first = asyncio.create_task(accounting._admit_io(slow_operation))
        second = None
        try:
            await _wait_for_thread_event(entered)
            second = asyncio.create_task(accounting._admit_io(lambda: calls.append("second") or "second"))
            await asyncio.sleep(0.02)
            assert not second.done()
            assert accounting.metrics()["admission_queue_depth"] == 1
            assert accounting.metrics()["admission_in_flight"] == 1
            release.set()
            assert await first == "first"
            assert await second == "second"
            assert calls == ["first", "second"]
        finally:
            release.set()
            await asyncio.gather(first, *(tuple([second]) if second is not None else ()), return_exceptions=True)
            await accounting.aclose()

    asyncio.run(run())


def test_saturated_admission_fails_only_after_queue_timeout():
    accounting = AsyncCostAccounting(
        config=None, billing={}, admission_workers=1, admission_timeout=0.3,
    )
    entered, release = threading.Event(), threading.Event()

    def slow_operation():
        entered.set()
        release.wait(2)
        return "finished"

    async def run():
        first = asyncio.create_task(accounting._admit_io(slow_operation))
        second = None
        try:
            await _wait_for_thread_event(entered)
            second_started = asyncio.get_running_loop().time()
            second = asyncio.create_task(accounting._admit_io(lambda: "must not run"))
            await asyncio.sleep(0.02)
            assert not second.done()
            assert accounting.metrics()["admission_queue_depth"] == 1
            with pytest.raises(CostSubsystemUnhealthy, match="cost admission timed out"):
                await second
            assert asyncio.get_running_loop().time() - second_started >= 0.25
            assert accounting.metrics()["admission_queue_depth"] == 0
            assert accounting.metrics()["admission_in_flight"] == 1
            # The active call and its queued successor each consume one deadline.
            assert accounting.metrics()["admission_timeouts"] == 2
            assert accounting.metrics()["admission_rejected"] == 1
            release.set()
            with pytest.raises(CostSubsystemUnhealthy, match="cost admission timed out"):
                await first
            await asyncio.wait_for(
                asyncio.gather(*tuple(accounting._admissions), return_exceptions=True), 1,
            )
        finally:
            release.set()
            await asyncio.gather(first, *(tuple([second]) if second is not None else ()), return_exceptions=True)
            await accounting.aclose()

    asyncio.run(run())


def test_readyz_recording_keeps_existing_keys_and_adds_admission_queue_depth():
    accounting = AsyncCostAccounting(config=None, billing={})
    required = {
        "admission_timeouts", "admission_rejected", "admission_failures",
        "admission_in_flight", "admission_capacity", "queue_depth", "queue_capacity",
        "records_written", "records_dropped", "record_failures", "usage_missing",
        "record_in_flight", "last_record_success_at",
    }
    with TestClient(app_for(lambda: accounting)) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        metrics = response.json()["cost_tracking"]["recording"]
        assert required <= metrics.keys()
        assert "admission_queue_depth" in metrics


@pytest.mark.parametrize(("value", "expected"), [
    ("", 10.0), ("0.5", 0.5), ("5", 5.0), ("30", 30.0),
])
def test_admission_timeout_env_boundaries(tmp_path, value, expected):
    accounting = build_async_cost_accounting(_cost_env(tmp_path) | {
        "COST_ADMISSION_TIMEOUT_SECONDS": value,
    })
    try:
        assert accounting.gate is not None
        assert accounting._admission_timeout == expected
    finally:
        asyncio.run(accounting.aclose())


@pytest.mark.parametrize("value", ["invalid", "0", "0.49", "30.01", "nan", "inf", "-inf"])
def test_invalid_admission_timeout_keeps_http_error_contract(tmp_path, value):
    from test_cost_postgres_api import HTTPChatProvider, chat_app, chat_payload

    env = _cost_env(tmp_path) | {"COST_ADMISSION_TIMEOUT_SECONDS": value}
    provider = HTTPChatProvider()
    accounting = build_async_cost_accounting(env)
    with TestClient(chat_app(accounting, provider)) as client:
        response = client.post("/v1/chat/completions", json=chat_payload())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "cost_config_error"
        assert provider.calls == 0
        assert client.get("/readyz").status_code == 503


@pytest.mark.parametrize("queued", [False, True])
def test_cancelled_admission_keeps_running_slot_and_releases_waiter(queued):
    accounting = AsyncCostAccounting(
        config=None, billing={}, admission_workers=1, admission_timeout=1.0,
    )
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_operation():
        entered.set()
        assert release.wait(2)
        calls.append("first")

    async def run():
        first = asyncio.create_task(accounting._admit_io(slow_operation))
        waiting = None
        try:
            await _wait_for_thread_event(entered)
            if queued:
                waiting = asyncio.create_task(accounting._admit_io(lambda: calls.append("cancelled")))
                async with asyncio.timeout(1):
                    while accounting.metrics()["admission_queue_depth"] != 1:
                        await asyncio.sleep(0)
            cancelled = waiting if queued else first
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            # 호출자가 취소해도 실행 중인 DB 작업이 끝날 때까지 슬롯을 유지한다.
            assert accounting.metrics()["admission_in_flight"] == 1
            assert accounting.metrics()["admission_queue_depth"] == 0
            release.set()
            await asyncio.gather(*tuple(accounting._admissions))
            assert await accounting._admit_io(lambda: "replacement") == "replacement"
            assert calls == ["first"]
            assert accounting.metrics()["admission_in_flight"] == 0
        finally:
            release.set()
            await asyncio.gather(first, *([waiting] if waiting else []), return_exceptions=True)
            await accounting.aclose()

    asyncio.run(run())


def test_admission_submission_failure_returns_slot_and_fails_closed(monkeypatch):
    accounting = AsyncCostAccounting(
        config=None, billing={}, admission_workers=1, admission_timeout=1.0,
    )
    submit = accounting._admission_executor.submit

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic executor detail")

    async def run():
        try:
            monkeypatch.setattr(accounting._admission_executor, "submit", fail)
            with pytest.raises(CostSubsystemUnhealthy, match="cost admission capacity unavailable") as error:
                await accounting._admit_io(lambda: pytest.fail("submission failure must block work"))
            assert "synthetic" not in str(error.value)
            assert accounting.metrics()["admission_rejected"] == 1
            assert accounting.metrics()["admission_in_flight"] == 0
            monkeypatch.setattr(accounting._admission_executor, "submit", submit)
            assert await accounting._admit_io(lambda: "replacement") == "replacement"
        finally:
            await accounting.aclose()

    asyncio.run(run())
