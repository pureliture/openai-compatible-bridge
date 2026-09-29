from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import postgres_helpers
import pytest
from test_cost_backend import cost_env

from openai_compatible_bridge.core.async_cost import (
    AsyncCostAccounting,
    build_async_cost_accounting,
)
from openai_compatible_bridge.core.cost_tracking import (
    CostBudgetExceeded,
    CostConfigError,
    CostSubsystemUnhealthy,
    CostTrackingConfig,
    NormalizedUsage,
    PricingCatalog,
)
from openai_compatible_bridge.core.postgres_cost_repository import (
    PostgresCostRepository,
    apply_migrations,
)

pg_dsn = postgres_helpers.pg_dsn


def service(tmp_path, pg_dsn, **kwargs):
    apply_migrations(pg_dsn)
    config = CostTrackingConfig.from_env(cost_env(tmp_path) | {
        "COST_LEDGER_BACKEND": "postgres", "COST_LEDGER_POSTGRES_DSN": pg_dsn,
    })
    return AsyncCostAccounting(
        config=config, ledger=PostgresCostRepository(pg_dsn), pricing=PricingCatalog.from_config(config),
        billing={"vertex": "metered", "ollama": "subscription", "foundry": "nonbillable"}, **kwargs,
    )


def context(accounting, provider="vertex", tokens=100):
    return accounting.reservation(
        endpoint="chat", model="gemini-2.5-flash", provider=provider,
        forecast_usage=NormalizedUsage(prompt_tokens=tokens),
    )


def test_each_attempt_is_reserved_and_subscription_never_touches_gate(tmp_path, pg_dsn):
    async def run():
        accounting = service(tmp_path, pg_dsn)
        try:
            async with context(accounting):
                first = await accounting.before_attempt("vertex")
                second = await accounting.before_attempt("vertex")
            assert first.reservation_id != second.reservation_id
            assert len(accounting.ledger.fetch_events()) == 2
            accounting.gate = None
            async with context(accounting, "ollama"):
                assert await accounting.before_attempt("ollama") is None
            async with context(accounting, "unknown"):
                with pytest.raises(CostConfigError):
                    await accounting.before_attempt("unknown")
        finally:
            await accounting.aclose()
    asyncio.run(run())


def test_slow_admission_is_bounded_without_blocking_event_loop(tmp_path, pg_dsn):
    async def run():
        accounting = service(tmp_path, pg_dsn, admission_workers=1, admission_timeout=0.05)
        release = threading.Event()
        entered = threading.Event()
        original = accounting.gate.preflight
        def slow(**kwargs):
            entered.set()
            release.wait(2)
            return original(**kwargs)
        accounting.gate.preflight = slow
        try:
            async with context(accounting):
                pending = asyncio.create_task(accounting.before_attempt("vertex"))
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                ticks = 0
                while not pending.done():
                    ticks += 1
                    await asyncio.sleep(0.005)
                with pytest.raises(CostSubsystemUnhealthy):
                    await pending
                assert ticks >= 3
                with pytest.raises(CostSubsystemUnhealthy):
                    await accounting.before_attempt("vertex")
                assert accounting.metrics()["admission_in_flight"] == 1
                assert accounting.metrics()["admission_timeouts"] == 1
                assert accounting.metrics()["admission_rejected"] == 1
        finally:
            release.set()
            await accounting.aclose()
    asyncio.run(run())


def test_record_queue_failure_and_missing_usage_do_not_poison_admission(tmp_path, pg_dsn):
    async def run():
        accounting = service(tmp_path, pg_dsn, queue_size=1)
        entered, release = threading.Event(), threading.Event()
        def failing(*args, **kwargs):
            entered.set()
            release.wait(2)
            raise OSError("secret must not appear")
        accounting.gate.finalize_success = failing
        try:
            async with context(accounting):
                holds = [await accounting.before_attempt("vertex") for _ in range(5)]
                accounting.record_attempt(holds[0], NormalizedUsage(prompt_tokens=1))
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                accounting.record_attempt(holds[1], NormalizedUsage(prompt_tokens=1))
                accounting.record_attempt(holds[2], NormalizedUsage(prompt_tokens=1))
                accounting.record_attempt(holds[3], None)
                assert accounting.metrics()["queue_depth"] == 1
                assert accounting.metrics()["records_dropped"] == 1
                assert accounting.metrics()["usage_missing"] == 1
                assert await accounting.before_attempt("vertex") is not None
                release.set()
                await accounting.flush()
                assert accounting.metrics()["record_failures"] == 2
                assert accounting.metrics()["records_dropped"] == 3
                assert accounting.metrics()["last_record_success_at"] is None
                assert all(row["status"] == "reserved" for row in accounting.ledger.fetch_events())
                assert (await accounting.readiness())["healthy"] is True
        finally:
            release.set()
            await accounting.aclose()
    asyncio.run(run())


def test_known_zero_is_distinct_from_missing_and_real_overrun_is_not_hidden(tmp_path, pg_dsn):
    async def run():
        accounting = service(tmp_path, pg_dsn)
        try:
            async with context(accounting):
                zero = await accounting.before_attempt("vertex")
                accounting.record_attempt(zero, NormalizedUsage())
                large = await accounting.before_attempt("vertex")
                accounting.record_attempt(large, NormalizedUsage(prompt_tokens=2_000_000))
                await accounting.flush()
                with pytest.raises(CostBudgetExceeded):
                    await accounting.before_attempt("vertex")
            rows = accounting.ledger.fetch_events()
            assert rows[0]["status"] == "finalized"
            assert Decimal(rows[0]["estimated_cost_usd"]) == 0
            assert Decimal(rows[1]["estimated_cost_usd"]) == 2
            assert accounting.metrics()["records_written"] == 2
            assert accounting.metrics()["last_record_success_at"] is not None
        finally:
            await accounting.aclose()
    asyncio.run(run())


def test_configuration_failure_does_not_block_explicit_subscription(tmp_path):
    async def run():
        env = cost_env(tmp_path) | {
            "COST_PROVIDER_BILLING_JSON": json.dumps({"vertex": "metered", "ollama": "subscription"}),
            "COST_PRICING_JSON": "{}",
        }
        accounting = build_async_cost_accounting(env)
        try:
            async with context(accounting, "ollama"):
                assert await accounting.before_attempt("ollama") is None
            async with context(accounting):
                with pytest.raises(CostConfigError):
                    await accounting.before_attempt("vertex")
            assert (await accounting.readiness())["healthy"] is False
        finally:
            await accounting.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("billing", ["{}", "[]", "invalid", '{"vertex":"free"}'])
def test_missing_or_invalid_contract_fails_closed(tmp_path, billing):
    async def run():
        accounting = build_async_cost_accounting(cost_env(tmp_path) | {"COST_PROVIDER_BILLING_JSON": billing})
        try:
            async with context(accounting):
                with pytest.raises(CostConfigError):
                    await accounting.before_attempt("vertex")
        finally:
            await accounting.aclose()
    asyncio.run(run())


def test_shutdown_drops_bounded_pending_records_and_restart_keeps_holds(tmp_path, pg_dsn):
    async def run():
        accounting = service(tmp_path, pg_dsn, queue_size=2, shutdown_timeout=0.02)
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def unavailable(*args):
            entered.set()
            try:
                release.wait(2)
                raise OSError("unavailable")
            finally:
                finished.set()
        accounting.gate.finalize_success = unavailable
        try:
            async with context(accounting):
                holds = [await accounting.before_attempt("vertex") for _ in range(3)]
                accounting.record_attempt(holds[0], NormalizedUsage(prompt_tokens=1))
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                for hold in holds[1:]:
                    accounting.record_attempt(hold, NormalizedUsage(prompt_tokens=1))
            await asyncio.wait_for(accounting.aclose(), 0.2)
            assert accounting.metrics()["queue_depth"] == 0
            assert accounting.metrics()["records_dropped"] == 3
            assert accounting.metrics()["records_written"] == 0
        finally:
            release.set()
            while not finished.is_set():
                await asyncio.sleep(0.001)
        restarted = service(tmp_path, pg_dsn)
        try:
            assert (await restarted.readiness())["healthy"] is True
            rows = restarted.ledger.fetch_events()
            assert len(rows) == 3 and all(row["status"] == "reserved" for row in rows)
            assert restarted.metrics()["records_dropped"] == 0
        finally:
            await restarted.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_unknown_usage_reservation_survives_age_prune_and_restart(tmp_path, pg_dsn, backend):
    if backend == "postgres":
        apply_migrations(pg_dsn)
    env = cost_env(tmp_path) | {
        "COST_LEDGER_BACKEND": backend, "COST_PROVIDER_BILLING_JSON": '{"vertex":"metered"}',
        "COST_SHORT_WINDOW_LIMIT_USD": "0.0001",
    }
    if backend == "postgres":
        env["COST_LEDGER_POSTGRES_DSN"] = pg_dsn
    async def run():
        first = build_async_cost_accounting(env)
        first.gate._now_fn = lambda: datetime.now(UTC) - timedelta(days=400)
        async with context(first):
            hold = await first.before_attempt("vertex")
            first.record_attempt(hold, None)
        assert (await first._admit_io(first.ledger.prune))["cost_events"] == 0
        await first.aclose()
        second = build_async_cost_accounting(env)
        try:
            async with context(second):
                with pytest.raises(CostBudgetExceeded):
                    await second.before_attempt("vertex")
        finally:
            await second.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("price", [
    {"input_per_million": "1"}, {"output_per_million": "2"}, {},
])
def test_missing_price_dimension_cannot_become_free(tmp_path, price):
    async def run():
        accounting = build_async_cost_accounting(cost_env(tmp_path) | {
            "COST_PROVIDER_BILLING_JSON": '{"vertex":"metered","ollama":"subscription"}',
            "COST_PRICING_JSON": json.dumps({"models": {"gemini-2.5-flash": {"chat": price}}}),
        })
        try:
            async with context(accounting):
                with pytest.raises(CostConfigError):
                    await accounting.before_attempt("vertex")
            async with context(accounting, "ollama"):
                assert await accounting.before_attempt("ollama") is None
            assert not (tmp_path / "cost.db").exists()
        finally:
            await accounting.aclose()
    asyncio.run(run())