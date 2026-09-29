from __future__ import annotations

import asyncio
import json
import multiprocessing
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import psycopg
import pytest
from postgres_helpers import pg_dsn

from openai_compatible_bridge.core.cost_tracking import (
    LEDGER_ALLOWED_FIELDS,
    BudgetGate,
    CostBudgetExceeded,
    CostLedgerValidationError,
    CostSubsystemUnhealthy,
    CostTrackingConfig,
    ICostRepository,
    NormalizedUsage,
    PricingCatalog,
    ReconciliationResult,
)
from openai_compatible_bridge.core.postgres_cost_repository import (
    ADVISORY_LOCK_KEY,
    CURRENT_SCHEMA_VERSION,
    SCHEMA,
    PostgresCostRepository,
    apply_migrations,
)

__all__ = ["pg_dsn"]

NOW = datetime(2026, 9, 26, 12, tzinfo=UTC)


@pytest.fixture
def repository(pg_dsn):
    apply_migrations(pg_dsn)
    repo = PostgresCostRepository(pg_dsn, now_fn=lambda: NOW)
    repo.initialize()
    yield repo
    repo.close()


def _event(**overrides):
    return {
        "provider": "synthetic-provider",
        "endpoint": "chat",
        "model": "synthetic-model",
        "status": "finalized",
        "billing_eligible": True,
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 0,
        "embedding_tokens": 7,
        "rerank_units": 0,
        "forecast_cost_usd": Decimal("0.1"),
        "estimated_cost_usd": Decimal("0.1"),
        "currency": "USD",
        "created_at": NOW,
        **overrides,
    }


def _gate(dsn, *, repo=None, now=NOW):
    pricing = PricingCatalog.from_json(json.dumps({
        "source": "synthetic", "version": "1", "currency": "USD",
        "models": {"synthetic-model": {"chat": {"input_per_million": "1000000"}}},
    }))
    return BudgetGate(
        config=CostTrackingConfig(
            enabled=True, short_window_seconds=60,
            short_window_limit_usd=Decimal(1), daily_limit_usd=Decimal(1),
        ),
        ledger=repo or PostgresCostRepository(dsn), pricing=pricing, now_fn=lambda: now,
    )


def _preflight(gate):
    return gate.preflight(
        endpoint="chat", model="synthetic-model", provider="synthetic-provider",
        forecast_usage=NormalizedUsage(prompt_tokens=1),
    )


def _process_preflight(dsn, barrier, results):
    try:
        gate = _gate(dsn)
        barrier.wait(timeout=15)
        try:
            _preflight(gate)
            results.put("accepted")
        except CostBudgetExceeded:
            results.put("blocked")
        finally:
            gate.close()
    except CostSubsystemUnhealthy:
        results.put("failed")


def _settle(repo, reservation_id="res-1", **overrides):
    repo.update_reservation(reservation_id, **{
        "status": "finalized", "billing_eligible": True,
        "usage": NormalizedUsage(prompt_tokens=2, completion_tokens=3),
        "estimated_cost_usd": Decimal("0.2"), "finalized_at": NOW.isoformat(),
        **overrides,
    })


def test_initialize_never_creates_schema(pg_dsn):
    repo = PostgresCostRepository(pg_dsn)
    with pytest.raises(CostSubsystemUnhealthy):
        repo.initialize()
    with psycopg.connect(pg_dsn) as conn:
        assert conn.execute("SELECT to_regnamespace(%s)", (SCHEMA,)).fetchone()[0] is None
    assert apply_migrations(pg_dsn) == CURRENT_SCHEMA_VERSION
    repo.initialize()
    repo.check_health()
    assert isinstance(repo, ICostRepository)
    assert apply_migrations(pg_dsn) == CURRENT_SCHEMA_VERSION


def test_prepare_insert_fetch_precision_and_utc(repository):
    tiny = Decimal("0.000000000000000000000000000000000000123456789123456789")
    prepared = repository.prepare_event(_event(
        estimated_cost_usd=tiny, forecast_cost_usd=tiny,
        created_at="2026-09-26T21:00:00+09:00",
    ))
    assert prepared["created_at"] == "2026-09-26T12:00:00Z"
    assert repository.fetch_events() == []
    assert repository.insert_event(prepared) == prepared
    stored = repository.fetch_events()[0]
    assert set(stored) == set(LEDGER_ALLOWED_FIELDS)
    assert isinstance(stored["estimated_cost_usd"], str)
    assert Decimal(stored["estimated_cost_usd"]) == tiny
    repository.record_event(_event(estimated_cost_usd=tiny))
    expected = Decimal("0.000000000000000000000000000000000000246913578246913578")
    assert repository.sum_estimated_since(NOW) == expected
    assert repository.daily_estimated_spend("2026-09-26") == expected
    summary = repository.usage_summary_since(NOW)
    assert summary.estimated_cost_usd == expected
    assert (summary.prompt_tokens, summary.completion_tokens, summary.total_tokens, summary.event_count) == (4, 6, 24, 2)


def test_reporting_filters_and_boundaries(repository):
    repository.record_event(_event())
    repository.record_event(_event(provider="other", estimated_cost_usd="2", total_tokens=42))
    repository.record_event(_event(status="blocked", billing_eligible=False, estimated_cost_usd="3"))
    repository.record_event(_event(created_at=NOW + timedelta(days=1), estimated_cost_usd="4"))
    repository.record_event(_event(created_at=NOW - timedelta(microseconds=1), estimated_cost_usd="5"))
    assert repository.sum_estimated_since(NOW, statuses=()) == 0
    assert repository.sum_estimated_since(NOW, providers=("other",)) == 2
    assert repository.sum_estimated_since(NOW, providers=()) == Decimal("6.1")
    assert repository.daily_estimated_spend("2026-09-26") == Decimal("7.1")
    assert repository.daily_estimated_spend("2026-09-26", providers=("other",)) == 2
    assert repository.usage_summary_since(NOW, statuses=()).event_count == 0
    assert repository.usage_summary_since(NOW, providers=("other",)).total_tokens == 42
    assert len(repository.fetch_events(limit=2)) == 2


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "-0.1", 0.1, "invalid"])
def test_money_validation_is_sanitized(repository, value):
    with pytest.raises(CostLedgerValidationError):
        repository.record_event(_event(estimated_cost_usd=value))
    assert repository.fetch_events() == []


def test_allowlist_and_timestamp_validation(repository):
    for fields in (
        {"synthetic-secret-key": "synthetic-secret-value"},
        _event(created_at="synthetic-secret-value"),
        _event(prompt_tokens=-1),
        _event(status="reserved", reservation_id=None),
    ):
        with pytest.raises(CostLedgerValidationError) as caught:
            repository.record_event(fields)
        assert "synthetic-secret" not in str(caught.value)
    assert repository.fetch_events() == []


def test_event_and_reservation_idempotence(repository):
    fields = _event(event_id="event-1", reservation_id="res-1", created_at=None)
    first = repository.record_event(fields)
    repository._now_fn = lambda: NOW + timedelta(days=1)
    assert repository.record_event(fields) == first
    assert repository.record_event(first) == first
    with pytest.raises(CostLedgerValidationError):
        repository.record_event({**fields, "estimated_cost_usd": "0.2"})
    with pytest.raises(CostLedgerValidationError):
        repository.record_event({**fields, "event_id": "event-2"})
    with pytest.raises(CostLedgerValidationError):
        repository.record_event({**fields, "created_at": NOW + timedelta(seconds=1)})
    assert len(repository.fetch_events()) == 1


@pytest.mark.parametrize("status", ["finalized", "estimated_only", "released_upstream_error"])
def test_settlement_is_idempotent_but_terminal(repository, status):
    repository.record_event(_event(status="reserved", reservation_id="res-1"))
    _settle(repository, status=status)
    first = repository.fetch_events()[0]
    _settle(repository, status=status, finalized_at=(NOW + timedelta(days=1)).isoformat())
    assert repository.fetch_events()[0] == first
    for changes in (
        {"estimated_cost_usd": Decimal("0.3")},
        {"usage": NormalizedUsage(prompt_tokens=3)},
        {"billing_eligible": False},
        {"status": "released_conflict"},
    ):
        with pytest.raises(CostLedgerValidationError):
            _settle(repository, **{"status": status, **changes})


def test_missing_and_invalid_settlement(repository):
    with pytest.raises(CostLedgerValidationError):
        _settle(repository, "missing")
    repository.record_event(_event(status="reserved", reservation_id="res-1"))
    with pytest.raises(CostLedgerValidationError):
        _settle(repository, status="reserved")
    assert repository.fetch_events()[0]["status"] == "reserved"


@pytest.mark.parametrize("changes", [
    {"estimated_cost_usd": None}, {"billing_eligible": None}, {"status": None},
])
def test_settlement_rejects_missing_required_values(repository, changes):
    repository.record_event(_event(status="reserved", reservation_id="res-1"))
    with pytest.raises(CostLedgerValidationError):
        _settle(repository, **changes)
    assert repository.fetch_events()[0]["status"] == "reserved"


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError])
def test_outer_transaction_rolls_back_base_exceptions(repository, failure):
    with pytest.raises(failure), repository.transaction():
        repository.record_event(_event(event_id="rolled-back"))
        raise failure()
    assert repository.fetch_events() == []
    repository.record_event(_event(event_id="fresh"))
    assert len(repository.fetch_events()) == 1


def test_nested_savepoint_recovers_aborted_transaction(repository):
    with repository.transaction():
        repository.record_event(_event(event_id="outer"))
        with pytest.raises(CostSubsystemUnhealthy), repository.transaction():
            repository.record_event(_event(event_id="inner"))
            repository.connection.execute("SELECT 1 / 0")
        repository.record_event(_event(event_id="after"))
    assert {row["event_id"] for row in repository.fetch_events()} == {"outer", "after"}


def test_swallowed_database_error_cannot_commit_partial_work(repository):
    with pytest.raises(CostSubsystemUnhealthy), repository.transaction():
        repository.record_event(_event())
        try:
            repository.connection.execute("SELECT 1 / 0")
        except psycopg.Error:
            pass
    assert repository.fetch_events() == []


def test_reconciliation_and_prune_are_in_outer_transaction(repository):
    repository.record_event(_event(created_at=NOW - timedelta(days=100)))
    result = ReconciliationResult("2026-09-26", Decimal("0.1"), Decimal("0.3"), Decimal("-0.2"), "mismatch", NOW.isoformat())
    with pytest.raises(RuntimeError), repository.transaction():
        repository.record_reconciliation_result(result)
        repository.prune()
        raise RuntimeError()
    assert repository.latest_reconciliation_result() is None
    assert len(repository.fetch_events()) == 1
    repository.record_reconciliation_result(result)
    assert repository.latest_reconciliation_result()["delta_usd"] == "-0.2"
    replacement = ReconciliationResult("2026-09-26", None, None, None, "pending", NOW.isoformat())
    repository.record_reconciliation_result(replacement)
    assert repository.latest_reconciliation_result()["status"] == "pending"


def test_late_reservations_remain_counted_and_cannot_be_pruned(repository, pg_dsn):
    old = NOW - timedelta(days=500)
    repository.record_event(_event(
        reservation_id="res-1", status="reserved", created_at=old,
        forecast_cost_usd="1", estimated_cost_usd="1",
    ))
    repository.record_event(_event(created_at=old))
    assert repository.sum_estimated_since(NOW, statuses=("reserved", "finalized")) == 1
    assert repository.sum_estimated_since(NOW, statuses=("finalized",)) == 0
    assert repository.sum_estimated_since(NOW) == 0
    assert repository.usage_summary_since(NOW).event_count == 0
    assert repository.daily_estimated_spend(NOW.date()) == 0
    assert repository.prune()["cost_events"] == 1
    assert len(repository.fetch_events()) == 1
    with pytest.raises(CostBudgetExceeded):
        _preflight(_gate(pg_dsn))
    _settle(repository, status="released_test", billing_eligible=False, estimated_cost_usd=Decimal(0))
    assert repository.sum_estimated_since(NOW, statuses=("reserved", "finalized")) == 0
    assert repository.prune()["cost_events"] == 1


@pytest.mark.parametrize("shared_repository", [False, True])
def test_concurrent_independent_gates_are_serialized(repository, pg_dsn, shared_repository):
    barrier = threading.Barrier(6)

    def run(_):
        gate = _gate(pg_dsn, repo=repository if shared_repository else None)
        barrier.wait(timeout=10)
        try:
            _preflight(gate)
            return "accepted"
        except CostBudgetExceeded:
            return "blocked"

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, range(6)))
    assert results.count("accepted") == 1
    assert results.count("blocked") == 5
    assert repository.sum_estimated_since(NOW, statuses=("reserved",)) == 1


def test_independent_process_gates_are_serialized(repository, pg_dsn):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    results = context.Queue()
    workers = [context.Process(target=_process_preflight, args=(pg_dsn, barrier, results)) for _ in range(3)]
    try:
        for worker in workers:
            worker.start()
        outcomes = [results.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert sorted(outcomes) == ["accepted", "blocked", "blocked"]
        assert repository.sum_estimated_since(NOW, statuses=("reserved",)) == 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        results.close()
        results.join_thread()


def test_connection_outage_rolls_back_and_next_call_connects_fresh(repository, pg_dsn):
    with pytest.raises(CostSubsystemUnhealthy), repository.transaction():
        repository.record_event(_event())
        pid = repository.connection.info.backend_pid
        with psycopg.connect(pg_dsn, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        repository.connection.execute("SELECT 1")
    assert repository.fetch_events() == []
    repository.record_event(_event())
    assert len(repository.fetch_events()) == 1


def test_connect_failure_is_bounded_and_sanitized(pg_dsn):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        dsn = psycopg.conninfo.make_conninfo(pg_dsn, port=listener.getsockname()[1], password="synthetic-secret")
        started = time.monotonic()
        with pytest.raises(CostSubsystemUnhealthy) as caught:
            PostgresCostRepository(dsn).initialize()
        assert time.monotonic() - started < 6
        assert "synthetic-secret" not in "".join(traceback.format_exception(caught.value))
        assert dsn not in str(caught.value)


def test_lock_timeout_is_bounded(repository, pg_dsn):
    with psycopg.connect(pg_dsn) as holder:
        holder.execute("SELECT pg_advisory_xact_lock(%s)", (ADVISORY_LOCK_KEY,))
        started = time.monotonic()
        with pytest.raises(CostSubsystemUnhealthy):
            repository.record_event(_event())
        assert 4 <= time.monotonic() - started < 8
    repository.record_event(_event())


def test_statement_timeout_and_transaction_settings(repository):
    with pytest.raises(CostSubsystemUnhealthy), repository.transaction():
        conn = repository.connection
        assert conn.execute("SHOW transaction_isolation").fetchone()["transaction_isolation"] == "read committed"
        assert conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "5s"
        assert conn.execute("SHOW lock_timeout").fetchone()["lock_timeout"] == "5s"
        repository.record_event(_event())
        conn.execute("SET LOCAL statement_timeout = 50")
        conn.execute("SELECT pg_sleep(1)")
    assert repository.fetch_events() == []


def test_database_money_and_unique_constraints(repository, pg_dsn):
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        for value in ("NaN", "Infinity", "-Infinity", "-1"):
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute(
                    "INSERT INTO bridge_cost.cost_events (event_id, created_at, estimated_cost_usd) VALUES ('invalid', %s, %s)",
                    (NOW, Decimal(value)),
                )
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute("INSERT INTO bridge_cost.cost_events (event_id, created_at, status) VALUES ('invalid', %s, 'reserved')", (NOW,))
        repository.record_event(_event(event_id="unique", reservation_id="unique-reservation"))
        for event_id, reservation_id in (("unique", "other"), ("other", "unique-reservation")):
            with pytest.raises(psycopg.errors.UniqueViolation):
                conn.execute(
                    "INSERT INTO bridge_cost.cost_events (event_id, reservation_id, created_at) VALUES (%s, %s, %s)",
                    (event_id, reservation_id, NOW),
                )


def test_aggregate_prune_boundary_and_null_status(repository):
    cutoff = NOW - timedelta(days=13 * 31)
    days = [(cutoff - timedelta(days=1)).date().isoformat(), cutoff.date().isoformat()]
    with repository.transaction():
        for day in days:
            repository.connection.execute(
                "INSERT INTO bridge_cost.cost_daily_aggregates (day, estimated_cost_usd, currency, updated_at) VALUES (%s, 0.1, 'USD', %s)",
                (day, NOW),
            )
            repository.record_reconciliation_result(ReconciliationResult(day, None, None, None, "pending", NOW.isoformat()))
        repository.record_event(_event(created_at=NOW - timedelta(days=90), status=None))
        repository.record_event(_event(created_at=NOW - timedelta(days=90, microseconds=1), status=None))
    assert repository.prune() == {"cost_events": 1, "cost_daily_aggregates": 1, "cost_reconciliation_results": 1}
    assert repository.latest_reconciliation_result()["day"] == days[1]
    assert len(repository.fetch_events()) == 1


def test_nullable_usage_matches_sqlite_reporting(repository):
    repository.record_event(_event(prompt_tokens=2, completion_tokens=None, embedding_tokens=None))
    summary = repository.usage_summary_since(NOW)
    assert summary.prompt_tokens == 2
    assert summary.completion_tokens == 0
    assert summary.total_tokens == 0


@pytest.mark.parametrize("change", [
    "UPDATE bridge_cost.schema_migrations SET checksum = 'bad'",
    "UPDATE bridge_cost.schema_migrations SET version = 999",
    "DELETE FROM bridge_cost.schema_migrations",
    "ALTER TABLE bridge_cost.cost_events DROP COLUMN model",
    "ALTER TABLE bridge_cost.cost_events ALTER COLUMN estimated_cost_usd TYPE NUMERIC(20, 10)",
    "ALTER TABLE bridge_cost.cost_events ALTER COLUMN estimated_cost_usd TYPE DOUBLE PRECISION",
    "ALTER TABLE bridge_cost.cost_events ENABLE ROW LEVEL SECURITY",
])
def test_schema_mismatch_fails_closed(repository, pg_dsn, change):
    with psycopg.connect(pg_dsn) as conn:
        conn.execute(change)
    with pytest.raises(CostSubsystemUnhealthy):
        repository.initialize()
    with pytest.raises(CostSubsystemUnhealthy):
        repository.check_health()
    with pytest.raises(CostSubsystemUnhealthy):
        repository.record_event(_event())
    with pytest.raises(CostSubsystemUnhealthy):
        apply_migrations(pg_dsn)


def test_runtime_privileges_and_read_only_initialize(repository, pg_dsn):
    with psycopg.connect(pg_dsn, autocommit=True) as admin:
        admin.execute("CREATE ROLE cost_runtime LOGIN")
        admin.execute("GRANT USAGE ON SCHEMA bridge_cost TO cost_runtime")
        admin.execute("GRANT SELECT ON ALL TABLES IN SCHEMA bridge_cost TO cost_runtime")
    dsn = psycopg.conninfo.make_conninfo(pg_dsn, user="cost_runtime")
    runtime = PostgresCostRepository(dsn)
    runtime.initialize()
    with pytest.raises(CostSubsystemUnhealthy):
        runtime.check_health()
    with pytest.raises(CostSubsystemUnhealthy):
        apply_migrations(dsn)
    with psycopg.connect(pg_dsn, autocommit=True) as admin:
        admin.execute("GRANT INSERT, UPDATE, DELETE ON bridge_cost.cost_events, bridge_cost.cost_daily_aggregates, bridge_cost.cost_reconciliation_results TO cost_runtime")
    runtime.initialize()
    runtime.check_health()
    runtime.record_event(_event())
    with pytest.raises(CostSubsystemUnhealthy):
        apply_migrations(dsn)
    with psycopg.connect(pg_dsn, autocommit=True) as admin:
        admin.execute("ALTER ROLE cost_runtime SET default_transaction_read_only = on")
    runtime.initialize()
    with pytest.raises(CostSubsystemUnhealthy):
        runtime.check_health()
    with pytest.raises(CostSubsystemUnhealthy):
        runtime.record_event(_event())