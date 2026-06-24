from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from openai_compatible_bridge.core.cost_tracking import (
    LEDGER_ALLOWED_FIELDS,
    BillingExportPermissionDenied,
    BudgetGate,
    CostAccountingStore,
    CostBudgetExceeded,
    CostConfigError,
    CostEstimator,
    CostLedger,
    CostLedgerValidationError,
    CostSubsystemHealth,
    CostSubsystemUnhealthy,
    CostTrackingConfig,
    NormalizedUsage,
    PricingCatalog,
    ReconciliationJob,
    ICostRepository,
    SQLiteCostRepository,
    InMemoryCostRepository,
)


def _pricing_json() -> str:
    return json.dumps(
        {
            "source": "unit-test",
            "version": "2026-06-22",
            "currency": "USD",
            "models": {
                "chat-model": {
                    "chat": {
                        "input_per_million": "0.10",
                        "output_per_million": "0.30",
                    }
                },
                "embedding-model": {
                    "embeddings": {
                        "embedding_per_million": "0.20",
                    }
                },
                "rerank-model": {
                    "rerank": {
                        "rerank_per_unit": "0.005",
                    }
                },
            },
        }
    )


def test_config_disabled_has_no_budget_defaults():
    config = CostTrackingConfig.from_env({})

    assert config.enabled is False
    assert config.short_window_limit_usd is None
    assert config.daily_limit_usd is None
    assert config.ledger_path is None


def test_config_enabled_requires_ledger_pricing_and_budgets(tmp_path):
    env = {
        "COST_TRACKING_ENABLED": "true",
        "COST_LEDGER_PATH": str(tmp_path / "cost.db"),
        "COST_PRICING_JSON": _pricing_json(),
        "COST_SHORT_WINDOW_SECONDS": "60",
        "COST_SHORT_WINDOW_LIMIT_USD": "1.25",
        "COST_DAILY_LIMIT_USD": "10.50",
    }

    config = CostTrackingConfig.from_env(env)

    assert config.enabled is True
    assert config.ledger_path == tmp_path / "cost.db"
    assert config.short_window_seconds == 60
    assert config.short_window_limit_usd == Decimal("1.25")
    assert config.daily_limit_usd == Decimal("10.50")


def test_config_enabled_fails_closed_when_budget_missing(tmp_path):
    env = {
        "COST_TRACKING_ENABLED": "true",
        "COST_LEDGER_PATH": str(tmp_path / "cost.db"),
        "COST_PRICING_JSON": _pricing_json(),
        "COST_SHORT_WINDOW_SECONDS": "60",
        "COST_SHORT_WINDOW_LIMIT_USD": "1.25",
    }

    with pytest.raises(CostConfigError, match="COST_DAILY_LIMIT_USD"):
        CostTrackingConfig.from_env(env)


def test_admin_enabled_requires_separate_admin_key(tmp_path):
    env = {
        "COST_TRACKING_ENABLED": "true",
        "COST_ADMIN_ENABLED": "true",
        "COST_LEDGER_PATH": str(tmp_path / "cost.db"),
        "COST_PRICING_JSON": _pricing_json(),
        "COST_SHORT_WINDOW_SECONDS": "60",
        "COST_SHORT_WINDOW_LIMIT_USD": "1.25",
        "COST_DAILY_LIMIT_USD": "10.50",
    }

    with pytest.raises(CostConfigError, match="COST_ADMIN_API_KEY"):
        CostTrackingConfig.from_env(env)


def test_pricing_catalog_estimates_chat_embeddings_and_rerank():
    catalog = PricingCatalog.from_json(_pricing_json())
    estimator = CostEstimator(catalog)

    chat_cost = estimator.estimate(
        model="chat-model",
        endpoint="chat",
        usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )
    embedding_cost = estimator.estimate(
        model="embedding-model",
        endpoint="embeddings",
        usage=NormalizedUsage(embedding_tokens=2000, total_tokens=2000),
    )
    rerank_cost = estimator.estimate(
        model="rerank-model",
        endpoint="rerank",
        usage=NormalizedUsage(rerank_units=3),
    )

    assert chat_cost == Decimal("0.00025")
    assert embedding_cost == Decimal("0.0004")
    assert rerank_cost == Decimal("0.015")


def test_pricing_catalog_rejects_negative_prices():
    raw = json.loads(_pricing_json())
    raw["models"]["chat-model"]["chat"]["input_per_million"] = "-0.01"

    with pytest.raises(CostConfigError, match="input_per_million"):
        PricingCatalog.from_json(json.dumps(raw))


def test_pricing_catalog_missing_entry_fails_closed():
    catalog = PricingCatalog.from_json(_pricing_json())
    estimator = CostEstimator(catalog)

    with pytest.raises(CostConfigError, match="missing pricing"):
        estimator.estimate(
            model="chat-model",
            endpoint="embeddings",
            usage=NormalizedUsage(embedding_tokens=1),
        )


def test_pricing_catalog_uses_explicit_ollama_wildcard_for_dynamic_chat_models():
    catalog = PricingCatalog.from_json(
        json.dumps(
            {
                "source": "unit-test",
                "version": "2026-06-23",
                "currency": "USD",
                "models": {
                    "ollama:*": {
                        "chat": {
                            "input_per_million": "0.25",
                            "output_per_million": "0.50",
                        }
                    }
                },
            }
        )
    )
    estimator = CostEstimator(catalog)

    cost = estimator.estimate(
        model="ollama:qwen3.5:cloud",
        endpoint="chat",
        usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=2000),
    )

    assert cost == Decimal("0.00125")


def test_pricing_catalog_exact_ollama_model_overrides_wildcard():
    catalog = PricingCatalog.from_json(
        json.dumps(
            {
                "source": "unit-test",
                "version": "2026-06-23",
                "currency": "USD",
                "models": {
                    "ollama:*": {
                        "chat": {
                            "input_per_million": "10.00",
                            "output_per_million": "10.00",
                        }
                    },
                    "ollama:qwen3.5:cloud": {
                        "chat": {
                            "input_per_million": "0.25",
                            "output_per_million": "0.50",
                        }
                    },
                },
            }
        )
    )
    estimator = CostEstimator(catalog)

    cost = estimator.estimate(
        model="ollama:qwen3.5:cloud",
        endpoint="chat",
        usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=2000),
    )

    assert cost == Decimal("0.00125")


def test_ledger_creates_allowlist_schema(tmp_path):
    ledger = CostLedger(tmp_path / "cost.db")
    ledger.initialize()

    rows = ledger.connection.execute("PRAGMA table_info(cost_events)").fetchall()
    columns = [row["name"] for row in rows]

    assert columns == list(LEDGER_ALLOWED_FIELDS)
    assert ledger.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cost_daily_aggregates'"
    ).fetchone()
    assert ledger.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cost_reconciliation_results'"
    ).fetchone()


def test_ledger_persists_only_allowlisted_fields(tmp_path):
    ledger = CostLedger(tmp_path / "cost.db")
    row = ledger.record_event(
        {
            "reservation_id": "res-1",
            "internal_request_id": "req-1",
            "endpoint": "chat",
            "model": "chat-model",
            "status": "finalized",
            "billing_eligible": True,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "forecast_cost_usd": Decimal("0.00001"),
            "estimated_cost_usd": Decimal("0.00002"),
            "currency": "USD",
            "pricing_source": "unit-test",
            "pricing_version": "2026-06-22",
            "created_at": "2026-06-22T00:00:00Z",
            "finalized_at": "2026-06-22T00:00:01Z",
            "reconciliation_status": "pending",
        }
    )

    stored = ledger.fetch_events()[0]

    assert set(stored) == set(LEDGER_ALLOWED_FIELDS)
    assert row["event_id"].startswith("costevt-")
    assert stored["billing_eligible"] == 1
    assert stored["estimated_cost_usd"] == "0.00002"
    assert stored["endpoint"] == "chat"


def test_ledger_rejects_non_allowlisted_payload_fields(tmp_path):
    ledger = CostLedger(tmp_path / "cost.db")

    with pytest.raises(CostLedgerValidationError, match="prompt"):
        ledger.record_event({"endpoint": "chat", "prompt": "do not persist me"})

    with pytest.raises(CostLedgerValidationError, match="raw_provider_response"):
        ledger.record_event({"endpoint": "chat", "raw_provider_response": {"secret": True}})

    assert ledger.fetch_events() == []


def test_ledger_retention_prunes_old_request_events(tmp_path):
    now = datetime(2026, 6, 22, tzinfo=timezone.utc)
    ledger = CostLedger(tmp_path / "cost.db", request_retention_days=90)
    ledger.record_event(
        {
            "endpoint": "chat",
            "model": "chat-model",
            "status": "finalized",
            "billing_eligible": True,
            "estimated_cost_usd": Decimal("1.00"),
            "created_at": "2026-02-01T00:00:00Z",
        }
    )
    ledger.record_event(
        {
            "endpoint": "chat",
            "model": "chat-model",
            "status": "finalized",
            "billing_eligible": True,
            "estimated_cost_usd": Decimal("2.00"),
            "created_at": "2026-06-01T00:00:00Z",
        }
    )

    deleted = ledger.prune(now=now)

    assert deleted["cost_events"] == 1
    events = ledger.fetch_events()
    assert len(events) == 1
    assert events[0]["estimated_cost_usd"] == "2.00"


def test_ledger_failure_marks_cost_subsystem_unhealthy(tmp_path, monkeypatch):
    ledger = CostLedger(tmp_path / "cost.db")
    health = CostSubsystemHealth()
    store = CostAccountingStore(ledger, health)

    def fail_write(fields):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(ledger, "record_event", fail_write)

    with pytest.raises(CostSubsystemUnhealthy, match="disk full"):
        store.record_event({"endpoint": "chat"})

    assert health.healthy is False
    with pytest.raises(CostSubsystemUnhealthy, match="disk full"):
        store.ensure_healthy()


def _enabled_config(tmp_path, *, short_limit="1.00", daily_limit="10.00") -> CostTrackingConfig:
    return CostTrackingConfig.from_env(
        {
            "COST_TRACKING_ENABLED": "true",
            "COST_LEDGER_PATH": str(tmp_path / "cost.db"),
            "COST_PRICING_JSON": _pricing_json(),
            "COST_SHORT_WINDOW_SECONDS": "60",
            "COST_SHORT_WINDOW_LIMIT_USD": short_limit,
            "COST_DAILY_LIMIT_USD": daily_limit,
        }
    )


def test_budget_gate_reserves_and_finalizes_success(tmp_path):
    config = _enabled_config(tmp_path)
    ledger = CostLedger(tmp_path / "cost.db")
    gate = BudgetGate(config=config, ledger=ledger, pricing=PricingCatalog.from_json(_pricing_json()))

    reservation = gate.preflight(
        endpoint="chat",
        model="chat-model",
        forecast_usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )
    gate.finalize_success(
        reservation,
        NormalizedUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    event = ledger.fetch_events()[0]
    assert event["status"] == "finalized"
    assert event["billing_eligible"] == 1
    assert event["prompt_tokens"] == 10
    assert event["completion_tokens"] == 20
    assert event["estimated_cost_usd"] == "0.000007"


def test_budget_gate_blocks_short_window_before_upstream(tmp_path):
    config = _enabled_config(tmp_path, short_limit="0.0001", daily_limit="10.00")
    ledger = CostLedger(tmp_path / "cost.db")
    gate = BudgetGate(config=config, ledger=ledger, pricing=PricingCatalog.from_json(_pricing_json()))

    with pytest.raises(CostBudgetExceeded) as excinfo:
        gate.preflight(
            endpoint="chat",
            model="chat-model",
            forecast_usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )

    block = excinfo.value.block
    assert block.limit_type == "short_window"
    assert block.configured_limit == Decimal("0.0001")
    event = ledger.fetch_events()[0]
    assert event["status"] == "blocked"
    assert event["billing_eligible"] == 0
    assert event["limit_type"] == "short_window"


def test_budget_gate_blocks_daily_window(tmp_path):
    config = _enabled_config(tmp_path, short_limit="10.00", daily_limit="0.0001")
    ledger = CostLedger(tmp_path / "cost.db")
    gate = BudgetGate(config=config, ledger=ledger, pricing=PricingCatalog.from_json(_pricing_json()))

    with pytest.raises(CostBudgetExceeded) as excinfo:
        gate.preflight(
            endpoint="chat",
            model="chat-model",
            forecast_usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )

    assert excinfo.value.block.limit_type == "daily"
    assert ledger.fetch_events()[0]["limit_type"] == "daily"


def test_budget_gate_release_nonbillable_removes_reserved_spend(tmp_path):
    config = _enabled_config(tmp_path, short_limit="0.0003", daily_limit="10.00")
    ledger = CostLedger(tmp_path / "cost.db")
    gate = BudgetGate(config=config, ledger=ledger, pricing=PricingCatalog.from_json(_pricing_json()))

    reservation = gate.preflight(
        endpoint="chat",
        model="chat-model",
        forecast_usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )
    gate.release_nonbillable(reservation, "upstream_error")
    second = gate.preflight(
        endpoint="chat",
        model="chat-model",
        forecast_usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )

    events = ledger.fetch_events()
    assert events[0]["status"] == "released_upstream_error"
    assert events[0]["billing_eligible"] == 0
    assert second.reservation_id != reservation.reservation_id


def test_budget_gate_estimated_only_keeps_forecast_spend(tmp_path):
    config = _enabled_config(tmp_path)
    ledger = CostLedger(tmp_path / "cost.db")
    gate = BudgetGate(config=config, ledger=ledger, pricing=PricingCatalog.from_json(_pricing_json()))

    reservation = gate.preflight(
        endpoint="chat",
        model="chat-model",
        forecast_usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )
    gate.finalize_estimated_only(reservation, "missing_usage")

    event = ledger.fetch_events()[0]
    assert event["status"] == "estimated_only"
    assert event["billing_eligible"] == 1
    assert event["estimated_cost_usd"] == event["forecast_cost_usd"]


class _FakeBillingAdapter:
    def __init__(self, value=None, exc=None):
        self.value = value
        self.exc = exc
        self.days = []

    def daily_cost(self, day):
        self.days.append(day)
        if self.exc is not None:
            raise self.exc
        return self.value


def _ledger_with_reconciliation_event(tmp_path, *, amount="1.00") -> CostLedger:
    ledger = CostLedger(tmp_path / "cost.db")
    ledger.record_event(
        {
            "endpoint": "chat",
            "model": "chat-model",
            "status": "finalized",
            "billing_eligible": True,
            "estimated_cost_usd": Decimal(amount),
            "created_at": "2026-06-21T12:00:00Z",
        }
    )
    return ledger


def test_reconciliation_matched_persists_result(tmp_path):
    ledger = _ledger_with_reconciliation_event(tmp_path, amount="1.00")
    adapter = _FakeBillingAdapter(value="1.00")
    job = ReconciliationJob(ledger=ledger, billing_adapter=adapter)

    result = job.reconcile_day("2026-06-21")

    assert result.status == "matched"
    assert result.wrapper_estimated_cost_usd == Decimal("1.00")
    assert result.billing_export_cost == Decimal("1.00")
    assert result.delta_usd == Decimal("0.00")
    assert ledger.latest_reconciliation_result()["status"] == "matched"


def test_reconciliation_mismatch_persists_delta(tmp_path):
    ledger = _ledger_with_reconciliation_event(tmp_path, amount="1.00")
    job = ReconciliationJob(ledger=ledger, billing_adapter=_FakeBillingAdapter(value="1.50"))

    result = job.reconcile_day("2026-06-21")

    assert result.status == "mismatch"
    assert result.delta_usd == Decimal("-0.50")
    latest = ledger.latest_reconciliation_result()
    assert latest["status"] == "mismatch"
    assert latest["delta_usd"] == "-0.50"


def test_reconciliation_pending_when_billing_row_missing(tmp_path):
    ledger = _ledger_with_reconciliation_event(tmp_path, amount="1.00")
    job = ReconciliationJob(ledger=ledger, billing_adapter=_FakeBillingAdapter(value=None))

    result = job.reconcile_day("2026-06-21")

    assert result.status == "pending"
    assert result.billing_export_cost is None
    assert ledger.latest_reconciliation_result()["status"] == "pending"


def test_reconciliation_unavailable_without_adapter(tmp_path):
    ledger = _ledger_with_reconciliation_event(tmp_path, amount="1.00")
    job = ReconciliationJob(ledger=ledger, billing_adapter=None)

    result = job.reconcile_day("2026-06-21")

    assert result.status == "unavailable"
    assert "not configured" in result.error_message
    assert ledger.latest_reconciliation_result()["status"] == "unavailable"


def test_reconciliation_permission_denied_is_error_not_request_path_failure(tmp_path):
    ledger = _ledger_with_reconciliation_event(tmp_path, amount="1.00")
    job = ReconciliationJob(
        ledger=ledger,
        billing_adapter=_FakeBillingAdapter(exc=BillingExportPermissionDenied("bigquery.jobs.create denied")),
    )

    result = job.reconcile_day("2026-06-21")

    assert result.status == "error"
    assert "permission denied" in result.error_message
    latest = ledger.latest_reconciliation_result()
    assert latest["status"] == "error"
    assert "bigquery.jobs.create denied" in latest["error_message"]


def test_in_memory_cost_repository_interface_compliance():
    repo = InMemoryCostRepository()
    assert isinstance(repo, ICostRepository)
    assert isinstance(repo, SQLiteCostRepository)


def test_in_memory_cost_repository_budget_gate_integration(tmp_path):
    config = _enabled_config(tmp_path, short_limit="0.0003", daily_limit="10.00")
    repo = InMemoryCostRepository()
    gate = BudgetGate(config=config, ledger=repo, pricing=PricingCatalog.from_json(_pricing_json()))

    reservation = gate.preflight(
        endpoint="chat",
        model="chat-model",
        forecast_usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )
    gate.finalize_success(
        reservation,
        NormalizedUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    event = repo.fetch_events()[0]
    assert event["status"] == "finalized"
    assert event["billing_eligible"] == 1
    assert event["estimated_cost_usd"] == "0.000007"

    with pytest.raises(CostBudgetExceeded):
        gate.preflight(
            endpoint="chat",
            model="chat-model",
            forecast_usage=NormalizedUsage(prompt_tokens=2000, completion_tokens=1000, total_tokens=3000),
        )


import contextlib
import copy
from typing import Mapping, Any, ContextManager, Iterator
from openai_compatible_bridge.core.cost_tracking import (
    ReconciliationResult,
    CostReservation,
    CostTrackingError,
    CostLedgerValidationError,
    _iso,
    _normalize_ledger_value,
    _coerce_day,
)

class MockCostRepository(ICostRepository):
    def __init__(self, now_fn=None) -> None:
        self.events: list[dict[str, Any]] = []
        self.reconciliation_results: dict[str, dict[str, Any]] = {}
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._backup_events = None
        self._backup_reconciliation = None
        self._in_transaction = False

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        was_in_trans = self._in_transaction
        if not was_in_trans:
            self._in_transaction = True
            self._backup_events = copy.deepcopy(self.events)
            self._backup_reconciliation = copy.deepcopy(self.reconciliation_results)
        try:
            yield
        except Exception:
            if not was_in_trans:
                self.events = self._backup_events
                self.reconciliation_results = self._backup_reconciliation
            raise
        finally:
            if not was_in_trans:
                self._in_transaction = False
                self._backup_events = None
                self._backup_reconciliation = None

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def record_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        with self.transaction():
            return self.insert_event(fields)

    def prepare_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        import uuid
        unknown = sorted(set(fields) - set(LEDGER_ALLOWED_FIELDS))
        if unknown:
            raise CostLedgerValidationError(f"cost ledger event has non-allowlisted fields: {unknown}")
        row = {name: fields.get(name) for name in LEDGER_ALLOWED_FIELDS}
        row["event_id"] = row["event_id"] or f"costevt-{uuid.uuid4().hex}"
        row["created_at"] = row["created_at"] or _iso(self._now_fn())
        return {name: _normalize_ledger_value(name, value) for name, value in row.items()}

    def insert_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        normalized = self.prepare_event(fields)
        self.events.append(normalized)
        return normalized

    def fetch_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        sorted_events = sorted(self.events, key=lambda e: e["created_at"])
        return sorted_events[:limit]

    def sum_estimated_since(self, cutoff: datetime, statuses: tuple[str, ...] | None = None) -> Decimal:
        cutoff_str = _iso(cutoff)
        total = Decimal("0")
        for e in self.events:
            if e["created_at"] < cutoff_str:
                continue
            if e["billing_eligible"] != 1:
                continue
            if statuses is not None and e["status"] not in statuses:
                continue
            if e["estimated_cost_usd"] is not None:
                total += Decimal(str(e["estimated_cost_usd"]))
        return total

    def daily_estimated_spend(self, day: str | date) -> Decimal:
        day_value = _coerce_day(day)
        start = datetime(day_value.year, day_value.month, day_value.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        start_str = _iso(start)
        end_str = _iso(end)
        total = Decimal("0")
        for e in self.events:
            if e["created_at"] < start_str or e["created_at"] >= end_str:
                continue
            if e["billing_eligible"] != 1:
                continue
            if e["status"] not in {"reserved", "finalized", "estimated_only"}:
                continue
            if e["estimated_cost_usd"] is not None:
                total += Decimal(str(e["estimated_cost_usd"]))
        return total

    def record_reconciliation_result(self, result: ReconciliationResult) -> None:
        self.reconciliation_results[result.day] = {
            "day": result.day,
            "wrapper_estimated_cost_usd": None if result.wrapper_estimated_cost_usd is None else str(result.wrapper_estimated_cost_usd),
            "billing_export_cost": None if result.billing_export_cost is None else str(result.billing_export_cost),
            "delta_usd": None if result.delta_usd is None else str(result.delta_usd),
            "status": result.status,
            "checked_at": result.checked_at,
            "error_message": result.error_message,
        }

    def latest_reconciliation_result(self) -> dict[str, Any] | None:
        if not self.reconciliation_results:
            return None
        sorted_results = sorted(self.reconciliation_results.values(), key=lambda r: r["checked_at"], reverse=True)
        return sorted_results[0]

    def prune(self, *, now: datetime | None = None) -> dict[str, int]:
        current = now or self._now_fn()
        request_cutoff = _iso(current - timedelta(days=90))
        aggregate_cutoff = (current - timedelta(days=13 * 31)).date().isoformat()
        new_events = []
        pruned_events = 0
        for e in self.events:
            if e["created_at"] < request_cutoff:
                pruned_events += 1
            else:
                new_events.append(e)
        self.events = new_events
        new_recon = {}
        pruned_recon = 0
        for day, r in self.reconciliation_results.items():
            if day < aggregate_cutoff:
                pruned_recon += 1
            else:
                new_recon[day] = r
        self.reconciliation_results = new_recon
        return {
            "cost_events": pruned_events,
            "cost_daily_aggregates": 0,
            "cost_reconciliation_results": pruned_recon,
        }

    def update_reservation(
        self,
        reservation_id: str,
        *,
        status: str,
        billing_eligible: bool,
        usage: NormalizedUsage,
        estimated_cost_usd: Decimal,
        finalized_at: str,
    ) -> None:
        total_tokens = usage.total_tokens or usage.prompt_tokens + usage.completion_tokens + usage.embedding_tokens
        for e in self.events:
            if e["reservation_id"] == reservation_id:
                e["status"] = status
                e["billing_eligible"] = int(billing_eligible)
                e["prompt_tokens"] = usage.prompt_tokens
                e["completion_tokens"] = usage.completion_tokens
                e["total_tokens"] = total_tokens
                e["embedding_tokens"] = usage.embedding_tokens
                e["rerank_units"] = usage.rerank_units
                e["estimated_cost_usd"] = str(estimated_cost_usd)
                e["finalized_at"] = finalized_at
                return
        raise CostTrackingError(f"reservation not found: {reservation_id}")


def test_mock_cost_repository_budget_gate_integration(tmp_path):
    config = _enabled_config(tmp_path, short_limit="0.0003", daily_limit="10.00")
    repo = MockCostRepository()
    gate = BudgetGate(config=config, ledger=repo, pricing=PricingCatalog.from_json(_pricing_json()))

    reservation = gate.preflight(
        endpoint="chat",
        model="chat-model",
        forecast_usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
    )
    gate.finalize_success(
        reservation,
        NormalizedUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    event = repo.fetch_events()[0]
    assert event["status"] == "finalized"
    assert event["billing_eligible"] == 1
    assert event["estimated_cost_usd"] == "0.000007"

    with pytest.raises(CostBudgetExceeded):
        gate.preflight(
            endpoint="chat",
            model="chat-model",
            forecast_usage=NormalizedUsage(prompt_tokens=2000, completion_tokens=1000, total_tokens=3000),
        )


def test_mock_cost_repository_transaction_rollback():
    repo = MockCostRepository()
    repo.insert_event({
        "reservation_id": "res-existing",
        "internal_request_id": "req-existing",
        "endpoint": "chat",
        "model": "chat-model",
        "status": "reserved",
        "billing_eligible": True,
        "forecast_cost_usd": Decimal("0.001"),
        "estimated_cost_usd": Decimal("0.001"),
        "currency": "USD",
        "pricing_source": "unit-test",
        "pricing_version": "1.0",
        "created_at": None,
        "finalized_at": None,
        "reconciliation_status": None,
    })

    with pytest.raises(ValueError, match="force rollback"):
        with repo.transaction():
            repo.insert_event({
                "reservation_id": "res-new",
                "internal_request_id": "req-new",
                "endpoint": "chat",
                "model": "chat-model",
                "status": "reserved",
                "billing_eligible": True,
                "forecast_cost_usd": Decimal("0.002"),
                "estimated_cost_usd": Decimal("0.002"),
                "currency": "USD",
                "pricing_source": "unit-test",
                "pricing_version": "1.0",
                "created_at": None,
                "finalized_at": None,
                "reconciliation_status": None,
            })
            raise ValueError("force rollback")

    assert len(repo.events) == 1
    assert repo.events[0]["reservation_id"] == "res-existing"

