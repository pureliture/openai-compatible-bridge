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
