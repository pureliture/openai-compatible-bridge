from __future__ import annotations

import json
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from openai_compatible_bridge.core.cost_tracking import (
    BudgetGate,
    CostConfigError,
    CostSubsystemUnhealthy,
    CostTrackingConfig,
    DisabledCostAccounting,
    InMemoryCostRepository,
    NormalizedUsage,
    PricingCatalog,
    SQLiteCostRepository,
    build_cost_accounting_from_env,
)
from openai_compatible_bridge.main import create_app


def cost_env(tmp_path):
    return {
        "COST_TRACKING_ENABLED": "true",
        "COST_LEDGER_PATH": str(tmp_path / "cost.db"),
        "COST_PRICING_JSON": json.dumps({"models": {
            "gemini-2.5-flash": {"chat": {"input_per_million": "1", "output_per_million": "2"}},
        }}),
        "COST_SHORT_WINDOW_SECONDS": "60",
        "COST_SHORT_WINDOW_LIMIT_USD": "1",
        "COST_DAILY_LIMIT_USD": "10",
    }


def test_sqlite_remains_default(tmp_path):
    gate = build_cost_accounting_from_env(cost_env(tmp_path))
    try:
        assert gate.config.backend == "sqlite"
        assert isinstance(gate.ledger, SQLiteCostRepository)
        assert gate.readiness()["healthy"] is True
    finally:
        gate.close()


def test_postgres_config_does_not_require_sqlite_path_and_hides_dsn(tmp_path):
    env = cost_env(tmp_path)
    del env["COST_LEDGER_PATH"]
    env.update(COST_LEDGER_BACKEND="postgres", COST_LEDGER_POSTGRES_DSN="postgresql://user:secret@localhost/test")
    config = CostTrackingConfig.from_env(env)
    assert config.backend == "postgres"
    assert config.ledger_path is None
    assert config.postgres_dsn == env["COST_LEDGER_POSTGRES_DSN"]
    assert "secret" not in repr(config)


@pytest.mark.parametrize("changes", [
    {"COST_LEDGER_BACKEND": "unknown"},
    {"COST_LEDGER_BACKEND": "postgres"},
    {"COST_LEDGER_POSTGRES_DSN": "postgresql://unused/test"},
])
def test_unsafe_backend_configuration_rejected(tmp_path, changes):
    with pytest.raises(CostConfigError):
        CostTrackingConfig.from_env(cost_env(tmp_path) | changes)


def test_explicit_empty_env_does_not_use_process_configuration(monkeypatch):
    monkeypatch.setenv("COST_TRACKING_ENABLED", "true")
    monkeypatch.setenv("COST_ADMIN_ENABLED", "true")
    accounting = build_cost_accounting_from_env({})
    assert isinstance(accounting, DisabledCostAccounting)
    assert accounting.enabled is False


def test_startup_database_failure_is_latched_and_sanitized(tmp_path, monkeypatch):
    def fail(_self):
        raise OSError("password=secret simulated disk failure")

    monkeypatch.setattr(SQLiteCostRepository, "initialize", fail)
    gate = build_cost_accounting_from_env(cost_env(tmp_path))
    assert gate.enabled is True
    assert gate.readiness()["healthy"] is False
    assert "secret" not in json.dumps(gate.readiness())
    with pytest.raises(CostSubsystemUnhealthy) as error:
        gate.preflight(endpoint="chat", model="gemini-2.5-flash", forecast_usage=NormalizedUsage(prompt_tokens=1))
    assert "secret" not in str(error.value)


def test_failed_settlement_retains_reservation_and_latches_readiness(tmp_path, monkeypatch, caplog):
    config = CostTrackingConfig.from_env(cost_env(tmp_path))
    ledger = InMemoryCostRepository()
    gate = BudgetGate(config=config, ledger=ledger, pricing=PricingCatalog.from_config(config))
    reservation = gate.preflight(endpoint="chat", model="gemini-2.5-flash", forecast_usage=NormalizedUsage(prompt_tokens=100))
    update = ledger.update_reservation

    def fail(*args, **kwargs):
        raise OSError("password=secret failed update")

    monkeypatch.setattr(ledger, "update_reservation", fail)
    gate.finalize_success(reservation, NormalizedUsage(prompt_tokens=50))
    assert ledger.fetch_events()[0]["status"] == "reserved"
    assert Decimal(ledger.fetch_events()[0]["estimated_cost_usd"]) == Decimal("0.0001")
    readiness = gate.readiness()
    assert readiness["database_available"] is True
    assert readiness["healthy"] is False
    assert "secret" not in json.dumps(readiness) + caplog.text
    assert reservation.reservation_id in caplog.text
    with pytest.raises(CostSubsystemUnhealthy):
        gate.preflight(endpoint="chat", model="gemini-2.5-flash", forecast_usage=NormalizedUsage(prompt_tokens=1))
    monkeypatch.setattr(ledger, "update_reservation", update)
    gate.finalize_success(reservation, NormalizedUsage(prompt_tokens=50))
    assert ledger.fetch_events()[0]["status"] == "finalized"
    assert gate.readiness()["healthy"] is False
    gate.close()


class NoCallProvider:
    async def close(self):
        pass

    async def generate(self, **kwargs):
        pytest.fail("database failure must block upstream calls")

    async def embed(self, **kwargs):
        pytest.fail("database failure must block upstream calls")

    async def rank(self, **kwargs):
        pytest.fail("database failure must block upstream calls")


def app_for(factory):
    return create_app(
        embedding_client_factory=NoCallProvider,
        chat_client_factory=NoCallProvider,
        rerank_client_factory=NoCallProvider,
        ollama_chat_client_factory=NoCallProvider,
        foundry_chat_client_factory=NoCallProvider,
        cost_accounting_factory=factory,
    )


@pytest.mark.parametrize("path,payload", [
    ("/v1/chat/completions", {"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "test"}]}),
    ("/v1/chat/completions", {"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "test"}], "stream": True}),
    ("/v1/embeddings", {"model": "gemini-embedding-001", "input": "test"}),
    ("/v1/rerank", {"model": "semantic-ranker-512@latest", "query": "test", "documents": ["test"]}),
])
def test_liveness_survives_startup_failure_but_paid_requests_do_not(tmp_path, monkeypatch, path, payload):
    from openai_compatible_bridge import main

    monkeypatch.setattr(main, "BRIDGE_API_KEY", "")
    def fail(_self):
        raise OSError("password=secret startup unavailable")

    monkeypatch.setattr(SQLiteCostRepository, "initialize", fail)
    with TestClient(app_for(lambda: build_cost_accounting_from_env(cost_env(tmp_path)))) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["cost_tracking"]["database_available"] is False
        assert "secret" not in ready.text
        response = client.post(path, json=payload)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "cost_tracking_unavailable"
        assert "secret" not in response.text


def test_disabled_cost_accounting_is_ready():
    with TestClient(app_for(DisabledCostAccounting)) as client:
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["cost_tracking"]["enabled"] is False