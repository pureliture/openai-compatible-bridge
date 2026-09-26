from __future__ import annotations

import contextlib
import socket
from decimal import Decimal

import postgres_helpers
import psycopg
import pytest
from fastapi.testclient import TestClient
from test_cost_backend import NoCallProvider, app_for, cost_env

from openai_compatible_bridge.core.cost_tracking import (
    NormalizedUsage,
    build_cost_accounting_from_env,
)
from openai_compatible_bridge.core.postgres_cost_repository import (
    PostgresCostRepository,
    apply_migrations,
)
from openai_compatible_bridge.main import create_app

pg_dsn = postgres_helpers.pg_dsn


def postgres_env(tmp_path, dsn):
    return cost_env(tmp_path) | {
        "COST_LEDGER_BACKEND": "postgres",
        "COST_LEDGER_POSTGRES_DSN": dsn,
        "COST_ADMIN_ENABLED": "true",
        "COST_ADMIN_API_KEY": "synthetic-admin-key",
    }


@contextlib.contextmanager
def unreachable_dsn():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        yield psycopg.conninfo.make_conninfo(
            host="127.0.0.1", port=listener.getsockname()[1], dbname="bridge_test",
            user="test", password="synthetic-password-must-not-leak",
        )


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    from openai_compatible_bridge import main

    monkeypatch.setattr(main, "BRIDGE_API_KEY", "")


class ChatProvider(NoCallProvider):
    def __init__(self, on_response=lambda: None):
        self.calls = 0
        self.on_response = on_response

    async def generate(self, **kwargs):
        self.calls += 1
        self.on_response()
        return {
            "text": "synthetic response",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        }

    async def stream_chat(self, **kwargs):
        result = await self.generate(**kwargs)
        yield {"delta_text": result["text"], "finish_reason": result["finish_reason"], "usage": result["usage"]}


def chat_app(gate, provider):
    return create_app(
        embedding_client_factory=NoCallProvider,
        chat_client_factory=lambda: provider,
        rerank_client_factory=NoCallProvider,
        ollama_chat_client_factory=NoCallProvider,
        foundry_chat_client_factory=NoCallProvider,
        cost_accounting_factory=lambda: gate,
    )


def chat_payload(stream=False):
    return {"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "test"}], "stream": stream}


@pytest.mark.parametrize("stream", [False, True])
def test_postgres_backend_http_settlement_and_admin(tmp_path, pg_dsn, stream):
    apply_migrations(pg_dsn)
    gate = build_cost_accounting_from_env(postgres_env(tmp_path, pg_dsn))
    provider = ChatProvider()
    with TestClient(chat_app(gate, provider)) as client:
        assert client.get("/readyz").json()["cost_tracking"]["backend"] == "postgres"
        response = client.post("/v1/chat/completions", json=chat_payload(stream))
        assert response.status_code == 200
        if stream:
            assert "data: [DONE]" in response.text
        else:
            assert response.json()["usage"]["total_tokens"] == 11
        assert provider.calls == 1
        status = client.get("/admin/cost/status", headers={"Authorization": "Bearer synthetic-admin-key"})
        assert status.status_code == 200
        assert status.json()["healthy"] is True
    assert not (tmp_path / "cost.db").exists()
    row = gate.ledger.fetch_events()[0]
    assert row["status"] == "finalized"
    assert row["prompt_tokens"] == 5
    assert row["completion_tokens"] == 6
    assert Decimal(row["estimated_cost_usd"]) == Decimal("0.000017")


@pytest.mark.parametrize("stream", [False, True])
def test_postgres_settlement_connection_failure_keeps_hold_and_blocks_next_request(tmp_path, pg_dsn, stream, caplog):
    apply_migrations(pg_dsn)
    gate = build_cost_accounting_from_env(postgres_env(tmp_path, pg_dsn))
    inspector = PostgresCostRepository(pg_dsn)
    reservations = []
    preflight = gate.preflight

    def capture(**kwargs):
        reservation = preflight(**kwargs)
        reservations.append(reservation)
        return reservation

    gate.preflight = capture
    with unreachable_dsn() as unavailable:
        provider = ChatProvider(lambda: setattr(gate.ledger, "_dsn", unavailable))
        with TestClient(chat_app(gate, provider)) as client:
            assert client.post("/v1/chat/completions", json=chat_payload(stream)).status_code == 200
            held = inspector.fetch_events()[0]
            assert held["status"] == "reserved"
            assert held["billing_eligible"] == 1
            assert client.get("/healthz").status_code == 200
            ready = client.get("/readyz")
            assert ready.status_code == 503
            assert ready.json()["cost_tracking"]["database_available"] is False
            assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 503
            assert provider.calls == 1
            assert "synthetic-password-must-not-leak" not in ready.text + caplog.text
            gate.ledger._dsn = pg_dsn
            assert client.get("/readyz").json()["cost_tracking"]["database_available"] is True
            assert client.get("/readyz").status_code == 503
            for _ in range(2):
                gate.finalize_success(reservations[0], NormalizedUsage(prompt_tokens=5, completion_tokens=6, total_tokens=11))
            assert inspector.fetch_events()[0]["status"] == "finalized"
            assert len(inspector.fetch_events()) == 1
            assert client.get("/readyz").status_code == 503
    assert not (tmp_path / "cost.db").exists()


def test_postgres_unreachable_at_startup_never_creates_sqlite(tmp_path, caplog):
    with (
        unreachable_dsn() as dsn,
        TestClient(app_for(lambda: build_cost_accounting_from_env(postgres_env(tmp_path, dsn)))) as client,
    ):
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["cost_tracking"]["backend"] == "postgres"
        response = client.post("/v1/chat/completions", json=chat_payload())
        assert response.status_code == 503
        assert "synthetic-password-must-not-leak" not in ready.text + response.text + caplog.text
    assert not (tmp_path / "cost.db").exists()


def test_runtime_does_not_implicitly_create_schema(tmp_path, pg_dsn):
    with TestClient(app_for(lambda: build_cost_accounting_from_env(postgres_env(tmp_path, pg_dsn)))) as client:
        assert client.get("/readyz").status_code == 503
        assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 503
    with psycopg.connect(pg_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM pg_namespace WHERE nspname = 'bridge_cost'").fetchone()[0] == 0
    assert not (tmp_path / "cost.db").exists()