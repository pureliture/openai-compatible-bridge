from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
from decimal import Decimal

import httpx
import postgres_helpers
import psycopg
import pytest
from fastapi.testclient import TestClient
from test_cost_backend import NoCallProvider, cost_env

from openai_compatible_bridge import main
from openai_compatible_bridge.core.async_cost import AsyncCostAccounting, build_async_cost_accounting
from openai_compatible_bridge.core.postgres_cost_repository import (
    ADVISORY_LOCK_KEY,
    PostgresCostRepository,
    apply_migrations,
)
from openai_compatible_bridge.main import create_app

pg_dsn = postgres_helpers.pg_dsn


def postgres_env(tmp_path, dsn):
    return cost_env(tmp_path) | {
        "COST_LEDGER_BACKEND": "postgres", "COST_LEDGER_POSTGRES_DSN": dsn,
        "COST_ADMIN_ENABLED": "true", "COST_ADMIN_API_KEY": "synthetic-admin-key",
        "COST_PROVIDER_BILLING_JSON": '{"vertex":"metered","ollama":"subscription","foundry":"nonbillable"}',
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


class HTTPChatProvider(NoCallProvider):
    def __init__(self, on_response=lambda: None, *, usage=True):
        self.calls = 0
        self.on_response = on_response
        self.usage = usage
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(self.respond))

    def respond(self, request):
        self.calls += 1
        self.on_response()
        result = {"text": "synthetic response", "finish_reason": "stop", "delta_text": "synthetic response"}
        if self.usage:
            result["usage"] = {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
        if request.url.path == "/stream":
            return httpx.Response(200, text=f"data: {json.dumps(result)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json=result)

    async def generate(self, **kwargs):
        response = await self.http.post("https://synthetic.invalid/chat")
        return response.json()

    async def stream_chat(self, **kwargs):
        async with self.http.stream("POST", "https://synthetic.invalid/stream") as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    yield json.loads(line[6:])

    async def close(self):
        await self.http.aclose()


def chat_app(accounting, provider, subscription=None):
    return create_app(
        embedding_client_factory=NoCallProvider, chat_client_factory=lambda: provider,
        rerank_client_factory=NoCallProvider, ollama_chat_client_factory=lambda: subscription or HTTPChatProvider(),
        foundry_chat_client_factory=NoCallProvider, cost_accounting_factory=lambda: accounting,
    )


def chat_payload(stream=False, *, subscription=False):
    return {
        "model": "ollama:synthetic" if subscription else "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "test"}], "stream": stream,
    }


@pytest.mark.parametrize("stream", [False, True])
def test_postgres_http_usage_and_admin(tmp_path, pg_dsn, stream):
    apply_migrations(pg_dsn)
    accounting = build_async_cost_accounting(postgres_env(tmp_path, pg_dsn))
    provider = HTTPChatProvider()
    with TestClient(chat_app(accounting, provider)) as client:
        assert client.get("/readyz").json()["cost_tracking"]["backend"] == "postgres"
        response = client.post("/v1/chat/completions", json=chat_payload(stream))
        assert response.status_code == 200
        assert "synthetic response" in response.text
        if stream:
            assert "data: [DONE]" in response.text
        else:
            assert response.json()["usage"]["total_tokens"] == 11
        client.portal.call(accounting.flush)
        status = client.get("/admin/cost/status", headers={"Authorization": "Bearer synthetic-admin-key"})
        assert status.json()["recording"]["records_written"] == 1
        assert status.json()["recording"]["last_record_success_at"] is not None
        assert status.json()["guarantee"] == "forecast_admission_only"
        assert provider.calls == 1
    row = accounting.ledger.fetch_events()[0]
    assert row["status"] == "finalized"
    assert row["prompt_tokens"] == 5 and row["completion_tokens"] == 6
    assert Decimal(row["estimated_cost_usd"]) == Decimal("0.000017")
    assert not (tmp_path / "cost.db").exists()


@pytest.mark.parametrize("stream", [False, True])
def test_record_db_failure_does_not_break_response_subscription_or_recovery(tmp_path, pg_dsn, stream, caplog):
    apply_migrations(pg_dsn)
    accounting = build_async_cost_accounting(postgres_env(tmp_path, pg_dsn))
    inspector = PostgresCostRepository(pg_dsn)
    subscription = HTTPChatProvider()
    with unreachable_dsn() as unavailable:
        provider = HTTPChatProvider(lambda: setattr(accounting.ledger, "_dsn", unavailable))
        with TestClient(chat_app(accounting, provider, subscription)) as client:
            response = client.post("/v1/chat/completions", json=chat_payload(stream))
            assert response.status_code == 200 and "synthetic response" in response.text
            if stream:
                assert "[DONE]" in response.text
            client.portal.call(accounting.flush)
            held = inspector.fetch_events()[0]
            assert held["status"] == "reserved" and held["billing_eligible"] == 1
            assert client.get("/healthz").status_code == 200
            ready = client.get("/readyz")
            assert ready.status_code == 503
            assert ready.json()["cost_tracking"]["recording"]["record_failures"] == 1
            assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 503
            assert provider.calls == 1
            for subscription_stream in (False, True):
                sub = client.post("/v1/chat/completions", json=chat_payload(subscription_stream, subscription=True))
                assert sub.status_code == 200 and "synthetic response" in sub.text
            assert subscription.calls == 2
            assert len(inspector.fetch_events()) == 1
            assert "synthetic-password-must-not-leak" not in ready.text + response.text + caplog.text
            accounting.ledger._dsn = pg_dsn
            provider.on_response = lambda: None
            assert client.get("/readyz").status_code == 200
            assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 200
        rows = inspector.fetch_events()
        assert [row["status"] for row in rows] == ["reserved", "finalized"]
        assert provider.calls == 2
    assert not (tmp_path / "cost.db").exists()


@pytest.mark.parametrize("stream", [False, True])
def test_unreachable_admission_never_sends_or_creates_sqlite(tmp_path, stream, caplog):
    with unreachable_dsn() as dsn:
        accounting = build_async_cost_accounting(postgres_env(tmp_path, dsn))
        provider = HTTPChatProvider()
        with TestClient(chat_app(accounting, provider)) as client:
            assert client.get("/healthz").status_code == 200
            ready = client.get("/readyz")
            assert ready.status_code == 503
            response = client.post("/v1/chat/completions", json=chat_payload(stream))
            assert response.status_code == (200 if stream else 503)
            assert "cost_tracking_unavailable" in response.text
            assert provider.calls == 0
            assert "synthetic-password-must-not-leak" not in ready.text + response.text + caplog.text
    assert not (tmp_path / "cost.db").exists()


def test_runtime_does_not_implicitly_create_schema(tmp_path, pg_dsn):
    accounting = build_async_cost_accounting(postgres_env(tmp_path, pg_dsn))
    provider = HTTPChatProvider()
    with TestClient(chat_app(accounting, provider)) as client:
        assert client.get("/readyz").status_code == 503
        assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 503
        assert provider.calls == 0
    with psycopg.connect(pg_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM pg_namespace WHERE nspname = 'bridge_cost'").fetchone()[0] == 0
    assert not (tmp_path / "cost.db").exists()


def test_independent_apps_serialize_each_http_attempt_at_exact_budget(tmp_path, pg_dsn):
    apply_migrations(pg_dsn)
    env = postgres_env(tmp_path, pg_dsn) | {"COST_SHORT_WINDOW_LIMIT_USD": "0.000003"}
    services = [build_async_cost_accounting(env) for _ in range(2)]
    for service in services:
        assert isinstance(service, AsyncCostAccounting)
        service._admission_timeout = 10
    providers = [HTTPChatProvider(usage=False) for _ in range(2)]
    apps = [chat_app(service, provider) for service, provider in zip(services, providers)]

    async def run():
        async with contextlib.AsyncExitStack() as stack:
            clients = []
            for app in apps:
                await stack.enter_async_context(app.router.lifespan_context(app))
                clients.append(await stack.enter_async_context(httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test",
                )))
            results = await asyncio.gather(*(
                clients[index % 2].post("/v1/chat/completions", json=chat_payload() | {"max_tokens": 1})
                for index in range(4)
            ))
            assert sorted(result.status_code for result in results) == [200, 429, 429, 429]
            assert sum(provider.calls for provider in providers) == 1
    asyncio.run(run())
    rows = PostgresCostRepository(pg_dsn).fetch_events()
    assert sum(row["status"] == "reserved" for row in rows) == 1
    assert sum(Decimal(row["estimated_cost_usd"]) for row in rows) == Decimal("0.000003")


@pytest.mark.parametrize("slow_lane", ["admission", "record"])
def test_slow_database_does_not_block_health_or_subscription_stream(tmp_path, pg_dsn, slow_lane):
    apply_migrations(pg_dsn)
    accounting = build_async_cost_accounting(postgres_env(tmp_path, pg_dsn))
    assert isinstance(accounting, AsyncCostAccounting)
    accounting._admission_timeout = 0.25
    paid_payload = chat_payload(True)
    entered, release = threading.Event(), threading.Event()
    if slow_lane == "admission":
        original = accounting._preflight
        def slow(**kwargs):
            entered.set()
            release.wait(15)
            return original(**kwargs)
        accounting._preflight = slow
    else:
        # Preparation performs genuine PostgreSQL health IO and BudgetGate admission.
        # The timed record lane isolates blocked recording, not the remote admission
        # deadline; the admission lane and other HTTP integration tests cover that.
        payload = main.OpenAIChatRequest.model_validate(paid_payload)
        model_config, model_error = main._resolve_chat_model(payload.model)
        assert model_error is None
        expected_preflight = {
            "endpoint": "chat", "model": payload.model,
            "forecast_usage": main._chat_forecast_usage(payload),
            "provider": (model_config or {}).get("provider", "vertex"),
        }
        assert expected_preflight["provider"] == "vertex"
        assert isinstance(accounting.ledger, PostgresCostRepository)
        original_preflight = accounting._preflight
        reservation = original_preflight(**expected_preflight)
        assert reservation is not None
        rows = PostgresCostRepository(pg_dsn).fetch_events()
        assert len(rows) == 1
        held = rows[0]
        assert held["reservation_id"] == reservation.reservation_id
        assert held["internal_request_id"] == reservation.internal_request_id
        assert held["status"] == "reserved" and held["billing_eligible"] == 1
        for field in ("endpoint", "model", "provider"):
            assert held[field] == expected_preflight[field] == getattr(reservation, field)
        forecast = expected_preflight["forecast_usage"]
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert held[field] == getattr(forecast, field)
        assert Decimal(held["forecast_cost_usd"]) == reservation.forecast_cost_usd
        assert Decimal(held["estimated_cost_usd"]) == reservation.forecast_cost_usd

        preflight_used = False
        preflight_lock = threading.Lock()
        def prepared_preflight(**kwargs):
            nonlocal preflight_used
            with preflight_lock:
                if preflight_used or kwargs != expected_preflight:
                    raise AssertionError("prepared reservation requires one exact matching attempt")
                preflight_used = True
                return reservation
        # Only replace IO admission: before_attempt still checks billing and context.
        accounting._preflight = prepared_preflight
        original = accounting.gate.finalize_success
        def slow(*args):
            assert args[0] is reservation
            entered.set()
            release.wait(15)
            return original(*args)
        accounting.gate.finalize_success = slow
    paid, subscription = HTTPChatProvider(), HTTPChatProvider()
    app = chat_app(accounting, paid, subscription)

    async def run():
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            pending = asyncio.create_task(client.post("/v1/chat/completions", json=paid_payload))
            try:
                async def wait_entered():
                    while not entered.is_set():
                        await asyncio.sleep(0.001)
                await asyncio.wait_for(wait_entered(), 10)
                health, sub = await asyncio.wait_for(asyncio.gather(
                    client.get("/healthz"),
                    client.post("/v1/chat/completions", json=chat_payload(True, subscription=True)),
                ), 0.15)
                assert health.status_code == 200
                assert "synthetic response" in sub.text and "[DONE]" in sub.text
                assert subscription.calls == 1
                if slow_lane == "record":
                    assert preflight_used
                    assert paid.calls == 1
                    response = await asyncio.wait_for(pending, 0.15)
                    assert "synthetic response" in response.text and "[DONE]" in response.text
                    assert accounting.metrics()["record_in_flight"] == 1
                    assert not release.is_set()
                else:
                    response = await pending
                    assert "cost_tracking_unavailable" in response.text
                    assert paid.calls == 0
            finally:
                release.set()
                await pending
    asyncio.run(run())


def test_actual_postgres_lock_wait_does_not_stop_other_requests(tmp_path, pg_dsn, monkeypatch):
    from openai_compatible_bridge.core import postgres_cost_repository

    apply_migrations(pg_dsn)
    accounting = build_async_cost_accounting(postgres_env(tmp_path, pg_dsn))
    assert isinstance(accounting, AsyncCostAccounting)
    assert isinstance(accounting.ledger, PostgresCostRepository)
    assert accounting.gate is not None
    accounting._admission_timeout = 0.15
    paid_payload = chat_payload()
    payload = main.OpenAIChatRequest.model_validate(paid_payload)
    model_config, model_error = main._resolve_chat_model(payload.model)
    assert model_error is None
    expected_preflight = {
        "endpoint": "chat", "model": payload.model,
        "forecast_usage": main._chat_forecast_usage(payload),
        "provider": (model_config or {}).get("provider", "vertex"),
    }
    # Real health IO is preparation, outside the timed advisory-lock admission lane.
    accounting.ledger.check_health()
    completion = threading.Event()
    admission_thread = threading.local()
    preflight_lock = threading.Lock()
    preflight_used = False
    admission_pid = None
    original_connect = postgres_cost_repository._connect
    original_preflight = accounting.gate.preflight

    def capture_connect(dsn):
        nonlocal admission_pid
        conn = original_connect(dsn)
        if getattr(admission_thread, "active", False):
            with preflight_lock:
                admission_pid = conn.info.backend_pid
        return conn

    monkeypatch.setattr(postgres_cost_repository, "_connect", capture_connect)

    def lock_preflight(**kwargs):
        nonlocal preflight_used
        try:
            with preflight_lock:
                if preflight_used or kwargs != expected_preflight:
                    raise AssertionError("lock admission requires one exact matching attempt")
                preflight_used = True
            admission_thread.active = True
            try:
                # Execute the real gate once; SQL errors must still propagate unchanged.
                reservation = original_preflight(**kwargs)
                assert reservation is not None
                return reservation
            finally:
                admission_thread.active = False
        finally:
            # Keep the worker in flight even if SQL fails before the HTTP assertion.
            completion.wait()

    accounting._preflight = lock_preflight
    paid, subscription = HTTPChatProvider(), HTTPChatProvider()
    app = chat_app(accounting, paid, subscription)

    async def run():
        async with (
            await psycopg.AsyncConnection.connect(pg_dsn, autocommit=True) as blocker,
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
        ):
            await blocker.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
            pending = asyncio.create_task(client.post("/v1/chat/completions", json=paid_payload))
            try:
                async def wait_for_lock():
                    while True:
                        with preflight_lock:
                            pid = admission_pid
                        if pid is not None:
                            cursor = await blocker.execute(
                                """
                                SELECT EXISTS (
                                    SELECT 1 FROM pg_stat_activity activity
                                    JOIN pg_locks waiter ON waiter.pid = activity.pid
                                    JOIN pg_locks held ON held.locktype = waiter.locktype
                                        AND held.database = waiter.database
                                        AND held.classid = waiter.classid
                                        AND held.objid = waiter.objid
                                        AND held.objsubid = waiter.objsubid
                                    WHERE activity.pid = %s
                                        AND activity.datname = current_database()
                                        AND waiter.database = (
                                            SELECT oid FROM pg_database
                                            WHERE datname = current_database()
                                        )
                                        AND waiter.locktype = 'advisory' AND NOT waiter.granted
                                        AND held.pid = %s AND held.granted
                                        AND held.pid = ANY(pg_blocking_pids(activity.pid))
                                )
                                """,
                                (pid, blocker.info.backend_pid),
                            )
                            if (await cursor.fetchone())[0]:
                                return
                        await asyncio.sleep(0.001)
                await asyncio.wait_for(wait_for_lock(), 8)
                health, stream = await asyncio.wait_for(asyncio.gather(
                    client.get("/healthz"),
                    client.post("/v1/chat/completions", json=chat_payload(True, subscription=True)),
                ), 0.1)
                assert health.status_code == 200 and "[DONE]" in stream.text
                response = await pending
                assert response.status_code == 503 and paid.calls == 0
                assert preflight_used
                assert accounting.metrics()["admission_in_flight"] == 1
                assert not completion.is_set()
            finally:
                try:
                    await blocker.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
                finally:
                    completion.set()
                    try:
                        await pending
                    finally:
                        async def drain_admission():
                            while accounting.metrics()["admission_in_flight"]:
                                await asyncio.sleep(0.001)
                        await asyncio.wait_for(drain_admission(), 8)
            rows = await asyncio.to_thread(accounting.ledger.fetch_events)
            assert len(rows) == 1 and rows[0]["status"] == "reserved"
            assert paid.calls == 0
    asyncio.run(run())
