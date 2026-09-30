"""Real provider serializer + HTTP boundary + disposable PostgreSQL; no paid traffic."""

import asyncio
import json
from dataclasses import replace
from decimal import Decimal

import httpx
import postgres_helpers
import pytest
from fastapi.testclient import TestClient
from test_context_compaction import (
    AFFINITY_HEADER,
    CLIENT_TOOL,
    _compact_call,
    _foundry_only_app,
    _messages,
    _register_alias,
    _restore_alias,
)
from test_cost_postgres_api import postgres_env

from openai_compatible_bridge.core.async_cost import (
    AsyncCostAccounting,
    build_async_cost_accounting,
)
from openai_compatible_bridge.core.cost_tracking import (
    CostConfigError,
    CostSubsystemUnhealthy,
)
from openai_compatible_bridge.core.postgres_cost_repository import apply_migrations
from openai_compatible_bridge.providers.foundry import FoundryChatClient

pg_dsn = postgres_helpers.pg_dsn


@pytest.mark.parametrize(
    "failure",
    [None, "budget", "config", "database", "missing_usage", "upstream", "loop"],
)
def test_internal_calls_are_individually_admitted_and_recorded(
    monkeypatch, tmp_path, pg_dsn, failure
):
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_MAX_INTERNAL_ROUNDS", "1")
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", "")
    apply_migrations(pg_dsn)
    env = postgres_env(tmp_path, pg_dsn)
    env["COST_PROVIDER_BILLING_JSON"] = '{"foundry":"metered"}'
    env["COST_PRICING_JSON"] = json.dumps(
        {
            "models": {
                "foundry:gpt-6-astra": {
                    "chat": {"input_per_million": "1", "output_per_million": "1"}
                }
            }
        }
    )
    accounting = build_async_cost_accounting(env)
    assert isinstance(accounting, AsyncCostAccounting)
    admitted = []
    before = accounting.before_attempt

    async def admit(provider):
        admitted.append(provider)
        if len(admitted) == 2:
            if failure == "budget":
                # Exercise the actual PostgreSQL-backed gate, not an injected exception.
                assert accounting.gate is not None
                accounting.gate.config = replace(
                    accounting.gate.config, daily_limit_usd=Decimal(0)
                )
            if failure == "config":
                raise CostConfigError("synthetic configuration unavailable")
            if failure == "database":
                raise CostSubsystemUnhealthy("synthetic database unavailable")
        return await before(provider)

    monkeypatch.setattr(accounting, "before_attempt", admit)
    sent = []

    def respond(request):
        assert request.url.host == "synthetic.invalid"
        sent.append(json.loads(request.content))
        if len(sent) == 2 and failure == "upstream":
            return httpx.Response(
                503, json={"error": {"message": "synthetic unavailable"}}
            )
        internal = len(sent) == 1 or failure == "loop"
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None if internal else "done",
                        **({"tool_calls": [_compact_call()]} if internal else {}),
                    },
                    "finish_reason": "tool_calls" if internal else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        if failure == "missing_usage" and len(sent) == 2:
            payload.pop("usage")
        return httpx.Response(200, json=payload)

    provider = FoundryChatClient(
        base_url="https://synthetic.invalid/chat", token="synthetic-only"
    )
    asyncio.run(provider.http.aclose())
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    old = _register_alias()
    try:
        app = _foundry_only_app(
            foundry_chat_client_factory=lambda: provider,
            cost_accounting_factory=lambda: accounting,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={AFFINITY_HEADER: "synthetic"},
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": _messages(),
                    "tools": [CLIENT_TOOL],
                },
            )
            expected = {
                "budget": 429,
                "config": 503,
                "database": 503,
                "upstream": 503,
                "loop": 502,
            }.get(failure, 200)
            assert response.status_code == expected, response.text
            assert client.portal is not None
            client.portal.call(accounting.flush)
            assert admitted == ["foundry", "foundry"]
            assert len(sent) == (
                1 if failure in {"budget", "config", "database"} else 2
            )
            if len(sent) == 2:
                assert (
                    sent[0]["messages"][2]["content"]
                    != sent[1]["messages"][2]["content"]
                )
            if failure is None:
                assert response.json()["usage"]["total_tokens"] == 24
                assert response.json()["choices"][0]["message"]["content"] == "done"
            assert "compact_context" not in response.text
        assert app.state.context_compaction_expiry_task.done()
        assert provider.http.is_closed
        rows = accounting.ledger.fetch_events()
        billable = [row for row in rows if row["billing_eligible"]]
        assert len(billable) == len(sent)
        if failure == "budget":
            assert len(rows) == len(sent) + 1
            assert sum(not row["billing_eligible"] for row in rows) == 1
        else:
            assert len(rows) == len(sent)
        finalized = [row for row in rows if row["status"] == "finalized"]
        assert len(finalized) == (
            1
            if failure in {"budget", "config", "database", "missing_usage", "upstream"}
            else 2
        )
        assert sum(row["prompt_tokens"] for row in finalized) == 10 * len(finalized)
        if failure in {"missing_usage", "upstream"}:
            assert any(row["status"] == "reserved" for row in rows)
    finally:
        _restore_alias(old)
