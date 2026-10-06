"""Real provider serializer + HTTP boundary + disposable PostgreSQL; no paid traffic."""

import asyncio
import json
from dataclasses import replace
from decimal import Decimal
from typing import Any

import httpx
import postgres_helpers
import pytest
from fastapi.testclient import TestClient
from test_context_compaction import (
    AFFINITY_HEADER,
    CLIENT_TOOL,
    _hide_tool_call,
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
from openai_compatible_bridge.providers.ollama import OllamaChatClient

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
    # These tests exercise per-call accounting, not a local-network deadline.
    accounting._admission_timeout = 10
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
                        **({"tool_calls": [_hide_tool_call()]} if internal else {}),
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
            assert "hide_context" not in response.text
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


@pytest.mark.parametrize("block_lfm", [False, True])
def test_lfm_summary_http_attempt_uses_ollama_cost_gate_and_ledger(
    monkeypatch, tmp_path, pg_dsn, block_lfm
):
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_LFM_ENABLED", "true")
    monkeypatch.setenv("CONTEXT_COMPACTION_MIN_CHARS", "100")
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", "")
    apply_migrations(pg_dsn)
    env = postgres_env(tmp_path, pg_dsn)
    env["COST_PROVIDER_BILLING_JSON"] = json.dumps({
        "foundry": "subscription" if block_lfm else "metered",
        "ollama": "metered",
    })
    env["COST_PRICING_JSON"] = json.dumps({
        "models": {
            "foundry:gpt-6-astra": {
                "chat": {"input_per_million": "1", "output_per_million": "1"}
            },
            "ollama:*": {
                "chat": {"input_per_million": "1", "output_per_million": "1"}
            },
        }
    })
    accounting = build_async_cost_accounting(env)
    assert isinstance(accounting, AsyncCostAccounting)
    # These tests exercise per-call accounting, not a local-network deadline.
    accounting._admission_timeout = 10
    if block_lfm:
        assert accounting.gate is not None
        accounting.gate.config = replace(accounting.gate.config, daily_limit_usd=Decimal(0))
    admitted: list[str] = []
    before = accounting.before_attempt

    async def track(provider_name: str):
        admitted.append(provider_name)
        return await before(provider_name)

    monkeypatch.setattr(accounting, "before_attempt", track)
    source_lines = [
        f"Catalog record for sample-addon component-{index:03d} supports exact-name lookup."
        for index in range(32)
    ]
    source_lines.extend([
        "The sample-addon package indexes catalog components for exact-name lookup.",
        "Command result: 12 tests passed.",
        "Build ID: synthetic-job-48001",
        "Artifact path: synthetic-output/catalog.json",
    ])
    source = "\n".join(source_lines)
    foundry_requests: list[dict[str, Any]] = []
    lfm_requests: list[dict[str, Any]] = []

    def respond_foundry(request):
        foundry_requests.append(json.loads(request.content))
        internal = len(foundry_requests) == 1
        result = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None if internal else "done",
                    **({"tool_calls": [_hide_tool_call()]} if internal else {}),
                },
                "finish_reason": "tool_calls" if internal else "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        return httpx.Response(200, json=result)

    def respond_ollama(request):
        lfm_requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "lfm2.5-thinking:latest",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "summary": "sample-addon 구성 요소의 이름 조회 목록을 확인했다."
                }),
            },
            "prompt_eval_count": 80,
            "eval_count": 18,
            "done": True,
        })

    foundry = FoundryChatClient(base_url="https://synthetic.invalid/chat", token="synthetic-only")
    asyncio.run(foundry.http.aclose())
    foundry.http = httpx.AsyncClient(transport=httpx.MockTransport(respond_foundry))
    ollama = OllamaChatClient(base_url="https://synthetic.invalid")
    asyncio.run(ollama.http.aclose())
    ollama.http = httpx.AsyncClient(transport=httpx.MockTransport(respond_ollama))
    old = _register_alias()
    try:
        app = _foundry_only_app(
            foundry_chat_client_factory=lambda: foundry,
            ollama_chat_client_factory=lambda: ollama,
            cost_accounting_factory=lambda: accounting,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={AFFINITY_HEADER: "synthetic-lfm-cost"},
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": _messages(source),
                    "tools": [CLIENT_TOOL],
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "done"
            assert len(foundry_requests) == 2
            assert admitted == ["foundry", "ollama", "foundry"]
            assert len(lfm_requests) == (0 if block_lfm else 1)
            items = app.state.context_compaction_store.items("synthetic-lfm-cost")
            assert len(items) == 1
            assert items[0].compaction_source == ("rule" if block_lfm else "lfm")
            assert client.portal is not None
            client.portal.call(accounting.flush)

        rows = accounting.ledger.fetch_events()
        if block_lfm:
            assert len(rows) == 1
            assert rows[0]["provider"] == "ollama"
            assert rows[0]["billing_eligible"] == 0
        else:
            finalized = [row for row in rows if row["status"] == "finalized"]
            assert len(finalized) == 3
            assert sorted(row["provider"] for row in finalized) == ["foundry", "foundry", "ollama"]
            lfm_row = next(row for row in finalized if row["provider"] == "ollama")
            assert lfm_row["prompt_tokens"] == 80
            assert lfm_row["completion_tokens"] == 18
        assert app.state.context_compaction_expiry_task.done()
    finally:
        _restore_alias(old)
