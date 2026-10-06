from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from openai_compatible_bridge.core.cost_tracking import CostSubsystemUnhealthy
from openai_compatible_bridge.core.postgres_cost_repository import PostgresCostRepository


def _health_repository(monkeypatch, *, schema=True, function=True, tables=True, read_only="off"):
    conn = Mock()
    conn.execute.side_effect = [
        Mock(fetchone=Mock(return_value={"transaction_read_only": read_only})),
        Mock(fetchone=Mock(return_value={
            "schema_allowed": schema, "function_allowed": function, "tables_allowed": tables,
        })),
    ]
    repo = PostgresCostRepository("unused-mocked-dsn")
    monkeypatch.setattr(repo, "_read", lambda: nullcontext(conn))
    return repo, conn


def test_health_batches_all_privileges_without_caching(monkeypatch):
    repo, conn = _health_repository(monkeypatch)
    repo.check_health()
    assert conn.execute.call_count == 2
    assert conn.execute.call_args_list[0].args == ("SHOW transaction_read_only",)
    sql, params = conn.execute.call_args_list[1].args
    normalized = " ".join(sql.split())
    assert "has_schema_privilege(%s, 'USAGE')" in normalized
    assert "has_function_privilege('pg_catalog.pg_advisory_xact_lock(bigint)', 'EXECUTE')" in normalized
    assert "bool_and(has_table_privilege(table_name, privilege))" in normalized
    assert "FROM unnest(%s::text[]) AS tables(table_name)" in normalized
    assert "CROSS JOIN unnest(%s::text[]) AS privileges(privilege)" in normalized
    assert params == (
        "bridge_cost",
        [
            "bridge_cost.cost_events",
            "bridge_cost.cost_daily_aggregates",
            "bridge_cost.cost_reconciliation_results",
        ],
        ["SELECT", "INSERT", "UPDATE", "DELETE"],
    )
    assert all(table not in sql for table in params[1])
    assert "CASE WHEN schema_allowed" in normalized
    assert "CASE WHEN schema_allowed AND function_allowed" in normalized

    conn.execute.side_effect = None
    conn.execute.return_value.fetchone.side_effect = [
        {"transaction_read_only": "off"},
        {"schema_allowed": True, "function_allowed": True, "tables_allowed": False},
    ]
    with pytest.raises(CostSubsystemUnhealthy) as caught:
        repo.check_health()
    assert str(caught.value) == "PostgreSQL 비용 저장소 runtime 권한이 부족합니다."
    assert conn.execute.call_count == 4
    assert conn.execute.call_args_list[3].args == (sql, params)


@pytest.mark.parametrize("schema,function,tables,message", [
    (False, True, True, "PostgreSQL 비용 schema 접근 권한이 없습니다."),
    (False, False, True, "PostgreSQL 비용 schema 접근 권한이 없습니다."),
    (False, True, False, "PostgreSQL 비용 schema 접근 권한이 없습니다."),
    (False, False, False, "PostgreSQL 비용 schema 접근 권한이 없습니다."),
    (True, False, True, "PostgreSQL 비용 lock 실행 권한이 없습니다."),
    (True, False, False, "PostgreSQL 비용 lock 실행 권한이 없습니다."),
    (True, True, False, "PostgreSQL 비용 저장소 runtime 권한이 부족합니다."),
    (None, True, True, "PostgreSQL 비용 schema 접근 권한이 없습니다."),
    (True, None, True, "PostgreSQL 비용 lock 실행 권한이 없습니다."),
    (True, True, None, "PostgreSQL 비용 저장소 runtime 권한이 부족합니다."),
])
def test_health_privilege_denial_preserves_message_and_precedence(
    monkeypatch, schema, function, tables, message,
):
    repo, conn = _health_repository(monkeypatch, schema=schema, function=function, tables=tables)
    with pytest.raises(CostSubsystemUnhealthy) as caught:
        repo.check_health()
    assert str(caught.value) == message
    assert conn.execute.call_count == 2


def test_health_read_only_denial_precedes_privilege_query(monkeypatch):
    repo, conn = _health_repository(monkeypatch, schema=False, function=False, tables=False, read_only="on")
    with pytest.raises(CostSubsystemUnhealthy) as caught:
        repo.check_health()
    assert str(caught.value) == "PostgreSQL 비용 저장소가 read-only 상태입니다."
    assert conn.execute.call_args_list == [(("SHOW transaction_read_only",), {})]
