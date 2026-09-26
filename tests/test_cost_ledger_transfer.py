from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from openai_compatible_bridge.core import cost_ledger_transfer as transfer
from openai_compatible_bridge.core.cost_tracking import LEDGER_ALLOWED_FIELDS, SQLiteCostRepository


pytest_plugins = ["postgres_helpers"]


STATUSES = ("reserved", "blocked", "released", "finalized", "estimated_only")
TINY_MONEY = "0.00000000000000000000000000000000000012345678901234567890123456789"
LARGE_MONEY = "12345678901234567890123456789.12345678901234567890123456789"
STAMP = "2026-09-26T01:02:03.123456Z"


def event(event_id: str, status: str = "finalized") -> dict:
    return {
        "event_id": event_id,
        "reservation_id": f"reservation-{event_id}",
        "internal_request_id": f"request-{event_id}",
        "provider": "synthetic-provider",
        "endpoint": "/v1/chat/completions",
        "model": "synthetic-model",
        "status": status,
        "billing_eligible": status not in {"blocked", "released"},
        "limit_type": "daily" if status == "blocked" else None,
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "embedding_tokens": 0,
        "rerank_units": 0,
        "forecast_cost_usd": TINY_MONEY,
        "estimated_cost_usd": TINY_MONEY,
        "currency": "USD",
        "pricing_source": "synthetic",
        "pricing_version": "version-1",
        "window_started_at": STAMP,
        "created_at": STAMP,
        "finalized_at": None if status == "reserved" else STAMP,
        "reconciliation_status": None,
    }


def populate(repository: SQLiteCostRepository) -> None:
    for status in STATUSES:
        repository.record_event(event(f"event-{status}", status))
    nullable = {name: None for name in LEDGER_ALLOWED_FIELDS}
    nullable.update(event_id="event-null", created_at=STAMP)
    repository.record_event(nullable)
    with repository.transaction():
        repository.connection.executemany(
            "INSERT INTO cost_daily_aggregates VALUES (?, ?, ?, ?)",
            [("2026-09-24", LARGE_MONEY, "USD", STAMP), ("2026-09-25", TINY_MONEY, "USD", STAMP)],
        )
        repository.connection.executemany(
            "INSERT INTO cost_reconciliation_results VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("2026-09-24", TINY_MONEY, "1.200", "-1.2", "mismatch", STAMP, None),
                ("2026-09-25", None, None, None, "unavailable", STAMP, "synthetic-sensitive-detail"),
            ],
        )


@pytest.fixture
def live_ledger(tmp_path):
    repository = SQLiteCostRepository(tmp_path / "live.sqlite")
    repository.initialize()
    populate(repository)
    yield repository
    repository.close()


@pytest.fixture
def snapshot(live_ledger, tmp_path):
    path = tmp_path / "snapshot.sqlite"
    transfer.backup_sqlite(live_ledger.path, path)
    return path


def assert_code(code, function, *args, **kwargs):
    with pytest.raises(transfer.LedgerTransferError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code, type(caught.value.__context__).__name__
    assert str(caught.value) == code


def test_backup_includes_wal_without_touching_source(live_ledger, tmp_path):
    source = live_ledger.path
    wal = Path(f"{source}-wal")
    assert wal.stat().st_size > 0
    before = (source.read_bytes(), wal.read_bytes())
    destination = tmp_path / "snapshot #1?.sqlite"

    result = transfer.backup_sqlite(source, destination)

    assert (source.read_bytes(), wal.read_bytes()) == before
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert result["tables"]["cost_events"]["row_count"] == 6
    assert not Path(f"{destination}-wal").exists()
    assert not Path(f"{destination}-shm").exists()
    with sqlite3.connect(destination) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0] == 6
    live_ledger.record_event(event("after-backup"))
    assert transfer.summarize_sqlite(destination) == result


def test_backup_refuses_missing_source_overwrite_and_source_itself(live_ledger, tmp_path):
    missing = tmp_path / "missing.sqlite"
    destination = tmp_path / "destination.sqlite"
    assert_code("source_not_found", transfer.backup_sqlite, missing, destination)
    assert not missing.exists()
    assert not destination.exists()
    destination.write_bytes(b"untouched")
    assert_code("destination_exists", transfer.backup_sqlite, live_ledger.path, destination)
    assert destination.read_bytes() == b"untouched"
    assert_code("destination_exists", transfer.backup_sqlite, live_ledger.path, live_ledger.path)


def test_backup_refuses_destination_symlink_and_orphan_wal(live_ledger, tmp_path):
    destination = tmp_path / "destination.sqlite"
    destination.symlink_to(tmp_path / "nonexistent-target")
    assert_code("destination_exists", transfer.backup_sqlite, live_ledger.path, destination)
    destination.unlink()
    wal = Path(f"{destination}-wal")
    wal.write_bytes(b"synthetic-orphan-wal")
    assert_code("destination_exists", transfer.backup_sqlite, live_ledger.path, destination)
    assert wal.read_bytes() == b"synthetic-orphan-wal"


def test_backup_bad_input_leaves_no_partial_destination(tmp_path):
    source = tmp_path / "invalid.sqlite"
    destination = tmp_path / "snapshot.sqlite"
    source.write_bytes(b"not a database")
    assert_code("sqlite_error", transfer.backup_sqlite, source, destination)
    assert not destination.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [source.name]


def test_summary_exact_totals_all_statuses_and_no_rows(snapshot):
    summary = transfer.summarize_sqlite(snapshot)
    events = summary["tables"]["cost_events"]
    assert events["status_counts"] == {**dict.fromkeys(STATUSES, 1), "null": 1}
    assert events["reservation_status_counts"] == dict.fromkeys(STATUSES, 1)
    with localcontext() as context:
        context.prec = 200
        assert Decimal(events["money_totals"]["estimated_cost_usd"]) == Decimal(TINY_MONEY) * 5
        aggregate_total = Decimal(LARGE_MONEY) + Decimal(TINY_MONEY)
    assert Decimal(summary["tables"]["cost_daily_aggregates"]["money_totals"]["estimated_cost_usd"]) == aggregate_total
    output = json.dumps(summary)
    for secret in ("event-finalized", "reservation-", "request-", "synthetic-model", "synthetic-sensitive-detail"):
        assert secret not in output
    assert all(len(table["digest"]) == 64 for table in summary["tables"].values())


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE cost_events SET model = 'changed' WHERE event_id = 'event-finalized'",
        "UPDATE cost_events SET prompt_tokens = 12 WHERE event_id = 'event-finalized'",
        "UPDATE cost_events SET finalized_at = NULL WHERE event_id = 'event-finalized'",
        "UPDATE cost_daily_aggregates SET currency = 'EUR'",
        "UPDATE cost_reconciliation_results SET error_message = 'changed'",
    ],
)
def test_digest_covers_non_money_columns(snapshot, sql):
    before = transfer.summarize_sqlite(snapshot)
    with sqlite3.connect(snapshot) as connection:
        connection.execute(sql)
    after = transfer.summarize_sqlite(snapshot)
    assert before != after
    for table in before["tables"]:
        assert before["tables"][table]["row_count"] == after["tables"][table]["row_count"]
        assert before["tables"][table]["money_totals"] == after["tables"][table]["money_totals"]


def test_digest_is_order_and_decimal_spelling_independent(snapshot):
    before = transfer.summarize_sqlite(snapshot)
    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE cost_reconciliation_results SET billing_export_cost = '1.2' WHERE billing_export_cost IS NOT NULL")
        connection.execute("UPDATE cost_events SET created_at = '2026-09-26T10:02:03.123456+09:00'")
        connection.execute("CREATE TEMP TABLE reordered AS SELECT * FROM cost_events ORDER BY event_id DESC")
        connection.execute("DELETE FROM cost_events")
        connection.execute("INSERT INTO cost_events SELECT * FROM reordered")
    assert transfer.summarize_sqlite(snapshot) == before


@pytest.mark.parametrize("money", ["NaN", "Infinity", "-Infinity", "not-money", "1_000", "-0.01", "1e999999"])
def test_invalid_money_is_rejected(snapshot, money):
    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE cost_events SET estimated_cost_usd = ? WHERE event_id = 'event-finalized'", (money,))
    assert_code("invalid_money", transfer.summarize_sqlite, snapshot)


@pytest.mark.parametrize("column", ["wrapper_estimated_cost_usd", "billing_export_cost"])
def test_only_reconciliation_delta_allows_negative_money(snapshot, column):
    with sqlite3.connect(snapshot) as connection:
        connection.execute(f"UPDATE cost_reconciliation_results SET {column} = '-0.01'")
    assert_code("invalid_money", transfer.summarize_sqlite, snapshot)


@pytest.mark.parametrize(
    "sql,code",
    [
        ("ALTER TABLE cost_events ADD COLUMN extra TEXT", "schema_mismatch"),
        ("ALTER TABLE cost_events RENAME COLUMN model TO renamed", "schema_mismatch"),
        ("DROP TABLE cost_daily_aggregates", "schema_mismatch"),
        ("CREATE TABLE sqliteX_hidden (value TEXT)", "schema_mismatch"),
        ("CREATE TRIGGER unexpected AFTER INSERT ON cost_events BEGIN SELECT 1; END", "schema_mismatch"),
        ("UPDATE cost_events SET reservation_id = 'duplicate'", "duplicate_reservation"),
        ("UPDATE cost_events SET event_id = NULL WHERE event_id = 'event-null'", "invalid_row"),
        ("UPDATE cost_events SET prompt_tokens = 'invalid'", "invalid_row"),
    ],
)
def test_schema_and_row_validation(snapshot, sql, code):
    with sqlite3.connect(snapshot) as connection:
        connection.execute(sql)
    assert_code(code, transfer.summarize_sqlite, snapshot)


def test_import_requires_frozen_snapshot(live_ledger):
    assert_code("snapshot_not_frozen", transfer.summarize_sqlite, live_ledger.path)


def test_snapshot_symlink_cannot_hide_live_wal(live_ledger, tmp_path):
    alias = tmp_path / "alias.sqlite"
    alias.symlink_to(live_ledger.path)
    assert_code("snapshot_not_frozen", transfer.summarize_sqlite, alias)


def test_unknown_status_not_exposed_in_summary(snapshot):
    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE cost_events SET status = 'synthetic-private-status'")
    summary = transfer.summarize_sqlite(snapshot)
    assert summary["tables"]["cost_events"]["status_counts"] == {"other": 6}
    assert "synthetic-private-status" not in json.dumps(summary)


def test_sqlite_reverse_serialization_preserves_entire_summary(snapshot, tmp_path):
    destination = tmp_path / "reverse.sqlite"
    before = snapshot.read_bytes()
    data = transfer._read_snapshot(snapshot)
    with transfer._fresh_sqlite(destination) as staging:
        transfer._write_sqlite(staging, data)
    assert transfer.summarize_sqlite(destination) == transfer.summarize_sqlite(snapshot)
    assert snapshot.read_bytes() == before
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_atomic_publish_never_replaces_racing_destination(tmp_path):
    destination = tmp_path / "racing.sqlite"
    with pytest.raises(transfer.LedgerTransferError, match="destination_exists"):
        with transfer._fresh_sqlite(destination) as staging:
            staging.write_bytes(b"new")
            destination.write_bytes(b"competing-writer")
    assert destination.read_bytes() == b"competing-writer"
    assert list(tmp_path.iterdir()) == [destination]


def test_summary_does_not_depend_on_decimal_context(snapshot):
    expected = transfer.summarize_sqlite(snapshot)
    with localcontext() as context:
        context.prec = 2
        assert transfer.summarize_sqlite(snapshot) == expected


@pytest.mark.parametrize("operation", ["import", "export", "migration"])
def test_api_mutations_require_apply(operation, tmp_path):
    path = tmp_path / "unused.sqlite"
    if operation == "import":
        assert_code("apply_required", transfer.import_sqlite, path, None)
    elif operation == "export":
        assert_code("apply_required", transfer.export_sqlite, None, path)
    else:
        assert_code("apply_required", transfer.migrate_schema)
    assert not path.exists()


def test_cli_missing_dsn_is_sanitized(monkeypatch, capsys):
    monkeypatch.delenv("COST_LEDGER_POSTGRES_DSN", raising=False)
    assert transfer.main(["migrate-schema", "--apply"]) == 1
    output = capsys.readouterr()
    assert json.loads(output.err) == {"ok": False, "code": "dsn_required"}
    assert not output.out


@pytest.mark.parametrize("command", ["migrate-schema", "import-sqlite", "export-sqlite"])
def test_cli_requires_apply_before_accessing_database(command, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COST_LEDGER_POSTGRES_DSN", "postgresql://synthetic-secret@invalid/synthetic")
    args = [command] if command == "migrate-schema" else [command, str(tmp_path / "database.sqlite")]
    assert transfer.main(args) != 0
    output = capsys.readouterr()
    assert json.loads(output.err) == {"ok": False, "code": "apply_required"}
    assert not output.out


def test_cli_redacts_errors_and_invalid_arguments(monkeypatch, capsys):
    def fail():
        raise RuntimeError("postgresql://user:synthetic-secret@host/db SELECT private_row")

    monkeypatch.setattr(transfer, "_open_repository", fail)
    assert transfer.main(["compare", "synthetic.sqlite"]) != 0
    output = capsys.readouterr()
    assert json.loads(output.err) == {"ok": False, "code": "operation_failed"}
    assert not output.out
    assert transfer.main(["compare", "synthetic.sqlite", "--dsn", "synthetic-secret"]) != 0
    output = capsys.readouterr()
    assert json.loads(output.err) == {"ok": False, "code": "invalid_arguments"}
    assert not output.out


def test_cli_module_backup(snapshot, tmp_path):
    destination = tmp_path / "cli.sqlite"
    environment = os.environ.copy()
    environment.pop("COST_LEDGER_POSTGRES_DSN", None)
    result = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-m", "openai_compatible_bridge.core.cost_ledger_transfer", "backup", str(snapshot), str(destination)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert not result.stderr
    output = json.loads(result.stdout)
    assert output["code"] == "backup_created"
    assert output["summary"] == transfer.summarize_sqlite(snapshot)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


@pytest.fixture
def pg_repository(pg_dsn):
    from openai_compatible_bridge.core.postgres_cost_repository import PostgresCostRepository, apply_migrations

    apply_migrations(pg_dsn)
    repository = PostgresCostRepository(pg_dsn)
    repository.initialize()
    yield repository
    repository.close()


def test_postgres_round_trip_repeat_and_reverse_after_new_record(snapshot, pg_repository, tmp_path):
    source_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    summary = transfer.summarize_sqlite(snapshot)
    imported = transfer.import_sqlite(snapshot, pg_repository, apply=True)
    assert imported == {"imported": True, "summary": summary}
    assert transfer.import_sqlite(snapshot, pg_repository, apply=True) == {"imported": False, "summary": summary}
    assert transfer.compare_sqlite(snapshot, pg_repository)["matches"] is True
    pg_repository.record_event(event("after-cutover", "reserved"))
    assert transfer.compare_sqlite(snapshot, pg_repository)["matches"] is False
    assert_code("target_not_empty", transfer.import_sqlite, snapshot, pg_repository, apply=True)
    assert_code("destination_exists", transfer.export_sqlite, pg_repository, snapshot, apply=True)
    destination = tmp_path / "rollback.sqlite"
    exported = transfer.export_sqlite(pg_repository, destination, apply=True)
    assert exported["tables"]["cost_events"]["row_count"] == 7
    assert transfer.compare_sqlite(destination, pg_repository)["matches"] is True
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == source_hash
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT status FROM cost_events WHERE event_id = 'after-cutover'").fetchone()[0] == "reserved"
        assert connection.execute("SELECT error_message FROM cost_reconciliation_results WHERE day = '2026-09-25'").fetchone()[0] == "synthetic-sensitive-detail"


def test_postgres_partial_import_failure_rolls_back_all_tables(snapshot, pg_repository):
    with pg_repository.transaction():
        pg_repository.connection.execute(
            "ALTER TABLE bridge_cost.cost_reconciliation_results ADD CONSTRAINT synthetic_failure CHECK (status <> 'unavailable')"
        )
    assert_code("postgres_error", transfer.import_sqlite, snapshot, pg_repository, apply=True)
    with pg_repository.transaction():
        for table in ("cost_events", "cost_daily_aggregates", "cost_reconciliation_results"):
            row = pg_repository.connection.execute(f"SELECT COUNT(*) AS count FROM bridge_cost.{table}").fetchone()
            assert (row["count"] if isinstance(row, dict) else row[0]) == 0


@pytest.mark.parametrize("table", ["cost_events", "cost_daily_aggregates", "cost_reconciliation_results"])
def test_postgres_compare_rejects_same_totals_but_changed_data(snapshot, pg_repository, table):
    transfer.import_sqlite(snapshot, pg_repository, apply=True)
    column = {"cost_events": "model", "cost_daily_aggregates": "currency", "cost_reconciliation_results": "error_message"}[table]
    with pg_repository.transaction():
        pg_repository.connection.execute(f"UPDATE bridge_cost.{table} SET {column} = 'changed'")
    comparison = transfer.compare_sqlite(snapshot, pg_repository)
    assert comparison["matches"] is False
    assert comparison["sqlite"]["tables"][table]["money_totals"] == comparison["postgres"]["tables"][table]["money_totals"]
    assert_code("target_not_empty", transfer.import_sqlite, snapshot, pg_repository, apply=True)


def test_postgres_cli_comparison_mismatch_nonzero(snapshot, pg_repository, pg_dsn, monkeypatch, capsys):
    monkeypatch.setenv("COST_LEDGER_POSTGRES_DSN", pg_dsn)
    assert transfer.main(["compare", str(snapshot)]) == 1
    output = capsys.readouterr()
    assert not output.err
    result = json.loads(output.out)
    assert result["ok"] is False
    assert result["code"] == "comparison_mismatch"
    assert result["comparison"]["matches"] is False
    assert pg_dsn not in output.out


def test_postgres_invalid_snapshot_never_imports_rows(snapshot, pg_repository):
    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE cost_daily_aggregates SET estimated_cost_usd = 'NaN'")
    assert_code("invalid_money", transfer.import_sqlite, snapshot, pg_repository, apply=True)
    with pg_repository.transaction():
        row = pg_repository.connection.execute("SELECT COUNT(*) AS count FROM bridge_cost.cost_events").fetchone()
        assert (row["count"] if isinstance(row, dict) else row[0]) == 0


def test_postgres_empty_snapshot_is_exact_noop(pg_repository, tmp_path):
    source = tmp_path / "empty.sqlite"
    repository = SQLiteCostRepository(source)
    repository.initialize()
    repository.close()
    snapshot = tmp_path / "empty-snapshot.sqlite"
    transfer.backup_sqlite(source, snapshot)
    assert transfer.import_sqlite(snapshot, pg_repository, apply=True)["imported"] is False
    assert transfer.compare_sqlite(snapshot, pg_repository)["matches"] is True


@pytest.mark.parametrize("table", ["cost_daily_aggregates", "cost_reconciliation_results"])
def test_postgres_import_refuses_target_with_only_non_event_data(snapshot, pg_repository, table):
    with pg_repository.transaction():
        if table == "cost_daily_aggregates":
            pg_repository.connection.execute(
                "INSERT INTO bridge_cost.cost_daily_aggregates (day, estimated_cost_usd, currency, updated_at) VALUES (%s, %s, %s, %s)",
                ("2026-09-26", "0", "USD", STAMP),
            )
        else:
            pg_repository.connection.execute(
                "INSERT INTO bridge_cost.cost_reconciliation_results (day, status, checked_at) VALUES (%s, %s, %s)",
                ("2026-09-26", "unavailable", STAMP),
            )
    assert_code("target_not_empty", transfer.import_sqlite, snapshot, pg_repository, apply=True)


def test_postgres_schema_drift_is_rejected(snapshot, pg_repository):
    with pg_repository.transaction():
        connection = pg_repository.connection
        connection.execute("ALTER TABLE bridge_cost.cost_events ADD COLUMN synthetic_extra TEXT")
        assert_code("schema_mismatch", transfer._check_postgres_schema, connection)
    assert_code("postgres_error", transfer.import_sqlite, snapshot, pg_repository, apply=True)
    assert_code("postgres_error", transfer.compare_sqlite, snapshot, pg_repository)


def test_postgres_cli_real_connection_error_is_redacted(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(
        "COST_LEDGER_POSTGRES_DSN",
        f"host={tmp_path / 'nonexistent-socket'} user=synthetic-user password=synthetic-secret dbname=synthetic connect_timeout=1",
    )
    assert transfer.main(["migrate-schema", "--apply"]) == 1
    output = capsys.readouterr()
    assert json.loads(output.err) == {"ok": False, "code": "postgres_error"}
    assert not output.out


def test_postgres_compare_and_export_support_read_only_transactions(snapshot, pg_repository, tmp_path):
    transfer.import_sqlite(snapshot, pg_repository, apply=True)
    with pg_repository.transaction():
        connection = pg_repository.connection
        connection.execute("SET TRANSACTION READ ONLY")
        assert connection.execute("SHOW transaction_read_only").fetchone()["transaction_read_only"] == "on"
        assert transfer.compare_sqlite(snapshot, pg_repository)["matches"] is True
        destination = tmp_path / "readonly-export.sqlite"
        assert transfer.export_sqlite(pg_repository, destination, apply=True) == transfer.summarize_sqlite(snapshot)


def test_postgres_cli_full_workflow(snapshot, pg_repository, pg_dsn, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("COST_LEDGER_POSTGRES_DSN", pg_dsn)
    destination = tmp_path / "cli-export.sqlite"
    for arguments, code in (
        (["migrate-schema", "--apply"], "schema_migrated"),
        (["import-sqlite", str(snapshot), "--apply"], "imported"),
        (["import-sqlite", str(snapshot), "--apply"], "already_imported"),
        (["compare", str(snapshot)], "match"),
        (["export-sqlite", str(destination), "--apply"], "export_created"),
    ):
        assert transfer.main(arguments) == 0
        output = capsys.readouterr()
        assert not output.err
        assert json.loads(output.out)["code"] == code
        for sensitive in (pg_dsn, "event-finalized", "synthetic-model", "synthetic-sensitive-detail"):
            assert sensitive not in output.out
    assert transfer.summarize_sqlite(destination) == transfer.summarize_sqlite(snapshot)