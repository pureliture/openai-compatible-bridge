"""오프라인 ledger 이전 도구. 운영자는 작업 전후 모든 writer를 차단해야 한다."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterator, Mapping

from .cost_tracking import LEDGER_ALLOWED_FIELDS


_COLUMNS = {
    "cost_events": LEDGER_ALLOWED_FIELDS,
    "cost_daily_aggregates": ("day", "estimated_cost_usd", "currency", "updated_at"),
    "cost_reconciliation_results": (
        "day", "wrapper_estimated_cost_usd", "billing_export_cost", "delta_usd", "status", "checked_at", "error_message",
    ),
}
_MONEY = {
    "cost_events": ("forecast_cost_usd", "estimated_cost_usd"),
    "cost_daily_aggregates": ("estimated_cost_usd",),
    "cost_reconciliation_results": ("wrapper_estimated_cost_usd", "billing_export_cost", "delta_usd"),
}
_INTEGERS = {"billing_eligible", "prompt_tokens", "completion_tokens", "total_tokens", "embedding_tokens", "rerank_units"}
_TIMESTAMPS = {"window_started_at", "created_at", "finalized_at", "updated_at", "checked_at"}
_REQUIRED = {
    "cost_events": {"event_id"},
    "cost_daily_aggregates": {"day", "estimated_cost_usd", "currency", "updated_at"},
    "cost_reconciliation_results": {"day", "status", "checked_at"},
}
_STATUSES = {"reserved", "blocked", "released", "finalized", "estimated_only", "pending", "unavailable", "error", "matched", "mismatch", "ok"}
_SIDECARS = ("-wal", "-shm", "-journal")
_DECIMAL_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


class LedgerTransferError(Exception):
    """행·SQL·접속 정보를 포함하지 않는 안정적인 오류 코드."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@contextlib.contextmanager
def _sanitize(default_code: str) -> Iterator[None]:
    try:
        yield
    except LedgerTransferError:
        raise
    except sqlite3.Error:
        raise LedgerTransferError("sqlite_error") from None
    except OSError:
        raise LedgerTransferError("filesystem_error") from None
    except Exception:
        raise LedgerTransferError(default_code) from None


def _require_apply(apply: bool) -> None:
    if apply is not True:
        raise LedgerTransferError("apply_required")


def _sidecars(path: Path) -> Iterator[Path]:
    return (path.with_name(path.name + suffix) for suffix in _SIDECARS)


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _check_destination(path: Path) -> None:
    if _exists(path) or any(_exists(sidecar) for sidecar in _sidecars(path)):
        raise LedgerTransferError("destination_exists")


@contextlib.contextmanager
def _fresh_sqlite(destination: Path) -> Iterator[Path]:
    _check_destination(destination)
    descriptor, name = tempfile.mkstemp(prefix=".cost-ledger-", suffix=".sqlite", dir=destination.parent)
    staging = Path(name)
    os.close(descriptor)
    try:
        yield staging
        with staging.open("rb") as handle:
            os.fsync(handle.fileno())
        _check_destination(destination)
        try:
            os.link(staging, destination)
        except FileExistsError:
            raise LedgerTransferError("destination_exists") from None
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        for path in (staging, *_sidecars(staging)):
            if _exists(path):
                path.unlink()


@contextlib.contextmanager
def _read_only_sqlite(path: Path) -> Iterator[sqlite3.Connection]:
    if not path.is_file():
        raise LedgerTransferError("source_not_found")
    uri = path.resolve().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("BEGIN")
        yield connection


def _check_integrity(connection: sqlite3.Connection) -> None:
    results = connection.execute("PRAGMA integrity_check").fetchall()
    if len(results) != 1 or results[0][0] != "ok":
        raise LedgerTransferError("integrity_check_failed")


def _check_sqlite_schema(connection: sqlite3.Connection) -> None:
    objects = connection.execute(
        "SELECT type, name FROM sqlite_master WHERE name NOT GLOB 'sqlite_*' AND type != 'index'"
    ).fetchall()
    if {(row[0], row[1]) for row in objects} != {("table", table) for table in _COLUMNS}:
        raise LedgerTransferError("schema_mismatch")
    for table, columns in _COLUMNS.items():
        info = connection.execute(f"PRAGMA table_xinfo({table})").fetchall()
        if {row[1] for row in info} != set(columns):
            raise LedgerTransferError("schema_mismatch")
        for row in info:
            name = row[1]
            expected_type = "INTEGER" if name in _INTEGERS else "TEXT"
            primary = name == columns[0]
            required = name in _REQUIRED[table] and not primary
            if row[2].upper() != expected_type or bool(row[5]) != primary or bool(row[3]) != required or row[4] is not None or row[6]:
                raise LedgerTransferError("schema_mismatch")


def _decimal_text(value: Decimal) -> str:
    if not value:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _money_value(column: str, value: Any) -> str:
    if not isinstance(value, (str, Decimal)) or not _DECIMAL_PATTERN.fullmatch(str(value)):
        raise LedgerTransferError("invalid_money")
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise LedgerTransferError("invalid_money") from None
    if not amount.is_finite() or (amount < 0 and column != "delta_usd"):
        raise LedgerTransferError("invalid_money")
    if amount.adjusted() > 131071 or amount.as_tuple().exponent < -16383:
        raise LedgerTransferError("invalid_money")
    return _decimal_text(amount)


def _timestamp(value: Any) -> str:
    try:
        if isinstance(value, str):
            fraction = re.search(r"[T ][0-9]{2}:[0-9]{2}:[0-9]{2}[.,]([0-9]+)", value)
            if fraction and any(digit != "0" for digit in fraction[1][6:]):
                raise ValueError
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime):
            raise ValueError
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (ValueError, OverflowError):
        raise LedgerTransferError("invalid_row") from None


def _normalize_row(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for column in _COLUMNS[table]:
        value = row[column]
        if value is None:
            if column in _REQUIRED[table]:
                raise LedgerTransferError("invalid_row")
        elif column in _MONEY[table]:
            value = _money_value(column, value)
        elif column in _INTEGERS:
            if not isinstance(value, int) or value < 0 or value > 9223372036854775807:
                raise LedgerTransferError("invalid_row")
            value = int(value)
            if column == "billing_eligible" and value not in (0, 1):
                raise LedgerTransferError("invalid_row")
        elif column in _TIMESTAMPS:
            value = _timestamp(value)
        elif column == "day":
            try:
                value = date.fromisoformat(str(value)).isoformat()
            except ValueError:
                raise LedgerTransferError("invalid_row") from None
        elif not isinstance(value, str) or "\x00" in value:
            raise LedgerTransferError("invalid_row")
        if column == _COLUMNS[table][0] and not value:
            raise LedgerTransferError("invalid_row")
        normalized[column] = value
    return normalized


def _read_rows(connection: Any, *, postgres: bool = False) -> dict[str, list[dict[str, Any]]]:
    data = {}
    for table, columns in _COLUMNS.items():
        qualified = f"bridge_cost.{table}" if postgres else table
        cursor = connection.execute(f"SELECT {', '.join(columns)} FROM {qualified}")
        rows = []
        keys: set[str] = set()
        reservations: set[str] = set()
        for values in cursor:
            row = dict(values) if isinstance(values, (Mapping, sqlite3.Row)) else dict(zip(columns, values))
            row = _normalize_row(table, row)
            key = row[columns[0]]
            if key in keys:
                raise LedgerTransferError("duplicate_key")
            keys.add(key)
            if table == "cost_events" and row["reservation_id"] is not None:
                reservation = row["reservation_id"]
                if reservation in reservations:
                    raise LedgerTransferError("duplicate_reservation")
                reservations.add(reservation)
            rows.append(row)
        data[table] = sorted(rows, key=lambda row: row[columns[0]])
    return data


def _exact_add(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = max(left.adjusted(), right.adjusted(), 0) - min(left.as_tuple().exponent, right.as_tuple().exponent, 0) + 2
        return left + right


def _status_bucket(value: str | None) -> str:
    return "null" if value is None else value if value in _STATUSES else "other"


def _summary(data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    tables = {}
    for table, columns in _COLUMNS.items():
        digest = hashlib.sha256()
        digest.update(json.dumps(["ledger-transfer-v1", table, columns], separators=(",", ":")).encode("utf-8"))
        totals = dict.fromkeys(_MONEY[table], Decimal(0))
        statuses: Counter[str] = Counter()
        reservations: Counter[str] = Counter()
        for row in data[table]:
            digest.update(b"\n")
            digest.update(json.dumps([row[column] for column in columns], ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
            for column in totals:
                if row[column] is not None:
                    totals[column] = _exact_add(totals[column], Decimal(row[column]))
            if "status" in columns:
                bucket = _status_bucket(row["status"])
                statuses[bucket] += 1
                if table == "cost_events" and row["reservation_id"] is not None:
                    reservations[bucket] += 1
        tables[table] = {
            "row_count": len(data[table]),
            "money_totals": {column: _decimal_text(total) for column, total in totals.items()},
            "status_counts": dict(sorted(statuses.items())),
            "reservation_status_counts": dict(sorted(reservations.items())),
            "digest": digest.hexdigest(),
        }
    return {"format_version": 1, "tables": tables}


def _read_snapshot(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise LedgerTransferError("source_not_found")
    path = path.resolve()
    if any(_exists(sidecar) for sidecar in _sidecars(path)):
        raise LedgerTransferError("snapshot_not_frozen")
    before = path.stat()
    with _read_only_sqlite(path) as connection:
        _check_integrity(connection)
        _check_sqlite_schema(connection)
        data = _read_rows(connection)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise LedgerTransferError("snapshot_changed")
    if any(_exists(sidecar) for sidecar in _sidecars(path)):
        raise LedgerTransferError("snapshot_not_frozen")
    return data


def summarize_sqlite(snapshot: str | Path) -> dict[str, Any]:
    """동결한 SQLite의 행 수·정확한 금액 합계·상태 수·전체 열 해시를 반환한다."""
    with _sanitize("sqlite_error"):
        return _summary(_read_snapshot(Path(snapshot)))


def backup_sqlite(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """read-only SQLite backup API로 WAL을 포함하고 0600 새 파일만 게시한다."""
    with _sanitize("sqlite_error"):
        with _read_only_sqlite(Path(source)) as source_connection:
            _check_integrity(source_connection)
            _check_sqlite_schema(source_connection)
            with _fresh_sqlite(Path(destination)) as staging:
                with contextlib.closing(sqlite3.connect(staging)) as target:
                    deadline = time.monotonic() + 30

                    def progress(status: int, remaining: int, total: int) -> None:
                        if time.monotonic() > deadline:
                            raise LedgerTransferError("backup_timeout")

                    source_connection.backup(target, pages=256, progress=progress, sleep=0.01)
                    target.execute("PRAGMA journal_mode=DELETE")
                summary = _summary(_read_snapshot(staging))
        return summary


def _check_postgres_schema(connection: Any) -> dict[str, dict[str, str]]:
    cursor = connection.execute(
        """
        SELECT c.table_name, c.column_name, c.data_type, c.numeric_scale
        FROM information_schema.columns c
        JOIN information_schema.tables t
          ON c.table_schema = t.table_schema AND c.table_name = t.table_name
        WHERE c.table_schema = 'bridge_cost' AND t.table_type = 'BASE TABLE'
          AND c.table_name IN ('cost_events', 'cost_daily_aggregates', 'cost_reconciliation_results')
        """
    )
    schema: dict[str, dict[str, str]] = {table: {} for table in _COLUMNS}
    for values in cursor:
        if isinstance(values, Mapping):
            table, column, kind, scale = (values[name] for name in ("table_name", "column_name", "data_type", "numeric_scale"))
        else:
            table, column, kind, scale = values
        schema[table][column] = kind
        if column in _MONEY[table]:
            valid = kind == "numeric" and scale is None
        elif column == "billing_eligible":
            valid = kind in {"boolean", "smallint", "integer", "bigint"}
        elif column in _INTEGERS:
            valid = kind in {"smallint", "integer", "bigint"}
        elif column in _TIMESTAMPS:
            valid = kind in {"text", "timestamp with time zone"}
        elif column == "day":
            valid = kind in {"text", "date"}
        else:
            valid = kind == "text"
        if not valid:
            raise LedgerTransferError("schema_mismatch")
    if any(set(schema[table]) != set(columns) for table, columns in _COLUMNS.items()):
        raise LedgerTransferError("schema_mismatch")
    return schema


@contextlib.contextmanager
def _postgres_transaction(repository: Any, *, writing: bool = False) -> Iterator[tuple[Any, dict[str, dict[str, str]]]]:
    with repository.transaction():
        connection = repository.connection
        mode = "SHARE ROW EXCLUSIVE" if writing else "SHARE"
        tables = ", ".join(f"bridge_cost.{table}" for table in _COLUMNS)
        connection.execute(f"LOCK TABLE {tables} IN {mode} MODE")
        yield connection, _check_postgres_schema(connection)


def import_sqlite(snapshot: str | Path, repository: Any, *, apply: bool = False) -> dict[str, Any]:
    """빈 PostgreSQL에 한 transaction으로 이전한다. 전체 내용이 같을 때만 재실행을 허용한다."""
    _require_apply(apply)
    with _sanitize("postgres_error"):
        source = _read_snapshot(Path(snapshot))
        summary = _summary(source)
        with _postgres_transaction(repository, writing=True) as (connection, schema):
            existing = _read_rows(connection, postgres=True)
            if _summary(existing) == summary:
                return {"imported": False, "summary": summary}
            if any(existing.values()):
                raise LedgerTransferError("target_not_empty")
            for table, columns in _COLUMNS.items():
                placeholders = ", ".join("%s" for _ in columns)
                sql = f"INSERT INTO bridge_cost.{table} ({', '.join(columns)}) VALUES ({placeholders})"
                for row in source[table]:
                    values = [
                        bool(row[column]) if schema[table][column] == "boolean" and row[column] is not None else row[column]
                        for column in columns
                    ]
                    connection.execute(sql, values)
            if _summary(_read_rows(connection, postgres=True)) != summary:
                raise LedgerTransferError("verification_failed")
        return {"imported": True, "summary": summary}


def compare_sqlite(snapshot: str | Path, repository: Any) -> dict[str, Any]:
    """DB 변경 없이 세 테이블의 모든 정규화된 열과 요약을 비교한다."""
    with _sanitize("postgres_error"):
        source = _summary(_read_snapshot(Path(snapshot)))
        with _postgres_transaction(repository) as (connection, _):
            target = _summary(_read_rows(connection, postgres=True))
        return {"matches": source == target, "sqlite": source, "postgres": target}


def _write_sqlite(path: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        with connection:
            for table, columns in _COLUMNS.items():
                definitions = []
                for column in columns:
                    kind = "INTEGER" if column in _INTEGERS else "TEXT"
                    constraint = " PRIMARY KEY" if column == columns[0] else " NOT NULL" if column in _REQUIRED[table] else ""
                    definitions.append(f"{column} {kind}{constraint}")
                connection.execute(f"CREATE TABLE {table} ({', '.join(definitions)})")
                placeholders = ", ".join("?" for _ in columns)
                connection.executemany(
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                    ([row[column] for column in columns] for row in data[table]),
                )
            connection.execute("CREATE INDEX idx_cost_events_created_at ON cost_events(created_at)")
            connection.execute("CREATE INDEX idx_cost_events_status ON cost_events(status)")


def export_sqlite(repository: Any, destination: str | Path, *, apply: bool = False) -> dict[str, Any]:
    """전환 후 행까지 포함한 PostgreSQL 전체를 새 SQLite로 내보내며 기존 백업은 덮어쓰지 않는다."""
    _require_apply(apply)
    with _sanitize("postgres_error"):
        with _fresh_sqlite(Path(destination)) as staging:
            with _postgres_transaction(repository) as (connection, _):
                data = _read_rows(connection, postgres=True)
                summary = _summary(data)
                _write_sqlite(staging, data)
                if _summary(_read_snapshot(staging)) != summary:
                    raise LedgerTransferError("verification_failed")
        return summary


def _dsn_from_env() -> str:
    dsn = os.environ.get("COST_LEDGER_POSTGRES_DSN", "").strip()
    if not dsn:
        raise LedgerTransferError("dsn_required")
    return dsn


def migrate_schema(*, apply: bool = False) -> None:
    """명시적인 운영자 승인으로만 schema migration을 실행한다."""
    _require_apply(apply)
    with _sanitize("postgres_error"):
        dsn = _dsn_from_env()
        from .postgres_cost_repository import apply_migrations

        apply_migrations(dsn)


@contextlib.contextmanager
def _open_repository() -> Iterator[Any]:
    with _sanitize("postgres_error"):
        dsn = _dsn_from_env()
        from .postgres_cost_repository import PostgresCostRepository

        repository = PostgresCostRepository(dsn)
        try:
            repository.initialize()
            yield repository
        finally:
            repository.close()


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise LedgerTransferError("invalid_arguments")


def main(argv: list[str] | None = None) -> int:
    parser = _ArgumentParser(prog="cost-ledger-transfer", description="writer 차단 후 사용하는 오프라인 ledger 이전 도구")
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup", help="SQLite WAL 포함 read-only 백업")
    backup.add_argument("source")
    backup.add_argument("destination")
    migration = commands.add_parser("migrate-schema", help="PostgreSQL schema 명시적 적용")
    migration.add_argument("--apply", action="store_true")
    importer = commands.add_parser("import-sqlite", help="동결된 SQLite를 빈 PostgreSQL로 이전")
    importer.add_argument("snapshot")
    importer.add_argument("--apply", action="store_true")
    compare = commands.add_parser("compare", help="세 테이블의 요약과 전체 열 해시 비교")
    compare.add_argument("snapshot")
    exporter = commands.add_parser("export-sqlite", help="현재 PostgreSQL 전체를 새 SQLite로 내보내기")
    exporter.add_argument("destination")
    exporter.add_argument("--apply", action="store_true")
    try:
        args = parser.parse_args(argv)
        if args.command in {"migrate-schema", "import-sqlite", "export-sqlite"}:
            _require_apply(args.apply)
        if args.command == "backup":
            result = {"ok": True, "code": "backup_created", "summary": backup_sqlite(args.source, args.destination)}
        elif args.command == "migrate-schema":
            migrate_schema(apply=True)
            result = {"ok": True, "code": "schema_migrated"}
        else:
            with _open_repository() as repository:
                if args.command == "import-sqlite":
                    imported = import_sqlite(args.snapshot, repository, apply=True)
                    result = {"ok": True, "code": "imported" if imported["imported"] else "already_imported", "summary": imported["summary"]}
                elif args.command == "export-sqlite":
                    result = {"ok": True, "code": "export_created", "summary": export_sqlite(repository, args.destination, apply=True)}
                else:
                    comparison = compare_sqlite(args.snapshot, repository)
                    result = {"ok": comparison["matches"], "code": "match" if comparison["matches"] else "comparison_mismatch", "comparison": comparison}
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    except LedgerTransferError as exc:
        print(json.dumps({"ok": False, "code": exc.code}), file=sys.stderr)
        return 1
    except Exception:
        print(json.dumps({"ok": False, "code": "operation_failed"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())