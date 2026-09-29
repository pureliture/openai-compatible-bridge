import json

import postgres_helpers
import pytest
from test_cost_postgres_api import unreachable_dsn

from openai_compatible_bridge.core.cost_schema import main
from openai_compatible_bridge.core.postgres_cost_repository import (
    PostgresCostRepository,
)

pg_dsn = postgres_helpers.pg_dsn


def test_schema_cli_requires_explicit_apply_and_rejects_transfer():
    for args in ([], ["--apply", "import-sqlite", "private.db"], ["--apply", "export-sqlite", "private.db"]):
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 2


def test_schema_cli_missing_dsn_and_error_are_redacted(monkeypatch, capsys):
    monkeypatch.delenv("COST_LEDGER_POSTGRES_DSN", raising=False)
    assert main(["--apply"]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "dsn_required"
    with unreachable_dsn() as dsn:
        monkeypatch.setenv("COST_LEDGER_POSTGRES_DSN", dsn)
        assert main(["--apply"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"ok": False, "code": "schema_migration_failed"}


def test_schema_cli_is_idempotent_and_starts_empty(pg_dsn, monkeypatch, capsys):
    monkeypatch.setenv("COST_LEDGER_POSTGRES_DSN", pg_dsn)
    for _ in range(2):
        assert main(["--apply"]) == 0
        assert json.loads(capsys.readouterr().out) == {"ok": True, "version": 1}
    assert PostgresCostRepository(pg_dsn).fetch_events() == []