"""No local PostgreSQL fallback; validate server before isolated DB writes."""
from unittest.mock import MagicMock
import pytest
from tests import postgres_helpers as pg


def test_missing_setting_cannot_start_local_postgres(monkeypatch):
    monkeypatch.delenv('ATLAS_TEST_PG_DSN', raising=False)
    with pytest.raises(pytest.fail.Exception, match='ATLAS_TEST_PG_DSN'):
        next(pg.pg_dsn.__wrapped__())


def test_wrong_identity_cannot_create_or_drop_database():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ('wrong-server', 'atlas_test_control', 'test_runner', False, True, False, False, False, 'atlas-isolated-test-server:e2-tiny:v1')
    with pytest.raises(ValueError, match='identity'):
        pg.validate_test_server(conn)
    assert conn.execute.call_count == 1


def test_owned_database_created_and_cleaned_on_exception(monkeypatch):
    monkeypatch.setenv('ATLAS_TEST_PG_DSN', 'host=127.0.0.1 port=25432 user=test_runner dbname=atlas_test_control')
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ('atlas-isolated-tests-e2-tiny', 'atlas_test_control', 'test_runner', False, True, False, False, False, 'atlas-isolated-test-server:e2-tiny:v1')
    monkeypatch.setattr(pg.psycopg, 'connect', lambda *a, **k: conn)
    with pytest.raises(RuntimeError):
        with pg.isolated_test_database() as dsn:
            assert 'atlas_test_bridge_' in dsn
            assert 'atlas_test_control' not in dsn
            raise RuntimeError('test failure')
    commands = [call.args[0] for call in conn.execute.call_args_list]
    assert len(commands) == 3
    assert 'CREATE DATABASE' in str(commands[1])
    assert 'template0' in str(commands[1])
    assert 'DROP DATABASE' in str(commands[2])
    assert str(commands[1]).split('Identifier(')[1].split(')')[0] == str(commands[2]).split('Identifier(')[1].split(')')[0]
    conn.close.assert_called_once()


def test_missing_setting_always_fails_even_without_required_flag(monkeypatch):
    monkeypatch.delenv('ATLAS_TEST_PG_DSN', raising=False)
    monkeypatch.delenv('COST_POSTGRES_TEST_REQUIRED', raising=False)
    with pytest.raises(pytest.fail.Exception, match='ATLAS_TEST_PG_DSN'):
        next(pg.pg_dsn.__wrapped__())
