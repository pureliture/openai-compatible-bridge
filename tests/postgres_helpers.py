"""One remote database per test; never launch local PostgreSQL."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest


def validate_test_server(conn) -> None:
    identity = conn.execute("""
        SELECT current_setting('cluster_name'), current_database(), current_user,
               rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls,
               shobj_description(d.oid, 'pg_database')
        FROM pg_roles r CROSS JOIN pg_database d
        WHERE r.rolname = current_user AND d.datname = current_database()
    """).fetchone()
    expected = ('atlas-isolated-tests-e2-tiny', 'atlas_test_control', 'test_runner',
                False, True, False, False, False, 'atlas-isolated-test-server:e2-tiny:v1')
    if identity != expected:
        raise ValueError('test PostgreSQL identity or permissions mismatch; no database mutation allowed')


@contextmanager
def isolated_test_database():
    dsn = os.environ.get('ATLAS_TEST_PG_DSN')
    if not dsn:
        raise ValueError('ATLAS_TEST_PG_DSN must reference the approved isolated test server')
    name = 'atlas_test_bridge_' + uuid4().hex
    conn = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
    created = False
    try:
        validate_test_server(conn)
        conn.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(name)))
        created = True
        yield psycopg.conninfo.make_conninfo(dsn, dbname=name, connect_timeout=10)
    finally:
        try:
            if created:
                # A different restricted runtime role can briefly remain visible
                # after its connection closes; never require signal-backend rights.
                deadline = time.monotonic() + 10
                while True:
                    try:
                        conn.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(name)))
                        break
                    except psycopg.errors.ObjectInUse:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.05)
        finally:
            conn.close()


@pytest.fixture
def pg_dsn():
    """Require explicit remote settings; absent config never starts initdb."""
    if not os.environ.get('ATLAS_TEST_PG_DSN'):
        message = 'ATLAS_TEST_PG_DSN is required; local PostgreSQL fallback is disabled'
        pytest.fail(message)
    with isolated_test_database() as dsn:
        yield dsn
