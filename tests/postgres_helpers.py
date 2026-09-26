from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import psycopg
import pytest


def _postgres_bin(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    pg_config = shutil.which("pg_config")
    if pg_config:
        result = subprocess.run([pg_config, "--bindir"], capture_output=True, text=True, timeout=10, check=False)
        candidate = Path(result.stdout.strip()) / name
        if result.returncode == 0 and candidate.is_file():
            return str(candidate)
    return None


@pytest.fixture
def pg_dsn(tmp_path):
    """외부 DSN을 사용하지 않는 테스트 전용 native PostgreSQL을 실행한다."""
    initdb = _postgres_bin("initdb")
    postgres = _postgres_bin("postgres")
    if not initdb or not postgres:
        message = "native PostgreSQL initdb/postgres가 필요합니다 (PATH 또는 pg_config)."
        if os.environ.get("COST_POSTGRES_TEST_REQUIRED") == "1":
            pytest.fail(message)
        pytest.skip(message)

    data = tmp_path / "postgres-data"
    result = subprocess.run(
        [initdb, "-D", str(data), "--auth=trust", "--username=postgres", "--no-locale", "--encoding=UTF8"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        pytest.fail("테스트 전용 PostgreSQL initdb 실행에 실패했습니다.")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    dsn = psycopg.conninfo.make_conninfo(
        host="127.0.0.1", port=port, user="postgres", dbname="postgres", connect_timeout=1,
    )
    process = subprocess.Popen(
        [
            postgres, "-D", str(data), "-h", "127.0.0.1", "-p", str(port),
            "-k", "", "-c", "fsync=off", "-c", "log_statement=none",
            "-c", "log_min_messages=panic", "-c", "log_min_error_statement=panic",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("테스트 전용 PostgreSQL이 시작 중 종료되었습니다.")
            try:
                with psycopg.connect(dsn):
                    break
            except psycopg.OperationalError:
                time.sleep(0.05)
        else:
            pytest.fail("테스트 전용 PostgreSQL 시작 대기 시간이 초과되었습니다.")
        yield dsn
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)