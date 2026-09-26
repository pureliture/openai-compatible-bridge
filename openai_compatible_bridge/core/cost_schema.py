"""전용 PostgreSQL의 빈 schema를 준비하는 명시적 운영자 명령."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .cost_tracking import CostSubsystemUnhealthy
from .postgres_cost_repository import apply_migrations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="브리지 전용 PostgreSQL schema migration (데이터 이관 없음)")
    parser.add_argument("--apply", action="store_true", required=True)
    parser.parse_args(argv)
    dsn = os.environ.get("COST_LEDGER_POSTGRES_DSN", "").strip()
    if not dsn:
        print(json.dumps({"ok": False, "code": "dsn_required"}), file=sys.stderr)
        return 1
    try:
        version = apply_migrations(dsn)
    except (CostSubsystemUnhealthy, ValueError):
        print(json.dumps({"ok": False, "code": "schema_migration_failed"}), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "version": version}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())