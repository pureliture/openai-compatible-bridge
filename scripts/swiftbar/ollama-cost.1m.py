#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DEFAULT_STATUS_URLS = (
    "http://127.0.0.1:8930/admin/cost/status",
    "http://127.0.0.1:8000/admin/cost/status",
)
INFINITY_SYMBOL = "∞"
UNLIMITED_VALUES = {"", "unlimited", INFINITY_SYMBOL}


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "openai_compatible_bridge").is_dir():
            return parent
    return None


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value or "0"))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _money(value: Any) -> str:
    amount = _decimal(value)
    if amount == 0:
        return "$0"
    rendered = format(amount.quantize(Decimal("0.000001")), "f")
    rendered = rendered.rstrip("0").rstrip(".")
    return f"${rendered or '0'}"


def _tokens(value: Any) -> str:
    amount = int(_decimal(value))
    if amount >= 1_000_000:
        rendered = f"{amount / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{rendered}M tok"
    if amount >= 1_000:
        rendered = f"{amount / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{rendered}k tok"
    return f"{amount} tok"


def _limit(value: Any) -> str:
    rendered = str(value or "").strip()
    if rendered.lower() in UNLIMITED_VALUES:
        return INFINITY_SYMBOL
    return _money(rendered)


def _status_urls() -> tuple[str, ...]:
    configured = os.getenv("OPENAI_BRIDGE_COST_STATUS_URL") or os.getenv("BRIDGE_COST_STATUS_URL")
    if configured:
        return (configured,)
    return DEFAULT_STATUS_URLS


def _admin_key() -> str | None:
    configured = os.getenv("OPENAI_BRIDGE_COST_ADMIN_API_KEY") or os.getenv("BRIDGE_COST_ADMIN_API_KEY")
    if configured:
        return configured
    key_file = Path(
        os.getenv(
            "OPENAI_BRIDGE_COST_ADMIN_API_KEY_FILE",
            "~/.config/openai-compatible-bridge/cost-admin-key",
        )
    ).expanduser()
    try:
        value = key_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _provider_status_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query["provider"] = _provider()
    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query))
    )


def _read_admin_status() -> tuple[dict[str, Any] | None, str | None]:
    key = _admin_key()
    last_error: str | None = None
    timeout = float(os.getenv("OPENAI_BRIDGE_COST_TIMEOUT_SECONDS", "2"))
    for url in _status_urls():
        status_url = _provider_status_url(url)
        request = urllib.request.Request(status_url)
        if key:
            request.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload, f"admin API: {status_url}"
        except urllib.error.HTTPError as exc:
            last_error = f"admin API HTTP {exc.code}"
        except (OSError, ValueError) as exc:
            last_error = str(exc)
    return None, last_error


def _ledger_path() -> Path:
    configured = os.getenv("OPENAI_BRIDGE_COST_LEDGER_PATH") or os.getenv("BRIDGE_COST_LEDGER_PATH")
    if configured:
        return Path(configured).expanduser()
    repo_root = _repo_root()
    if repo_root is not None:
        return repo_root / "data" / "cost-ledger.db"
    return Path.home() / "openai-compatible-bridge" / "data" / "cost-ledger.db"


def _provider() -> str:
    return os.getenv("OPENAI_BRIDGE_COST_PROVIDER", "ollama").strip().lower() or "ollama"


def _read_bridge_usage_summary(path: Path, provider: str) -> dict[str, Any] | None:
    repo_root = _repo_root()
    if repo_root is not None and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from openai_compatible_bridge.core.cost_tracking import read_provider_daily_usage_summary_from_sqlite

    summary = read_provider_daily_usage_summary_from_sqlite(path, provider)
    return None if summary is None else summary.as_dict()


def _read_ledger_status() -> tuple[dict[str, Any], str]:
    path = _ledger_path()
    limit = _limit(os.getenv("COST_DAILY_LIMIT_USD", "unlimited"))
    provider = _provider()
    if not path.exists():
        return {
            "enabled": True,
            "healthy": True,
            "currency": "USD",
            "daily": {
                "estimated_spend": "0",
                "usage_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "limit": limit,
                "blocked": False,
            },
            "short_window": {"estimated_spend": "0", "limit": "unlimited", "blocked": False},
        }, f"ledger waiting: {path}"

    daily = _read_bridge_usage_summary(path, provider)
    if daily is None:
        return {
            "enabled": True,
            "healthy": True,
            "currency": "USD",
            "daily": {
                "estimated_spend": "0",
                "usage_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "limit": limit,
                "blocked": False,
            },
            "short_window": {"estimated_spend": "0", "limit": "unlimited", "blocked": False},
        }, f"ledger schema waiting for provider column: {path}"
    daily["limit"] = limit
    daily["blocked"] = False
    return {
        "enabled": True,
        "healthy": True,
        "currency": "USD",
        "daily": daily,
        "short_window": {"estimated_spend": "0", "limit": "unlimited", "blocked": False},
    }, f"ledger: {path}"


def _print_status(status: dict[str, Any], source: str) -> None:
    daily = status.get("daily") or {}
    short_window = status.get("short_window") or {}
    spend = _money(daily.get("estimated_spend", "0"))
    usage_tokens = daily.get("usage_tokens")
    usage = spend
    daily_limit = _limit(daily.get("limit", "unlimited"))
    blocked = bool(daily.get("blocked") or short_window.get("blocked"))
    health = "healthy" if status.get("healthy", True) else "unhealthy"
    prefix = "Ollama"
    suffix = " blocked" if blocked else ""

    print(f"{prefix} {usage} / {daily_limit}{suffix}")
    print("---")
    print(f"Daily cost: {usage} / {daily_limit}")
    if usage_tokens is not None:
        print(f"Daily tokens: {_tokens(usage_tokens)}")
        print(f"Prompt tokens: {_tokens(daily.get('prompt_tokens', 0))}")
        print(f"Completion tokens: {_tokens(daily.get('completion_tokens', 0))}")
    print(f"Short window: {_money(short_window.get('estimated_spend', '0'))} / {_limit(short_window.get('limit', 'unlimited'))}")
    print(f"Health: {health}")
    print(f"Source: {source}")
    print("Refresh | refresh=true")


def main() -> int:
    status, source = _read_admin_status()
    if status is None:
        try:
            status, source = _read_ledger_status()
        except Exception:
            status = None
        if status is None:
            print("Ollama cost unavailable")
            print("---")
            print(f"Error: {source}")
            print("Refresh | refresh=true")
            return 0
    _print_status(status, source or "unknown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
