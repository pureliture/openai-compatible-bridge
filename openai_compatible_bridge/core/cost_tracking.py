from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, ContextManager, Iterator, Mapping, Protocol, runtime_checkable


LEDGER_ALLOWED_FIELDS: tuple[str, ...] = (
    "event_id",
    "reservation_id",
    "internal_request_id",
    "endpoint",
    "model",
    "status",
    "billing_eligible",
    "limit_type",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "embedding_tokens",
    "rerank_units",
    "forecast_cost_usd",
    "estimated_cost_usd",
    "currency",
    "pricing_source",
    "pricing_version",
    "window_started_at",
    "created_at",
    "finalized_at",
    "reconciliation_status",
)

_TEXT_COLUMNS = {
    "event_id",
    "reservation_id",
    "internal_request_id",
    "endpoint",
    "model",
    "status",
    "limit_type",
    "forecast_cost_usd",
    "estimated_cost_usd",
    "currency",
    "pricing_source",
    "pricing_version",
    "window_started_at",
    "created_at",
    "finalized_at",
    "reconciliation_status",
}

_INTEGER_COLUMNS = {
    "billing_eligible",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "embedding_tokens",
    "rerank_units",
}

_COST_LOGGER = logging.getLogger("cost_tracking")


class CostTrackingError(Exception):
    """Base error for cost tracking failures."""


class CostConfigError(CostTrackingError):
    """Raised when cost tracking config is unsafe or incomplete."""


class CostLedgerValidationError(CostTrackingError):
    """Raised when a ledger event violates the allowlist contract."""


class CostSubsystemUnhealthy(CostTrackingError):
    """Raised when cost tracking can no longer enforce hard limits safely."""


class BillingExportPermissionDenied(CostTrackingError):
    """Raised when the billing export adapter lacks BigQuery read/job permissions."""


@dataclass(frozen=True)
class BudgetBlock:
    limit_type: str
    reset_at: str
    current_estimated_spend: Decimal
    configured_limit: Decimal
    currency: str = "USD"


class CostBudgetExceeded(CostTrackingError):
    def __init__(self, block: BudgetBlock) -> None:
        super().__init__(f"cost budget exceeded: {block.limit_type}")
        self.block = block


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _require_nonempty(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise CostConfigError(f"{name} is required when cost tracking is enabled")
    return value


def _parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CostConfigError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise CostConfigError(f"{name} must be > 0")
    return parsed


def _parse_decimal(value: Any, name: str, *, allow_zero: bool) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CostConfigError(f"{name} must be a decimal number") from exc
    if not parsed.is_finite():
        raise CostConfigError(f"{name} must be finite")
    if parsed < 0 or (parsed == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise CostConfigError(f"{name} must be {comparator}")
    return parsed


@dataclass(frozen=True)
class CostTrackingConfig:
    enabled: bool
    ledger_path: Path | None = None
    pricing_json: str | None = None
    pricing_path: Path | None = None
    short_window_seconds: int | None = None
    short_window_limit_usd: Decimal | None = None
    daily_limit_usd: Decimal | None = None
    request_retention_days: int = 90
    aggregate_retention_months: int = 13
    admin_enabled: bool = False
    admin_api_key: str | None = None
    reconciliation_enabled: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "CostTrackingConfig":
        enabled = _parse_bool(env.get("COST_TRACKING_ENABLED"))
        admin_enabled = _parse_bool(env.get("COST_ADMIN_ENABLED"))
        reconciliation_enabled = _parse_bool(env.get("COST_RECONCILIATION_ENABLED"))

        if not enabled:
            if admin_enabled:
                raise CostConfigError("COST_ADMIN_ENABLED requires COST_TRACKING_ENABLED=true")
            return cls(enabled=False, reconciliation_enabled=reconciliation_enabled)

        ledger_path = Path(_require_nonempty(env, "COST_LEDGER_PATH"))
        pricing_json = env.get("COST_PRICING_JSON", "").strip() or None
        pricing_path_raw = env.get("COST_PRICING_PATH", "").strip()
        pricing_path = Path(pricing_path_raw) if pricing_path_raw else None
        if pricing_json is None and pricing_path is None:
            raise CostConfigError("COST_PRICING_JSON or COST_PRICING_PATH is required")

        short_window_seconds = _parse_positive_int(
            _require_nonempty(env, "COST_SHORT_WINDOW_SECONDS"),
            "COST_SHORT_WINDOW_SECONDS",
        )
        short_window_limit_usd = _parse_decimal(
            _require_nonempty(env, "COST_SHORT_WINDOW_LIMIT_USD"),
            "COST_SHORT_WINDOW_LIMIT_USD",
            allow_zero=False,
        )
        daily_limit_usd = _parse_decimal(
            _require_nonempty(env, "COST_DAILY_LIMIT_USD"),
            "COST_DAILY_LIMIT_USD",
            allow_zero=False,
        )
        request_retention_days = _parse_positive_int(
            env.get("COST_RETENTION_REQUEST_DAYS", "90"),
            "COST_RETENTION_REQUEST_DAYS",
        )
        aggregate_retention_months = _parse_positive_int(
            env.get("COST_RETENTION_AGGREGATE_MONTHS", "13"),
            "COST_RETENTION_AGGREGATE_MONTHS",
        )

        admin_api_key = env.get("COST_ADMIN_API_KEY", "").strip() or None
        if admin_enabled and admin_api_key is None:
            raise CostConfigError("COST_ADMIN_API_KEY is required when COST_ADMIN_ENABLED=true")

        return cls(
            enabled=True,
            ledger_path=ledger_path,
            pricing_json=pricing_json,
            pricing_path=pricing_path,
            short_window_seconds=short_window_seconds,
            short_window_limit_usd=short_window_limit_usd,
            daily_limit_usd=daily_limit_usd,
            request_retention_days=request_retention_days,
            aggregate_retention_months=aggregate_retention_months,
            admin_enabled=admin_enabled,
            admin_api_key=admin_api_key,
            reconciliation_enabled=reconciliation_enabled,
        )


@dataclass(frozen=True)
class PricingEntry:
    model: str
    endpoint: str
    input_per_million: Decimal = Decimal("0")
    output_per_million: Decimal = Decimal("0")
    embedding_per_million: Decimal = Decimal("0")
    rerank_per_unit: Decimal = Decimal("0")
    currency: str = "USD"
    source: str = "manual"
    version: str = "unversioned"


class PricingCatalog:
    def __init__(self, entries: Mapping[tuple[str, str], PricingEntry]) -> None:
        self._entries = dict(entries)

    @classmethod
    def from_json(cls, raw: str) -> "PricingCatalog":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CostConfigError("COST_PRICING_JSON must be valid JSON") from exc
        if not isinstance(data, dict):
            raise CostConfigError("pricing config must be a JSON object")

        source = str(data.get("source") or "manual")
        version = str(data.get("version") or "unversioned")
        currency = str(data.get("currency") or "USD").upper()
        if currency != "USD":
            raise CostConfigError("only USD pricing is supported")

        models = data.get("models")
        if not isinstance(models, dict) or not models:
            raise CostConfigError("pricing config requires a non-empty models object")

        entries: dict[tuple[str, str], PricingEntry] = {}
        for model, endpoint_map in models.items():
            if not isinstance(endpoint_map, dict):
                raise CostConfigError(f"pricing for {model!r} must be an object")
            for endpoint, price in endpoint_map.items():
                if not isinstance(price, dict):
                    raise CostConfigError(f"pricing for {model!r}/{endpoint!r} must be an object")
                entry_currency = str(price.get("currency") or currency).upper()
                if entry_currency != "USD":
                    raise CostConfigError(f"only USD pricing is supported for {model}/{endpoint}")
                entry = PricingEntry(
                    model=str(model),
                    endpoint=str(endpoint),
                    input_per_million=_price_decimal(price, "input_per_million"),
                    output_per_million=_price_decimal(price, "output_per_million"),
                    embedding_per_million=_price_decimal(price, "embedding_per_million"),
                    rerank_per_unit=_price_decimal(price, "rerank_per_unit"),
                    currency=entry_currency,
                    source=str(price.get("source") or source),
                    version=str(price.get("version") or version),
                )
                entries[(entry.model, entry.endpoint)] = entry

        if not entries:
            raise CostConfigError("pricing config contains no entries")
        return cls(entries)

    @classmethod
    def from_file(cls, path: Path) -> "PricingCatalog":
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_config(cls, config: CostTrackingConfig) -> "PricingCatalog":
        if config.pricing_json:
            return cls.from_json(config.pricing_json)
        if config.pricing_path:
            return cls.from_file(config.pricing_path)
        raise CostConfigError("pricing config is missing")

    def price_for(self, *, model: str, endpoint: str) -> PricingEntry:
        exact_key = (model, endpoint)
        if exact_key in self._entries:
            return self._entries[exact_key]

        for fallback_model in _pricing_model_fallbacks(model):
            fallback_key = (fallback_model, endpoint)
            if fallback_key in self._entries:
                return self._entries[fallback_key]

        raise CostConfigError(f"missing pricing for {model}/{endpoint}")


def _pricing_model_fallbacks(model: str) -> tuple[str, ...]:
    if model.startswith("ollama:") and model != "ollama:*":
        return ("ollama:*",)
    return ()


def _price_decimal(price: Mapping[str, Any], name: str) -> Decimal:
    value = price.get(name, 0)
    return _parse_decimal(value, name, allow_zero=True)


@dataclass(frozen=True)
class NormalizedUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    rerank_units: int = 0

    def __post_init__(self) -> None:
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "embedding_tokens",
            "rerank_units",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")


@dataclass(frozen=True)
class CostReservation:
    reservation_id: str
    internal_request_id: str
    endpoint: str
    model: str
    forecast_cost_usd: Decimal
    currency: str
    pricing_source: str
    pricing_version: str
    window_started_at: str
    created_at: str


@dataclass(frozen=True)
class ReconciliationResult:
    day: str
    wrapper_estimated_cost_usd: Decimal | None
    billing_export_cost: Decimal | None
    delta_usd: Decimal | None
    status: str
    checked_at: str
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "wrapper_estimated_cost_usd": (
                None if self.wrapper_estimated_cost_usd is None else str(self.wrapper_estimated_cost_usd)
            ),
            "billing_export_cost": None if self.billing_export_cost is None else str(self.billing_export_cost),
            "delta_usd": None if self.delta_usd is None else str(self.delta_usd),
            "status": self.status,
            "checked_at": self.checked_at,
            "error_message": self.error_message,
        }


class CostEstimator:
    def __init__(self, pricing: PricingCatalog) -> None:
        self.pricing = pricing

    def estimate(self, *, model: str, endpoint: str, usage: NormalizedUsage) -> Decimal:
        price = self.pricing.price_for(model=model, endpoint=endpoint)
        return (
            Decimal(usage.prompt_tokens) * price.input_per_million / Decimal(1_000_000)
            + Decimal(usage.completion_tokens) * price.output_per_million / Decimal(1_000_000)
            + Decimal(usage.embedding_tokens) * price.embedding_per_million / Decimal(1_000_000)
            + Decimal(usage.rerank_units) * price.rerank_per_unit
        )


@dataclass
class CostSubsystemHealth:
    healthy: bool = True
    reason: str | None = None

    def ensure_healthy(self) -> None:
        if not self.healthy:
            raise CostSubsystemUnhealthy(self.reason or "cost subsystem is unhealthy")

    def mark_unhealthy(self, reason: str) -> None:
        self.healthy = False
        self.reason = reason


@runtime_checkable
class ICostRepository(Protocol):
    def transaction(self) -> ContextManager[None]:
        ...

    def initialize(self) -> None:
        ...

    def close(self) -> None:
        ...

    def record_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        ...

    def prepare_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        ...

    def insert_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        ...

    def fetch_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        ...

    def sum_estimated_since(self, cutoff: datetime, statuses: tuple[str, ...] | None = None) -> Decimal:
        ...

    def daily_estimated_spend(self, day: str | date) -> Decimal:
        ...

    def record_reconciliation_result(self, result: ReconciliationResult) -> None:
        ...

    def latest_reconciliation_result(self) -> dict[str, Any] | None:
        ...

    def prune(self, *, now: datetime | None = None) -> dict[str, int]:
        ...

    def update_reservation(
        self,
        reservation_id: str,
        *,
        status: str,
        billing_eligible: bool,
        usage: NormalizedUsage,
        estimated_cost_usd: Decimal,
        finalized_at: str,
    ) -> None:
        ...



class SQLiteCostRepository(ICostRepository):
    def __init__(
        self,
        path: Path,
        *,
        request_retention_days: int = 90,
        aggregate_retention_months: int = 13,
        now_fn: Any = _utcnow,
    ) -> None:
        self.path = Path(path)
        self.request_retention_days = request_retention_days
        self.aggregate_retention_months = aggregate_retention_months
        self._now_fn = now_fn
        self._conn: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.initialize()
        assert self._conn is not None
        return self._conn

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        conn = self.connection
        in_trans = conn.in_transaction
        if not in_trans:
            conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            if not in_trans:
                conn.commit()
        except Exception:
            if not in_trans:
                conn.rollback()
            raise

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(_create_cost_events_sql())
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cost_daily_aggregates (
              day TEXT PRIMARY KEY,
              estimated_cost_usd TEXT NOT NULL,
              currency TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cost_reconciliation_results (
              day TEXT PRIMARY KEY,
              wrapper_estimated_cost_usd TEXT,
              billing_export_cost TEXT,
              delta_usd TEXT,
              status TEXT NOT NULL,
              checked_at TEXT NOT NULL,
              error_message TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cost_events_created_at ON cost_events(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cost_events_status ON cost_events(status)")
        conn.commit()
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def record_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        with self.transaction():
            return self.insert_event(fields)

    def prepare_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(fields) - set(LEDGER_ALLOWED_FIELDS))
        if unknown:
            raise CostLedgerValidationError(f"cost ledger event has non-allowlisted fields: {unknown}")

        row = {name: fields.get(name) for name in LEDGER_ALLOWED_FIELDS}
        row["event_id"] = row["event_id"] or f"costevt-{uuid.uuid4().hex}"
        row["created_at"] = row["created_at"] or _iso(self._now_fn())
        return {name: _normalize_ledger_value(name, value) for name, value in row.items()}

    def insert_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        normalized = self.prepare_event(fields)
        placeholders = ", ".join("?" for _ in LEDGER_ALLOWED_FIELDS)
        columns = ", ".join(LEDGER_ALLOWED_FIELDS)
        values = [normalized[name] for name in LEDGER_ALLOWED_FIELDS]
        self.connection.execute(f"INSERT INTO cost_events ({columns}) VALUES ({placeholders})", values)
        return normalized

    def fetch_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM cost_events ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def sum_estimated_since(self, cutoff: datetime, statuses: tuple[str, ...] | None = None) -> Decimal:
        if statuses is None:
            row = self.connection.execute(
                """
                SELECT estimated_cost_usd
                FROM cost_events
                WHERE created_at >= ? AND billing_eligible = 1
                """,
                (_iso(cutoff),),
            ).fetchall()
        else:
            placeholders = ", ".join("?" for _ in statuses)
            row = self.connection.execute(
                f"""
                SELECT estimated_cost_usd
                FROM cost_events
                WHERE created_at >= ?
                  AND billing_eligible = 1
                  AND status IN ({placeholders})
                """,
                (_iso(cutoff), *statuses),
            ).fetchall()
        total = Decimal("0")
        for item in row:
            if item["estimated_cost_usd"] is not None:
                total += Decimal(str(item["estimated_cost_usd"]))
        return total

    def daily_estimated_spend(self, day: str | date) -> Decimal:
        day_value = _coerce_day(day)
        start = datetime(day_value.year, day_value.month, day_value.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        rows = self.connection.execute(
            """
            SELECT estimated_cost_usd
            FROM cost_events
            WHERE created_at >= ?
              AND created_at < ?
              AND billing_eligible = 1
              AND status IN ('reserved', 'finalized', 'estimated_only')
            """,
            (_iso(start), _iso(end)),
        ).fetchall()
        total = Decimal("0")
        for row in rows:
            if row["estimated_cost_usd"] is not None:
                total += Decimal(str(row["estimated_cost_usd"]))
        return total

    def update_reservation(
        self,
        reservation_id: str,
        *,
        status: str,
        billing_eligible: bool,
        usage: NormalizedUsage,
        estimated_cost_usd: Decimal,
        finalized_at: str,
    ) -> None:
        total_tokens = usage.total_tokens or usage.prompt_tokens + usage.completion_tokens + usage.embedding_tokens
        cursor = self.connection.execute(
            """
            UPDATE cost_events
            SET status = ?,
                billing_eligible = ?,
                prompt_tokens = ?,
                completion_tokens = ?,
                total_tokens = ?,
                embedding_tokens = ?,
                rerank_units = ?,
                estimated_cost_usd = ?,
                finalized_at = ?
            WHERE reservation_id = ?
            """,
            (
                status,
                int(billing_eligible),
                usage.prompt_tokens,
                usage.completion_tokens,
                total_tokens,
                usage.embedding_tokens,
                usage.rerank_units,
                str(estimated_cost_usd),
                finalized_at,
                reservation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise CostTrackingError(f"reservation not found: {reservation_id}")


    def record_reconciliation_result(self, result: ReconciliationResult) -> None:
        self.connection.execute(
            """
            INSERT INTO cost_reconciliation_results (
              day,
              wrapper_estimated_cost_usd,
              billing_export_cost,
              delta_usd,
              status,
              checked_at,
              error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(day) DO UPDATE SET
              wrapper_estimated_cost_usd = excluded.wrapper_estimated_cost_usd,
              billing_export_cost = excluded.billing_export_cost,
              delta_usd = excluded.delta_usd,
              status = excluded.status,
              checked_at = excluded.checked_at,
              error_message = excluded.error_message
            """,
            (
                result.day,
                None if result.wrapper_estimated_cost_usd is None else str(result.wrapper_estimated_cost_usd),
                None if result.billing_export_cost is None else str(result.billing_export_cost),
                None if result.delta_usd is None else str(result.delta_usd),
                result.status,
                result.checked_at,
                result.error_message,
            ),
        )
        self.connection.commit()

    def latest_reconciliation_result(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT *
            FROM cost_reconciliation_results
            ORDER BY checked_at DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row is not None else None

    def prune(self, *, now: datetime | None = None) -> dict[str, int]:
        current = now or self._now_fn()
        request_cutoff = _iso(current - timedelta(days=self.request_retention_days))
        aggregate_cutoff = (current - timedelta(days=self.aggregate_retention_months * 31)).date().isoformat()

        event_cursor = self.connection.execute(
            "DELETE FROM cost_events WHERE created_at < ?",
            (request_cutoff,),
        )
        aggregate_cursor = self.connection.execute(
            "DELETE FROM cost_daily_aggregates WHERE day < ?",
            (aggregate_cutoff,),
        )
        reconciliation_cursor = self.connection.execute(
            "DELETE FROM cost_reconciliation_results WHERE day < ?",
            (aggregate_cutoff,),
        )
        self.connection.commit()
        return {
            "cost_events": event_cursor.rowcount,
            "cost_daily_aggregates": aggregate_cursor.rowcount,
            "cost_reconciliation_results": reconciliation_cursor.rowcount,
        }


CostLedger = SQLiteCostRepository


class InMemoryCostRepository(SQLiteCostRepository, ICostRepository):
    def __init__(
        self,
        *,
        request_retention_days: int = 90,
        aggregate_retention_months: int = 13,
        now_fn: Any = _utcnow,
    ) -> None:
        super().__init__(
            path=Path(":memory:"),
            request_retention_days=request_retention_days,
            aggregate_retention_months=aggregate_retention_months,
            now_fn=now_fn,
        )


class CostAccountingStore:
    def __init__(self, ledger: ICostRepository, health: CostSubsystemHealth | None = None) -> None:
        self.ledger = ledger
        self.health = health or CostSubsystemHealth()

    def ensure_healthy(self) -> None:
        self.health.ensure_healthy()

    def mark_unhealthy(self, reason: str) -> None:
        self.health.mark_unhealthy(reason)

    def record_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        self.health.ensure_healthy()
        try:
            return self.ledger.record_event(fields)
        except Exception as exc:
            self.health.mark_unhealthy(f"cost ledger write failed: {exc}")
            raise CostSubsystemUnhealthy(self.health.reason) from exc


class ReconciliationJob:
    def __init__(
        self,
        *,
        ledger: ICostRepository,
        billing_adapter: Any | None,
        tolerance_usd: Decimal = Decimal("0.01"),
        now_fn: Any = _utcnow,
    ) -> None:
        self.ledger = ledger
        self.billing_adapter = billing_adapter
        self.tolerance_usd = tolerance_usd
        self._now_fn = now_fn

    def reconcile_day(self, day: str | date) -> ReconciliationResult:
        day_value = _coerce_day(day)
        day_key = day_value.isoformat()
        checked_at = _iso(self._now_fn())

        if self.billing_adapter is None:
            result = ReconciliationResult(
                day=day_key,
                wrapper_estimated_cost_usd=None,
                billing_export_cost=None,
                delta_usd=None,
                status="unavailable",
                checked_at=checked_at,
                error_message="billing export adapter is not configured",
            )
            self.ledger.record_reconciliation_result(result)
            return result

        wrapper_cost = self.ledger.daily_estimated_spend(day_value)
        try:
            billing_cost = self.billing_adapter.daily_cost(day_key)
        except BillingExportPermissionDenied as exc:
            result = ReconciliationResult(
                day=day_key,
                wrapper_estimated_cost_usd=wrapper_cost,
                billing_export_cost=None,
                delta_usd=None,
                status="error",
                checked_at=checked_at,
                error_message=f"permission denied: {exc}",
            )
            self.ledger.record_reconciliation_result(result)
            return result
        except Exception as exc:
            result = ReconciliationResult(
                day=day_key,
                wrapper_estimated_cost_usd=wrapper_cost,
                billing_export_cost=None,
                delta_usd=None,
                status="error",
                checked_at=checked_at,
                error_message=str(exc),
            )
            self.ledger.record_reconciliation_result(result)
            return result

        if billing_cost is None:
            result = ReconciliationResult(
                day=day_key,
                wrapper_estimated_cost_usd=wrapper_cost,
                billing_export_cost=None,
                delta_usd=None,
                status="pending",
                checked_at=checked_at,
                error_message=None,
            )
            self.ledger.record_reconciliation_result(result)
            return result

        billing_decimal = _parse_decimal(billing_cost, "billing_export_cost", allow_zero=True)
        delta = wrapper_cost - billing_decimal
        status = "matched" if abs(delta) <= self.tolerance_usd else "mismatch"
        result = ReconciliationResult(
            day=day_key,
            wrapper_estimated_cost_usd=wrapper_cost,
            billing_export_cost=billing_decimal,
            delta_usd=delta,
            status=status,
            checked_at=checked_at,
            error_message=None,
        )
        self.ledger.record_reconciliation_result(result)
        return result


class BudgetReservationContext:
    def __init__(
        self,
        accounting: Any,
        endpoint: str,
        model: str,
        forecast_usage: NormalizedUsage,
    ) -> None:
        self.accounting = accounting
        self.endpoint = endpoint
        self.model = model
        self.forecast_usage = forecast_usage

        self.current_reservation: CostReservation | None = None
        self.finalized: bool = False
        self.is_stream: bool = False
        self.stream_saw_event: bool = False

        self.actual_usage: NormalizedUsage | None = None
        self.estimated_reason: str | None = None
        self._preflight_done: bool = False

    def preflight_now(self) -> "BudgetReservationContext":
        if self._preflight_done:
            return self
        if self.accounting and getattr(self.accounting, "enabled", False):
            self.current_reservation = self.accounting.preflight(
                endpoint=self.endpoint,
                model=self.model,
                forecast_usage=self.forecast_usage,
            )
        self._preflight_done = True
        return self

    async def __aenter__(self) -> "BudgetReservationContext":
        self.preflight_now()
        return self

    @staticmethod
    def _has_actual_usage(usage: NormalizedUsage) -> bool:
        return bool(
            usage.total_tokens
            or usage.prompt_tokens
            or usage.completion_tokens
            or usage.embedding_tokens
            or usage.rerank_units
        )

    def _finalize_with_usage_or_estimate(
        self,
        usage: NormalizedUsage,
        estimated_reason: str,
        *,
        accept_empty_usage: bool,
    ) -> None:
        if accept_empty_usage or self._has_actual_usage(usage):
            self.accounting.finalize_success(self.current_reservation, usage)
            return
        self.accounting.finalize_estimated_only(self.current_reservation, estimated_reason)

    def _success_estimated_reason(self) -> str:
        if self.estimated_reason is not None:
            return self.estimated_reason
        if self.is_stream:
            return "stream_missing_usage"
        return "missing_usage"

    def _finalize_successful_exit(self) -> None:
        estimated_reason = self._success_estimated_reason()
        if self.actual_usage is not None:
            self._finalize_with_usage_or_estimate(
                self.actual_usage,
                estimated_reason,
                accept_empty_usage=self.is_stream,
            )
            return
        self.accounting.finalize_estimated_only(self.current_reservation, estimated_reason)

    def _finalize_failed_exit(self, release_reason: str = "upstream_error") -> None:
        if self.is_stream and self.stream_saw_event:
            self.accounting.finalize_estimated_only(
                self.current_reservation,
                "stream_error_after_start",
            )
            return
        self.accounting.release_nonbillable(
            self.current_reservation,
            release_reason,
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if self.finalized:
            return False

        if exc_type is None:
            self._finalize_successful_exit()
        else:
            self._finalize_failed_exit()
        self.finalized = True
        return False

    def finalize_interrupted_stream(self, reason: str) -> None:
        if self.finalized:
            return
        if self.actual_usage is not None:
            self._finalize_successful_exit()
        else:
            self._finalize_failed_exit(release_reason=reason)
        self.finalized = True

    def complete(self, usage: NormalizedUsage) -> None:
        self.actual_usage = usage

    def complete_attempt(self, usage: NormalizedUsage, estimated_reason: str) -> None:
        if self.finalized:
            return
        self._finalize_with_usage_or_estimate(
            usage,
            estimated_reason,
            accept_empty_usage=False,
        )
        self.finalized = True

    def release_attempt(self, reason: str) -> None:
        if self.finalized:
            return
        self.accounting.release_nonbillable(self.current_reservation, reason)
        self.finalized = True

    def renew(self, *, model: str, forecast_usage: NormalizedUsage) -> None:
        if not self.finalized and self.current_reservation is not None:
            self.accounting.release_nonbillable(self.current_reservation, "replaced")
            self.finalized = True

        if self.accounting and getattr(self.accounting, "enabled", False):
            self.current_reservation = self.accounting.preflight(
                endpoint=self.endpoint,
                model=model,
                forecast_usage=forecast_usage,
            )
            self.finalized = False
        else:
            self.current_reservation = None
            self.finalized = False


class DisabledCostAccounting:
    enabled = False

    def reservation(
        self,
        *,
        endpoint: str,
        model: str,
        forecast_usage: NormalizedUsage,
    ) -> BudgetReservationContext:
        return BudgetReservationContext(self, endpoint, model, forecast_usage)

    def preflight(self, *, endpoint: str, model: str, forecast_usage: NormalizedUsage) -> None:
        return None

    def finalize_success(
        self,
        reservation: CostReservation | None,
        usage: NormalizedUsage,
        response_status: int = 200,
    ) -> None:
        return None

    def release_nonbillable(self, reservation: CostReservation | None, reason: str) -> None:
        return None

    def finalize_estimated_only(self, reservation: CostReservation | None, reason: str) -> None:
        return None

    def mark_unhealthy(self, reason: str) -> None:
        return None

    def close(self) -> None:
        return None

    def admin_status(self) -> dict[str, Any]:
        return {"enabled": False}

    def admin_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return []

    def admin_reconciliation(self) -> dict[str, Any]:
        return {"status": "unavailable"}


class MisconfiguredCostAccounting(DisabledCostAccounting):
    enabled = True

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def preflight(self, *, endpoint: str, model: str, forecast_usage: NormalizedUsage) -> None:
        raise CostConfigError(self.reason)


class BudgetGate:
    enabled = True

    def reservation(
        self,
        *,
        endpoint: str,
        model: str,
        forecast_usage: NormalizedUsage,
    ) -> BudgetReservationContext:
        return BudgetReservationContext(self, endpoint, model, forecast_usage)

    def __init__(
        self,
        *,
        config: CostTrackingConfig,
        ledger: ICostRepository,
        pricing: PricingCatalog,
        health: CostSubsystemHealth | None = None,
        now_fn: Any = _utcnow,
    ) -> None:
        if not config.enabled:
            raise CostConfigError("BudgetGate requires enabled cost tracking config")
        if config.short_window_seconds is None:
            raise CostConfigError("short window seconds are required")
        if config.short_window_limit_usd is None:
            raise CostConfigError("short window limit is required")
        if config.daily_limit_usd is None:
            raise CostConfigError("daily limit is required")
        self.config = config
        self.ledger = ledger
        self.estimator = CostEstimator(pricing)
        self.health = health or CostSubsystemHealth()
        self._now_fn = now_fn

    def preflight(
        self,
        *,
        endpoint: str,
        model: str,
        forecast_usage: NormalizedUsage,
    ) -> CostReservation:
        self.health.ensure_healthy()
        price = self.estimator.pricing.price_for(model=model, endpoint=endpoint)
        forecast_cost = self.estimator.estimate(model=model, endpoint=endpoint, usage=forecast_usage)
        now = self._now_fn()
        created_at = _iso(now)
        short_cutoff = now - timedelta(seconds=self.config.short_window_seconds or 0)
        day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        reservation_id = f"costres-{uuid.uuid4().hex}"
        internal_request_id = f"req-{uuid.uuid4().hex}"
        window_started_at = _iso(short_cutoff)

        block_to_raise: BudgetBlock | None = None
        try:
            with self.ledger.transaction():
                short_spend = self.ledger.sum_estimated_since(
                    short_cutoff,
                    statuses=("reserved", "finalized", "estimated_only")
                )
                daily_spend = self.ledger.sum_estimated_since(
                    day_start,
                    statuses=("reserved", "finalized", "estimated_only")
                )

                short_limit = self.config.short_window_limit_usd or Decimal("0")
                daily_limit = self.config.daily_limit_usd or Decimal("0")
                block: BudgetBlock | None = None
                if short_spend + forecast_cost > short_limit:
                    block = BudgetBlock(
                        limit_type="short_window",
                        reset_at=_iso(now + timedelta(seconds=self.config.short_window_seconds or 0)),
                        current_estimated_spend=short_spend,
                        configured_limit=short_limit,
                        currency=price.currency,
                    )
                elif daily_spend + forecast_cost > daily_limit:
                    block = BudgetBlock(
                        limit_type="daily",
                        reset_at=_iso(day_start + timedelta(days=1)),
                        current_estimated_spend=daily_spend,
                        configured_limit=daily_limit,
                        currency=price.currency,
                    )

                if block is not None:
                    self.ledger.insert_event(
                        {
                            "reservation_id": reservation_id,
                            "internal_request_id": internal_request_id,
                            "endpoint": endpoint,
                            "model": model,
                            "status": "blocked",
                            "billing_eligible": False,
                            "limit_type": block.limit_type,
                            "prompt_tokens": forecast_usage.prompt_tokens,
                            "completion_tokens": forecast_usage.completion_tokens,
                            "total_tokens": forecast_usage.total_tokens,
                            "embedding_tokens": forecast_usage.embedding_tokens,
                            "rerank_units": forecast_usage.rerank_units,
                            "forecast_cost_usd": forecast_cost,
                            "estimated_cost_usd": Decimal("0"),
                            "currency": price.currency,
                            "pricing_source": price.source,
                            "pricing_version": price.version,
                            "window_started_at": window_started_at,
                            "created_at": created_at,
                        }
                    )
                    block_to_raise = block
                else:
                    self.ledger.insert_event(
                        {
                            "reservation_id": reservation_id,
                            "internal_request_id": internal_request_id,
                            "endpoint": endpoint,
                            "model": model,
                            "status": "reserved",
                            "billing_eligible": True,
                            "prompt_tokens": forecast_usage.prompt_tokens,
                            "completion_tokens": forecast_usage.completion_tokens,
                            "total_tokens": forecast_usage.total_tokens,
                            "embedding_tokens": forecast_usage.embedding_tokens,
                            "rerank_units": forecast_usage.rerank_units,
                            "forecast_cost_usd": forecast_cost,
                            "estimated_cost_usd": forecast_cost,
                            "currency": price.currency,
                            "pricing_source": price.source,
                            "pricing_version": price.version,
                            "window_started_at": window_started_at,
                            "created_at": created_at,
                        }
                    )
        except Exception as exc:
            self.health.mark_unhealthy(f"cost ledger preflight failed: {exc}")
            raise CostSubsystemUnhealthy(self.health.reason) from exc

        if block_to_raise is not None:
            _log_cost_event(
                "blocked",
                endpoint=endpoint,
                model=model,
                status="blocked",
                reservation_id=reservation_id,
                estimated_cost_usd=Decimal("0"),
                forecast_cost_usd=forecast_cost,
                limit_type=block_to_raise.limit_type,
                billing_eligible=False,
            )
            raise CostBudgetExceeded(block_to_raise)

        _log_cost_event(
            "reserved",
            endpoint=endpoint,
            model=model,
            status="reserved",
            reservation_id=reservation_id,
            estimated_cost_usd=forecast_cost,
            forecast_cost_usd=forecast_cost,
            billing_eligible=True,
        )
        return CostReservation(
            reservation_id=reservation_id,
            internal_request_id=internal_request_id,
            endpoint=endpoint,
            model=model,
            forecast_cost_usd=forecast_cost,
            currency=price.currency,
            pricing_source=price.source,
            pricing_version=price.version,
            window_started_at=window_started_at,
            created_at=created_at,
        )

    def finalize_success(
        self,
        reservation: CostReservation | None,
        usage: NormalizedUsage,
        response_status: int = 200,
    ) -> None:
        if reservation is None:
            return
        try:
            estimated_cost = self.estimator.estimate(
                model=reservation.model,
                endpoint=reservation.endpoint,
                usage=usage,
            )
            total_tokens = usage.total_tokens or usage.prompt_tokens + usage.completion_tokens + usage.embedding_tokens
            finalized_at = _iso(self._now_fn())
            with self.ledger.transaction():
                self.ledger.update_reservation(
                    reservation.reservation_id,
                    status="finalized",
                    billing_eligible=True,
                    usage=NormalizedUsage(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=total_tokens,
                        embedding_tokens=usage.embedding_tokens,
                        rerank_units=usage.rerank_units,
                    ),
                    estimated_cost_usd=estimated_cost,
                    finalized_at=finalized_at,
                )
            _log_cost_event(
                "finalized",
                endpoint=reservation.endpoint,
                model=reservation.model,
                status="finalized",
                reservation_id=reservation.reservation_id,
                estimated_cost_usd=estimated_cost,
                forecast_cost_usd=reservation.forecast_cost_usd,
                billing_eligible=True,
            )
        except Exception as exc:
            self.health.mark_unhealthy(f"cost ledger finalization failed: {exc}")

    def release_nonbillable(self, reservation: CostReservation | None, reason: str) -> None:
        if reservation is None:
            return
        try:
            finalized_at = _iso(self._now_fn())
            with self.ledger.transaction():
                self.ledger.update_reservation(
                    reservation.reservation_id,
                    status=f"released_{reason}",
                    billing_eligible=False,
                    usage=NormalizedUsage(),
                    estimated_cost_usd=Decimal("0"),
                    finalized_at=finalized_at,
                )
            _log_cost_event(
                "released",
                endpoint=reservation.endpoint,
                model=reservation.model,
                status=f"released_{reason}",
                reservation_id=reservation.reservation_id,
                estimated_cost_usd=Decimal("0"),
                forecast_cost_usd=reservation.forecast_cost_usd,
                billing_eligible=False,
            )
        except Exception as exc:
            self.health.mark_unhealthy(f"cost ledger release failed: {exc}")

    def finalize_estimated_only(self, reservation: CostReservation | None, reason: str) -> None:
        if reservation is None:
            return
        try:
            finalized_at = _iso(self._now_fn())
            with self.ledger.transaction():
                self.ledger.update_reservation(
                    reservation.reservation_id,
                    status="estimated_only",
                    billing_eligible=True,
                    usage=NormalizedUsage(),
                    estimated_cost_usd=reservation.forecast_cost_usd,
                    finalized_at=finalized_at,
                )
            _log_cost_event(
                "estimated_only",
                endpoint=reservation.endpoint,
                model=reservation.model,
                status="estimated_only",
                reservation_id=reservation.reservation_id,
                estimated_cost_usd=reservation.forecast_cost_usd,
                forecast_cost_usd=reservation.forecast_cost_usd,
                billing_eligible=True,
            )
        except Exception as exc:
            self.health.mark_unhealthy(f"cost ledger estimated-only finalization failed: {exc}")

    def mark_unhealthy(self, reason: str) -> None:
        self.health.mark_unhealthy(reason)

    def close(self) -> None:
        self.ledger.close()

    def admin_status(self) -> dict[str, Any]:
        now = self._now_fn()
        short_cutoff = now - timedelta(seconds=self.config.short_window_seconds or 0)
        day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        short_spend = self.ledger.sum_estimated_since(short_cutoff)
        daily_spend = self.ledger.sum_estimated_since(day_start)
        short_limit = self.config.short_window_limit_usd or Decimal("0")
        daily_limit = self.config.daily_limit_usd or Decimal("0")
        return {
            "enabled": True,
            "healthy": self.health.healthy,
            "unhealthy_reason": self.health.reason,
            "currency": "USD",
            "short_window": {
                "seconds": self.config.short_window_seconds,
                "estimated_spend": str(short_spend),
                "limit": str(short_limit),
                "reset_at": _iso(now + timedelta(seconds=self.config.short_window_seconds or 0)),
                "blocked": short_spend >= short_limit,
            },
            "daily": {
                "estimated_spend": str(daily_spend),
                "limit": str(daily_limit),
                "reset_at": _iso(day_start + timedelta(days=1)),
                "blocked": daily_spend >= daily_limit,
            },
            "reconciliation": self.admin_reconciliation(),
        }

    def admin_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(limit, 500))
        return self.ledger.fetch_events(limit=bounded_limit)

    def admin_reconciliation(self) -> dict[str, Any]:
        if self.config.reconciliation_enabled:
            latest = self.ledger.latest_reconciliation_result()
            if latest is not None:
                return latest
            return {"status": "pending"}
        return {"status": "unavailable"}


def build_cost_accounting_from_env(env: Mapping[str, str] | None = None) -> DisabledCostAccounting | MisconfiguredCostAccounting | BudgetGate:
    source = env or os.environ
    try:
        config = CostTrackingConfig.from_env(source)
        if not config.enabled:
            return DisabledCostAccounting()
        pricing = PricingCatalog.from_config(config)
        assert config.ledger_path is not None
        ledger = SQLiteCostRepository(
            config.ledger_path,
            request_retention_days=config.request_retention_days,
            aggregate_retention_months=config.aggregate_retention_months,
        )
        ledger.initialize()
        return BudgetGate(config=config, ledger=ledger, pricing=pricing)
    except CostConfigError as exc:
        if _parse_bool(source.get("COST_TRACKING_ENABLED")):
            return MisconfiguredCostAccounting(str(exc))
        raise


def _create_cost_events_sql() -> str:
    column_sql: list[str] = []
    for name in LEDGER_ALLOWED_FIELDS:
        if name == "event_id":
            column_sql.append("event_id TEXT PRIMARY KEY")
        elif name in _TEXT_COLUMNS:
            column_sql.append(f"{name} TEXT")
        elif name in _INTEGER_COLUMNS:
            column_sql.append(f"{name} INTEGER")
        else:
            raise AssertionError(f"unclassified cost ledger column: {name}")
    return f"CREATE TABLE IF NOT EXISTS cost_events ({', '.join(column_sql)})"


def _normalize_ledger_value(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name in _INTEGER_COLUMNS:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        raise CostLedgerValidationError(f"{name} must be an integer")
    if name in {"forecast_cost_usd", "estimated_cost_usd"}:
        return str(_parse_decimal(value, name, allow_zero=True))
    if name in _TEXT_COLUMNS:
        return str(value)
    raise CostLedgerValidationError(f"{name} is not a known ledger field")


def _coerce_day(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CostConfigError(f"invalid reconciliation day: {value!r}") from exc


def _log_cost_event(
    action: str,
    *,
    endpoint: str,
    model: str,
    status: str,
    reservation_id: str,
    estimated_cost_usd: Decimal,
    forecast_cost_usd: Decimal,
    billing_eligible: bool,
    limit_type: str | None = None,
) -> None:
    payload = {
        "event": "cost_event",
        "action": action,
        "endpoint": endpoint,
        "model": model,
        "status": status,
        "reservation_id": reservation_id,
        "billing_eligible": billing_eligible,
        "forecast_cost_usd": str(forecast_cost_usd),
        "estimated_cost_usd": str(estimated_cost_usd),
    }
    if limit_type is not None:
        payload["limit_type"] = limit_type
    try:
        _COST_LOGGER.info(json.dumps(payload, sort_keys=True))
    except Exception:
        _COST_LOGGER.info("cost_event log_failed")
