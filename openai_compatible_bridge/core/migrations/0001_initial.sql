CREATE SCHEMA bridge_cost;
REVOKE ALL ON SCHEMA bridge_cost FROM PUBLIC;

CREATE TABLE bridge_cost.schema_migrations (
    version INTEGER PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE bridge_cost.cost_events (
    event_id TEXT PRIMARY KEY,
    reservation_id TEXT UNIQUE,
    internal_request_id TEXT,
    provider TEXT,
    endpoint TEXT,
    model TEXT,
    status TEXT,
    billing_eligible INTEGER CHECK (billing_eligible IN (0, 1)),
    limit_type TEXT,
    prompt_tokens BIGINT CHECK (prompt_tokens >= 0),
    completion_tokens BIGINT CHECK (completion_tokens >= 0),
    total_tokens BIGINT CHECK (total_tokens >= 0),
    embedding_tokens BIGINT CHECK (embedding_tokens >= 0),
    rerank_units BIGINT CHECK (rerank_units >= 0),
    forecast_cost_usd NUMERIC CHECK (
        forecast_cost_usd >= 0 AND forecast_cost_usd NOT IN ('NaN', 'Infinity', '-Infinity')
    ),
    estimated_cost_usd NUMERIC CHECK (
        estimated_cost_usd >= 0 AND estimated_cost_usd NOT IN ('NaN', 'Infinity', '-Infinity')
    ),
    currency TEXT,
    pricing_source TEXT,
    pricing_version TEXT,
    window_started_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    finalized_at TIMESTAMPTZ,
    reconciliation_status TEXT,
    CHECK (status IS DISTINCT FROM 'reserved' OR (
        reservation_id IS NOT NULL AND reservation_id <> ''
        AND billing_eligible IS NOT NULL AND billing_eligible = 1
        AND forecast_cost_usd IS NOT NULL AND estimated_cost_usd IS NOT NULL
        AND forecast_cost_usd = estimated_cost_usd
    ))
);

CREATE INDEX idx_cost_events_created_at ON bridge_cost.cost_events (created_at);
CREATE INDEX idx_cost_events_status ON bridge_cost.cost_events (status);

CREATE TABLE bridge_cost.cost_daily_aggregates (
    day TEXT PRIMARY KEY,
    estimated_cost_usd NUMERIC NOT NULL CHECK (
        estimated_cost_usd >= 0 AND estimated_cost_usd NOT IN ('NaN', 'Infinity', '-Infinity')
    ),
    currency TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE bridge_cost.cost_reconciliation_results (
    day TEXT PRIMARY KEY,
    wrapper_estimated_cost_usd NUMERIC CHECK (
        wrapper_estimated_cost_usd >= 0 AND wrapper_estimated_cost_usd NOT IN ('NaN', 'Infinity', '-Infinity')
    ),
    billing_export_cost NUMERIC CHECK (
        billing_export_cost >= 0 AND billing_export_cost NOT IN ('NaN', 'Infinity', '-Infinity')
    ),
    delta_usd NUMERIC CHECK (delta_usd NOT IN ('NaN', 'Infinity', '-Infinity')),
    status TEXT NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL,
    error_message TEXT
);