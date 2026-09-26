# PostgreSQL cost ledger: Atlas cutover draft

## Status and boundary

This is a development candidate, not an authorization to deploy. No production
database, role, Secret, PVC, Deployment, GitOps state, or real ledger data was
created, changed, or migrated during development. Atlas owns provisioning and
cutover approval. Do not run the legacy Jenkins publish/GitOps stages for this
validation task.

Read-only runtime recheck on 2026-09-26: the bridge still had one ready replica
with `Recreate`; TCP connectivity from the bridge Pod on e2 to the HomeLab
PostgreSQL cluster service succeeded. This proves network reachability only,
not database authorization, TLS verification, migration compatibility, or
production capacity. Development integration tests use a disposable native
PostgreSQL instance, not the HomeLab database and not Mac Docker. Ubuntu runtime
deployment evidence remains an Atlas responsibility.

## Storage contract

- SQLite remains the default. Existing `COST_LEDGER_PATH` configuration continues
  to work. PostgreSQL requires explicit `COST_LEDGER_BACKEND=postgres` and
  `COST_LEDGER_POSTGRES_DSN`. A DSN with the SQLite backend is a configuration
  error; an obsolete SQLite path with the PostgreSQL backend is never used.
- Use a **dedicated database**, fixed schema `bridge_cost`, and dedicated roles
  on the shared server. Never point the bridge at the Neurons database or grant
  access to its tables. The DSN is provisioned out-of-band, never passed on the
  command line, printed, checked into Git, or copied to the handoff report.
- Schema changes are versioned SQL with migration metadata and explicit operator
  application. Runtime startup only validates; the runtime role has no DDL
  permissions. Old, missing, or incompatible schema prevents paid admission.
- Budget check and reservation insert share a serialized PostgreSQL transaction
  across bridge processes. All cooperating writers use the same database lock.
  Do not mix independent databases, fork the locking contract, or let operators
  bypass it with live ad-hoc writes. All replicas must use identical prices,
  provider scope, limits, UTC clocks, and backend settings.
- Money is PostgreSQL `NUMERIC` / Python `Decimal`, never binary floating-point.
  Event and reservation identifiers remain the existing opaque prefixed UUID
  strings. Identical ledger retries are idempotent; conflicting duplicates and
  conflicting terminal settlements are rejected, not overwritten.
- A PostgreSQL `reserved` amount continues to consume budget even after its
  original time window/day and is excluded from retention deletion. This is an
  intentional conservative difference from SQLite: uncertain charges must not
  disappear by age. Normal finalized/released reporting remains date-based.
- Serialization protects **forecast admission**, not an absolute invoice ceiling.
  Provider usage, token estimates, price changes, missing usage and billing-export
  delays retain the existing estimation limitations. Provision conservative
  forecasts/limits; do not present this ledger as the provider's billing SoT.
- Each operation uses bounded connections/transactions. There is no blind retry
  of an ambiguous commit and no HTTP-level idempotency guarantee. Never retry a
  paid provider request merely to recover its ledger settlement.

## Provisioning review (Atlas only; not executed)

1. Choose a dedicated database such as `bridge_cost_ledger`, owned by an isolated
   no-login owner. Use distinct migration and runtime login roles, neither
   superuser nor `CREATEDB`/`CREATEROLE`/replication. Remove public connection/create
   permissions in this dedicated database; allow only approved roles/networks.
2. Permit the migrator to create/own `bridge_cost` and apply the versioned schema.
   The runtime role receives only database `CONNECT`, schema `USAGE`, `SELECT` on
   migration metadata, and `SELECT, INSERT, UPDATE, DELETE` on the three runtime
   ledger tables (the delete privilege supports retention). No schema `CREATE`,
   ownership, membership in the migrator/owner role, or Neurons access. Restrict
   migration-table changes to the migrator. Use a separate, time-limited operator
   identity for transfer/recovery; revoke it afterwards.
3. Confirm that the actual runtime identity can check schema, reserve, settle,
   query, and prune synthetic data in an approved **non-production** database,
   while DDL, migration mutation and cross-database/table access fail.
4. Select TLS/network isolation and server authentication policy. Prefer
   `sslmode=verify-full` with the appropriate CA/hostname where TLS is configured;
   plaintext inside a private cluster is not automatically an approved policy.
5. Establish server backups/PITR, restore tests, capacity and connection limits,
   monitoring, and an independent maintenance window. Do not enable runtime
   migrations or reuse a database-admin credential to avoid provisioning work.

## Observability and incident handling

`/healthz` remains process liveness. `/readyz` returns 200 only when the cost
subsystem is ready (or intentionally disabled), otherwise 503. Its
`cost_tracking` object includes `enabled`, `backend`, `database_available`,
`healthy` and a sanitized reason. No DSN, ledger rows, or SQL error details are
returned. A restored database can report `database_available=true` while
`healthy=false`: the process recovery latch has **not** been cleared.

Atlas must separately approve changing readiness to `/readyz` while retaining
liveness at `/healthz`. Allow a probe timeout consistent with bounded DB
connection/statement/lock waits, and avoid restart storms during DB outages.
No probe manifest was changed in this candidate. Alert on readiness failure,
`cost_ledger_failure` structured logs, outstanding reservation age/count,
connection/lock timeouts, failed comparisons, and storage/backup health.

Before upstream admission, database failure returns 503 and the provider is not
called. Initialization and processing failures latch that process unhealthy;
reconnecting alone does not clear it. There is no SQLite fallback, bypass, or
automatic switch to disabled accounting.

If a provider response is already obtained but settlement fails, the response
(or already-started stream) cannot be unbilled or recalled. Preserve its existing
delivery behavior; keep the committed forecast reservation billable, log the
operation plus reservation identifier without payload/credentials, and block
subsequent tracked requests on that process. Other in-flight settlements may
still complete. The latch is process-local, not a distributed incident switch:
Atlas must fence **all** replicas for recovery. Other replicas still count the
outstanding reservation conservatively, but its actual charge may exceed its
forecast. Do not claim this is an exactly-once durable settlement outbox.

### Settlement recovery

1. Stop admission on every replica and drain streams/requests. Preserve logs,
   snapshots and provider usage evidence privately. Do not delete/release old
   reservations, disable accounting, or restart repeatedly to erase the latch.
2. Restore DB access and determine the outcome of each affected reservation by
   its identifier. A lost acknowledgement can mean the transaction committed.
   Inspect the persisted state before retrying. An exact settlement replay must
   not double charge; a conflicting terminal value requires investigation.
3. Reconcile against retained provider usage/billing evidence and the price
   version applicable to the original request. Record actual usage/cost via the
   repository's transaction and `update_reservation` contract. If usage cannot
   be established, keep the reservation and the fence: a forecast is not proof
   of the actual charge. Only explicitly evidenced nonbillable requests may be
   released. There is no automatic TTL release or automatic finalization retry.
4. Preserve the authorization/evidence trail outside public logs. If process
   loss also lost actual usage, this candidate cannot reconstruct it from the
   forecast; Atlas must obtain external evidence before clearing the incident.
5. Recompare ledger counts, money and reservation states; resolve outstanding
   uncertainty before restarting with the same validated settings. A new
   process clears only the process-local latch, not ledger reservations. Check
   readiness and an approved canary before gradually reopening admission.

There is no automatic recovery worker or settlement CLI in this candidate.
Approved manual replay uses
`openai_compatible_bridge.core.postgres_cost_repository.PostgresCostRepository`
and `update_reservation()` with the original reservation identifier,
`NormalizedUsage`, exact `Decimal` cost and UTC finalization timestamp, inside
`transaction()`. Only `reserved` may transition to a terminal state. Replaying
the same terminal settlement is harmless (the retry timestamp is ignored), but
changing its cost/status/usage, including `estimated_only` to `finalized`, is
rejected. Reconciliation evidence is separate; never bypass this guard with a
live table update.

## Transfer commands (Atlas only; not executed on production)

These examples use placeholder private filesystem locations. Atlas must first
approve and substitute the real paths, fence writers and securely inject
`COST_LEDGER_POSTGRES_DSN` for the appropriate migration/operator identity.
Do not include its value in shell history, process arguments or transcripts.

```bash
uv run python -m openai_compatible_bridge.core.cost_ledger_transfer backup /approved/private/live.db /approved/private/pre-cutover.sqlite
uv run python -m openai_compatible_bridge.core.cost_ledger_transfer migrate-schema --apply
uv run python -m openai_compatible_bridge.core.cost_ledger_transfer import-sqlite /approved/private/pre-cutover.sqlite --apply
uv run python -m openai_compatible_bridge.core.cost_ledger_transfer compare /approved/private/pre-cutover.sqlite
```

For rollback **after any target writes**, while all writers remain fenced:

```bash
uv run python -m openai_compatible_bridge.core.cost_ledger_transfer export-sqlite /approved/private/current-rollback.sqlite --apply
uv run python -m openai_compatible_bridge.core.cost_ledger_transfer compare /approved/private/current-rollback.sqlite
```

- Existing destinations, including SQLite sidecars, are refused. Published
  snapshots have mode `0600`; secure/encrypt their parent storage and backups.
  Backup opens the existing source read-only and uses SQLite's online backup
  API; it never creates an empty replacement for a missing source.
- `--apply` is required for schema application, import and export. Omitting it
  is a refusal, **not** a dry run. `compare` is read-only and uses the common
  transaction lock for a consistent view relative to cooperating writers.
- Exit code `0` means success; `1` means failure or comparison mismatch. JSON
  output contains stable result codes and only counts, exact decimal totals,
  status counts and canonical SHA-256 digests. No raw records are emitted.
- Compare covers events, daily aggregates and reconciliation results, including
  nullable fields and usage/cost/state changes to existing IDs. An exact repeated
  import returns `already_imported`; a different nonempty target is rejected.
  Incompatible schema/checksum, corrupt/invalid amounts, duplicate IDs and
  incomplete transfers are stop conditions, never reasons to weaken checks.
- These are offline transfer tools, not online replication or schema downgrade
  tools. Rehearse size, execution time, memory and backup/restore capacity on an
  authorized copy before setting the production maintenance window.

## Forward cutover checklist

1. Approve the dedicated DB/roles, migration version, credential delivery, image
   candidate, monitoring, backup location, recovery owner and stop conditions.
   Recheck live bridge/DB state; previous TCP evidence is not deployment approval.
2. Freeze pricing/config changes and retention/reconciliation writers. Fence all
   bridge replicas and drain in-flight requests, including streams and repair
   attempts. No SQLite/PostgreSQL dual writers are allowed.
3. Make a SQLite backup using the tool's SQLite backup API, not a raw copy of the
   database without its WAL. Preserve the original volume, immutable snapshot,
   integrity result, file hash and secure access controls. Never query/print raw
   cost rows in shared logs. Resolve uncertain reservations before cutover where
   possible; retained unresolved entries must remain budget-consuming.
4. Apply schema using the migration identity, then import the frozen snapshot
   into an empty target in one transaction. An exact repeat is harmless; a
   non-identical nonempty target is an abort, not a merge request.
5. Require count, exact money, reservation-state and canonical-content agreement
   for **all three tables**. A successful network probe or matching grand total
   alone is insufficient. Abort on duplicates, invalid amounts/schema, digest
   mismatch or concurrent writes; never edit source data to make a check pass.
6. Atlas alone changes runtime configuration/Secret/probes and deploys the
   approved image, initially one replica. Keep the SQLite snapshot and volume
   untouched. Verify runtime identity, readiness, admin reports, reservation and
   settlement of an explicitly approved canary, then take a new checkpoint.
7. Consider a second replica only after identical configuration and concurrency
   evidence have been accepted. Observe locks/latency and budget headroom before
   lifting the fence. Archive the evidence; revoke temporary migration access.

## Rollback, including post-cutover writes

- If **no PostgreSQL write has occurred** after the accepted import, fence/drain,
  prove full equality against the preserved snapshot, then Atlas may restore the
  original SQLite backend/config. Unset the PostgreSQL DSN when selecting SQLite.
- If any new reservation, settlement, release, reconciliation or retention write
  exists, **never simply point at the old SQLite file**. That loses costs and
  reopens budget. Freeze all writers, back up PostgreSQL, recover uncertain
  reservations, export the **entire latest consistent ledger** to a new SQLite
  file, and require full comparison before selecting that file. Do not append
  only new event IDs: updates to pre-existing reservations matter too.
- Do not resume the SQLite backend with unresolved old reservations: its legacy
  time-window/retention behavior does not preserve PostgreSQL's indefinite hold.
  Resolve the uncertainty first, or remain fenced on PostgreSQL.
- If PostgreSQL is unavailable and its post-cutover writes cannot be recovered,
  rollback is blocked. Keep paid traffic stopped and restore from verified
  PostgreSQL backups/WAL; do not discard unknown charges for availability.
- Preserve both old and new snapshots and comparison results; never overwrite
  the evidence. Prevent split brain before and after any backend change.

## Development verification

The regression command is `uv run pytest -q`. To require rather than silently
skip disposable PostgreSQL tests when local binaries are absent:

```bash
COST_POSTGRES_TEST_REQUIRED=1 uv run pytest -q
```

The tests initialize their own native PostgreSQL cluster with synthetic records,
not a production DSN, and shut it down after the run. PostgreSQL server binaries
must be on `PATH`; the GitHub Actions test job installs them and requires these
tests. Do not trigger publishing/deployment to run this command. The legacy
Jenkins test image does not install PostgreSQL; use the required command in a
prepared test environment for release evidence, not a silently skipped run.

The application is run directly from source (`tool.uv.package=false`);
`Dockerfile` copies the full `openai_compatible_bridge` directory, including
`core/migrations/0001_initial.sql`. No container image was built/published and
no deployment workflow was triggered for this task. See
[verification.md](verification.md) for actual local results and unperformed work.