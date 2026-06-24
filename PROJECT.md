# Project: Cost Tracking Layer Refactoring

## Architecture
- 비용 추적 시스템을 기존의 SQLite 밀착형 `CostLedger`에서 추상화된 Repository 패턴으로 전환합니다.
- `ICostRepository` 인터페이스(Protocol)를 도입하여 상위 비즈니스 로직(`BudgetGate`, `CostAccountingStore`)과 저장소 계층(SQLite, InMemory)을 결합 해제(Decouple)합니다.
- 비용 예약과 정산 수명 주기를 제어하는 `BudgetReservationContext` Context Manager를 설계하여 `main.py` 라우터와 구조화된 출력 복구 로프의 정산 코드를 대폭 간소화합니다.

```
[main.py 라우터] (BudgetReservationContext 사용)
       │
       ▼
[BudgetGate] / [CostAccountingStore]
       │
       ▼ (의존성 주입)
   [ICostRepository] (Protocol)
       ├── [SQLiteCostRepository] (SQLite I/O 캡슐화)
       └── [InMemoryCostRepository] (테스트용 메모리 저장소)
```

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| 1 | ICostRepository Protocol 정의 및 구현체 설계 | `ICostRepository` 선언, `SQLiteCostRepository`, `InMemoryCostRepository` 작성 | none | DONE |
| 2 | BudgetGate 및 CostAccountingStore DI 개선 | 생성자 주입 방식으로 리팩터링 | 1 | DONE |
| 3 | BudgetReservationContext 구현 및 main.py 라우터 연동 | `async with` 수명 주기 관리와 stream preflight 선처리 적용 | 2 | DONE |
| 4 | 이중 청구 회피 로직 Context Manager 이동 | 구조화 출력 복구 시도별 예약/정산을 Context Manager로 이동 | 3 | DONE |
| 5 | test_cost_tracking.py 업데이트 | Repository fake, InMemory repository, Context Manager 단위 테스트 작성 | 1, 2, 3, 4 | DONE |
| 6 | 통합 테스트 및 정적 점검 | cost tracking 핵심 테스트와 수정 파일 ruff/diff 검증 | 5 | DONE |

## Interface Contracts
### ICostRepository (typing.Protocol)
- `transaction(self) -> ContextManager[None]`
- `initialize(self) -> None`
- `close(self) -> None`
- `record_event(self, fields: Mapping[str, Any]) -> dict[str, Any]`
- `prepare_event(self, fields: Mapping[str, Any]) -> dict[str, Any]`
- `insert_event(self, fields: Mapping[str, Any]) -> dict[str, Any]`
- `fetch_events(self, *, limit: int = 100) -> list[dict[str, Any]]`
- `sum_estimated_since(self, cutoff: datetime, statuses: tuple[str, ...] | None = None) -> Decimal`
- `daily_estimated_spend(self, day: str | date) -> Decimal`
- `record_reconciliation_result(self, result: ReconciliationResult) -> None`
- `latest_reconciliation_result(self) -> dict | None`
- `prune(self, *, now: datetime | None = None) -> dict[str, int]`
- `update_reservation(...) -> None`

## Code Layout
- `core/cost_tracking.py`: `ICostRepository`, `SQLiteCostRepository`, `InMemoryCostRepository`, `BudgetReservationContext`, `BudgetGate`, `CostAccountingStore` 구현 위치
- `main.py`: `BudgetReservationContext` 적용 및 라우터 간소화 대상
- `tests/test_cost_tracking.py`: 리팩터링된 컴포넌트 검증 테스트
