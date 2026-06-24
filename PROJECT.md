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
| 3 | BudgetReservationContext 구현 및 main.py 라우터 연동 | `async with`로 수명 주기 관리 적용, main.py 리팩터링 | 2 | PLANNED |
| 4 | 이중 청구 회피 로직 Context Manager 이동 | `_generate_ollama_with_structured_output_repair` 리팩터링 | 3 | PLANNED |
| 5 | test_cost_tracking.py 업데이트 | mock/DI 구조 수정 및 InMemoryCostRepository 유닛 테스트 작성 | 1, 2, 3, 4 | PLANNED |
| 6 | 통합 테스트 및 Audit 검증 | `uv run pytest` 100% 통과 및 Forensic Auditor 무결성 검증 | 5 | PLANNED |

## Interface Contracts
### ICostRepository (typing.Protocol)
- `initialize(self) -> None`
- `close(self) -> None`
- `record_event(self, event: dict) -> None`
- `fetch_events(self, limit: int = 100) -> list[dict]`
- `sum_estimated_since(self, since: datetime) -> float`
- `daily_estimated_spend(self, date: date) -> float`
- `record_reconciliation_result(self, result: dict) -> None`
- `latest_reconciliation_result(self) -> dict | None`
- `prune(self, before: datetime) -> int`

## Code Layout
- `core/cost_tracking.py`: `ICostRepository`, `SQLiteCostRepository`, `InMemoryCostRepository`, `BudgetReservationContext`, `BudgetGate`, `CostAccountingStore` 구현 위치
- `main.py`: `BudgetReservationContext` 적용 및 라우터 간소화 대상
- `tests/test_cost_tracking.py`: 리팩터링된 컴포넌트 검증 테스트
