# Result-only hide milestones

- 실행 계약: agentic-execution, 사용자 최신 명시 구현 승인.
- 요구사항/설계: [design.md](design.md). 이전 의도 기반 설계를 축소하는 최신 결정 우선.
- 기준 HEAD: 7763855c12275b6f74010a9c27d3e873ff09b801. 기존 dirty summary 변경을 보존·검증해서 필요한 부분만 채택한다.
- 실제 세션 원문은 workspace artifacts 로컬 전용, 모델 및 Git으로 복사하지 않는다.
- PG 관련 동시 미커밋 변경은 건드리지 않는다. ATLAS_TEST_PG_DSN 확인 전 DB 사용 금지, local/운영 PG 대체 금지.

| 단계 | 상태 | observable result/evidence |
|---|---|---|
| 준비 | 완료 | 최신 범위와 R01–R05 기대 본문 정렬, LBrain 빈 context, 소스 상태 확인 |
| A 결과-only hide/복원·exit guard | 검증 통과·체크포인트 대기 | 부모 직접 175 passed/9 opt-in skipped/1 기존 warning, returncode0 및 diff check. context/invocation 전송 없음, context 호환 무시, exit unknown/다중 claim guard, exact restore fixture. B 변경과 분리해 최종 리뷰 후 소유 파일만 commit |
| B 실제 LFM 결과 품질 | 대기 | 새 버전 합성 corpus의 주요 사실/경고/모순, 3회 actual local 생성, 원문 보호·복원 구분 |
| C 전체회귀/리뷰/인계 | 대기 | full suite와 PG skip/성공 명시, 코드 리뷰, local commit/PR body, Atlas 인계 |

활성 A. 책임 재사용: 기존 store/rule selector/metered Ollama/내부 도구. 삭제할 기능 요구: 의도 기반 생성/세필드 실행 보고. 영문 단일 summary로 축소. 운영 기본 OFF, 배포/push/merge/secret/모델 변경 없음.
