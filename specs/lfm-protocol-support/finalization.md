# 브리지 완료 점검 — 플러그인 설계 보류

## 사용자 범위
Hermes/Pi 공통 모듈화·플러그인 설계는 보류하고 기존 브리지 기능을 먼저 완성한다. 기존 agentic 구현 계약을 유지한다. 운영 작업은 Atlas 소유이며 배포/활성화/Secret/push/merge는 이번 점검에서 하지 않았다.

## 개발 정리
기능 코드는 4e37f6d21bca7a456f55e23558aee3310a88522f에 저장되어 있다. Foundry 5protocol의 비스트리밍/스트리밍 결과-only hide/list/unhide 구현. README/.env.example의 이전 비스트리밍 전용 설명을 실제 코드 범위와 맞추고 응답 보류·원문 복원 한계·context 호환 무시를 기록했다. 확장 플러그인을 만들지 않았다.

## 승인된 시험 PG의 실제 검증
현재 셸에 DSN이 없었으나 기존 문서에서 승인된 connection.env 주입 절차를 확인했다. 접속값을 출력하지 않고 서버 cluster/control DB/role/권한/comment를 현재 remote fixture의 identity guard로 확인했다. 운영/로컬 PG는 사용하지 않았다.
- 최초 합산 검사: 78 passed,5 failed,4 errors. 동시성 설정 미지정 및 connection_slots_exhausted 확인. max_connections=10, 다른 idle 연결 존재. 서버 설정·다른 연결·기존 DB를 변경/종료하지 않음.
- ATLAS_TEST_SUITE_CONCURRENCY=2, COST_POSTGRES_TEST_REQUIRED=1을 적용하고 순차 재실행:
  - compaction cost + async cost: 25 passed
  - cost PostgreSQL API: 11 passed
  - repository + remote fixture: 51 passed
- 동일 설정 전체 suite: **1288 passed,32 skipped,1 기존 Starlette warning**,187.61초,exit0.
- 실제 local LFM 통합/품질 묶음: **87 passed,1 warning**,13.02초,exit0. protocol integration 10회 + 기존 result-only/envelope/longhide 실제 생성 포함.
- git diff --check 통과. bridge 시험DB count 실행전후7 동일. 기존7개 소유/원인 추정하지 않고 삭제하지 않음. count 불변은 동시 타작업까지 포함한 절대 누수 증명은 아니며 fixture isolation/teardown tests도 함께 통과함.

명령(접속값은 출력하지 않음):
```
set -a; . "${ATLAS_TEST_POSTGRES_ENV_FILE:?set approved test connection file}"; set +a
ATLAS_TEST_SUITE_CONCURRENCY=2 COST_POSTGRES_TEST_REQUIRED=1 .venv/bin/python -m pytest -q
RUN_LFM_PROTOCOL_INTEGRATION=1 RUN_LFM_RESULT_ONLY_SMOKE=1 RUN_LFM_RESULT_ENVELOPES=1 .venv/bin/python -m pytest -q tests/test_compaction_protocols_live_lfm.py tests/test_lfm_result_only.py tests/test_lfm_result_envelopes.py tests/test_lfm_lossless_source.py
```

## 증거의 소스 범위
위 PG 검증은 현재 worktree의 **별도 미커밋 remote-only 시험 전환 파일**을 사용했다. 기능 커밋 HEAD의 native local PG fixture를 실행한 것이 아니다. 해당 파일들은 다른 작업 소유이므로 이번 변경에서 수정·커밋하지 않았다. 따라서 source commit만 checkout한 환경에서 같은 원격시험 결과를 재현했다고 주장하지 않는다. Atlas가 배포 전 검증 시 remote fixture 변경의 소유자 승인/검증된 commit을 별도로 확보해야 한다. 이 사유 때문에 clean worktree라고 보고하지 않는다.

## 남은 실제 사용 완료 조건
개발 코드·보조 LFM·비용 DB 회귀는 위 범위에서 확인했다. 실제 유료 Foundry 모델의 hide 선택과 소비자 동작, zeon 배포, Pod→맥미니 DNS 실제 생성, 운영 지연/비용은 미검증이다. 기존 Atlas 배포 세션은 과거 e358cf 기준이므로 최신 기능 commit으로 재개해야 한다. 승인된 배포 지시와 개발 repo 임의 main merge 금지를 혼동하지 않는다. 유료 시험의 별도 비용 경계는 유지한다.

맥미니 OLLAMA_BASE_URL은 환경변수로 받는다. Atlas가 확인한 private Ollama DNS은 기존 운영근거이고 이 점검에서는 원격운영 재확인 안 함. feature flags 기본false, 초기 OFF 배포/검증 뒤 제한 활성화. 기존 rollback/메모리TTL/단일replica 영향은 verification-and-pr.md 및 Atlas 점검에 따른다.
