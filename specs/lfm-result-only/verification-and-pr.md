# Result-only LFM hide: 최종 검증 및 로컬 PR 본문

## 결론
최신 승인된 결과-only 범위에서 구현·고정 합성 LFM 품질·원문 복원을 직접 검증했다. 기존 실패 사례를 없애지 않고 입력 정규화/무손실 반복 encoding/warning 관계와 optional 한정어 보호로 해결했다. 일반 요약 정확성 또는 운영 소비자 성공을 보장하지 않는다.

## PR 요약
- 원래 호출·인자·명령·ID 유지. 결과 본문만 단일 English summary로 교체.
- invocation/context는 LFM 미전송. context는 검증 후 무시하는 로컬 호환 인자로만 유지.
- JSON content/output/files 정규화, read_file 연속 줄번호 처리, 완전 TOML 그룹 관계 유지.
- 긴 반복 plain text는 verbatim unique lines+알파벳 순서로 무손실 표현. 모든 source lines/개행/삽입지시 유지. 원문 저장/검증 권위는 unchanged.
- 명시적 warning만 별도 표시, annotation 경고 오분류와 optional 한정어 누락 거부.
- 실제 원문 숫자/경로/ID, 확인되지 않은 exit code, 메타 설명 검사. 모든 의미관계를 자동 증명하는 validator는 아님.
- 기존 Rule/Laya/store/TTL/cost-metered/pending/digest/unhide 재사용. 기본 feature flags false. provider/config/model/운영 변경 없음.

## 부모 직접 실행
```sh
RUN_LFM_RESULT_ONLY_SMOKE=1 RUN_LFM_RESULT_ENVELOPES=1 .venv/bin/python -m pytest -q tests/test_lfm_result_only.py tests/test_lfm_result_envelopes.py tests/test_lfm_lossless_source.py tests/test_lfm_summary.py
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q openai_compatible_bridge tests
git diff --check
```
- 첫 명령: **107 passed**,8.47초. 실제localLFM21회(반복9,현실envelope9,longhide3) 포함.
- 전체: **876 passed,91 skipped,1 warning**,13.33초,exit0. Skip은 PG DSN/opt-in 조건없는 검사 포함. skipped를 passed로 세지 않음.
- compile/diff 통과. 기존 Starlette/httpx deprecation warning.
- 독립 read-only review deleg_921ff6ea: NO_ACTIONABLE_FINDING. 리뷰는 정적 검토이며 부모의 실제 실행을 독립 재실행한 것이 아님.

## 실제 생성 Before/After
합성 원문: orbit-widget exact-name matching, Warning optional descriptions omitted +55개 동일 일반 annotation.
실제 LFM: `The orbit-widget catalog uses exact-name matching with a warning about omitted optional descriptions.`
생성문 수정·사후 정답 합성 없이 model response 그대로 검증·적용.
- checks: `The output indicates 2 synthetic checks passed with exit code 0, and repeated annotations were noted.` 실제 결과의 반복 언급이 불필요할 수 있으나 핵심 사실/상태 유지.
- unknown exit: `The amber-widget component is identified with exact-name matching for catalog entries.` 없는 종료 코드나 성공 만들지 않음.
- file TOML: Python>=3.12, runtime dependency4종/version, pytest development group 보존.
- file search: source3경로 그대로 유지.
- terminal envelope: 2passed0.04s exit0 noerror 유지.
- longhide3회: original1199→compacted468UTF8bytes,1LFMcall,after/list/exactunhide,assistant 불변.

## 미검증 및 경계
- ATLAS_TEST_PG_DSN 현재 미설정. 현재 remote-only fixture는 다른 작업의 미커밋 변경이라 수정/커밋하지 않았다. 승인 격리 PG identity 검증 후 ledger/async 비용 재검증 필요. 운영/로컬 DB 대체하지 않음.
- 실제 주 모델의 자율 hide 판단과 소비자 canary, 운영 비용/캐시/장애 대응은 미검증.
- 짧거나 오류/업무 위험 결과는 보호 유지. 긴 단일 escaped line의 필수 증거240자 제한은 여전히 안전 거부.
- 모델 입력은 source byte 예산12KiB/output384/timeout60초/turn1call 유지. 모든 임의 입력 품질 보장 아님.
- raw session 예시는 workspace local artifacts만. Git/model에는 새 합성자료만 사용.

## 운영 적용/롤백(Atlas 별도 승인)
구현 결과가 있다고 자동 배포하지 않는다. Atlas가 검토·빌드/승인된isolatedPG회귀·Pod→Ollama/replica affinity 확인 뒤 synthetic consumer canary를 별도로 수행한다. 기존 OLLAMA_BASE_URL은 운영 Pod에서 도달가능한 주소 사용, 개발127.0.0.1 복사 금지. 두 featureflags 기본false. 영속store/DBmigration없음.
- LFM중지 CONTEXT_COMPACTION_LFM_ENABLED=false.
- internal hide전체중지 CONTEXT_COMPACTION_ENABLED=false.
- Atlas 승인 이전image rollback. 재시작/replica이동 후 memory originals 복구보장없어 Hermes 원문 유지.

## Git/PR
로컬 feature branch만 commit. push/PR remote creation/merge/배포 없음. 다른dirtyPG/docs/old evaluator 변경은 제외. docs/handoffs/lfm-result-only-atlas.md는 ignored local기록이므로 이 tracked검증문서가 최신 완료결과를 우선한다.
