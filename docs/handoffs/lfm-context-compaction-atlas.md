# Atlas 인계: LFM 결과-only hide 요약 개선

## 현재 판정
- **후속 PR 검토 가능:** 실제 코드 반영, 깨끗한 후보 작업 폴더의 전체·원격 DB·합성 LFM 시험 완료.
- **운영 적용 준비 완료는 아님:** 선행 PR #23 미머지, 후속 PR CI/리뷰와 배포 승인, 실제 소비자 검증이 남아 있다.
- 구현: `79204f5` (PR #23 head `36c7a1b69d7771b877b5ff3c155e56338e98869d` 기반).
- 관련 이슈: #24. 선행 PR #23을 대체·병합·종료하지 않는다.

## 계약과 과거 기록
현재 계약은 [결과-only 설계](../../specs/lfm-result-only/design.md)다. 기존 사용자 승인에 따라 작업 의도·목적별 정보 선택·execution/result/limitations 생성은 제외하고 도구 결과만 요약한다. 이전 의미 중심 실험의 Q05/Q06 목적별 요구는 현재 합격 조건이 아니다. 과거 실험 실패를 합격으로 바꾸거나 삭제하지 않는다. 과거 기록은 과거 계약의 증거이지 현재 운영 판정을 덮어쓰는 기준이 아니다.

개발 통과가 임의 입력의 의미 정확성이나 운영 활성화를 보장하지 않는다. 이번 결과는 고정 합성 4개 사례의 반복 및 기존 live hide/list/unhide 경로 검증이다.

## 변경 사항
- 일반 텍스트: 간결한 결과-only system 지시와 원문 source JSON.
- 구조화 envelope 또는 기존 무손실 반복 표현: 기존 system과 정규화 source JSON.
- 구조로 분기하며 사례 이름·예상 답은 사용하지 않는다.
- validator, 모델, 토큰·시간 제한, 원문 보관·복원·비용 gate 및 ledger는 변경하지 않는다.
- 원래 assistant 호출·인자·명령·ID·사용자 메시지는 그대로 두고 결과 본문만 요청 사본에서 교체한다.
- hide_context는 내부 도구이며 Hermes UI/MCP나 자체 대화 압축 기능과 별개다. unhide_context는 재실행 없이 원문으로 되돌린다.
- 이번 변경은 브리지 LFMSummarizer의 요청 조립에 적용된다. vendor snapshot은 원본 추출 커밋 그대로이며 이 후속 변경에 동기화됐다고 주장하지 않는다. main의 요약 호출은 브리지 LFMSummarizer를 사용한다.

## 직접 검증 결과
PR 기반의 깨끗한 전용 작업 폴더에서 다음을 실행했다. 다른 작업의 미커밋 파일을 복사하지 않았다.

| 검사 | 결과 |
|---|---|
| vendor provenance 검사 | 14개 파일 무결성 통과 |
| 전체 회귀 및 승인 원격 PG | 1304 passed, 32 skipped, 기존 경고 1개, 170.51초, exit 0 |
| 실제 목표 LFM, 합성 4사례 | 각 10회, 총 40회 성공 |
| 실제 hide → 후속 요청 → list → exact unhide | 10회 성공 |
| 압축 결과 | 3128 → 550바이트, 필수 원문 근거 3줄 유지 |

목표 모델 lfm2.5-thinking:latest, temperature 0, think=false, 출력 상한 384, timeout 60초, 기존 JSON schema/validator 사용. 동일 입력 반복은 독립적 통계 신뢰도를 증명하지 않는다. 32개 opt-in skip은 pass로 세지 않으며 위 실제 LFM 시험은 별도 수행했다. 주 모델의 도구 선택은 scripted 합성 응답이고 실제 Foundry 소비자 시험이 아니다.

PR #23에는 원격 전용 PG fixture와 identity 검증이 포함돼 있다. 따라서 이전 로컬 개선 커밋에 있던 '별도 미커밋 fixture 필요' 조건은 이 PR 기반에서 해소됐다. 테스트는 승인 연결 계약과 공통 HOME suite 잠금 아래 순차 실행했다. 로컬 PG 기동·운영 PG·유료 Foundry 호출은 하지 않았다. CI는 오프라인 시험이며 원격 DB 검증과 구분한다.

## 재현
```sh
uv sync --frozen
uv run python scripts/verify_vendor_context_hide.py
# Atlas 승인 connection.env를 안전하게 로드하고 공통 pg_test_suite_guard 아래 실행:
# uv run pytest -q
LFM_INTEGRATION_BASE_URL=<approved-ollama-url> LFM_INTEGRATION_TIMEOUT_SECONDS=60 uv run python scripts/verify_lfm_prompt_routing.py <new-output-jsonl>
```
DSN이 없거나 서버 identity가 틀리면 DB 시험은 실패해야 한다. 운영 DB나 로컬 서버로 대체하지 않는다. 반복 검증 출력 파일은 새 경로를 사용한다. 합성 데이터만 전송한다.

## 설정
새 설정 없음. 기본값: CONTEXT_COMPACTION_ENABLED=false, CONTEXT_COMPACTION_LFM_ENABLED=false, CONTEXT_COMPACTION_LFM_MODEL=lfm2.5-thinking:latest, 출력 384토큰, timeout 60초, 입력 12288바이트/50000문자. 한 turn 요약 최대 1회. 해당 출력 예산이 모든 결과의 최적값이라는 주장은 하지 않는다.

## 적용과 활성화는 별도 승인
1. 선행 PR #23과 후속 PR 리뷰/CI를 확인한다. 이 문서로 머지 권한이 생기지 않는다.
2. main merge는 저장소의 자동 이미지 게시·운영 handoff를 유발할 수 있다. 컨테이너 교체 직전 중지 조건이 있으므로 승인 없이 머지·workflow dispatch하지 않는다.
3. Atlas가 승인된 배포에서 이미지·Ollama DNS 접근·목표 모델·비용 분류·예산을 확인한다. 운영 환경값·Secret·Pod 변경은 이번 작업에서 하지 않았다.
4. 기능 OFF 배포와 제한된 canary 활성화를 분리한다. 실제 Foundry 호출은 비용 승인 후 합성 입력만 사용한다.
5. 실제 주 모델의 내부 도구 발견·소비, 두 후속 요청 결과 교체, list와 정확한 unhide, 외부 도구·stream·비용 회계를 검증한다. HTTP 200이나 안전 fallback을 요약 성공으로 세지 않는다.

## 원문 및 운영 제한
- 저장소는 프로세스 메모리·TTL 방식이다. 재시작/Pod 이동 후 복원 보장 없음.
- raw conversation 상관 키는 인증된 사용자·분기 권한을 대신하지 않는다. 실제 replica 라우팅과 세션 격리를 확인해야 한다.
- hide는 비밀 삭제·접근 제어가 아니다. 결과를 모델 서버에 보내도 되는지 운영자가 확인해야 한다.
- 실제 지원 protocol/stream 조합은 [프로토콜 지원 설계](../../specs/lfm-protocol-support/design.md)와 해당 시험을 기준으로 읽는다. 초기 non-stream 한정 기록을 현재 계약으로 사용하지 않는다.

## 롤백
Atlas 승인 절차로 LFM 또는 전체 기능 스위치를 OFF로 하고 코드 문제면 이전 이미지로 복구한다. 새 DB migration 없음. 호스트의 원래 결과를 보존하며 프로세스 원문 유실 시 복원을 보장하지 않는다.

현재 이슈/후속 PR까지의 개발 변경만 승인됐으며 main push·머지·배포·활성화·Secret 변경·운영 컨테이너 교체는 수행하지 않는다.
