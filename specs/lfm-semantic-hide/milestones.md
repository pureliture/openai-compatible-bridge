# LFM 의미 중심 hide 실행 상태

## 승인과 실행 계약
- 요구사항/설계: `<local-plan>/2026-10-01_202607-lfm-semantic-hide-context.md` (10절 우선).
- 사용자 최신 승인: 해당 계획을 구현하며 mode=agentic, 준비가 부족하면 준비부터 수행.
- 선택 계약: `agentic-execution`. 승인된 목표를 유지하는 최소 설계 변경만 자율 결정한다.
- 기존 streaming 작업의 root milestones.md는 보존하고 이 작업은 별도 문서로 관리한다.
- 전용 worktree/브랜치: `.worktrees/lfm-context-summary`, `daedalus/lfm-context-summary`.
- baseline HEAD: c2dcb899df5c1a53dd4afeaad06b4c984f926a3a.
- 운영 배포/활성화/secret 변경/push/merge 금지. 실제 Foundry 유료 사용성 테스트 B는 별도 승인 전 실행하지 않는다.

## 준비 결과
- LBrain read-only context: 항목 없음. 로컬 코드/설계로 계속.
- clean 작업 트리 및 현재 브랜치 확인. baseline focused: `uv run pytest -q tests/test_context_compaction.py tests/test_lfm_summary.py` → 36 passed.
- 로컬 Ollama /api/tags 정상, lfm2.5-thinking:latest 존재. 생성 품질은 아직 미검증.
- 실행 도구 schema(현재 세션 노출): terminal command/workdir, read_file path/offset/limit, search_files pattern/target/path/file_glob 확인. namespace 접두사를 임의 제거하지 않고 실제 assistant 함수 이름을 기준으로 adapter를 적용한다.
- terminal 부가 옵션은 timeout 등 실행 방법의 보조 설정만 생략 가능. background/pty/persist_on_release처럼 결과 의미를 바꾸는 옵션이 활성화되면 의미 요약 거부. search_files의 offset/limit/output_mode/context/order는 의미를 바꾸므로 기본이 아닌 사용은 첫 adapter에서 거부한다. 임의 unknown 인자도 거부한다.
- context optional 확장, 한국어 생성 출력, 민감 인자 거부, 위험 로그 원문 유지는 계획 구현 지시에 따라 개발 계약으로 수용한다. 운영 command 전송 승인은 별도 Atlas 경계로 유지.
- pending 생성 상한은 process당 4로 고정 제안. 이미 한 turn LFM 1회 및 timeout/입력 12,288 byte 한도가 있어 별도 queue/자동 재시도는 만들지 않는다. 메모리 RSS 보장은 아니다.

## 관찰 가능한 순차 단계
| 단계 | 상태 | 관찰 결과와 직접 증거 |
|---|---|---|
| 준비 | 완료 | baseline 36 passed, clean, 로컬 모델과 tool schema 확인 |
| S1 실행 맥락 의미 hide/후속 적용/복원 | 완료 | 부모 직접 focused 130 passed, git diff --check 통과. 실제 invocation+optional context와 구조 출력, digest visibility, exact unhide 및 pending/cancel fixture 검증. 로컬 checkpoint 커밋에 이 상태 포함 |
| S2 안전·동시성·비용 경계 | 대기 | 중복 생성/충돌/취소/민감정보/TTL/비용 게이트/외부 도구/stream 회귀의 직접 검증 및 커밋 |
| S3 실제 LFM 품질 비교 | 대기 | 사전 고정 Q01–Q12, 비교군 반복 결과와 내용 판정, 실제 LFM 적용 및 복원 |
| S4 최종 회귀·리뷰·인계 | 대기 | 전체 suite/compile/diff, 독립 리뷰, Atlas 인계와 로컬 PR 본문, 로컬 커밋 |

## 활성 단계
S2. S1 부모 focused 130 passed 후 checkpoint `5394ccc` 작성. 부모 비용/비동기/stream 회귀 68 passed, SysV SHM 항목 전후 3개 동일. 실제 LFM smoke는 `invalid_schema`로 rule fallback하여 실패(원본 출력 1790857389_pytest.log), 성공으로 표시하지 않음. 현재 실제 응답/프롬프트를 조사 중이며 active S2의 끝단 증거로 수정 후 재실행한다. agy 독립 리뷰 4분 timeout으로 미완료; 다른 검토 경로 사용. S3 평가기 작성은 준비 작업만 병행하며 실제 품질 단계는 아직 활성화하지 않는다. delegated 보고는 부모가 직접 재검증해야 닫힌다.

## 미검증/유예
- 실제 Foundry 주 모델이 context를 제공하고 hide/unhide를 적절히 선택하는지(B)는 미검증 유지.
- 실패 로그 숨김 확대·영속 저장·branch 인증·Hermes UI/MCP·운영 활성화는 비범위.
- 실제 LFM 품질/맥락 개선은 실험 이후에만 판정. fallback을 요약 성공으로 표시하지 않는다.
