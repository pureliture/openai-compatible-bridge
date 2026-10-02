# Foundry 전체 protocol hide — 검증 결과·로컬 PR 본문

## 결론과 범위
승인된 Foundry Chat Completions/Responses/Anthropic/Google/xAI의 비스트리밍·스트리밍 코드 경로를 구현했다. 내부 hide/list/unhide 처리, 결과-only 요약, 후속 유지, exact restore와 public 외부 호출을 보존한다. Streaming은 사용자 승인대로 native upstream을 수집하고 private loop 이후 최종 응답만 SSE로 전달하므로 최초 출력이 늦어질 수 있다.

직접 Vertex/Ollama 주모델 확대, 운영 모델 전환, 배포/Secret/push/merge는 수행하지 않았다. 기존 dirty PG/docs 작업은 보존하고 이번 commit에서 제외한다.

## 직접 실행 증거
최종 코드가 고정된 상태에서 부모 실행:
```
.venv/bin/python -m pytest -q
RUN_LFM_PROTOCOL_INTEGRATION=1 .venv/bin/python -m pytest -q tests/test_compaction_protocols_live_lfm.py
.venv/bin/python -m compileall -q openai_compatible_bridge tests
git diff --check
```
- 전체: 1219 passed,101 skipped,1 기존 Starlette warning,15.45초,exit0.
- 5protocol×2mode 실제 auxiliary LFM 통합: 10 passed,1 warning,4.33초,exit0.
- compile/diff 통과.
- Skip은 격리 PG 접속/opt-in 조건 미충족 검사 포함. skipped를 passed로 계산하지 않는다.

## 무엇을 실제로 확인했는가
주 모델 요청/응답은 native HTTP/SSE MockTransport 시험 자료다. 실제 FoundryChatClient body conversion과 stream_chat(stream=True), chunk 조립, main endpoint, 내부 도구 continuation을 실행했다. 유료 Foundry 요청이 아니다.
보조 요약은 실제 로컬 Ollama lfm2.5-thinking:latest, 합성 입력만,384outputtokens/60초/12KiB/turn1call로 실행했다. 각 case 3442→536UTF8bytes, compaction_source=lfm, after/list/rendered-body exactunhide/caller 원문 및 call ID 불변을 확인했다. Rule fallback이면 실패다.
실제 생성: `The output indicates a synthetic catalog with metadata, specifying Build ID and artifact path.` 이 테스트는 protocol 접합과 복원을 검증하며 모든 입력의 의미 품질 벤치마크를 대신하지 않는다. 별도 결과-only 품질 사례는 specs/lfm-result-only/verification-and-pr.md에서 구분한다.

## 구현 책임 재사용과 변경
- context_compaction plan/run_turn/store/TTL/digest/pending/LFM result-only 재사용. 다섯 protocol gate를 비스트리밍/스트리밍에 연결.
- 기존 native adapters 재사용. Responses call_id/item_id, Anthropic tool_use/tool_result, Google functionCall/functionResponse ID/thoughtSignature/parallel-result, xAI call_id를 검증.
- main bounded native stream collector가 private 응답을 보류하고 최종 public SSE/finish/usage/[DONE] 제공. generate를 SSE로 포장해 upstream streaming을 대체하지 않음.
- cache 사용량은 실제 반환 값만 보존. 미제공/잘못된 usage를 계측0으로 만들지 않음. 실패·취소·EOF/시간/크기 제한 검증.
- 기존 forcedtool/collision/header없음/disabled 경로 유지. stale 미지원 테스트를 승인된 지원 범위로 갱신하고 미확인 protocol/비Foundry 거부 검사는 유지.
- 새 서비스/DBschema/queue 없음. streaming structured-output repair는 기존 Ollama-only 책임이며 이 Foundry 경로와 분리.

## 독립 리뷰 발견과 수정
정적 리뷰의 세 후보를 TDD로 실제 재현하고 수정했다. 새 회귀39개와 관련567개 통과 보고 후 부모 전체/live 다시 검증.
1. finish+usage 이후 DONE없는 EOF가 실패여도 HTTP 기록 usage를 성공값으로 저장: native parser 종료 예외를 HTTP context exit에 전달해 실패 usage unknown 처리. 정상 usage 유지.
2. native 오류 문자열이 private tool명/인자를 public에 출력: 활성 private 경로에서 고정 안전 메시지/허용 코드로 정제. 제외 경로 기존 오류 보존.
3. Google nonstream native part 무제한 보관: thoughtSignature/인자/ID/mapping overhead까지 직렬화 총량 제한. 초과 시 안전502로 continuation/LFM 미실행. 기존 stream과 같은 token-derived byte cap 사용.

## 미검증·운영 경계
- 실제 유료 Foundry consumer 선택/작업 정확도/canary, 실제 네트워크 disconnect, 운영 지연/캐시 순이익 미검증.
- PG DSN 미설정으로 PostgreSQL 영속 원장 회귀 이번 실행은 미검증. MockHTTP metered admission/attempt/modelcontext복원은 검증했으나 PG증거로 주장하지 않음.
- 로컬 프로세스 메모리/TTL 원문 저장. 재시작/다른replica 복원보장없음. x-hermes-conversation은 상관키이지 인증/분기ID아님.
- featureflags 기본false, OLLAMA_BASE_URL 환경변수 재사용. 맥미니Tailscale DNS 운영 설정·Pod연결·배포는 Atlas 별도 실행.

## Atlas 후속
최종 commit 정확 소스로 빌드하고 기존 미커밋 전체폴더 빌드 금지. 운영 Foundry protocol을 ChatCompletions로 바꿀 필요 없이 해당 native경로 사용. 승인 격리PG와 운영 syntheticconsumer 검증 후 제한 활성화. 결과를 읽어 private비노출/실제LFM적용/정확복원을 각각 확인한다. 유료call한도 별도 승인 필요.
Rollback 기존 두flags OFF와 이전image/config복구. 이번명세 구현이 운영배포 승인이나 mainpush/merge승인을 추가하지 않는다.
