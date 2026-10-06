# Foundry protocol hide 지원 설계

상태: 승인됨 — 사용자가 요구사항·설계 두 문서를 승인했다.
요구사항: [requirements.md](requirements.md), 승인됨.

## 재사용 경계
- context_compaction.py의 plan_request/run_turn, 원문 저장/가시성/복원과 LFM 결과-only 요약을 재사용한다.
- main.py의 기존 _maybe_context_compaction에서 metered Ollama 생성과 주 모델 비용 context 복원을 재사용한다.
- Foundry provider의 기존 protocol별 body 변환/응답 정규화를 사용한다. 운영 모델을 다른 요청 방식으로 바꾸지 않는다.
- 기존 원문 호출과 public 함수 도구 유지. 내부 함수만 소비한다. Hermes UI나 MCP는 변경하지 않는다.

## 비스트리밍
현재 Foundry+Chat Completions 제한을 대상 protocol별 검증 후 확장한다. protocol 이름만 allowlist에 추가하고 동작한다고 간주하지 않는다. 각 adapter의 실제 요청 body에서 tools/선택정책/assistant 함수 호출/도구 결과가 맞게 변환되는지 확인하고 응답이 run_turn의 정규화 형태로 들어오는지 검증한다.

Responses부터 Anthropic, Google, xAI 순서로 한 단계씩 진행한다. 각 단계는 실제 native HTTP fixture를 통과하고 hide→요약 적용→후속 유지→list→unhide의 관찰 결과로 닫는다. 외부 도구/mixed calls/강제 도구/충돌의 기존 안전 처리와 비용 계상 회귀를 포함한다.

## 스트리밍
사용자가 첫 출력 지연을 수용했으므로 내부 도구 여부를 확정하기 전에는 public SSE를 보내지 않는다. 대상은 기능을 켠 요청이며 기존 제외 조건에서 일반 streaming 경로를 유지한다.

provider의 실제 upstream stream 이벤트를 정규화해 주 모델 응답 하나를 수집한다. 내부 함수 호출은 브리지에서 처리하고 제한 횟수 안에서 모델을 다시 호출한다. 최종 응답만 OpenAI-compatible SSE로 전달한다. 외부 도구 호출은 클라이언트 실행 대상으로 보존한다. private 중간 답변/함수명/인자는 출력하지 않는다. 마지막 finish reason, usage(요청된 경우), [DONE]을 유지한다.

Responses/Anthropic/Google/xAI/기존 Chat Completions stream 각각의 조각난 인자·여러 tool calls·text·finish/usage를 protocol fixture로 확인한다. 일반 generate 호출을 SSE로 포장한 것만으로 upstream streaming 지원을 완료했다고 주장하지 않는다.

연결 중단/timeout/중간 provider 오류 시 원문 보존과 비용 시도 기록을 확인한다. SSE 시작 전 오류와 시작 뒤 오류를 구분해 기존 public 오류 계약을 유지한다. 기존 호출 round/LFM 최대1회/timeout/입력·출력 제한은 유지한다. structured output repair와 스트리밍 경계는 기존 책임을 먼저 확인하고 중복 처리하지 않는다.

## 증거와 완료 판정
- 각 대상 native request/response 또는 SSE fixture는 어댑터 변환을 검증한다.
- 합성 local LFM은 실제 요약 생성/본문 적용/복원을 검증한다.
- 위 둘을 실제 Foundry 공급자 canary 성공으로 합쳐 보고하지 않는다. 실제 호출은 별도 비용 승인 필요.
- 모든 대상 비스트리밍과 streaming의 내부 소비/외부 전달/usage/복원을 순차 검증한다.
- 최종 focused/full suite/compile/diff/독립 review 후 local commit과 PR 본문 작성. 운영 배포는 Atlas에 별도 전달한다.

## 제외 범위
직접 Vertex/Ollama 주 요청 지원 확대, 호출 명령 자체 압축, 목적 기반 요약, 영속 저장/분기 인증, 운영 모델 전환/배포·Secret 변경은 제외한다.

**이 설계 문서는 사용자 승인을 받았다.**
