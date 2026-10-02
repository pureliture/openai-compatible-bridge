# Foundry 전체 요청 방식의 LFM hide 지원 요구사항

상태: 승인됨 — 사용자가 요구사항·설계 두 문서를 승인했다.
실행 계약: 기존 agentic-execution 유지. 문서 승인은 구현 착수와 별개다.

## 목표
현재 운영 Foundry 모델의 요청 방식을 바꾸지 않고 hide_context/list_context_items/unhide_context를 사용할 수 있도록 한다.

## 확인한 사실
- 기존 브리지는 Foundry OpenAI Chat Completions 비스트리밍에만 hide 기능을 적용한다. streaming 요청과 다른 protocol은 제외한다.
- Atlas 운영 점검에서 등록된 방식은 openai_responses, anthropic_messages, google_generate_content, xai_responses이며 Chat Completions 별칭은 확인되지 않았다.
- 이 세션 모델은 foundry:gpt-6.1-sol이고 저장소 예시의 해당 별칭 protocol은 openai_responses다. 로컬 profile 설정 파싱은 PyYAML 부재로 확인하지 못했다. 운영 사실은 Atlas 점검 근거로 구분한다.
- LBrain read-only 프로젝트 조회는 항목 없음이었다. 로컬 코드와 Atlas 보고를 근거로 범위를 정했다.

## 사용자 결정
- Foundry의 위 네 방식과 기존 Chat Completions를 모두 지원한다. 직접 Vertex/Ollama 주 모델 경로는 제외한다.
- 비스트리밍과 스트리밍을 모두 포함한다. 순차 단계로 구현·검증한다.
- streaming은 기능을 켠 해당 요청에서 내부 호출 여부가 확정될 때까지 전송 보류를 허용한다. 내부 처리가 끝난 최종 응답을 SSE로 전달한다. 첫 출력 지연 가능성을 수용한다.
- 결과-only 계약 유지: 원래 호출 이름/인자/명령/ID와 Hermes 원본 대화는 그대로 유지하고 과거 도구 결과 본문만 LFM으로 요약한다. 작업 의도·다음 행동·명령 압축은 제외한다.

## 필수 동작
- 모든 대상 protocol에서 내부 도구를 주 모델에 제공하고 브리지 안에서 소비한다. MCP/UI 변경 없음.
- 실제 도구 호출의 정규화와 각 protocol의 결과 연결을 유지한다. 원래 외부 도구 호출을 클라이언트에 정상 전달한다.
- hide→실제 LFM 생성→본문 적용→후속 유지→list→exact unhide를 검증한다. 실패 fallback과 LFM 성공을 구분한다.
- streaming에서 private 도구 이름/인자/중간 답변을 SSE에 노출하지 않는다. 최종 응답의 외부 도구 호출·finish reason·usage·[DONE]을 보존한다.
- 기존 세션 상관 키/TTL/메모리/원문 보호/호출·시간·크기 제한/비용 gate와 원장을 유지한다. stream 중 취소·오류 시 계상과 자원 해제 검증.
- 비활성/헤더 없음/도구 충돌 등 기존 제외 조건의 일반 처리 경로를 유지한다.

## 순차 완료 조건
1. Responses 비스트리밍: protocol 원본 HTTP fixture와 브리지 내부 처리·요약/복원 테스트.
2. Anthropic 비스트리밍: tool_use/tool_result 연결과 위 공통 동작.
3. Google 비스트리밍: functionCall/functionResponse 연결과 위 공통 동작.
4. xAI 비스트리밍: 실제 해당 adapter 계약과 위 공통 동작. 기존 Chat Completions 회귀 유지.
5. 각 protocol streaming을 위 순서로 검증: upstream SSE 조립, private 소비, 최종 public SSE, 실패·취소·usage.
6. focused/full suite/diff, 합성 local LFM 검증, 독립 리뷰, 로컬 commit/PR 본문. 단계별 직접 검증 후 다음 단계.

## 권한과 미검증 경계
- 운영 배포·설정·Secret·모델 변경은 Atlas 소유. 이번 명세는 코드 확장 범위다.
- 유료 외부 모델 시험은 비용 승인 없이 실행하지 않는다. native protocol HTTP fixture와 합성 local LFM 통합을 실제 공급자 사용성 증거로 주장하지 않는다.
- PG 검증은 승인된 격리 서버만. 접속값이 없으면 비용 DB 검사 미검증 명시. 운영/로컬 PG 대체 금지.
- 기존 dirty PG/이전 문서 작업 보존, 관계없는 변경 금지.

**이 요구사항 문서는 사용자 승인을 받았다.**
