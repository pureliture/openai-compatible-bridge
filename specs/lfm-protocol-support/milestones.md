# Foundry protocol hide 실행 기록

## Admission
요구사항 requirements.md 및 설계 design.md는 승인됨. 최신 사용자 `/agentic-execution + agentic모드로 구현시작해`가 구현 권한과 실행 계약을 명시했다. 실행 계약 agentic-execution, TDD 및 부모 직접 검증. 운영·유료 주모델·push/merge 제외. 기존 다른 dirty PG/docs/test 파일 보존.

Baseline HEAD e358cf9936a5e955ae4b711bf6afe208e41935de. 부모 baseline tests/test_context_compaction.py + test_foundry_tool_calling.py 53 passed/기존 warning1. LBrain read-only 항목 없음. 현재 Foundry adapter는 native 각 body/response를 정규화하며 plan_request protocol gate만 Chat Completions 비stream으로 제한. protocol 단순 allowlist 추가는 완료 증거가 아님.

## 순차 observable slices
| 단계 | 상태 | 직접 evidence gate |
|---|---|---|
| R Responses 비스트리밍 | 완료 | 부모 직접 native HTTP fixture/bridge focused91passed, diff통과. distinct item_id/call_id 연결, private hide/after/list/exactrestore/외부calls/usage와 cached6 합산 확인. 실제 유료Foundry/localLFM은 이 단계 미실행. local checkpoint에 포함 |
| A Anthropic 비스트리밍 | 완료 | 부모 무제외 focused133passed/diff통과. native tool_use/tool_result 연결·hide/after/list/exactunhide·mixed/external·cache계수 보존 확인. 실제외부provider 미호출 |
| G Google 비스트리밍 | 완료 | 부모 무제외 focused160passed/diff통과. functionCall/functionResponse ID·thoughtSignature·병렬 결과·충돌 방지·cache계수·hide/after/list/exactunhide 확인. native fixture 증거, 실제외부provider 미호출 |
| X xAI 비스트리밍 | 완료 | 부모 무제외 focused167passed/diff통과. native call_id/continuation·private hide/after/list/exactrestore·mixed/external·strict cache계수 확인. 실제외부provider 미호출 |
| RS Responses streaming | 완료 | 부모 무제외 focused257passed/diff통과. native chunked upstream stream=True, private 보류/최종publicSSE/usage·cache/완료EOF/timeout/buffer/cancel/metered모델복원 fixture 검증. 실제provider/네트워크disconnect 미검증 |
| AS Anthropic streaming | 완료 | 부모 무제외244passed/diff통과. native message_start/tool_use/input_json_delta/message_stop→private continuation/finalSSE/after/list/exactrestore, usage출처·cache·0/missing·EOF/cancel/metered fixture 검증. 실제외부provider 미호출 |
| GS Google streaming | 완료 | 부모 무제외279passed/diff통과. native streamGenerateContent SSE→snapshot호출/ID/thoughtSignature/반복방지/private continuation/finalSSE/exactrestore/usage/cache/취소 fixture검증. 실제외부provider 미호출 |
| XS xAI streaming | 완료 | 부모 무제외281passed/diff통과. native stream=True chunked SSE·partial/done arguments·call_id·private 소비·after/list/exactrestore·cache/usage/EOF/cancel/metered fixture 검증. 실제provider 미호출 |
| CS ChatCompletions streaming | 완료 | 부모 무제외357passed/diff통과. native chat chunked SSE/None default/fragmentcalls/private소비/finalSSE/usage/[DONE]/EOF/cancel/meteredfixture 검증. 실제provider 미호출 |
| V 최종검증 | 대기 | full suite/실제localLFMsynthetic/독립review/localcommit |

현재 CS Chat Completions 스트리밍만 활성. XS checkpoint 50c4895, GS checkpoint e4f43de. RS checkpoint 4dd8a1d, AS checkpoint 6eac072. R f485a14, A 98e30ac, G 7dd4a85, X 04e517c checkpoint 완료. delegated 보고는 부모가 command/diff를 재확인하고 checkpoint commit 이후 닫는다. 서비스/schema/큐 신설 없이 기존 run_turn/store/LFM/metered/adapter 재사용. streaming 보류 사용자 승인, 최종응답 SSE 전달. 실제 유료 Foundry 소비자 검증은 별도 승인 전 미실행. PG DSN없으면 skip 명시, local/운영PG 대체 금지.
