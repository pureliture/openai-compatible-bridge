# Foundry protocol hide 실행 기록

## Admission
요구사항 requirements.md 및 설계 design.md는 승인됨. 최신 사용자 `/agentic-execution + agentic모드로 구현시작해`가 구현 권한과 실행 계약을 명시했다. 실행 계약 agentic-execution, TDD 및 부모 직접 검증. 운영·유료 주모델·push/merge 제외. 기존 다른 dirty PG/docs/test 파일 보존.

Baseline HEAD e358cf9936a5e955ae4b711bf6afe208e41935de. 부모 baseline tests/test_context_compaction.py + test_foundry_tool_calling.py 53 passed/기존 warning1. LBrain read-only 항목 없음. 현재 Foundry adapter는 native 각 body/response를 정규화하며 plan_request protocol gate만 Chat Completions 비stream으로 제한. protocol 단순 allowlist 추가는 완료 증거가 아님.

## 순차 observable slices
| 단계 | 상태 | 직접 evidence gate |
|---|---|---|
| R Responses 비스트리밍 | 완료 | 부모 직접 native HTTP fixture/bridge focused91passed, diff통과. distinct item_id/call_id 연결, private hide/after/list/exactrestore/외부calls/usage와 cached6 합산 확인. 실제 유료Foundry/localLFM은 이 단계 미실행. local checkpoint에 포함 |
| A Anthropic 비스트리밍 | 대기 | tool_use/tool_result 원형 fixture 및 공통 경로 |
| G Google 비스트리밍 | 대기 | functionCall/functionResponse 원형 fixture 및 공통 경로 |
| X xAI 비스트리밍 | 대기 | xAI adapter native fixture 및 공통 경로, ChatCompletions 회귀 |
| RS Responses streaming | 대기 | 실제 upstream SSE 조립/private 소비/final SSE/usage/cancel |
| AS Anthropic streaming | 대기 | 해당 native SSE+공통 gate |
| GS Google streaming | 대기 | 해당 native SSE+공통 gate |
| XS xAI streaming | 대기 | 해당 native SSE+공통 gate |
| CS ChatCompletions streaming | 대기 | 해당 native SSE+공통 gate |
| V 최종검증 | 대기 | full suite/실제localLFMsynthetic/독립review/localcommit |

현재 R만 활성. delegated 보고는 부모가 command/diff를 재확인하고 checkpoint commit 이후 닫는다. 서비스/schema/큐 신설 없이 기존 run_turn/store/LFM/metered/adapter 재사용. streaming 보류 사용자 승인, 최종응답 SSE 전달. 실제 유료 Foundry 소비자 검증은 별도 승인 전 미실행. PG DSN없으면 skip 명시, local/운영PG 대체 금지.
