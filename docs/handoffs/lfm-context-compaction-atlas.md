# Atlas 인계: LFM 컨텍스트 축약

## 상태와 권한

- 구현 위치: `/Users/ddalkak/Projects/openai-compatible-bridge/.worktrees/lfm-context-summary`
- 브랜치: `daedalus/lfm-context-summary`
- 구현 커밋: 로컬 커밋 후 이 항목을 실제 SHA로 갱신한다.
- 운영 반영/활성화: 하지 않았다. `CONTEXT_COMPACTION_ENABLED`와 `CONTEXT_COMPACTION_LFM_ENABLED`의 기본값은 모두 `false`다.
- Daedalus는 로컬 코드와 문서만 변경했다. 운영 설정·Secret·클러스터·main·원격 브랜치·PR은 변경하지 않았다. 빌드 배포와 운영 검증은 Atlas가 별도 승인 범위에서 수행한다.

## 실제 동작 범위

LFM 요약은 Hermes UI 버튼이나 Hermes 자체 대화 자동 축약이 아니다. 기존 브리지의 `POST /v1/chat/completions` 처리 중, 주 모델이 브리지 내부 `compact_context` 함수를 호출할 때 선택적으로 실행된다. 대상은 Foundry의 OpenAI Chat Completions 비스트리밍 요청뿐이다. 스트리밍과 다른 제공업체/프로토콜에서는 축약하지 않는다.

브리지는 특정 과거 `role=tool` 결과 하나를 기존 Ollama client로 전송하고 요약을 검증한다. 생성 요약은 비신뢰 정보로 표시하고, 테스트 결과·오류·명령 결과·ID·경로 등 필수 줄은 원문 그대로 별도 보존한다. `unhide_context`는 저장된 원문을 복구하며 명령을 다시 실행하지 않는다. 실패하거나 검증되지 않은 요약은 LFM 성공으로 기록하지 않고, 기존의 안전한 규칙 발췌로 대체하거나 원문을 유지한다. Laya 선택 기능은 별도이며 변경하지 않았다.

## 변경 파일

- `openai_compatible_bridge/context_compaction.py`: LFM 선택, 검증된 압축 적용, 원문 복원, 호출 횟수/결과 측정, payload를 남기지 않는 `lfm_calls`·`lfm_applied`·`lfm_fallback` 로그 필드.
- `openai_compatible_bridge/lfm_summary.py`: Ollama 요약 prompt, JSON 형식 검사, 비신뢰 입력 경계, 생성 결과의 제한적 검증.
- `openai_compatible_bridge/main.py`: 기존 Foundry 비스트리밍 compact 경로에 metered Ollama LFM 호출을 연결하고 예산 예약·복원 처리.
- `openai_compatible_bridge/providers/ollama.py`: 내부 요약 요청별 timeout과 Ollama 옵션을 지원.
- `tests/test_lfm_summary.py`: 검증·안전 fallback·내부 호출·압축·복원 및 내용 없는 성공 로그 테스트.
- `tests/test_lfm_live_integration.py`: opt-in 합성 데이터를 사용한 실제 Ollama LFM 요청, 압축, `unhide_context` 원문 동일성 테스트.
- `tests/test_compaction_cost_integration.py`: Ollama 호출의 metered budget gate와 비용 원장 테스트.
- `.env.example`: 선택 기능의 기본 설정 예시. 이 저장소는 이미 `.env.example`을 환경 변수 안내 파일로 사용한다.
- `specs/context-compaction/design.md`, `specs/context-compaction/implementation-handoff.md`: Laya와 다른 생성 요약 계약, 실제 진입점, 제한 사항을 기록.

## 설정과 기본값

| 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `CONTEXT_COMPACTION_ENABLED` | `false` | 전체 내부 compact/list/unhide 기능의 스위치 |
| `CONTEXT_COMPACTION_LFM_ENABLED` | `false` | Ollama LFM 생성 요약 스위치. 위 전체 스위치도 켜야 함 |
| `CONTEXT_COMPACTION_LFM_MODEL` | `lfm2.5-thinking:latest` | Ollama 모델 이름 |
| `CONTEXT_COMPACTION_LFM_MAX_INPUT_CHARS` | `50000` | 입력 JSON 문자 제한 |
| `CONTEXT_COMPACTION_LFM_MAX_INPUT_BYTES` | `12288` | system 지시문 텍스트와 source JSON을 합친 UTF-8 byte 제한 |
| `CONTEXT_COMPACTION_LFM_MAX_OUTPUT_TOKENS` | `384` | Ollama `num_predict` 출력 제한 |
| `CONTEXT_COMPACTION_LFM_TIMEOUT_SECONDS` | `60` | 개별 LFM HTTP 요청 timeout |
| `CONTEXT_COMPACTION_MAX_INTERNAL_ROUNDS` | `3` | 한 turn의 내부 주 모델 호출 최대 횟수 |
| `CONTEXT_COMPACTION_TTL_SECONDS` | `86400` | 원문/축약 항목의 프로세스 내 보관 시간 |
| `CONTEXT_COMPACTION_MAX_BYTES` | `67108864` | 프로세스별 전체 저장 예산 |
| `OLLAMA_BASE_URL` | `.env.example`은 `http://host.docker.internal:11434` | bridge가 접근할 Ollama native API 주소 |

입력 byte 제한은 토크나이저로 센 입력 token 수가 아니다. 출력은 token 상한이 있지만 입력은 문자/byte로 제한된다. 설정값의 최대 허용치는 입력 문자 100,000, 입력 byte 12,288, 출력 token 1,024, timeout 300초다. LFM 호출은 한 turn에 최대 1회다.

실제 합성 live test는 `http://100.97.224.92:11434`의 `lfm2.5-thinking:latest`를 사용했다. 운영에서 `OLLAMA_BASE_URL`이 실제 bridge pod에서 도달 가능한지 Atlas가 다시 확인해야 한다. 테스트에서 쓴 주소를 운영 설정에 자동 복사하지 않는다.

`COST_TRACKING_ENABLED=true`인 운영 환경에서는 기존 비용 정책대로 Ollama의 `COST_PROVIDER_BILLING_JSON` 분류와 필요한 경우 실제 계약에 맞는 `COST_PRICING_JSON` 가격을 확인해야 한다. 가격을 추정해 입력하지 않는다. 비용 gate가 LFM 시도를 차단하면 LFM이 성공한 것으로 표시되지 않고 안전한 기존 규칙 경로로 돌아간다.

## 테스트 및 검증 결과

모든 자동 테스트는 `/Users/ddalkak/Projects/openai-compatible-bridge/.worktrees/lfm-context-summary`에서 실행했다. PostgreSQL을 사용하는 테스트 fixture는 외부 DSN을 사용하지 않고 `tmp_path` 아래에 임시 native PostgreSQL을 만든다. 운영 PostgreSQL은 사용하지 않았다.

- 집중 회귀: `uv run pytest tests/test_lfm_summary.py tests/test_lfm_live_integration.py tests/test_context_compaction.py tests/test_laya_compaction.py tests/test_compaction_protection.py tests/test_compaction_cost_integration.py tests/test_foundry_chat.py tests/test_foundry_tool_calling.py -q` — 145 passed, 0 failed, 1 skipped.
- 실제 LFM 통합: `RUN_LFM_LIVE_INTEGRATION=1 uv run pytest -s tests/test_lfm_live_integration.py -q` — 1 passed. 합성 catalog/tool 결과만 전송했다. 응답의 주제 관련성, 필수 command result/ID/path 유지, 원문 대비 20% 초과 축소, 다음 turn의 원문 동일성 복원을 검사했다. 안전 fallback만 성공한 경우는 `compaction_source == "lfm"` 조건에서 실패하도록 테스트한다.
- 전체 suite: `uv run pytest -q` — 816 passed, 0 failed, 1 skipped. 기본 전체 실행에서는 opt-in live test가 skip되며 위에서 별도로 실행해 통과시켰다.
- `git diff --check` — 통과.
- 집중 회귀에는 Foundry 비동기/non-stream 및 SSE/tool-call streaming 테스트가 포함됐다. streaming에서는 LFM 요약 경로를 실행하지 않는 기존 정책을 검증했다.
- 비용 테스트는 metered Ollama 요청이 허용될 때 원장에 실제 Ollama prompt/completion usage를 남기고, 예산 차단 때 Ollama HTTP 요청 없이 규칙 기반 fallback으로 이어지는지 확인했다.

실제 live test의 성공 기준은 HTTP 200만이 아니다. 저장된 항목이 LFM 출처여야 하고, 구체적인 synthetic 주제를 포함하며, 압축 및 `unhide_context` 원문 동일성이 모두 통과해야 한다.

## Atlas 운영 적용 순서

1. 먼저 이 로컬 커밋의 코드/테스트를 Atlas 승인된 PR·릴리스 경로로 검토한다. 이 인계만으로 push, merge, 배포, 운영 활성화 권한이 생기지 않는다.
2. 배포 전 bridge pod에서 Ollama 주소와 `lfm2.5-thinking:latest` 사용 가능 여부를 확인한다. Secret은 이 기능에 새로 필요하지 않으며 코드/로그에 넣지 않는다.
3. replica 수와 라우팅을 확인한다. 저장소는 pod별 프로세스 메모리이며 공유 DB가 아니다. 여러 replica를 사용하면 같은 상관 헤더 요청이 같은 pod에 도달하도록 sticky routing이 필요하다. 이를 보장할 수 없으면 기능을 운영 활성화하지 않는다.
4. 비용 추적이 켜져 있으면 Ollama payer 분류와 실제 가격표를 확인한다. `metered`이면 기존 예산 gate가 요청을 허용하는지 사전 확인한다.
5. 우선 코드를 배포하더라도 두 기능 스위치를 `false`로 둔다. canary 절차와 복구 담당자를 확인한 뒤, 제한된 canary에서만 `CONTEXT_COMPACTION_ENABLED=true` 및 `CONTEXT_COMPACTION_LFM_ENABLED=true`를 적용한다. 전체 스위치는 기존 compaction 기능도 켜므로 LFM만 켜는 설정은 아니다.
6. 실제 소비자 canary는 승인된 Foundry OpenAI alias와 안정적인 `x-hermes-conversation` 헤더로 수행한다. 대화와 tool output은 synthetic fixture만 사용하고, 짧지 않은 과거 `role=tool` 결과 및 정상적인 `assistant.tool_calls`/결과 ID 짝을 보낸다. 주 모델에게 결과를 압축해 이후에 참조하도록 요청하며, `tool_choice`를 외부 도구로 강제하지 않는다.
7. bridge 로그에서 `context_compaction ... lfm_calls=1 lfm_applied=True lfm_fallback=False`를 확인한다. 이 로그에는 입력 본문이 포함되지 않는다. 성공 판정은 반드시 `lfm_applied=True`와 `lfm_fallback=False`로 한다. `lfm_calls`는 요약 함수 진입 횟수이며 비용 gate가 outbound HTTP 전에 막아도 증가할 수 있다. `calls`는 주 모델 호출 횟수다.
8. 같은 consumer 대화와 같은 pod에서 다음 요청에 synthetic sentinel 원문 복원을 요청한다. 사용 가능한 내부 목록/복원 도구 흐름을 거쳐 sentinel이 돌아오는지 확인하고, 내부 도구 이름이 consumer 응답에 노출되지 않는지 확인한다. byte-for-byte 복원은 별도 실제 live integration test에서 이미 직접 확인했다.
9. canary가 성공해도 범위를 확대하기 전까지 stream·다른 Foundry 프로토콜·다른 provider는 미지원으로 유지한다. 이 기능은 tool output의 비밀 정보를 지우지 않는다. 민감 자료가 들어간 출력을 Ollama에 보내도 되는지는 운영자가 별도로 확인해야 한다.

## 롤백

- LFM만 중지하고 기존 규칙 기반 compaction을 유지하려면 `CONTEXT_COMPACTION_LFM_ENABLED=false`로 바꾼다.
- 전체 hide/compact 내부 도구를 끄려면 `CONTEXT_COMPACTION_ENABLED=false`로 바꾼다.
- 코드 문제가 있으면 Atlas 승인 절차에 따라 이전 bridge 이미지/릴리스로 되돌린다. 새 DB schema나 migration은 없다.
- 프로세스 재시작 또는 다른 pod로의 이동 뒤에는 해당 프로세스의 `unhide_context` 항목이 없을 수 있다. 저장소를 영속 원문 보관소로 간주하지 않는다. rollback 전후 모두 원본 대화/tool 결과를 consumer가 보존하는지 확인한다.

## 남은 제한과 위험

- `x-hermes-conversation`은 raw 상관 키이지 인증된 사용자나 분기 ID가 아니다. 같은 값을 공유하는 분기/cron/위임 작업은 같은 프로세스 상태를 사용할 수 있다.
- 원문과 축약 상태는 pod별 메모리·TTL에만 저장된다. 재시작/replica 이동 후 목록 및 `unhide_context` 복구를 보장하지 않는다.
- 생성 요약의 사실 정확성을 완전히 증명할 수 없다. 검증은 일반 문구·비신뢰 지시·원문에 없는 숫자/경로/ID 등을 제한하며, 필수 원문 증거를 별도 보존한다. 이것은 secret scrubber나 접근 제어가 아니다.
- `CONTEXT_COMPACTION_ENABLED`를 켜면 LFM 외 기존 compaction 도구도 활성화된다. 운영에서는 먼저 canary 범위를 제한한다.
- 운영 배포/활성화, pod-to-Ollama 연결 재확인, 운영 consumer canary는 수행하지 않았다. Atlas가 담당한다.
