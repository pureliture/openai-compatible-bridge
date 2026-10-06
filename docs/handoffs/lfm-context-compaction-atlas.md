# Atlas 인계: LFM 컨텍스트 축약

## 상태와 권한

- 구현 위치: `<bridge-repository>/.worktrees/lfm-context-summary`
- 브랜치: `daedalus/lfm-context-summary`
- 구현 커밋: `53ac2de8cedb555988619bd059d4edce58f05ba2` (로컬 전용, push하지 않음). 검증 결과와 Atlas 인계 내용은 별도의 후속 로컬 커밋에 기록했다.
- 운영 반영/활성화: 하지 않았다. `CONTEXT_COMPACTION_ENABLED`와 `CONTEXT_COMPACTION_LFM_ENABLED`의 기본값은 모두 `false`다.
- Daedalus는 로컬 코드와 문서만 변경했다. 운영 설정·Secret·클러스터·main·원격 브랜치·PR은 변경하지 않았다. 빌드 배포와 운영 검증은 Atlas가 별도 승인 범위에서 수행한다.

## 실제 동작 범위

LFM 요약은 Hermes UI 버튼이나 Hermes 자체 `/compress`·`/compact` 대화 축약이 아니다. 기존 브리지의 `POST /v1/chat/completions`에서 주 모델에게 `hide_context`, `list_context_items`, `unhide_context`를 OpenAI 함수 도구 형식으로 제공하고 브리지 내부에서 소비한다. 정식 숨김 진입점은 `hide_context` 하나뿐이다. 대상은 Foundry OpenAI Chat Completions 비스트리밍 요청이며 streaming 및 다른 provider/protocol 경로는 지원하지 않는다.

`hide_context`는 필수 인자 `tool_call_id` 하나를 받고 현재 대화 안에서 유일한 과거 `role=tool` 결과에만 적용한다. 사용자 메시지나 assistant/tool call 자체는 숨기지 않는다. 원문을 보관하고 upstream 요청 사본의 해당 결과 본문만 LFM 생성 요약과 필수 원문 발췌로 대체한다. 원래 `assistant.tool_calls`, 그 `tool_call_id`, 외부 도구 호출 흐름은 유지하며 도구를 재실행하지 않는다. `item_id`는 `list_context_items`로 찾은 뒤 `unhide_context`의 필수 인자로 쓴다. `unhide_context`는 저장 원문만 복원하며 새 요약을 만들지 않는다.

LFM 생성 문장은 비신뢰 참고 정보이고, 테스트 결과·오류·명령 결과·ID·경로 등 필수 줄은 원문 그대로 별도 보존한다. 실패/잘림/검증 거부는 `lfm_applied=false`이며 안전한 기존 규칙 발췌 또는 원문 유지로 fallback한다. fallback 성공은 LFM 성공이 아니다. Laya 선택 기능은 별도이며 변경하지 않았다.

## 변경 파일

- `openai_compatible_bridge/context_compaction.py`: LFM 선택, 검증된 숨김 적용, 원문 복원, 호출 횟수/결과 측정, payload를 남기지 않는 `lfm_calls`·`lfm_applied`·`lfm_fallback` 로그 필드.
- 주 모델에게 제공하는 도구는 정확히 `hide_context`, `list_context_items`, `unhide_context`다. `hide_context` JSON schema는 `tool_call_id` 필수, `item_id` 불허이고 runtime dispatch도 이를 검증한다. tool_call_id는 현재 대화에 매칭되는 유일한 과거 `role=tool` message를 가리켜야 한다. `list_context_items`는 숨긴 item의 `item_id`와 메타데이터를 돌려주고 `unhide_context`는 그 `item_id`로 원문을 복원한다. 추가 인자, 누락 인자, 없는 결과, 중복 tool_call_id는 실패 처리한다.
- `openai_compatible_bridge/lfm_summary.py`: Ollama 요약 prompt, JSON 형식 검사, 비신뢰 입력 경계, 생성 결과의 제한적 검증.
- `openai_compatible_bridge/main.py`: 기존 Foundry 비스트리밍 `hide_context` 경로에 metered Ollama LFM 호출을 연결하고 예산 예약·복원 처리.
- `openai_compatible_bridge/providers/ollama.py`: 내부 요약 요청별 timeout과 Ollama 옵션을 지원.
- `tests/test_lfm_summary.py`: 검증·안전 fallback·내부 호출·압축·복원 및 내용 없는 성공 로그 테스트.
- `tests/test_lfm_live_integration.py`: opt-in 합성 데이터를 사용한 실제 Ollama LFM 요청, 압축, `unhide_context` 원문 동일성 테스트.
- `tests/test_compaction_cost_integration.py`: Ollama 호출의 metered budget gate와 비용 원장 테스트.
- `.env.example`: 선택 기능의 기본 설정 예시. 이 저장소는 이미 `.env.example`을 환경 변수 안내 파일로 사용한다.
- `specs/context-compaction/design.md`, `specs/context-compaction/implementation-handoff.md`: Laya와 다른 생성 요약 계약, 실제 진입점, 제한 사항을 기록.

## 설정과 기본값

| 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `CONTEXT_COMPACTION_ENABLED` | `false` | 전체 내부 hide/list/unhide 기능의 스위치 |
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

실제 합성 live test는 로컬 Ollama `http://127.0.0.1:11434`의 `lfm2.5-thinking:latest`를 사용했다. `tests/test_lfm_live_integration.py`는 `LFM_INTEGRATION_BASE_URL`, `LFM_INTEGRATION_MODEL`, `LFM_INTEGRATION_TIMEOUT_SECONDS` 환경변수로 테스트 대상을 선택하며 기본 동작 모드는 `ollama-native`다. 이 로컬 주소를 운영 설정에 복사하지 않는다. 운영에서 `OLLAMA_BASE_URL`이 bridge pod에서 도달 가능한지는 Atlas가 별도로 확인해야 한다.

`COST_TRACKING_ENABLED=true`인 운영 환경에서는 기존 비용 정책대로 Ollama의 `COST_PROVIDER_BILLING_JSON` 분류와 필요한 경우 실제 계약에 맞는 `COST_PRICING_JSON` 가격을 확인해야 한다. 가격을 추정해 입력하지 않는다. 비용 gate가 LFM 시도를 차단하면 LFM이 성공한 것으로 표시되지 않고 안전한 기존 규칙 경로로 돌아간다.

## 테스트 및 검증 결과

모든 자동 테스트는 `<bridge-repository>/.worktrees/lfm-context-summary`에서 실행했다. PostgreSQL을 사용하는 테스트 fixture는 외부 DSN을 사용하지 않고 `tmp_path` 아래에 임시 native PostgreSQL을 만든다. 운영 PostgreSQL은 사용하지 않았다.

- 최종 집중 회귀: `uv run pytest -q tests/test_context_compaction.py tests/test_laya_compaction.py tests/test_lfm_summary.py tests/test_compaction_cost_integration.py tests/test_lfm_live_integration.py tests/test_foundry_chat.py tests/test_foundry_tool_calling.py tests/test_async_cost.py tests/test_cost_postgres_api.py tests/test_stream_usage.py tests/test_ollama_chat.py` — **203 passed, 0 failed, 1 skipped**. 기본 opt-in live test는 이 묶음에서 skip했고, 아래에서 실제 호출을 별도로 통과시켰다.
- PostgreSQL fixture 수명주기: `uv run pytest -q tests/test_async_cost.py` — **16 passed**. 테스트 전후 SysV 공유 메모리 세그먼트는 3개로 같았고, 실행 중 새로 생겼다가 남은 임시 fixture PostgreSQL 프로세스는 없었다. fixture 종료 코드는 `terminate` 후 `wait`, 제한 시간 초과 시 `kill` 후 `wait`를 수행한다. 운영 PostgreSQL이나 다른 프로필 DB를 사용·수정하지 않았다.
- 전체 suite: `uv run pytest -q` — **818 passed, 0 failed, 1 skipped**. pytest-xdist가 설치되어 있지 않아 단일 pytest 프로세스로 순차 실행했다. Skip은 opt-in 실제 Ollama live test다.
- 실제 LFM live: `RUN_LFM_LIVE_INTEGRATION=1 LFM_INTEGRATION_BASE_URL=http://127.0.0.1:11434 LFM_INTEGRATION_MODEL=lfm2.5-thinking:latest LFM_INTEGRATION_TIMEOUT_SECONDS=180 uv run pytest -s tests/test_lfm_live_integration.py -q` — **1 passed**. 합성 데이터로 실제 LFM 호출이 1회 발생하고 `compaction_source=lfm`, `lfm_applied=true`였음을 검증했다. 다음 upstream 요청에 생성 요약과 필수 원문 증거가 반영되고, `list_context_items`가 숨김 항목 ID를 돌려주며, `unhide_context`가 원문을 byte-for-byte 동일하게 복원했다. 테스트는 주제 관련성, 증거 3줄 보존, 축약 크기 20% 초과 감소, assistant tool call ID 연결 및 내부 도구 호출 비노출도 확인한다. 안전 fallback만 발생하면 테스트가 실패한다.
- `uv run python -m compileall -q openai_compatible_bridge tests`와 `git diff --check` — 통과.
- 테스트에서 확인된 비차단 경고: Starlette의 `TestClient`가 현재 `httpx` 조합에 대해 deprecation warning을 출력했다. streaming 경로는 LFM hide 처리를 하지 않고 기존 SSE 동작을 유지한다. 비용 회귀는 metered Ollama HTTP 호출의 비용 원장 기록과 예산 차단 시 fallback을 검사한다.

실제 live test의 성공 기준은 HTTP 200만이 아니다. 저장된 항목이 LFM 출처여야 하고, 구체적인 synthetic 주제를 포함하며, 압축·목록 조회·`unhide_context` 원문 동일성이 모두 통과해야 한다. 2026-10-01에 로컬 Ollama endpoint에서 이 기준을 통과했다. bridge API 모드 및 운영 Foundry consumer canary는 검증하지 않았다.

## Atlas 운영 적용 순서

1. 먼저 이 로컬 커밋의 코드/테스트를 Atlas 승인된 PR·릴리스 경로로 검토한다. 이 인계만으로 push, merge, 배포, 운영 활성화 권한이 생기지 않는다.
2. 배포 전 bridge pod에서 Ollama 주소와 `lfm2.5-thinking:latest` 사용 가능 여부를 확인한다. Secret은 이 기능에 새로 필요하지 않으며 코드/로그에 넣지 않는다.
3. replica 수와 라우팅을 확인한다. 저장소는 pod별 프로세스 메모리이며 공유 DB가 아니다. 여러 replica를 사용하면 같은 상관 헤더 요청이 같은 pod에 도달하도록 sticky routing이 필요하다. 이를 보장할 수 없으면 기능을 운영 활성화하지 않는다.
4. 비용 추적이 켜져 있으면 Ollama payer 분류와 실제 가격표를 확인한다. `metered`이면 기존 예산 gate가 요청을 허용하는지 사전 확인한다.
5. 우선 코드를 배포하더라도 두 기능 스위치를 `false`로 둔다. canary 절차와 복구 담당자를 확인한 뒤, 제한된 canary에서만 `CONTEXT_COMPACTION_ENABLED=true` 및 `CONTEXT_COMPACTION_LFM_ENABLED=true`를 적용한다. 전체 스위치는 기존 hide/list/unhide 기능도 켜므로 LFM만 켜는 설정은 아니다.
6. 실제 소비자 canary는 승인된 Foundry OpenAI alias와 안정적인 `x-hermes-conversation` 헤더로 수행한다. 대화와 tool output은 synthetic fixture만 사용하고, 짧지 않은 과거 `role=tool` 결과 및 정상적인 `assistant.tool_calls`/결과 ID 짝을 보낸다. 주 모델이 `hide_context`에 필수 `tool_call_id` 하나를 전달하도록 하고, 다음 upstream 요청에서 해당 결과 본문만 바뀌는지 확인한다. `list_context_items`에서 `item_id`를 읽어 `unhide_context(item_id)`를 호출하고 원문 sentinel이 정확히 돌아오는지 확인한다. `hide_context`가 모델 응답이나 client 실행 도구로 노출되지 않는지 검증하며, `tool_choice`를 외부 도구로 강제하지 않는다.
7. bridge 로그에서 `context_compaction ... lfm_calls=1 lfm_applied=True lfm_fallback=False`를 확인한다. 이 로그에는 입력 본문이 포함되지 않는다. 성공 판정은 반드시 `lfm_applied=True`와 `lfm_fallback=False`로 한다. `lfm_calls`는 요약 함수 진입 횟수이며 비용 gate가 outbound HTTP 전에 막아도 증가할 수 있다. `calls`는 주 모델 호출 횟수다.
8. hide/list/unhide가 브리지 내부에서 소비되어 client에게 function call로 반환되지 않는지 확인한다. 숨김 시 upstream 전송 사본에서 지정한 tool result content만 바뀌고, assistant call ID와 `tool_call_id` 연결은 유지되어야 한다. byte-for-byte 복원은 합성 live integration test가 통과한 뒤에만 확인된 것으로 기록한다.
9. canary가 성공해도 범위를 확대하기 전까지 stream·다른 Foundry 프로토콜·다른 provider는 미지원으로 유지한다. 이 기능은 tool output의 비밀 정보를 지우지 않는다. 민감 자료가 들어간 출력을 Ollama에 보내도 되는지는 운영자가 별도로 확인해야 한다.

## 롤백

- LFM만 중지하고 기존 규칙 기반 compaction을 유지하려면 `CONTEXT_COMPACTION_LFM_ENABLED=false`로 바꾼다.
- 전체 `hide_context`/`list_context_items`/`unhide_context` 내부 도구를 끄려면 `CONTEXT_COMPACTION_ENABLED=false`로 바꾼다.
- 코드 문제가 있으면 Atlas 승인 절차에 따라 이전 bridge 이미지/릴리스로 되돌린다. 새 DB schema나 migration은 없다.
- 프로세스 재시작 또는 다른 pod로의 이동 뒤에는 해당 프로세스의 `unhide_context` 항목이 없을 수 있다. 저장소를 영속 원문 보관소로 간주하지 않는다. rollback 전후 모두 원본 대화/tool 결과를 consumer가 보존하는지 확인한다.

## 남은 제한과 위험

- `x-hermes-conversation`은 raw 상관 키이지 인증된 사용자나 분기 ID가 아니다. 같은 값을 공유하는 분기/cron/위임 작업은 같은 프로세스 상태를 사용할 수 있다.
- 원문과 축약 상태는 pod별 메모리·TTL에만 저장된다. 재시작/replica 이동 후 목록 및 `unhide_context` 복구를 보장하지 않는다.
- 생성 요약의 사실 정확성을 완전히 증명할 수 없다. 검증은 일반 문구·비신뢰 지시·원문에 없는 숫자/경로/ID 등을 제한하며, 필수 원문 증거를 별도 보존한다. 이것은 secret scrubber나 접근 제어가 아니다.
- `CONTEXT_COMPACTION_ENABLED`를 켜면 LFM 외 기존 compaction 도구도 활성화된다. 운영에서는 먼저 canary 범위를 제한한다.
- 로컬 source 검색에서는 기존 `compact_context`를 부르는 해당 bridge 외부 consumer 코드를 찾지 못했다. Neurons 쪽 `compact_context_pack`은 무관한 개념이다. 그러나 외부 사용자 설정 전체가 조사된 것은 아니므로 소비자 부재를 단정하지 않는다. 합의 계약대로 bridge는 `hide_context`만 선언/소비한다. client가 `compact_context`를 자신의 public tool로 명시 제공하면 이름 충돌 없이 외부 client tool로 전달한다. 배포 전 Atlas는 배포된 client tool registry나 승인된 사용 기록에 기존 tool name 의존이 있는지 확인해야 한다. 발견되면 별도 계약 결정 전 alias를 임의로 추가하지 않는다.
- 운영 배포/활성화, pod-to-Ollama 연결 재확인, 운영 consumer canary는 수행하지 않았다. Atlas가 담당한다.
