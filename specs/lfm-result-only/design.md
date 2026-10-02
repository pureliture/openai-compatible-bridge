# 결과 본문만 요약하는 hide — 승인된 축소 계약

## 승인
최신 사용자가 결과만 요약하도록 확정하고, 실제 input/expected output을 맞춘 뒤 agentic-execution mode=agentic으로 즉시 구현하도록 승인했다. 이 문서는 이전 의도 기반/한국어 세 필드 계획보다 우선한다.

## 목표와 비범위
원래 assistant.tool_calls의 도구 이름·인자·명령·ID는 그대로 유지한다. 과거 role=tool 결과 본문만 짧은 내용 요약으로 교체한다. 작업 의도/다음 행동/추측 원인/명령 설명을 생성하지 않는다. 단일 JSON summary, 영어 허용. 원문 보관과 exact unhide, 보호/fallback/세션/TTL/예산/비용/1call-turn/stream skip 재사용. 긴 명령 압축·실패 보호 정책 완화·모델 변경·운영·push/merge·유료 주 모델 호출은 제외.

## 실제 사례 기대 결과 정렬
실제 세션 자료는 workspace artifacts/lfm-hide-examples/real-input-expected-output.json과 .md에 로컬 보관한다. Git 및 모델 호출에 넣지 않는다. 테스트에는 별도 합성 변형만 사용한다. 아래 결과는 실제 응답이 아닌 승인 기준 초안의 축소형이다.

- R01 파일 설정: `The project requires Python 3.11 or newer. Its runtime dependencies include FastAPI, httpx, cachetools and jsonschema; pytest is a development dependency.` 파일 읽기/설치 설명은 필요 없음. 전체 dependency version을 보존하는 기능이 아니라 개요 요약이다. 세부 정보는 원문 복원.
- R02 파일 검색: `Three AGENTS.md matches: repository root, project-context component, and discovery test fixture.` 검색된 사실·구분 보존. 검색 결과로 파일의 권위를 확정하지 않음. 해당 파일을 읽었다고 말하지 않음.
- R03 파일 없음: `AGENTS.md was not found. README.md and milestones.md were suggested as similar files.` 실패 보호 때문에 실제 hide는 원문 유지 가능. 빈 파일 읽기 성공으로 변환 금지.
- R04 terminal: `2 tests passed; exit code 0.` 원래 결과가 평가 판정을 보여주지 않으므로 평가 성공이나 chained command 실행 설명을 요약에 추가하지 않음. 원래 command는 이미 남음.
- R05 terminal 모순: `Visible output reports 12 passing tests, but exit code is 1; the failure cause is not shown.` 원문 유지 guard를 유지. 다음 행동 지시 추가 금지.

명령 성공/실패는 결과의 명시적 값으로만 확인. source row 0 숫자로 exit code 0을 만들어낼 수 없도록 검증기 수정 필수. source JSON exit_code unknown이면 생성된 숫자 exit claim 거부. 원문 숫자/경로/ID가 나타났다는 이유만으로 관계 정확성을 증명하지 않음.

## 입력/도구 계약
hide_context(tool_call_id) 기본 유지. 이미 로컬 optional context가 있으므로 공개 인자를 즉시 삭제하지 말고 호환 입력으로 검증 후 무시한다. 모델에게 새 힌트 사용을 권장하지 않는다. LFM 입력은 결과 원문과 원문 기반 필수 근거만. invocation은 연결/충돌 검증 용도로 브리지 내부에만 사용하고 모델에 보내지 않음. context metadata는 신규 요약에 저장/표시하지 않음. 오래된 메모리 항목/혼합 버전 범위는 테스트하고 영속 호환을 주장하지 않음.

## 합격 기준
1. 호출 원문/ID/사용자 대화는 불변, 본문만 요약되고 후속 요청 및 exact unhide 정상.
2. 생성 결과는 결과 본문의 핵심 관찰·성공/실패·경고를 보존하고 원문에 없는 목적·다음 행동·원인·종료 코드 없음.
3. 기존 작업 의도별 사실 Q05/Q06와 execution-field 요구는 새 목표의 합격 기준에서 제외. 기존 corpus/증거는 그대로 보존하고 새로운 result-only corpus를 버전 구분해 작성.
4. 실제 자료의 원문 전송은 이번 승인에서 추정하지 않음. 합성 파생사례로 local Ollama 3회 반복. 짧은/보호 사례는 안전 유지로 판정, 실제 생성 성공과 구분.
5. 보호·정확 발췌/호출 제한·메모리·budget/ledger/async/stream 회귀 및 전체 suite 실행. PG fixture의 현 remote-only 변경은 별도 미커밋 작업이므로 수정/커밋하지 않음. approved isolated server만 identity 검증 후 허용. DSN 없으면 PG skip을 명시, 로컬/운영 PG 대체 금지.

## 실행 단계
- A 결과-only hide→후속 적용→복원과 exit validation: TDD/focused/로컬 체크포인트.
- B 합성 결과-only actual LFM 내용 평가: 입력 정답 사전 고정, 실제 적용 요약 수동 평가/3회 반복. 불합격은 숨기지 않고 agentic 최소 입력 변경으로 수정·증명.
- C 전체 회귀/독립 리뷰/Atlas 인계/로컬 커밋. 별도 PG 미커밋 파일 제외. 운영·PR remote push 없음.
