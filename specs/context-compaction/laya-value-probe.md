# Laya의 tool 결과 축약 효용: 새 합성 사례 평가

## 판단 범위

현재 코드는 command/도구 호출 본문을 축약하지 않고 **이미 읽은 tool 결과 본문**만 축약한다. M2의 Laya는 규칙 발췌에 원문 줄을 추가하고, M3의 Laya는 숨겨진 결과 목록을 정렬한다. 안전 보호 대상인 배송·결제·점검 등의 결과는 원문을 유지하므로 효용 시험에서 제외했다.

## 실행과 결과 구분

```sh
# fix/compaction-cleanup 작업 트리의 저장소 루트에서 실행
.venv/bin/python -m pytest -q tests/test_laya_value_evaluation.py tests/test_laya_http.py tests/test_laya_compaction.py --tb=short
.venv/bin/python -m scripts.evaluate_laya_value
# Atlas의 보관 정책 확인 후, 허용된 개발 측 서버 origin을 환경에서만 제공할 때:
LAYA_BASE_URL='<approved-origin>' .venv/bin/python -m scripts.evaluate_laya_value
```

스크립트는 합성 입력만 사용하고 결과에는 사례명·누락 수·길이·호출 수·시간·순위 판정만 출력한다. 서버 주소나 원문 응답을 출력하지 않는다. **현재 세션에 `LAYA_BASE_URL`이 없어 실제 Laya 모델은 호출하지 못했다.** 따라서 아래 대조군은 모델 정확도나 원격 지연이 아니다. 실제 서버 평가를 실행할 때도 요청 본문은 서버로 전송되므로 보관 정책 확인이 선행되어야 한다. 현재 스크립트는 한 번 실행하는 제한된 사례 측정이며 반복 실행·통계적 정확도나 운영 p95 검증이 아니다.

로컬 실행: 새 평가 테스트 **4 passed**(exit 0), Laya 관련 집중 테스트 **41 passed**(exit 0), 전체 테스트 **750 passed, 기존 Starlette 경고 1건**(exit 0, 74.28초). `uvx --offline ruff check scripts/evaluate_laya_value.py tests/test_laya_value_evaluation.py`, `uv lock --check`, `git diff --check`는 exit 0. 스크립트는 기본값에서 `controls_only: true`를 표시한다.

| 구분 | 규칙 | 항상 skip 대조군 | 정답 지정 대조군 |
| --- | ---: | ---: | ---: |
| M2: 축약 가능한 기술 결과 4건 중 필수 중간 줄 누락 | 4/4 | 4/4 | 0/4 |
| M2: 추가 선택 호출 | 0 | 8(모의) | 8(모의) |
| M3: 규칙이 틀리는 3건 중 정답 1위 | 0/3 | 0/3 | 3/3 |
| M3: 선택 호출 | 0 | 3(모의) | 3(모의) |

M2의 추가 줄을 남기는 경로는 축약본 길이를 늘린다. 사례별 원문/규칙/후보 길이 및 로컬 경과 시간은 스크립트 JSON으로 다시 측정한다. M3의 세 사례 모두 전체 항목을 유지하고 자동 unhide하지 않는다. 이것은 평가기의 긍정·부정 대조가 동작한다는 뜻이며, **Laya가 정답을 고른다는 뜻은 아니다.** 이전 Atlas의 원격 평가 및 위험 업무 결과 8건 보호 평가와 사례·조건이 다르므로 합쳐 점수를 내지 않는다.

## 다음 게이트

1. Atlas가 Laya 서버의 요청 본문·로그 보관 정책과 승인된 시험 접근 경로를 확인한다. 확인 전에는 도메인이 합성이어도 원격 전송을 보류한다.
2. 동일 사례에 실제 Laya를 적용하여 M2 누락·새 누락·추가 축약 길이, M3 순위 개선·오선택·무개선, 실제 호출 수·지연을 기록한다. 실패 응답과 타임아웃은 규칙 복귀로 별도 집계한다.
3. 실제 Laya가 위 기술 사례에서 규칙보다 반복적으로 낫지 않으면 Laya는 OFF로 유지한다. 개선이 있어도 위험 결과 보호, 운영 보관 게이트 및 비용/지연 판단을 별도로 통과해야 한다.

Draft PR #19는 이 평가만으로 머지·운영 활성화하지 않는다.
