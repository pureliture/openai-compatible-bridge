# context-hide

호스트 중립적인 도구 결과 요약/압축(Context Compaction) 공통 엔진.

## 핵심 특징
- **호스트 중립성**: OpenAI, Foundry, Starlette, FastAPI, 환경변수(`os.environ`)에 일절 의존하지 않는 순수 Python 라이브러리.
- **엄격한 증거성**: 소스 결과에 없는 환각(종료 코드, 수치, 파일 경로) 엄격 차단 및 권위 있는 메타데이터 검증.
- **가역적 원문 복원 (Unhide)**: 재실행이나 모델 재호출 없이 메모리 저장소에서 원본 100% 즉시 복원.
- **안전한 정책 보호 (Policy Protection)**: 미해결 오류, 비즈니스 핵심 상태, 보안 민감값 자동 감지 및 fail-open 보호.
- **최소 의존성**: 오직 `cachetools` 하나에만 의존하며 표준 라이브러리 기반으로 구동.

## 설치
```bash
uv pip install context-hide
```

## 빠른 시작
```python
import hashlib
from context_hide import ContextHideEngine, Scope, ToolResultRecord

engine = ContextHideEngine()
scope = Scope(adapter_id="bridge", host_profile="default", session_id="session-1")

record = ToolResultRecord(
    call_id="call-123",
    result_position=0,
    content="...대용량 도구 실행 결과...",
    content_sha256=hashlib.sha256(b"...").hexdigest(),
    invocation={"tool_name": "terminal", "arguments": {"command": "pytest"}},
    invocation_digest="digest-abc",
    host_handle={"message_index": 3},
)

# 요약/압축 실행
result = engine.hide_sync(scope, record)
if result.ok and result.plan:
    print("치환 텍스트:", result.plan.replacement_text)
    print("호스트 핸들:", result.plan.host_handle)

# 원문 복원 (Unhide)
engine.unhide(scope, result.plan.item_id)
```

## 개발 및 테스트
```bash
uv sync --dev
uv run pytest
uv build
```
