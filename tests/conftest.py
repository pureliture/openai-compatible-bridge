"""테스트 격리용 공통 픽스처.

일부 레지스트리 테스트는 `importlib.reload(vertex)`로 환경변수를 반영한다.
reload는 Vertex provider 모듈의 클래스 객체(VertexAPIError, VertexChatClient,
VertexEmbeddingClient 등)를 새로 만들기 때문에, 이미 `from vertex import ...`로
그 심볼들을 바인딩해 둔 bridge app 모듈의 참조가 stale해진다. 그 결과 app의
`except VertexAPIError`가 reload 후 raise된 새 VertexAPIError를 잡지 못하는
정체성 불일치가 발생할 수 있다.

아래 autouse 픽스처는 각 테스트가 끝난 뒤 Vertex provider가 reload되어 app의 참조와
어긋났는지 감지하고, 어긋났다면 app 모듈이 보는 vertex 심볼들을 현재 vertex
모듈 기준으로 재바인딩해 세션 전역 상태를 일관되게 회복한다. 프로덕션 런타임은
reload를 하지 않으므로 영향이 없다.
"""

from __future__ import annotations

import pytest

import openai_compatible_bridge.main as _app
import openai_compatible_bridge.providers.vertex as _vertex

# bridge app이 Vertex provider에서 가져오는 심볼 이름들.
_REBOUND_SYMBOLS = (
    "VertexAPIError",
    "VertexChatClient",
    "VertexEmbeddingClient",
    "allowed_models",
    "model_config",
)


def _resync_app_with_vertex() -> None:
    """app 모듈의 vertex 유래 심볼을 현재 vertex 모듈 기준으로 재바인딩한다."""
    for name in _REBOUND_SYMBOLS:
        if hasattr(_vertex, name):
            setattr(_app, name, getattr(_vertex, name))


@pytest.fixture(autouse=True)
def _keep_app_in_sync_with_vertex():
    # 테스트 실행 전후로 app <-> vertex 심볼 정체성을 맞춰 reload 누수를 차단한다.
    _resync_app_with_vertex()
    yield
    _resync_app_with_vertex()
