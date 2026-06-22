FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir uv

# 의존성 먼저 설치 (레이어 캐시)
COPY pyproject.toml uv.lock /app/
RUN uv sync --no-dev

COPY openai_compatible_bridge /app/openai_compatible_bridge

EXPOSE 80

CMD ["uv", "run", "uvicorn", "openai_compatible_bridge.main:app", "--host", "0.0.0.0", "--port", "80"]
