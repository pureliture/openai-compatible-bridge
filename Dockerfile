FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Step 1: 공식 초경량 uv 바이너리 복사 (pip 설치 시간 5초 절감)
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

# Step 2: 의존성 정의 파일 복사 및 설치 (GHA 캐시 영속화 대상 레이어)
COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Step 3: 애플리케이션 소스 코드 복사 (509 KB)
COPY openai_compatible_bridge /app/openai_compatible_bridge

# Step 4: 메타데이터 ARG 및 LABEL (의존성 레이어 캐시 파괴 방지를 위해 하단 배치)
ARG NEURONS_SOURCE_COMMIT=""
ARG NEURONS_SOURCE_REPOSITORY=""
LABEL org.opencontainers.image.revision="${NEURONS_SOURCE_COMMIT}" \
      org.opencontainers.image.source="${NEURONS_SOURCE_REPOSITORY}"

EXPOSE 80

CMD ["uv", "run", "uvicorn", "openai_compatible_bridge.main:app", "--host", "0.0.0.0", "--port", "80"]
