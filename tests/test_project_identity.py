from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_pyproject_uses_bridge_name():
    pyproject = tomllib.loads(_read("pyproject.toml"))

    assert pyproject["project"]["name"] == "openai-compatible-bridge"


def test_dockerfile_uses_package_entrypoint():
    dockerfile = _read("Dockerfile")

    assert "COPY openai_compatible_bridge" in dockerfile
    assert "openai_compatible_bridge.main:app" in dockerfile
    assert "app.py" not in dockerfile
    assert "vertex.py" not in dockerfile
    assert "cost_tracking.py" not in dockerfile


def test_compose_and_env_use_bridge_identity():
    compose = _read("docker-compose.yml")
    env_example = _read(".env.example")

    assert "openai-compatible-bridge" in compose
    assert "BRIDGE_API_KEY" in compose
    assert "BRIDGE_API_KEY" in env_example
    assert "MODEL_REGISTRY_JSON" in compose
    assert "MODEL_REGISTRY_JSON" in env_example
    assert "OLLAMA_BASE_URL" in compose
    assert "OLLAMA_BASE_URL" in env_example
    assert "OLLAMA_HTTP_TIMEOUT_SECONDS" in compose
    assert "OLLAMA_HTTP_TIMEOUT_SECONDS" in env_example
    assert "OLLAMA_THINK" in compose
    assert "OLLAMA_THINK" in env_example
    assert "wrapper-vertex-ai-api" not in compose
    assert "WRAPPER_API_KEY" not in compose
    assert "WRAPPER_API_KEY" not in env_example


def test_readme_uses_bridge_identity_without_legacy_product_aliases():
    readme = _read("README.md")

    assert "openai-compatible-bridge" in readme
    assert "vertex-ai-api-wrapper" not in readme
    assert "wrapper-vertex-ai-api" not in readme
    assert "WRAPPER_API_KEY" not in readme
