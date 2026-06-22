# Milestones - openai-compatible-bridge

## M1 Spec SoT and worktree setup
- status: done
- evidence: `git status --short --branch` showed `codex/openai-compatible-bridge-spec`; `uv run pytest` passed with 185 tests and 1 StarletteDeprecationWarning.

## M2 Package shell and app factory seam
- status: done
- evidence: RED `test_bridge_package_exposes_fastapi_app` failed with `ModuleNotFoundError`; GREEN added `openai_compatible_bridge.main:app`. RED `test_create_app_accepts_provider_factories` failed with missing `create_app`; GREEN added factory seam. `uv run pytest` passed with 187 tests and 1 StarletteDeprecationWarning.

## M3 Model registry migration and provider boundary
- status: done
- evidence: RED `test_model_registry_json_accepts_ollama_chat_provider` failed with `invalid api=None`; GREEN added provider-aware registry normalization. RED `test_routes_resolve_models_from_current_registry` failed because routes used import-time `ALLOWED_MODELS`; GREEN changed routes to current registry lookup. `uv run pytest` passed with 189 tests and 1 StarletteDeprecationWarning.

## M4 Vertex adapter preservation
- status: done
- evidence: Moved root implementation files into `openai_compatible_bridge/`; updated tests/imports to package paths; root `app.py`, `vertex.py`, and `cost_tracking.py` no longer exist as implementation modules. `uv run pytest` passed with 189 tests and 1 StarletteDeprecationWarning.

## M5 Ollama chat completions adapter
- status: done
- evidence: RED `test_ollama_chat_alias_routes_to_ollama_provider` failed due missing `ollama_chat_client_factory`; GREEN added Ollama dispatch. RED adapter tests added `OllamaChatClient`; GREEN implemented non-stream and stream normalization. M5 acceptance tests cover route dispatch, streaming SSE, auth-before-dispatch, embeddings/rerank rejection, cost success, missing pricing fail-closed, and connection error mapping. `uv run pytest` passed with 199 tests and 1 StarletteDeprecationWarning.

## M6 Documentation and migration cleanup
- status: done
- evidence: RED identity tests failed on old project/Docker/docs names; GREEN updated pyproject, uv.lock, Dockerfile, docker-compose, .env.example, and README. `uv run pytest tests/test_project_identity.py` passed with 4 tests. `uv run pytest` passed with 204 tests and 1 StarletteDeprecationWarning. Import smoke printed `openai-compatible-bridge` and confirmed `/healthz` route exists.

## M7 Final verification
- status: done
- evidence: RED `test_module_app_uses_bridge_factory_lifespan` failed because module-level `app` lacked `ollama_chat_client`; GREEN made exported `app` use the same lazy `create_app()` lifecycle as test/factory apps. `uv run pytest tests/test_bridge_package.py tests/test_ollama_chat.py -q` passed with 15 tests and 1 StarletteDeprecationWarning. `uv run pytest -q` passed with 205 tests and 1 StarletteDeprecationWarning.
- final checks: import smoke printed `openai-compatible-bridge`, `True` for `/healthz`, and `True` for `/v1/chat/completions`; `git diff --check` passed with no output; `uv lock --check` resolved 43 packages; stale-name grep across README, Docker/config, pyproject, and package code returned no matches.
- review follow-up: multi-agent review found streaming error handling with cost tracking disabled, Vertex alias native-id resolution, and missing Ollama Docker/env config surface. Added RED tests for each, then fixed null-safe streaming cost handling, provider-native `provider_model` resolution with resolved config handoff, and `OLLAMA_BASE_URL` compose/env examples. `uv run pytest -q` passed with 208 tests and 1 StarletteDeprecationWarning.
