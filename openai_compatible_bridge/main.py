"""OpenAI-compatible embedding proxy for Google Cloud Vertex AI (Gemini Enterprise Agent Platform).

RAGFlow 0.25.6의 "OpenAI-API-Compatible" 임베딩 provider가 호출하는
POST /v1/embeddings 를 받아서 Vertex AI :predict 엔드포인트로 통역한다.

- 인증: 서비스 계정(ADC)에서 short-lived OAuth2 access token을 자동 발급/캐시/갱신.
- 배치 분할: RAGFlow는 요청당 16개 텍스트를 보내지만, Vertex 모델별 요청당 instance
  한도에 맞춰 쪼개 병렬 호출.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict

from openai_compatible_bridge.core.cost_tracking import (
    BudgetBlock,
    CostBudgetExceeded,
    CostConfigError,
    CostReservation,
    CostSubsystemUnhealthy,
    NormalizedUsage,
    build_cost_accounting_from_env,
)
from openai_compatible_bridge.providers.ollama import OllamaChatClient
from openai_compatible_bridge.providers.vertex import (
    SUPPORTED_RESPONSE_FORMAT_TYPES,
    VertexAPIError,
    VertexChatClient,
    VertexEmbeddingClient,
    VertexRerankClient,
    allowed_models,
    model_config,
)

SUPPORTED_TASK_TYPES = {
    "UNSPECIFIED",
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
}

VERTEX_TASK_TYPE_DEFAULT = os.getenv("VERTEX_TASK_TYPE_DEFAULT", "RETRIEVAL_DOCUMENT")
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY")
ALLOWED_MODELS = allowed_models()
OLLAMA_DYNAMIC_MODEL_PREFIX = "ollama:"
STRUCTURED_OUTPUT_REPAIR_LOGGER = logging.getLogger("structured_output_repair")
STRUCTURED_OUTPUT_REPAIR_LOGGER.setLevel(logging.INFO)
STRUCTURED_OUTPUT_REPAIR_RUNTIME_LOGGER = logging.getLogger("uvicorn.error")
STRUCTURED_OUTPUT_REPAIR_DEFAULT_MODELS = (
    "ollama:qwen3.5:cloud",
    "ollama:gemma4:31b-cloud",
    "ollama:glm-5.2:cloud",
)


def current_allowed_models() -> set[str]:
    return allowed_models()


def _resolve_chat_model(model: str) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    if model.startswith(OLLAMA_DYNAMIC_MODEL_PREFIX):
        provider_model = model[len(OLLAMA_DYNAMIC_MODEL_PREFIX):].strip()
        if not provider_model:
            return None, openai_error_response(
                message="Dynamic Ollama model must use the form 'ollama:<native-model>'.",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_model",
                param="model",
            )
        return {
            "provider": "ollama",
            "kind": "chat",
            "provider_model": provider_model,
        }, None

    if model not in current_allowed_models():
        return None, openai_error_response(
            message=f"The model '{model}' does not exist.",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    cfg = model_config(model)
    if cfg and cfg.get("kind") != "chat":
        return None, openai_error_response(
            message=f"The model '{model}' is not a chat model.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_model",
            param="model",
        )
    return cfg, None


class OpenAIEmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: str | list[str]
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


class OpenAIChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any]


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[OpenAIChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    user: str | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    reasoning: dict[str, Any] | None = None


class CohereRerankRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    query: str
    documents: list[str | dict[str, Any]]
    top_n: int | None = None


def openai_error_response(
    *,
    message: str,
    status_code: int,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": param, "code": code}},
    )


def budget_exceeded_response(block: BudgetBlock) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "Cost budget exceeded.",
                "type": "rate_limit_error",
                "param": None,
                "code": "budget_exceeded",
                "limit_type": block.limit_type,
                "reset_at": block.reset_at,
                "current_estimated_spend": str(block.current_estimated_spend),
                "configured_limit": str(block.configured_limit),
                "currency": block.currency,
            }
        },
    )


def map_vertex_status_to_openai_type(status_code: int) -> str:
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "invalid_request_error",
        429: "rate_limit_error",
    }.get(status_code, "api_error" if status_code >= 500 else "invalid_request_error")


def coerce_string_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise ValueError("This wrapper supports only string or array[string] inputs.")


def encode_embedding(values: list[float], fmt: str) -> list[float] | str:
    if fmt == "base64":
        import base64
        import struct
        return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")
    return values


def _estimate_text_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 0 if value == "" else max(1, (len(value) + 3) // 4)
    if isinstance(value, list):
        return sum(_estimate_text_tokens(item) for item in value)
    if isinstance(value, dict):
        total = 0
        for key in ("text", "content"):
            item = value.get(key)
            if isinstance(item, (str, list, dict)):
                total += _estimate_text_tokens(item)
        return total
    return 0


def _default_chat_completion_tokens() -> int:
    try:
        return max(1, int(os.getenv("COST_CHAT_DEFAULT_MAX_OUTPUT_TOKENS", "4096")))
    except ValueError:
        return 4096


def _chat_forecast_usage(payload: OpenAIChatRequest) -> NormalizedUsage:
    prompt_tokens = sum(_estimate_text_tokens(message.content) for message in payload.messages)
    completion_tokens = (
        payload.max_tokens if payload.max_tokens is not None and payload.max_tokens > 0
        else _default_chat_completion_tokens()
    )
    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _embedding_forecast_usage(texts: list[str]) -> NormalizedUsage:
    embedding_tokens = sum(_estimate_text_tokens(text) for text in texts)
    return NormalizedUsage(embedding_tokens=embedding_tokens, total_tokens=embedding_tokens)


def _chat_usage_from_mapping(usage: MappingLike) -> NormalizedUsage:
    prompt_tokens = _safe_int(usage.get("prompt_tokens"))
    completion_tokens = _safe_int(usage.get("completion_tokens"))
    total_tokens = _safe_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _safe_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


MappingLike = dict[str, Any]


def _cost_accounting(request: Request) -> Any:
    return getattr(request.app.state, "cost_accounting", None)


def _preflight_cost(
    request: Request,
    *,
    endpoint: str,
    model: str,
    forecast_usage: NormalizedUsage,
) -> CostReservation | None | JSONResponse:
    accounting = _cost_accounting(request)
    if accounting is None:
        return None
    try:
        return accounting.preflight(endpoint=endpoint, model=model, forecast_usage=forecast_usage)
    except CostBudgetExceeded as exc:
        return budget_exceeded_response(exc.block)
    except CostConfigError as exc:
        return openai_error_response(
            message=str(exc),
            status_code=503,
            error_type="api_error",
            code="cost_config_error",
        )
    except CostSubsystemUnhealthy as exc:
        return openai_error_response(
            message=str(exc),
            status_code=503,
            error_type="api_error",
            code="cost_tracking_unavailable",
        )


def _finalize_cost_success(request: Request, reservation: CostReservation | None, usage: NormalizedUsage) -> None:
    accounting = _cost_accounting(request)
    if accounting is not None:
        accounting.finalize_success(reservation, usage)


def _release_cost_nonbillable(request: Request, reservation: CostReservation | None, reason: str) -> None:
    accounting = _cost_accounting(request)
    if accounting is not None:
        accounting.release_nonbillable(reservation, reason)


def _finalize_cost_estimated_only(request: Request, reservation: CostReservation | None, reason: str) -> None:
    accounting = _cost_accounting(request)
    if accounting is not None:
        accounting.finalize_estimated_only(reservation, reason)


def _openai_error_json_response_to_vertex_error(response: JSONResponse) -> VertexAPIError:
    try:
        body = json.loads(response.body.decode("utf-8"))
    except Exception:
        body = {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    return VertexAPIError(
        response.status_code,
        str(error.get("message") or "Request failed."),
        code=str(error.get("code") or response.status_code),
        raw={"cost_managed": True},
    )


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_json_schema_response_format(response_format: dict[str, Any] | None) -> bool:
    return isinstance(response_format, dict) and response_format.get("type") == "json_schema"


def _structured_output_repair_enabled() -> bool:
    return _env_flag("STRUCTURED_OUTPUT_REPAIR_ENABLED")


def _structured_output_repair_model_ids() -> tuple[str, ...]:
    raw = os.getenv("STRUCTURED_OUTPUT_REPAIR_MODELS")
    if raw:
        values = tuple(item.strip() for item in raw.split(",") if item.strip())
    else:
        values = STRUCTURED_OUTPUT_REPAIR_DEFAULT_MODELS
    return values[:3]


def _dynamic_ollama_native_model(model_id: str) -> str | None:
    if not model_id.startswith(OLLAMA_DYNAMIC_MODEL_PREFIX):
        return None
    native = model_id[len(OLLAMA_DYNAMIC_MODEL_PREFIX):].strip()
    return native or None


def _is_ollama_cloud_native_model(native_model: str) -> bool:
    return native_model.endswith(":cloud") or native_model.endswith("-cloud")


def _structured_output_repair_applies(
    *,
    payload: "OpenAIChatRequest",
    provider: str,
    provider_model: str,
) -> bool:
    return (
        _structured_output_repair_enabled()
        and provider == "ollama"
        and payload.model.startswith(OLLAMA_DYNAMIC_MODEL_PREFIX)
        and _is_ollama_cloud_native_model(provider_model)
        and _is_json_schema_response_format(payload.response_format)
    )


def _structured_output_repair_native_models() -> tuple[str, ...]:
    models: list[str] = []
    invalid: list[str] = []
    for model_id in _structured_output_repair_model_ids():
        native = _dynamic_ollama_native_model(model_id)
        if native is not None and _is_ollama_cloud_native_model(native):
            models.append(native)
        else:
            invalid.append(model_id)
    if invalid or not models:
        raise VertexAPIError(
            503,
            "STRUCTURED_OUTPUT_REPAIR_MODELS must contain dynamic Ollama Cloud model ids.",
            code="cost_config_error",
            raw={"cost_managed": True},
        )
    return tuple(models)


def _build_structured_output_repair_messages(
    *,
    messages: list[dict[str, Any]],
    response_format: dict[str, Any] | None,
    failure_code: str | None,
) -> list[dict[str, Any]]:
    json_schema_obj = response_format.get("json_schema") if isinstance(response_format, dict) else None
    schema = json_schema_obj.get("schema") if isinstance(json_schema_obj, dict) else {}
    schema_json = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    instruction = (
        "Return only JSON that validates against the provided JSON Schema. "
        "Do not include markdown, code fences, commentary, or reasoning text. "
        f"Previous failure category: {failure_code or 'invalid_schema_output'}. "
        f"JSON Schema: {schema_json}"
    )
    return [{"role": "system", "content": instruction}, *messages]


async def _generate_ollama_chat_once(
    chat_client: VertexChatClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
    payload: "OpenAIChatRequest",
    repair_attempt: bool = False,
) -> dict[str, Any]:
    temperature = payload.temperature
    if repair_attempt and temperature is None:
        temperature = 0
    return await chat_client.generate(
        model=model,
        messages=messages,
        max_tokens=payload.max_tokens,
        temperature=temperature,
        top_p=payload.top_p,
        stop=payload.stop,
        response_format=payload.response_format,
        reasoning_effort=payload.reasoning_effort,
        reasoning=payload.reasoning,
    )


def _structured_output_latency_bucket(elapsed_seconds: float) -> str:
    if elapsed_seconds < 1:
        return "lt_1s"
    if elapsed_seconds < 5:
        return "lt_5s"
    if elapsed_seconds < 15:
        return "lt_15s"
    if elapsed_seconds < 60:
        return "lt_60s"
    return "gte_60s"


def _log_structured_output_repair_event(
    *,
    attempted_models: list[str],
    final_status: str,
    failure_category: str | None,
    content_chars: int,
    started_at: float,
) -> None:
    event = {
        "event": "structured_output_repair",
        "enabled": True,
        "attempt_count": max(0, len(attempted_models) - 1),
        "attempted_models": attempted_models,
        "final_status": final_status,
        "failure_category": failure_category or "invalid_schema_output",
        "latency_bucket": _structured_output_latency_bucket(time.monotonic() - started_at),
        "content_chars": max(0, content_chars),
    }
    STRUCTURED_OUTPUT_REPAIR_LOGGER.info(
        "structured_output_repair %s",
        json.dumps(event, ensure_ascii=False, sort_keys=True),
    )
    STRUCTURED_OUTPUT_REPAIR_RUNTIME_LOGGER.info(
        "structured_output_repair %s",
        json.dumps(event, ensure_ascii=False, sort_keys=True),
    )


def _usage_from_error(exc: VertexAPIError) -> NormalizedUsage:
    raw = exc.raw if isinstance(exc.raw, dict) else {}
    usage = raw.get("usage") if isinstance(raw, dict) else None
    return _chat_usage_from_mapping(usage if isinstance(usage, dict) else {})


def _add_usage(left: NormalizedUsage, right: NormalizedUsage) -> NormalizedUsage:
    prompt_tokens = left.prompt_tokens + right.prompt_tokens
    completion_tokens = left.completion_tokens + right.completion_tokens
    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _usage_to_mapping(usage: NormalizedUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _finalize_attempt_cost(
    request: Request | None,
    reservation: CostReservation | None,
    usage: NormalizedUsage,
    *,
    estimated_reason: str,
) -> None:
    if request is None:
        return
    if usage.total_tokens or usage.prompt_tokens or usage.completion_tokens:
        _finalize_cost_success(request, reservation, usage)
    else:
        _finalize_cost_estimated_only(request, reservation, estimated_reason)


def _preflight_repair_cost_or_raise(
    request: Request | None,
    *,
    model: str,
    payload: "OpenAIChatRequest",
) -> CostReservation | None:
    if request is None:
        return None
    cost_preflight = _preflight_cost(
        request,
        endpoint="chat",
        model=model,
        forecast_usage=_chat_forecast_usage(payload),
    )
    if isinstance(cost_preflight, JSONResponse):
        raise _openai_error_json_response_to_vertex_error(cost_preflight)
    return cost_preflight


async def _generate_ollama_with_structured_output_repair(
    chat_client: VertexChatClient,
    *,
    payload: "OpenAIChatRequest",
    messages: list[dict[str, Any]],
    provider_model: str,
    request: Request | None = None,
    reservation: CostReservation | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    attempted_models = [payload.model]
    aggregate_usage = NormalizedUsage()
    try:
        return await _generate_ollama_chat_once(
            chat_client,
            model=provider_model,
            messages=messages,
            payload=payload,
        )
    except VertexAPIError as exc:
        if not (
            exc.code == "invalid_schema_output"
            and _structured_output_repair_applies(
                payload=payload,
                provider="ollama",
                provider_model=provider_model,
            )
        ):
            raise
        last_failure_code = exc.code
        initial_usage = _usage_from_error(exc)
        aggregate_usage = _add_usage(aggregate_usage, initial_usage)
        _finalize_attempt_cost(
            request,
            reservation,
            initial_usage,
            estimated_reason="structured_output_repair_initial_schema_failure",
        )

    repair_messages = _build_structured_output_repair_messages(
        messages=messages,
        response_format=payload.response_format,
        failure_code=last_failure_code,
    )
    for repair_model in _structured_output_repair_native_models():
        repair_model_id = f"{OLLAMA_DYNAMIC_MODEL_PREFIX}{repair_model}"
        attempted_models.append(repair_model_id)
        repair_reservation = _preflight_repair_cost_or_raise(
            request,
            model=repair_model_id,
            payload=payload,
        )
        try:
            result = await _generate_ollama_chat_once(
                chat_client,
                model=repair_model,
                messages=repair_messages,
                payload=payload,
                repair_attempt=True,
            )
            result_usage = _chat_usage_from_mapping(result.get("usage", {}))
            aggregate_usage = _add_usage(aggregate_usage, result_usage)
            _finalize_attempt_cost(
                request,
                repair_reservation,
                result_usage,
                estimated_reason="structured_output_repair_success_missing_usage",
            )
            result["usage"] = _usage_to_mapping(aggregate_usage)
            result["_cost_managed"] = True
            _log_structured_output_repair_event(
                attempted_models=attempted_models,
                final_status="success",
                failure_category=last_failure_code,
                content_chars=len(str(result.get("text") or "")),
                started_at=started_at,
            )
            return result
        except VertexAPIError as exc:
            if exc.code != "invalid_schema_output":
                _release_cost_nonbillable(request, repair_reservation, "upstream_error")
                raise VertexAPIError(
                    exc.status_code,
                    "Ollama structured output repair attempt failed.",
                    code="structured_output_repair_error",
                    raw={"cost_managed": True},
                ) from exc
            last_failure_code = exc.code
            attempt_usage = _usage_from_error(exc)
            aggregate_usage = _add_usage(aggregate_usage, attempt_usage)
            _finalize_attempt_cost(
                request,
                repair_reservation,
                attempt_usage,
                estimated_reason="structured_output_repair_schema_failure",
            )

    _log_structured_output_repair_event(
        attempted_models=attempted_models,
        final_status="failure",
        failure_category=last_failure_code,
        content_chars=0,
        started_at=started_at,
    )
    raise VertexAPIError(
        502,
        "Ollama structured output repair failed after configured attempts.",
        code="invalid_schema_output",
        raw={"cost_managed": True},
    )


def _authorize_cost_admin(request: Request, authorization: str | None) -> JSONResponse | None:
    accounting = _cost_accounting(request)
    config = getattr(accounting, "config", None)
    if accounting is None or not getattr(accounting, "enabled", False) or config is None or not config.admin_enabled:
        return openai_error_response(
            message="Cost admin API is disabled.",
            status_code=404,
            error_type="invalid_request_error",
            code="cost_admin_disabled",
        )
    if not config.admin_api_key:
        return openai_error_response(
            message="Cost admin API is disabled.",
            status_code=404,
            error_type="invalid_request_error",
            code="cost_admin_disabled",
        )
    if authorization is None:
        return openai_error_response(
            message="Missing cost admin API key.",
            status_code=401,
            error_type="authentication_error",
            code="missing_admin_api_key",
        )
    if authorization != f"Bearer {config.admin_api_key}":
        return openai_error_response(
            message="Invalid cost admin API key.",
            status_code=403,
            error_type="permission_error",
            code="invalid_admin_api_key",
        )
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vertex_client = VertexEmbeddingClient()
    app.state.vertex_chat_client = VertexChatClient()
    app.state.vertex_rerank_client = VertexRerankClient()
    app.state.cost_accounting = build_cost_accounting_from_env(os.environ)
    try:
        yield
    finally:
        cost_accounting = getattr(app.state, "cost_accounting", None)
        if cost_accounting is not None:
            cost_accounting.close()
        await app.state.vertex_client.close()
        await app.state.vertex_chat_client.close()
        await app.state.vertex_rerank_client.close()


app = FastAPI(title="openai-compatible-bridge", version="0.1.0", lifespan=lifespan)
app.router.redirect_slashes = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = exc.errors()
    msg = detail[0].get("msg", "Invalid request") if detail else "Invalid request"
    loc = detail[0].get("loc") if detail else None
    param = ".".join(str(p) for p in loc[1:]) if loc and len(loc) > 1 else None
    return openai_error_response(
        message=str(msg),
        status_code=400,
        error_type="invalid_request_error",
        code="invalid_request",
        param=param,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/admin/cost/status", response_model=None)
async def admin_cost_status(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | dict[str, Any]:
    auth_error = _authorize_cost_admin(request, authorization)
    if auth_error is not None:
        return auth_error
    return _cost_accounting(request).admin_status()


@app.get("/admin/cost/events", response_model=None)
async def admin_cost_events(
    request: Request,
    authorization: str | None = Header(default=None),
    limit: int = 100,
) -> JSONResponse | dict[str, Any]:
    auth_error = _authorize_cost_admin(request, authorization)
    if auth_error is not None:
        return auth_error
    return {"data": _cost_accounting(request).admin_events(limit=limit)}


@app.get("/admin/cost/reconciliation", response_model=None)
async def admin_cost_reconciliation(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | dict[str, Any]:
    auth_error = _authorize_cost_admin(request, authorization)
    if auth_error is not None:
        return auth_error
    return _cost_accounting(request).admin_reconciliation()


def _model_object(model_id: str) -> dict[str, Any]:
    return {"id": model_id, "object": "model", "created": 0, "owned_by": "google"}


@app.get("/v1/models", response_model=None)
async def list_models(authorization: str | None = Header(default=None)) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    return {"object": "list", "data": [_model_object(m) for m in sorted(current_allowed_models())]}


@app.get("/v1/models/{model_id}", response_model=None)
async def retrieve_model(
    model_id: str, authorization: str | None = Header(default=None)
) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    if model_id not in current_allowed_models():
        return openai_error_response(
            message=f"The model '{model_id}' does not exist.",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )
    return _model_object(model_id)


@app.post("/v1/embeddings", response_model=None)
async def create_embeddings(
    payload: OpenAIEmbeddingsRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_vertex_task_type: str | None = Header(default=None, alias="X-Vertex-Task-Type"),
    x_vertex_title: str | None = Header(default=None, alias="X-Vertex-Title"),
) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    allowed = current_allowed_models()
    if payload.model not in allowed:
        return openai_error_response(
            message=f"The model '{payload.model}' does not exist. Allowed: {sorted(allowed)}",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    _embed_cfg = model_config(payload.model)
    if _embed_cfg and _embed_cfg.get("kind") != "embedding":
        return openai_error_response(
            message=f"The model '{payload.model}' is not an embedding model.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_model",
            param="model",
        )

    if payload.dimensions is not None and payload.dimensions < 1:
        return openai_error_response(
            message="dimensions must be >= 1.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_dimensions",
            param="dimensions",
        )

    try:
        texts = coerce_string_inputs(payload.input)
    except ValueError as exc:
        return openai_error_response(
            message=str(exc),
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_input_shape",
            param="input",
        )
    if not texts:
        return openai_error_response(
            message="input must not be empty.",
            status_code=400,
            error_type="invalid_request_error",
            code="empty_input",
            param="input",
        )

    task_type = x_vertex_task_type or VERTEX_TASK_TYPE_DEFAULT
    if task_type not in SUPPORTED_TASK_TYPES:
        return openai_error_response(
            message=f"Unsupported task type: {task_type}",
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_task_type",
            param="X-Vertex-Task-Type",
        )

    cost_preflight = _preflight_cost(
        request,
        endpoint="embeddings",
        model=payload.model,
        forecast_usage=_embedding_forecast_usage(texts),
    )
    if isinstance(cost_preflight, JSONResponse):
        return cost_preflight
    reservation = cost_preflight

    vertex_client: VertexEmbeddingClient = request.app.state.vertex_client
    provider_model = (_embed_cfg or {}).get("provider_model", payload.model)

    try:
        chunk_results = await vertex_client.embed(
            model=provider_model,
            texts=texts,
            dimensions=payload.dimensions,
            task_type=task_type,
            title=x_vertex_title,
            resolved_config=_embed_cfg,
        )
    except VertexAPIError as exc:
        _release_cost_nonbillable(request, reservation, "upstream_error")
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )

    data: list[dict[str, Any]] = []
    total_tokens = 0
    for index, item in enumerate(chunk_results):
        values = item.get("values")
        if not isinstance(values, list):
            _finalize_cost_estimated_only(request, reservation, "malformed_upstream_response")
            return openai_error_response(
                message="Malformed Vertex AI response: embeddings.values missing.",
                status_code=502,
                error_type="api_error",
                code="bad_gateway",
            )
        try:
            total_tokens += int(item.get("token_count", 0))
        except (TypeError, ValueError):
            pass
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": encode_embedding(values, payload.encoding_format),
            }
        )

    _finalize_cost_success(
        request,
        reservation,
        NormalizedUsage(embedding_tokens=total_tokens, total_tokens=total_tokens),
    )
    return {
        "object": "list",
        "data": data,
        "model": payload.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


def _new_chat_completion_id() -> str:
    """매 요청 고유한 OpenAI 호환 chat completion id를 생성한다."""
    return f"chatcmpl-{uuid.uuid4().hex[:16]}"


def _chat_completions_stream(
    chat_client: VertexChatClient,
    payload: "OpenAIChatRequest",
    messages: list[dict[str, Any]],
    cost_accounting: Any,
    reservation: CostReservation | None,
    provider_model: str | None = None,
    resolved_config: dict[str, Any] | None = None,
    provider: str = "vertex",
) -> StreamingResponse:
    """stream=true 요청을 OpenAI 호환 SSE로 변환하는 StreamingResponse를 만든다."""
    completion_id = _new_chat_completion_id()

    def _chunk(delta: dict[str, Any], finish_reason: str | None) -> str:
        obj = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": payload.model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def event_generator():
        first = True
        final_finish_reason: str | None = None
        final_usage: NormalizedUsage | None = None
        saw_stream_event = False
        finalized_cost = False
        try:
            stream_kwargs = {
                "model": provider_model or payload.model,
                "messages": messages,
                "max_tokens": payload.max_tokens,
                "temperature": payload.temperature,
                "top_p": payload.top_p,
                "stop": payload.stop,
                "response_format": payload.response_format,
            }
            if provider == "ollama":
                stream_kwargs["reasoning_effort"] = payload.reasoning_effort
                stream_kwargs["reasoning"] = payload.reasoning
            elif resolved_config is not None:
                stream_kwargs["resolved_config"] = resolved_config
            async for event in chat_client.stream_chat(**stream_kwargs):
                saw_stream_event = True
                delta_text = event.get("delta_text", "") or ""
                fr = event.get("finish_reason")
                if fr is not None:
                    final_finish_reason = fr
                usage = event.get("usage")
                if isinstance(usage, dict):
                    final_usage = _chat_usage_from_mapping(usage)

                delta: dict[str, Any] = {}
                if first:
                    delta["role"] = "assistant"
                    first = False
                if delta_text:
                    delta["content"] = delta_text

                # 내용 또는 role이 있는 청크만 델타로 내보낸다.
                if delta:
                    yield _chunk(delta, None)
        except VertexAPIError as exc:
            if cost_accounting is not None:
                if saw_stream_event:
                    cost_accounting.finalize_estimated_only(reservation, "stream_error_after_start")
                else:
                    cost_accounting.release_nonbillable(reservation, "upstream_error")
            finalized_cost = True
            # 스트림 시작 전/도중 에러: OpenAI 에러 형태를 SSE data로 흘려보낸 뒤 종료.
            err_obj = {
                "error": {
                    "message": exc.message,
                    "type": map_vertex_status_to_openai_type(exc.status_code),
                    "param": None,
                    "code": exc.code,
                }
            }
            yield f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        finally:
            if reservation is not None and cost_accounting is not None and not finalized_cost:
                if final_usage is not None:
                    cost_accounting.finalize_success(reservation, final_usage)
                else:
                    cost_accounting.finalize_estimated_only(reservation, "stream_missing_usage")

        # 첫 청크가 한 번도 안 나갔다면(빈 스트림) role 청크라도 보낸다.
        if first:
            yield _chunk({"role": "assistant"}, None)

        # 종료 청크: finish_reason 담기 (없으면 stop으로 폴백).
        yield _chunk({}, final_finish_reason or "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _chat_completions_stream_buffered_structured_repair(
    chat_client: VertexChatClient,
    payload: "OpenAIChatRequest",
    messages: list[dict[str, Any]],
    request: Request,
    reservation: CostReservation | None,
    *,
    provider_model: str,
) -> StreamingResponse:
    completion_id = _new_chat_completion_id()

    def _chunk(delta: dict[str, Any], finish_reason: str | None) -> str:
        obj = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": payload.model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def event_generator():
        try:
            result = await _generate_ollama_with_structured_output_repair(
                chat_client,
                payload=payload,
                messages=messages,
                provider_model=provider_model,
                request=request,
                reservation=reservation,
            )
        except VertexAPIError as exc:
            if not (isinstance(exc.raw, dict) and exc.raw.get("cost_managed")):
                _release_cost_nonbillable(request, reservation, "upstream_error")
            err_obj = {
                "error": {
                    "message": exc.message,
                    "type": map_vertex_status_to_openai_type(exc.status_code),
                    "param": None,
                    "code": exc.code,
                }
            }
            yield f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        if not result.get("_cost_managed"):
            _finalize_cost_success(request, reservation, _chat_usage_from_mapping(result.get("usage", {})))
        yield _chunk({"role": "assistant"}, None)
        yield _chunk({"content": result["text"]}, None)
        yield _chunk({}, result.get("finish_reason") or "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completions(
    payload: OpenAIChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    _chat_cfg, model_error = _resolve_chat_model(payload.model)
    if model_error is not None:
        return model_error

    # response_format.type 유효성 검사: 지원하지 않는 type은 400으로 거부한다.
    # 허용 타입은 vertex.SUPPORTED_RESPONSE_FORMAT_TYPES 단일 출처를 공유한다(드리프트 방지).
    if payload.response_format is not None:
        rf_type = payload.response_format.get("type")
        if rf_type not in SUPPORTED_RESPONSE_FORMAT_TYPES:
            return openai_error_response(
                message=f"Unsupported response_format.type: {rf_type!r}. "
                        f"Allowed: {sorted(SUPPORTED_RESPONSE_FORMAT_TYPES)}",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_request",
                param="response_format",
            )

    provider = (_chat_cfg or {}).get("provider", "vertex")
    provider_model = (_chat_cfg or {}).get("provider_model", payload.model)
    chat_client = (
        request.app.state.ollama_chat_client
        if provider == "ollama"
        else request.app.state.vertex_chat_client
    )

    messages = [{"role": m.role, "content": m.content} for m in payload.messages]
    cost_preflight = _preflight_cost(
        request,
        endpoint="chat",
        model=payload.model,
        forecast_usage=_chat_forecast_usage(payload),
    )
    if isinstance(cost_preflight, JSONResponse):
        return cost_preflight
    reservation = cost_preflight

    if payload.stream:
        if _structured_output_repair_applies(
            payload=payload,
            provider=provider,
            provider_model=provider_model,
        ):
            return _chat_completions_stream_buffered_structured_repair(
                chat_client,
                payload,
                messages,
                request,
                reservation,
                provider_model=provider_model,
            )
        return _chat_completions_stream(
            chat_client,
            payload,
            messages,
            _cost_accounting(request),
            reservation,
            provider_model=provider_model,
            resolved_config=_chat_cfg if provider == "vertex" else None,
            provider=provider,
        )

    try:
        generate_kwargs = {
            "model": provider_model,
            "messages": messages,
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "stop": payload.stop,
            "response_format": payload.response_format,
        }
        if provider == "vertex":
            generate_kwargs["resolved_config"] = _chat_cfg
            result = await chat_client.generate(**generate_kwargs)
        else:
            result = await _generate_ollama_with_structured_output_repair(
                chat_client,
                payload=payload,
                messages=messages,
                provider_model=provider_model,
                request=request,
                reservation=reservation,
            )
    except VertexAPIError as exc:
        if not (isinstance(exc.raw, dict) and exc.raw.get("cost_managed")):
            _release_cost_nonbillable(request, reservation, "upstream_error")
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )

    if not result.get("_cost_managed"):
        _finalize_cost_success(request, reservation, _chat_usage_from_mapping(result.get("usage", {})))
    return {
        "id": _new_chat_completion_id(),
        "object": "chat.completion",
        "created": 0,
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": result["usage"],
    }


@app.post("/v1/rerank", response_model=None)
async def rerank(
    payload: CohereRerankRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    if payload.model not in current_allowed_models():
        return openai_error_response(
            message=f"The model '{payload.model}' does not exist.",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    _rank_cfg = model_config(payload.model)
    if _rank_cfg and _rank_cfg.get("kind") != "rerank":
        return openai_error_response(
            message=f"The model '{payload.model}' is not a rerank model.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_model",
            param="model",
        )

    if not payload.documents:
        return openai_error_response(
            message="documents must not be empty.",
            status_code=400,
            error_type="invalid_request_error",
            code="empty_documents",
            param="documents",
        )

    rerank_client: VertexRerankClient = request.app.state.vertex_rerank_client

    records: list[dict[str, Any]] = []
    for i, doc in enumerate(payload.documents):
        content = doc.get("text", "") if isinstance(doc, dict) else str(doc)
        records.append({"id": str(i), "content": content})

    cost_preflight = _preflight_cost(
        request,
        endpoint="rerank",
        model=payload.model,
        forecast_usage=NormalizedUsage(rerank_units=1),
    )
    if isinstance(cost_preflight, JSONResponse):
        return cost_preflight
    reservation = cost_preflight

    try:
        records_out = await rerank_client.rank(
            model=(_rank_cfg or {}).get("provider_model", payload.model),
            query=payload.query,
            records=records,
            top_n=payload.top_n,
            ignore_record_details_in_response=True,
            location=_rank_cfg.get("location", "global") if _rank_cfg else "global",
        )
    except VertexAPIError as exc:
        _release_cost_nonbillable(request, reservation, "upstream_error")
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )

    results = []
    for r in records_out:
        try:
            idx = int(r.get("id", "0"))
        except ValueError:
            idx = 0
        results.append({
            "index": idx,
            "relevance_score": float(r.get("score", 0.0))
        })

    _finalize_cost_success(request, reservation, NormalizedUsage(rerank_units=1))
    return {"results": results}


def _lifespan_with_factories(
    *,
    embedding_client_factory: Any,
    chat_client_factory: Any,
    rerank_client_factory: Any,
    ollama_chat_client_factory: Any,
    cost_accounting_factory: Any,
) -> Any:
    @asynccontextmanager
    async def managed_lifespan(app: FastAPI):
        app.state.vertex_client = embedding_client_factory()
        app.state.vertex_chat_client = chat_client_factory()
        app.state.vertex_rerank_client = rerank_client_factory()
        app.state.ollama_chat_client = ollama_chat_client_factory()
        app.state.cost_accounting = cost_accounting_factory()
        try:
            yield
        finally:
            cost_accounting = getattr(app.state, "cost_accounting", None)
            if cost_accounting is not None:
                cost_accounting.close()
            await app.state.vertex_client.close()
            await app.state.vertex_chat_client.close()
            await app.state.vertex_rerank_client.close()
            await app.state.ollama_chat_client.close()

    return managed_lifespan


_ROUTE_SOURCE_APP = app


def create_app(
    *,
    embedding_client_factory: Any | None = None,
    chat_client_factory: Any | None = None,
    rerank_client_factory: Any | None = None,
    ollama_chat_client_factory: Any | None = None,
    cost_accounting_factory: Any | None = None,
) -> FastAPI:
    embedding_factory = embedding_client_factory or (lambda: VertexEmbeddingClient())
    chat_factory = chat_client_factory or (lambda: VertexChatClient())
    rerank_factory = rerank_client_factory or (lambda: VertexRerankClient())
    ollama_factory = ollama_chat_client_factory or (lambda: OllamaChatClient())
    cost_factory = cost_accounting_factory or (lambda: build_cost_accounting_from_env(os.environ))
    bridge_app = FastAPI(
        title="openai-compatible-bridge",
        version="0.1.0",
        lifespan=_lifespan_with_factories(
            embedding_client_factory=embedding_factory,
            chat_client_factory=chat_factory,
            rerank_client_factory=rerank_factory,
            ollama_chat_client_factory=ollama_factory,
            cost_accounting_factory=cost_factory,
        ),
    )
    bridge_app.router.redirect_slashes = False
    bridge_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    bridge_app.add_exception_handler(RequestValidationError, validation_exception_handler)
    for route in _ROUTE_SOURCE_APP.router.routes:
        if isinstance(route, APIRoute):
            bridge_app.router.routes.append(route)
    return bridge_app


app = create_app()
