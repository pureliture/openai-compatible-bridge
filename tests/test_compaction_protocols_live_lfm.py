"""Opt-in native HTTP/SSE fixtures + real local LFM; NOT a Foundry canary.

Only synthetic result text reaches Ollama. The scripted main provider never
leaves MockTransport; real auxiliary calls are restricted to local Ollama.
Run: RUN_LFM_PROTOCOL_INTEGRATION=1 .venv/bin/python -m pytest -sv <this file>
"""
from __future__ import annotations

import asyncio
import copy
import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.context_compaction import AFFINITY_HEADER, HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.ollama import OllamaChatClient
import test_compaction_anthropic as anthropic
import test_compaction_anthropic_stream as anthropic_stream
import test_compaction_chat_stream as chat_stream
import test_compaction_foundry_protocols as responses
import test_compaction_google as google
import test_compaction_google_stream as google_stream
import test_compaction_responses_stream as responses_stream
import test_compaction_xai_stream as xai_stream

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LFM_PROTOCOL_INTEGRATION") != "1",
    reason="set RUN_LFM_PROTOCOL_INTEGRATION=1 for real local Ollama/LFM integration",
)

PROTOCOLS = (
    "openai_chat_completions", "openai_responses", "anthropic_messages",
    "google_generate_content", "xai_responses",
)
MODEL = "lfm2.5-thinking:latest"
ALIAS = "foundry:synthetic-live-protocol"
AFFINITY = "synthetic-live-protocol-conversation"
SOURCE = responses.SOURCE + "\nBuild ID: synthetic-job-48291\nArtifact path: synthetic-output/reports/catalog.json"
EVIDENCE = ("Build ID: synthetic-job-48291", "Artifact path: synthetic-output/reports/catalog.json")


class RecordedLocalLFM(OllamaChatClient):
    """Record real requests/results, without substituting generated text or usage."""

    def __init__(self):
        super().__init__(base_url="http://127.0.0.1:11434")
        self.calls = []
        self.results = []

    async def generate(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        assert len(self.calls) == 1, "this roundtrip must make exactly one real LFM call"
        assert kwargs["model"] == MODEL
        assert kwargs["max_tokens"] == 384
        assert kwargs["timeout_seconds"] == 60
        assert len(json.dumps(kwargs["messages"], ensure_ascii=False).encode()) <= 12_288
        result = await super().generate(**kwargs)
        self.results.append(copy.deepcopy(result))
        return result


def _reply(protocol, stream, *, name=None, args=None, identity="private-call", text=None):
    adapter = (anthropic if protocol == "anthropic_messages" else
               google if protocol == "google_generate_content" else
               chat_stream if protocol == "openai_chat_completions" else responses)
    calls = [adapter.native_call(name, args, identity)] if name and adapter is not chat_stream else (
        [chat_stream.call(name, args, identity)] if name else [])
    if stream:
        wire = {
            "openai_chat_completions": chat_stream.wire,
            "openai_responses": responses_stream.wire,
            "anthropic_messages": anthropic_stream.wire,
            "google_generate_content": google_stream.wire,
            "xai_responses": xai_stream.wire,
        }[protocol]
        return wire(*calls, **{"content" if protocol == "openai_chat_completions" else "text": text})
    if protocol == "openai_chat_completions":
        return {"id": "private-upstream-id", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})},
                "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
    return adapter.native_response(*calls, text=text)


def _output(protocol, body, identity="original-call", name="terminal"):
    if protocol == "anthropic_messages":
        return anthropic.output(body, identity)
    if protocol == "google_generate_content":
        result = google.output(body, name)
        return result["content"] if name == "terminal" else json.dumps(result)
    if protocol == "openai_chat_completions":
        return chat_stream.output(body, identity)
    return next(i["output"] for i in body["input"]
                if i.get("type") == "function_call_output" and i["call_id"] == identity)


def _assert_original(protocol, body, messages):
    if protocol == "anthropic_messages":
        anthropic.assert_original_call(body)
    elif protocol == "google_generate_content":
        google.assert_original_call(body)
    elif protocol == "openai_chat_completions":
        assert body["messages"][:2] == messages[:2]
    else:
        assert body["input"][0] == messages[0]
        assert {"type": "function_call", "call_id": "original-call", "name": "terminal",
                "arguments": responses.ARGS} in body["input"]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_native_protocol_real_lfm_hide_list_exact_unhide(monkeypatch, protocol, stream):
    for key, value in {
        "CONTEXT_COMPACTION_ENABLED": "true", "CONTEXT_COMPACTION_LFM_ENABLED": "true",
        "CONTEXT_COMPACTION_LAYA_ENABLED": "false", "CONTEXT_COMPACTION_LFM_MODEL": MODEL,
        "CONTEXT_COMPACTION_LFM_MAX_OUTPUT_TOKENS": "384",
        "CONTEXT_COMPACTION_LFM_TIMEOUT_SECONDS": "60",
        "CONTEXT_COMPACTION_LFM_MAX_INPUT_BYTES": "12288",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("openai_compatible_bridge.main.BRIDGE_API_KEY", "")
    monkeypatch.setitem(vertex.MODEL_REGISTRY, ALIAS, {
        "provider": "foundry", "kind": "chat", "provider_model": "synthetic-google",
        "protocol": protocol,
    })
    bodies, replies, streams = [], [], []
    messages = copy.deepcopy(responses.MESSAGES)
    messages[2]["content"] = SOURCE
    original_messages = copy.deepcopy(messages)

    async def handler(request):
        body = json.loads(request.content)
        suffix = {
            "openai_chat_completions": "/openai/v1/chat/completions",
            "openai_responses": "/openai/v1/responses",
            "anthropic_messages": "/anthropic/v1/messages",
            "google_generate_content": "/google/v1/models/synthetic-google:" + ("streamGenerateContent" if stream else "generateContent"),
            "xai_responses": "/xai/v1/responses",
        }[protocol]
        assert request.url.path.endswith(suffix)
        if protocol == "google_generate_content":
            assert "stream" not in body and "messages" not in body
            assert request.url.query == (b"alt=sse" if stream else b"")
        else:
            assert body["stream"] is stream, "must exercise actual upstream stream mode"
        if protocol == "anthropic_messages":
            assert request.headers["anthropic-version"] == "2023-06-01"
        bodies.append(body)
        assert replies, "unexpected main-provider HTTP call"
        reply = replies.pop(0)
        if stream:
            chunked = responses_stream.Chunked(reply)
            streams.append(chunked)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=chunked)
        return httpx.Response(200, json=reply)

    provider = FoundryChatClient(
        base_url="https://foundry.example/api/v2/llm/proxy/openai/v1/chat/completions", token="synthetic",
    )
    asyncio.run(provider.http.aclose())
    provider.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if stream:
        async def forbidden_generate(**kwargs):
            pytest.fail("actual native upstream streaming must not fall back to generate")
        monkeypatch.setattr(provider, "generate", forbidden_generate)
    lfm = RecordedLocalLFM()
    app = create_app(
        embedding_client_factory=responses.Unused, chat_client_factory=responses.Unused,
        rerank_client_factory=responses.Unused, ollama_chat_client_factory=lambda: lfm,
        foundry_chat_client_factory=lambda: provider, cost_accounting_factory=lambda: None,
    )

    def queue(*, name=None, args=None, identity="private-call", final):
        if name:
            replies.append(_reply(protocol, stream, name=name, args=args, identity=identity,
                                  text="PRIVATE intermediate answer"))
        replies.append(_reply(protocol, stream, text=final))

    with TestClient(app) as client:
        def post(expected, submitted=None):
            body = {"model": ALIAS, "messages": copy.deepcopy(messages if submitted is None else submitted),
                    "tools": [responses.TOOL], "stream": stream}
            if stream:
                body["stream_options"] = {"include_usage": True}
            response = client.post("/v1/chat/completions", headers={AFFINITY_HEADER: AFFINITY}, json=body)
            assert response.status_code == 200, response.text
            assert all(s not in response.text for s in (
                HIDE_TOOL, LIST_TOOL, UNHIDE_TOOL, "private-call", "list-call", "unhide-call",
                "fc-distinct-item", "PRIVATE", "private-upstream-id", "resp-private", "msg-private",
            )), response.text
            if stream:
                rows = responses_stream.public(response)
                assert responses_stream.text(rows) == expected
                assert rows[-2]["choices"][0]["finish_reason"] == "stop"
                assert rows[-1]["usage"]["total_tokens"] > 0
            else:
                data = response.json()
                assert data["choices"][0]["message"]["content"] == expected
                assert not data["choices"][0]["message"].get("tool_calls")
                assert data["choices"][0]["finish_reason"] == "stop"
            return response

        queue(name=HIDE_TOOL, args={"tool_call_id": "original-call"}, final="hidden")
        post("hidden")
        measurement = app.state.context_compaction_store.last_measurement
        item = app.state.context_compaction_store.items(AFFINITY)[0]
        assert item.compaction_source == "lfm", (
            "rule fallback is NOT a live LFM pass: " + str(measurement.lfm_fallback_reason)
            + "; real output=" + json.dumps(lfm.results, ensure_ascii=False)
        )
        assert measurement.lfm_applied and measurement.lfm_calls == 1
        assert measurement.lfm_fallback_reason is None and measurement.provider_calls == 2
        assert len(lfm.calls) == len(lfm.results) == 1
        real_summary = json.loads(lfm.results[0]["text"])["summary"]
        assert real_summary and real_summary in item.compacted
        assert "catalog" in real_summary.casefold() and "metadata" in real_summary.casefold()
        assert all(line in item.compacted for line in EVIDENCE)
        assert len(item.compacted.encode()) < len(SOURCE.encode())
        assert _output(protocol, bodies[0]) == SOURCE
        assert _output(protocol, bodies[1]) == item.compacted != SOURCE
        assert json.loads(_output(protocol, bodies[1], "private-call", HIDE_TOOL))["ok"]
        packet = json.loads(lfm.calls[0]["messages"][1]["content"].split("\n\n", 1)[1])
        assert set(packet) == {"result", "required_evidence"}
        assert responses.ARGS not in json.dumps(lfm.calls)
        assert messages[0]["content"] not in json.dumps(lfm.calls)

        queue(final="after")
        post("after")
        assert _output(protocol, bodies[-1]) == item.compacted
        queue(name=LIST_TOOL, args={}, identity="list-call", final="listed")
        post("listed")
        listed = json.loads(_output(protocol, bodies[-1], "list-call", LIST_TOOL))
        assert listed["ok"] and len(listed["items"]) == 1
        assert listed["items"][0]["item_id"] == item.item_id
        assert listed["items"][0]["tool_call_id"] == "original-call"
        assert listed["items"][0]["original_available"] is True
        rendered = copy.deepcopy(messages)
        rendered[2]["content"] = item.compacted
        queue(name=UNHIDE_TOOL, args={"item_id": item.item_id}, identity="unhide-call", final="restored")
        post("restored", rendered)
        assert _output(protocol, bodies[-2]) == item.compacted
        assert _output(protocol, bodies[-1]).encode() == SOURCE.encode()
        assert json.loads(_output(protocol, bodies[-1], "unhide-call", UNHIDE_TOOL))["visibility"] == "visible"
        queue(final="still restored")
        post("still restored", rendered)
        assert _output(protocol, bodies[-1]).encode() == SOURCE.encode()
        restored = app.state.context_compaction_store.get(AFFINITY, item.item_id)
        assert restored.visibility == "original" and restored.original == SOURCE
        assert len(lfm.calls) == 1 and not replies and len(bodies) == 8
        assert messages == original_messages and rendered[2]["content"] == item.compacted
        for body in bodies:
            _assert_original(protocol, body, messages)
        assert all(s.closed for s in streams)
        assert len(streams) == (8 if stream else 0)
        print("REAL_AUX_LFM_NATIVE_FIXTURE " + json.dumps({
            "protocol": protocol, "stream": stream, "main_evidence": "scripted-native-http-fixture",
            "aux_evidence": "real-local-ollama", "model": MODEL, "lfm_calls": len(lfm.calls),
            "source_bytes": len(SOURCE.encode()), "compacted_bytes": len(item.compacted.encode()),
            "actual_summary": real_summary, "actual_aux_usage": lfm.results[0].get("usage"),
            "compaction_source": item.compaction_source, "exact_unhide": True,
        }, ensure_ascii=False))
