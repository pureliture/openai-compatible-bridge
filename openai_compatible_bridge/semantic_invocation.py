"""Small, conservative adapters for semantic summaries, never tool execution."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


class InvocationRejected(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def context_hint(args: dict) -> dict[str, str] | None:
    if set(args) - {"tool_call_id", "context"}:
        raise InvocationRejected("invalid_arguments")
    if "context" not in args:
        return None
    hint = args["context"]
    if (not isinstance(hint, dict) or not hint or set(hint) - {"purpose", "retain_for"}
            or any(not isinstance(v, str) or not v.strip() or len(v) > 300 for v in hint.values())):
        raise InvocationRejected("invalid_arguments")
    return dict(hint)


def match_invocation(messages: list[dict], ident: str) -> dict:
    calls = [(i, c) for i, m in enumerate(messages) if m.get("role") == "assistant"
             for c in (m.get("tool_calls") or []) if isinstance(c, dict) and c.get("id") == ident]
    results = [(i, m) for i, m in enumerate(messages) if m.get("role") == "tool" and m.get("tool_call_id") == ident]
    if not results or not calls:
        raise InvocationRejected("not_found")
    if len(calls) != 1 or len(results) != 1:
        raise InvocationRejected("ambiguous")
    if calls[0][0] >= results[0][0]:
        raise InvocationRejected("invalid_invocation_order")
    if not isinstance(results[0][1].get("content"), str):
        raise InvocationRejected("unsupported_content")
    function = calls[0][1].get("function") or {}
    name = function.get("name")
    raw = function.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        raise InvocationRejected("invalid_invocation") from None
    if not isinstance(args, dict):
        raise InvocationRejected("invalid_invocation")
    if name in {"hide_context", "list_context_items", "unhide_context"}:
        raise InvocationRejected("internal_target")
    fields = {"terminal": {"command", "workdir"}, "read_file": {"path", "offset", "limit"},
              "search_files": {"pattern", "target", "path", "file_glob"}}
    if name not in fields:
        raise InvocationRejected("unsupported_tool")
    defaults = {"terminal": {"background": False, "pty": False, "persist_on_release": False,
                              "heartbeat": 0, "notify": False},
                "read_file": {}, "search_files": {"limit": 50, "offset": 0, "order": "discovery",
                                                      "output_mode": "content", "context": 0}}
    allowed_extra = set(defaults[name]) | ({"timeout"} if name == "terminal" else set())
    if set(args) - fields[name] - allowed_extra:
        raise InvocationRejected("unsupported_arguments")
    if any(k in args and (type(args[k]) is not type(v) or args[k] != v) for k, v in defaults[name].items()):
        raise InvocationRejected("unsupported_arguments")
    if "timeout" in args and (type(args["timeout"]) is not int or args["timeout"] < 1):
        raise InvocationRejected("unsupported_arguments")
    required = {"terminal": "command", "read_file": "path", "search_files": "pattern"}[name]
    if not isinstance(args.get(required), str) or not args[required].strip():
        raise InvocationRejected("invalid_invocation")
    for key in fields[name] & set(args):
        if key in {"offset", "limit"}:
            if type(args[key]) is not int or args[key] < 1:
                raise InvocationRejected("invalid_invocation")
        elif not isinstance(args[key], str):
            raise InvocationRejected("invalid_invocation")
    text = canonical(args)
    # Known patterns only; this is not a complete secret scrubber.
    if re.search(r"(?i)authorization|bearer\s|password|passwd|credential|secret|api[_-]?key|access[_-]?token|--token\b|\b[A-Za-z_][A-Za-z_0-9]*=|\benv\s|\bexport\s|://[^\s/]+:[^\s/]+@", text):
        raise InvocationRejected("sensitive_invocation")
    invocation = {"tool_name": name, "arguments": {k: args[k] for k in sorted(fields[name] & set(args))}}
    if len(canonical(invocation).encode()) > 2048:
        raise InvocationRejected("invocation_too_large")
    return invocation


def invocation_digest(invocation: dict, messages: list[dict] | None = None, ident: str | None = None) -> str:
    identity = dict(invocation)
    if messages is not None:
        call = next(c for m in messages if m.get("role") == "assistant"
                    for c in (m.get("tool_calls") or []) if c.get("id") == ident)
        raw = call["function"]["arguments"]
        args = json.loads(raw) if isinstance(raw, str) else raw
        identity["omitted_options"] = {k: v for k, v in args.items() if k not in invocation["arguments"]}
    return hashlib.sha256(canonical(identity).encode()).hexdigest()
