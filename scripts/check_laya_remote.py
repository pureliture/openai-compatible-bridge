"""Read-only smoke probe of an operator-provided Laya URL.

Run: LAYA_BASE_URL=http://<tailscale-host>:8000 uv run --no-sync python -m scripts.check_laya_remote
Prints only sanitized status and timing, never the URL, inputs or response text.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx

from openai_compatible_bridge.laya_http import LayaClient, LayaUnavailable


async def probe(base_url: str, http: httpx.AsyncClient, *, approved_origin: str = "") -> dict[str, Any]:
    # Validate the operator-supplied origin before either request.
    client = LayaClient(base_url, approved_origin=approved_origin, http=http)
    origin = client.base_url
    result: dict[str, Any] = {"health_ok": False, "multilingual_loaded": False,
                              "device": None, "inference_ok": False, "routed_multilingual": False,
                              "inference_seconds": None, "status": None}
    try:
        health = await http.get(origin + "/health", follow_redirects=False)
        result["health_status"] = health.status_code
        body = health.json() if health.status_code == 200 else {}
        if not isinstance(body, dict):
            body = {}
        result["health_ok"] = body.get("status") == "ok"
        result["multilingual_loaded"] = "multilingual" in body.get("loaded", []) if isinstance(body.get("loaded"), list) else False
        result["device"] = body.get("device") if body.get("device") in {"cpu", "cuda", "mps"} else None
    except (httpx.HTTPError, ValueError):
        result["status"] = "health_unavailable"
        return result
    if not (result["health_ok"] and result["multilingual_loaded"]):
        result["status"] = "checkpoint_not_ready"
        return result
    started = time.monotonic()
    try:
        await client.choose(
            "영수증이 두 번 청구되었습니다. A duplicated invoice was issued.",
            {"kind": {"type": "choice", "instructions": "Classify the issue",
                      "criteria": {"billing": "payment and billing", "technical": "software error"}}},
            "kind",
        )
    except LayaUnavailable as exc:
        result["inference_seconds"] = round(time.monotonic() - started, 3)
        result["status"] = str(exc)  # adapter exposes only fixed failure categories
        return result
    result["inference_seconds"] = round(time.monotonic() - started, 3)
    result["inference_status"] = 200
    result["routed_multilingual"] = True
    result["inference_ok"] = True
    result["status"] = "ok"
    return result


async def main() -> int:
    base_url = os.getenv("LAYA_BASE_URL", "")
    if not base_url:
        print("LAYA_BASE_URL is required (not provided by the screenshot).", file=sys.stderr)
        return 2
    if os.getenv("LAYA_REMOTE_TEST_APPROVED") != "true":
        print("Remote test approval is required after log/retention verification.", file=sys.stderr)
        return 2
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(65.0), trust_env=False) as http:
            result = await probe(base_url, http, approved_origin=os.getenv("LAYA_APPROVED_ORIGIN", ""))
    except ValueError:
        print("Laya origin must match the approved tailnet destination.", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["inference_ok"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
