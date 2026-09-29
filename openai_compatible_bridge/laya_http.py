"""Bounded HTTP adapter for a remote, opt-in Laya System-One server.

No Laya package or model is installed on the bridge. This is not an OpenAI API.
The result is a candidate only, never an authority for safety or secret removal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import urlsplit

import httpx


class LayaUnavailable(Exception):
    """Bad transport, HTTP status or unverifiable model decision; no raw body exposed."""


class LayaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 60.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or timeout_seconds <= 0
        ):
            raise ValueError("LAYA_BASE_URL must be an HTTP(S) origin without credentials or path")
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    async def close(self) -> None:
        await self._http.aclose()

    async def health(self) -> bool:
        """A health probe is informative only; it never authorizes unsafe compaction."""
        try:
            response = await self._http.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError):
            return False
        return (
            isinstance(data, dict)
            and data.get("status") == "ok"
            and isinstance(data.get("loaded"), list)
            and "multilingual" in data["loaded"]
        )

    async def choose(
        self,
        state: str,
        questions: dict[str, Any],
        question_id: str,
    ) -> str:
        """Return a validated choice key; otherwise ask the caller to use the rule baseline."""
        question = questions.get(question_id)
        criteria = question.get("criteria") if isinstance(question, dict) else None
        if (
            not isinstance(state, str)
            or not state
            or len(state) > 50000
            or not isinstance(criteria, dict)
            or not criteria
            or len(criteria) > 100
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in criteria.items())
        ):
            raise LayaUnavailable("invalid_request")
        try:
            response = await self._http.post(
                f"{self.base_url}/v1/systemone",
                json={"model": "multilingual", "state": state, "questions": questions,
                      "max_len": 8192, "head_max_len": 4096},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise LayaUnavailable(f"http_{exc.response.status_code}") from None
        except (httpx.HTTPError, ValueError):
            raise LayaUnavailable("transport_or_json_error") from None

        if not isinstance(data, Mapping) or not isinstance(data.get("routing"), Mapping):
            raise LayaUnavailable("invalid_response")
        # The response's `model` is always the decision head, NOT the checkpoint.
        if data["routing"].get("model") != "multilingual":
            raise LayaUnavailable("wrong_checkpoint")
        answers = data.get("answers")
        answer = answers.get(question_id) if isinstance(answers, Mapping) else None
        if not isinstance(answer, Mapping) or answer.get("type") != "choice":
            raise LayaUnavailable("invalid_answer")
        choice = answer.get("choice")
        probabilities = answer.get("probabilities")
        if not isinstance(choice, str) or choice not in criteria or not isinstance(probabilities, Mapping):
            raise LayaUnavailable("invalid_choice")
        prob = probabilities.get(choice)
        confidence = answer.get("answer_confidence")
        if not self._probability(prob) or not self._probability(confidence):
            raise LayaUnavailable("invalid_probability")
        selected = cast(float, prob)
        reported = cast(float, confidence)
        if abs(selected - reported) > 0.001:
            raise LayaUnavailable("inconsistent_probability")
        # A conservative rejection heuristic, NOT calibrated accuracy. Domain evaluation
        # is required before the operator may enable this selector at all.
        if selected < 0.75:
            raise LayaUnavailable("uncertain_answer")
        return choice

    @staticmethod
    def _probability(value: Any) -> bool:
        return (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
        )
