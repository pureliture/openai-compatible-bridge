"""Offline acceptance checks for exact source-grounded exit-code claims."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
from typing import Callable, cast

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_lfm_prompt_routing.py"
SCRIPT_NAMESPACE = runpy.run_path(str(SCRIPT))
KNOWN_EXIT = json.dumps({"output": "2 passed units\n" * 60, "exit_code": 0})
UNKNOWN_EXIT = json.dumps({"output": "2 passed units\n" * 60, "exit_code": None})


def _check_exit_claim(source: str, summary: str) -> bool:
    helper = SCRIPT_NAMESPACE.get("exit_claim_matches_source")
    assert callable(helper), "verification script needs a pure exit-claim helper"
    return cast(Callable[[str, str], bool], helper)(source, summary)


def test_unrelated_test_count_and_zero_do_not_satisfy_known_exit_claim():
    assert not _check_exit_claim(KNOWN_EXIT, "2 passed units; 0 items were omitted.")


def test_exact_explicit_known_exit_claim_is_accepted():
    assert _check_exit_claim(KNOWN_EXIT, "2 passed units; exit code 0.")
    assert _check_exit_claim(KNOWN_EXIT, "2 passed units; return code 0.")


@pytest.mark.parametrize(
    "summary",
    [
        "2 passed units.",
        "2 passed units; exit code 1.",
        "2 passed units; exit code 0; return code 0.",
        "2 passed units; exit code 0 and exit status 1.",
        "2 passed units; exit code 0 or 1.",
        "2 passed units; exit code 0 was not reported.",
        "2 passed units; no exit code 0 was reported.",
    ],
)
def test_known_exit_rejects_missing_wrong_multiple_and_negated_claims(summary: str):
    assert not _check_exit_claim(KNOWN_EXIT, summary)


@pytest.mark.parametrize(
    "summary",
    [
        "2 passed units; exit code 0.",
        "2 passed units; exit_code: 7.",
        "2 passed units; exit-status was 23.",
        "2 passed units; process exited with code 4.",
    ],
)
def test_unknown_exit_rejects_any_invented_numeric_exit_claim(summary: str):
    assert not _check_exit_claim(UNKNOWN_EXIT, summary)


@pytest.mark.parametrize(
    "summary",
    [
        "2 passed units; exitcode: 7.",
        "2 passed units; returncode 23.",
        "2 passed units; exit code unknown.",
    ],
)
def test_unknown_exit_rejects_nonstandard_and_nonnumeric_exit_mentions(summary: str):
    assert not _check_exit_claim(UNKNOWN_EXIT, summary)


def test_unknown_exit_allows_unrelated_zero_without_exit_claim():
    assert _check_exit_claim(UNKNOWN_EXIT, "2 passed units; 0 items were omitted.")
