"""Pre-generation synthetic corpus; no user transcripts or real credentials.

Ground-truth sentences are review rubrics, not fabricated provider output.
This file is fixed before candidate live generation. Do not tune to responses.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticCase:
    case_id: str
    tool_name: str
    arguments: dict
    source: str
    context: dict
    required_facts: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    expected: str = "semantic"


def _long(*facts: str) -> str:
    noise = [f"Synthetic detail row {i}: ordinary component description for synthetic corpus." for i in range(40)]
    noise[10:10] = facts
    return "\n".join(noise)


_TESTS = _long("Command result: 10 tests passed.", "Build ID: synthetic-job-741", "Artifact path: synthetic-output/test-report.json")
_LOOKUP = _long("The synthetic resolve_component function maps a component name to its catalog entry.", "Lookup mode is exact-name matching.")
_SEARCH = _long("Search matches: resolve_component appears in the synthetic catalog module.", "Match path: synthetic-src/catalog.py")
_CHOICES = _long("The synthetic catalog describes exact-name component lookup.", "Alpha topic: cache reuse retains an existing component entry.", "Beta topic: locale selection chooses the Korean description.")

CASES = (
    SemanticCase("Q01", "terminal", {"command": "uv run pytest tests/test_example.py -q"}, _TESTS, {},
                 ("pytest ran for tests/test_example.py", "10 tests passed"), ("entire suite passed", "files were modified")),
    SemanticCase("Q02", "terminal", {"command": "uv run pytest -q"}, _TESTS, {},
                 ("pytest ran without a selected test file", "10 tests passed"), ("only tests/test_example.py ran", "files were modified")),
    SemanticCase("Q03", "read_file", {"path": "synthetic-src/catalog.py", "offset": 1, "limit": 100}, _LOOKUP, {},
                 ("catalog.py was read", "resolve_component maps a component name to a catalog entry"), ("file modified", "function executed")),
    SemanticCase("Q04", "search_files", {"pattern": "resolve_component", "target": "content", "path": "synthetic-src", "file_glob": "*.py"}, _SEARCH, {},
                 ("content searched for resolve_component in synthetic-src Python files", "catalog.py matched"), ("file created", "code executed")),
    SemanticCase("Q05", "read_file", {"path": "synthetic-src/catalog.py"}, _CHOICES, {"retain_for": "Alpha topic: cache reuse"},
                 ("file read", "exact-name component lookup", "cache reuse retains an existing component entry"), ("cache was cleared", "file changed")),
    SemanticCase("Q06", "read_file", {"path": "synthetic-src/catalog.py"}, _CHOICES, {"retain_for": "Beta topic: locale selection"},
                 ("file read", "exact-name component lookup", "locale selection chooses the Korean description"), ("locale was changed by this read", "file changed")),
    SemanticCase("Q07", "terminal", {"command": "synthetic-inspector list-components"}, _long("Warning: optional component descriptions are omitted.", "The synthetic inspector lists component names."), {},
                 ("inspector listed component names", "optional descriptions omitted", "exit code not provided"), ("exit code zero", "all checks passed")),
    SemanticCase("Q08", "terminal", {"command": "uv run pytest -q"}, _long("FAILED synthetic check: component entry missing."), {"purpose": "Confirm all checks passed"},
                 ("original retained",), ("lfm success",), "original"),
    SemanticCase("Q09", "terminal", {"command": "SYNTHETIC_TOKEN=not-a-real-secret synthetic-inspector"}, _LOOKUP, {},
                 ("original retained and no summary generation",), ("credential sent",), "original"),
    SemanticCase("Q10", "terminal", {"command": "synthetic-inspector"}, _LOOKUP, {},
                 ("duplicate assistant/result rejected",), ("summary generated",), "ambiguous"),
    SemanticCase("Q11", "terminal", {"command": "synthetic-inspector " + "long-argument " * 300}, _LOOKUP, {},
                 ("input refused without command truncation",), ("summary success",), "original"),
    SemanticCase("Q12", "terminal", {"command": "uv run pytest tests/test_example.py -q"}, _TESTS, {},
                 ("summary applied on follow-up", "list returns original item", "unhide restores exact UTF-8 original", "no regeneration"), ("tool rerun",), "restore"),
)
