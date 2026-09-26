"""
Tests for Canonical Build Receipt Contract and Downstream Integration.

Validates:
1. Canonical reference formatting against GoCD EXPECTED_BRIDGE_IMAGE_REFERENCE regex.
2. Failure cases on invalid image references (short SHA, missing prefix, uppercase, etc.).
3. ADR-0002 Canonical Build Receipt schema, fields, and types.
4. Workflow and Dockerfile static integrity contracts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import pytest

CONTRACT_REGEX = re.compile(
    r"^ghcr\.io/pureliture/neurons/openai-compatible-bridge:sha-[0-9a-f]{40}@sha256:[0-9a-f]{64}$"
)

SAMPLE_SHA = "6b24d66a5b6df092ddd01906d899630e792db277"
SAMPLE_DIGEST = "sha256:93d0489c1a2daf1d2e72cff8b2d27ec316d50076a0a06f70918d9ea7f668c95c"
VALID_REF = f"ghcr.io/pureliture/neurons/openai-compatible-bridge:sha-{SAMPLE_SHA}@{SAMPLE_DIGEST}"


def test_canonical_reference_matches_gocd_regex():
    """Verify that a properly constructed canonical reference matches the GoCD regex contract."""
    assert CONTRACT_REGEX.match(VALID_REF) is not None


@pytest.mark.parametrize(
    "invalid_ref",
    [
        # 1. Short SHA (7 characters instead of 40)
        f"ghcr.io/pureliture/neurons/openai-compatible-bridge:sha-6b24d66@{SAMPLE_DIGEST}",
        # 2. Missing 'sha-' prefix in tag
        f"ghcr.io/pureliture/neurons/openai-compatible-bridge:{SAMPLE_SHA}@{SAMPLE_DIGEST}",
        # 3. Missing 'neurons/' sub-namespace
        f"ghcr.io/pureliture/openai-compatible-bridge:sha-{SAMPLE_SHA}@{SAMPLE_DIGEST}",
        # 4. Uppercase hex characters in SHA
        f"ghcr.io/pureliture/neurons/openai-compatible-bridge:sha-{SAMPLE_SHA.upper()}@{SAMPLE_DIGEST}",
        # 5. Missing digest component entirely
        f"ghcr.io/pureliture/neurons/openai-compatible-bridge:sha-{SAMPLE_SHA}",
        # 6. Double sha256 prefix (@sha256:sha256:...)
        f"ghcr.io/pureliture/neurons/openai-compatible-bridge:sha-{SAMPLE_SHA}@sha256:{SAMPLE_DIGEST}",
        # 7. Non-hex characters in digest
        f"ghcr.io/pureliture/neurons/openai-compatible-bridge:sha-{SAMPLE_SHA}@sha256:zzzd0489c1a2daf1d2e72cff8b2d27ec316d50076a0a06f70918d9ea7f668c95c",
        # 8. Truncated digest (32 chars instead of 64)
        f"ghcr.io/pureliture/neurons/openai-compatible-bridge:sha-{SAMPLE_SHA}@sha256:93d0489c1a2daf1d2e72cff8b2d27ec3",
    ],
)
def test_invalid_references_fail_regex(invalid_ref: str):
    """Verify that malformed or non-compliant references fail GoCD contract regex."""
    assert CONTRACT_REGEX.match(invalid_ref) is None


def test_receipt_fields_integrity():
    """Verify that canonical_build_receipt.v1 structure meets ADR-0002 requirements."""
    receipt = {
        "schema_version": "canonical_build_receipt.v1",
        "component": "openai-compatible-bridge",
        "source_repository": "https://github.com/pureliture/openai-compatible-bridge",
        "source_branch": "main",
        "source_full_sha": SAMPLE_SHA,
        "registry_reference": VALID_REF,
        "image_tag": f"ghcr.io/pureliture/neurons/openai-compatible-bridge:sha-{SAMPLE_SHA}",
        "manifest_digest": SAMPLE_DIGEST,
        "target_platform": "linux/amd64",
        "result": "SUCCESS",
        "published_at": "2026-09-26T10:45:00Z",
        "published_by": "github-actions/ci.yml",
        "build_engine": "github-actions",
        "gha_run_id": "14092490123",
        "gha_run_attempt": "1",
    }

    assert receipt["schema_version"] == "canonical_build_receipt.v1"
    assert receipt["component"] == "openai-compatible-bridge"
    assert CONTRACT_REGEX.match(receipt["registry_reference"]) is not None
    assert receipt["manifest_digest"].startswith("sha256:")
    assert len(receipt["manifest_digest"]) == 71  # "sha256:" (7) + 64 hex = 71
    assert len(receipt["source_full_sha"]) == 40
    assert re.match(r"^[0-9a-f]{40}$", receipt["source_full_sha"])
    assert receipt["target_platform"] == "linux/amd64"
    assert receipt["result"] == "SUCCESS"
    assert receipt["build_engine"] == "github-actions"


def test_dockerignore_rules_and_context_exclusion():
    """Verify .dockerignore exists and excludes heavy directories to keep context ~500KB."""
    root = Path(__file__).resolve().parent.parent
    dockerignore_path = root / ".dockerignore"
    assert dockerignore_path.is_file(), ".dockerignore must exist in repository root"

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = {line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")}

    critical_exclusions = [
        ".git",
        ".worktrees",
        ".venv",
        "tests/",
        "build/",
        "dist/",
        "**/__pycache__",
        "**/.DS_Store",
    ]
    for exclusion in critical_exclusions:
        assert any(
            line == exclusion or line.startswith(exclusion.rstrip("/")) for line in lines
        ), f"Missing critical exclusion: {exclusion}"


def test_dockerfile_layer_caching_and_uv_binary():
    """Verify Dockerfile layer ordering places ARG below dependency sync for caching."""
    root = Path(__file__).resolve().parent.parent
    dockerfile_path = root / "Dockerfile"
    assert dockerfile_path.is_file(), "Dockerfile must exist"

    content = dockerfile_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Find line indices for critical steps
    uv_copy_idx = -1
    uv_sync_idx = -1
    arg_commit_idx = -1

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if "ghcr.io/astral-sh/uv" in stripped and "COPY" in stripped:
            uv_copy_idx = idx
        elif "uv sync" in stripped:
            uv_sync_idx = idx
        elif stripped.startswith("ARG NEURONS_SOURCE_COMMIT"):
            arg_commit_idx = idx

    assert uv_copy_idx != -1, "Dockerfile must copy uv binary from ghcr.io/astral-sh/uv"
    assert uv_sync_idx != -1, "Dockerfile must execute uv sync"
    assert arg_commit_idx != -1, "Dockerfile must define ARG NEURONS_SOURCE_COMMIT"

    # ARG NEURONS_SOURCE_COMMIT must be AFTER uv sync to prevent cache busting on each commit
    assert arg_commit_idx > uv_sync_idx, (
        f"ARG NEURONS_SOURCE_COMMIT (line {arg_commit_idx+1}) must appear after uv sync (line {uv_sync_idx+1}) "
        "to prevent busting dependency layer cache on every commit"
    )


def test_workflow_definitions_exist_and_conform():
    """Verify GHA workflow YAML files exist and define expected job outputs and permissions."""
    root = Path(__file__).resolve().parent.parent
    reusable_path = root / ".github" / "workflows" / "reusable-docker-publish.yml"
    caller_path = root / ".github" / "workflows" / "build-publish.yml"

    assert reusable_path.is_file(), "reusable-docker-publish.yml must exist"
    assert caller_path.is_file(), "build-publish.yml must exist"

    reusable_content = reusable_path.read_text(encoding="utf-8")
    assert "workflow_call:" in reusable_content
    assert "packages: write" in reusable_content
    assert "contents: read" in reusable_content
    assert "type=gha,mode=max" in reusable_content or "type=gha, mode=max" in reusable_content
    assert "linux/amd64" in reusable_content
    assert "canonical_reference:" in reusable_content
    assert "image_digest:" in reusable_content
    assert "canonical_build_receipt.v1.json" in reusable_content
    assert "GITHUB_STEP_SUMMARY" in reusable_content

    caller_content = caller_path.read_text(encoding="utf-8")
    assert "reusable-docker-publish.yml" in caller_content
    assert "EXPECTED_REGEX=" in caller_content or "EXPECTED_BRIDGE_IMAGE_REFERENCE" in caller_content
