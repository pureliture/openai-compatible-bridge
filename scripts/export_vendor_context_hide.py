#!/usr/bin/env python3
"""Vendor Snapshot Exporter for Context Hide.

Exports a clean commit of context-hide into vendor/context-hide:
1. Validates that the engine repository is clean (git status --porcelain must be empty).
2. Obtains HEAD commit SHA (40 chars) and package metadata from pyproject.toml.
3. Copies allowlisted files (pyproject.toml, LICENSE, README.md, src/context_hide/**/*.py,
   src/context_hide/py.typed, contracts/*.json) into vendor/context-hide.
4. Strictly excludes .git, tests, __pycache__, and unnecessary files.
5. Computes SHA-256 for all exported files and generates vendor/context-hide/provenance.json.
6. Invokes scripts/verify_vendor_context_hide.py to verify snapshot integrity immediately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ENGINE_DIR = Path(
    os.environ.get("CONTEXT_HIDE_ENGINE_DIR", str(Path.home() / "Projects" / "context-hide"))
)
DEFAULT_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "context-hide"
VERIFIER_SCRIPT = Path(__file__).resolve().parent / "verify_vendor_context_hide.py"


def check_git_clean(engine_dir: Path) -> str:
    """Ensure engine git repository is clean and return HEAD commit SHA (40 chars)."""
    # Check if git repo
    is_git = subprocess.run(
        ["git", "-C", str(engine_dir), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if is_git.returncode != 0:
        raise RuntimeError(f"Engine path is not a git worktree: {engine_dir}")

    # Check for dirty working tree
    status_proc = subprocess.run(
        ["git", "-C", str(engine_dir), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status_proc.returncode != 0:
        raise RuntimeError(f"Failed to check git status in {engine_dir}")

    if status_proc.stdout.strip():
        print("Dirty engine repository. Clean commit required.", file=sys.stderr)
        sys.exit(1)

    # Get HEAD commit SHA
    head_proc = subprocess.run(
        ["git", "-C", str(engine_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head_proc.returncode != 0:
        raise RuntimeError(f"Failed to resolve HEAD commit in {engine_dir}")

    commit_sha = head_proc.stdout.strip()
    if len(commit_sha) != 40:
        raise ValueError(f"Invalid commit SHA retrieved: '{commit_sha}'")

    return commit_sha


def collect_allowlist_files(engine_dir: Path) -> list[Path]:
    """Collect allowlist relative paths from engine_dir."""
    allowed_rel_paths: list[Path] = []

    # 1. Root files
    for root_file in ["pyproject.toml", "LICENSE", "README.md"]:
        p = engine_dir / root_file
        if not p.is_file():
            raise FileNotFoundError(f"Required engine file not found: {p}")
        allowed_rel_paths.append(Path(root_file))

    # 2. Source files
    src_dir = engine_dir / "src" / "context_hide"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Engine source directory not found: {src_dir}")

    for p in sorted(src_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(engine_dir)
        if "__pycache__" in rel.parts or rel.name.endswith(".pyc"):
            continue
        if rel.name.endswith(".py") or rel.name == "py.typed":
            allowed_rel_paths.append(rel)

    # 3. Contract files
    contracts_dir = engine_dir / "contracts"
    if contracts_dir.is_dir():
        for p in sorted(contracts_dir.glob("*.json")):
            if p.is_file():
                allowed_rel_paths.append(p.relative_to(engine_dir))

    return sorted(allowed_rel_paths, key=lambda p: p.as_posix())


def export_vendor_snapshot(engine_dir: Path, vendor_dir: Path) -> dict:
    """Export clean context-hide snapshot and generate provenance.json."""
    commit_sha = check_git_clean(engine_dir)

    # Parse pyproject.toml
    pyproject_file = engine_dir / "pyproject.toml"
    if not pyproject_file.is_file():
        raise FileNotFoundError(f"Missing pyproject.toml at {pyproject_file}")

    pyproject_data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
    pkg_name = pyproject_data.get("project", {}).get("name", "context-hide")
    pkg_version = pyproject_data.get("project", {}).get("version")
    if not pkg_version:
        raise ValueError("Could not determine package version from pyproject.toml")

    allowlist_rel_paths = collect_allowlist_files(engine_dir)

    # Clean existing vendor directory to prevent stale files
    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    vendor_dir.mkdir(parents=True, exist_ok=True)

    # Copy allowlisted files
    files_manifest: dict[str, str] = {}
    for rel_path in allowlist_rel_paths:
        src = engine_dir / rel_path
        dst = vendor_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        content = dst.read_bytes()
        sha256_hash = hashlib.sha256(content).hexdigest()
        files_manifest[rel_path.as_posix()] = sha256_hash

    # Generate provenance.json
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": "1.0.0",
        "package_name": pkg_name,
        "package_version": pkg_version,
        "engine_commit": commit_sha,
        "exported_at": now_utc,
        "files": files_manifest,
    }

    provenance_path = vendor_dir / "provenance.json"
    provenance_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(files_manifest)} files to {vendor_dir} (commit: {commit_sha[:7]})")

    # Immediate verification
    verify_proc = subprocess.run(
        [sys.executable, str(VERIFIER_SCRIPT), "--vendor-dir", str(vendor_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify_proc.returncode != 0:
        print(f"Post-export verification failed:\n{verify_proc.stderr}", file=sys.stderr)
        sys.exit(1)
    else:
        print(verify_proc.stdout.strip())

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export clean commit snapshot of context-hide to vendor/context-hide"
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=DEFAULT_ENGINE_DIR,
        help=f"Path to context-hide repository (default: {DEFAULT_ENGINE_DIR})",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=DEFAULT_VENDOR_DIR,
        help=f"Destination vendor directory (default: {DEFAULT_VENDOR_DIR})",
    )
    args = parser.parse_args()

    export_vendor_snapshot(args.engine_dir.resolve(), args.vendor_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
