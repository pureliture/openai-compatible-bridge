#!/usr/bin/env python3
"""Vendor Snapshot Integrity Verifier for Context Hide.

Verifies that vendor/context-hide matches the recorded provenance.json:
1. Validates provenance.json schema and required fields.
2. Checks pyproject.toml version consistency with provenance metadata.
3. Ensures no unmanifested files exist on disk and no manifest files are missing.
4. Re-computes SHA-256 for all snapshot files and compares against provenance.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

DEFAULT_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "context-hide"
REQUIRED_FIELDS = (
    "schema_version",
    "package_name",
    "package_version",
    "engine_commit",
    "exported_at",
    "files",
)


def verify_snapshot(vendor_dir: Path) -> tuple[bool, str]:
    """Verify integrity of vendor snapshot directory against its provenance.json.

    Returns:
        (True, success_message) on verification success,
        (False, error_message) on any verification failure.
    """
    if not vendor_dir.exists() or not vendor_dir.is_dir():
        return False, f"Vendor directory not found or not a directory: {vendor_dir}"

    provenance_path = vendor_dir / "provenance.json"
    if not provenance_path.is_file():
        return False, f"Missing provenance.json at {provenance_path}"

    try:
        manifest_text = provenance_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"Failed to parse provenance.json: {exc}"

    if not isinstance(manifest, dict):
        return False, "provenance.json root must be a JSON object"

    for field in REQUIRED_FIELDS:
        if field not in manifest:
            return False, f"Missing required field in provenance.json: {field}"

    if manifest.get("schema_version") != "1.0.0":
        return False, f"Unsupported schema_version: {manifest.get('schema_version')}"

    commit = manifest.get("engine_commit")
    if not isinstance(commit, str) or len(commit) != 40 or not all(c in "0123456789abcdefABCDEF" for c in commit):
        return False, f"Invalid engine_commit SHA (expected 40 hex chars): {commit}"

    files_manifest = manifest.get("files")
    if not isinstance(files_manifest, dict):
        return False, "Field 'files' must be a dictionary of relative_path -> sha256"

    # Verify vendor pyproject.toml matches package metadata
    pyproject_path = vendor_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        return False, f"Missing pyproject.toml at {pyproject_path}"

    try:
        pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        pkg_version = pyproject_data.get("project", {}).get("version")
        pkg_name = pyproject_data.get("project", {}).get("name")
        if pkg_version != manifest.get("package_version"):
            return False, (
                f"Version mismatch: pyproject.toml has '{pkg_version}', "
                f"provenance.json has '{manifest.get('package_version')}'"
            )
        if pkg_name != manifest.get("package_name"):
            return False, (
                f"Package name mismatch: pyproject.toml has '{pkg_name}', "
                f"provenance.json has '{manifest.get('package_name')}'"
            )
    except (tomllib.TOMLDecodeError, OSError) as exc:
        return False, f"Failed to parse vendor pyproject.toml: {exc}"

    # Collect files on disk (excluding provenance.json and python cache artifacts)
    disk_files: set[str] = set()
    for path in vendor_dir.rglob("*"):
        if path.is_file():
            rel_posix = path.relative_to(vendor_dir).as_posix()
            if rel_posix == "provenance.json":
                continue
            if "__pycache__" in rel_posix or rel_posix.endswith((".pyc", "/.DS_Store")) or rel_posix == ".DS_Store":
                continue
            disk_files.add(rel_posix)

    manifest_files: set[str] = set(files_manifest.keys())

    # Check for unmanifested files
    unmanifested = disk_files - manifest_files
    if unmanifested:
        return False, f"Unmanifested file found: {', '.join(sorted(unmanifested))}"

    # Check for missing files
    missing = manifest_files - disk_files
    if missing:
        return False, f"Missing file: {', '.join(sorted(missing))}"

    # Re-compute and compare SHA-256 hashes
    for rel_posix in sorted(manifest_files):
        target_file = vendor_dir / rel_posix
        try:
            content = target_file.read_bytes()
        except OSError as exc:
            return False, f"Failed to read file {rel_posix}: {exc}"

        actual_sha256 = hashlib.sha256(content).hexdigest()
        expected_sha256 = files_manifest[rel_posix]
        if actual_sha256 != expected_sha256:
            return False, (
                f"SHA-256 mismatch for {rel_posix}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    success_msg = f"[OK] vendor/context-hide snapshot integrity verified ({len(manifest_files)} files, commit {commit})."
    return True, success_msg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify vendor/context-hide snapshot integrity against provenance.json"
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=DEFAULT_VENDOR_DIR,
        help=f"Target vendor directory (default: {DEFAULT_VENDOR_DIR})",
    )
    args = parser.parse_args()

    ok, message = verify_snapshot(args.vendor_dir)
    if ok:
        print(message)
        return 0
    else:
        print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
