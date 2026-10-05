"""Tests for Context Hide vendor snapshot and verification tools.

Covers:
1. Active snapshot integrity verification (exit 0, expected manifest & commit).
2. Provenance metadata validation (schema, versions, commit SHA, relative paths).
3. Tamper detection on file modification (exit 1, SHA-256 mismatch).
4. Unmanifested file detection when unexpected files are introduced (exit 1).
5. Missing file detection when manifest files are removed (exit 1).
6. Corrupt provenance / version mismatch error handling.
7. Exporter dirty repository detection and clean export validation.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_vendor_context_hide.py"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_vendor_context_hide.py"
VENDOR_DIR = REPO_ROOT / "vendor" / "context-hide"
ENGINE_REPO = Path("/Users/ddalkak/Projects/context-hide")


def run_verifier(vendor_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--vendor-dir", str(vendor_dir)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_active_vendor_snapshot_integrity_passes():
    """Verify that the repository's vendor/context-hide snapshot is valid and passes verification."""
    assert VENDOR_DIR.is_dir(), f"Vendor directory does not exist: {VENDOR_DIR}"
    assert (VENDOR_DIR / "provenance.json").is_file()

    result = run_verifier(VENDOR_DIR)
    assert result.returncode == 0, f"Verifier failed with stderr: {result.stderr}"
    assert "[OK] vendor/context-hide snapshot integrity verified" in result.stdout
    assert "14 files" in result.stdout
    assert "commit a1ed62b8936ad20f89fe132a87b070055453e764" in result.stdout


def test_provenance_json_schema_and_integrity():
    """Validate provenance.json fields and cross-check disk files."""
    provenance_path = VENDOR_DIR / "provenance.json"
    data = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert data["schema_version"] == "1.0.0"
    assert data["package_name"] == "context-hide"
    assert data["package_version"] == "0.1.0"
    assert len(data["engine_commit"]) == 40
    assert "T" in data["exported_at"] and data["exported_at"].endswith("Z")

    files = data["files"]
    assert isinstance(files, dict)
    assert len(files) == 14

    expected_allowlist_keys = {
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "contracts/errors.json",
        "contracts/replacement_plan.json",
        "contracts/tool_result_record.json",
        "src/context_hide/__init__.py",
        "src/context_hide/engine.py",
        "src/context_hide/model.py",
        "src/context_hide/policy.py",
        "src/context_hide/py.typed",
        "src/context_hide/store.py",
        "src/context_hide/summary.py",
        "src/context_hide/transport.py",
    }
    assert set(files.keys()) == expected_allowlist_keys

    # Ensure no absolute paths or test/cache files
    for rel_path in files:
        assert not rel_path.startswith("/"), f"Path must be relative: {rel_path}"
        assert not rel_path.startswith("\\"), f"Path must be relative: {rel_path}"
        assert "tests" not in rel_path.split("/")
        assert "__pycache__" not in rel_path


def test_verifier_detects_file_tampering(tmp_path: Path):
    """File content modification must cause verification failure with exit code 1."""
    test_vendor = tmp_path / "context-hide"
    shutil.copytree(VENDOR_DIR, test_vendor)

    target_file = test_vendor / "src" / "context_hide" / "model.py"
    target_file.write_text(target_file.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    result = run_verifier(test_vendor)
    assert result.returncode == 1
    assert "SHA-256 mismatch for src/context_hide/model.py" in result.stderr


def test_verifier_detects_unmanifested_file(tmp_path: Path):
    """Adding an unmanifested file must cause verification failure with exit code 1."""
    test_vendor = tmp_path / "context-hide"
    shutil.copytree(VENDOR_DIR, test_vendor)

    extra_file = test_vendor / "src" / "context_hide" / "unmanifested.py"
    extra_file.write_text("# untracked file\n", encoding="utf-8")

    result = run_verifier(test_vendor)
    assert result.returncode == 1
    assert "Unmanifested file found: src/context_hide/unmanifested.py" in result.stderr


def test_verifier_detects_missing_file(tmp_path: Path):
    """Removing a manifested file must cause verification failure with exit code 1."""
    test_vendor = tmp_path / "context-hide"
    shutil.copytree(VENDOR_DIR, test_vendor)

    (test_vendor / "LICENSE").unlink()

    result = run_verifier(test_vendor)
    assert result.returncode == 1
    assert "Missing file: LICENSE" in result.stderr


def test_verifier_detects_corrupt_provenance(tmp_path: Path):
    """Corrupted provenance JSON must cause verification failure with exit code 1."""
    test_vendor = tmp_path / "context-hide"
    shutil.copytree(VENDOR_DIR, test_vendor)

    (test_vendor / "provenance.json").write_text("{invalid json", encoding="utf-8")

    result = run_verifier(test_vendor)
    assert result.returncode == 1
    assert "Failed to parse provenance.json" in result.stderr


def test_verifier_detects_version_mismatch(tmp_path: Path):
    """Mismatch between pyproject.toml and provenance.json package_version must fail."""
    test_vendor = tmp_path / "context-hide"
    shutil.copytree(VENDOR_DIR, test_vendor)

    pyproject_file = test_vendor / "pyproject.toml"
    content = pyproject_file.read_text(encoding="utf-8").replace('version = "0.1.0"', 'version = "0.2.0"')
    pyproject_file.write_text(content, encoding="utf-8")

    result = run_verifier(test_vendor)
    assert result.returncode == 1
    assert "Version mismatch" in result.stderr


def test_exporter_rejects_dirty_repository(tmp_path: Path):
    """Exporter must reject dirty repository with error message and exit code 1."""
    fake_engine = tmp_path / "fake-engine"
    fake_engine.mkdir()

    # Initialize a git repository with an initial commit
    subprocess.run(["git", "init", str(fake_engine)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(fake_engine), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(fake_engine), "config", "user.name", "Test"], check=True)

    dummy_file = fake_engine / "dummy.txt"
    dummy_file.write_text("hello", encoding="utf-8")
    subprocess.run(["git", "-C", str(fake_engine), "add", "dummy.txt"], check=True)
    subprocess.run(["git", "-C", str(fake_engine), "commit", "-m", "init"], check=True)

    # Make working tree dirty
    dummy_file.write_text("hello dirty", encoding="utf-8")

    fake_vendor = tmp_path / "fake-vendor"
    proc = subprocess.run(
        [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--engine-dir",
            str(fake_engine),
            "--vendor-dir",
            str(fake_vendor),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "Dirty engine repository. Clean commit required." in proc.stderr


def test_exporter_runs_clean_and_verifies_successfully(tmp_path: Path):
    """Exporter exports clean context-hide and successfully performs verification."""
    dest_vendor = tmp_path / "exported-vendor"

    proc = subprocess.run(
        [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--engine-dir",
            str(ENGINE_REPO),
            "--vendor-dir",
            str(dest_vendor),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, f"Export failed with stderr:\n{proc.stderr}"
    assert "Exported 14 files" in proc.stdout
    assert "[OK] vendor/context-hide snapshot integrity verified" in proc.stdout

    # Independently verify with the verifier script
    verify_res = run_verifier(dest_vendor)
    assert verify_res.returncode == 0
    assert "[OK] vendor/context-hide snapshot integrity verified" in verify_res.stdout
