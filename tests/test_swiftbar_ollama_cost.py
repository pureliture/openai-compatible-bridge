from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_swiftbar_ledger_fallback_keeps_unlimited_limit_as_infinity(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "swiftbar" / "ollama-cost.1m.py"
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_BRIDGE_COST_STATUS_URL": "http://127.0.0.1:9/admin/cost/status",
            "OPENAI_BRIDGE_COST_LEDGER_PATH": str(tmp_path / "missing-cost-ledger.db"),
            "COST_DAILY_LIMIT_USD": "unlimited",
        }
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines()[0] == "Ollama $0 / ∞"
