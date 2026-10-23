"""Passive browser recorder contracts with a fake clock and WebSocket; no devices."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_passive_numeric_recorder():
    script = Path(__file__).parent / "js" / "real_acceptance_recorder.js"
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
