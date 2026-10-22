"""Run camera-free frontend scheduling and drawing regressions under Node."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frontend_scheduling_and_drawing():
    script = Path(__file__).parent / "js" / "frontend_perf.js"
    out = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
