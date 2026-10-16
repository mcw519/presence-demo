"""Monitor player jitter buffer (AudioWorklet code run under node): bursts must not become permanent delay."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from sources import ui_source

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_player_buffer_stays_bounded(tmp_path):
    html = ui_source()
    src = re.search(r"const WORKLET=`(.*?)`;", html, re.S).group(1)
    w = tmp_path / "worklet.js"  # JS source only (no audio)
    w.write_text(src, encoding="utf-8")
    out = subprocess.run(["node", str(ROOT / "tests" / "js" / "player_sim.js"), str(w)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    m = re.search(r"max buffer ms (\d+) stats (\{.*\})", out.stdout)
    assert m, out.stdout
    assert int(m.group(1)) <= 170  # old player kept 400+ ms after one 400 ms stall
    assert json.loads(m.group(2))["drop_ms"] > 0
