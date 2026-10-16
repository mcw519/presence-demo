"""v0.11.1: tuned defaults, presets, diagnostics (session_open, track lifecycle, face misses, client reports,
host_suspend), normal close on client disconnect."""

import json
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from presence_demo import diag
from presence_demo.tracker import Config

ROOT = Path(__file__).parents[1] / "presence_demo"
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
CLIENT_ONLY = {"fps", "vad_k", "vad_min", "vad_hang"}  # browser-side send rate / VAD
SRV = (ROOT / "server.py").read_text(encoding="utf-8")


def test_tuned_defaults():
    c = Config()
    assert (c.min_conf, c.leave_halflife, c.force_detect_seconds) == (0.4, 1.5, 2.0)
    assert (c.reid_decision_timeout, c.reid_face_wait) == (10.0, 3.0)
    for k, v in (("min_conf", 0.4), ("leave_halflife", 1.5), ("force_detect_seconds", 2), ("reid_face_wait", 3)):
        assert re.search(rf"[,{{\s]{k}:{v}[,}}]", HTML), k  # client defaults agree


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_presets_valid_and_recommended_matches_field_run():
    src = HTML[HTML.index("const REC=") : HTML.index("const KEY=")]
    mod = HTML[HTML.index("const MOD_PRESETS=") :]
    mod = mod[: mod.index("};") + 2]
    js = src + mod + "console.log(JSON.stringify({REC,PRESETS,MOD_PRESETS}))"
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    rec = out["MOD_PRESETS"]["recommended"]
    assert rec == out["REC"]
    want = dict(
        tracker_mode="kalman",
        presence_model="hmm",
        intent_model="score",
        pose_enabled=True,
        distance_enabled=True,
        audio_pipeline="puresound",
        near_range=1.6,
        min_conf=0.4,
        leave_halflife=1.5,
    )
    assert {k: rec[k] for k in want} == want
    allp = {**out["PRESETS"], **out["MOD_PRESETS"]}
    assert set(allp) >= {"recommended", "lowcam", "single", "group", "noisy", "lowpower"}
    for name, p in allp.items():
        c = Config()
        p = {k: v for k, v in p.items() if k not in CLIENT_ONLY}
        warns = c.update(p)
        assert not warns, (name, warns)
        for k, v in p.items():  # in range: nothing was clamped
            got = asdict(c)[k]
            assert (list(got) if isinstance(got, tuple) else got) == pytest.approx(v), (name, k)
    for name in ("lowcam", "single", "group", "noisy"):
        assert out["PRESETS"][name]["tracker_mode"] == "kalman"  # scenario presets build on the recommended one
    for b in ("recommended", "lowcam", "lowpower", "single", "group", "noisy"):
        assert f'data-p="{b}"' in HTML


def test_diagnostic_events_present():
    for ev in (
        '"session_open"',
        '"track_new"',
        '"track_lost"',
        '"track_recovered"',
        '"track_dropped"',
        '"face_miss_summary"',
        '"asr_skip"',
        '"host_suspend"',
        '"client"',
        '"sender_closed"',
    ):
        assert ev in SRV, ev
    assert 'reason=tr_ev[-1].get("reason")' in SRV
    for what in ("'visibility'", "'autopause'", "'media_ended'", "'reconnect'", "'stale_socket'", "'client_suspend'"):
        assert what in HTML, what
    assert "performance.now()-lastRx>8000" in HTML  # half-open socket is closed and reconnected


def test_client_report_and_normal_close_over_ws():
    from fastapi.testclient import TestClient

    from presence_demo import server

    q = f"?token={server.TOKEN}" if server.TOKEN else ""
    with TestClient(server.app).websocket_connect("/ws" + q) as ws:
        ws.send_json({"type": "client", "what": "visibility", "hidden": True, "level": "x", "big": "y" * 500})
        ws.send_json({"type": "ping", "t": 1})
        for _ in range(20):
            if ws.receive_json().get("type") == "pong":
                break
    ev = [e for e in diag.recent(200) if e["event"] == "client"]
    assert ev and ev[-1]["what"] == "visibility" and ev[-1]["level"] == "info" and len(ev[-1]["big"]) <= 64
    crashes = [e for e in diag.recent(50) if e["event"] == "task_crash" and e.get("task") == "sender"]
    assert not crashes
