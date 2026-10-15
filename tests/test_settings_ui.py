"""Settings UI (v0.10.0): every config key belongs to a module; server config keys and client defaults agree."""

import re
from dataclasses import fields
from pathlib import Path

from presence_demo.tracker import Config

HTML = (Path(__file__).parents[1] / "presence_demo" / "static" / "index.html").read_text(encoding="utf-8")


def _def_keys():
    body = re.search(r"const DEF=\{(.*?)\};", HTML, re.S).group(1)
    return set(re.findall(r"(?:^|[,{\s])([a-z_0-9]+):", body))


def test_every_setting_is_in_a_module():
    mods = HTML[HTML.index("const MODULES=[") : HTML.index("const MOD_PRESETS")]
    placed = set(re.findall(r"'([a-z_0-9]+)'", mods))
    missing = {k for k in _def_keys() if k != "monitor" and k not in placed}
    assert not missing, missing


def test_client_defaults_cover_server_config():
    client_only = {"monitor", "mirror_view", "fps", "vad_k", "vad_min", "vad_hang", "show_face_roi", "show_pose_roi"}
    server = {f.name for f in fields(Config)}
    ui = _def_keys() - client_only
    assert ui <= server, ui - server
