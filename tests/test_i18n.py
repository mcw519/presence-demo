"""UI internationalization: English and Traditional Chinese string tables have identical keys, and every key the
UI uses (t('…'), data-i18n attributes, setting metadata, module names) exists in both."""

import json
import re
import shutil
import subprocess

import pytest
from sources import STATIC

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
LOCALES = ("en", "zh-TW")


def _tables() -> dict[str, dict[str, str]]:
    js = "const I18N={};" + "".join((STATIC / "i18n" / f"{lang}.js").read_text(encoding="utf-8") for lang in LOCALES)
    out = subprocess.run(["node", "-e", js + "console.log(JSON.stringify(I18N))"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _meta() -> dict:
    src = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")
    start = src.index("const META=")
    meta = src[start : src.index("};", start) + 2]
    mods = src[src.index("const MODULES=") :]
    out = subprocess.run(
        [
            "node",
            "-e",
            "const t=k=>k,ON=()=>0,EQ=()=>0,PS=()=>0,DIST=()=>0;"
            + meta
            + mods[: mods.index("}];") + 3]
            + "console.log(JSON.stringify({META,mods:MODULES.map(m=>m.id)}))",
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_locales_have_identical_keys():
    tabs = _tables()
    assert set(tabs) == set(LOCALES)
    en, zh = set(tabs["en"]), set(tabs["zh-TW"])
    assert en == zh, {"only_en": sorted(en - zh)[:20], "only_zh": sorted(zh - en)[:20]}
    for lang, tab in tabs.items():
        empty = [k for k, v in tab.items() if not str(v).strip()]
        assert not empty, (lang, empty[:10])


def test_placeholders_match_between_locales():
    tabs = _tables()
    ph = re.compile(r"\{(\w+)\}")
    bad = [k for k in tabs["en"] if set(ph.findall(tabs["en"][k])) != set(ph.findall(tabs["zh-TW"][k]))]
    assert not bad, bad[:10]


def test_every_used_key_exists():
    keys = set(_tables()["en"])
    code = "\n".join(p.read_text(encoding="utf-8") for p in (STATIC / "js").glob("*.js"))
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    used = set(re.findall(r"\b[tT]\('([\w.\-]+)'", code))
    used |= set(re.findall(r'data-i18n(?:-[\w-]+)?="([\w.\-]+)"', html))
    used |= set(re.findall(r"'(need\.[\w]+)'", code))  # rule reasons are translated when shown
    used = {k for k in used if not k.endswith(".")}  # prefixes completed at runtime (e.g. t('state.'+s))
    missing = sorted(used - keys)
    assert not missing, missing


def test_settings_and_modules_are_translated():
    keys = set(_tables()["en"])
    m = _meta()
    need = set()
    for k, v in m["META"].items():
        need |= {f"s.{k}", f"s.{k}.h"} | {f"s.{k}.o.{o}" for o in v.get("opts", {})}
        if str(v.get("u", "")).startswith("@"):
            need.add("unit." + v["u"][1:])
    for mid in m["mods"]:
        need |= {f"mod.{mid}", f"mod.{mid}.short"}
    for s in ("ACTIVE", "IDLE", "MAYBE_AWAY", "AWAY", "ENDED"):
        need.add(f"state.{s}")
    assert not sorted(need - keys), sorted(need - keys)[:20]


def test_no_hardcoded_chinese_in_scripts():
    cjk = re.compile(r"[\u3400-\u9fff]")
    for p in (STATIC / "js").glob("*.js"):
        lines = [i + 1 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()) if cjk.search(line)]
        allowed = {"i18n.js"}  # the language switch shows its own name "中文"
        assert p.name in allowed or not lines, (p.name, lines[:5])


def test_language_switch_and_default():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="langSw"' in html and 'data-l="en"' in html and 'data-l="zh-TW"' in html
    i18n = (STATIC / "js" / "i18n.js").read_text(encoding="utf-8")
    assert "localStorage.getItem('presence-lang')" in i18n and "navigator.languages" in i18n
    assert "return 'en';" in i18n  # fallback
    assert html.index("/static/js/i18n.js") < html.index("/static/i18n/en.js") < html.index("/static/js/core.js")
