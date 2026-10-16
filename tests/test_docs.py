"""Documentation checks: bilingual mirrors, language-switch links, working relative links and images, and an
up-to-date generated parameter reference."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ZH = DOCS / "zh-TW"
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
NEUTRAL = re.compile(r"kebbi|collibot|tailscale|helio", re.I)


def pages() -> list[Path]:
    root = [ROOT / n for n in ("README.md", "README.zh-TW.md", "CONTRIBUTING.md", "CONTRIBUTING.zh-TW.md")]
    return root + sorted(DOCS.glob("*.md")) + sorted(ZH.glob("*.md"))


def test_every_english_page_has_a_chinese_mirror_and_vice_versa():
    en = {p.name for p in DOCS.glob("*.md")}
    zh = {p.name for p in ZH.glob("*.md")}
    assert en == zh


def test_language_switch_links():
    for p in pages():
        head = "\n".join(p.read_text(encoding="utf-8").splitlines()[:16])
        assert "English" in head and "繁體中文" in head, p


def test_relative_links_and_images_exist():
    missing = []
    for p in pages():
        for target in LINK.findall(p.read_text(encoding="utf-8")):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if path and not (p.parent / path).exists():
                missing.append(f"{p.relative_to(ROOT)} -> {target}")
    assert not missing, missing


def test_docs_are_vendor_neutral():
    hits = [str(p.relative_to(ROOT)) for p in pages() if NEUTRAL.search(p.read_text(encoding="utf-8"))]
    assert not hits, hits


def test_generated_parameter_reference_is_up_to_date():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_config_docs.py"), "--check"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
