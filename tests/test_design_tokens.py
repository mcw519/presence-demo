"""v0.10.2: the UI uses the PureSound web design system (vendored, pinned) instead of its own token copy."""

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from presence_demo import server

VENDOR = Path(server.STATIC) / "vendor" / "puresound"
# unmodified upstream files at puresound 121bf623b491cb7f18805bec65099543ecc5323f (see vendor/puresound/README.md)
PINNED = {
    "tokens.css": "d788571678a05894d084e8bda54f4445b8d1c3072ff6e1ecf3718c8934c3822b",
    "components.css": "cbb5a2580dd69b3451e7e77f2b2794bf86a4559835aa2addd1cadd87b0a590e3",
}


def test_vendored_tokens_match_pinned_upstream():
    for name, digest in PINNED.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name
    for f in ("LICENSE", "NOTICE", "README.md", "fonts/LICENSE.txt"):
        assert (VENDOR / f).is_file(), f


def test_pin_matches_puresound_dependency():
    rev = "121bf623b491cb7f18805bec65099543ecc5323f"
    assert rev in (VENDOR / "README.md").read_text()
    assert rev in (Path(server.__file__).parents[1] / "pyproject.toml").read_text()


def test_index_uses_shared_tokens_and_no_own_palette():
    html = (Path(server.STATIC) / "index.html").read_text()
    assert "/static/vendor/puresound/tokens.css" in html
    style = html[html.index("<style>") : html.index("</style>")]
    # no redefinition of upstream roles in the app's own CSS
    for role in ("--page:", "--canvas:", "--ink:", "--accent:", "--good:", "--bad:", "--font-sans:"):
        assert role not in style, role


def test_vendor_assets_public_but_app_still_protected(monkeypatch):
    monkeypatch.setattr(server, "TOKEN", "tok")
    c = TestClient(server.app)
    r = c.get("/static/vendor/puresound/tokens.css")
    assert r.status_code == 200 and "--accent" in r.text
    assert c.get("/static/vendor/puresound/fonts/IBMPlexSans-400.woff2").status_code == 200
    assert c.get("/").status_code == 403
    r = c.get("/?token=tok")
    assert r.status_code == 200 and "__VERSION__" not in r.text
    assert c.get("/static/vendor/../index.html").status_code in (403, 404)
