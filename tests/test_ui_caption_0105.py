"""UI v0.10.5: audio panel under the camera; live ASR caption overlay."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1] / "presence_demo" / "static"
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "app.css").read_text(encoding="utf-8")


def test_audio_panel_is_in_left_column_below_camera():
    main = HTML.index('class="workbench-main presence-stack"')
    view, audio, right = HTML.index('id="view"'), HTML.index('id="aCard"'), HTML.index('<div class="presence-stack">')
    assert main < view < audio < right


def test_caption_overlay_outside_mirrored_layers():
    view = re.search(r'<div id="view">(.*?)</div>\s*</div>\s*<div class="panel panel-pad" id="aCard"', HTML, re.S).group(1)
    assert 'id="cap"' in view and "hidden" in view
    assert "#view video.mirror" in CSS and ".caption" in CSS and "scaleX" not in CSS[CSS.index("v0.10.5") :]


def test_caption_logic():
    assert "caption(m);" in HTML  # every partial/final updates in place
    assert "m.kind==='final'?CAP_HOLD_MS" in HTML  # fade out after final
    assert "if(cfg.asr_provider==='none')captionOff();" in HTML  # hidden when ASR off
    assert "textContent=m.text" in HTML  # no HTML injection from transcripts
