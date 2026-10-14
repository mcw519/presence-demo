"""Alias of eval_tracking.py (tracking + presence-model comparison)."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("eval_tracking.py")), run_name="__main__")
