"""Create an offline real-device acceptance preparation pack; no capture or cloud calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presence_demo.real_acceptance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
