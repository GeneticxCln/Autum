"""WP1 voice-pipeline benchmark. See docs/m1-benchmark-methodology.md."""

import sys
from pathlib import Path

# The benchmark speaks replies with Autum's own chunker (src/autum/text.py).
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
