"""Make the `src/` modules importable as bare names (e.g. `import env`),
matching how they're imported when run directly as scripts (`python src/train.py`).
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
