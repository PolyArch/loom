"""Shared pytest configuration for DSE tests.

Ensures the repository root is on sys.path so that
`from scripts.dse.* import ...` works regardless of the
current working directory.
"""

import sys
from pathlib import Path


def _repo_root() -> Path:
    """Walk up from this file to find the repository root."""
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts" / "dse").exists():
            return p
        p = p.parent
    raise RuntimeError("Cannot locate repository root from conftest.py")


_ROOT = str(_repo_root())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
