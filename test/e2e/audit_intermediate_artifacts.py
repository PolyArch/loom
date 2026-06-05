#!/usr/bin/env python3
"""CLI wrapper for deterministic intermediate artifact content audits."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(intermediate_artifacts.main(["audit", *sys.argv[1:]]))
