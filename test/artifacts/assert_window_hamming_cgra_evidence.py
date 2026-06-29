#!/usr/bin/env python3
"""Compatibility wrapper for window_hamming signal-window evidence checks."""

from __future__ import annotations

import sys

from assert_signal_window_cgra_evidence import main


if __name__ == "__main__":
    raise SystemExit(main([sys.argv[0], "--case", "window_hamming", *sys.argv[1:]]))
