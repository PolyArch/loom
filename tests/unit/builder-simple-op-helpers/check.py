#!/usr/bin/env python3

from pathlib import Path
import sys


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing expected text: {needle}")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check.py <case-name> <output-dir>")

    output_dir = Path(sys.argv[2])
    map_text = (output_dir / "dfg.map.txt").read_text(encoding="utf-8")

    require(map_text, "arith.trunci")
    require(map_text, "arith.addi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
