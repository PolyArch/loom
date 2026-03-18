#!/usr/bin/env python3

from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check.py <case-name> <output-dir>")
    output_dir = Path(sys.argv[2])
    map_text = (output_dir / "dfg.map.txt").read_text(encoding="utf-8")
    if map_text.count("arith.addi") < 2:
        raise SystemExit("expected two arith.addi mappings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
