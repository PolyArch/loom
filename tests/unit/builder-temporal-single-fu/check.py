#!/usr/bin/env python3

from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check.py <case-name> <output-dir>", file=sys.stderr)
        return 2

    case_name = sys.argv[1]
    output_dir = Path(sys.argv[2])
    fabric_file = output_dir / "builder-temporal-single-fu.fabric.mlir"
    map_file = output_dir / "dfg.map.txt"

    text = fabric_file.read_text(encoding="utf-8")
    if "fabric.temporal_pe @add_tpe" not in text:
        print(f"{case_name}: expected temporal PE definition", file=sys.stderr)
        return 1
    if "num_instruction = 1 : i64" not in text:
        print(f"{case_name}: expected temporal PE instruction count", file=sys.stderr)
        return 1

    map_text = map_file.read_text(encoding="utf-8")
    if "UNROUTED" in map_text:
        print(f"{case_name}: unexpected unrouted edge", file=sys.stderr)
        return 1
    if "arith.addi" not in map_text:
        print(f"{case_name}: expected addi mapping", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
