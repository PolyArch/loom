#!/usr/bin/env python3

from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check.py <case-name> <output-dir>")

    output_dir = Path(sys.argv[2])
    fabric_file = output_dir / "builder-index-width.fabric.mlir"
    map_file = output_dir / "dfg.map.txt"

    fabric_text = fabric_file.read_text(encoding="utf-8")
    if "!fabric.tagged<!fabric.bits<32>, i1>" not in fabric_text:
        raise SystemExit("expected default index width probe to use bits<32>")
    if "fabric.add_tag" not in fabric_text or "fabric.del_tag" not in fabric_text:
        raise SystemExit("expected index width probe tag chain in generated fabric")

    map_text = map_file.read_text(encoding="utf-8")
    if map_text.count("arith.addi") < 2:
        raise SystemExit("expected two arith.addi mappings")
    if "UNROUTED" in map_text:
        raise SystemExit("unexpected unrouted edge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
