#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))

    node_mappings = {entry["sw_op"]: entry for entry in mapping["node_mappings"]}
    mul = node_mappings.get("arith.muli")
    add = node_mappings.get("arith.addi")
    if mul is None or add is None:
        raise SystemExit("expected both arith.muli and arith.addi to map")
    if mul.get("hw_name") != "fu_mac" or add.get("hw_name") != "fu_mac":
        raise SystemExit("expected both software ops to map onto raw-body fu_mac")

    return 0


if __name__ == "__main__":
    sys.exit(main())
