#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    case_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))

    node_mappings = [entry for entry in mapping["node_mappings"] if entry["sw_op"] == "arith.muli"]
    if len(node_mappings) != 1:
        raise SystemExit("expected one arith.muli node mapping")
    if node_mappings[0].get("pe_name") != "pe_1_1":
        raise SystemExit("expected arith.muli to map to the customized cell pe_1_1")

    return 0


if __name__ == "__main__":
    sys.exit(main())
