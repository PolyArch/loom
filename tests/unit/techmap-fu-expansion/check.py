#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def find_map_json(out_dir: Path) -> Path:
    """Find the mapping JSON file in the output directory."""
    candidates = sorted(out_dir.glob("*.map.json"))
    if not candidates:
        raise SystemExit("no *.map.json found in " + str(out_dir))
    return candidates[0]


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    mapping_path = find_map_json(out_dir)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    node_mappings = {entry["sw_op"]: entry for entry in mapping["node_mappings"]}

    # Verify that the OpCompat alias pair count is at least 3.
    techmap = mapping.get("techmap", {})
    op_alias_pairs = techmap.get("op_alias_pair_count", 0)
    if op_alias_pairs < 3:
        raise SystemExit(
            "expected at least 3 op alias pairs, got: " + str(op_alias_pairs)
        )

    # Verify full coverage.
    coverage = techmap.get("coverage_score", 0.0)
    if coverage < 1.0:
        raise SystemExit(
            "expected coverage_score 1.0, got: " + str(coverage)
        )

    # If the DFG has a MAC pattern (muli + addi), verify they map to fu_mac.
    mul = node_mappings.get("arith.muli")
    add = node_mappings.get("arith.addi")
    if mul is not None and add is not None:
        if mul.get("hw_name") != "fu_mac" or add.get("hw_name") != "fu_mac":
            raise SystemExit(
                "expected both software ops to map onto fu_mac, got: "
                + str(mul.get("hw_name")) + " and " + str(add.get("hw_name"))
            )

    # If the DFG has a compare-select pattern (cmpi + select), verify they
    # fuse onto the same FU.
    cmpi = node_mappings.get("arith.cmpi")
    sel = node_mappings.get("arith.select")
    if cmpi is not None and sel is not None:
        if cmpi.get("hw_name") != sel.get("hw_name"):
            raise SystemExit(
                "expected cmpi and select to map onto the same FU, got: "
                + str(cmpi.get("hw_name")) + " and " + str(sel.get("hw_name"))
            )
        if cmpi.get("hw_name") != "fu_cmp_sel":
            raise SystemExit(
                "expected cmpi/select to map onto fu_cmp_sel, got: "
                + str(cmpi.get("hw_name"))
            )

    # Verify no unmapped fallback nodes.
    fallback_no_candidate = techmap.get("fallback_no_candidate_count", 0)
    if fallback_no_candidate != 0:
        raise SystemExit(
            "expected 0 unmapped fallback nodes, got: "
            + str(fallback_no_candidate)
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
