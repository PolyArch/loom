#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    case_name = sys.argv[1]
    out_dir = Path(sys.argv[2])

    fabric_path = out_dir / "lattice-3x3.fabric.mlir"
    text = fabric_path.read_text(encoding="utf-8")
    for marker in ("@__lattice_sw_6x6_", "@__lattice_sw_7x7_", "@__lattice_sw_8x8_"):
        if marker not in text:
            raise SystemExit(f"missing generated switch template {marker}")

    sidecar = json.loads((out_dir / "lattice-3x3.fabric.viz.json").read_text(encoding="utf-8"))
    counts = {}
    for comp in sidecar["components"]:
        counts[comp["kind"]] = counts.get(comp["kind"], 0) + 1
    if counts.get("spatial_pe") != 9:
        raise SystemExit("expected 9 spatial_pe instances in lattice sidecar")
    if counts.get("spatial_sw") != 9:
        raise SystemExit("expected 9 spatial_sw instances in lattice sidecar")

    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))
    ops = {entry["sw_op"] for entry in mapping["node_mappings"]}
    if "arith.addi" not in ops:
        raise SystemExit("expected arith.addi to be mapped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
