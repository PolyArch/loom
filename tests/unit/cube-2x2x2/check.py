#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])

    fabric_path = out_dir / "cube-2x2x2.fabric.mlir"
    text = fabric_path.read_text(encoding="utf-8")
    for marker in ("@__cube_sw_6x4_", "@__cube_sw_4x5_", "@__cube_sw_14x14_"):
        if marker not in text:
            raise SystemExit(f"missing generated switch template {marker}")

    sidecar = json.loads(
        (out_dir / "cube-2x2x2.fabric.viz.json").read_text(encoding="utf-8")
    )
    counts = {}
    for comp in sidecar["components"]:
        counts[comp["kind"]] = counts.get(comp["kind"], 0) + 1
    if counts.get("spatial_pe") != 8:
        raise SystemExit("expected 8 spatial_pe instances in cube sidecar")
    if counts.get("spatial_sw") != 27:
        raise SystemExit("expected 27 spatial_sw instances in cube sidecar")

    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))
    ops = {entry["sw_op"] for entry in mapping["node_mappings"]}
    if "arith.addi" not in ops:
        raise SystemExit("expected arith.addi to be mapped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
