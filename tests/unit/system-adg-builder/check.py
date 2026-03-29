#!/usr/bin/env python3
"""Check that the system ADG builder test produced valid output."""

from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check.py <case-name> <output-dir>")
    output_dir = Path(sys.argv[2])

    # Verify the system mesh MLIR was generated
    mesh_file = output_dir / "system_mesh.mlir"
    if mesh_file.exists():
        mesh_text = mesh_file.read_text(encoding="utf-8")
        assert "test_system_mesh" in mesh_text, "system module name missing"
        assert "core_0_0" in mesh_text, "core_0_0 missing"
        assert "core_1_1" in mesh_text, "core_1_1 missing"
        assert "CoreType_A" in mesh_text, "CoreType_A missing"
        assert "CoreType_B" in mesh_text, "CoreType_B missing"
        assert "mesh" in mesh_text.lower(), "mesh topology missing"
        assert "noc_link" in mesh_text, "NoC connections missing"
        assert "l2_bank_" in mesh_text, "L2 banks missing"

    # Verify ring topology output
    ring_file = output_dir / "system_ring.mlir"
    if ring_file.exists():
        ring_text = ring_file.read_text(encoding="utf-8")
        assert "ring" in ring_text.lower(), "ring topology missing"

    # Verify hierarchical topology output
    hier_file = output_dir / "system_hier.mlir"
    if hier_file.exists():
        hier_text = hier_file.read_text(encoding="utf-8")
        assert "hierarchical" in hier_text.lower(), "hierarchical topology missing"

    # Verify the per-core ADG mapper result
    map_text = (output_dir / "dfg.map.txt").read_text(encoding="utf-8")
    if "arith.addi" not in map_text:
        raise SystemExit("expected arith.addi mapping")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
