#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check.py <case-name> <output-dir>")

    output_dir = Path(sys.argv[2])
    config_path = output_dir / "dfg.config.json"
    map_path = output_dir / "dfg.map.json"
    if not config_path.exists():
      raise SystemExit("missing dfg.config.json")
    if not map_path.exists():
      raise SystemExit("missing dfg.map.json")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    mapping = json.loads(map_path.read_text(encoding="utf-8"))

    slices = {
        (entry.get("name"), entry.get("kind")): entry
        for entry in config.get("slices", [])
    }

    sw_slice = slices.get(("sw_0", "spatial_sw"))
    if sw_slice is None:
        raise SystemExit("missing spatial_sw slice for sw_0")
    if not sw_slice.get("complete", False):
        raise SystemExit("tagged spatial_sw slice should be complete")

    edge_kinds = {
        entry.get("edge"): entry.get("kind", "unknown")
        for entry in mapping.get("edge_routings", [])
    }
    unrouted = [edge_id for edge_id, kind in edge_kinds.items() if kind == "unrouted"]
    if unrouted:
        raise SystemExit(f"unexpected unrouted edges: {unrouted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
