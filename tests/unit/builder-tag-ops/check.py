#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))

    kinds = [entry["kind"] for entry in config["slices"]]
    if "add_tag" not in kinds:
        raise SystemExit("missing add_tag slice")
    if "map_tag" not in kinds:
        raise SystemExit("missing map_tag slice")
    port_table = {port["id"]: port for port in mapping.get("port_table", [])}
    route_kinds = set()
    for edge in mapping.get("edge_routings", []):
        path = edge.get("path", [])
        if not path:
            continue
        route_kinds.add(
            tuple(port_table[port_id]["kind"] for port_id in path if port_id in port_table)
        )

    if not any(
        "add_tag" in kinds and "map_tag" in kinds and "del_tag" in kinds
        for kinds in route_kinds
    ):
        raise SystemExit("expected routed path through add_tag/map_tag/del_tag")

    return 0


if __name__ == "__main__":
    sys.exit(main())
