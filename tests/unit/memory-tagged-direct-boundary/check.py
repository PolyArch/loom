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
    html = (out_dir / "dfg.viz.html").read_text(encoding="utf-8")

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    for key in (("tpe_0", "temporal_pe"), ("sw_0", "spatial_sw"),
                ("tsw_0", "temporal_sw"), ("tsw_1", "temporal_sw")):
        entry = slices.get(key)
        if entry is None:
            raise SystemExit(f"missing slice {key}")
        if not entry["complete"]:
            raise SystemExit(f"incomplete slice {key}")

    unrouted = [entry for entry in mapping["edge_routings"] if entry["kind"] == "unrouted"]
    if unrouted:
        raise SystemExit(f"unexpected unrouted edges: {unrouted}")

    port_info = {entry["id"]: entry for entry in mapping["port_table"]}
    saw_direct_ingress = False
    saw_direct_egress = False
    for entry in mapping["edge_routings"]:
        if entry["kind"] != "routed":
            continue
        path_ports = [port_info[pid] for pid in entry.get("path", []) if pid in port_info]
        path_kinds = [port["kind"] for port in path_ports]
        if ("fu" in path_kinds and "sw" in path_kinds and "memory" in path_kinds and
                "add_tag" not in path_kinds):
            if any(port["kind"] == "sw" and port.get("name") == "sw_0" for port in path_ports):
                saw_direct_ingress = True
        if ("memory" in path_kinds and "temporal_sw" in path_kinds and "fu" in path_kinds and
                "del_tag" not in path_kinds):
            if any(port["kind"] == "temporal_sw" and port.get("name") == "tsw_0"
                   for port in path_ports):
                saw_direct_egress = True

    if not saw_direct_ingress:
        raise SystemExit("expected ingress path to terminate at tagged temporal_pe boundary")
    if not saw_direct_egress:
        raise SystemExit("expected egress path to terminate at tagged temporal_pe boundary")

    if '"kind": "temporal_pe"' not in html:
        raise SystemExit("viz html missing temporal_pe payload")
    if html.count('"kind": "temporal_sw"') < 2 or '"kind": "spatial_sw"' not in html:
        raise SystemExit("viz html missing expected switch payloads")

    return 0


if __name__ == "__main__":
    sys.exit(main())
