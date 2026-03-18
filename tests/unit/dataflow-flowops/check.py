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
    for pe_name in ("pe_stream", "pe_gate", "pe_carry"):
        pe_slice = slices.get((pe_name, "spatial_pe"))
        if pe_slice is None:
            raise SystemExit(f"missing spatial_pe slice for {pe_name}")
        if not pe_slice["complete"]:
            raise SystemExit(f"slice for {pe_name} marked incomplete")

    expected_nodes = {
        "dataflow.stream": "fu_stream",
        "dataflow.gate": "fu_gate",
        "dataflow.carry": "fu_carry",
    }
    node_mappings = {entry["sw_op"]: entry["hw_name"] for entry in mapping["node_mappings"]}
    for sw_op, hw_name in expected_nodes.items():
        if node_mappings.get(sw_op) != hw_name:
            raise SystemExit(f"{sw_op} should map to {hw_name}, got {node_mappings}")

    port_info = {entry["id"]: entry for entry in mapping["port_table"]}
    routed = [entry for entry in mapping["edge_routings"] if entry["kind"] == "routed"]
    if len(routed) < 3:
        raise SystemExit(f"expected routed edges for flowops case, got {routed}")

    saw_stream_to_gate = False
    saw_gate_to_carry = False
    for entry in routed:
        path = entry.get("path", [])
        pe_names = []
        for port_id in path:
            info = port_info.get(port_id)
            if not info:
                continue
            if info.get("kind") == "fu":
                pe_names.append(info.get("pe"))
        if "pe_stream" in pe_names and "pe_gate" in pe_names:
            saw_stream_to_gate = True
        if "pe_gate" in pe_names and "pe_carry" in pe_names:
            saw_gate_to_carry = True

    if not saw_stream_to_gate:
        raise SystemExit("expected a routed edge between pe_stream and pe_gate")
    if not saw_gate_to_carry:
        raise SystemExit("expected a routed edge between pe_gate and pe_carry")

    for op_name in ("dataflow.stream", "dataflow.gate", "dataflow.carry"):
        if op_name not in html:
            raise SystemExit(f"viz html missing {op_name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
