#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def decode_temporal_switch(words: list[int], slot_count: int, tag_width: int,
                           route_bits: int) -> list[dict[str, object]]:
    bits: list[int] = []
    for word in words:
        for shift in range(32):
            bits.append((word >> shift) & 1)

    cursor = 0
    slots = []
    for _ in range(slot_count):
        valid = bits[cursor]
        cursor += 1
        tag = 0
        for bit in range(tag_width):
            tag |= bits[cursor] << bit
            cursor += 1
        routes = bits[cursor:cursor + route_bits]
        cursor += route_bits
        slots.append({"valid": valid, "tag": tag, "routes": routes})
    return slots


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))
    html = (out_dir / "dfg.viz.html").read_text(encoding="utf-8")

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    for sw_name in ("sw_0", "sw_1", "sw_2"):
        sw_slice = slices.get((sw_name, "spatial_sw"))
        if sw_slice is None or not sw_slice["complete"]:
            raise SystemExit(f"missing or incomplete ingress spatial_sw slice {sw_name}")
        words = config["words"][
            sw_slice["word_offset"]: sw_slice["word_offset"] + sw_slice["word_count"]
        ]
        if words != [3]:
            raise SystemExit(f"unexpected config for {sw_name}: {words}")
    for tsw_name in ("tsw_0", "tsw_1", "tsw_2"):
        tsw_slice = slices.get((tsw_name, "temporal_sw"))
        if tsw_slice is None or not tsw_slice["complete"]:
            raise SystemExit(f"missing or incomplete egress temporal_sw slice {tsw_name}")
        words = config["words"][
            tsw_slice["word_offset"]: tsw_slice["word_offset"] + tsw_slice["word_count"]
        ]
        slots = decode_temporal_switch(words, slot_count=2, tag_width=1, route_bits=2)
        if slots != [
            {"valid": 1, "tag": 0, "routes": [1, 0]},
            {"valid": 1, "tag": 1, "routes": [0, 1]},
        ]:
            raise SystemExit(f"unexpected config for {tsw_name}: {slots}")

    port_info = {entry["id"]: entry for entry in mapping["port_table"]}
    unrouted = [entry for entry in mapping["edge_routings"] if entry["kind"] == "unrouted"]
    if unrouted:
        raise SystemExit(f"unexpected unrouted edges: {unrouted}")

    saw_ingress = False
    saw_egress = False
    for entry in mapping["edge_routings"]:
        if entry["kind"] != "routed":
            continue
        path_kinds = [port_info[pid]["kind"] for pid in entry.get("path", []) if pid in port_info]
        if "add_tag" in path_kinds and "sw" in path_kinds and "memory" in path_kinds:
            saw_ingress = True
        if "memory" in path_kinds and "temporal_sw" in path_kinds and "del_tag" in path_kinds:
            saw_egress = True
    if not saw_ingress:
        raise SystemExit("expected tagged spatial_sw ingress route into memory")
    if not saw_egress:
        raise SystemExit("expected temporal_sw egress route out of memory")

    if html.count('"kind": "spatial_sw"') < 3 or html.count('"kind": "temporal_sw"') < 3:
        raise SystemExit("viz html missing expected switch component payloads")

    return 0


if __name__ == "__main__":
    sys.exit(main())
