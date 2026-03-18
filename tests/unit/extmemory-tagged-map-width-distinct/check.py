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
    for key in (("sw_0", "spatial_sw"), ("map_tag_0", "map_tag"),
                ("map_tag_1", "map_tag"), ("tsw_0", "temporal_sw"),
                ("tsw_1", "temporal_sw")):
        entry = slices.get(key)
        if entry is None or not entry["complete"]:
            raise SystemExit(f"missing or incomplete slice {key}")

    data_slice = slices[("tsw_0", "temporal_sw")]
    data_words = config["words"][
        data_slice["word_offset"]: data_slice["word_offset"] + data_slice["word_count"]
    ]
    data_slots = decode_temporal_switch(data_words, slot_count=2, tag_width=3, route_bits=2)
    if data_slots != [
        {"valid": 1, "tag": 0, "routes": [1, 0]},
        {"valid": 1, "tag": 1, "routes": [0, 1]},
    ]:
        raise SystemExit(f"unexpected load-data demux config: {data_slots}")

    port_info = {entry["id"]: entry for entry in mapping["port_table"]}
    unrouted = [entry for entry in mapping["edge_routings"] if entry["kind"] == "unrouted"]
    if unrouted:
        raise SystemExit(f"unexpected unrouted edges: {unrouted}")

    saw_width_adapt_ingress = False
    for entry in mapping["edge_routings"]:
        if entry["kind"] != "routed":
            continue
        path_ports = [port_info[pid] for pid in entry.get("path", []) if pid in port_info]
        path_kinds = [port["kind"] for port in path_ports]
        if "map_tag" in path_kinds and "sw" in path_kinds and "memory" in path_kinds:
            saw_width_adapt_ingress = True
            break
    if not saw_width_adapt_ingress:
        raise SystemExit("expected mapped-width ingress route into extmemory")

    if html.count('"kind": "map_tag"') < 2 or '"kind": "spatial_sw"' not in html:
        raise SystemExit("viz html missing expected map_tag or spatial_sw payload")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
