#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def read_bits(words: list[int], offset: int, width: int) -> int:
    value = 0
    for bit in range(width):
        word_index = (offset + bit) // 32
        bit_index = (offset + bit) % 32
        if word_index < len(words) and ((words[word_index] >> bit_index) & 1):
            value |= 1 << bit
    return value


def decode_temporal_sw_slice(config: dict, entry: dict, tag_width: int, route_bits: int) -> dict[int, int]:
    words = config["words"][entry["word_offset"]:entry["word_offset"] + entry["word_count"]]
    entries: dict[int, int] = {}
    bit_pos = 0
    for _ in range(2):
        valid = read_bits(words, bit_pos, 1)
        bit_pos += 1
        tag = read_bits(words, bit_pos, tag_width)
        bit_pos += tag_width
        route_mask = read_bits(words, bit_pos, route_bits)
        bit_pos += route_bits
        if valid:
            entries[tag] = route_mask
    return entries


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))
    html = (out_dir / "dfg.viz.html").read_text(encoding="utf-8")

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    for key in (("sw_0", "spatial_sw"), ("sw_1", "spatial_sw"),
                ("sw_2", "spatial_sw"), ("map_tag_0", "map_tag"),
                ("map_tag_1", "map_tag"), ("tsw_0", "temporal_sw"),
                ("tsw_1", "temporal_sw")):
        entry = slices.get(key)
        if entry is None or not entry["complete"]:
            raise SystemExit(f"missing or incomplete slice {key}")

    unrouted = [entry for entry in mapping["edge_routings"] if entry["kind"] == "unrouted"]
    if unrouted:
        raise SystemExit(f"unexpected unrouted edges: {unrouted}")

    data_routes = decode_temporal_sw_slice(config, slices[("tsw_0", "temporal_sw")], 3, 2)
    done_routes = decode_temporal_sw_slice(config, slices[("tsw_1", "temporal_sw")], 3, 2)
    if data_routes != {1: 0b01, 2: 0b10}:
        raise SystemExit(f"unexpected load-data temporal routes: {data_routes}")
    if done_routes != {1: 0b01, 2: 0b10}:
        raise SystemExit(f"unexpected load-done temporal routes: {done_routes}")

    port_info = {entry["id"]: entry for entry in mapping["port_table"]}
    saw_egress = False
    for entry in mapping["edge_routings"]:
        if entry["kind"] != "routed":
            continue
        path_kinds = [port_info[pid]["kind"] for pid in entry.get("path", []) if pid in port_info]
        if ("memory" in path_kinds and "map_tag" in path_kinds and
                "sw" in path_kinds and "temporal_sw" in path_kinds and
                "del_tag" in path_kinds):
            saw_egress = True
            break
    if not saw_egress:
        raise SystemExit(
            "expected width-adapted tagged egress route through map_tag, "
            "spatial_sw, and temporal_sw")

    if (html.count('"kind": "map_tag"') < 2 or
            html.count('"kind": "spatial_sw"') < 3 or
            html.count('"kind": "temporal_sw"') < 2):
        raise SystemExit("viz html missing expected egress tagged payloads")

    return 0


if __name__ == "__main__":
    sys.exit(main())
