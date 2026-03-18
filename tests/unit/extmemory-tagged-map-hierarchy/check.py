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


def expect_temporal_route(mapping: dict, port_info: dict[int, dict], component: str,
                          expected_tag_by_sink: dict[str, int], route_entries: dict[int, int]) -> None:
    saw = set()
    for entry in mapping["edge_routings"]:
        if entry["kind"] != "routed":
            continue
        path = entry.get("path", [])
        tsw_out = None
        sink = None
        for pid in path:
            port = port_info.get(pid)
            if not port:
                continue
            if port["kind"] == "temporal_sw" and port.get("name") == component and port["dir"] == "out":
                tsw_out = port["index"]
            if port["kind"] == "fu" and port.get("pe") in expected_tag_by_sink and port["dir"] == "in":
                sink = port["pe"]
        if tsw_out is None or sink is None:
            continue
        expected_tag = expected_tag_by_sink[sink]
        mask = route_entries.get(expected_tag)
        if mask is None or ((mask >> tsw_out) & 1) == 0:
            raise SystemExit(
                f"missing temporal route for component {component}, sink {sink}, "
                f"tag {expected_tag}, output {tsw_out}, entries={route_entries}"
            )
        saw.add(sink)
    if saw != set(expected_tag_by_sink):
        raise SystemExit(f"incomplete temporal route coverage for {component}: saw {sorted(saw)}")


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))
    html = (out_dir / "dfg.viz.html").read_text(encoding="utf-8")

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    for sw_name in ("sw_leaf0", "sw_leaf1", "sw_root"):
        sw_slice = slices.get((sw_name, "spatial_sw"))
        if sw_slice is None or not sw_slice["complete"]:
            raise SystemExit(f"missing or incomplete spatial_sw slice {sw_name}")
    for tsw_name in ("tsw_0", "tsw_1"):
        tsw_slice = slices.get((tsw_name, "temporal_sw"))
        if tsw_slice is None or not tsw_slice["complete"]:
            raise SystemExit(f"missing or incomplete temporal_sw slice {tsw_name}")

    map_tag_slices = [entry for entry in config["slices"] if entry["kind"] == "map_tag"]
    if len(map_tag_slices) < 3:
        raise SystemExit("expected at least three map_tag slices")

    unrouted = [entry for entry in mapping["edge_routings"] if entry["kind"] == "unrouted"]
    if unrouted:
        raise SystemExit(f"unexpected unrouted edges: {unrouted}")

    port_info = {entry["id"]: entry for entry in mapping["port_table"]}
    saw_ingress = False
    saw_egress = False
    for entry in mapping["edge_routings"]:
        if entry["kind"] != "routed":
            continue
        path_kinds = [port_info[pid]["kind"] for pid in entry.get("path", []) if pid in port_info]
        if path_kinds.count("sw") >= 2 and "add_tag" in path_kinds and "map_tag" in path_kinds and "memory" in path_kinds:
            saw_ingress = True
        if "memory" in path_kinds and "map_tag" in path_kinds and "temporal_sw" in path_kinds and "del_tag" in path_kinds:
            saw_egress = True
    if not saw_ingress:
        raise SystemExit("expected hierarchical tagged ingress route into extmemory")
    if not saw_egress:
        raise SystemExit("expected tagged egress route through map_tag and temporal_sw")

    data_routes = decode_temporal_sw_slice(config, slices[("tsw_0", "temporal_sw")], 1, 2)
    done_routes = decode_temporal_sw_slice(config, slices[("tsw_1", "temporal_sw")], 1, 2)
    expect_temporal_route(mapping, port_info, "tsw_0", {"pe_0": 1, "pe_1": 0}, data_routes)

    done_saw = set()
    for entry in mapping["edge_routings"]:
        if entry["kind"] != "routed":
            continue
        path = entry.get("path", [])
        tsw_out = None
        module_out = None
        for pid in path:
            port = port_info.get(pid)
            if not port:
                continue
            if port["kind"] == "temporal_sw" and port.get("name") == "tsw_1" and port["dir"] == "out":
                tsw_out = port["index"]
            if port["kind"] == "module_out" and port["dir"] == "in":
                module_out = port["index"]
        if tsw_out is None or module_out not in (2, 3):
            continue
        expected_tag = 1 if module_out == 2 else 0
        mask = done_routes.get(expected_tag)
        if mask is None or ((mask >> tsw_out) & 1) == 0:
            raise SystemExit(
                f"missing done temporal route for tag {expected_tag}, output {tsw_out}, entries={done_routes}"
            )
        done_saw.add(module_out)
    if done_saw != {2, 3}:
        raise SystemExit(f"incomplete done-route coverage: {sorted(done_saw)}")

    if html.count('"kind": "map_tag"') < 3:
        raise SystemExit("viz html missing expected map_tag payloads")
    if html.count('"kind": "spatial_sw"') < 3 or html.count('"kind": "temporal_sw"') < 2:
        raise SystemExit("viz html missing expected switch payloads")

    return 0


if __name__ == "__main__":
    sys.exit(main())
