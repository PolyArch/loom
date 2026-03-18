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
    ingress = slices.get(("sw_0", "spatial_sw"))
    data_demux = slices.get(("tsw_0", "temporal_sw"))
    done_demux = slices.get(("tsw_1", "temporal_sw"))
    if ingress is None:
        raise SystemExit("missing tagged spatial_sw ingress slice")
    if data_demux is None or done_demux is None:
        raise SystemExit("missing temporal_sw egress slices")
    if not ingress["complete"] or not data_demux["complete"] or not done_demux["complete"]:
        raise SystemExit("expected ingress and egress switch slices to be complete")

    ingress_words = config["words"][
        ingress["word_offset"]: ingress["word_offset"] + ingress["word_count"]
    ]
    if ingress_words != [3]:
        raise SystemExit(f"unexpected spatial ingress config: {ingress_words}")

    data_words = config["words"][
        data_demux["word_offset"]: data_demux["word_offset"] + data_demux["word_count"]
    ]
    data_slots = decode_temporal_switch(data_words, slot_count=2, tag_width=1, route_bits=2)
    if data_slots != [
        {"valid": 1, "tag": 0, "routes": [1, 0]},
        {"valid": 1, "tag": 1, "routes": [0, 1]},
    ]:
        raise SystemExit(f"unexpected load-data demux config: {data_slots}")

    done_words = config["words"][
        done_demux["word_offset"]: done_demux["word_offset"] + done_demux["word_count"]
    ]
    done_slots = decode_temporal_switch(done_words, slot_count=2, tag_width=1, route_bits=2)
    if done_slots != [
        {"valid": 1, "tag": 0, "routes": [1, 0]},
        {"valid": 1, "tag": 1, "routes": [0, 1]},
    ]:
        raise SystemExit(f"unexpected load-done demux config: {done_slots}")

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
        raise SystemExit("expected tagged spatial_sw ingress route into extmemory")
    if not saw_egress:
        raise SystemExit("expected temporal_sw egress route out of extmemory")

    if '"kind": "spatial_sw"' not in html or '"kind": "temporal_sw"' not in html:
        raise SystemExit("viz html missing spatial_sw or temporal_sw payload")

    return 0


if __name__ == "__main__":
    sys.exit(main())
