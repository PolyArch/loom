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


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))
    html = (out_dir / "dfg.viz.html").read_text(encoding="utf-8")

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    map_tag_slice = slices.get(("map_tag_0", "map_tag"))
    if map_tag_slice is None:
        raise SystemExit("missing map_tag slice")
    if not map_tag_slice["complete"]:
        raise SystemExit("map_tag slice marked incomplete")

    words = config["words"][
        map_tag_slice["word_offset"]:map_tag_slice["word_offset"] + map_tag_slice["word_count"]
    ]
    if read_bits(words, 0, 16) != 2:
        raise SystemExit(f"unexpected map_tag table_size words: {words}")
    if read_bits(words, 16, 16) != 0 or read_bits(words, 32, 16) != 1:
        raise SystemExit(f"unexpected map_tag table contents: {words}")

    port_info = {entry["id"]: entry for entry in mapping["port_table"]}
    routed = [entry for entry in mapping["edge_routings"] if entry["kind"] == "routed"]
    if len(routed) != 1:
        raise SystemExit(f"expected one routed edge, got {routed}")
    path = routed[0]["path"]
    if not path:
        raise SystemExit("expected non-empty routed path through map_tag")
    path_kinds = [port_info[pid]["kind"] for pid in path if pid in port_info]
    if "add_tag" not in path_kinds or "map_tag" not in path_kinds or "del_tag" not in path_kinds:
        raise SystemExit(f"expected add_tag/map_tag/del_tag in routed path, got {path_kinds}")

    if '"kind": "map_tag"' not in html:
        raise SystemExit("viz html does not export map_tag component data")
    if '"edge_routings"' not in html:
        raise SystemExit("viz html does not export mapping data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
