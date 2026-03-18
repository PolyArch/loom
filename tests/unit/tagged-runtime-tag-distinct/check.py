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
        raise SystemExit("usage: check.py <case-name> <output-dir>")
    _case_name = sys.argv[1]
    out_dir = Path(sys.argv[2])

    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    slices = {entry["name"]: entry for entry in config["slices"]}
    split_slice = slices.get("tsw_split")
    if not split_slice:
        raise SystemExit("missing temporal_sw slice for tsw_split")

    words = config["words"][
        split_slice["word_offset"]: split_slice["word_offset"] + split_slice["word_count"]
    ]
    slots = decode_temporal_switch(words, slot_count=2, tag_width=3, route_bits=2)
    if slots[0]["valid"] != 1 or slots[0]["tag"] != 1 or slots[0]["routes"] != [1, 0]:
        raise SystemExit(f"unexpected slot0 config: {slots[0]}")
    if slots[1]["valid"] != 1 or slots[1]["tag"] != 2 or slots[1]["routes"] != [0, 1]:
        raise SystemExit(f"unexpected slot1 config: {slots[1]}")

    for expected_name in ("sw_merge", "map_tag_0", "tsw_split"):
        if expected_name not in slices:
            raise SystemExit(f"missing config slice for {expected_name}")

    viz_html = out_dir / "dfg.viz.html"
    if not viz_html.exists():
        raise SystemExit("missing dfg.viz.html output")
    html_text = viz_html.read_text(encoding="utf-8")
    if "tsw_split" not in html_text or "sw_merge" not in html_text:
        raise SystemExit("expected sw_merge and tsw_split to appear in viz HTML")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
