#!/usr/bin/env python3

import json
import math
import sys
from pathlib import Path


def bit_width_for_choices(count: int) -> int:
    return math.ceil(math.log2(count)) if count > 1 else 0


def read_bits(words: list[int], offset: int, width: int) -> int:
    value = 0
    for bit in range(width):
        word_index = (offset + bit) // 32
        bit_index = (offset + bit) % 32
        if word_index < len(words) and ((words[word_index] >> bit_index) & 1):
            value |= 1 << bit
    return value


def decode_mux(words: list[int], offset: int, sel_bits: int) -> tuple[int, bool, bool]:
    sel = read_bits(words, offset, sel_bits)
    discard = bool(read_bits(words, offset + sel_bits, 1))
    disconnect = bool(read_bits(words, offset + sel_bits + 1, 1))
    return sel, discard, disconnect


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    case_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    if case_name == "techmap-add":
        stem = "dfg-add"
    elif case_name == "techmap-mul":
        stem = "dfg-mul"
    elif case_name == "techmap-mac":
        stem = "dfg-mac"
    else:
        stem = "dfg"
    config = json.loads((out_dir / f"{stem}.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / f"{stem}.map.json").read_text(encoding="utf-8"))

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    pe_slice = slices.get(("pe_0", "spatial_pe"))
    if pe_slice is None:
        raise SystemExit("missing spatial_pe slice for pe_0")
    if not pe_slice["complete"]:
        raise SystemExit("spatial_pe slice marked incomplete")

    words = config["words"][pe_slice["word_offset"]:pe_slice["word_offset"] + pe_slice["word_count"]]

    opcode_bits = bit_width_for_choices(2)
    input_sel_bits = bit_width_for_choices(3)
    input_field_width = input_sel_bits + 2
    output_field_width = 2

    offset = 0
    enable = read_bits(words, offset, 1)
    offset += 1
    opcode = read_bits(words, offset, opcode_bits)
    offset += opcode_bits
    mux0 = decode_mux(words, offset, input_sel_bits)
    offset += input_field_width
    mux1 = decode_mux(words, offset, input_sel_bits)
    offset += input_field_width
    mux2 = decode_mux(words, offset, input_sel_bits)
    offset += input_field_width
    demux0 = decode_mux(words, offset, 0)
    offset += output_field_width
    fu_cfg = read_bits(words, offset, 3)

    techmap_add_hw = None
    if case_name == "techmap-add":
        techmap_add_hw = None

    node_mappings = {entry["sw_op"]: entry["hw_name"] for entry in mapping["node_mappings"]}

    if case_name == "techmap-add":
        techmap_add_hw = node_mappings.get("arith.addi")
        if techmap_add_hw == "fu_add":
            expected = {"opcode": 0, "mux2": (0, False, True), "fu_cfg": 0}
        elif techmap_add_hw == "fu_mac":
            expected = {"opcode": 1, "mux2": (0, False, True), "fu_cfg": 0}
        else:
            raise SystemExit("arith.addi should map to fu_add or fu_mac")
    else:
        expected = {
            "techmap-mul": {"opcode": 1, "mux2": (0, False, True), "fu_cfg": 0},
            "techmap-mac": {"opcode": 1, "mux2": (2, False, False), "fu_cfg": 1},
        }[case_name]

    if enable != 1:
        raise SystemExit("spatial_pe enable bit should be 1")
    if opcode != expected["opcode"]:
        raise SystemExit(f"unexpected opcode {opcode}")
    if mux0 != (0, False, False):
        raise SystemExit(f"unexpected input mux0 {mux0}")
    if mux1 != (1, False, False):
        raise SystemExit(f"unexpected input mux1 {mux1}")
    if mux2 != expected["mux2"]:
        raise SystemExit(f"unexpected input mux2 {mux2}")
    if demux0 != (0, False, False):
        raise SystemExit(f"unexpected output demux {demux0}")
    if fu_cfg != expected["fu_cfg"]:
        raise SystemExit(f"unexpected spatial FU config {fu_cfg}")

    if case_name == "techmap-add" and techmap_add_hw not in ("fu_add", "fu_mac"):
        raise SystemExit("arith.addi should map to fu_add or fu_mac")
    if case_name == "techmap-mul" and node_mappings.get("arith.muli") != "fu_mac":
        raise SystemExit("arith.muli should map to fu_mac")
    if case_name == "techmap-mac":
        if node_mappings.get("arith.muli") != "fu_mac" or node_mappings.get("arith.addi") != "fu_mac":
            raise SystemExit("MAC subgraph should map to fu_mac")

    return 0


if __name__ == "__main__":
    sys.exit(main())
