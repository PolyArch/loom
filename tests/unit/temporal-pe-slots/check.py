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


def decode_slot(words: list[int], offset: int, tag_bits: int, opcode_bits: int,
                input_count: int, input_sel_bits: int, output_count: int,
                output_sel_bits: int, result_tag_bits: int) -> tuple[dict[str, object], int]:
    slot = {}
    slot["valid"] = read_bits(words, offset, 1)
    offset += 1
    slot["tag"] = read_bits(words, offset, tag_bits)
    offset += tag_bits
    slot["opcode"] = read_bits(words, offset, opcode_bits)
    offset += opcode_bits
    slot["inputs"] = []
    for _ in range(input_count):
        slot["inputs"].append(decode_mux(words, offset, input_sel_bits))
        offset += input_sel_bits + 2
    slot["outputs"] = []
    for _ in range(output_count):
        slot["outputs"].append(decode_mux(words, offset, output_sel_bits))
        offset += output_sel_bits + 2
    slot["results"] = []
    for _ in range(output_count):
        slot["results"].append(read_bits(words, offset, result_tag_bits))
        offset += result_tag_bits
    return slot, offset


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    pe_slice = slices.get(("tpe_0", "temporal_pe"))
    if pe_slice is None:
        raise SystemExit("missing temporal_pe slice for tpe_0")
    if not pe_slice["complete"]:
        raise SystemExit("temporal_pe slice marked incomplete")

    for sw_name in ("tsw_in0", "tsw_in1", "tsw_out"):
        sw_slice = slices.get((sw_name, "temporal_sw"))
        if sw_slice is None:
            raise SystemExit(f"missing temporal_sw slice for {sw_name}")
        if not sw_slice["complete"]:
            raise SystemExit(f"temporal_sw slice {sw_name} marked incomplete")

    words = config["words"][pe_slice["word_offset"]:pe_slice["word_offset"] + pe_slice["word_count"]]
    offset = 0
    slot0, offset = decode_slot(words, offset, 1, bit_width_for_choices(2), 2,
                                bit_width_for_choices(2), 1,
                                bit_width_for_choices(1), 1)
    slot1, offset = decode_slot(words, offset, 1, bit_width_for_choices(2), 2,
                                bit_width_for_choices(2), 1,
                                bit_width_for_choices(1), 1)

    expected0 = {
        "valid": 1,
        "tag": 0,
        "opcode": 0,
        "inputs": [(0, False, False), (1, False, False)],
        "outputs": [(0, False, False)],
        "results": [0],
    }
    expected1 = {
        "valid": 1,
        "tag": 1,
        "opcode": 1,
        "inputs": [(0, False, False), (1, False, False)],
        "outputs": [(0, False, False)],
        "results": [1],
    }
    if slot0 != expected0:
        raise SystemExit(f"unexpected slot0 config: {slot0}")
    if slot1 != expected1:
        raise SystemExit(f"unexpected slot1 config: {slot1}")

    node_mappings = {entry["sw_op"]: entry["hw_name"] for entry in mapping["node_mappings"]}
    if node_mappings.get("arith.addi") != "fu_add":
        raise SystemExit("arith.addi should map to fu_add")
    if node_mappings.get("arith.muli") != "fu_mul":
        raise SystemExit("arith.muli should map to fu_mul")

    return 0


if __name__ == "__main__":
    sys.exit(main())
