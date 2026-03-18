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


def decode_spatial(words: list[int], num_inputs: int, num_outputs: int) -> dict[str, object]:
    offset = 0
    enable = read_bits(words, offset, 1)
    offset += 1
    opcode = read_bits(words, offset, bit_width_for_choices(1))
    offset += bit_width_for_choices(1)
    inputs = []
    for _ in range(num_inputs):
        inputs.append(decode_mux(words, offset, bit_width_for_choices(num_inputs)))
        offset += bit_width_for_choices(num_inputs) + 2
    outputs = []
    for _ in range(num_outputs):
        outputs.append(decode_mux(words, offset, bit_width_for_choices(num_outputs)))
        offset += bit_width_for_choices(num_outputs) + 2
    return {"enable": enable, "opcode": opcode, "inputs": inputs, "outputs": outputs}


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    add_slice = slices.get(("pe_add", "spatial_pe"))
    sub_slice = slices.get(("pe_sub", "spatial_pe"))
    if add_slice is None or sub_slice is None:
        raise SystemExit("missing spatial_pe slices")

    add_words = config["words"][add_slice["word_offset"]:add_slice["word_offset"] + add_slice["word_count"]]
    sub_words = config["words"][sub_slice["word_offset"]:sub_slice["word_offset"] + sub_slice["word_count"]]
    add_cfg = decode_spatial(add_words, 2, 1)
    sub_cfg = decode_spatial(sub_words, 2, 1)

    if add_cfg["enable"] != 1:
        raise SystemExit("pe_add should be enabled")
    if add_cfg["inputs"] != [(0, False, False), (1, False, False)]:
        raise SystemExit(f"unexpected pe_add inputs: {add_cfg['inputs']}")
    if add_cfg["outputs"] != [(0, False, False)]:
        raise SystemExit(f"unexpected pe_add outputs: {add_cfg['outputs']}")

    if sub_cfg["enable"] != 0:
        raise SystemExit("pe_sub should be disabled")
    if sub_cfg["inputs"] != [(0, False, True), (0, False, True)]:
        raise SystemExit(f"unexpected pe_sub inputs: {sub_cfg['inputs']}")
    if sub_cfg["outputs"] != [(0, False, True)]:
        raise SystemExit(f"unexpected pe_sub outputs: {sub_cfg['outputs']}")

    node_mappings = {entry["sw_op"]: entry["hw_name"] for entry in mapping["node_mappings"]}
    if node_mappings.get("arith.addi") != "fu_add":
        raise SystemExit("arith.addi should map to fu_add")

    return 0


if __name__ == "__main__":
    sys.exit(main())
