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


def decode_spatial_slice(words: list[int], num_inputs: int, num_outputs: int) -> dict[str, object]:
    offset = 0
    enable = read_bits(words, offset, 1)
    offset += 1
    opcode_bits = 0
    opcode = 0
    input_sel_bits = bit_width_for_choices(num_inputs)
    output_sel_bits = bit_width_for_choices(num_outputs)
    input_muxes = []
    output_demuxes = []
    for _ in range(num_inputs):
        input_muxes.append(decode_mux(words, offset, input_sel_bits))
        offset += input_sel_bits + 2
    for _ in range(num_outputs):
        output_demuxes.append(decode_mux(words, offset, output_sel_bits))
        offset += output_sel_bits + 2
    return {
        "enable": enable,
        "opcode_bits": opcode_bits,
        "opcode": opcode,
        "input_muxes": input_muxes,
        "output_demuxes": output_demuxes,
    }


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    expected = {
        "pe_add": {"num_inputs": 2, "num_outputs": 1, "muxes": [(0, False, False), (1, False, False)]},
        "pe_const": {"num_inputs": 1, "num_outputs": 1, "muxes": [(0, False, False)]},
        "pe_mul": {"num_inputs": 2, "num_outputs": 1, "muxes": [(0, False, False), (1, False, False)]},
        "pe_join": {"num_inputs": 1, "num_outputs": 1, "muxes": [(0, False, False)]},
    }

    for pe_name, spec in expected.items():
        pe_slice = slices.get((pe_name, "spatial_pe"))
        if pe_slice is None:
            raise SystemExit(f"missing spatial_pe slice for {pe_name}")
        if not pe_slice["complete"]:
            raise SystemExit(f"slice for {pe_name} marked incomplete")
        words = config["words"][pe_slice["word_offset"]:pe_slice["word_offset"] + pe_slice["word_count"]]
        decoded = decode_spatial_slice(words, spec["num_inputs"], spec["num_outputs"])
        if decoded["enable"] != 1:
            raise SystemExit(f"{pe_name} should be enabled")
        if decoded["input_muxes"] != spec["muxes"]:
            raise SystemExit(f"unexpected input muxes for {pe_name}: {decoded['input_muxes']}")
        if decoded["output_demuxes"] != [(0, False, False)]:
            raise SystemExit(f"unexpected output demuxes for {pe_name}: {decoded['output_demuxes']}")

    hw_names = {entry["sw_op"]: entry["hw_name"] for entry in mapping["node_mappings"]}
    expected_nodes = {
        "arith.addi": "fu_add",
        "handshake.constant": "fu_constant",
        "arith.muli": "fu_mul",
        "handshake.join": "fu_join",
    }
    for sw_op, hw_name in expected_nodes.items():
        if hw_names.get(sw_op) != hw_name:
            raise SystemExit(f"{sw_op} should map to {hw_name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
