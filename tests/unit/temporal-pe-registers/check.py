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


def decode_operand(words: list[int], offset: int, reg_bits: int) -> tuple[int, bool]:
    reg_idx = read_bits(words, offset, reg_bits)
    is_reg = bool(read_bits(words, offset + reg_bits, 1))
    return reg_idx, is_reg


def decode_result(words: list[int], offset: int, tag_bits: int, reg_bits: int) -> tuple[int, int, bool]:
    tag = read_bits(words, offset, tag_bits)
    reg_idx = read_bits(words, offset + tag_bits, reg_bits)
    is_reg = bool(read_bits(words, offset + tag_bits + reg_bits, 1))
    return tag, reg_idx, is_reg


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.fabric.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.fabric.map.json").read_text(encoding="utf-8"))

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    pe_slice = slices.get(("tpe_0", "temporal_pe"))
    if pe_slice is None:
        raise SystemExit("missing temporal_pe slice for tpe_0")
    if not pe_slice["complete"]:
        raise SystemExit("temporal_pe slice marked incomplete")

    words = config["words"][pe_slice["word_offset"]:pe_slice["word_offset"] + pe_slice["word_count"]]
    tag_bits = 1
    opcode_bits = bit_width_for_choices(2)
    reg_bits = bit_width_for_choices(2)
    operand_width = 1 + reg_bits
    input_sel_bits = bit_width_for_choices(2)
    input_width = input_sel_bits + 2
    output_width = 2
    result_width = tag_bits + reg_bits + 1

    offset = 0

    slot0_valid = read_bits(words, offset, 1)
    offset += 1
    slot0_tag = read_bits(words, offset, tag_bits)
    offset += tag_bits
    slot0_opcode = read_bits(words, offset, opcode_bits)
    offset += opcode_bits
    slot0_op0 = decode_operand(words, offset, reg_bits)
    offset += operand_width
    slot0_op1 = decode_operand(words, offset, reg_bits)
    offset += operand_width
    slot0_mux0 = decode_mux(words, offset, input_sel_bits)
    offset += input_width
    slot0_mux1 = decode_mux(words, offset, input_sel_bits)
    offset += input_width
    slot0_demux0 = decode_mux(words, offset, 0)
    offset += output_width
    slot0_result0 = decode_result(words, offset, tag_bits, reg_bits)
    offset += result_width

    slot1_valid = read_bits(words, offset, 1)
    offset += 1
    slot1_tag = read_bits(words, offset, tag_bits)
    offset += tag_bits
    slot1_opcode = read_bits(words, offset, opcode_bits)
    offset += opcode_bits
    slot1_op0 = decode_operand(words, offset, reg_bits)
    offset += operand_width
    slot1_op1 = decode_operand(words, offset, reg_bits)
    offset += operand_width
    slot1_mux0 = decode_mux(words, offset, input_sel_bits)
    offset += input_width
    slot1_mux1 = decode_mux(words, offset, input_sel_bits)
    offset += input_width
    slot1_demux0 = decode_mux(words, offset, 0)
    offset += output_width
    slot1_result0 = decode_result(words, offset, tag_bits, reg_bits)

    if slot0_valid != 1 or slot1_valid != 1:
        raise SystemExit("both temporal slots should be valid")
    if slot0_tag != 0 or slot1_tag != 1:
        raise SystemExit("unexpected temporal slot tags")
    if slot0_opcode != 0 or slot1_opcode != 1:
        raise SystemExit("unexpected opcode assignment")

    if slot0_op0 != (0, False) or slot0_op1 != (0, False):
        raise SystemExit(f"slot0 operands should use inputs, got {slot0_op0} {slot0_op1}")
    if slot1_op0 != (0, True):
        raise SystemExit(f"slot1 operand0 should read reg0, got {slot1_op0}")
    if slot1_op1 != (0, False):
        raise SystemExit(f"slot1 operand1 should use external input, got {slot1_op1}")

    if slot0_mux0 != (0, False, False) or slot0_mux1 != (1, False, False):
        raise SystemExit(f"unexpected slot0 mux config {slot0_mux0} {slot0_mux1}")
    if slot1_mux0 != (0, False, True):
        raise SystemExit(f"slot1 mux0 should be disconnected, got {slot1_mux0}")
    if slot1_mux1 != (0, False, False):
        raise SystemExit(f"slot1 mux1 should select input0, got {slot1_mux1}")

    if slot0_demux0 != (0, True, False):
        raise SystemExit(f"slot0 output should discard, got {slot0_demux0}")
    if slot1_demux0 != (0, False, False):
        raise SystemExit(f"slot1 output should drive port0, got {slot1_demux0}")

    if slot0_result0 != (0, 0, True):
        raise SystemExit(f"slot0 result should write reg0, got {slot0_result0}")
    if slot1_result0 != (1, 0, False):
        raise SystemExit(f"slot1 result should drive output with tag1, got {slot1_result0}")

    node_mappings = {entry["sw_op"]: entry["hw_name"] for entry in mapping["node_mappings"]}
    if node_mappings.get("arith.addi") != "fu_add":
        raise SystemExit("arith.addi should map to fu_add")
    if node_mappings.get("arith.muli") != "fu_mul":
        raise SystemExit("arith.muli should map to fu_mul")

    edge_kinds = {entry["sw_edge"]: entry["kind"] for entry in mapping["edge_routings"]}
    temporal_edges = [edge_id for edge_id, kind in edge_kinds.items() if kind == "temporal_reg"]
    if len(temporal_edges) != 1:
        raise SystemExit(f"expected exactly one temporal_reg edge, got {temporal_edges}")

    reg_bindings = mapping.get("temporal_registers", [])
    if len(reg_bindings) != 1:
        raise SystemExit(f"expected one temporal register binding, got {reg_bindings}")
    binding = reg_bindings[0]
    if binding["register_index"] != 0:
        raise SystemExit(f"expected register index 0, got {binding['register_index']}")
    if binding["writer_output_index"] != 0 or binding["reader_input_index"] != 0:
        raise SystemExit(f"unexpected temporal register endpoints {binding}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
