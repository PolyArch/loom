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
    opcode_bits = bit_width_for_choices(1)
    opcode = read_bits(words, offset, opcode_bits)
    offset += opcode_bits
    inputs = []
    input_sel_bits = bit_width_for_choices(num_inputs)
    for _ in range(num_inputs):
        inputs.append(decode_mux(words, offset, input_sel_bits))
        offset += input_sel_bits + 2
    outputs = []
    output_sel_bits = bit_width_for_choices(num_outputs)
    for _ in range(num_outputs):
        outputs.append(decode_mux(words, offset, output_sel_bits))
        offset += output_sel_bits + 2
    return {"enable": enable, "opcode": opcode, "inputs": inputs, "outputs": outputs}


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))
    html = (out_dir / "dfg.viz.html").read_text(encoding="utf-8")

    node_mappings = {entry["sw_op"]: entry["hw_name"] for entry in mapping["node_mappings"]}
    if node_mappings.get("dataflow.invariant") != "fu_invariant":
        raise SystemExit(f"dataflow.invariant should map to fu_invariant, got {node_mappings}")

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}
    inv_slice = slices.get(("pe_invariant", "spatial_pe"))
    gate_slice = slices.get(("pe_gate", "spatial_pe"))
    if inv_slice is None or gate_slice is None:
        raise SystemExit("missing spatial_pe slices for invariant test")

    inv_words = config["words"][inv_slice["word_offset"]:inv_slice["word_offset"] + inv_slice["word_count"]]
    gate_words = config["words"][gate_slice["word_offset"]:gate_slice["word_offset"] + gate_slice["word_count"]]
    inv_cfg = decode_spatial(inv_words, 2, 1)
    gate_cfg = decode_spatial(gate_words, 2, 1)

    if inv_cfg["enable"] != 1:
        raise SystemExit("pe_invariant should be enabled")
    if inv_cfg["inputs"] != [(0, False, False), (1, False, False)]:
        raise SystemExit(f"unexpected pe_invariant inputs: {inv_cfg['inputs']}")
    if inv_cfg["outputs"] != [(0, False, False)]:
        raise SystemExit(f"unexpected pe_invariant outputs: {inv_cfg['outputs']}")

    if gate_cfg["enable"] != 0:
        raise SystemExit("pe_gate should be disabled")
    if gate_cfg["inputs"] != [(0, False, True), (0, False, True)]:
        raise SystemExit(f"unexpected pe_gate inputs: {gate_cfg['inputs']}")
    if gate_cfg["outputs"] != [(0, False, True)]:
        raise SystemExit(f"unexpected pe_gate outputs: {gate_cfg['outputs']}")

    if "dataflow.invariant" not in html or "fu_invariant" not in html:
        raise SystemExit("viz html missing invariant op/component payload")

    return 0


if __name__ == "__main__":
    sys.exit(main())
