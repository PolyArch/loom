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


def get_fu_field(mapping: dict, pe_name: str, kind: str) -> dict:
    for entry in mapping["fu_configs"]:
        if entry["pe_name"] != pe_name:
            continue
        for field in entry["fields"]:
            if field["kind"] == kind:
                return field
    raise SystemExit(f"missing {kind} field for {pe_name}")


def main() -> int:
    if len(sys.argv) != 3:
        return 2

    out_dir = Path(sys.argv[2])
    config = json.loads((out_dir / "dfg.config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "dfg.map.json").read_text(encoding="utf-8"))
    html = (out_dir / "dfg.viz.html").read_text(encoding="utf-8")

    slices = {(entry["name"], entry["kind"]): entry for entry in config["slices"]}

    def slice_words(name: str) -> list[int]:
        entry = slices.get((name, "spatial_pe"))
        if entry is None:
            raise SystemExit(f"missing spatial_pe slice for {name}")
        if not entry["complete"]:
            raise SystemExit(f"slice for {name} marked incomplete")
        start = entry["word_offset"]
        end = start + entry["word_count"]
        return config["words"][start:end]

    const_words = slice_words("pe_const")
    cmpi_words = slice_words("pe_cmpi")
    cmpf_words = slice_words("pe_cmpf")
    stream_words = slice_words("pe_stream")

    const_offset = 0
    if read_bits(const_words, const_offset, 1) != 1:
        raise SystemExit("pe_const should be enabled")
    const_offset += 1
    if decode_mux(const_words, const_offset, 0) != (0, False, False):
        raise SystemExit("unexpected const input mux")
    const_offset += 2
    if decode_mux(const_words, const_offset, 0) != (0, False, False):
        raise SystemExit("unexpected const output demux")
    const_offset += 2
    if read_bits(const_words, const_offset, 32) != 42:
        raise SystemExit("unexpected handshake.constant config value")

    cmpi_field = get_fu_field(mapping, "pe_cmpi", "cmpi_predicate")
    cmpi_offset = 1
    cmpi_sel_bits = bit_width_for_choices(2)
    if decode_mux(cmpi_words, cmpi_offset, cmpi_sel_bits) != (0, False, False):
        raise SystemExit("unexpected cmpi input mux0")
    cmpi_offset += cmpi_sel_bits + 2
    if decode_mux(cmpi_words, cmpi_offset, cmpi_sel_bits) != (1, False, False):
        raise SystemExit("unexpected cmpi input mux1")
    cmpi_offset += cmpi_sel_bits + 2
    if decode_mux(cmpi_words, cmpi_offset, 0) != (0, False, False):
        raise SystemExit("unexpected cmpi output demux")
    cmpi_offset += 2
    if read_bits(cmpi_words, cmpi_offset, 4) != cmpi_field["value"]:
        raise SystemExit("cmpi predicate bits do not match fu_configs")

    cmpf_field = get_fu_field(mapping, "pe_cmpf", "cmpf_predicate")
    cmpf_offset = 1
    cmpf_sel_bits = bit_width_for_choices(2)
    if decode_mux(cmpf_words, cmpf_offset, cmpf_sel_bits) != (0, False, False):
        raise SystemExit("unexpected cmpf input mux0")
    cmpf_offset += cmpf_sel_bits + 2
    if decode_mux(cmpf_words, cmpf_offset, cmpf_sel_bits) != (1, False, False):
        raise SystemExit("unexpected cmpf input mux1")
    cmpf_offset += cmpf_sel_bits + 2
    if decode_mux(cmpf_words, cmpf_offset, 0) != (0, False, False):
        raise SystemExit("unexpected cmpf output demux")
    cmpf_offset += 2
    if read_bits(cmpf_words, cmpf_offset, 4) != cmpf_field["value"]:
        raise SystemExit("cmpf predicate bits do not match fu_configs")

    stream_field = get_fu_field(mapping, "pe_stream", "stream_cont_cond")
    stream_offset = 1
    stream_in_bits = bit_width_for_choices(3)
    stream_out_bits = bit_width_for_choices(2)
    if decode_mux(stream_words, stream_offset, stream_in_bits) != (0, False, False):
        raise SystemExit("unexpected stream input mux0")
    stream_offset += stream_in_bits + 2
    if decode_mux(stream_words, stream_offset, stream_in_bits) != (1, False, False):
        raise SystemExit("unexpected stream input mux1")
    stream_offset += stream_in_bits + 2
    if decode_mux(stream_words, stream_offset, stream_in_bits) != (2, False, False):
        raise SystemExit("unexpected stream input mux2")
    stream_offset += stream_in_bits + 2
    if decode_mux(stream_words, stream_offset, stream_out_bits) != (0, False, False):
        raise SystemExit("unexpected stream output demux0")
    stream_offset += stream_out_bits + 2
    if decode_mux(stream_words, stream_offset, stream_out_bits) != (1, False, False):
        raise SystemExit("unexpected stream output demux1")
    stream_offset += stream_out_bits + 2
    if read_bits(stream_words, stream_offset, 5) != stream_field["value"]:
        raise SystemExit("stream cont_cond bits do not match fu_configs")

    expected_pairs = {
        ("handshake.constant", "fu_constant"),
        ("arith.cmpi", "fu_cmpi"),
        ("arith.cmpf", "fu_cmpf"),
        ("dataflow.stream", "fu_stream"),
    }
    observed_pairs = {(entry["sw_op"], entry["hw_name"]) for entry in mapping["node_mappings"]}
    missing = expected_pairs - observed_pairs
    if missing:
        raise SystemExit(f"missing expected node mappings: {sorted(missing)}")

    for text in ("value=42", "predicate=", "cont_cond=!="):
        if text not in html:
            raise SystemExit(f"viz html missing config text {text!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
