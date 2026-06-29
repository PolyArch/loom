#!/usr/bin/env python3
"""Assert row-complete window_hamming CGRA-sim evidence."""

from __future__ import annotations

import json
import math
import re
import struct
import sys
from pathlib import Path


GRAPH = "g_t_window_hamming_kernel_0_0"
HARDWARE = "shared_signal_window_adg"
MAPPING_ID = "window_hamming__g_t_window_hamming_kernel_0_0__shared_signal_window_adg"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing artifact: {path}")
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"artifact must be a JSON object: {path}")
    return data


def f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def parse_const(source: Path, name: str) -> float:
    text = source.read_text()
    pattern = rf"constexpr\s+(?:float|uint32_t)\s+{name}\s*=\s*(?P<value>[^;]+);"
    match = re.search(pattern, text)
    require(match is not None, f"missing constexpr {name} in {source}")
    value = match.group("value").strip()
    return float(value.rstrip("fuFU"))


def parse_float_literal(value: str) -> float:
    return float(value.strip().rstrip("fuFU"))


def parse_window_source(source: Path) -> tuple[int, float, float, float, float, float]:
    text = source.read_text()
    size = int(parse_const(source, "kSize"))
    input_pi = parse_const(source, "kInputPi")
    window_pi = parse_const(source, "kWindowPi")
    denominator_match = re.search(
        r"std::sin\(\s*2\.0f\s*\*\s*kInputPi\s*\*\s*static_cast<float>\(i\)\s*/\s*([0-9.]+)f?\s*\)",
        text,
        re.S,
    )
    require(denominator_match is not None, f"missing window_hamming input denominator in {source}")
    window_match = re.search(
        r"const\s+float\s+window\s*=\s*([0-9.]+)f?\s*-\s*([0-9.]+)f?\s*\*\s*std::cos",
        text,
    )
    require(window_match is not None, f"missing Hamming window coefficients in {source}")
    input_denominator = parse_float_literal(denominator_match.group(1))
    base = parse_float_literal(window_match.group(1))
    amplitude = parse_float_literal(window_match.group(2))
    return size, input_pi, window_pi, input_denominator, base, amplitude


def parse_float_token(value: str) -> float:
    prefix, raw = value.split(":", 1)
    require(prefix == "f32", f"expected f32 token, got {value!r}")
    return float(raw)


def assert_float_tokens_close(values: object, expected: list[float], label: str) -> None:
    require(isinstance(values, list), f"{label} should be a token list")
    require(len(values) == len(expected), f"{label} length mismatch: expected {len(expected)}, got {len(values)}")
    for index, (actual_token, expected_value) in enumerate(zip(values, expected)):
        require(isinstance(actual_token, str), f"{label}[{index}] should be serialized: {actual_token!r}")
        actual = parse_float_token(actual_token)
        require(
            math.isclose(actual, expected_value, rel_tol=1.0e-5, abs_tol=1.0e-6),
            f"{label}[{index}] should be close to {expected_value}, got {actual_token}",
        )


def source_values() -> tuple[list[float], list[float]]:
    source = Path(__file__).resolve().parents[2] / "test/app/window_hamming/main_func.cpp"
    size, input_pi, window_pi, input_denominator, base, amplitude = parse_window_source(source)
    twopi = f32(2.0 * f32(window_pi))
    denominator = float(size - 1)
    inputs = [math.sin(2.0 * input_pi * float(index) / input_denominator) for index in range(size)]
    outputs = [
        inputs[index] * (base - amplitude * math.cos(twopi * float(index) / denominator))
        for index in range(size)
    ]
    return inputs, outputs


def require_fire_counts(dfg: dict[str, object]) -> None:
    expected_counts = {
        "arith.divf": 256,
        "arith.index_cast": 256,
        "arith.mulf": 512,
        "dataflow.load": 256,
        "dataflow.store": 256,
        "dataflow.sync": 256,
        "llvm.intr.fmuladd": 256,
        "llvm.trunc": 256,
        "llvm.uitofp": 256,
        "math.cos": 256,
    }
    counts = dfg.get("operation_fire_counts")
    require(isinstance(counts, dict), f"DFG report should expose operation fire counts: {dfg}")
    for op_name, expected in expected_counts.items():
        require(counts.get(op_name) == expected, f"{op_name} fire count should be {expected}, got {counts.get(op_name)}")


def require_route_edges(mapping: dict[str, object]) -> None:
    expected_edges = {
        "arith.divf#0.result0->math.cos#0.operand0",
        "llvm.trunc#0.result0->llvm.uitofp#0.operand0",
        "llvm.uitofp#0.result0->arith.mulf#0.operand0",
        "math.cos#0.result0->llvm.intr.fmuladd#0.operand0",
    }
    routes = mapping.get("routes")
    require(isinstance(routes, list), f"mapping should expose route records: {mapping}")
    by_edge = {
        str(route.get("edge_ref")): route
        for route in routes
        if isinstance(route, dict) and route.get("edge_ref") is not None
    }
    for edge_ref in expected_edges:
        route = by_edge.get(edge_ref)
        require(route is not None, f"missing route for {edge_ref}: {mapping}")
        require(route.get("status") == "routed", f"route should be routed for {edge_ref}: {route}")
        segments = route.get("segments")
        require(isinstance(segments, list) and len(segments) >= 3, f"route should be multihop for {edge_ref}: {route}")
        saw_switch = False
        previous_sink = None
        segment_kinds: set[str] = set()
        for segment in segments:
            require(isinstance(segment, dict), f"route segment should be an object for {edge_ref}: {route}")
            segment_kinds.add(str(segment.get("segment_kind", "")))
            for field in ("hardware_ref", "source_endpoint", "sink_endpoint"):
                value = segment.get(field)
                require(isinstance(value, str) and "::" in value, f"segment lacks structured {field}: {segment}")
                require(".out" not in value and ".in" not in value, f"segment uses placeholder endpoint: {segment}")
            if previous_sink is not None:
                require(segment["source_endpoint"] == previous_sink, f"route is not contiguous for {edge_ref}: {route}")
            previous_sink = segment["sink_endpoint"]
            saw_switch = saw_switch or "fabric.switch" in str(segment.get("hardware_ref", ""))
        require(saw_switch, f"route should use a real switch for {edge_ref}: {route}")
        require({"module_path", "resource_edge"}.issubset(segment_kinds), f"route lacks concrete segments: {route}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} EVIDENCE_DIR")
    evidence = Path(argv[1])
    expected_input, expected_output = source_values()

    dfg = read_json(evidence / "window_hamming.dfg.report.json")
    mapping = read_json(evidence / "window_hamming.mapping.json")
    cgra = read_json(evidence / "window_hamming.cgra.report.json")
    comparison = read_json(evidence / "window_hamming.sim-comparison-report.json")

    require(
        dfg.get("status") == "pass"
        and dfg.get("kind") == "dfg_sim_report"
        and dfg.get("workload") == "window_hamming"
        and dfg.get("graph") == GRAPH
        and dfg.get("dynamic_work_items") == 256
        and dfg.get("optimistic_cycles") == 14346
        and dfg.get("final_outputs") == ["none"]
        and dfg.get("diagnostics") == [],
        f"window_hamming DFG evidence should cover the real Hamming loop: {dfg}",
    )
    require_fire_counts(dfg)
    memory = dfg.get("final_memory_state")
    require(isinstance(memory, dict), f"DFG final memory should be present: {dfg}")
    assert_float_tokens_close(memory.get("arg5"), expected_input, "window_hamming input")
    assert_float_tokens_close(memory.get("arg6"), expected_output, "window_hamming output")

    output_tokens = memory.get("arg6")
    require(isinstance(output_tokens, list), "window_hamming output should be a token list")
    require(
        parse_float_token(str(output_tokens[8])) > 0.08
        and parse_float_token(str(output_tokens[24])) < -0.15
        and parse_float_token(str(output_tokens[136])) > 0.95
        and parse_float_token(str(output_tokens[-1])) < -0.01,
        f"window_hamming output should expose a nontrivial tapered waveform: {output_tokens[:8]}",
    )

    require(
        mapping.get("status") == "pass"
        and mapping.get("kind") == "pnr_mapping"
        and mapping.get("workload") == "window_hamming"
        and mapping.get("graph") == GRAPH
        and mapping.get("hardware") == HARDWARE
        and mapping.get("mapping_id") == MAPPING_ID
        and mapping.get("placed_records") == 10
        and mapping.get("routed_edges") == 10
        and mapping.get("unrouted_edges") == 0
        and mapping.get("unplaced_records") == 0,
        f"window_hamming should map to the shared signal-window ADG: {mapping}",
    )
    require_route_edges(mapping)

    require(
        cgra.get("status") == "pass"
        and cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == "window_hamming"
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == MAPPING_ID
        and cgra.get("dfg_cycles") == 14346
        and cgra.get("hardware_aware_cycles") == 14416
        and cgra.get("placed_records") == 10
        and cgra.get("routed_edges") == 10
        and cgra.get("route_segments") == 44
        and cgra.get("config_records") == 247
        and cgra.get("functional_state_source") == "carried_from_dfg_sim_report",
        f"window_hamming CGRA evidence should carry DFG final state: {cgra}",
    )
    cgra_memory = cgra.get("final_memory_state")
    require(isinstance(cgra_memory, dict), f"CGRA final memory should be present: {cgra}")
    assert_float_tokens_close(cgra_memory.get("arg6"), expected_output, "window_hamming CGRA output")

    require(
        comparison.get("status") == "pass"
        and comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == "window_hamming"
        and comparison.get("dfg_sim_report_identity") == "window_hamming.dfg.report"
        and comparison.get("mapping_artifact_identity") == "window_hamming.mapping"
        and comparison.get("cgra_sim_report_identity") == "window_hamming.cgra.report",
        f"comparison should bind the window_hamming artifacts: {comparison}",
    )
    require(comparison.get("functional_comparison_status") == "pass", f"functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"performance comparison failed: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
