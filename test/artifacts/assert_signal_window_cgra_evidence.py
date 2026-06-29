#!/usr/bin/env python3
"""Assert row-complete signal-window CGRA-sim evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from signal_window_fixtures import WindowFixture, fixture_for_case


HARDWARE = "shared_signal_window_adg"


EXPECTED_METRICS: dict[str, dict[str, int]] = {
    "window_hamming": {
        "dfg_cycles": 14346,
        "hardware_aware_cycles": 14416,
        "placed_records": 10,
        "routed_edges": 10,
        "route_segments": 44,
        "config_records": 247,
    },
    "window_hanning": {
        "dfg_cycles": 13578,
        "hardware_aware_cycles": 13654,
        "placed_records": 11,
        "routed_edges": 11,
        "route_segments": 49,
        "config_records": 275,
    },
    "window_blackman": {
        "dfg_cycles": 21258,
        "hardware_aware_cycles": 21351,
        "placed_records": 13,
        "routed_edges": 14,
        "route_segments": 64,
        "config_records": 351,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing artifact: {path}")
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"artifact must be a JSON object: {path}")
    return data


def parse_float_token(value: str) -> float:
    prefix, raw = value.split(":", 1)
    require(prefix == "f32", f"expected f32 token, got {value!r}")
    return float(raw)


def assert_float_tokens_close(values: object, expected: tuple[float, ...], label: str) -> None:
    require(isinstance(values, list), f"{label} should be a token list")
    require(len(values) == len(expected), f"{label} length mismatch: expected {len(expected)}, got {len(values)}")
    for index, (actual_token, expected_value) in enumerate(zip(values, expected)):
        require(isinstance(actual_token, str), f"{label}[{index}] should be serialized: {actual_token!r}")
        actual = parse_float_token(actual_token)
        require(
            math.isclose(actual, expected_value, rel_tol=1.0e-5, abs_tol=1.0e-6),
            f"{label}[{index}] should be close to {expected_value}, got {actual_token}",
        )


def require_nontrivial_output(values: object, case: str) -> None:
    require(isinstance(values, list), f"{case} output should be a token list")
    parsed = [parse_float_token(str(value)) for value in values]
    rounded = {round(value, 5) for value in parsed}
    require(len(rounded) > 16, f"{case} output should expose a nontrivial waveform")
    require(max(parsed) > 0.05, f"{case} output should have positive signal content")
    require(min(parsed) < -0.01, f"{case} output should have negative signal content")


def require_fire_counts(dfg: dict[str, object], fixture: WindowFixture) -> None:
    counts = dfg.get("operation_fire_counts")
    require(isinstance(counts, dict), f"DFG report should expose operation fire counts: {dfg}")
    for op_name, expected in fixture.expected_fire_counts.items():
        require(counts.get(op_name) == expected, f"{op_name} fire count should be {expected}, got {counts.get(op_name)}")


def require_route_segments(mapping: dict[str, object], fixture: WindowFixture) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping should expose route records: {mapping}")
    saw_switch = False
    saw_cos = False
    saw_uitofp = False
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        edge_ref = str(route.get("edge_ref", ""))
        saw_cos = saw_cos or "math.cos" in edge_ref
        saw_uitofp = saw_uitofp or "llvm.uitofp" in edge_ref
        segments = route.get("segments")
        require(isinstance(segments, list) and len(segments) >= 1, f"route lacks segments: {route}")
        previous_sink = None
        segment_kinds: set[str] = set()
        for segment in segments:
            require(isinstance(segment, dict), f"route segment should be an object: {route}")
            segment_kinds.add(str(segment.get("segment_kind", "")))
            for field in ("hardware_ref", "source_endpoint", "sink_endpoint"):
                value = segment.get(field)
                require(isinstance(value, str) and "::" in value, f"segment lacks structured {field}: {segment}")
                require(".out" not in value and ".in" not in value, f"segment uses placeholder endpoint: {segment}")
            if previous_sink is not None:
                require(segment["source_endpoint"] == previous_sink, f"route is not contiguous: {route}")
            previous_sink = segment["sink_endpoint"]
            saw_switch = saw_switch or "fabric.switch" in str(segment.get("hardware_ref", ""))
        require("resource_edge" in segment_kinds, f"route lacks resource-edge segment: {route}")
    require(saw_switch, f"{fixture.case} route should use a real switch")
    require(saw_cos, f"{fixture.case} route should include math.cos data movement")
    require(saw_uitofp, f"{fixture.case} route should include llvm.uitofp data movement")


def require_metrics(dfg: dict[str, object], mapping: dict[str, object], cgra: dict[str, object], fixture: WindowFixture) -> None:
    expected = EXPECTED_METRICS.get(fixture.case)
    if expected is None:
        require(isinstance(dfg.get("optimistic_cycles"), int), f"{fixture.case} DFG cycles should be present")
        require(isinstance(cgra.get("hardware_aware_cycles"), int), f"{fixture.case} CGRA cycles should be present")
        require(cgra["hardware_aware_cycles"] >= dfg["optimistic_cycles"], f"{fixture.case} CGRA cycles must not be optimistic")
        require(mapping.get("placed_records", 0) > 0, f"{fixture.case} should place records")
        require(mapping.get("routed_edges", 0) > 0, f"{fixture.case} should route edges")
        require(mapping.get("unrouted_edges") == 0, f"{fixture.case} should have no unrouted edges")
        require(cgra.get("route_segments", 0) > 0, f"{fixture.case} should report route segments")
        return
    require(dfg.get("optimistic_cycles") == expected["dfg_cycles"], f"{fixture.case} DFG cycles changed: {dfg}")
    require(cgra.get("dfg_cycles") == expected["dfg_cycles"], f"{fixture.case} CGRA report should carry DFG cycles: {cgra}")
    require(
        cgra.get("hardware_aware_cycles") == expected["hardware_aware_cycles"],
        f"{fixture.case} hardware-aware cycles changed: {cgra}",
    )
    for key in ("placed_records", "routed_edges"):
        require(mapping.get(key) == expected[key], f"{fixture.case} mapping {key} changed: {mapping}")
        require(cgra.get(key) == expected[key], f"{fixture.case} CGRA {key} changed: {cgra}")
    require(mapping.get("unrouted_edges") == 0, f"{fixture.case} should have no unrouted edges")
    require(mapping.get("unplaced_records") == 0, f"{fixture.case} should have no unplaced records")
    require(cgra.get("route_segments") == expected["route_segments"], f"{fixture.case} route segments changed: {cgra}")
    require(cgra.get("config_records") == expected["config_records"], f"{fixture.case} config records changed: {cgra}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args(argv[1:])

    fixture = fixture_for_case(args.case)
    evidence = args.evidence_dir
    mapping_id = f"{fixture.case}__{fixture.graph}__{HARDWARE}"

    dfg = read_json(evidence / f"{fixture.case}.dfg.report.json")
    mapping = read_json(evidence / f"{fixture.case}.mapping.json")
    cgra = read_json(evidence / f"{fixture.case}.cgra.report.json")
    comparison = read_json(evidence / f"{fixture.case}.sim-comparison-report.json")

    require(
        dfg.get("status") == "pass"
        and dfg.get("kind") == "dfg_sim_report"
        and dfg.get("workload") == fixture.case
        and dfg.get("graph") == fixture.graph
        and dfg.get("dynamic_work_items") == fixture.size
        and dfg.get("final_outputs") == ["none"]
        and dfg.get("diagnostics") == [],
        f"{fixture.case} DFG evidence should cover the source loop: {dfg}",
    )
    require_fire_counts(dfg, fixture)
    memory = dfg.get("final_memory_state")
    require(isinstance(memory, dict), f"DFG final memory should be present: {dfg}")
    assert_float_tokens_close(memory.get(f"arg{fixture.input_arg}"), fixture.inputs, f"{fixture.case} input")
    assert_float_tokens_close(memory.get(f"arg{fixture.output_arg}"), fixture.outputs, f"{fixture.case} output")
    require_nontrivial_output(memory.get(f"arg{fixture.output_arg}"), fixture.case)

    require(
        mapping.get("status") == "pass"
        and mapping.get("kind") == "pnr_mapping"
        and mapping.get("workload") == fixture.case
        and mapping.get("graph") == fixture.graph
        and mapping.get("hardware") == HARDWARE
        and mapping.get("mapping_id") == mapping_id,
        f"{fixture.case} should map to the shared signal-window ADG: {mapping}",
    )
    require_route_segments(mapping, fixture)

    require(
        cgra.get("status") == "pass"
        and cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == fixture.case
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == mapping_id
        and cgra.get("functional_state_source") == "carried_from_dfg_sim_report",
        f"{fixture.case} CGRA evidence should carry DFG final state: {cgra}",
    )
    cgra_memory = cgra.get("final_memory_state")
    require(isinstance(cgra_memory, dict), f"CGRA final memory should be present: {cgra}")
    assert_float_tokens_close(cgra_memory.get(f"arg{fixture.output_arg}"), fixture.outputs, f"{fixture.case} CGRA output")
    require_metrics(dfg, mapping, cgra, fixture)

    require(
        comparison.get("status") == "pass"
        and comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == fixture.case
        and comparison.get("dfg_sim_report_identity") == f"{fixture.case}.dfg.report"
        and comparison.get("mapping_artifact_identity") == f"{fixture.case}.mapping"
        and comparison.get("cgra_sim_report_identity") == f"{fixture.case}.cgra.report",
        f"comparison should bind the {fixture.case} artifacts: {comparison}",
    )
    require(comparison.get("functional_comparison_status") == "pass", f"functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"performance comparison failed: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
