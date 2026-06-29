#!/usr/bin/env python3
"""Assert row-complete distance_point CGRA-sim evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from distance_point_fixtures import DistancePointFixture, fixture_from_source


HARDWARE = "shared_signal_window_adg"


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
            math.isclose(actual, expected_value, rel_tol=1.0e-5, abs_tol=1.0e-5),
            f"{label}[{index}] should be close to {expected_value}, got {actual_token}",
        )


def require_nontrivial_output(values: object) -> None:
    require(isinstance(values, list), "distance_point output should be a token list")
    parsed = [parse_float_token(str(value)) for value in values]
    rounded = {round(value, 4) for value in parsed}
    require(len(rounded) > 8, f"distance_point output should expose distinct distances: {values}")
    require(max(parsed) > 6.0, f"distance_point output should contain the long-distance end: {values}")
    require(min(parsed) < 2.4, f"distance_point output should contain the short-distance middle: {values}")


def require_fire_counts(dfg: dict[str, object], fixture: DistancePointFixture) -> None:
    counts = dfg.get("operation_fire_counts")
    require(isinstance(counts, dict), f"DFG report should expose operation fire counts: {dfg}")
    for op_name, expected in fixture.expected_fire_counts.items():
        require(counts.get(op_name) == expected, f"{op_name} fire count should be {expected}, got {counts.get(op_name)}")


def require_routes(mapping: dict[str, object]) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping should expose route records: {mapping}")
    saw_switch = False
    saw_sqrt = False
    saw_fma = False
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        edge_ref = str(route.get("edge_ref", ""))
        saw_sqrt = saw_sqrt or "math.sqrt" in edge_ref
        saw_fma = saw_fma or "llvm.intr.fmuladd" in edge_ref
        segments = route.get("segments")
        require(isinstance(segments, list) and segments, f"route lacks segments: {route}")
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
    require(saw_switch, "distance_point route should use a real switch")
    require(saw_sqrt, "distance_point route should include math.sqrt data movement")
    require(saw_fma, "distance_point route should include fmuladd data movement")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args(argv[1:])

    fixture = fixture_from_source()
    evidence = args.evidence_dir
    mapping_id = f"{fixture.case}__{fixture.graph}__{HARDWARE}"

    dfg = read_json(evidence / "distance_point.dfg.report.json")
    mapping = read_json(evidence / "distance_point.mapping.json")
    cgra = read_json(evidence / "distance_point.cgra.report.json")
    comparison = read_json(evidence / "distance_point.sim-comparison-report.json")

    require(
        dfg.get("status") == "pass"
        and dfg.get("kind") == "dfg_sim_report"
        and dfg.get("workload") == fixture.case
        and dfg.get("graph") == fixture.graph
        and dfg.get("dynamic_work_items") == fixture.size
        and dfg.get("final_outputs") == ["none"]
        and dfg.get("diagnostics") == [],
        f"distance_point DFG evidence should cover the source loop: {dfg}",
    )
    require_fire_counts(dfg, fixture)
    memory = dfg.get("final_memory_state")
    require(isinstance(memory, dict), f"DFG final memory should be present: {dfg}")
    assert_float_tokens_close(memory.get(f"arg{fixture.a_arg}"), fixture.a_values, "distance_point input A")
    assert_float_tokens_close(memory.get(f"arg{fixture.b_arg}"), fixture.b_values, "distance_point input B")
    assert_float_tokens_close(memory.get(f"arg{fixture.output_arg}"), fixture.outputs, "distance_point output")
    require_nontrivial_output(memory.get(f"arg{fixture.output_arg}"))

    require(
        mapping.get("status") == "pass"
        and mapping.get("kind") == "pnr_mapping"
        and mapping.get("workload") == fixture.case
        and mapping.get("graph") == fixture.graph
        and mapping.get("hardware") == HARDWARE
        and mapping.get("mapping_id") == mapping_id
        and mapping.get("unrouted_edges") == 0
        and mapping.get("unplaced_records") == 0,
        f"distance_point should map to the shared signal-window ADG: {mapping}",
    )
    require_routes(mapping)

    require(
        cgra.get("status") == "pass"
        and cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == fixture.case
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == mapping_id
        and cgra.get("functional_state_source") == "carried_from_dfg_sim_report",
        f"distance_point CGRA evidence should carry DFG final state: {cgra}",
    )
    cgra_memory = cgra.get("final_memory_state")
    require(isinstance(cgra_memory, dict), f"CGRA final memory should be present: {cgra}")
    assert_float_tokens_close(cgra_memory.get(f"arg{fixture.output_arg}"), fixture.outputs, "distance_point CGRA output")
    require(isinstance(dfg.get("optimistic_cycles"), int), f"distance_point DFG cycles should be present: {dfg}")
    require(cgra.get("dfg_cycles") == dfg.get("optimistic_cycles"), f"CGRA should carry DFG cycles: {cgra}")
    require(cgra.get("hardware_aware_cycles", 0) >= dfg.get("optimistic_cycles", 0), f"CGRA cycles too optimistic: {cgra}")
    require(cgra.get("route_segments", 0) > 0, f"CGRA report should expose route segments: {cgra}")

    require(
        comparison.get("status") == "pass"
        and comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == fixture.case
        and comparison.get("functional_comparison_status") == "pass"
        and comparison.get("memory_comparison_status") == "pass"
        and comparison.get("performance_comparison_status") == "pass",
        f"distance_point comparison should pass with real final state: {comparison}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
