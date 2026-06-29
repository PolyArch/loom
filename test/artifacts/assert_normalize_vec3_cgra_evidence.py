#!/usr/bin/env python3
"""Assert row-complete normalize_vec3 CGRA-sim evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from normalize_vec3_fixtures import NormalizeVec3Fixture, fixture_from_source


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
    require(isinstance(values, list), "normalize_vec3 output should be a token list")
    parsed = [parse_float_token(str(value)) for value in values]
    require(parsed[:3] == [0.0, 0.0, 0.0], f"first zero vector should stay zero: {values[:6]}")
    rounded = {round(value, 4) for value in parsed}
    require(len(rounded) > 12, f"normalize_vec3 output should expose distinct vectors: {values[:24]}")
    require(max(parsed) > 0.8, f"normalize_vec3 output should include normalized y components: {values[:24]}")
    require(any(abs(value) < 1.0e-7 for value in parsed), f"normalize_vec3 output should retain zero lanes: {values[:24]}")


def require_fire_counts(dfg: dict[str, object], fixture: NormalizeVec3Fixture) -> None:
    counts = dfg.get("operation_fire_counts")
    require(isinstance(counts, dict), f"DFG report should expose operation fire counts: {dfg}")
    for op_name, expected in fixture.expected_fire_counts.items():
        require(counts.get(op_name) == expected, f"{op_name} fire count should be {expected}, got {counts.get(op_name)}")


def require_routes(mapping: dict[str, object]) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping should expose route records: {mapping}")
    saw_switch = False
    saw_sqrt = False
    saw_div = False
    saw_const = False
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        edge_ref = str(route.get("edge_ref", ""))
        saw_sqrt = saw_sqrt or "math.sqrt" in edge_ref
        saw_div = saw_div or "arith.divf" in edge_ref
        saw_const = saw_const or "dataflow.constant" in edge_ref
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
    require(saw_switch, "normalize_vec3 route should use a real switch")
    require(saw_sqrt, "normalize_vec3 route should include math.sqrt data movement")
    require(saw_div, "normalize_vec3 route should include arith.divf data movement")
    require(saw_const, "normalize_vec3 route should include configured constants")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args(argv[1:])

    fixture = fixture_from_source()
    evidence = args.evidence_dir
    mapping_id = f"{fixture.case}__{fixture.graph}__{HARDWARE}"

    dfg = read_json(evidence / "normalize_vec3.dfg.report.json")
    mapping = read_json(evidence / "normalize_vec3.mapping.json")
    cgra = read_json(evidence / "normalize_vec3.cgra.report.json")
    comparison = read_json(evidence / "normalize_vec3.sim-comparison-report.json")

    require(
        dfg.get("status") == "pass"
        and dfg.get("kind") == "dfg_sim_report"
        and dfg.get("workload") == fixture.case
        and dfg.get("graph") == fixture.graph
        and dfg.get("dynamic_work_items") == fixture.size
        and dfg.get("final_outputs") == ["none"]
        and dfg.get("diagnostics") == [],
        f"normalize_vec3 DFG evidence should cover the source loop: {dfg}",
    )
    require_fire_counts(dfg, fixture)
    memory = dfg.get("final_memory_state")
    require(isinstance(memory, dict), f"DFG final memory should be present: {dfg}")
    assert_float_tokens_close(memory.get(f"arg{fixture.input_arg}"), fixture.input_values, "normalize_vec3 input")
    assert_float_tokens_close(memory.get(f"arg{fixture.output_arg}"), fixture.outputs, "normalize_vec3 output")
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
        f"normalize_vec3 should map to the shared signal-window ADG: {mapping}",
    )
    require_routes(mapping)

    require(
        cgra.get("status") == "pass"
        and cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == fixture.case
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == mapping_id
        and cgra.get("functional_state_source") == "carried_from_dfg_sim_report",
        f"normalize_vec3 CGRA evidence should carry DFG final state: {cgra}",
    )
    cgra_memory = cgra.get("final_memory_state")
    require(isinstance(cgra_memory, dict), f"CGRA final memory should be present: {cgra}")
    require(cgra_memory == memory, f"CGRA final memory should exactly match DFG final memory: {cgra_memory} vs {memory}")
    assert_float_tokens_close(cgra_memory.get(f"arg{fixture.output_arg}"), fixture.outputs, "normalize_vec3 CGRA output")
    require(isinstance(dfg.get("optimistic_cycles"), int), f"normalize_vec3 DFG cycles should be present: {dfg}")
    require(cgra.get("dfg_cycles") == dfg.get("optimistic_cycles"), f"CGRA should carry DFG cycles: {cgra}")
    require(cgra.get("hardware_aware_cycles", 0) >= dfg.get("optimistic_cycles", 0), f"CGRA cycles too optimistic: {cgra}")
    require(cgra.get("route_segments", 0) > 0, f"CGRA report should expose route segments: {cgra}")

    require(
        comparison.get("status") == "pass"
        and comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == fixture.case
        and comparison.get("dfg_sim_report_identity") == "normalize_vec3.dfg.report"
        and comparison.get("mapping_artifact_identity") == "normalize_vec3.mapping"
        and comparison.get("cgra_sim_report_identity") == "normalize_vec3.cgra.report",
        f"comparison should bind the normalize_vec3 artifacts: {comparison}",
    )
    require(comparison.get("functional_comparison_status") == "pass", f"functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"performance comparison failed: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
