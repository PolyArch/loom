#!/usr/bin/env python3
"""Assert row-complete interpolate_linear CGRA-sim evidence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from interpolate_linear_fixtures import InterpolateLinearFixture, fixture_from_source


HARDWARE = "shared_signal_window_adg"
EXPECTED_DFG_CYCLES = 32471
EXPECTED_HARDWARE_CYCLES = 32699
EXPECTED_PLACED_RECORDS = 28
EXPECTED_ROUTED_EDGES = 36
EXPECTED_ROUTE_SEGMENTS = 162
EXPECTED_CONFIG_RECORDS = 881


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing artifact: {path}")
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"artifact must be a JSON object: {path}")
    return data


def artifact_path(evidence: Path, *names: str) -> Path:
    for name in names:
        path = evidence / name
        if path.is_file():
            return path
    return evidence / names[0]


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


def require_output_shape(values: object, fixture: InterpolateLinearFixture) -> None:
    require(isinstance(values, list), f"{fixture.case} output should be a token list")
    parsed = [parse_float_token(str(value)) for value in values]
    rounded = {round(value, 5) for value in parsed}
    require(len(rounded) > 40, f"{fixture.case} output should expose many distinct interpolation values")
    require(min(parsed) == 0.0, f"{fixture.case} output should preserve the zero endpoint")
    require(max(parsed) == fixture.outputs[-1], f"{fixture.case} output should preserve the final endpoint")
    require(math.isclose(math.fsum(parsed), 20351.5, rel_tol=0.0, abs_tol=1.0e-5), f"bad checksum: {parsed}")


def require_fire_counts(dfg: dict[str, object], fixture: InterpolateLinearFixture) -> None:
    counts = dfg.get("operation_fire_counts")
    require(isinstance(counts, dict), f"DFG report should expose operation fire counts: {dfg}")
    for op_name, expected in fixture.expected_fire_counts.items():
        require(counts.get(op_name) == expected, f"{op_name} fire count should be {expected}, got {counts.get(op_name)}")


def require_route_segments(mapping: dict[str, object]) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping should expose route records: {mapping}")
    saw_switch = False
    saw_index_cast = False
    saw_fmuladd = False
    saw_load_addr = False
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        edge_ref = str(route.get("edge_ref", ""))
        saw_index_cast = saw_index_cast or "arith.index_cast" in edge_ref
        saw_fmuladd = saw_fmuladd or "llvm.intr.fmuladd" in edge_ref
        saw_load_addr = saw_load_addr or "dataflow.load" in edge_ref
        segments = route.get("segments")
        require(isinstance(segments, list) and len(segments) >= 1, f"route lacks segments: {route}")
        previous_sink = None
        segment_kinds: set[str] = set()
        for segment in segments:
            require(isinstance(segment, dict), f"route segment should be an object: {segment}")
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
    require(saw_switch, "interpolate_linear route should use a real switch")
    require(saw_index_cast, "interpolate_linear route should include arith.index_cast data movement")
    require(saw_fmuladd, "interpolate_linear route should include fmuladd data movement")
    require(saw_load_addr, "interpolate_linear route should include load-address data movement")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} EVIDENCE_DIR")
    fixture = fixture_from_source()
    evidence = Path(argv[1])
    mapping_id = f"{fixture.case}__{fixture.graph}__{HARDWARE}"

    dfg = read_json(artifact_path(evidence, "interpolate_linear.dfg.report.json", "interpolate_linear-dfg-sim-report.json"))
    mapping = read_json(artifact_path(evidence, "interpolate_linear.mapping.json", "pnr-mapping.json"))
    cgra = read_json(artifact_path(evidence, "interpolate_linear.cgra.report.json", "interpolate_linear-cgra-sim-report.json"))
    comparison = read_json(artifact_path(evidence, "interpolate_linear.sim-comparison-report.json", "sim-comparison-report.json"))

    require(
        dfg.get("status") == "pass"
        and dfg.get("kind") == "dfg_sim_report"
        and dfg.get("workload") == fixture.case
        and dfg.get("graph") == fixture.graph
        and dfg.get("dynamic_work_items") == fixture.query_count
        and dfg.get("final_outputs") == ["none"]
        and dfg.get("diagnostics") == [],
        f"interpolate_linear DFG evidence should cover the source loop: {dfg}",
    )
    require_fire_counts(dfg, fixture)
    memory = dfg.get("final_memory_state")
    require(isinstance(memory, dict), f"DFG final memory should be present: {dfg}")
    assert_float_tokens_close(memory.get(f"arg{fixture.input_xq_arg}"), fixture.input_xq, "input_xq")
    assert_float_tokens_close(memory.get(f"arg{fixture.input_x_arg}"), fixture.input_x, "input_x")
    assert_float_tokens_close(memory.get(f"arg{fixture.input_y_arg}"), fixture.input_y, "input_y")
    assert_float_tokens_close(memory.get(f"arg{fixture.output_arg}"), fixture.outputs, "output_yq")
    require_output_shape(memory.get(f"arg{fixture.output_arg}"), fixture)

    require(
        mapping.get("status") == "pass"
        and mapping.get("kind") == "pnr_mapping"
        and mapping.get("workload") == fixture.case
        and mapping.get("graph") == fixture.graph
        and mapping.get("hardware") == HARDWARE
        and mapping.get("mapping_id") == mapping_id
        and mapping.get("placed_records") == EXPECTED_PLACED_RECORDS
        and mapping.get("routed_edges") == EXPECTED_ROUTED_EDGES
        and mapping.get("unrouted_edges") == 0
        and mapping.get("unplaced_records") == 0,
        f"interpolate_linear should map to the shared signal-window ADG: {mapping}",
    )
    require_route_segments(mapping)

    require(
        cgra.get("status") == "pass"
        and cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == fixture.case
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == mapping_id
        and cgra.get("functional_state_source") == "carried_from_dfg_sim_report",
        f"interpolate_linear CGRA evidence should carry DFG final state: {cgra}",
    )
    cgra_memory = cgra.get("final_memory_state")
    require(isinstance(cgra_memory, dict), f"CGRA final memory should be present: {cgra}")
    assert_float_tokens_close(cgra_memory.get(f"arg{fixture.output_arg}"), fixture.outputs, "CGRA output_yq")
    require(cgra.get("dfg_cycles") == EXPECTED_DFG_CYCLES, f"DFG cycles changed: {cgra}")
    require(cgra.get("hardware_aware_cycles") == EXPECTED_HARDWARE_CYCLES, f"hardware cycles changed: {cgra}")
    require(cgra.get("hardware_aware_cycles") >= dfg.get("optimistic_cycles"), f"CGRA cycles must not be optimistic: {cgra}")
    require(cgra.get("route_segments") == EXPECTED_ROUTE_SEGMENTS, f"route segments changed: {cgra}")
    require(cgra.get("config_records") == EXPECTED_CONFIG_RECORDS, f"config records changed: {cgra}")

    require(
        comparison.get("status") == "pass"
        and comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == fixture.case
        and comparison.get("dfg_sim_report_identity") in {"interpolate_linear.dfg.report", "interpolate_linear-dfg-sim-report"}
        and comparison.get("mapping_artifact_identity") in {"interpolate_linear.mapping", "pnr-mapping"}
        and comparison.get("cgra_sim_report_identity") in {"interpolate_linear.cgra.report", "interpolate_linear-cgra-sim-report"},
        f"comparison should bind the interpolate_linear artifacts: {comparison}",
    )
    require(comparison.get("functional_comparison_status") == "pass", f"functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"performance comparison failed: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
