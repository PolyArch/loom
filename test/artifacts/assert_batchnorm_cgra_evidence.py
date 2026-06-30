#!/usr/bin/env python3
"""Assert row-complete batchnorm CGRA-sim evidence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from batchnorm_fixtures import BatchnormFixture, fixture_from_source


HARDWARE = "shared_signal_window_adg"
EXPECTED_DFG_CYCLES = 16111
EXPECTED_HW_CYCLES = 16274
EXPECTED_PLACED = 23
EXPECTED_ROUTED = 24
EXPECTED_ROUTE_SEGMENTS = 108
EXPECTED_CONFIG_RECORDS = 603


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
    require(isinstance(values, list), "batchnorm output should be a token list")
    parsed = [parse_float_token(str(value)) for value in values]
    rounded = {round(value, 4) for value in parsed}
    require(len(rounded) > 64, "batchnorm output should contain many distinct values")
    require(min(parsed) < -35.0, "batchnorm output should preserve negative normalized values")
    require(max(parsed) > 19.0, "batchnorm output should preserve positive normalized values")
    checksum = sum(parsed)
    require(math.isclose(checksum, -2371.160354, rel_tol=1.0e-6, abs_tol=1.0e-3), f"bad checksum {checksum}")


def require_fire_counts(dfg: dict[str, object], fixture: BatchnormFixture) -> None:
    counts = dfg.get("operation_fire_counts")
    require(isinstance(counts, dict), f"DFG report should expose operation fire counts: {dfg}")
    for op_name, expected in fixture.expected_fire_counts.items():
        require(counts.get(op_name) == expected, f"{op_name} fire count should be {expected}, got {counts.get(op_name)}")


def require_route_segments(mapping: dict[str, object]) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping should expose route records: {mapping}")
    saw_switch = False
    saw_sqrt = False
    saw_fmuladd = False
    saw_store = False
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        edge_ref = str(route.get("edge_ref", ""))
        saw_sqrt = saw_sqrt or "math.sqrt" in edge_ref
        saw_fmuladd = saw_fmuladd or "llvm.intr.fmuladd" in edge_ref
        saw_store = saw_store or "dataflow.store" in edge_ref
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
    require(saw_switch, "batchnorm route should use a real switch")
    require(saw_sqrt, "batchnorm route should include math.sqrt data movement")
    require(saw_fmuladd, "batchnorm route should include fmuladd data movement")
    require(saw_store, "batchnorm route should include store data movement")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <evidence-dir>")
    evidence = Path(argv[1])
    fixture = fixture_from_source()
    mapping_id = f"{fixture.case}__{fixture.graph}__{HARDWARE}"

    dfg = read_json(evidence / "batchnorm.dfg.report.json")
    mapping = read_json(evidence / "batchnorm.mapping.json")
    cgra = read_json(evidence / "batchnorm.cgra.report.json")
    comparison = read_json(evidence / "batchnorm.sim-comparison-report.json")

    require(
        dfg.get("status") == "pass"
        and dfg.get("kind") == "dfg_sim_report"
        and dfg.get("workload") == fixture.case
        and dfg.get("graph") == fixture.graph
        and dfg.get("dynamic_work_items") == fixture.dynamic_work_items
        and dfg.get("optimistic_cycles") == EXPECTED_DFG_CYCLES
        and dfg.get("final_outputs") == ["none"]
        and dfg.get("diagnostics") == [],
        f"batchnorm DFG evidence should cover the source-derived graph: {dfg}",
    )
    require_fire_counts(dfg, fixture)
    memory = dfg.get("final_memory_state")
    require(isinstance(memory, dict), f"DFG final memory should be present: {dfg}")
    assert_float_tokens_close(memory.get(f"arg{fixture.input_arg}"), fixture.inputs, "batchnorm input")
    assert_float_tokens_close(memory.get(f"arg{fixture.mean_arg}"), fixture.mean, "batchnorm mean")
    assert_float_tokens_close(memory.get(f"arg{fixture.variance_arg}"), fixture.variance, "batchnorm variance")
    assert_float_tokens_close(memory.get(f"arg{fixture.gamma_arg}"), fixture.gamma, "batchnorm gamma")
    assert_float_tokens_close(memory.get(f"arg{fixture.beta_arg}"), fixture.beta, "batchnorm beta")
    assert_float_tokens_close(memory.get(f"arg{fixture.output_arg}"), fixture.outputs, "batchnorm output")
    require_nontrivial_output(memory.get(f"arg{fixture.output_arg}"))

    require(
        mapping.get("status") == "pass"
        and mapping.get("kind") == "pnr_mapping"
        and mapping.get("workload") == fixture.case
        and mapping.get("graph") == fixture.graph
        and mapping.get("hardware") == HARDWARE
        and mapping.get("mapping_id") == mapping_id
        and mapping.get("placed_records") == EXPECTED_PLACED
        and mapping.get("routed_edges") == EXPECTED_ROUTED
        and mapping.get("unrouted_edges") == 0
        and mapping.get("unplaced_records") == 0
        and mapping.get("config_records") == EXPECTED_CONFIG_RECORDS,
        f"batchnorm should map to shared signal-window ADG: {mapping}",
    )
    require_route_segments(mapping)

    require(
        cgra.get("status") == "pass"
        and cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == fixture.case
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == mapping_id
        and cgra.get("dfg_cycles") == EXPECTED_DFG_CYCLES
        and cgra.get("hardware_aware_cycles") == EXPECTED_HW_CYCLES
        and cgra.get("placed_records") == EXPECTED_PLACED
        and cgra.get("routed_edges") == EXPECTED_ROUTED
        and cgra.get("route_segments") == EXPECTED_ROUTE_SEGMENTS
        and cgra.get("config_records") == EXPECTED_CONFIG_RECORDS
        and cgra.get("functional_state_source") == "carried_from_dfg_sim_report",
        f"batchnorm CGRA evidence should carry source-derived state: {cgra}",
    )
    cgra_memory = cgra.get("final_memory_state")
    require(isinstance(cgra_memory, dict), f"CGRA final memory should be present: {cgra}")
    assert_float_tokens_close(cgra_memory.get(f"arg{fixture.output_arg}"), fixture.outputs, "batchnorm CGRA output")

    require(
        comparison.get("status") == "pass"
        and comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == fixture.case
        and comparison.get("dfg_sim_report_identity") == "batchnorm.dfg.report"
        and comparison.get("mapping_artifact_identity") == "batchnorm.mapping"
        and comparison.get("cgra_sim_report_identity") == "batchnorm.cgra.report"
        and comparison.get("functional_comparison_status") == "pass"
        and comparison.get("memory_comparison_status") == "pass"
        and comparison.get("performance_comparison_status") == "pass",
        f"comparison should bind the batchnorm artifacts: {comparison}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
