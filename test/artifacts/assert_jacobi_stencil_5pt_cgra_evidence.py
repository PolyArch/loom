#!/usr/bin/env python3
"""Assert row-complete Jacobi stencil CGRA-sim evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from jacobi_stencil_5pt_fixtures import JacobiStencil5ptFixture, fixture_from_source


HARDWARE = "shared_signal_window_adg"
EXPECTED_DFG_CYCLES = 223
EXPECTED_CGRA_CYCLES = 379
EXPECTED_ROUTE_SEGMENTS = 112
EXPECTED_CONFIG_RECORDS = 615


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing artifact: {path}")
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"artifact must be a JSON object: {path}")
    return data


def parse_float_token(value: object) -> float:
    require(isinstance(value, str), f"expected serialized f32 token, got {value!r}")
    prefix, raw = value.split(":", 1)
    require(prefix == "f32", f"expected f32 token, got {value!r}")
    return float(raw)


def assert_float_tokens_close(values: object, expected: tuple[float, ...], label: str) -> None:
    require(isinstance(values, list), f"{label} should be a token list")
    require(len(values) == len(expected), f"{label} length mismatch: expected {len(expected)}, got {len(values)}")
    for index, (actual_token, expected_value) in enumerate(zip(values, expected)):
        actual = parse_float_token(actual_token)
        require(
            math.isclose(actual, expected_value, rel_tol=1.0e-6, abs_tol=1.0e-6),
            f"{label}[{index}] should be close to {expected_value}, got {actual_token}",
        )


def require_fire_counts(dfg: dict[str, object], fixture: JacobiStencil5ptFixture) -> None:
    counts = dfg.get("operation_fire_counts")
    require(isinstance(counts, dict), f"DFG report should expose operation fire counts: {dfg}")
    for op_name, expected in fixture.expected_fire_counts.items():
        require(counts.get(op_name) == expected, f"{op_name} fire count should be {expected}, got {counts.get(op_name)}")


def require_route_segments(mapping: dict[str, object]) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping should expose route records: {mapping}")
    saw_switch = False
    saw_load_address_route = False
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        edge_ref = str(route.get("edge_ref", ""))
        saw_load_address_route = saw_load_address_route or "->dataflow.load" in edge_ref
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
    require(saw_switch, "Jacobi mapping should use a real switch route")
    require(saw_load_address_route, "Jacobi mapping should route computed load addresses")


def assert_source_derived_memory(memory: object, fixture: JacobiStencil5ptFixture, label: str) -> None:
    require(isinstance(memory, dict), f"{label} final memory should be present")
    assert_float_tokens_close(memory.get(f"arg{fixture.input_arg}"), fixture.input_values, f"{label} input")
    assert_float_tokens_close(memory.get(f"arg{fixture.interior_arg}"), fixture.interior_values, f"{label} interior")
    parsed_interior = [parse_float_token(value) for value in memory[f"arg{fixture.interior_arg}"]]
    require(
        len({round(value, 6) for value in parsed_interior}) == len(parsed_interior),
        f"{label} interior should expose distinct stencil outputs",
    )


def source_checksum(values: tuple[float, ...]) -> float:
    return math.fsum((index + 1) * value for index, value in enumerate(values))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args(argv[1:])

    fixture = fixture_from_source()
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
        and dfg.get("dynamic_work_items") == fixture.interior_count
        and dfg.get("optimistic_cycles") == EXPECTED_DFG_CYCLES
        and dfg.get("final_outputs") == ["none"]
        and dfg.get("diagnostics") == [],
        f"Jacobi DFG evidence should cover the source-derived interior stencil: {dfg}",
    )
    require_fire_counts(dfg, fixture)
    assert_source_derived_memory(dfg.get("final_memory_state"), fixture, "DFG")

    require(
        mapping.get("status") == "pass"
        and mapping.get("kind") == "pnr_mapping"
        and mapping.get("workload") == fixture.case
        and mapping.get("graph") == fixture.graph
        and mapping.get("hardware") == HARDWARE
        and mapping.get("mapping_id") == mapping_id
        and mapping.get("placed_records") == 20
        and mapping.get("routed_edges") == 26
        and mapping.get("unrouted_edges") == 0
        and mapping.get("unplaced_records") == 0,
        f"Jacobi should map to the shared signal-window ADG: {mapping}",
    )
    require_route_segments(mapping)

    require(
        cgra.get("status") == "pass"
        and cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == fixture.case
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == mapping_id
        and cgra.get("functional_state_source") == "carried_from_dfg_sim_report"
        and cgra.get("config_records") == EXPECTED_CONFIG_RECORDS
        and cgra.get("route_segments") == EXPECTED_ROUTE_SEGMENTS
        and cgra.get("final_outputs") == ["none"],
        f"Jacobi CGRA evidence should carry DFG final state: {cgra}",
    )
    assert_source_derived_memory(cgra.get("final_memory_state"), fixture, "CGRA")
    require(cgra.get("dfg_cycles") == EXPECTED_DFG_CYCLES, f"CGRA should carry DFG cycle evidence: {cgra}")
    require(cgra.get("hardware_aware_cycles") == EXPECTED_CGRA_CYCLES, f"CGRA cycles changed: {cgra}")
    require(EXPECTED_CGRA_CYCLES >= EXPECTED_DFG_CYCLES, "CGRA cycles must not be optimistic")
    require(source_checksum(fixture.final_values) == 15897.5, "source-derived full-grid checksum changed")

    require(
        comparison.get("status") == "pass"
        and comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == fixture.case
        and comparison.get("dfg_sim_report_identity") == f"{fixture.case}.dfg.report"
        and comparison.get("mapping_artifact_identity") == f"{fixture.case}.mapping"
        and comparison.get("cgra_sim_report_identity") == f"{fixture.case}.cgra.report",
        f"comparison should bind the Jacobi artifacts: {comparison}",
    )
    require(comparison.get("functional_comparison_status") == "pass", f"functional comparison failed: {comparison}")
    require(comparison.get("memory_comparison_status") == "pass", f"memory comparison failed: {comparison}")
    require(comparison.get("performance_comparison_status") == "pass", f"performance comparison failed: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
