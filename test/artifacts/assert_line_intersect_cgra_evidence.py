#!/usr/bin/env python3
"""Assert row-complete line_intersect CGRA-sim evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from line_intersect_fixtures import LineIntersectFixture, fixture_from_source


HARDWARE = "shared_signal_window_adg"
EXPECTED_DFG_CYCLES = 10432
EXPECTED_CGRA_CYCLES = 35968
EXPECTED_PLACED_RECORDS = 2752
EXPECTED_ROUTED_EDGES = 4544
EXPECTED_ROUTE_SEGMENTS = 20032
EXPECTED_CONFIG_RECORDS = 105664


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


def parse_i32_token(value: object) -> int:
    require(isinstance(value, str), f"expected serialized i32 token, got {value!r}")
    prefix, raw = value.split(":", 1)
    require(prefix == "i32", f"expected i32 token, got {value!r}")
    return int(raw)


def assert_float_tokens_close(values: object, expected: tuple[float, ...], label: str) -> None:
    require(isinstance(values, list), f"{label} should be a token list")
    require(len(values) == len(expected), f"{label} length mismatch: expected {len(expected)}, got {len(values)}")
    for index, (actual_token, expected_value) in enumerate(zip(values, expected)):
        actual = parse_float_token(actual_token)
        require(
            math.isclose(actual, expected_value, rel_tol=1.0e-6, abs_tol=1.0e-6),
            f"{label}[{index}] should be close to {expected_value}, got {actual_token}",
        )


def component_output_key(index: int) -> str:
    return f"line_intersect.dfg-sim-idx{index}.report:arg11"


def component_input_key(index: int, arg: int) -> str:
    return f"line_intersect.dfg-sim-idx{index}.report:arg{arg}"


def require_fire_counts(dfg: dict[str, object], fixture: LineIntersectFixture) -> None:
    counts = dfg.get("operation_fire_counts")
    require(isinstance(counts, dict), f"DFG report should expose operation fire counts: {dfg}")
    for op_name, expected in fixture.expected_fire_counts.items():
        require(counts.get(op_name) == expected, f"{op_name} fire count should be {expected}, got {counts.get(op_name)}")


def assert_component_memory(memory: object, fixture: LineIntersectFixture, label: str) -> None:
    require(isinstance(memory, dict), f"{label} final memory should be present")
    combined = [0 for _ in range(fixture.size)]
    for index, expected_value in enumerate(fixture.outputs):
        out_key = component_output_key(index)
        values = memory.get(out_key)
        require(isinstance(values, list), f"{label} component {index} output missing: {out_key}")
        require(len(values) == fixture.size, f"{label} component {index} output length mismatch")
        parsed = [parse_i32_token(value) for value in values]
        for lane, actual in enumerate(parsed):
            expected = expected_value if lane == index else 0
            require(actual == expected, f"{label} component {index} lane {lane} expected {expected}, got {actual}")
        combined[index] = parsed[index]

        assert_float_tokens_close(
            memory.get(component_input_key(index, fixture.line_a_arg)),
            fixture.line_a_values,
            f"{label} component {index} line A",
        )
        assert_float_tokens_close(
            memory.get(component_input_key(index, fixture.line_b_arg)),
            fixture.line_b_values,
            f"{label} component {index} line B",
        )
    require(tuple(combined) == fixture.outputs, f"{label} combined output mismatch: {combined}")
    require(sum((idx + 1) * value for idx, value in enumerate(combined)) == 2075, f"{label} checksum mismatch")
    require(sum(combined) == 62, f"{label} output should contain 62 intersections")
    require(combined[:5] == [1, 0, 0, 1, 1], f"{label} should preserve the nontrivial early cases: {combined[:5]}")


def require_component_references(evidence: Path, aggregate: dict[str, object], field: str, expected_count: int) -> None:
    identities = aggregate.get(field)
    require(isinstance(identities, list), f"aggregate lacks {field}: {aggregate}")
    require(len(identities) == expected_count, f"{field} should have {expected_count} entries, got {len(identities)}")
    require(len(set(identities)) == expected_count, f"{field} contains duplicates: {identities}")
    fingerprints = aggregate.get("input_artifact_fingerprints")
    require(isinstance(fingerprints, dict), f"aggregate lacks input artifact fingerprints: {aggregate}")
    for identity in identities:
        require(isinstance(identity, str) and identity.startswith("line_intersect."), f"bad component identity: {identity}")
        path = evidence / f"{identity}.json"
        require(path.is_file(), f"component identity does not resolve: {identity}")
        require(identity in fingerprints, f"missing component fingerprint: {identity}")


def require_routes(mapping: dict[str, object]) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping should expose routes: {mapping}")
    saw_switch = False
    saw_fabs = False
    saw_mux = False
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        edge_ref = str(route.get("edge_ref", ""))
        saw_fabs = saw_fabs or "llvm.intr.fabs" in edge_ref
        saw_mux = saw_mux or "dataflow.mux" in edge_ref
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
    require(saw_switch, "line_intersect routes should use real switches")
    require(saw_fabs, "line_intersect routes should include llvm.intr.fabs data movement")
    require(saw_mux, "line_intersect routes should include dataflow.mux data movement")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args(argv[1:])

    fixture = fixture_from_source()
    evidence = args.evidence_dir
    mapping_id = f"{fixture.case}__workload_graph_set__{HARDWARE}"

    dfg = read_json(evidence / "line_intersect.dfg.report.json")
    mapping = read_json(evidence / "line_intersect.mapping.json")
    cgra = read_json(evidence / "line_intersect.cgra.report.json")
    comparison = read_json(evidence / "line_intersect.sim-comparison-report.json")

    require(
        dfg.get("status") == "pass"
        and dfg.get("kind") == "dfg_sim_report"
        and dfg.get("workload") == fixture.case
        and dfg.get("graph") == "workload_graph_set"
        and dfg.get("aggregation_kind") == "workload_graph_set"
        and dfg.get("dynamic_work_items") == fixture.size
        and dfg.get("optimistic_cycles") == EXPECTED_DFG_CYCLES
        and dfg.get("diagnostics") == ["derived workload graph-set DFG report from component DFG simulator reports"],
        f"line_intersect DFG aggregate should cover the full source-derived workload: {dfg}",
    )
    require_component_references(evidence, dfg, "component_dfg_sim_report_identities", fixture.size)
    require_fire_counts(dfg, fixture)
    assert_component_memory(dfg.get("final_memory_state"), fixture, "DFG")

    require(
        mapping.get("status") == "pass"
        and mapping.get("kind") == "pnr_mapping"
        and mapping.get("workload") == fixture.case
        and mapping.get("graph") == "workload_graph_set"
        and mapping.get("hardware") == HARDWARE
        and mapping.get("mapping_id") == mapping_id
        and mapping.get("placed_records") == EXPECTED_PLACED_RECORDS
        and mapping.get("routed_edges") == EXPECTED_ROUTED_EDGES
        and mapping.get("unrouted_edges") == 0
        and mapping.get("unplaced_records") == 0
        and mapping.get("route_segments") == EXPECTED_ROUTE_SEGMENTS
        and mapping.get("config_records") == EXPECTED_CONFIG_RECORDS,
        f"line_intersect should map to shared signal-window ADG: {mapping}",
    )
    require_component_references(evidence, mapping, "component_mapping_artifact_identities", fixture.size)
    require_routes(mapping)

    require(
        cgra.get("status") == "pass"
        and cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == fixture.case
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == mapping_id
        and cgra.get("functional_state_source") == "component_cgra_sim_reports_carried_from_dfg_sim_reports"
        and cgra.get("dfg_cycles") == EXPECTED_DFG_CYCLES
        and cgra.get("hardware_aware_cycles") == EXPECTED_CGRA_CYCLES
        and cgra.get("route_segments") == EXPECTED_ROUTE_SEGMENTS
        and cgra.get("config_records") == EXPECTED_CONFIG_RECORDS,
        f"line_intersect CGRA aggregate should carry component final state: {cgra}",
    )
    require_component_references(evidence, cgra, "component_dfg_sim_report_identities", fixture.size)
    require_component_references(evidence, cgra, "component_cgra_sim_report_identities", fixture.size)
    assert_component_memory(cgra.get("final_memory_state"), fixture, "CGRA")
    require(EXPECTED_CGRA_CYCLES >= EXPECTED_DFG_CYCLES, "CGRA cycles must not be optimistic")

    require(
        comparison.get("status") == "pass"
        and comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == fixture.case
        and comparison.get("functional_comparison_status") == "pass"
        and comparison.get("memory_comparison_status") == "pass"
        and comparison.get("performance_comparison_status") == "pass",
        f"line_intersect comparison should pass with real final state: {comparison}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
