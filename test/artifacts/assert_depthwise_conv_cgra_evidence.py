#!/usr/bin/env python3
"""Assert row-complete depthwise_conv CGRA-sim evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from depthwise_conv_fixtures import DepthwiseConvFixture, fixture_from_source


HARDWARE = "shared_memory_reduction_adg"
EXPECTED_DFG_CYCLES = 40896
EXPECTED_CGRA_CYCLES = 50976
EXPECTED_PLACED_RECORDS = 1584
EXPECTED_ROUTED_EDGES = 1296
EXPECTED_ROUTE_SEGMENTS = 6768
EXPECTED_CONFIG_RECORDS = 37872


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


def assert_close(actual: float, expected: float, label: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1.0e-5, abs_tol=1.0e-5), f"{label}: expected {expected}, got {actual}")


def assert_float_tokens_close(values: object, expected: tuple[float, ...], label: str) -> None:
    require(isinstance(values, list), f"{label} should be a token list")
    require(len(values) == len(expected), f"{label} length mismatch: expected {len(expected)}, got {len(values)}")
    for index, (actual_token, expected_value) in enumerate(zip(values, expected)):
        assert_close(parse_float_token(actual_token), expected_value, f"{label}[{index}]")


def output_values(report: dict[str, object], fixture: DepthwiseConvFixture, label: str) -> tuple[float, ...]:
    values = report.get("final_outputs")
    require(isinstance(values, list), f"{label} final_outputs should be a list")
    require(len(values) == fixture.size * 2, f"{label} final_outputs length mismatch: {len(values)}")
    outputs: list[float] = []
    for index in range(fixture.size):
        require(values[2 * index] == "none", f"{label} output {index} control token should be none")
        outputs.append(parse_float_token(values[2 * index + 1]))
    for index, (actual, expected) in enumerate(zip(outputs, fixture.outputs)):
        assert_close(actual, expected, f"{label} output[{index}]")
    assert_close(sum((idx + 1) * value for idx, value in enumerate(outputs)), 1689.0, f"{label} checksum")
    for index, expected in enumerate((2.7, 0.5, -1.7, -1.9, -1.1, -1.3)):
        assert_close(outputs[index], expected, f"{label} first row output[{index}]")
    return tuple(outputs)


def component_input_key(index: int, arg: int) -> str:
    return f"depthwise_conv.dfg-sim-idx{index}.report:arg{arg}"


def assert_component_memory(memory: object, fixture: DepthwiseConvFixture, label: str) -> None:
    require(isinstance(memory, dict), f"{label} final memory should be present")
    for component in fixture.components:
        assert_float_tokens_close(
            memory.get(component_input_key(component.index, fixture.input_arg)),
            fixture.input_values,
            f"{label} component {component.index} input",
        )
        assert_float_tokens_close(
            memory.get(component_input_key(component.index, fixture.kernel_arg)),
            component.kernel_values,
            f"{label} component {component.index} channel kernel",
        )


def require_component_references(evidence: Path, aggregate: dict[str, object], field: str, expected_count: int) -> None:
    identities = aggregate.get(field)
    require(isinstance(identities, list), f"aggregate lacks {field}: {aggregate}")
    require(len(identities) == expected_count, f"{field} should have {expected_count} entries, got {len(identities)}")
    require(len(set(identities)) == expected_count, f"{field} contains duplicates")
    fingerprints = aggregate.get("input_artifact_fingerprints")
    require(isinstance(fingerprints, dict), f"aggregate lacks input artifact fingerprints: {aggregate}")
    for identity in identities:
        require(isinstance(identity, str) and identity.startswith("depthwise_conv."), f"bad component identity: {identity}")
        require((evidence / f"{identity}.json").is_file(), f"component identity does not resolve: {identity}")
        require(identity in fingerprints, f"missing component fingerprint: {identity}")


def require_fire_counts(dfg: dict[str, object], fixture: DepthwiseConvFixture) -> None:
    counts = dfg.get("operation_fire_counts")
    require(isinstance(counts, dict), f"DFG report should expose operation fire counts: {dfg}")
    for op_name, expected in fixture.expected_fire_counts.items():
        require(counts.get(op_name) == expected, f"{op_name} fire count should be {expected}, got {counts.get(op_name)}")


def require_routes(mapping: dict[str, object]) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"mapping should expose routes: {mapping}")
    saw_fmuladd = False
    saw_memory = False
    saw_switch = False
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        edge_ref = str(route.get("edge_ref", ""))
        saw_fmuladd = saw_fmuladd or "llvm.intr.fmuladd" in edge_ref
        segments = route.get("segments")
        require(isinstance(segments, list) and segments, f"route lacks segments: {route}")
        previous_sink = None
        for segment in segments:
            require(isinstance(segment, dict), f"route segment should be an object: {route}")
            for field in ("hardware_ref", "source_endpoint", "sink_endpoint"):
                value = segment.get(field)
                require(isinstance(value, str) and "::" in value, f"segment lacks structured {field}: {segment}")
                require(".out" not in value and ".in" not in value, f"segment uses placeholder endpoint: {segment}")
            if previous_sink is not None:
                require(segment["source_endpoint"] == previous_sink, f"route is not contiguous: {route}")
            previous_sink = segment["sink_endpoint"]
            hardware_ref = str(segment.get("hardware_ref", ""))
            source_endpoint = str(segment.get("source_endpoint", ""))
            sink_endpoint = str(segment.get("sink_endpoint", ""))
            saw_memory = saw_memory or "mem.load#" in hardware_ref or "mem.load#" in source_endpoint or "mem.load#" in sink_endpoint
            saw_switch = saw_switch or "fabric.switch" in hardware_ref
    require(saw_fmuladd, "depthwise_conv routes should carry fmuladd data")
    require(saw_memory, "depthwise_conv routes should include memory load resources")
    require(saw_switch, "depthwise_conv routes should use real switches")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args(argv[1:])

    fixture = fixture_from_source()
    evidence = args.evidence_dir
    mapping_id = f"{fixture.case}__workload_graph_set__{HARDWARE}"

    dfg = read_json(evidence / "depthwise_conv.dfg.report.json")
    mapping = read_json(evidence / "depthwise_conv.mapping.json")
    cgra = read_json(evidence / "depthwise_conv.cgra.report.json")
    comparison = read_json(evidence / "depthwise_conv.sim-comparison-report.json")

    require(
        dfg.get("status") == "pass"
        and dfg.get("kind") == "dfg_sim_report"
        and dfg.get("workload") == fixture.case
        and dfg.get("graph") == "workload_graph_set"
        and dfg.get("aggregation_kind") == "workload_graph_set"
        and dfg.get("dynamic_work_items") == fixture.size * 3
        and dfg.get("optimistic_cycles") == EXPECTED_DFG_CYCLES,
        f"depthwise_conv DFG aggregate should cover all output elements: {dfg}",
    )
    require_component_references(evidence, dfg, "component_dfg_sim_report_identities", fixture.size)
    require_fire_counts(dfg, fixture)
    output_values(dfg, fixture, "DFG")
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
        f"depthwise_conv should map to shared memory-reduction ADG: {mapping}",
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
        f"depthwise_conv CGRA aggregate should carry component final state: {cgra}",
    )
    require_component_references(evidence, cgra, "component_dfg_sim_report_identities", fixture.size)
    require_component_references(evidence, cgra, "component_cgra_sim_report_identities", fixture.size)
    output_values(cgra, fixture, "CGRA")
    assert_component_memory(cgra.get("final_memory_state"), fixture, "CGRA")
    require(EXPECTED_CGRA_CYCLES >= EXPECTED_DFG_CYCLES, "CGRA cycles must not be optimistic")

    require(
        comparison.get("status") == "pass"
        and comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == fixture.case
        and comparison.get("functional_comparison_status") == "pass"
        and comparison.get("memory_comparison_status") == "pass"
        and comparison.get("performance_comparison_status") == "pass",
        f"depthwise_conv comparison should pass with real final state: {comparison}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
