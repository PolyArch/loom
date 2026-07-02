#!/usr/bin/env python3
"""Assert row-complete normalize CGRA-sim evidence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from normalize_fixtures import NormalizeFixture, fixture_from_source, float_tokens


CASE = "normalize"
HARDWARE = "shared_signal_window_adg"
MAPPING_ID = f"{CASE}__workload_graph_set__{HARDWARE}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing artifact: {path}")
    data = json.loads(path.read_text())
    require(isinstance(data, dict), f"artifact must be a JSON object: {path}")
    return data


def require_status(label: str, artifact: dict[str, object]) -> None:
    require(artifact.get("status") == "pass", f"{label} evidence must pass: {artifact}")


def parse_f32(token: str) -> float:
    require(token.startswith("f32:"), f"expected f32 token, got {token}")
    return float(token.split(":", 1)[1])


def assert_float_tokens_close(tokens: object, expected: tuple[float, ...], label: str) -> None:
    require(isinstance(tokens, list), f"{label} should be a token list: {tokens}")
    require(len(tokens) == len(expected), f"{label} length mismatch: {tokens}")
    for index, (token, value) in enumerate(zip(tokens, expected)):
        require(isinstance(token, str), f"{label}[{index}] should be a string token: {token}")
        actual = parse_f32(token)
        require(
            math.isclose(actual, value, rel_tol=1.0e-5, abs_tol=1.0e-5),
            f"{label}[{index}] mismatch: {actual} != {value}",
        )


def require_real_routes(mapping: dict[str, object], fixture: NormalizeFixture) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"normalize mapping should expose routes: {mapping}")
    edges = {str(route.get("edge_ref", "")) for route in routes if isinstance(route, dict)}
    for edge in (
        "dataflow.load#0.result0->arith.addf#0.operand1",
        "dataflow.load#0.result0->arith.cmpf#0.operand0",
        "arith.select#0.result0->dataflow.carry#0.operand2",
        "dataflow.load#0.result0->arith.mulf#0.operand1",
        "arith.mulf#0.result0->dataflow.store#0.operand2",
    ):
        require(edge in edges, f"normalize route edge missing {edge}: {edges}")

    saw_switch = False
    saw_load = False
    saw_store = False
    saw_select = any("arith.select" in edge for edge in edges)
    saw_module_path = False
    component_graphs: set[str] = set()
    segment_count = 0
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        graph = route.get("component_graph")
        require(graph in fixture.graphs, f"route should name a normalize component graph: {route}")
        component_graphs.add(str(graph))
        segments = route.get("segments")
        require(isinstance(segments, list) and segments, f"route lacks segments: {route}")
        previous_sink = None
        for segment in segments:
            require(isinstance(segment, dict), f"route segment should be an object: {segment}")
            kind = segment.get("segment_kind")
            require(kind in {"resource_edge", "module_path", "buffer"}, f"bad route segment kind: {segment}")
            saw_module_path = saw_module_path or kind == "module_path"
            for field in ("hardware_ref", "source_endpoint", "sink_endpoint"):
                value = segment.get(field)
                require(isinstance(value, str) and "::" in value, f"segment lacks structured {field}: {segment}")
                require(".out" not in value and ".in" not in value, f"segment uses placeholder endpoint: {segment}")
            if previous_sink is not None:
                require(segment["source_endpoint"] == previous_sink, f"route is not contiguous: {route}")
            previous_sink = segment["sink_endpoint"]
            text = " ".join(
                str(segment.get(field, ""))
                for field in ("hardware_ref", "source_endpoint", "sink_endpoint")
            )
            saw_switch = saw_switch or "fabric.switch" in text
            saw_load = saw_load or "mem.load" in text
            saw_store = saw_store or "mem.store" in text
            segment_count += 1
    require(component_graphs == set(fixture.graphs), f"normalize routes missed components: {component_graphs}")
    require(segment_count > 30, f"normalize should route a nontrivial component graph set, got {segment_count}")
    require(saw_switch, "normalize routes should traverse a real switch")
    require(saw_load, "normalize routes should target real memory load ports")
    require(saw_store, "normalize routes should target real memory store ports")
    require(saw_select, "normalize routes should include the max select path")
    require(saw_module_path, "normalize routes should include Fabric module paths")


def require_memory(memory: object, fixture: NormalizeFixture, label: str) -> None:
    require(isinstance(memory, dict), f"{label} final memory must be present: {memory}")
    expected = fixture.expected_memory
    for key, values in expected.items():
        assert_float_tokens_close(memory.get(key), tuple(parse_f32(token) for token in values), f"{label} {key}")
    output = memory.get(f"{fixture.scale_graph}:arg3")
    require(output != float_tokens(fixture.zero_output_values), f"{label} output should not remain zero-filled")


def require_outputs(outputs: object, fixture: NormalizeFixture, label: str) -> None:
    require(isinstance(outputs, list), f"{label} final outputs should be a list: {outputs}")
    expected = ["none", f"f32:{fixture.sum_value:.6g}", "none", f"f32:{fixture.max_value:.6g}", "none"]
    require(outputs == expected, f"{label} final outputs should expose sum/max/scale components: {outputs}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} EVIDENCE_DIR")
    evidence = Path(argv[1])
    fixture = fixture_from_source()

    dfg = read_json(evidence / f"{CASE}.dfg.report.json")
    mapping = read_json(evidence / f"{CASE}.mapping.json")
    cgra = read_json(evidence / f"{CASE}.cgra.report.json")
    comparison = read_json(evidence / f"{CASE}.sim-comparison-report.json")

    for label, artifact in (
        ("DFG", dfg),
        ("mapping", mapping),
        ("CGRA", cgra),
        ("comparison", comparison),
    ):
        require_status(label, artifact)

    require(
        dfg.get("kind") == "dfg_sim_report"
        and dfg.get("workload") == CASE
        and dfg.get("graph") == "workload_graph_set"
        and dfg.get("component_graphs") == list(fixture.graphs)
        and dfg.get("dynamic_work_items") == fixture.size * 2 + fixture.size - 1
        and dfg.get("operation_fire_counts") == fixture.aggregate_fire_counts,
        f"normalize DFG aggregate should carry source-derived component evidence: {dfg}",
    )
    require_outputs(dfg.get("final_outputs"), fixture, "DFG")
    require_memory(dfg.get("final_memory_state"), fixture, "DFG")

    require(
        mapping.get("kind") == "pnr_mapping"
        and mapping.get("workload") == CASE
        and mapping.get("hardware") == HARDWARE
        and mapping.get("mapping_id") == MAPPING_ID
        and mapping.get("graph") == "workload_graph_set"
        and mapping.get("component_graphs") == list(fixture.graphs)
        and mapping.get("status") == "pass"
        and mapping.get("unrouted_edges") == 0
        and mapping.get("unplaced_records") == 0,
        f"normalize should map to shared signal-window ADG: {mapping}",
    )
    require_real_routes(mapping, fixture)

    require(
        cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == CASE
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == MAPPING_ID
        and cgra.get("functional_state_source") == "component_cgra_sim_reports_carried_from_dfg_sim_reports"
        and cgra.get("component_graphs") == list(fixture.graphs)
        and isinstance(cgra.get("dfg_cycles"), int)
        and isinstance(cgra.get("hardware_aware_cycles"), int)
        and cgra["hardware_aware_cycles"] >= cgra["dfg_cycles"],
        f"normalize CGRA report should carry hardware-aware aggregate evidence: {cgra}",
    )
    require_outputs(cgra.get("final_outputs"), fixture, "CGRA")
    require_memory(cgra.get("final_memory_state"), fixture, "CGRA")

    require(
        comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == CASE
        and comparison.get("functional_comparison_status") == "pass"
        and comparison.get("memory_comparison_status") == "pass"
        and comparison.get("performance_comparison_status") == "pass"
        and comparison.get("difference_classification") == "expected_hardware_constraint",
        f"normalize comparison should pass real functional and memory checks: {comparison}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
