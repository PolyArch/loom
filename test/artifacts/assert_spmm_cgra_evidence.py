#!/usr/bin/env python3
"""Assert row-complete spmm CGRA-sim evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from spmm_fixtures import fixture_from_source


CASE = "spmm"
GRAPH = "g_spmm_kernel_0"
HARDWARE = "shared_memory_reduction_adg"
MAPPING_ID = f"{CASE}__{GRAPH}__{HARDWARE}"
REQUIRED_ROUTE_EDGES = {
    "dataflow.constant#2.result0->arith.index_cast#1.operand0",
    "arith.index_cast#1.result0->arith.addi#1.operand1",
    "arith.index_cast#0.result0->dataflow.load#0.operand1",
    "arith.addi#5.result0->dataflow.store#1.operand1",
    "dataflow.load#4.result0->arith.muli#3.operand0",
    "dataflow.load#5.result0->arith.addi#4.operand0",
}


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


def require_memory(memory: object, expected_memory: dict[str, list[str]], label: str) -> None:
    require(isinstance(memory, dict), f"{label} final memory must be present: {memory}")
    for key, expected in expected_memory.items():
        require(memory.get(key) == expected, f"{label} final_memory_state.{key} mismatch: {memory.get(key)}")
    require(memory.get("arg5") == ["i32:11", "i32:14", "i32:29", "i32:36"], f"{label} output mismatch: {memory}")


def require_real_routes(mapping: dict[str, object]) -> None:
    routes = mapping.get("routes")
    require(isinstance(routes, list) and routes, f"spmm mapping should expose routes: {mapping}")
    edges = {str(route.get("edge_ref", "")) for route in routes if isinstance(route, dict)}
    missing = REQUIRED_ROUTE_EDGES.difference(edges)
    require(not missing, f"spmm route edge set missed required edges: {missing}: {edges}")
    saw_switch = False
    saw_load = False
    saw_store = False
    saw_buffer = False
    saw_module_path = False
    saw_adapter_edge = False
    segment_count = 0
    for route in routes:
        require(isinstance(route, dict), f"route should be an object: {route}")
        require(route.get("status") == "routed", f"route should be routed: {route}")
        edge = str(route.get("edge_ref", ""))
        segments = route.get("segments")
        require(isinstance(segments, list) and segments, f"route lacks segments: {route}")
        previous_sink = None
        for segment in segments:
            require(isinstance(segment, dict), f"route segment should be an object: {segment}")
            kind = segment.get("segment_kind")
            require(kind in {"resource_edge", "module_path", "buffer"}, f"bad route segment kind: {segment}")
            saw_buffer = saw_buffer or kind == "buffer"
            saw_module_path = saw_module_path or kind == "module_path"
            for field in ("hardware_ref", "source_endpoint", "sink_endpoint"):
                value = segment.get(field)
                require(isinstance(value, str) and "::" in value, f"segment lacks structured {field}: {segment}")
                require(not value.endswith(".out") and not value.endswith(".in"), f"segment uses placeholder endpoint: {segment}")
            if previous_sink is not None:
                require(segment["source_endpoint"] == previous_sink, f"route is not contiguous: {route}")
            previous_sink = segment["sink_endpoint"]
            saw_switch = saw_switch or "fabric.switch" in str(segment.get("hardware_ref", ""))
            saw_load = saw_load or "mem.load" in str(segment.get("sink_endpoint", ""))
            saw_store = saw_store or "mem.store" in str(segment.get("sink_endpoint", ""))
            if edge == "arith.index_cast#1.result0->arith.addi#1.operand1":
                saw_adapter_edge = saw_adapter_edge or kind == "buffer"
            segment_count += 1
    require(segment_count == 177, f"spmm should route 177 segments, got {segment_count}")
    require(saw_switch, "spmm routes should traverse a real switch")
    require(saw_load, "spmm routes should target real memory load ports")
    require(saw_store, "spmm routes should target real memory store ports")
    require(saw_buffer, "spmm routes should include real Fabric buffers")
    require(saw_module_path, "spmm routes should include Fabric module paths")
    require(saw_adapter_edge, "spmm computed-address adapter route should traverse a real buffer")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} EVIDENCE_DIR")
    evidence = Path(argv[1])
    fixture = fixture_from_source()
    expected_memory = fixture.expected_memory

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
        and dfg.get("graph") == GRAPH
        and dfg.get("dynamic_work_items") == 4
        and dfg.get("optimistic_cycles") == 488
        and dfg.get("operation_fire_counts") == fixture.expected_fire_counts
        and dfg.get("final_outputs") == fixture.final_outputs,
        f"spmm DFG report should carry source-derived sparse-dense evidence: {dfg}",
    )
    require_memory(dfg.get("final_memory_state"), expected_memory, "DFG")

    require(
        mapping.get("kind") == "pnr_mapping"
        and mapping.get("workload") == CASE
        and mapping.get("graph") == GRAPH
        and mapping.get("hardware") == HARDWARE
        and mapping.get("mapping_id") == MAPPING_ID
        and mapping.get("placed_records") == 40
        and mapping.get("routed_edges") == 35
        and mapping.get("unrouted_edges") == 0
        and mapping.get("unplaced_records") == 0
        and mapping.get("config_records") == 993,
        f"spmm should map to shared memory-reduction ADG: {mapping}",
    )
    require_real_routes(mapping)

    require(
        cgra.get("kind") == "cgra_sim_report"
        and cgra.get("workload") == CASE
        and cgra.get("hardware") == HARDWARE
        and cgra.get("mapping_id") == MAPPING_ID
        and cgra.get("functional_state_source") == "carried_from_dfg_sim_report"
        and cgra.get("dfg_cycles") == 488
        and cgra.get("hardware_aware_cycles") == 740
        and cgra.get("route_segments") == 177
        and cgra.get("config_records") == 993
        and cgra.get("placed_records") == 40
        and cgra.get("routed_edges") == 35
        and cgra.get("final_outputs") == fixture.final_outputs,
        f"spmm CGRA report should carry hardware-aware evidence: {cgra}",
    )
    require_memory(cgra.get("final_memory_state"), expected_memory, "CGRA")
    require(740 >= 488, "CGRA cycles must not be more optimistic than DFG")

    require(
        comparison.get("kind") == "sim_comparison_report"
        and comparison.get("workload") == CASE
        and comparison.get("functional_comparison_status") == "pass"
        and comparison.get("memory_comparison_status") == "pass"
        and comparison.get("performance_comparison_status") == "pass"
        and comparison.get("dfg_sim_cycles") == 488
        and comparison.get("cgra_sim_cycles") == 740
        and comparison.get("performance_delta_cycles") == 252
        and comparison.get("difference_classification") == "expected_hardware_constraint",
        f"spmm comparison should pass real functional and memory checks: {comparison}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
