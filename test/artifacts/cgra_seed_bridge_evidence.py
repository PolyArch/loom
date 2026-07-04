#!/usr/bin/env python3
"""Content checks for CGRA seed rows bridged into LoomBench."""

from __future__ import annotations

import json
from pathlib import Path


def _load_reports(evidence_dir: Path, case: str) -> tuple[dict, dict, dict, dict]:
    dfg = json.loads((evidence_dir / f"{case}.dfg.report.json").read_text())
    mapping = json.loads((evidence_dir / f"{case}.mapping.json").read_text())
    cgra = json.loads((evidence_dir / f"{case}.cgra.report.json").read_text())
    comparison = json.loads((evidence_dir / f"{case}.sim-comparison-report.json").read_text())
    return dfg, mapping, cgra, comparison


def _assert_operation_fire_counts(case: str, dfg: dict, expected_counts: dict[str, int]) -> None:
    actual_counts = dfg.get("operation_fire_counts", {})
    for op_name, expected in expected_counts.items():
        actual = actual_counts.get(op_name)
        if actual != expected:
            raise AssertionError(f"{case} {op_name} fire count should be {expected}, got {actual}: {dfg}")


def _assert_mapping(
    case: str,
    mapping: dict,
    *,
    hardware: str,
    placed_records: int,
    routed_edges: int,
    config_records: int,
    required_edges: set[str],
    expected_diagnostics: list[str] | None = None,
) -> None:
    diagnostics = (
        ["mapped software graph to fabric resources"] if expected_diagnostics is None else expected_diagnostics
    )
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != hardware
        or mapping.get("placed_records") != placed_records
        or mapping.get("routed_edges") != routed_edges
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != config_records
        or mapping.get("diagnostics") != diagnostics
    ):
        raise AssertionError(f"{case} mapping should route on {hardware}: {mapping}")
    routes = mapping.get("routes")
    if not isinstance(routes, list):
        raise AssertionError(f"{case} mapping should expose routes: {mapping}")
    by_edge = {
        str(route.get("edge_ref")): route
        for route in routes
        if isinstance(route, dict) and route.get("edge_ref") is not None
    }
    missing_edges = required_edges - set(by_edge)
    if missing_edges:
        raise AssertionError(f"{case} mapping missed required route edges {sorted(missing_edges)}: {mapping}")
    for edge_ref in required_edges:
        route = by_edge[edge_ref]
        if route.get("status") != "routed":
            raise AssertionError(f"{case} route should be routed for {edge_ref}: {route}")
        segments = route.get("segments")
        if not isinstance(segments, list) or not segments:
            raise AssertionError(f"{case} route should expose concrete segments for {edge_ref}: {route}")
        for segment in segments:
            if not isinstance(segment, dict):
                raise AssertionError(f"{case} route should use structured segments for {edge_ref}: {route}")
            endpoints = (
                str(segment.get("source_endpoint", "")),
                str(segment.get("sink_endpoint", "")),
            )
            if any(endpoint.endswith(".out") or endpoint.endswith(".in") for endpoint in endpoints):
                raise AssertionError(f"{case} mapping uses placeholder endpoint for {edge_ref}: {route}")


def _assert_cgra_and_comparison(
    case: str,
    dfg: dict,
    cgra: dict,
    comparison: dict,
    *,
    hardware: str,
    dfg_cycles: int,
    cgra_cycles: int,
    routed_edges: int,
    route_segments: int,
    final_outputs: list[str],
    functional_state_source: str = "carried_from_dfg_sim_report",
) -> None:
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != hardware
        or cgra.get("dfg_cycles") != dfg_cycles
        or cgra.get("hardware_aware_cycles") != cgra_cycles
        or cgra.get("routed_edges") != routed_edges
        or cgra.get("route_segments") != route_segments
        or cgra.get("final_outputs") != final_outputs
        or cgra.get("functional_state_source") != functional_state_source
        or cgra.get("dfg_cycles") != dfg.get("optimistic_cycles")
        or cgra.get("hardware_aware_cycles", 0) < cgra.get("dfg_cycles", 0)
    ):
        raise AssertionError(f"{case} CGRA evidence should preserve DFG state and routed timing: {cgra}")
    if (
        comparison.get("status") != "pass"
        or comparison.get("functional_comparison_status") != "pass"
        or comparison.get("memory_comparison_status") != "pass"
        or comparison.get("performance_comparison_status") != "pass"
        or comparison.get("dfg_sim_cycles") != dfg_cycles
        or comparison.get("cgra_sim_cycles") != cgra_cycles
        or comparison.get("cgra_sim_cycles", 0) < comparison.get("dfg_sim_cycles", 0)
    ):
        raise AssertionError(f"{case} comparison should pass functional, memory, and timing checks: {comparison}")


def _assert_memory_window(
    case: str,
    memory: dict,
    key: str,
    *,
    length: int,
    head: list[str],
    tail: list[str],
) -> None:
    values = memory.get(key)
    if not isinstance(values, list) or len(values) != length or values[: len(head)] != head or values[-len(tail) :] != tail:
        raise AssertionError(f"{case} memory {key} should match source-derived window: {memory}")


def _i32_values(values: list[int]) -> list[str]:
    return [f"i32:{value}" for value in values]


def _f32_values(values: list[str]) -> list[str]:
    return [f"f32:{value}" for value in values]


def _assert_memory_windows(
    case: str,
    dfg: dict,
    cgra: dict,
    windows: dict[str, tuple[int, list[str], list[str]]],
) -> None:
    dfg_memory = dfg.get("final_memory_state")
    cgra_memory = cgra.get("final_memory_state")
    if not isinstance(dfg_memory, dict) or not isinstance(cgra_memory, dict):
        raise AssertionError(f"{case} should expose final memory state in both simulators: {dfg} {cgra}")
    for key, (length, head, tail) in windows.items():
        _assert_memory_window(case, dfg_memory, key, length=length, head=head, tail=tail)
        _assert_memory_window(case, cgra_memory, key, length=length, head=head, tail=tail)


def _assert_seed_case_evidence(
    evidence_dir: Path,
    case: str,
    *,
    graph: str,
    dynamic_work_items: int,
    event_count: int,
    dfg_cycles: int,
    cgra_cycles: int,
    final_outputs: list[str],
    hardware: str,
    placed_records: int,
    routed_edges: int,
    config_records: int,
    route_segments: int,
    operation_fire_counts: dict[str, int],
    required_edges: set[str],
    memory_windows: dict[str, tuple[int, list[str], list[str]]],
    functional_state_source: str = "carried_from_dfg_sim_report",
    expected_diagnostics: list[str] | None = None,
    expected_mapping_diagnostics: list[str] | None = None,
) -> None:
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    diagnostics = [] if expected_diagnostics is None else expected_diagnostics
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != graph
        or dfg.get("dynamic_work_items") != dynamic_work_items
        or dfg.get("event_count") != event_count
        or dfg.get("optimistic_cycles") != dfg_cycles
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("diagnostics") != diagnostics
    ):
        raise AssertionError(f"{case} DFG evidence should match source-derived execution: {dfg}")
    _assert_operation_fire_counts(case, dfg, operation_fire_counts)
    _assert_mapping(
        case,
        mapping,
        hardware=hardware,
        placed_records=placed_records,
        routed_edges=routed_edges,
        config_records=config_records,
        required_edges=required_edges,
        expected_diagnostics=expected_mapping_diagnostics,
    )
    _assert_memory_windows(case, dfg, cgra, memory_windows)
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware=hardware,
        dfg_cycles=dfg_cycles,
        cgra_cycles=cgra_cycles,
        routed_edges=routed_edges,
        route_segments=route_segments,
        final_outputs=final_outputs,
        functional_state_source=functional_state_source,
    )


def assert_correlation_evidence(evidence_dir: Path) -> None:
    case = "correlation"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    final_outputs = ["none", "f32:16"]
    expected_memory = {"arg6": ["f32:1"] * 16, "arg7": ["f32:1"] * 16}
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_correlation_kernel_0_0"
        or dfg.get("dynamic_work_items") != 16
        or dfg.get("optimistic_cycles") != 485
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real constant-window correlation: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addi": 17,
            "arith.andi": 17,
            "arith.index_cast": 66,
            "dataflow.carry": 17,
            "dataflow.invariant": 38,
            "dataflow.load": 32,
            "dataflow.stream": 17,
            "dataflow.sync": 16,
            "llvm.intr.fmuladd": 16,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=10,
        routed_edges=15,
        config_records=317,
        required_edges={
            "dataflow.load#0.result0->llvm.intr.fmuladd#0.operand0",
            "dataflow.load#1.result0->llvm.intr.fmuladd#0.operand1",
            "llvm.intr.fmuladd#0.result0->dataflow.carry#0.operand2",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry real input windows: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=485,
        cgra_cycles=562,
        routed_edges=15,
        route_segments=57,
        final_outputs=final_outputs,
    )


def assert_downsample_avg_evidence(evidence_dir: Path) -> None:
    case = "downsample_avg"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    final_outputs = ["none", "f32:22", "f32:5.500000"]
    expected_memory = {"arg4": ["f32:1", "f32:4", "f32:7", "f32:10"]}
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_downsample_avg_0_0"
        or dfg.get("dynamic_work_items") != 4
        or dfg.get("optimistic_cycles") != 77
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real downsample average: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addf": 4,
            "arith.index_cast": 5,
            "arith.mulf": 5,
            "dataflow.carry": 5,
            "dataflow.invariant": 7,
            "dataflow.load": 4,
            "dataflow.stream": 5,
            "dataflow.sync": 4,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=7,
        routed_edges=9,
        config_records=190,
        required_edges={
            "dataflow.load#0.result0->arith.addf#0.operand1",
            "arith.addf#0.result0->dataflow.carry#0.operand2",
            "dataflow.carry#0.result0->arith.mulf#0.operand0",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry real input window: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=77,
        cgra_cycles=123,
        routed_edges=9,
        route_segments=33,
        final_outputs=final_outputs,
    )


def assert_integrate_trapz_evidence(evidence_dir: Path) -> None:
    case = "integrate_trapz"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    final_outputs = ["none", "i64:1", "f32:0.335938"]
    expected_memory = {
        "arg4": [
            "f32:0",
            "f32:0.125000",
            "f32:0.250000",
            "f32:0.375000",
            "f32:0.500000",
            "f32:0.625000",
            "f32:0.750000",
            "f32:0.875000",
            "f32:1",
        ],
        "arg5": [
            "f32:0",
            "f32:0.015625",
            "f32:0.062500",
            "f32:0.140625",
            "f32:0.250000",
            "f32:0.390625",
            "f32:0.562500",
            "f32:0.765625",
            "f32:1",
        ],
    }
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_integrate_trapz_red_0_0"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 386
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real trapezoid inputs: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addf": 8,
            "arith.addi": 9,
            "arith.index_cast": 28,
            "arith.mulf": 8,
            "arith.subf": 8,
            "dataflow.carry": 20,
            "dataflow.invariant": 21,
            "dataflow.load": 32,
            "dataflow.stream": 9,
            "dataflow.sync": 8,
            "llvm.intr.fmuladd": 8,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=15,
        routed_edges=25,
        config_records=534,
        required_edges={
            "arith.subf#0.result0->llvm.intr.fmuladd#0.operand1",
            "arith.mulf#0.result0->llvm.intr.fmuladd#0.operand0",
            "llvm.intr.fmuladd#0.result0->dataflow.carry#1.operand2",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry real trapezoid inputs: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=386,
        cgra_cycles=519,
        routed_edges=25,
        route_segments=99,
        final_outputs=final_outputs,
    )


def assert_rotate_bits_evidence(evidence_dir: Path) -> None:
    case = "rotate_bits"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    expected_memory = {
        "arg1": [f"i32:{-1985229329 + 16909320 * index}" for index in range(32)],
        "arg3": [f"i32:{index}" for index in range(32)],
        "arg5": [
            "i32:0",
            "i32:8388608",
            "i32:1",
            "i32:384",
            "i32:131072",
            "i32:41943040",
            "i32:3",
            "i32:896",
            "i32:262144",
            "i32:75497472",
            "i32:5",
            "i32:1408",
            "i32:393216",
            "i32:109051904",
            "i32:7",
            "i32:1920",
            "i32:524288",
            "i32:142606336",
            "i32:9",
            "i32:2432",
            "i32:655360",
            "i32:176160768",
            "i32:11",
            "i32:2944",
            "i32:786432",
            "i32:209715200",
            "i32:13",
            "i32:3456",
            "i32:917504",
            "i32:243269632",
            "i32:15",
            "i32:3968",
        ],
    }
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_rotate_bits_0_0"
        or dfg.get("dynamic_work_items") != 32
        or dfg.get("optimistic_cycles") != 551
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real rotate fixtures: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.andi": 32,
            "arith.cmpi": 32,
            "arith.select": 32,
            "dataflow.load": 64,
            "dataflow.store": 32,
            "dataflow.sync": 32,
            "llvm.intr.fshl": 32,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=8,
        routed_edges=12,
        config_records=247,
        required_edges={
            "llvm.intr.fshl#0.result0->arith.select#0.operand2",
            "arith.select#0.result0->dataflow.store#0.operand2",
            "dataflow.load#1.result0->llvm.intr.fshl#0.operand0",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry rotated outputs: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=551,
        cgra_cycles=618,
        routed_edges=12,
        route_segments=44,
        final_outputs=["none"],
    )


def assert_rle_encode_evidence(evidence_dir: Path) -> None:
    case = "rle_encode"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    final_outputs = ["none", "i32:2", "i32:7", "i32:6"]
    expected_memory = {
        "arg4": [
            "i32:1",
            "i32:1",
            "i32:1",
            "i32:2",
            "i32:2",
            "i32:3",
            "i32:3",
            "i32:3",
            "i32:3",
            "i32:4",
            "i32:4",
            "i32:4",
            "i32:4",
            "i32:4",
            "i32:5",
            "i32:6",
            "i32:6",
            "i32:6",
            "i32:7",
            "i32:7",
        ],
        "arg6": ["i32:1", "i32:2", "i32:3", "i32:4", "i32:5", "i32:6"] + ["i32:0"] * 14,
        "arg7": ["i32:3", "i32:2", "i32:4", "i32:5", "i32:1", "i32:3"] + ["i32:0"] * 14,
    }
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_rle_encode_kernel_red_0_0"
        or dfg.get("dynamic_work_items") != 19
        or dfg.get("optimistic_cycles") != 297
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real run-length encoding: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addi": 19,
            "arith.cmpi": 19,
            "arith.index_cast": 37,
            "dataflow.load": 25,
            "dataflow.store": 12,
            "scf.if": 19,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=9,
        routed_edges=3,
        config_records=116,
        required_edges={
            "arith.index_cast#0.result0->dataflow.load#0.operand1",
            "arith.index_cast#1.result0->dataflow.load#1.operand1",
            "dataflow.load#0.result0->arith.cmpi#0.operand0",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry encoded runs: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=297,
        cgra_cycles=343,
        routed_edges=3,
        route_segments=17,
        final_outputs=final_outputs,
    )


def assert_stream_update_evidence(evidence_dir: Path) -> None:
    case = "stream_update"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    final_outputs = ["none", "i32:1976", "i32:30"]
    expected_memory = {"arg4": [f"i32:{value}" for value in range(2, 66, 2)]}
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_stream_update_kernel_red_0_0"
        or dfg.get("dynamic_work_items") != 10
        or dfg.get("optimistic_cycles") != 1086
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real stream update: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addi": 120,
            "arith.cmpi": 60,
            "arith.index_cast": 60,
            "arith.remui": 60,
            "arith.shrui": 60,
            "dataflow.load": 60,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_memory_reduction_adg",
        placed_records=6,
        routed_edges=4,
        config_records=114,
        required_edges={
            "arith.addi#0.result0->arith.remui#0.operand0",
            "arith.remui#0.result0->dataflow.load#0.operand1",
            "dataflow.load#0.result0->arith.addi#1.operand0",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry real stream memory: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_memory_reduction_adg",
        dfg_cycles=1086,
        cgra_cycles=1115,
        routed_edges=4,
        route_segments=18,
        final_outputs=final_outputs,
    )


def assert_jacobi_stencil_7pt_evidence(evidence_dir: Path) -> None:
    case = "jacobi_stencil_7pt"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    expected_output = [
        "f32:0.579209",
        "f32:4.262399",
        "f32:0.241253",
        "f32:-3.239180",
        "f32:1.247716",
        "f32:0.685203",
        "f32:0.297809",
        "f32:0.649615",
    ]
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_jacobi_stencil_7pt_kernel_0_0"
        or dfg.get("dynamic_work_items") != 8
        or dfg.get("optimistic_cycles") != 627
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real stencil execution: {dfg}")
    memory = dfg.get("final_memory_state", {})
    if not isinstance(memory, dict) or memory.get("arg13") != expected_output:
        raise AssertionError(f"{case} DFG evidence should write real stencil outputs: {dfg}")
    _assert_memory_window(
        case,
        memory,
        "arg6",
        length=64,
        head=[
            "f32:3.929384",
            "f32:-4.277213",
            "f32:-5.462971",
            "f32:1.026295",
            "f32:4.389380",
            "f32:-1.537871",
            "f32:9.615284",
            "f32:3.696595",
        ],
        tail=["f32:3.386276", "f32:1.718731", "f32:2.498070", "f32:3.493781"],
    )
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addf": 40,
            "arith.addi": 48,
            "arith.andi": 24,
            "arith.index_cast": 104,
            "arith.mulf": 8,
            "arith.ori": 16,
            "arith.shli": 16,
            "dataflow.load": 48,
            "dataflow.store": 8,
            "dataflow.sync": 8,
            "llvm.trunc": 24,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_signal_window_adg",
        placed_records=30,
        routed_edges=40,
        config_records=950,
        required_edges={
            "arith.addf#4.result0->arith.mulf#0.operand0",
            "arith.mulf#0.result0->dataflow.store#0.operand2",
            "dataflow.store#0.result0->dataflow.sync#0.operand6",
        },
    )
    cgra_memory = cgra.get("final_memory_state", {})
    if not isinstance(cgra_memory, dict) or cgra_memory.get("arg13") != expected_output:
        raise AssertionError(f"{case} CGRA evidence should carry real stencil outputs: {cgra}")
    _assert_memory_window(
        case,
        cgra_memory,
        "arg6",
        length=64,
        head=[
            "f32:3.929384",
            "f32:-4.277213",
            "f32:-5.462971",
            "f32:1.026295",
            "f32:4.389380",
            "f32:-1.537871",
            "f32:9.615284",
            "f32:3.696595",
        ],
        tail=["f32:3.386276", "f32:1.718731", "f32:2.498070", "f32:3.493781"],
    )
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_signal_window_adg",
        dfg_cycles=627,
        cgra_cycles=859,
        routed_edges=40,
        route_segments=174,
        final_outputs=["none"],
    )


def assert_vecnorm_l1_evidence(evidence_dir: Path) -> None:
    case = "vecnorm_l1"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    final_outputs = ["none", "i32:171"]
    expected_memory = {"arg4": [f"i32:{[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5][index % 11]}" for index in range(64)]}
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_vecnorm_l1_red_0_0"
        or dfg.get("dynamic_work_items") != 64
        or dfg.get("optimistic_cycles") != 714
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real L1 vector norm: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addi": 64,
            "arith.index_cast": 65,
            "dataflow.carry": 65,
            "dataflow.load": 64,
            "dataflow.stream": 65,
            "dataflow.sync": 64,
            "llvm.intr.abs": 64,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=6,
        routed_edges=7,
        config_records=156,
        required_edges={
            "dataflow.load#0.result0->llvm.intr.abs#0.operand0",
            "llvm.intr.abs#0.result0->arith.addi#0.operand0",
            "arith.addi#0.result0->dataflow.carry#0.operand2",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry real vector input: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=714,
        cgra_cycles=752,
        routed_edges=7,
        route_segments=27,
        final_outputs=final_outputs,
    )


def assert_gemv_evidence(evidence_dir: Path) -> None:
    case = "gemv"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    final_outputs = ["none", "i32:55", "i32:110"]
    expected_memory = {
        "arg4": ["i32:1", "i32:2", "i32:3", "i32:4", "i32:5"],
        "arg5": ["i32:1", "i32:2", "i32:3", "i32:4", "i32:5"],
    }
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_gemv_kernel_0_0"
        or dfg.get("dynamic_work_items") != 5
        or dfg.get("optimistic_cycles") != 116
        or dfg.get("event_count") != 57
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real matrix-vector inputs: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addi": 5,
            "arith.index_cast": 6,
            "arith.muli": 5,
            "arith.shli": 6,
            "dataflow.carry": 6,
            "dataflow.invariant": 8,
            "dataflow.load": 10,
            "dataflow.stream": 6,
            "dataflow.sync": 5,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=9,
        routed_edges=13,
        config_records=282,
        required_edges={
            "dataflow.load#0.result0->arith.muli#0.operand1",
            "dataflow.load#1.result0->arith.muli#0.operand0",
            "arith.muli#0.result0->arith.addi#0.operand0",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry real matrix-vector inputs: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=116,
        cgra_cycles=186,
        routed_edges=13,
        route_segments=51,
        final_outputs=final_outputs,
    )


def assert_matmul_evidence(evidence_dir: Path) -> None:
    case = "matmul"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    final_outputs = ["none", "i32:58"]
    expected_memory = {
        "arg5": ["i32:1", "i32:2", "i32:3", "i32:4", "i32:5", "i32:6"],
        "arg8": ["i32:7", "i32:8", "i32:9", "i32:10", "i32:11", "i32:12"],
    }
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_matmul_kernel_0_0"
        or dfg.get("dynamic_work_items") != 3
        or dfg.get("optimistic_cycles") != 125
        or dfg.get("event_count") != 78
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real matrix product inputs: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addi": 11,
            "arith.index_cast": 17,
            "arith.muli": 7,
            "dataflow.carry": 4,
            "dataflow.invariant": 18,
            "dataflow.load": 6,
            "dataflow.stream": 4,
            "dataflow.sync": 3,
            "llvm.trunc": 8,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=15,
        routed_edges=21,
        config_records=470,
        required_edges={
            "dataflow.load#0.result0->arith.muli#1.operand1",
            "dataflow.load#1.result0->arith.muli#1.operand0",
            "arith.muli#1.result0->arith.addi#2.operand0",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry real matrix product inputs: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=125,
        cgra_cycles=235,
        routed_edges=21,
        route_segments=85,
        final_outputs=final_outputs,
    )


def assert_matvec_evidence(evidence_dir: Path) -> None:
    case = "matvec"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    final_outputs = ["none", "i32:55"]
    expected_memory = {
        "arg4": ["i32:1", "i32:2", "i32:3", "i32:4", "i32:5"],
        "arg5": ["i32:1", "i32:2", "i32:3", "i32:4", "i32:5"],
    }
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_matvec_kernel_0_0"
        or dfg.get("dynamic_work_items") != 5
        or dfg.get("optimistic_cycles") != 100
        or dfg.get("event_count") != 43
        or dfg.get("final_outputs") != final_outputs
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real matrix-vector reduction: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.addi": 5,
            "arith.index_cast": 6,
            "arith.muli": 5,
            "dataflow.carry": 6,
            "dataflow.load": 10,
            "dataflow.stream": 6,
            "dataflow.sync": 5,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=7,
        routed_edges=10,
        config_records=221,
        required_edges={
            "dataflow.load#0.result0->arith.muli#0.operand1",
            "dataflow.load#1.result0->arith.muli#0.operand0",
            "arith.muli#0.result0->arith.addi#0.operand0",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry real matrix-vector reduction inputs: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=100,
        cgra_cycles=156,
        routed_edges=10,
        route_segments=40,
        final_outputs=final_outputs,
    )


def assert_upsample_evidence(evidence_dir: Path) -> None:
    case = "upsample"
    dfg, mapping, cgra, comparison = _load_reports(evidence_dir, case)
    expected_memory = {
        "arg1": ["f32:2", "f32:5", "f32:8", "f32:11"],
        "arg3": [
            "f32:2",
            "f32:0",
            "f32:0",
            "f32:0",
            "f32:5",
            "f32:0",
            "f32:0",
            "f32:0",
            "f32:8",
            "f32:0",
            "f32:0",
            "f32:0",
            "f32:11",
            "f32:0",
            "f32:0",
            "f32:0",
        ],
    }
    if (
        dfg.get("status") != "pass"
        or dfg.get("graph") != "g_t_upsample_0_0"
        or dfg.get("dynamic_work_items") != 4
        or dfg.get("optimistic_cycles") != 67
        or dfg.get("event_count") != 28
        or dfg.get("final_outputs") != ["none"]
        or dfg.get("final_memory_state") != expected_memory
        or dfg.get("diagnostics") != []
    ):
        raise AssertionError(f"{case} DFG evidence should match real sparse upsample output: {dfg}")
    _assert_operation_fire_counts(
        case,
        dfg,
        {
            "arith.index_cast": 4,
            "arith.shli": 4,
            "arith.shrui": 4,
            "dataflow.constant": 4,
            "dataflow.load": 4,
            "dataflow.store": 4,
            "dataflow.sync": 4,
        },
    )
    _assert_mapping(
        case,
        mapping,
        hardware="shared_reduction_adg",
        placed_records=6,
        routed_edges=6,
        config_records=141,
        required_edges={
            "arith.shli#0.result0->arith.shrui#0.operand0",
            "arith.shrui#0.result0->dataflow.store#0.operand1",
            "dataflow.load#0.result0->dataflow.store#0.operand2",
        },
    )
    if cgra.get("final_memory_state") != expected_memory:
        raise AssertionError(f"{case} CGRA evidence should carry real sparse upsample output: {cgra}")
    _assert_cgra_and_comparison(
        case,
        dfg,
        cgra,
        comparison,
        hardware="shared_reduction_adg",
        dfg_cycles=67,
        cgra_cycles=111,
        routed_edges=6,
        route_segments=24,
        final_outputs=["none"],
    )


def assert_vecsum_while_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "vecsum-while",
        graph="g_t_vecsum_while_kernel_red_0_0",
        dynamic_work_items=16,
        event_count=99,
        dfg_cycles=169,
        cgra_cycles=203,
        final_outputs=["none", "i32:120"],
        hardware="shared_reduction_adg",
        placed_records=5,
        routed_edges=6,
        config_records=137,
        route_segments=24,
        operation_fire_counts={
            "arith.addi": 16,
            "arith.index_cast": 17,
            "dataflow.carry": 17,
            "dataflow.load": 16,
            "dataflow.stream": 17,
            "dataflow.sync": 16,
        },
        required_edges={
            "arith.addi#0.result0->dataflow.carry#0.operand2",
            "dataflow.load#0.result0->arith.addi#0.operand0",
            "dataflow.stream#0.result0->dataflow.load#0.operand1",
        },
        memory_windows={
            "arg4": (16, _i32_values(list(range(8))), _i32_values(list(range(8, 16)))),
        },
    )


def assert_reduction_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "reduction",
        graph="g_t_reduce_sum_red_0_0",
        dynamic_work_items=128,
        event_count=771,
        dfg_cycles=1289,
        cgra_cycles=1323,
        final_outputs=["none", "i32:8128"],
        hardware="shared_reduction_adg",
        placed_records=5,
        routed_edges=6,
        config_records=137,
        route_segments=24,
        operation_fire_counts={
            "arith.addi": 128,
            "arith.index_cast": 129,
            "dataflow.carry": 129,
            "dataflow.load": 128,
            "dataflow.stream": 129,
            "dataflow.sync": 128,
        },
        required_edges={
            "arith.addi#0.result0->dataflow.carry#0.operand2",
            "dataflow.load#0.result0->arith.addi#0.operand0",
            "dataflow.stream#0.result0->dataflow.load#0.operand1",
        },
        memory_windows={
            "arg4": (128, _i32_values(list(range(8))), _i32_values(list(range(120, 128)))),
        },
    )


def assert_prefix_sum_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "prefix_sum",
        graph="g_t_prefix_sum_red_0_0",
        dynamic_work_items=64,
        event_count=451,
        dfg_cycles=1034,
        cgra_cycles=1091,
        final_outputs=["none", "i32:2016"],
        hardware="shared_reduction_adg",
        placed_records=6,
        routed_edges=9,
        config_records=202,
        route_segments=37,
        operation_fire_counts={
            "arith.addi": 64,
            "arith.index_cast": 65,
            "dataflow.carry": 65,
            "dataflow.load": 64,
            "dataflow.store": 64,
            "dataflow.stream": 65,
            "dataflow.sync": 64,
        },
        required_edges={
            "arith.addi#0.result0->dataflow.store#0.operand2",
            "dataflow.store#0.result0->dataflow.sync#0.operand1",
            "dataflow.stream#0.result0->dataflow.store#0.operand1",
        },
        memory_windows={
            "arg4": (64, _i32_values(list(range(8))), _i32_values(list(range(56, 64)))),
            "arg5": (
                64,
                _i32_values([0, 1, 3, 6, 10, 15, 21, 28]),
                _i32_values([1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016]),
            ),
        },
    )


def assert_prefix_sum_inclusive_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "prefix_sum_inclusive",
        graph="g_t_prefix_sum_inclusive_kernel_red_0_0",
        dynamic_work_items=1023,
        event_count=7164,
        dfg_cycles=16378,
        cgra_cycles=16435,
        final_outputs=["none", "i32:5620"],
        hardware="shared_reduction_adg",
        placed_records=6,
        routed_edges=9,
        config_records=202,
        route_segments=37,
        operation_fire_counts={
            "arith.addi": 1023,
            "arith.index_cast": 1024,
            "dataflow.carry": 1024,
            "dataflow.load": 1023,
            "dataflow.store": 1023,
            "dataflow.stream": 1024,
            "dataflow.sync": 1023,
        },
        required_edges={
            "arith.addi#0.result0->dataflow.store#0.operand2",
            "dataflow.store#0.result0->dataflow.sync#0.operand1",
            "dataflow.stream#0.result0->dataflow.store#0.operand1",
        },
        memory_windows={
            "arg4": (
                1024,
                _i32_values([1, 2, 3, 4, 5, 6, 7, 8]),
                _i32_values([7, 8, 9, 10, 1, 2, 3, 4]),
            ),
            "arg5": (
                1024,
                _i32_values([0, 3, 6, 10, 15, 21, 28, 36]),
                _i32_values([5583, 5591, 5600, 5610, 5611, 5613, 5616, 5620]),
            ),
        },
    )


def assert_vecnorm_l2_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "vecnorm_l2",
        graph="g_t_vecnorm_l2_red_0_0",
        dynamic_work_items=64,
        event_count=451,
        dfg_cycles=842,
        cgra_cycles=886,
        final_outputs=["none", "i32:619"],
        hardware="shared_reduction_adg",
        placed_records=6,
        routed_edges=8,
        config_records=179,
        route_segments=32,
        operation_fire_counts={
            "arith.addi": 64,
            "arith.index_cast": 65,
            "arith.muli": 64,
            "dataflow.carry": 65,
            "dataflow.load": 64,
            "dataflow.stream": 65,
            "dataflow.sync": 64,
        },
        required_edges={
            "arith.muli#0.result0->arith.addi#0.operand0",
            "dataflow.load#0.result0->arith.muli#0.operand0",
            "dataflow.load#0.result0->arith.muli#0.operand1",
        },
        memory_windows={
            "arg4": (
                64,
                _i32_values([-5, -4, -3, -2, -1, 0, 1, 2]),
                _i32_values([-4, -3, -2, -1, 0, 1, 2, 3]),
            ),
        },
    )


def assert_variance_evidence(evidence_dir: Path) -> None:
    window = (
        16,
        _f32_values(["-2.750000", "-1.750000", "-0.750000", "0.250000", "1.250000", "2.250000", "3.250000", "-2.750000"]),
        _f32_values(["-1.750000", "-0.750000", "0.250000", "1.250000", "2.250000", "3.250000", "-2.750000", "-1.750000"]),
    )
    _assert_seed_case_evidence(
        evidence_dir,
        "variance",
        graph="workload_graph_set",
        dynamic_work_items=32,
        event_count=305,
        dfg_cycles=662,
        cgra_cycles=775,
        final_outputs=["none", "f32:-1", "f32:-0.062500", "none", "f32:67.437500", "f32:4.214844"],
        hardware="shared_reduction_adg",
        placed_records=16,
        routed_edges=22,
        config_records=472,
        route_segments=84,
        operation_fire_counts={
            "arith.addf": 16,
            "arith.index_cast": 34,
            "arith.mulf": 34,
            "arith.subf": 16,
            "dataflow.carry": 34,
            "dataflow.invariant": 57,
            "dataflow.load": 32,
            "dataflow.stream": 34,
            "dataflow.sync": 32,
            "llvm.intr.fmuladd": 16,
        },
        required_edges={
            "arith.subf#0.result0->llvm.intr.fmuladd#0.operand0",
            "dataflow.load#0.result0->arith.addf#0.operand1",
            "llvm.intr.fmuladd#0.result0->dataflow.carry#0.operand2",
        },
        memory_windows={
            "g_t_variance_red_0_0:arg4": window,
            "g_t_variance_red_1_0:arg4": window,
        },
        functional_state_source="component_cgra_sim_reports_carried_from_dfg_sim_reports",
        expected_diagnostics=["derived workload graph-set DFG report from component DFG simulator reports"],
        expected_mapping_diagnostics=["derived workload graph-set mapping artifact from component PnR mapping artifacts"],
    )


def assert_spmv_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "spmv",
        graph="g_t_spmv_kernel_red_0_0",
        dynamic_work_items=2,
        event_count=23,
        dfg_cycles=58,
        cgra_cycles=125,
        final_outputs=["none", "i32:12"],
        hardware="shared_reduction_adg",
        placed_records=8,
        routed_edges=12,
        config_records=255,
        route_segments=46,
        operation_fire_counts={
            "arith.addi": 2,
            "arith.index_cast": 5,
            "arith.muli": 2,
            "dataflow.carry": 3,
            "dataflow.load": 6,
            "dataflow.stream": 3,
            "dataflow.sync": 2,
        },
        required_edges={
            "arith.muli#0.result0->arith.addi#0.operand0",
            "dataflow.load#1.result0->dataflow.load#2.operand1",
            "dataflow.load#2.result0->arith.muli#0.operand0",
        },
        memory_windows={
            "arg4": (2, _i32_values([2, 3]), _i32_values([2, 3])),
            "arg5": (2, _i32_values([0, 2]), _i32_values([0, 2])),
            "arg6": (5, _i32_values([3, 4, 2, 5, 6]), _i32_values([3, 4, 2, 5, 6])),
        },
    )


def assert_spmm_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "spmm",
        graph="g_spmm_kernel_0",
        dynamic_work_items=4,
        event_count=246,
        dfg_cycles=488,
        cgra_cycles=740,
        final_outputs=["none"],
        hardware="shared_memory_reduction_adg",
        placed_records=40,
        routed_edges=35,
        config_records=993,
        route_segments=177,
        operation_fire_counts={
            "arith.addi": 36,
            "arith.cmpi": 7,
            "arith.index_cast": 88,
            "arith.muli": 34,
            "dataflow.constant": 9,
            "dataflow.load": 28,
            "dataflow.store": 12,
            "llvm.trunc": 18,
            "llvm.zext": 6,
            "scf.if": 8,
        },
        required_edges={
            "arith.muli#3.result0->arith.addi#4.operand1",
            "dataflow.load#4.result0->arith.muli#3.operand0",
            "llvm.zext#0.result0->arith.cmpi#4.operand1",
        },
        memory_windows={
            "arg1": (4, _i32_values([1, 2, 3, 4]), _i32_values([1, 2, 3, 4])),
            "arg2": (4, _i32_values([0, 2, 1, 2]), _i32_values([0, 2, 1, 2])),
            "arg3": (3, _i32_values([0, 2, 4]), _i32_values([0, 2, 4])),
            "arg4": (6, _i32_values([1, 2, 3, 4, 5, 6]), _i32_values([1, 2, 3, 4, 5, 6])),
            "arg5": (4, _i32_values([11, 14, 29, 36]), _i32_values([11, 14, 29, 36])),
        },
    )
