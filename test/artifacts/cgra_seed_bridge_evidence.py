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
) -> None:
    if (
        mapping.get("status") != "pass"
        or mapping.get("hardware") != hardware
        or mapping.get("placed_records") != placed_records
        or mapping.get("routed_edges") != routed_edges
        or mapping.get("unrouted_edges") != 0
        or mapping.get("unplaced_records") != 0
        or mapping.get("config_records") != config_records
        or mapping.get("diagnostics") != ["mapped software graph to fabric resources"]
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
) -> None:
    if (
        cgra.get("status") != "pass"
        or cgra.get("hardware") != hardware
        or cgra.get("dfg_cycles") != dfg_cycles
        or cgra.get("hardware_aware_cycles") != cgra_cycles
        or cgra.get("routed_edges") != routed_edges
        or cgra.get("route_segments") != route_segments
        or cgra.get("final_outputs") != final_outputs
        or cgra.get("functional_state_source") != "carried_from_dfg_sim_report"
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
