#!/usr/bin/env python3
"""Content checks for remaining shared ADG seed rows."""

from __future__ import annotations

from pathlib import Path

from cgra_seed_bridge_evidence import _assert_seed_case_evidence, _f32_values, _i32_values


def assert_gemm_evidence(evidence_dir: Path) -> None:
    assert _f32_values([str(value) for value in range(8)]) == [
        "f32:0",
        "f32:1",
        "f32:2",
        "f32:3",
        "f32:4",
        "f32:5",
        "f32:6",
        "f32:7",
    ]
    _assert_seed_case_evidence(
        evidence_dir,
        "gemm",
        graph="g_t__ZN12_GLOBAL__N_14gemmEPKfS1_Pfiii_0_0",
        dynamic_work_items=8,
        event_count=112,
        dfg_cycles=242,
        cgra_cycles=317,
        final_outputs=["none", "f32:28"],
        hardware="shared_reduction_adg",
        placed_records=10,
        routed_edges=14,
        config_records=303,
        route_segments=54,
        operation_fire_counts={
            "arith.index_cast": 26,
            "arith.shli": 9,
            "arith.shrui": 8,
            "dataflow.carry": 9,
            "dataflow.constant": 8,
            "dataflow.invariant": 11,
            "dataflow.load": 16,
            "dataflow.stream": 9,
            "dataflow.sync": 8,
            "llvm.intr.fmuladd": 8,
        },
        required_edges={
            "arith.shrui#0.result0->dataflow.load#1.operand1",
            "dataflow.load#0.result0->llvm.intr.fmuladd#0.operand0",
            "dataflow.load#1.result0->llvm.intr.fmuladd#0.operand1",
            "llvm.intr.fmuladd#0.result0->dataflow.carry#0.operand2",
        },
        memory_windows={
            "arg4": (
                8,
                _f32_values([str(value) for value in range(8)]),
                _f32_values([str(value) for value in range(8)]),
            ),
            "arg6": (225, _f32_values(["1"] * 8), _f32_values(["1"] * 8)),
        },
    )


def assert_hash_mix_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "hash_mix",
        graph="g_t_main_1_0",
        dynamic_work_items=64,
        event_count=576,
        dfg_cycles=1287,
        cgra_cycles=1365,
        final_outputs=["none"],
        hardware="shared_reduction_adg",
        placed_records=9,
        routed_edges=13,
        config_records=289,
        route_segments=53,
        operation_fire_counts={
            "arith.addi": 64,
            "arith.muli": 64,
            "arith.xori": 64,
            "dataflow.load": 128,
            "dataflow.store": 64,
            "dataflow.sync": 64,
            "llvm.intr.fshl": 128,
        },
        required_edges={
            "arith.addi#0.result0->llvm.intr.fshl#0.operand0",
            "dataflow.load#1.result0->arith.xori#0.operand1",
            "llvm.intr.fshl#0.result0->arith.xori#0.operand0",
            "llvm.intr.fshl#1.result0->dataflow.store#0.operand2",
        },
        memory_windows={
            "arg1": (
                64,
                _i32_values([1732584193 + value for value in range(8)]),
                _i32_values([1732584193 + value for value in range(56, 64)]),
            ),
            "arg2": (
                64,
                _i32_values([-271733879 + 13 * value for value in range(8)]),
                _i32_values([-271733879 + 13 * value for value in range(56, 64)]),
            ),
            "arg6": (
                64,
                _i32_values(
                    [
                        180967326,
                        482358742,
                        -840625257,
                        -163121667,
                        233430344,
                        -1721851285,
                        1079349901,
                        1880063498,
                    ]
                ),
                _i32_values(
                    [
                        1686262881,
                        -1092274563,
                        1660031372,
                        515634785,
                        -1779250925,
                        1993843780,
                        1203406767,
                        182220187,
                    ]
                ),
            ),
        },
    )


def assert_relu_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "relu",
        graph="workload_graph_set",
        dynamic_work_items=64,
        event_count=355,
        dfg_cycles=750,
        cgra_cycles=821,
        final_outputs=["none", "none", "f32:42"],
        hardware="shared_reduction_adg",
        placed_records=10,
        routed_edges=12,
        config_records=257,
        route_segments=44,
        operation_fire_counts={
            "arith.addf": 32,
            "arith.cmpf": 32,
            "arith.index_cast": 33,
            "arith.select": 32,
            "dataflow.carry": 33,
            "dataflow.load": 64,
            "dataflow.store": 32,
            "dataflow.stream": 33,
            "dataflow.sync": 64,
        },
        required_edges={
            "arith.cmpf#0.result0->arith.select#0.operand0",
            "arith.select#0.result0->dataflow.store#0.operand2",
            "dataflow.load#0.result0->arith.select#0.operand1",
        },
        memory_windows={
            "g_t_main_red_0_0:arg4": (
                32,
                _f32_values(["0", "0", "0", "0", "0", "0", "0", "1"]),
                _f32_values(["5", "6", "0", "0", "0", "0", "0", "0"]),
            ),
            "g_t_relu_0_0:arg1": (
                32,
                _f32_values(["-6", "-5", "-4", "-3", "-2", "-1", "0", "1"]),
                _f32_values(["5", "6", "-6", "-5", "-4", "-3", "-2", "-1"]),
            ),
            "g_t_relu_0_0:arg3": (
                32,
                _f32_values(["0", "0", "0", "0", "0", "0", "0", "1"]),
                _f32_values(["5", "6", "0", "0", "0", "0", "0", "0"]),
            ),
        },
        functional_state_source="component_cgra_sim_reports_carried_from_dfg_sim_reports",
        expected_diagnostics=["derived workload graph-set DFG report from component DFG simulator reports"],
        expected_mapping_diagnostics=[
            "derived workload graph-set mapping artifact from component PnR mapping artifacts"
        ],
    )


def assert_sbox_lookup_evidence(evidence_dir: Path) -> None:
    _assert_seed_case_evidence(
        evidence_dir,
        "sbox_lookup",
        graph="g_t_main_2_0",
        dynamic_work_items=64,
        event_count=448,
        dfg_cycles=1093,
        cgra_cycles=1133,
        final_outputs=["none"],
        hardware="shared_reduction_adg",
        placed_records=5,
        routed_edges=6,
        config_records=119,
        route_segments=20,
        operation_fire_counts={
            "arith.andi": 64,
            "arith.index_cast": 128,
            "dataflow.load": 128,
            "dataflow.store": 64,
            "dataflow.sync": 64,
        },
        required_edges={
            "arith.andi#0.result0->dataflow.load#1.operand1",
            "dataflow.load#1.result0->dataflow.store#0.operand2",
            "dataflow.store#0.result0->dataflow.sync#0.operand2",
        },
        memory_windows={
            "arg1": (
                64,
                _i32_values([(17 + 13 * value) % 256 for value in range(8)]),
                _i32_values([(17 + 13 * value) % 256 for value in range(56, 64)]),
            ),
            "arg3": (
                256,
                _i32_values([(31 + 7 * value) % 256 for value in range(8)]),
                _i32_values([(31 + 7 * value) % 256 for value in range(248, 256)]),
            ),
            "arg4": (
                64,
                _i32_values([150, 241, 76, 167, 2, 93, 184, 19]),
                _i32_values([126, 217, 52, 143, 234, 69, 160, 251]),
            ),
        },
    )
