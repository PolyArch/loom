#!/usr/bin/env python3
"""Run bounded CMSIS DFG-sim attempts for row-level status evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


@dataclass(frozen=True)
class Attempt:
    suite: str
    case: str
    stem: str
    graph: str
    dfg_dir_arg: str
    args: tuple[str, ...]
    memrefs: tuple[str, ...]
    global_memrefs: tuple[str, ...] = ()
    cmsis_dsp_global_tables: tuple[str, ...] = ()
    hardware_mlir: str = ""
    hardware: str = ""
    artifact_stem: str = ""
    aggregate_stem: str = ""
    expected_dynamic_work_items: int | None = None
    expected_operation_fire_counts: tuple[tuple[str, int], ...] = ()
    expected_final_outputs: tuple[str, ...] = ()
    expected_final_memory_state: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class AttemptResult:
    attempt: Attempt
    dfg_mlir: Path
    dfg_report: Path
    mapping_summary: Path | None = None
    mapping_artifact: Path | None = None
    cgra_report: Path | None = None


ATTEMPTS = (
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_abs_f32.c",
        stem="arm_abs_f32",
        graph="g_t_arm_abs_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
        memrefs=(
            "4=-1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_mult_f32.c",
        stem="arm_mult_f32",
        graph="g_t_arm_mult_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
        memrefs=(
            "4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
            "6=2.000000e+00,-1.000000e+00,3.000000e+00,5.000000e-01",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_add_q15.c",
        stem="arm_add_q15",
        graph="g_t_arm_add_q15_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
        memrefs=(
            "4=1000,20000,-30000,32760",
            "5=0,0,0,0",
            "6=2000,15000,-10000,1000",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="BasicMathFunctions/arm_offset_f32.c",
        stem="arm_offset_f32",
        graph="g_t_arm_offset_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "4=1.250000e+00",
        ),
        memrefs=(
            "5=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "6=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="MatrixFunctions/arm_mat_add_f32.c",
        stem="arm_mat_add_f32",
        graph="g_t_arm_mat_add_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
        memrefs=(
            "4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "5=2.000000e+00,-1.000000e+00,3.000000e+00,5.000000e-01",
            "6=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="MatrixFunctions/arm_mat_mult_f32.c",
        stem="arm_mat_mult_f32",
        graph="g_t_arm_mat_mult_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "1=1",
            "2=0",
            "3=-1",
            "5=0.000000e+00",
            "6=3",
            "7=3",
            "8=false",
            "11=0",
        ),
        memrefs=(
            "4=0.000000e+00,0.000000e+00,0.000000e+00",
            "9=7.000000e+00,8.000000e+00,9.000000e+00,"
            "1.000000e+01,1.100000e+01,1.200000e+01,"
            "1.300000e+01,1.400000e+01,1.500000e+01",
            "10=1.000000e+00,2.000000e+00,3.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="StatisticsFunctions/arm_mean_f32.c",
        stem="arm_mean_f32",
        graph="g_t_arm_mean_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "4=0.000000e+00",
        ),
        memrefs=("5=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="StatisticsFunctions/arm_var_f32.c",
        stem="arm_var_f32",
        graph="g_t_arm_var_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "5=0.000000e+00",
        ),
        memrefs=("4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        artifact_stem="arm_var_f32.red0",
        aggregate_stem="arm_var_f32",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="StatisticsFunctions/arm_var_f32.c",
        stem="arm_var_f32",
        graph="g_t_arm_var_f32_red_1_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "4=9.375000e-01",
            "6=0.000000e+00",
        ),
        memrefs=("5=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        artifact_stem="arm_var_f32.red1",
        aggregate_stem="arm_var_f32",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="SupportFunctions/arm_copy_f32.c",
        stem="arm_copy_f32",
        graph="g_t_arm_copy_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
        memrefs=(
            "4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="SupportFunctions/arm_fill_f32.c",
        stem="arm_fill_f32",
        graph="g_t_arm_fill_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "4=3.250000e+00",
        ),
        memrefs=("5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="FilteringFunctions/arm_biquad_cascade_df1_f32.c",
        stem="arm_biquad_cascade_df1_f32",
        graph="g_t_arm_biquad_cascade_df1_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "4=5.000000e-01",
            "5=2.500000e-01",
            "6=1.250000e-01",
            "7=6.250000e-02",
            "8=3.125000e-02",
            "10=0.000000e+00",
            "11=0.000000e+00",
            "12=0.000000e+00",
            "13=0.000000e+00",
        ),
        memrefs=(
            "9=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",
            "14=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="FilteringFunctions/arm_fir_f32.c",
        stem="arm_fir_f32",
        graph="g_t_arm_fir_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=4",
            "2=0",
            "3=-1",
            "4=0.000000e+00",
            "5=0",
            "7=false",
        ),
        memrefs=(
            "6=2.500000e-01,5.000000e-01,7.500000e-01,1.000000e+00",
            "8=1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00",
            "9=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
            "10=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
            "11=1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        artifact_stem="arm_fir_f32.red0",
        aggregate_stem="arm_fir_f32",
    ),
    Attempt(
        suite="cmsis-dsp",
        case="FilteringFunctions/arm_fir_f32.c",
        stem="arm_fir_f32",
        graph="g_t_arm_fir_f32_red_1_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "1=4", "2=0", "3=-1"),
        memrefs=(
            "4=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00",
            "5=1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        artifact_stem="arm_fir_f32.red1",
        aggregate_stem="arm_fir_f32",
        expected_dynamic_work_items=4,
        expected_operation_fire_counts=(
            ("dataflow.load", 4),
            ("dataflow.store", 4),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg4", ("f32:1", "f32:2", "f32:3", "f32:4")),
            ("arg5", ("f32:1", "f32:2", "f32:3", "f32:4")),
        ),
    ),
    Attempt(
        suite="cmsis-dsp",
        case="TransformFunctions/arm_cfft_f32.c",
        stem="arm_cfft_f32",
        graph="g_t_arm_cfft_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "0=none", "0=none", "1=0", "2=2", "3=1"),
        memrefs=("4=1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00",),
        hardware_mlir="test/pnr/shared_signal_window_adg.mlir",
        hardware="shared_signal_window_adg",
        artifact_stem="arm_cfft_f32.red0",
        aggregate_stem="arm_cfft_f32",
        expected_dynamic_work_items=3,
        expected_operation_fire_counts=(
            ("dataflow.load", 2),
            ("dataflow.store", 2),
            ("llvm.fneg", 2),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            (
                "arg4",
                (
                    "f32:-1",
                    "f32:2",
                    "f32:-3",
                    "f32:4",
                    "f32:5",
                    "f32:6",
                    "f32:7",
                    "f32:8",
                ),
            ),
        ),
    ),
    Attempt(
        suite="cmsis-dsp",
        case="FastMathFunctions/arm_sin_f32.c",
        stem="arm_sin_f32",
        graph="g_arm_sin_f32_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "1=5.000000e-01"),
        memrefs=(),
        cmsis_dsp_global_tables=("sinTable_f32",),
        hardware_mlir="test/pnr/shared_signal_window_adg.mlir",
        hardware="shared_signal_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("dataflow.constant", 7),
            ("llvm.mlir.addressof", 1),
            ("llvm.load", 2),
            ("llvm.fptosi", 1),
            ("llvm.sitofp", 1),
            ("llvm.intr.fmuladd", 1),
        ),
        expected_final_outputs=("none", "f32:0.479419"),
    ),
    Attempt(
        suite="cmsis-dsp",
        case="FastMathFunctions/arm_sqrt_q15.c",
        stem="arm_sqrt_q15",
        graph="g_arm_sqrt_q15_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=("0=none", "1=16384"),
        memrefs=("2=0",),
        cmsis_dsp_global_tables=("sqrt_initial_lut_q15",),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("dataflow.constant", 15),
            ("dataflow.load", 1),
            ("dataflow.store", 1),
            ("llvm.intr.ctlz", 1),
            ("llvm.mlir.addressof", 1),
            ("llvm.sext", 1),
            ("llvm.trunc", 1),
            ("llvm.zext", 1),
            ("scf.if", 1),
        ),
        expected_final_outputs=("none", "i32:0"),
        expected_final_memory_state=(("arg2", ("i16:23172",)),),
    ),
    Attempt(
        suite="cmsis-dsp",
        case="TransformFunctions/arm_cfft_f32.c",
        stem="arm_cfft_f32",
        graph="g_t_arm_cfft_f32_red_1_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "1=2",
            "2=0",
            "3=-1",
        ),
        memrefs=(
            "4=1,2,3,4,5,6,7,8,9,10,11,12",
            "5=2,3,4,5,6,7,8,9,10,11,12,13",
            "6=3,4,5,6,7,8,9,10,11,12,13,14",
            "7=4,5,6,7,8,9,10,11,12,13,14,15",
            "8=5,6,7,8,9,10,11,12,13,14,15,16",
        ),
        hardware_mlir="test/pnr/shared_signal_window_adg.mlir",
        hardware="shared_signal_window_adg",
        artifact_stem="arm_cfft_f32.red1",
        aggregate_stem="arm_cfft_f32",
        expected_dynamic_work_items=2,
        expected_operation_fire_counts=(
            ("arith.mulf", 32),
            ("dataflow.load", 10),
            ("llvm.load", 30),
            ("llvm.store", 24),
            ("dataflow.store", 8),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            (
                "arg4",
                (
                    "f32:5",
                    "f32:7",
                    "f32:9",
                    "f32:11",
                    "f32:13",
                    "f32:15",
                    "f32:17",
                    "f32:19",
                    "f32:9",
                    "f32:10",
                    "f32:11",
                    "f32:12",
                ),
            ),
            (
                "arg5",
                (
                    "f32:5",
                    "f32:7",
                    "f32:9",
                    "f32:11",
                    "f32:13",
                    "f32:15",
                    "f32:17",
                    "f32:19",
                    "f32:10",
                    "f32:11",
                    "f32:12",
                    "f32:13",
                ),
            ),
            (
                "arg6",
                (
                    "f32:1",
                    "f32:11",
                    "f32:1",
                    "f32:15",
                    "f32:1",
                    "f32:19",
                    "f32:1",
                    "f32:23",
                    "f32:11",
                    "f32:12",
                    "f32:13",
                    "f32:14",
                ),
            ),
            (
                "arg7",
                (
                    "f32:-33",
                    "f32:3",
                    "f32:-45",
                    "f32:3",
                    "f32:-57",
                    "f32:3",
                    "f32:-69",
                    "f32:3",
                    "f32:12",
                    "f32:13",
                    "f32:14",
                    "f32:15",
                ),
            ),
            (
                "arg8",
                (
                    "f32:5",
                    "f32:6",
                    "f32:7",
                    "f32:8",
                    "f32:9",
                    "f32:10",
                    "f32:11",
                    "f32:12",
                    "f32:13",
                    "f32:14",
                    "f32:15",
                    "f32:16",
                ),
            ),
        ),
    ),
    Attempt(
        suite="cmsis-dsp",
        case="TransformFunctions/arm_cfft_f32.c",
        stem="arm_cfft_f32",
        graph="g_t_arm_cfft_f32_red_2_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "1=2",
            "2=0",
            "3=-1",
        ),
        memrefs=tuple(
            f"{arg}:32=" + ",".join(str(arg * 100 + offset) for offset in range(24))
            for arg in range(4, 22)
        ),
        hardware_mlir="test/pnr/shared_signal_window_adg.mlir",
        hardware="shared_signal_window_adg",
        artifact_stem="arm_cfft_f32.red2",
        aggregate_stem="arm_cfft_f32",
        expected_dynamic_work_items=2,
        expected_operation_fire_counts=(
            ("arith.addf", 46),
            ("arith.mulf", 48),
            ("arith.subf", 56),
            ("dataflow.carry", 54),
            ("dataflow.gate", 54),
            ("dataflow.load", 22),
            ("dataflow.store", 16),
            ("llvm.load", 30),
            ("llvm.store", 16),
        ),
        expected_final_outputs=("none",),
    ),
    Attempt(
        suite="cmsis-dsp",
        case="TransformFunctions/arm_cfft_f32.c",
        stem="arm_cfft_f32",
        graph="g_t_arm_cfft_f32_red_3_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "1=0",
            "2=2",
            "3=1",
            "4=5.000000e-01",
        ),
        memrefs=("5=1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00",),
        hardware_mlir="test/pnr/shared_signal_window_adg.mlir",
        hardware="shared_signal_window_adg",
        artifact_stem="arm_cfft_f32.red3",
        aggregate_stem="arm_cfft_f32",
        expected_dynamic_work_items=3,
        expected_operation_fire_counts=(
            ("dataflow.load", 2),
            ("dataflow.store", 2),
            ("llvm.fneg", 2),
            ("llvm.store", 2),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            (
                "arg5",
                (
                    "f32:0.500000",
                    "f32:-1",
                    "f32:1.500000",
                    "f32:-2",
                    "f32:5",
                    "f32:6",
                    "f32:7",
                    "f32:8",
                ),
            ),
        ),
    ),
    Attempt(
        suite="cmsis-dsp",
        case="StatisticsFunctions/arm_max_f32.c",
        stem="arm_max_f32",
        graph="g_t_arm_max_f32_red_0_0",
        dfg_dir_arg="cmsis_dsp_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "1=3",
            "2=0",
            "3=-1",
            "4=4",
            "5=0",
            "6=1.000000e+00",
        ),
        memrefs=("7=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-nn",
        case="ActivationFunctions/arm_relu_q15.c",
        stem="arm_relu_q15",
        graph="g_t_arm_relu_q15_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "1=2",
            "2=0",
            "3=-1",
            "4=15",
            "5=65537",
            "6=0",
            "7=-1",
        ),
        memrefs=("8=-2147516401,65538",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-nn",
        case="ActivationFunctions/arm_relu6_s8.c",
        stem="arm_relu6_s8",
        graph="g_t_arm_relu6_s8_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=("0=none", "2=0", "3=6", "4=2"),
        memrefs=("1=0,2,9",),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("dataflow.load", 1),
            ("llvm.intr.smax", 1),
            ("llvm.intr.umin", 1),
            ("dataflow.store", 1),
            ("dataflow.sync", 1),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg1", ("i8:0", "i8:2", "i8:6")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="BasicMathFunctions/arm_elementwise_add_s8.c",
        stem="arm_elementwise_add_s8",
        graph="g_arm_elementwise_add_s8_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "3=0",
            "4=1073741824",
            "5=0",
            "6=0",
            "7=1073741824",
            "8=0",
            "9=0",
            "11=0",
            "12=1073741824",
            "13=0",
            "14=-128",
            "15=127",
            "16=3",
        ),
        memrefs=(
            "1=1,-2,3",
            "2=4,5,-6",
            "10=0,0,0",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=3,
        expected_operation_fire_counts=(
            ("dataflow.load", 6),
            ("dataflow.store", 3),
            ("scf.if", 2),
        ),
        expected_final_outputs=("none", "i32:0"),
        expected_final_memory_state=(
            ("arg1", ("i8:1", "i8:-2", "i8:3")),
            ("arg2", ("i8:4", "i8:5", "i8:-6")),
            ("arg10", ("i8:2", "i8:1", "i8:0")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="NNSupportFunctions/arm_q7_to_q15_with_offset.c",
        stem="arm_q7_to_q15_with_offset",
        graph="g_arm_q7_to_q15_with_offset_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=("0=none", "3=4", "4=2"),
        memrefs=(
            "1=83690239",
            "2=0,0",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("dataflow.constant", 10),
            ("dataflow.load", 1),
            ("dataflow.store", 2),
            ("llvm.intr.fshl", 1),
            ("llvm.arm.pkhbt", 2),
            ("llvm.arm.pkhtb", 1),
            ("llvm.arm.sxtab16", 2),
            ("arith.remsi", 1),
            ("scf.if", 2),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg1", ("i32:83690239",)),
            ("arg2", ("i32:262145", "i32:458751")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ActivationFunctions/arm_relu_q7.c",
        stem="arm_relu_q7",
        graph="g_t_arm_relu_q7_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "1=2",
            "2=0",
            "3=-1",
            "4=7",
            "5=16843009",
            "6=0",
            "7=-1",
        ),
        memrefs=("8=-2139062144,2130706433",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        artifact_stem="arm_relu_q7.red0",
        aggregate_stem="arm_relu_q7",
    ),
    Attempt(
        suite="cmsis-nn",
        case="ActivationFunctions/arm_relu_q7.c",
        stem="arm_relu_q7",
        graph="g_t_arm_relu_q7_red_1_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=("0=none", "0=none", "0=none", "1=3", "2=0", "3=-1", "4=0"),
        memrefs=("5=-1,2,-3",),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        artifact_stem="arm_relu_q7.red1",
        aggregate_stem="arm_relu_q7",
        expected_dynamic_work_items=3,
        expected_operation_fire_counts=(
            ("dataflow.load", 3),
            ("arith.cmpi", 3),
            ("arith.select", 3),
            ("dataflow.store", 3),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg5", ("i8:0", "i8:2", "i8:0")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ConcatenationFunctions/arm_concatenation_s8_x.c",
        stem="arm_concatenation_s8_x",
        graph="g_t_arm_concatenation_s8_x_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "1=0",
            "2=2",
            "3=1",
            "4=2",
            "5=2",
        ),
        memrefs=(
            "6=1,2,3,4",
            "7=0,0,0,0,0,0",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        expected_dynamic_work_items=4,
        expected_operation_fire_counts=(
            ("dataflow.load", 4),
            ("dataflow.store", 4),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg6", ("i8:1", "i8:2", "i8:3", "i8:4")),
            ("arg7", ("i8:1", "i8:2", "i8:3", "i8:4", "i8:0", "i8:0")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ConcatenationFunctions/arm_concatenation_s8_w.c",
        stem="arm_concatenation_s8_w",
        graph="g_arm_concatenation_s8_w_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "0=none",
            "0=none",
            "0=none",
            "2=2",
            "3=1",
            "4=1",
            "5=2",
            "7=1",
        ),
        memrefs=(
            "1=1,2,3,4",
            "6=0,0,0,0,0,0",
        ),
        hardware_mlir="test/pnr/shared_signal_window_adg.mlir",
        hardware="shared_signal_window_adg",
        expected_dynamic_work_items=4,
        expected_operation_fire_counts=(
            ("dataflow.load", 4),
            ("dataflow.store", 4),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg1", ("i8:1", "i8:2", "i8:3", "i8:4")),
            ("arg6", ("i8:0", "i8:0", "i8:1", "i8:2", "i8:3", "i8:4")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ReshapeFunctions/arm_reshape_s8.c",
        stem="arm_reshape_s8",
        graph="g_arm_reshape_s8_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=("0=none", "0=none", "0=none", "0=none", "3=4"),
        memrefs=(
            "1=1,2,3,4",
            "2=0,0,0,0",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
        expected_dynamic_work_items=4,
        expected_operation_fire_counts=(
            ("dataflow.load", 4),
            ("dataflow.store", 4),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg1", ("i8:1", "i8:2", "i8:3", "i8:4")),
            ("arg2", ("i8:1", "i8:2", "i8:3", "i8:4")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="FullyConnectedFunctions/arm_vector_sum_s8.c",
        stem="arm_vector_sum_s8",
        graph="g_t_arm_vector_sum_s8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=2",
            "3=1",
            "4=2",
            "5=1",
            "6=0",
            "7=1",
        ),
        memrefs=(
            "8=0,0",
            "9=1,2,3,4",
        ),
        hardware_mlir="test/pnr/shared_reduction_adg.mlir",
        hardware="shared_reduction_adg",
    ),
    Attempt(
        suite="cmsis-nn",
        case="PoolingFunctions/arm_max_pool_s8.c",
        stem="arm_max_pool_s8",
        graph="g_t_arm_max_pool_s8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=1",
            "3=1",
            "4=1",
            "5=1",
            "6=1",
            "7=1",
            "8=1",
            "9=1",
            "10=24",
            "11=16",
            "12=8",
            "13=255",
            "14=65280",
            "15=16711680",
            "16=-16777216",
            "17=1",
            "18=false",
            "19=1",
            "20=false",
            "21=false",
            "22=1",
            "23=1",
            "24=0",
            "25=false",
            "26=1",
            "27=1",
            "28=0",
            "29=false",
            "30=-128",
            "31=127",
            "32=1",
            "33=false",
            "34=1",
            "35=false",
            "36=0",
        ),
        memrefs=(
            "37=1,2,3,4",
            "38=0,0,0,0",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("dataflow.load", 1),
            ("dataflow.store", 1),
            ("scf.if", 3),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg38", ("i8:0", "i8:0", "i8:0", "i8:0")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="PoolingFunctions/arm_avgpool_s8.c",
        stem="arm_avgpool_s8",
        graph="g_arm_avgpool_s8_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(),
        memrefs=(),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ConvolutionFunctions/arm_depthwise_conv_s8.c",
        stem="arm_depthwise_conv_s8",
        artifact_stem="arm_depthwise_conv_s8.red0",
        aggregate_stem="arm_depthwise_conv_s8",
        graph="g_t_arm_depthwise_conv_s8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=1",
            "3=1",
            "4=1",
            "5=1",
            "6=1",
            "8=1",
            "9=1",
            "10=1",
            "13=0",
            "14=1",
            "15=1",
            "16=1",
            "17=1",
            "18=1",
            "19=1",
            "20=-128",
            "21=127",
            "22=1",
            "23=0",
            "24=0",
            "25=false",
            "29=false",
            "30=1",
            "31=1",
            "32=0",
            "33=false",
            "34=0",
            "36=0",
        ),
        memrefs=(
            "7=1,2,3,4",
            "11=1,1,1,1",
            "12=0,0,0,0",
            "26=0,0,0,0",
            "27=0,0,0,0",
            "28=0,0,0,0",
            "35=0,0,0,0",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("arith.addi", 1),
            ("arith.subi", 1),
            ("llvm.intr.smax", 1),
            ("scf.if", 1),
        ),
        expected_final_outputs=("none", "i32:0"),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ConvolutionFunctions/arm_depthwise_conv_s8.c",
        stem="arm_depthwise_conv_s8",
        artifact_stem="arm_depthwise_conv_s8.red1",
        aggregate_stem="arm_depthwise_conv_s8",
        graph="g_t_arm_depthwise_conv_s8_red_1_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=1",
            "3=1",
            "4=1",
            "5=0",
            "6=16",
            "7=1",
            "8=1",
            "9=-1",
            "10=1",
            "11=1",
            "12=1",
            "13=0",
            "14=1",
            "15=1",
            "16=1",
            "17=1",
            "18=1",
            "19=false",
            "20=false",
            "22=false",
            "24=1",
            "25=1",
            "26=0",
            "29=1073741824",
            "30=31",
            "31=0",
            "32=-128",
            "33=127",
            "34=0",
            "36=false",
            "37=false",
            "38=1",
            "39=false",
            "40=1",
            "41=false",
            "42=1",
            "44=0",
        ),
        memrefs=(
            "21=0,0,0,0",
            "23=1,1,1,1",
            "27=1,1,1,1",
            "28=0,0,0,0",
            "35=0,0,0,0",
            "43=0,0,0,0",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("arith.divsi", 4),
            ("dataflow.load", 5),
            ("dataflow.mux", 4),
            ("dataflow.store", 1),
            ("llvm.trunc", 2),
            ("llvm.zext", 1),
            ("scf.if", 7),
        ),
        expected_final_outputs=("none", "i32:1"),
        expected_final_memory_state=(
            ("arg21", ("i32:0", "i32:0", "i32:0", "i32:0")),
            ("arg23", ("i8:1", "i8:1", "i8:1", "i8:1")),
            ("arg27", ("i32:1", "i32:1", "i32:1", "i32:1")),
            ("arg28", ("i32:0", "i32:0", "i32:0", "i32:0")),
            ("arg35", ("i8:0", "i8:0", "i8:0", "i8:0")),
            ("arg43", ("i8:0", "i8:0", "i8:0", "i8:0")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
        stem="arm_convolve_1x1_s8_fast",
        artifact_stem="arm_convolve_1x1_s8_fast.red0",
        aggregate_stem="arm_convolve_1x1_s8_fast",
        graph="g_t_arm_nn_mat_mult_nt_t_s8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=1",
            "3=1",
            "5=1",
            "6=0",
            "7=0",
        ),
        memrefs=("4=98,-9",),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("dataflow.load", 1),
            ("llvm.load", 1),
        ),
        expected_final_outputs=("none", "i32:98", "i32:-9", "i32:98", "i32:-9"),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
        stem="arm_convolve_1x1_s8_fast",
        artifact_stem="arm_convolve_1x1_s8_fast.red1",
        aggregate_stem="arm_convolve_1x1_s8_fast",
        graph="g_t_arm_nn_mat_mult_nt_t_s8_red_1_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=1",
            "2=0",
            "3=-1",
            "4=21648",
            "5=32462",
            "6=0",
            "7=1",
            "9=true",
            "12=1073741824",
            "13=31",
            "14=31",
            "17=0",
            "18=-128",
            "19=127",
            "20=1",
            "21=1",
            "22=0",
        ),
        memrefs=(
            "8=1,2,3,4",
            "10=1073741824",
            "11=0",
            "15=1073741824",
            "16=0",
            "23=0,0,0,0",
            "24=0,0,0,0",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("dataflow.load", 4),
            ("dataflow.store", 4),
            ("scf.if", 1),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(("arg24", ("i8:127", "i8:127", "i8:127", "i8:0")),),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
        stem="arm_convolve_1x1_s8_fast",
        artifact_stem="arm_convolve_1x1_s8_fast.red2",
        aggregate_stem="arm_convolve_1x1_s8_fast",
        graph="g_t_arm_nn_mat_mult_nt_t_s8_red_2_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=19",
            "2=0",
            "3=-1",
            "5=18048",
            "6=32384",
        ),
        memrefs=(
            "4=115,33,-36,-34,21,91,74,-18,-65,6,-106,44,11,-44,-102,-105,5,-65,74",
            "7=98,-9,-56,95,97,3,-112,-42,127,-120,102,22,96,-105,-122,67,-18,45,85,"
            "3,-8,-56,3,-32,62,-23,127,70,-33,-126,65,-120,42,-92,73,106,114,-34",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=19,
        expected_operation_fire_counts=(
            ("arith.muli", 38),
            ("dataflow.load", 38),
            ("dataflow.stream", 20),
            ("dataflow.sync", 19),
            ("llvm.load", 19),
        ),
        expected_final_outputs=("none", "i32:21648", "i32:32462"),
    ),
    Attempt(
        suite="cmsis-nn",
        case="ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
        stem="arm_convolve_1x1_s8_fast",
        artifact_stem="arm_convolve_1x1_s8_fast.red3",
        aggregate_stem="arm_convolve_1x1_s8_fast",
        graph="g_t_arm_nn_mat_mult_nt_t_s8_red_3_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=1",
            "3=1",
            "5=true",
            "6=0",
            "7=0",
            "8=1",
            "10=true",
            "11=0",
            "12=0",
            "15=1073741824",
            "16=31",
            "17=1",
            "18=0",
            "19=-128",
            "20=127",
            "21=0",
        ),
        memrefs=(
            "4=0",
            "9=1,2,3,4",
            "13=1073741824",
            "14=0",
            "22=0,0,0,0",
            "23=0,0,0,0",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=1,
        expected_operation_fire_counts=(
            ("dataflow.load", 2),
            ("dataflow.store", 1),
            ("scf.if", 2),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(("arg22", ("i8:0", "i8:0", "i8:0", "i8:0")),),
    ),
    Attempt(
        suite="cmsis-nn",
        case="FullyConnectedFunctions/arm_fully_connected_s8.c",
        stem="arm_fully_connected_s8",
        graph="g_t_arm_fully_connected_s8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=1",
            "2=0",
            "3=-1",
            "16=1",
        ),
        memrefs=(
            "4=1",
            "5=3",
            "6=1073741824",
            "7=1",
            "8=3",
            "9=2",
            "10=-128",
            "11=127",
            "12=-2",
            "13=4,-1,2,-3,5,1",
            "14=0,0",
            "15=10,-4",
            "17=1,-2,3",
            "18=0,0",
        ),
        hardware_mlir="test/pnr/shared_memory_reduction_adg.mlir",
        hardware="shared_memory_reduction_adg",
        expected_dynamic_work_items=3,
        expected_operation_fire_counts=(
            ("arith.addi", 32),
            ("arith.andi", 2),
            ("arith.cmpi", 16),
            ("arith.muli", 16),
            ("arith.select", 14),
            ("arith.shli", 4),
            ("arith.shrsi", 6),
            ("arith.subi", 4),
            ("dataflow.constant", 25),
            ("dataflow.load", 25),
            ("dataflow.store", 2),
            ("llvm.getelementptr", 2),
            ("llvm.icmp", 2),
            ("llvm.sext", 16),
            ("llvm.trunc", 4),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg4", ("i32:1",)),
            ("arg5", ("i32:3",)),
            ("arg6", ("i32:1073741824",)),
            ("arg7", ("i32:1",)),
            ("arg8", ("i32:3",)),
            ("arg9", ("i32:2",)),
            ("arg10", ("i32:-128",)),
            ("arg11", ("i32:127",)),
            ("arg12", ("i32:-2",)),
            ("arg13", ("i8:4", "i8:-1", "i8:2", "i8:-3", "i8:5", "i8:1")),
            ("arg15", ("i32:10", "i32:-4")),
            ("arg17", ("i8:1", "i8:-2", "i8:3")),
            ("arg18", ("i8:20", "i8:-18")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="BasicMathFunctions/arm_minimum_s8.c",
        stem="arm_minimum_s8",
        graph="g_t_arm_minimum_s8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=1",
            "3=1",
            "4=-1",
            "5=3",
            "6=true",
            "8=false",
            "9=0",
            "10=false",
            "11=false",
            "12=0",
            "13=false",
            "14=0",
            "15=false",
            "16=false",
            "17=0",
            "18=false",
            "19=0",
            "20=false",
            "21=false",
            "22=false",
            "23=false",
            "24=false",
            "25=0",
            "26=0",
            "27=0",
            "28=false",
            "29=false",
            "30=false",
            "31=false",
            "32=0",
            "33=0",
            "34=0",
            "35=false",
            "36=false",
            "37=true",
            "38=0",
            "39=0",
        ),
        memrefs=(
            "7=0",
            "40=3,-4,7",
            "41=2,5,-9",
            "42=0,0,0",
        ),
        hardware_mlir="test/pnr/shared_memory_reduction_adg.mlir",
        hardware="shared_memory_reduction_adg",
        expected_dynamic_work_items=3,
        expected_operation_fire_counts=(
            ("dataflow.load", 6),
            ("llvm.intr.smin", 3),
            ("dataflow.store", 3),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg40", ("i8:3", "i8:-4", "i8:7")),
            ("arg41", ("i8:2", "i8:5", "i8:-9")),
            ("arg42", ("i8:2", "i8:-4", "i8:-9")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="BasicMathFunctions/arm_maximum_s8.c",
        stem="arm_maximum_s8",
        graph="g_t_arm_maximum_s8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=1",
            "3=1",
            "4=-1",
            "5=3",
            "6=true",
            "8=false",
            "9=0",
            "10=false",
            "11=false",
            "12=0",
            "13=false",
            "14=0",
            "15=false",
            "16=false",
            "17=0",
            "18=false",
            "19=0",
            "20=false",
            "21=false",
            "22=false",
            "23=false",
            "24=false",
            "25=0",
            "26=0",
            "27=0",
            "28=false",
            "29=false",
            "30=false",
            "31=false",
            "32=0",
            "33=0",
            "34=0",
            "35=false",
            "36=false",
            "37=true",
            "38=0",
            "39=0",
        ),
        memrefs=(
            "7=0",
            "40=3,-4,7",
            "41=2,5,-9",
            "42=0,0,0",
        ),
        hardware_mlir="test/pnr/shared_memory_reduction_adg.mlir",
        hardware="shared_memory_reduction_adg",
        expected_dynamic_work_items=3,
        expected_operation_fire_counts=(
            ("dataflow.load", 6),
            ("llvm.intr.smax", 3),
            ("dataflow.store", 3),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg40", ("i8:3", "i8:-4", "i8:7")),
            ("arg41", ("i8:2", "i8:5", "i8:-9")),
            ("arg42", ("i8:3", "i8:5", "i8:7")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="SoftmaxFunctions/arm_softmax_u8.c",
        stem="arm_softmax_u8",
        graph="g_t_arm_softmax_u8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=1",
            "3=1",
            "4=3",
            "5=true",
            "6=-128",
            "7=0",
            "8=1",
            "9=-1",
            "10=1073741824",
            "11=-1073741823",
            "12=1",
            "13=2147483648",
            "14=-2147483648",
            "15=false",
            "16=2147483647",
            "17=-16777216",
            "18=5",
            "19=268435456",
            "20=31",
            "21=3",
            "22=2",
            "23=715827883",
            "24=31",
            "25=1895147668",
            "26=1895147668",
            "27=1672461947",
            "28=16777216",
            "29=1302514674",
            "30=33554432",
            "31=790015084",
            "32=67108864",
            "33=290630308",
            "34=134217728",
            "35=39332535",
            "36=720401",
            "37=536870912",
            "38=242",
            "39=1073741824",
            "40=12",
            "41=11",
            "42=true",
            "43=35",
            "44=-1010580540",
            "45=1515870810",
            "46=-2147483648",
            "47=-536870911",
            "48=-1073741823",
            "49=0",
            "50=1",
            "51=false",
            "52=255",
            "53=true",
        ),
        memrefs=(
            "54=1,2,3",
            "55=0,0,0",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=3,
        expected_operation_fire_counts=(
            ("dataflow.load", 9),
            ("dataflow.mux", 6),
            ("dataflow.store", 3),
            ("llvm.intr.ctlz", 1),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg54", ("i8:1", "i8:2", "i8:3")),
            ("arg55", ("i8:85", "i8:85", "i8:85")),
        ),
    ),
    Attempt(
        suite="cmsis-nn",
        case="SoftmaxFunctions/arm_softmax_s8.c",
        stem="arm_softmax_s8",
        graph="g_t_arm_nn_softmax_common_s8_red_0_0",
        dfg_dir_arg="cmsis_nn_dfg_dir",
        args=(
            "0=none",
            "1=0",
            "2=1",
            "3=1",
            "4=5",
            "5=true",
            "6=-3968",
            "7=19",
            "8=1077952640",
            "9=-1",
            "10=1073741824",
            "11=-1073741823",
            "12=1077952640",
            "13=2147483648",
            "14=-2147483648",
            "15=false",
            "16=2147483647",
            "17=-16777216",
            "18=5",
            "19=268435456",
            "20=31",
            "21=3",
            "22=2",
            "23=715827883",
            "24=31",
            "25=1895147668",
            "26=1895147668",
            "27=1672461947",
            "28=16777216",
            "29=1302514674",
            "30=33554432",
            "31=790015084",
            "32=67108864",
            "33=290630308",
            "34=134217728",
            "35=39332535",
            "36=720401",
            "37=536870912",
            "38=242",
            "39=1073741824",
            "40=12",
            "41=11",
            "42=true",
            "43=-1010580540",
            "44=1515870810",
            "45=-2147483648",
            "46=-536870911",
            "47=-1073741823",
            "49=27",
            "50=-32768",
            "51=1077952640",
            "52=false",
            "53=65535",
            "54=true",
            "55=35",
            "56=-128",
            "57=1077952640",
            "58=false",
            "59=255",
            "60=true",
            "61=false",
        ),
        memrefs=(
            "48=0,0,0,0,0",
            "62=101,49,6,-34,-75",
        ),
        hardware_mlir="test/pnr/shared_quantized_window_adg.mlir",
        hardware="shared_quantized_window_adg",
        expected_dynamic_work_items=5,
        expected_operation_fire_counts=(
            ("dataflow.load", 15),
            ("dataflow.mux", 10),
            ("dataflow.store", 5),
            ("llvm.intr.ctlz", 1),
        ),
        expected_final_outputs=("none",),
        expected_final_memory_state=(
            ("arg48", ("i8:-57", "i8:-70", "i8:-79", "i8:-86", "i8:-92")),
            ("arg62", ("i8:101", "i8:49", "i8:6", "i8:-34", "i8:-75")),
        ),
    ),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmsis-dsp-dfg-dir", required=True)
    parser.add_argument("--cmsis-nn-dfg-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--attempt-stem",
        action="append",
        default=[],
        help=(
            "run only attempts matching this artifact stem, input stem, or aggregate stem; "
            "may be repeated"
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="run only attempts for this full CMSIS source row; may be repeated",
    )
    parser.add_argument("--loom-dfg-sim")
    parser.add_argument("--loom-raise-opt")
    parser.add_argument("--loom-pnr-map")
    parser.add_argument("--loom-cgra-sim")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--jobs", type=positive_int, default=None)
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return positive_int(value)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"[cmsis-dfg-sim] {name} {exc}") from exc


def cmsis_dfg_sim_jobs(args: argparse.Namespace, attempt_count: int) -> int:
    if attempt_count < 1:
        return 1
    budget = (
        args.jobs
        or positive_env_int("LOOM_CMSIS_DFG_SIM_JOBS")
        or positive_env_int("LOOM_TEST_JOBS")
        or positive_env_int("JOBS")
        or (os.cpu_count() or 1)
    )
    return max(1, min(attempt_count, budget))


def attempt_matches_stem(attempt: Attempt, stem: str) -> bool:
    return stem in {attempt.stem, attempt.artifact_stem, attempt.aggregate_stem}


def select_attempts(args: argparse.Namespace) -> tuple[Attempt, ...]:
    requested_stems = tuple(args.attempt_stem)
    requested_cases = tuple(args.case)
    blank_selectors = [
        selector
        for selector in (*requested_stems, *requested_cases)
        if not selector.strip()
    ]
    if blank_selectors:
        raise SystemExit("[cmsis-dfg-sim] CMSIS attempt selectors must not be blank")
    if not requested_stems and not requested_cases:
        return ATTEMPTS

    selected: list[Attempt] = []
    matched_stems: set[str] = set()
    matched_cases: set[str] = set()
    for attempt in ATTEMPTS:
        include = False
        for stem in requested_stems:
            if attempt_matches_stem(attempt, stem):
                include = True
                matched_stems.add(stem)
        for case in requested_cases:
            if attempt.case == case:
                include = True
                matched_cases.add(case)
        if include:
            selected.append(attempt)

    missing = [stem for stem in requested_stems if stem not in matched_stems]
    missing.extend(case for case in requested_cases if case not in matched_cases)
    if missing:
        available = sorted(
            {
                label
                for attempt in ATTEMPTS
                for label in (attempt.stem, attempt.artifact_stem, attempt.aggregate_stem, attempt.case)
                if label
            }
        )
        raise SystemExit(
            "[cmsis-dfg-sim] unknown CMSIS attempt selector(s): "
            f"{', '.join(missing)}; available selectors include: {', '.join(available)}"
        )
    return tuple(selected)


def resolve_tool(explicit: str | None, env_var: str, default: Path) -> Path:
    value = explicit or os.environ.get(env_var)
    if value:
        candidate = Path(value)
        if candidate.is_file():
            return candidate
        resolved = shutil.which(value)
        if resolved:
            return Path(resolved)
    return default


def require_tool(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"[cmsis-dfg-sim] {label} not executable: {path}")


def normalize_c_numeric_literal(value: str) -> str:
    token = value.strip()
    while token and token[-1] in "fFuUlL":
        token = token[:-1]
    return token


def parse_c_initializer_values(body: str) -> list[str]:
    body = re.sub(r"/\*.*?\*/", " ", body, flags=re.S)
    body = re.sub(r"//.*", " ", body)
    values = re.findall(
        r"[-+]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[fFuUlL]*",
        body,
    )
    return [normalize_c_numeric_literal(value) for value in values]


def cmsis_dsp_common_table_memref(symbol: str) -> str:
    if not symbol or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
        raise SystemExit(f"[cmsis-dfg-sim] invalid CMSIS-DSP table symbol: {symbol!r}")
    source = ROOT / "externals" / "cmsis-dsp" / "Source" / "CommonTables" / "arm_common_tables.c"
    if not source.is_file():
        raise SystemExit(f"[cmsis-dfg-sim] missing CMSIS-DSP common table source: {source}")
    text = source.read_text(errors="replace")
    pattern = (
        r"const\s+[A-Za-z_][A-Za-z0-9_]*\s+"
        + re.escape(symbol)
        + r"\s*\[[^\]]+\][^=]*=\s*\{(?P<body>.*?)\};"
    )
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise SystemExit(f"[cmsis-dfg-sim] missing CMSIS-DSP common table: {symbol}")
    values = parse_c_initializer_values(match.group("body"))
    if not values:
        raise SystemExit(f"[cmsis-dfg-sim] CMSIS-DSP common table is empty: {symbol}")
    return f"{symbol}=" + ",".join(values)


def attempt_global_memrefs(attempt: Attempt) -> list[str]:
    resolved = list(attempt.global_memrefs)
    resolved.extend(
        cmsis_dsp_common_table_memref(symbol)
        for symbol in attempt.cmsis_dsp_global_tables
    )
    return resolved


def run_command(
    command: list[str],
    timeout_seconds: int,
    label: str,
    env_overrides: dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"[cmsis-dfg-sim] {label} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def validate_attempt_report(attempt: Attempt, data: dict[str, object], output: Path) -> None:
    if data.get("status") != "pass":
        return

    label = attempt.artifact_stem or attempt.stem
    diagnostics: list[str] = []
    if (
        attempt.expected_dynamic_work_items is not None
        and data.get("dynamic_work_items") != attempt.expected_dynamic_work_items
    ):
        diagnostics.append(
            f"dynamic_work_items={data.get('dynamic_work_items')!r}, "
            f"expected {attempt.expected_dynamic_work_items!r}"
        )

    if attempt.expected_operation_fire_counts:
        counts = data.get("operation_fire_counts")
        if not isinstance(counts, dict):
            diagnostics.append("operation_fire_counts is not an object")
        else:
            for op_name, expected_count in attempt.expected_operation_fire_counts:
                actual = counts.get(op_name)
                if actual != expected_count:
                    diagnostics.append(
                        f"operation_fire_counts[{op_name!r}]={actual!r}, "
                        f"expected {expected_count!r}"
                    )

    if attempt.expected_final_outputs and data.get("final_outputs") != list(attempt.expected_final_outputs):
        diagnostics.append(
            f"final_outputs={data.get('final_outputs')!r}, "
            f"expected {list(attempt.expected_final_outputs)!r}"
        )

    if attempt.expected_final_memory_state:
        memory_state = data.get("final_memory_state")
        if not isinstance(memory_state, dict):
            diagnostics.append("final_memory_state is not an object")
        else:
            for argument, expected_values in attempt.expected_final_memory_state:
                actual = memory_state.get(argument)
                expected = list(expected_values)
                if actual != expected:
                    diagnostics.append(
                        f"final_memory_state[{argument!r}]={actual!r}, expected {expected!r}"
                    )

    if diagnostics:
        raise SystemExit(
            f"[cmsis-dfg-sim] {attempt.case} {label} failed expected simulator evidence "
            f"guard at {output}: {'; '.join(diagnostics)}"
        )


def run_attempt(
    dfg_tool: Path,
    lower_tool: Path,
    pnr_tool: Path,
    cgra_tool: Path,
    output_dir: Path,
    args: argparse.Namespace,
    attempt: Attempt,
) -> AttemptResult:
    dfg_dir = Path(getattr(args, attempt.dfg_dir_arg))
    dfg_mlir = dfg_dir / f"{attempt.stem}.dfg.mlir"
    if not dfg_mlir.is_file():
        raise SystemExit(f"[cmsis-dfg-sim] missing DFG MLIR for {attempt.case}: {dfg_mlir}")

    output_stem = attempt.artifact_stem or attempt.stem
    lowered_mlir = output_dir / f"{output_stem}.lowered.dfg.mlir"
    output = output_dir / f"{output_stem}.dfg.report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(lower_tool),
            "--loom-lower-graph-memory",
            str(dfg_mlir),
            "-o",
            str(lowered_mlir),
        ],
        args.timeout_seconds,
        f"{attempt.case} graph-memory lowering",
    )
    command = [
        str(dfg_tool),
        str(lowered_mlir),
        "--graph",
        attempt.graph,
        "--workload",
        attempt.case,
        "--output",
        str(output),
    ]
    for arg in attempt.args:
        command.extend(["--arg", arg])
    for memref in attempt.memrefs:
        command.extend(["--memref", memref])
    for global_memref in attempt_global_memrefs(attempt):
        command.extend(["--global-memref", global_memref])

    run_command(command, args.timeout_seconds, attempt.case)
    if not output.is_file():
        raise SystemExit(f"[cmsis-dfg-sim] {attempt.case} produced no report: {output}")
    data = json.loads(output.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"[cmsis-dfg-sim] {attempt.case} report is not a JSON object: {output}")
    data["input_artifact_fingerprints"] = intermediate_artifacts.input_artifact_fingerprints(
        [dfg_mlir, lowered_mlir]
    )
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    validate_attempt_report(attempt, data, output)

    if not attempt.hardware_mlir:
        return AttemptResult(attempt=attempt, dfg_mlir=dfg_mlir, dfg_report=output)
    if data.get("status") != "pass":
        return AttemptResult(attempt=attempt, dfg_mlir=dfg_mlir, dfg_report=output)

    hardware_mlir = ROOT / attempt.hardware_mlir
    mapping_output = output_dir / f"{output_stem}.mapping.csv"
    mapping_artifact = output_dir / f"{output_stem}.mapping.json"
    run_command(
        [
            sys.executable,
            str(ROOT / "test" / "pnr" / "mapping_summary.py"),
            "--dfg-mlir",
            str(lowered_mlir),
            "--graph",
            attempt.graph,
            "--hardware-mlir",
            str(hardware_mlir),
            "--hardware",
            attempt.hardware,
            "--workload",
            attempt.case,
            "--output",
            str(mapping_output),
            "--artifact",
            str(mapping_artifact),
        ],
        args.timeout_seconds,
        f"{attempt.case} PnR",
        {"LOOM_PNR_MAP": str(pnr_tool)},
    )
    require_tool(cgra_tool, "loom-cgra-sim")
    cgra_report = output_dir / f"{output_stem}.cgra.report.json"
    run_command(
        [
            str(cgra_tool),
            "--dfg-report",
            str(output),
            "--mapping-artifact",
            str(mapping_artifact),
            "--hardware-mlir",
            str(hardware_mlir),
            "--output",
            str(cgra_report),
        ],
        args.timeout_seconds,
        f"{attempt.case} CGRA-sim",
    )
    return AttemptResult(
        attempt=attempt,
        dfg_mlir=dfg_mlir,
        dfg_report=output,
        mapping_summary=mapping_output,
        mapping_artifact=mapping_artifact,
        cgra_report=cgra_report,
    )


def aggregate_mapping_id(case: str, hardware: str) -> str:
    return f"{quote(case, safe='')}__workload_graph_set__{hardware}"


def run_aggregates(
    output_dir: Path,
    results: list[AttemptResult],
    timeout_seconds: int,
) -> list[Path]:
    grouped: dict[tuple[str, str, str], list[AttemptResult]] = {}
    for result in results:
        stem = result.attempt.aggregate_stem
        if not stem:
            continue
        key = (result.attempt.case, result.attempt.hardware, stem)
        grouped.setdefault(key, []).append(result)

    artifacts: list[Path] = []
    for (case, hardware, stem), group in grouped.items():
        if len(group) < 2:
            continue
        if any(
            item.mapping_artifact is None or item.mapping_summary is None or item.cgra_report is None
            for item in group
        ):
            continue
        dfg_output = output_dir / f"{stem}.dfg.report.json"
        mapping_output = output_dir / f"{stem}.mapping.json"
        cgra_output = output_dir / f"{stem}.cgra.report.json"
        mapping_summary = output_dir / f"{stem}.mapping.csv"
        command = [
            sys.executable,
            str(ROOT / "test" / "e2e" / "aggregate_workload_graph_artifacts.py"),
            "--workload",
            case,
            "--hardware",
            hardware,
            "--mapping-id",
            aggregate_mapping_id(case, hardware),
            "--dfg-output",
            str(dfg_output),
            "--mapping-output",
            str(mapping_output),
            "--cgra-output",
            str(cgra_output),
            "--mapping-summary-output",
            str(mapping_summary),
        ]
        for dfg_mlir in sorted({result.dfg_mlir for result in group}):
            command.extend(["--source-dfg-mlir", str(dfg_mlir)])
        for result in group:
            command.extend(["--dfg-report", str(result.dfg_report)])
        for result in group:
            if result.mapping_artifact is not None:
                command.extend(["--mapping-artifact", str(result.mapping_artifact)])
        for result in group:
            if result.cgra_report is not None:
                command.extend(["--cgra-report", str(result.cgra_report)])
        run_command(command, timeout_seconds, f"{case} aggregate")
        artifacts.extend([dfg_output, mapping_output, cgra_output, mapping_summary])
    return artifacts


def run_attempts(
    dfg_tool: Path,
    lower_tool: Path,
    pnr_tool: Path,
    cgra_tool: Path,
    output_dir: Path,
    args: argparse.Namespace,
    attempts: tuple[Attempt, ...],
) -> list[AttemptResult]:
    results: list[AttemptResult | None] = [None] * len(attempts)
    jobs = cmsis_dfg_sim_jobs(args, len(attempts))
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                run_attempt,
                dfg_tool,
                lower_tool,
                pnr_tool,
                cgra_tool,
                output_dir,
                args,
                attempt,
            ): index
            for index, attempt in enumerate(attempts)
        }
        for future in as_completed(futures):
            index = futures[future]
            attempt = attempts[index]
            try:
                results[index] = future.result()
            except SystemExit:
                raise
            except BaseException as exc:
                raise SystemExit(f"[cmsis-dfg-sim] {attempt.case} failed: {exc}") from exc
    return [result for result in results if result is not None]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    selected_attempts = select_attempts(args)
    dfg_tool = resolve_tool(
        args.loom_dfg_sim,
        "LOOM_DFG_SIM",
        ROOT / "build" / "tools" / "loom-dfg-sim" / "loom-dfg-sim",
    )
    lower_tool = resolve_tool(
        args.loom_raise_opt,
        "LOOM_RAISE_OPT",
        ROOT / "build" / "bin" / "loom-raise-opt",
    )
    pnr_tool = resolve_tool(
        args.loom_pnr_map,
        "LOOM_PNR_MAP",
        ROOT / "build" / "tools" / "loom-pnr-map" / "loom-pnr-map",
    )
    cgra_tool = resolve_tool(
        args.loom_cgra_sim,
        "LOOM_CGRA_SIM",
        ROOT / "build" / "tools" / "loom-cgra-sim" / "loom-cgra-sim",
    )
    require_tool(dfg_tool, "loom-dfg-sim")
    require_tool(lower_tool, "loom-raise-opt")
    if any(attempt.hardware_mlir for attempt in selected_attempts):
        require_tool(pnr_tool, "loom-pnr-map")
    output_dir = Path(args.output_dir)
    results = run_attempts(
        dfg_tool,
        lower_tool,
        pnr_tool,
        cgra_tool,
        output_dir,
        args,
        selected_attempts,
    )
    artifacts: list[Path] = []
    for result in results:
        artifacts.append(result.dfg_report)
        for artifact in (result.mapping_summary, result.mapping_artifact, result.cgra_report):
            if artifact is not None:
                artifacts.append(artifact)
    artifacts.extend(run_aggregates(output_dir, results, args.timeout_seconds))
    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
