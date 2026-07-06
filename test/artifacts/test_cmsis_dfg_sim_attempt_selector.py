#!/usr/bin/env python3
"""Regression tests for selectable CMSIS DFG-sim attempts."""

from __future__ import annotations

import sys
import subprocess
import json
from pathlib import Path

import artifact_test_common


def load_attempt_module(repo: Path):
    sys.path.insert(0, str(repo / "test" / "e2e"))
    import run_cmsis_dfg_sim_attempts as attempts  # noqa: E402

    return attempts


def labels(selected) -> list[str]:
    return [attempt.artifact_stem or attempt.stem for attempt in selected]


def run_rollup(repo: Path, out_dir: Path, evidence_dir: Path, stem: str) -> None:
    result = subprocess.run(
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--sim-evidence-dir",
            str(evidence_dir),
            "--cmsis-sim-attempt-stem",
            stem,
            "--jobs",
            "8",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"selected CMSIS rollup failed for {stem} with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def row_by_case(csv_path: Path, suite: str, case: str) -> dict[str, str]:
    import csv

    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["suite"] == suite and row["case"] == case:
                return row
    raise AssertionError(f"missing {suite} row for {case}")


def assert_selected_rollup_drops_stale_cmsis_evidence(repo: Path) -> None:
    with artifact_test_common.repo_temp_dir(repo, "cmsis-selector-stale-") as tmp:
        root = Path(tmp)
        out_dir = root / "rollup"
        evidence_dir = root / "shared-sim-evidence"

        run_rollup(repo, out_dir, evidence_dir, "arm_relu_q15")
        relu_first = row_by_case(
            out_dir / "cgra-status-summary.csv",
            "cmsis-nn",
            "ActivationFunctions/arm_relu_q15.c",
        )
        if relu_first["status"] != "pass" or relu_first["cgra_status"] != "pass":
            raise AssertionError(f"initial selected relu evidence should pass: {relu_first}")

        run_rollup(repo, out_dir, evidence_dir, "arm_add_q15")
        add_second = row_by_case(
            out_dir / "cgra-status-summary.csv",
            "cmsis-dsp",
            "BasicMathFunctions/arm_add_q15.c",
        )
        relu_second = row_by_case(
            out_dir / "cgra-status-summary.csv",
            "cmsis-nn",
            "ActivationFunctions/arm_relu_q15.c",
        )
        if add_second["status"] != "pass" or add_second["cgra_status"] != "pass":
            raise AssertionError(f"second selected add evidence should pass: {add_second}")
        if relu_second["status"] == "pass" or relu_second["cgra_status"] != "not_run":
            raise AssertionError(f"stale relu evidence survived selected rerun: {relu_second}")
        if relu_second["comparison_report"] or relu_second["comparison_status"] != "not_run":
            raise AssertionError(f"stale relu comparison survived selected rerun: {relu_second}")


def assert_default_batch_rollup_promotes_bounded_rows(repo: Path) -> None:
    with artifact_test_common.repo_temp_dir(repo, "cmsis-default-batch-") as tmp:
        out_dir = Path(tmp) / "rollup"
        result = subprocess.run(
            [
                "bash",
                "test/e2e/run_cmsis_cgra_status_rollup.sh",
                "--output-dir",
                str(out_dir),
                "--cmsis-sim-default-batch",
                "--jobs",
                "8",
            ],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"CMSIS default-batch rollup failed with {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        data = json.loads((out_dir / "cgra-status-summary.json").read_text())
        expected_counts = {
            "cmsis-dsp": {
                "total": 16,
                "pass": 16,
                "fail": 0,
                "blocked": 0,
                "unsupported": 0,
                "missing_status": 0,
            },
            "cmsis-nn": {
                "total": 18,
                "pass": 17,
                "fail": 0,
                "blocked": 0,
                "unsupported": 1,
                "missing_status": 0,
            },
        }
        for suite, expected in expected_counts.items():
            actual = data.get("counts", {}).get(suite)
            if actual != expected:
                raise AssertionError(f"{suite} default-batch counts {actual}, expected {expected}")

        add = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        abs_f32 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-dsp", "BasicMathFunctions/arm_abs_f32.c")
        fill = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-dsp", "SupportFunctions/arm_fill_f32.c")
        sin_f32 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-dsp", "FastMathFunctions/arm_sin_f32.c")
        sqrt_q15 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-dsp", "FastMathFunctions/arm_sqrt_q15.c")
        q7_to_q15 = row_by_case(
            out_dir / "cgra-status-summary.csv",
            "cmsis-nn",
            "NNSupportFunctions/arm_q7_to_q15_with_offset.c",
        )
        relu_q15 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
        relu_q7 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "ActivationFunctions/arm_relu_q7.c")
        relu6 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "ActivationFunctions/arm_relu6_s8.c")
        reshape = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "ReshapeFunctions/arm_reshape_s8.c")
        softmax_s8 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "SoftmaxFunctions/arm_softmax_s8.c")
        softmax = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "SoftmaxFunctions/arm_softmax_u8.c")
        avgpool = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "PoolingFunctions/arm_avgpool_s8.c")
        depthwise = row_by_case(
            out_dir / "cgra-status-summary.csv",
            "cmsis-nn",
            "ConvolutionFunctions/arm_depthwise_conv_s8.c",
        )
        vector_sum = row_by_case(
            out_dir / "cgra-status-summary.csv",
            "cmsis-nn",
            "FullyConnectedFunctions/arm_vector_sum_s8.c",
        )
        fully_connected = row_by_case(
            out_dir / "cgra-status-summary.csv",
            "cmsis-nn",
            "FullyConnectedFunctions/arm_fully_connected_s8.c",
        )
        for row in (
            add,
            abs_f32,
            fill,
            sin_f32,
            sqrt_q15,
            q7_to_q15,
            relu_q15,
            relu_q7,
            relu6,
            reshape,
            depthwise,
            vector_sum,
            fully_connected,
        ):
            if row["status"] != "pass" or row["cgra_status"] != "pass" or row["comparison_status"] != "pass":
                raise AssertionError(f"default-batch row should expose CGRA-sim pass evidence: {row}")
        if (
            fully_connected["diagnostic_class"] != "cgra_sim_pass"
            or fully_connected["blocking_prerequisite"] != ""
            or fully_connected["mapping_status"] != "pass"
            or fully_connected["final_outputs_present"] != "true"
            or fully_connected["final_memory_state_present"] != "true"
        ):
            raise AssertionError(
                "default-batch rollup should record fully connected as real CGRA-sim pass evidence: "
                f"{fully_connected}"
            )
        if (
            softmax["status"] != "pass"
            or softmax["dfg_status"] != "pass"
            or softmax["mapping_status"] != "pass"
            or softmax["cgra_status"] != "pass"
            or softmax["comparison_status"] != "pass"
            or softmax["final_outputs_present"] != "true"
            or softmax["final_memory_state_present"] != "true"
            or softmax["diagnostic_class"] != "cgra_sim_pass"
        ):
            raise AssertionError(f"default-batch rollup should promote softmax with real CGRA-sim evidence: {softmax}")
        if (
            softmax_s8["status"] != "pass"
            or softmax_s8["dfg_status"] != "pass"
            or softmax_s8["mapping_status"] != "pass"
            or softmax_s8["cgra_status"] != "pass"
            or softmax_s8["comparison_status"] != "pass"
            or softmax_s8["final_outputs_present"] != "true"
            or softmax_s8["final_memory_state_present"] != "true"
            or softmax_s8["diagnostic_class"] != "cgra_sim_pass"
            or softmax_s8["blocking_prerequisite"] != ""
        ):
            raise AssertionError(f"default-batch rollup should promote softmax_s8 with real CGRA-sim evidence: {softmax_s8}")
        if (
            depthwise["diagnostic_class"] != "cgra_sim_pass"
            or depthwise["blocking_prerequisite"] != ""
            or depthwise["dfg_status"] != "pass"
            or depthwise["mapping_status"] != "pass"
            or depthwise["final_outputs_present"] != "true"
            or depthwise["final_memory_state_present"] != "true"
        ):
            raise AssertionError(f"default-batch rollup should promote depthwise with real CGRA-sim evidence: {depthwise}")

        evidence_dir = out_dir / "current-sim-cycle"
        for artifact in (
            evidence_dir / "arm_add_q15.dfg.report.json",
            evidence_dir / "arm_add_q15.mapping.json",
            evidence_dir / "arm_add_q15.cgra.report.json",
            evidence_dir / "arm_abs_f32.dfg.report.json",
            evidence_dir / "arm_abs_f32.mapping.json",
            evidence_dir / "arm_abs_f32.cgra.report.json",
            evidence_dir / "arm_sqrt_q15.dfg.report.json",
            evidence_dir / "arm_sqrt_q15.mapping.json",
            evidence_dir / "arm_sqrt_q15.cgra.report.json",
            evidence_dir / "arm_q7_to_q15_with_offset.dfg.report.json",
            evidence_dir / "arm_q7_to_q15_with_offset.mapping.json",
            evidence_dir / "arm_q7_to_q15_with_offset.cgra.report.json",
            evidence_dir / "arm_relu_q15.dfg.report.json",
            evidence_dir / "arm_relu_q15.mapping.json",
            evidence_dir / "arm_relu_q15.cgra.report.json",
            evidence_dir / "arm_fill_f32.dfg.report.json",
            evidence_dir / "arm_fill_f32.mapping.json",
            evidence_dir / "arm_fill_f32.cgra.report.json",
            evidence_dir / "arm_relu6_s8.dfg.report.json",
            evidence_dir / "arm_relu6_s8.mapping.json",
            evidence_dir / "arm_relu6_s8.cgra.report.json",
            evidence_dir / "arm_fully_connected_s8.dfg.report.json",
            evidence_dir / "arm_fully_connected_s8.mapping.json",
            evidence_dir / "arm_fully_connected_s8.cgra.report.json",
            evidence_dir / "arm_softmax_u8.dfg.report.json",
            evidence_dir / "arm_softmax_u8.mapping.json",
            evidence_dir / "arm_softmax_u8.cgra.report.json",
            evidence_dir / "arm_softmax_s8.dfg.report.json",
            evidence_dir / "arm_softmax_s8.mapping.json",
            evidence_dir / "arm_softmax_s8.cgra.report.json",
            evidence_dir / "arm_depthwise_conv_s8.dfg.report.json",
            evidence_dir / "arm_depthwise_conv_s8.mapping.json",
            evidence_dir / "arm_depthwise_conv_s8.cgra.report.json",
            evidence_dir / "arm_avgpool_s8.dfg.report.json",
        ):
            if not artifact.is_file():
                raise AssertionError(f"default-batch rollup did not emit {artifact}")
        for artifact in (
            evidence_dir / "arm_avgpool_s8.mapping.json",
            evidence_dir / "arm_avgpool_s8.cgra.report.json",
        ):
            if artifact.exists():
                raise AssertionError(f"avgpool DFG unsupported evidence must not emit {artifact}")
        if (
            avgpool["status"] != "unsupported"
            or avgpool["diagnostic_class"] != "dfg_report_unsupported"
            or avgpool["blocking_prerequisite"] != "dfg_report"
            or avgpool["dfg_status"] != "unsupported"
            or avgpool["mapping_status"] != "not_run"
            or avgpool["cgra_status"] != "not_run"
            or avgpool["comparison_status"] != "not_run"
            or avgpool["dfg_report"] == ""
            or avgpool["mapping_artifact"] != ""
            or avgpool["cgra_report"] != ""
            or "dataflow.graph.func 'g_arm_avgpool_s8_0' was not found" not in avgpool["diagnostic"]
        ):
            raise AssertionError(f"default-batch avgpool row should expose exact DFG unsupported evidence: {avgpool}")
        avgpool_dfg = json.loads((evidence_dir / "arm_avgpool_s8.dfg.report.json").read_text())
        if (
            avgpool_dfg.get("status") != "unsupported"
            or avgpool_dfg.get("graph") != "g_arm_avgpool_s8_0"
            or avgpool_dfg.get("workload") != "PoolingFunctions/arm_avgpool_s8.c"
            or avgpool_dfg.get("dynamic_work_items") != 0
            or avgpool_dfg.get("final_outputs") != []
            or avgpool_dfg.get("final_memory_state") != {}
            or "dataflow.graph.func 'g_arm_avgpool_s8_0' was not found" not in avgpool_dfg.get("diagnostics", [])
        ):
            raise AssertionError(f"unexpected avgpool DFG unsupported report: {avgpool_dfg}")
        abs_dfg = json.loads((evidence_dir / "arm_abs_f32.dfg.report.json").read_text())
        abs_cgra = json.loads((evidence_dir / "arm_abs_f32.cgra.report.json").read_text())
        expected_abs_memory = {
            "arg4": ["f32:-1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:1", "f32:2", "f32:3.500000", "f32:4.250000"],
        }
        for label, report in (("DFG", abs_dfg), ("CGRA", abs_cgra)):
            if report.get("status") != "pass" or report.get("final_memory_state") != expected_abs_memory:
                raise AssertionError(f"arm_abs_f32 {label} report should preserve real abs final state: {report}")


def assert_convolve_selector_records_row_pass_from_all_components(repo: Path) -> None:
    with artifact_test_common.repo_temp_dir(repo, "cmsis-convolve-attempt-") as tmp:
        root = Path(tmp)
        out_dir = root / "rollup"
        evidence_dir = root / "sim-evidence"

        run_rollup(repo, out_dir, evidence_dir, "arm_convolve_1x1_s8_fast")
        row = row_by_case(
            out_dir / "cgra-status-summary.csv",
            "cmsis-nn",
            "ConvolutionFunctions/arm_convolve_1x1_s8_fast.c",
        )
        if (
            row["status"] != "pass"
            or row["dfg_status"] != "pass"
            or row["mapping_status"] != "pass"
            or row["cgra_status"] != "pass"
            or row["comparison_status"] != "pass"
            or row["diagnostic_class"] != "cgra_sim_pass"
            or row["blocking_prerequisite"] != ""
            or row["required_slice_count"] != "4"
            or row["final_outputs_present"] != "true"
            or "arm_convolve_1x1_s8_fast.dfg.report.json" not in row["dfg_report"]
            or "arm_convolve_1x1_s8_fast.mapping.json" not in row["mapping_artifact"]
            or "arm_convolve_1x1_s8_fast.cgra.report.json" not in row["cgra_report"]
        ):
            raise AssertionError(f"convolve selector should promote all component evidence to a row pass: {row}")

        aggregate_dfg = json.loads((evidence_dir / "arm_convolve_1x1_s8_fast.dfg.report.json").read_text())
        expected_graphs = [
            "g_t_arm_nn_mat_mult_nt_t_s8_red_0_0",
            "g_t_arm_nn_mat_mult_nt_t_s8_red_1_0",
            "g_t_arm_nn_mat_mult_nt_t_s8_red_2_0",
            "g_t_arm_nn_mat_mult_nt_t_s8_red_3_0",
        ]
        if (
            aggregate_dfg.get("status") != "pass"
            or aggregate_dfg.get("graph") != "workload_graph_set"
            or aggregate_dfg.get("component_graphs") != expected_graphs
            or aggregate_dfg.get("final_outputs") != [
                "none",
                "i32:98",
                "i32:-9",
                "i32:98",
                "i32:-9",
                "none",
                "none",
                "i32:21648",
                "i32:32462",
                "none",
            ]
        ):
            raise AssertionError(f"convolve aggregate DFG should cover all four source graphs: {aggregate_dfg}")

        for component in ("red0", "red1", "red2", "red3"):
            for suffix in ("dfg.report.json", "mapping.json", "cgra.report.json"):
                artifact = evidence_dir / f"arm_convolve_1x1_s8_fast.{component}.{suffix}"
                if not artifact.is_file():
                    raise AssertionError(f"convolve component evidence is missing {artifact}")

        red0 = json.loads((evidence_dir / "arm_convolve_1x1_s8_fast.red0.dfg.report.json").read_text())
        if (
            red0.get("status") != "pass"
            or red0.get("graph") != "g_t_arm_nn_mat_mult_nt_t_s8_red_0_0"
            or red0.get("dynamic_work_items") != 1
            or red0.get("operation_fire_counts", {}).get("dataflow.load") != 1
            or red0.get("operation_fire_counts", {}).get("llvm.load") != 1
            or red0.get("final_outputs") != ["none", "i32:98", "i32:-9", "i32:98", "i32:-9"]
            or red0.get("diagnostics") != []
        ):
            raise AssertionError(f"convolve red0 report should expose real source-derived evidence: {red0}")

        red1 = json.loads((evidence_dir / "arm_convolve_1x1_s8_fast.red1.dfg.report.json").read_text())
        if (
            red1.get("status") != "pass"
            or red1.get("graph") != "g_t_arm_nn_mat_mult_nt_t_s8_red_1_0"
            or red1.get("dynamic_work_items") != 1
            or red1.get("operation_fire_counts", {}).get("dataflow.store") != 4
            or red1.get("final_memory_state", {}).get("arg24") != ["i8:127", "i8:127", "i8:127", "i8:0"]
            or red1.get("diagnostics") != []
        ):
            raise AssertionError(f"convolve red1 report should expose real quantized store evidence: {red1}")

        report_path = evidence_dir / "arm_convolve_1x1_s8_fast.red2.dfg.report.json"
        report = json.loads(report_path.read_text())
        if (
            report.get("status") != "pass"
            or report.get("graph") != "g_t_arm_nn_mat_mult_nt_t_s8_red_2_0"
            or report.get("dynamic_work_items") != 19
            or report.get("operation_fire_counts", {}).get("dataflow.load") != 38
            or report.get("operation_fire_counts", {}).get("llvm.load") != 19
            or report.get("operation_fire_counts", {}).get("dataflow.sync") != 19
            or report.get("final_outputs") != ["none", "i32:21648", "i32:32462"]
            or report.get("diagnostics") != []
        ):
            raise AssertionError(f"convolve red2 report should expose complete DFG-sim evidence: {report}")

        mapping_path = evidence_dir / "arm_convolve_1x1_s8_fast.red2.mapping.json"
        mapping = json.loads(mapping_path.read_text())
        if (
            mapping.get("status") != "pass"
            or mapping.get("graph") != "g_t_arm_nn_mat_mult_nt_t_s8_red_2_0"
            or mapping.get("unplaced_records") != 0
            or mapping.get("unrouted_edges") != 0
            or mapping.get("resource_pressure") not in (None, [])
        ):
            raise AssertionError(f"convolve red2 mapping should pass on the shared quantized ADG: {mapping}")
        placed_ops = {placement.get("operation") for placement in mapping.get("placements", [])}
        if not {"dataflow.stream", "dataflow.carry", "dataflow.invariant"} <= placed_ops:
            raise AssertionError(f"convolve red2 mapping missed loop-control placements: {mapping}")

        cgra_path = evidence_dir / "arm_convolve_1x1_s8_fast.red2.cgra.report.json"
        cgra = json.loads(cgra_path.read_text())
        if (
            cgra.get("status") != "pass"
            or cgra.get("kind") != "cgra_sim_report"
            or "g_t_arm_nn_mat_mult_nt_t_s8_red_2_0" not in str(cgra.get("mapping_id", ""))
            or cgra.get("hardware") != "shared_quantized_window_adg"
            or cgra.get("final_outputs") != ["none", "i32:21648", "i32:32462"]
            or int(cgra.get("placed_records", 0)) <= 0
            or int(cgra.get("routed_edges", 0)) <= 0
            or int(cgra.get("route_segments", 0)) <= 0
        ):
            raise AssertionError(f"convolve red2 CGRA report should pass after mapping succeeds: {cgra}")

        red3 = json.loads((evidence_dir / "arm_convolve_1x1_s8_fast.red3.dfg.report.json").read_text())
        if (
            red3.get("status") != "pass"
            or red3.get("graph") != "g_t_arm_nn_mat_mult_nt_t_s8_red_3_0"
            or red3.get("dynamic_work_items") != 1
            or red3.get("operation_fire_counts", {}).get("dataflow.store") != 1
            or red3.get("final_memory_state", {}).get("arg22") != ["i8:0", "i8:0", "i8:0", "i8:0"]
            or red3.get("diagnostics") != []
        ):
            raise AssertionError(f"convolve red3 report should expose real quantized store evidence: {red3}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    attempts = load_attempt_module(repo)

    add_args = attempts.parse_args(
        [
            "--cmsis-dsp-dfg-dir",
            "dsp",
            "--cmsis-nn-dfg-dir",
            "nn",
            "--output-dir",
            "out",
            "--attempt-stem",
            "arm_add_q15",
        ]
    )
    add_selected = attempts.select_attempts(add_args)
    if labels(add_selected) != ["arm_add_q15"]:
        raise AssertionError(f"arm_add_q15 selector chose unexpected attempts: {labels(add_selected)}")

    relu_args = attempts.parse_args(
        [
            "--cmsis-dsp-dfg-dir",
            "dsp",
            "--cmsis-nn-dfg-dir",
            "nn",
            "--output-dir",
            "out",
            "--case",
            "ActivationFunctions/arm_relu_q7.c",
        ]
    )
    relu_selected = attempts.select_attempts(relu_args)
    if labels(relu_selected) != ["arm_relu_q7.red0", "arm_relu_q7.red1"]:
        raise AssertionError(f"arm_relu_q7 case selector chose unexpected attempts: {labels(relu_selected)}")

    aggregate_args = attempts.parse_args(
        [
            "--cmsis-dsp-dfg-dir",
            "dsp",
            "--cmsis-nn-dfg-dir",
            "nn",
            "--output-dir",
            "out",
            "--attempt-stem",
            "arm_var_f32",
        ]
    )
    aggregate_selected = attempts.select_attempts(aggregate_args)
    if labels(aggregate_selected) != ["arm_var_f32.red0", "arm_var_f32.red1"]:
        raise AssertionError(f"aggregate selector chose unexpected attempts: {labels(aggregate_selected)}")

    max_pool_args = attempts.parse_args(
        [
            "--cmsis-dsp-dfg-dir",
            "dsp",
            "--cmsis-nn-dfg-dir",
            "nn",
            "--output-dir",
            "out",
            "--attempt-stem",
            "arm_max_pool_s8",
        ]
    )
    max_pool_selected = attempts.select_attempts(max_pool_args)
    if labels(max_pool_selected) != ["arm_max_pool_s8"]:
        raise AssertionError(f"arm_max_pool_s8 selector chose unexpected attempts: {labels(max_pool_selected)}")

    avgpool_args = attempts.parse_args(
        [
            "--cmsis-dsp-dfg-dir",
            "dsp",
            "--cmsis-nn-dfg-dir",
            "nn",
            "--output-dir",
            "out",
            "--attempt-stem",
            "arm_avgpool_s8",
        ]
    )
    avgpool_selected = attempts.select_attempts(avgpool_args)
    if labels(avgpool_selected) != ["arm_avgpool_s8"]:
        raise AssertionError(f"arm_avgpool_s8 selector chose unexpected attempts: {labels(avgpool_selected)}")

    depthwise_args = attempts.parse_args(
        [
            "--cmsis-dsp-dfg-dir",
            "dsp",
            "--cmsis-nn-dfg-dir",
            "nn",
            "--output-dir",
            "out",
            "--attempt-stem",
            "arm_depthwise_conv_s8",
        ]
    )
    depthwise_selected = attempts.select_attempts(depthwise_args)
    if labels(depthwise_selected) != ["arm_depthwise_conv_s8.red0", "arm_depthwise_conv_s8.red1"]:
        raise AssertionError(f"arm_depthwise_conv_s8 selector chose unexpected attempts: {labels(depthwise_selected)}")

    convolve_args = attempts.parse_args(
        [
            "--cmsis-dsp-dfg-dir",
            "dsp",
            "--cmsis-nn-dfg-dir",
            "nn",
            "--output-dir",
            "out",
            "--attempt-stem",
            "arm_convolve_1x1_s8_fast",
        ]
    )
    convolve_selected = attempts.select_attempts(convolve_args)
    if labels(convolve_selected) != [
        "arm_convolve_1x1_s8_fast.red0",
        "arm_convolve_1x1_s8_fast.red1",
        "arm_convolve_1x1_s8_fast.red2",
        "arm_convolve_1x1_s8_fast.red3",
    ]:
        raise AssertionError(
            f"arm_convolve_1x1_s8_fast selector chose unexpected attempts: {labels(convolve_selected)}"
        )

    bad_args = attempts.parse_args(
        [
            "--cmsis-dsp-dfg-dir",
            "dsp",
            "--cmsis-nn-dfg-dir",
            "nn",
            "--output-dir",
            "out",
            "--attempt-stem",
            "not_a_cmsis_attempt",
        ]
    )
    try:
        attempts.select_attempts(bad_args)
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("unknown CMSIS attempt selector unexpectedly passed")
    if "not_a_cmsis_attempt" not in message or "arm_add_q15" not in message:
        raise AssertionError(f"unknown selector diagnostic is not actionable: {message}")

    blank_args = attempts.parse_args(
        [
            "--cmsis-dsp-dfg-dir",
            "dsp",
            "--cmsis-nn-dfg-dir",
            "nn",
            "--output-dir",
            "out",
            "--attempt-stem",
            "",
        ]
    )
    try:
        attempts.select_attempts(blank_args)
    except SystemExit as exc:
        blank_message = str(exc)
    else:
        raise AssertionError("blank CMSIS attempt selector unexpectedly passed")
    if "must not be blank" not in blank_message:
        raise AssertionError(f"blank selector diagnostic is not actionable: {blank_message}")

    bad_cli = subprocess.run(
        [
            sys.executable,
            "test/e2e/run_cmsis_dfg_sim_attempts.py",
            "--cmsis-dsp-dfg-dir",
            "missing-dsp",
            "--cmsis-nn-dfg-dir",
            "missing-nn",
            "--output-dir",
            "temp/cmsis-selector-bad-cli",
            "--attempt-stem",
            "not_a_cmsis_attempt",
            "--loom-dfg-sim",
            "definitely-missing-loom-dfg-sim",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if bad_cli.returncode == 0:
        raise AssertionError("unknown CMSIS attempt selector CLI unexpectedly passed")
    if "not_a_cmsis_attempt" not in bad_cli.stderr or "not executable" in bad_cli.stderr:
        raise AssertionError(f"CLI should validate selectors before tools: {bad_cli.stderr}")

    selector_default = subprocess.run(
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            "temp/cmsis-selector-no-mode",
            "--no-legacy-loombench",
            "--cmsis-sim-attempt-stem",
            "arm_add_q15",
            "--jobs",
            "8",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if selector_default.returncode != 0:
        raise AssertionError(
            "CMSIS sim selector without an explicit sim evidence mode should run the selected row\n"
            f"stdout:\n{selector_default.stdout}\nstderr:\n{selector_default.stderr}"
        )
    selector_default_data = json.loads((repo / "temp/cmsis-selector-no-mode/cgra-status-summary.json").read_text())
    selector_default_counts = selector_default_data.get("counts", {}).get("cmsis-dsp")
    if selector_default_counts != {
        "total": 16,
        "pass": 1,
        "fail": 0,
        "blocked": 15,
        "unsupported": 0,
        "missing_status": 0,
    }:
        raise AssertionError(f"selected default CMSIS counts are wrong: {selector_default_counts}")
    selector_default_row = row_by_case(
        repo / "temp/cmsis-selector-no-mode/cgra-status-summary.csv",
        "cmsis-dsp",
        "BasicMathFunctions/arm_add_q15.c",
    )
    if (
        selector_default_row["status"] != "pass"
        or selector_default_row["dfg_status"] != "pass"
        or selector_default_row["mapping_status"] != "pass"
        or selector_default_row["cgra_status"] != "pass"
        or selector_default_row["comparison_status"] != "pass"
    ):
        raise AssertionError(f"selected default CMSIS row should expose real CGRA-sim evidence: {selector_default_row}")
    for artifact in (
        repo / "temp/cmsis-selector-no-mode/current-sim-cycle/arm_add_q15.dfg.report.json",
        repo / "temp/cmsis-selector-no-mode/current-sim-cycle/arm_add_q15.mapping.json",
        repo / "temp/cmsis-selector-no-mode/current-sim-cycle/arm_add_q15.cgra.report.json",
    ):
        if not artifact.is_file():
            raise AssertionError(f"selected default CMSIS rollup did not emit {artifact}")

    legacy_alias = subprocess.run(
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--cmsis-sim-seed-batch",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if legacy_alias.returncode == 0:
        raise AssertionError("legacy CMSIS batch alias without output dir unexpectedly passed")
    if "--output-dir is required" not in legacy_alias.stderr or "unknown argument" in legacy_alias.stderr:
        raise AssertionError(f"legacy CMSIS batch alias should remain accepted: {legacy_alias.stderr}")

    assert_selected_rollup_drops_stale_cmsis_evidence(repo)
    assert_default_batch_rollup_promotes_bounded_rows(repo)
    assert_convolve_selector_records_row_pass_from_all_components(repo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
