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
                "pass": 14,
                "fail": 0,
                "blocked": 0,
                "unsupported": 2,
                "missing_status": 0,
            },
            "cmsis-nn": {
                "total": 18,
                "pass": 12,
                "fail": 0,
                "blocked": 0,
                "unsupported": 6,
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
        relu_q15 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
        relu_q7 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "ActivationFunctions/arm_relu_q7.c")
        relu6 = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "ActivationFunctions/arm_relu6_s8.c")
        reshape = row_by_case(out_dir / "cgra-status-summary.csv", "cmsis-nn", "ReshapeFunctions/arm_reshape_s8.c")
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
        for row in (add, abs_f32, fill, relu_q15, relu_q7, relu6, reshape, vector_sum):
            if row["status"] != "pass" or row["cgra_status"] != "pass" or row["comparison_status"] != "pass":
                raise AssertionError(f"default-batch row should expose CGRA-sim pass evidence: {row}")
        if (
            fully_connected["status"] != "unsupported"
            or fully_connected["diagnostic_class"] != "dfg_report_unsupported"
            or fully_connected["blocking_prerequisite"] != "dfg_report"
            or fully_connected["dfg_status"] != "unsupported"
            or "unsupported op: llvm.call" not in fully_connected["diagnostic"]
        ):
            raise AssertionError(
                "default-batch rollup should record the fully connected row's exact DFG blocker: "
                f"{fully_connected}"
            )

        evidence_dir = out_dir / "current-sim-cycle"
        for artifact in (
            evidence_dir / "arm_add_q15.dfg.report.json",
            evidence_dir / "arm_add_q15.mapping.json",
            evidence_dir / "arm_add_q15.cgra.report.json",
            evidence_dir / "arm_abs_f32.dfg.report.json",
            evidence_dir / "arm_abs_f32.mapping.json",
            evidence_dir / "arm_abs_f32.cgra.report.json",
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
        ):
            if not artifact.is_file():
                raise AssertionError(f"default-batch rollup did not emit {artifact}")
        abs_dfg = json.loads((evidence_dir / "arm_abs_f32.dfg.report.json").read_text())
        abs_cgra = json.loads((evidence_dir / "arm_abs_f32.cgra.report.json").read_text())
        expected_abs_memory = {
            "arg4": ["f32:-1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:1", "f32:2", "f32:3.500000", "f32:4.250000"],
        }
        for label, report in (("DFG", abs_dfg), ("CGRA", abs_cgra)):
            if report.get("status") != "pass" or report.get("final_memory_state") != expected_abs_memory:
                raise AssertionError(f"arm_abs_f32 {label} report should preserve real abs final state: {report}")


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

    no_mode = subprocess.run(
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            "temp/cmsis-selector-no-mode",
            "--cmsis-sim-attempt-stem",
            "arm_add_q15",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if no_mode.returncode == 0:
        raise AssertionError("CMSIS sim selector without a sim evidence mode unexpectedly passed")
    if "require --cmsis-sim-default or --sim-evidence-dir" not in no_mode.stderr:
        raise AssertionError(f"CMSIS sim selector mode diagnostic is not actionable: {no_mode.stderr}")

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
