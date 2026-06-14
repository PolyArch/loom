#!/usr/bin/env python3
"""Regression test for real CMSIS DFG evidence in CGRA status rollup."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import artifact_test_common

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cgra_status_summary import assert_sha256_file, one_row, read_rows  # noqa: E402


def run(repo: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed with {result.returncode}: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def assert_counts(data: dict[str, object], suite: str, expected: dict[str, int]) -> None:
    counts = data.get("counts")
    if not isinstance(counts, dict) or counts.get(suite) != expected:
        raise AssertionError(f"unexpected {suite} counts: {counts.get(suite) if isinstance(counts, dict) else counts}")


def assert_no_cmsis_pass(rows: list[dict[str, str]]) -> None:
    passed = [row for row in rows if row["suite"] in {"cmsis-dsp", "cmsis-nn"} and row["status"] == "pass"]
    if passed:
        raise AssertionError(f"CMSIS DFG-only rollup must not claim CGRA pass rows: {passed[:3]}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "cmsis-cgra-status-rollup-") as raw_out_dir:
        out_dir = Path(raw_out_dir)
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cmsis_cgra_status_rollup.sh",
                "--output-dir",
                str(out_dir),
            ],
        )

        csv_output = out_dir / "cgra-status-summary.csv"
        json_output = out_dir / "cgra-status-summary.json"
        audit_output = out_dir / "cgra-status-generic-audit.json"
        for artifact in (csv_output, json_output, audit_output):
            if not artifact.is_file():
                raise AssertionError(f"missing expected rollup artifact: {artifact}")

        rows = read_rows(csv_output)
        data = json.loads(json_output.read_text())
        assert_counts(
            data,
            "cmsis-dsp",
            {
                "total": 16,
                "pass": 0,
                "fail": 0,
                "blocked": 14,
                "unsupported": 2,
                "missing_status": 0,
            },
        )
        assert_counts(
            data,
            "cmsis-nn",
            {
                "total": 18,
                "pass": 0,
                "fail": 0,
                "blocked": 10,
                "unsupported": 8,
                "missing_status": 0,
            },
        )
        assert_no_cmsis_pass(rows)

        dsp_add = one_row(rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        if (
            dsp_add["status"] != "blocked"
            or dsp_add["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or dsp_add["blocking_prerequisite"] != "dfg_sim_report"
            or dsp_add["required_slice_count"] != "1"
            or "g_t_arm_add_q15_red_0_0" not in dsp_add["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-DSP add row should be an exact DFG-sim blocker: {dsp_add}")
        assert_sha256_file(dsp_add["dfg_mlir"], dsp_add["dfg_mlir_fingerprint"], repo)

        dsp_sin = one_row(rows, "cmsis-dsp", "FastMathFunctions/arm_sin_f32.c")
        if (
            dsp_sin["status"] != "unsupported"
            or dsp_sin["diagnostic_class"] != "cmsis_no_dataflow_graph"
            or dsp_sin["blocking_prerequisite"] != "dataflow_graph"
            or dsp_sin["required_slice_count"] != "0"
        ):
            raise AssertionError(f"CMSIS-DSP no-graph row should be structured unsupported: {dsp_sin}")
        assert_sha256_file(dsp_sin["dfg_mlir"], dsp_sin["dfg_mlir_fingerprint"], repo)

        nn_relu = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
        if (
            nn_relu["status"] != "blocked"
            or nn_relu["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or nn_relu["blocking_prerequisite"] != "dfg_sim_report"
            or "g_t_arm_relu_q15_red_0_0" not in nn_relu["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-NN relu row should be an exact DFG-sim blocker: {nn_relu}")
        assert_sha256_file(nn_relu["dfg_mlir"], nn_relu["dfg_mlir_fingerprint"], repo)

        nn_relu6 = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu6_s8.c")
        if (
            nn_relu6["status"] != "unsupported"
            or nn_relu6["diagnostic_class"] != "cmsis_no_dataflow_graph"
            or nn_relu6["blocking_prerequisite"] != "dataflow_graph"
            or nn_relu6["required_slice_count"] != "0"
        ):
            raise AssertionError(f"CMSIS-NN no-graph row should be structured unsupported: {nn_relu6}")
        assert_sha256_file(nn_relu6["dfg_mlir"], nn_relu6["dfg_mlir_fingerprint"], repo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
