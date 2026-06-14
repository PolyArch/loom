#!/usr/bin/env python3
"""Regression test for real CMSIS DFG evidence in CGRA status rollup."""

from __future__ import annotations

import json
import subprocess
import sys
import csv
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def assert_manifest_projection(json_path: Path, csv_path: Path) -> None:
    data = json.loads(json_path.read_text())
    rows = read_csv_rows(csv_path)
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise AssertionError(f"LoomBench manifest JSON cases must be a list: {data}")
    if data.get("csv_projection") != str(csv_path):
        raise AssertionError(f"LoomBench manifest JSON should name CSV projection: {data}")
    if data.get("case_count") != len(cases) or len(cases) != len(rows):
        raise AssertionError(f"LoomBench manifest JSON/CSV row counts diverged: {data}, {rows}")
    by_case = {row["case"]: row for row in rows}
    for case_data in cases:
        if not isinstance(case_data, dict):
            raise AssertionError(f"LoomBench manifest case is not an object: {case_data}")
        row = by_case.get(str(case_data.get("case", "")))
        if row is None:
            raise AssertionError(f"LoomBench manifest CSV missing case: {case_data}")
        for key in (
            "case",
            "source_row",
            "software_root",
            "source_fingerprint",
            "import_state",
            "manifest_case",
            "owner",
            "reason",
        ):
            if row.get(key, "") != str(case_data.get(key, "")):
                raise AssertionError(f"LoomBench manifest CSV/JSON mismatch for {key}: {row}, {case_data}")


def assert_no_legacy_mode(repo: Path, out_dir: Path) -> None:
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
        ],
    )
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    loombench_rows = [row for row in rows if row["suite"] == "loombench"]
    if loombench_rows:
        raise AssertionError(f"no-legacy rollup should not emit LoomBench rows: {loombench_rows[:3]}")
    if (out_dir / "loombench-manifest.json").exists() or (out_dir / "loombench-manifest.csv").exists():
        raise AssertionError("no-legacy rollup should not emit LoomBench manifest artifacts")


def write_legacy_case(root: Path, name: str, *, with_header: bool = True) -> None:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    (case_dir / "main.cpp").write_text("int main() { return 0; }\n")
    (case_dir / f"{name}.cpp").write_text(f'#include "{name}.h"\n')
    if with_header:
        (case_dir / f"{name}.h").write_text("#pragma once\n")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "cmsis-cgra-status-rollup-") as raw_out_dir:
        out_dir = Path(raw_out_dir)
        assert_no_legacy_mode(repo, out_dir / "no-legacy")
        legacy_root = out_dir / "legacy-loombench"
        write_legacy_case(legacy_root, "legacy_missing")
        write_legacy_case(legacy_root, "vecadd")
        write_legacy_case(legacy_root, "blocked_case", with_header=False)
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cmsis_cgra_status_rollup.sh",
                "--output-dir",
                str(out_dir),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
        )

        csv_output = out_dir / "cgra-status-summary.csv"
        json_output = out_dir / "cgra-status-summary.json"
        audit_output = out_dir / "cgra-status-generic-audit.json"
        manifest_output = out_dir / "loombench-manifest.json"
        manifest_csv_output = out_dir / "loombench-manifest.csv"
        for artifact in (csv_output, json_output, audit_output, manifest_output, manifest_csv_output):
            if not artifact.is_file():
                raise AssertionError(f"missing expected rollup artifact: {artifact}")
        assert_manifest_projection(manifest_output, manifest_csv_output)

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
        assert_counts(
            data,
            "loombench",
            {
                "total": 3,
                "pass": 0,
                "fail": 0,
                "blocked": 2,
                "unsupported": 1,
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

        loombench_vecadd = one_row(rows, "loombench", "vecadd")
        if (
            loombench_vecadd["status"] != "blocked"
            or loombench_vecadd["diagnostic_class"] != "loombench_workload_identity_fingerprint_missing"
            or loombench_vecadd["blocking_prerequisite"] != "loombench_workload_identity_fingerprint"
        ):
            raise AssertionError(f"LoomBench accepted row should block on fingerprint bridge: {loombench_vecadd}")
        loombench_deferred = one_row(rows, "loombench", "legacy_missing")
        if (
            loombench_deferred["status"] != "blocked"
            or loombench_deferred["diagnostic_class"] != "loombench_import_deferred"
            or loombench_deferred["blocking_prerequisite"] != "app_import"
        ):
            raise AssertionError(f"LoomBench deferred row should be a structured blocker: {loombench_deferred}")
        loombench_excluded = one_row(rows, "loombench", "blocked_case")
        if (
            loombench_excluded["status"] != "unsupported"
            or loombench_excluded["diagnostic_class"] != "loombench_import_excluded"
            or loombench_excluded["blocking_prerequisite"] != "legacy_source"
        ):
            raise AssertionError(f"LoomBench excluded row should be structured unsupported: {loombench_excluded}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
