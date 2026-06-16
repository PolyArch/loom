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


def run(
    repo: Path, argv: list[str], *, expect_success: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"command failed with {result.returncode}: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(
            f"command unexpectedly passed: {' '.join(argv)}\n"
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
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_synthetic_root = out_dir / "no-legacy-loombench-root"
    stale_synthetic_root.mkdir()
    (stale_synthetic_root / "stale_case").mkdir()
    stale_manifest = out_dir / "loombench-manifest.json"
    stale_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "loombench_manifest",
                "csv_projection": "",
                "case_count": 1,
                "cases": [
                    {
                        "case": "stale_legacy_case",
                        "source_row": "stale_legacy_case",
                        "software_root": "stale",
                        "source_fingerprint": "0" * 64,
                        "main_source": "main.cpp",
                        "implementation_sources": [],
                        "headers": [],
                        "feature_tags": [],
                        "import_state": "deferred",
                        "manifest_case": "",
                        "oracle": "legacy_reference",
                        "input_profile": "legacy_default",
                        "tier_states": {
                            "source": "blocked",
                            "raise": "blocked",
                            "dataflow": "blocked",
                            "cgra_status": "blocked",
                        },
                        "owner": "test",
                        "reason": "stale sidecar should not be consumed without a legacy root",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
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
    if (out_dir / "loombench-manifest.csv").exists():
        raise AssertionError("no-legacy rollup should not emit LoomBench manifest CSV artifacts")
    stale_data = json.loads(stale_manifest.read_text())
    if stale_data.get("cases", [{}])[0].get("case") != "stale_legacy_case":
        raise AssertionError("no-legacy rollup should not overwrite stale LoomBench manifest sidecar")


def assert_direct_cmsis_dfg_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    csv_output = out_dir / "cgra-status-summary.csv"
    json_output = out_dir / "cgra-status-summary.json"
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cgra_status_summary.sh",
            "--output",
            str(csv_output),
            "--json-output",
            str(json_output),
            "--legacy-loombench-root",
            str(legacy_root),
            "--cmsis-dfg-auto",
        ],
    )
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cgra_status_audit.sh",
            "--input",
            str(csv_output),
            "--json-input",
            str(json_output),
            "--legacy-loombench-root",
            str(legacy_root),
        ],
    )
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
    for artifact in (
        out_dir / "cmsis-dsp-dfg" / "arm_add_q15.dfg.mlir",
        out_dir / "cmsis-nn-dfg" / "arm_relu_q15.dfg.mlir",
    ):
        if not artifact.is_file():
            raise AssertionError(f"direct CMSIS DFG mode should emit {artifact}")
    dsp_add = one_row(rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
    nn_relu = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
    for row in (dsp_add, nn_relu):
        if (
            row["status"] != "blocked"
            or row["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or row["blocking_prerequisite"] != "dfg_sim_report"
            or not row["graph_ids"]
        ):
            raise AssertionError(f"direct CMSIS DFG mode should publish exact DFG blockers: {row}")
        assert_sha256_file(row["dfg_mlir"], row["dfg_mlir_fingerprint"], repo)


def assert_cmsis_dfg_report(
    sim_evidence: Path,
    stem: str,
    workload: str,
    fired_op: str,
    final_memory_state: dict[str, list[str]],
) -> None:
    report = sim_evidence / f"{stem}.dfg.report.json"
    if not report.is_file():
        raise AssertionError(f"CMSIS DFG-sim evidence mode should emit {report}")
    report_data = json.loads(report.read_text())
    if (
        report_data.get("kind") != "dfg_sim_report"
        or report_data.get("workload") != workload
        or report_data.get("status") != "pass"
        or report_data.get("dynamic_work_items") != 4
        or report_data.get("operation_fire_counts", {}).get(fired_op) != 4
    ):
        raise AssertionError(f"unexpected CMSIS DFG-sim report: {report_data}")
    for key, expected in final_memory_state.items():
        if report_data.get("final_memory_state", {}).get(key) != expected:
            raise AssertionError(f"unexpected CMSIS DFG-sim memory for {stem}: {report_data}")
    input_fingerprints = report_data.get("input_artifact_fingerprints")
    if not isinstance(input_fingerprints, dict) or f"{stem}.dfg" not in input_fingerprints:
        raise AssertionError(f"CMSIS DFG-sim report should fingerprint its input DFG MLIR: {report_data}")


def assert_cmsis_cgra_pass_row(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
    suite: str,
    case: str,
    stem: str,
) -> None:
    row = one_row(rows, suite, case)
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["blocking_prerequisite"] != ""
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["hardware_system"] != "shared_reduction_adg"
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
    ):
        raise AssertionError(f"CMSIS row should expose real CGRA-sim evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
    for artifact in (
        sim_evidence / f"{stem}.dfg.report.json",
        sim_evidence / f"{stem}.mapping.json",
        sim_evidence / f"{stem}.cgra.report.json",
    ):
        if not artifact.is_file():
            raise AssertionError(f"CMSIS evidence mode should emit {artifact}")


def assert_cmsis_add_q15_shared_adg_evidence(sim_evidence: Path) -> None:
    mapping_artifact = json.loads((sim_evidence / "arm_add_q15.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "placed_records": 14,
        "routed_edges": 19,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 430,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_add_q15 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 19:
        raise AssertionError(f"arm_add_q15 mapping should expose every routed edge: {mapping_artifact}")
    required_edges = {
        "dataflow.load#0.result0->llvm.sext#0.operand0",
        "dataflow.load#1.result0->llvm.sext#1.operand0",
        "llvm.sext#0.result0->llvm.arm.qadd16#0.operand0",
        "llvm.sext#1.result0->llvm.arm.qadd16#0.operand1",
        "llvm.arm.qadd16#0.result0->llvm.trunc#0.operand0",
        "llvm.trunc#0.result0->dataflow.store#0.operand2",
    }
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_add_q15 mapping missed q15 datapath route evidence: {mapping_artifact}")
    for route in routes:
        if route.get("edge_ref") not in required_edges:
            continue
        segments = route.get("segments", [])
        if not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in segments):
            raise AssertionError(f"arm_add_q15 route should traverse Fabric paths: {route}")

    cgra_report = json.loads((sim_evidence / "arm_add_q15.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "hardware_aware_cycles": 196,
        "performance_delta_cycles": 89,
        "route_segments": 77,
        "config_records": 430,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_add_q15 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")


def assert_cmsis_fill_shared_adg_evidence(sim_evidence: Path) -> None:
    mapping_artifact = json.loads((sim_evidence / "arm_fill_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "placed_records": 9,
        "routed_edges": 11,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 246,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_fill_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 11:
        raise AssertionError(f"arm_fill_f32 mapping should expose every routed edge: {mapping_artifact}")
    required_edges = {
        "dataflow.carry#0.result0->dataflow.store#0.operand1",
        "dataflow.invariant#0.result0->dataflow.store#0.operand2",
        "dataflow.store#0.result0->dataflow.sync#0.operand0",
    }
    actual_edges = {
        route.get("edge_ref")
        for route in routes
        if isinstance(route, dict)
    }
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_fill_f32 mapping missed store route evidence: {mapping_artifact}")
    multi_hop_routes = [
        route
        for route in routes
        if route.get("edge_ref") in required_edges
        and any(
            segment.get("segment_kind") == "module_path"
            for segment in route.get("segments", [])
            if isinstance(segment, dict)
        )
    ]
    if len(multi_hop_routes) != len(required_edges):
        raise AssertionError(f"arm_fill_f32 mapping should route store edges through Fabric paths: {mapping_artifact}")

    cgra_report = json.loads((sim_evidence / "arm_fill_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "hardware_aware_cycles": 112,
        "performance_delta_cycles": 47,
        "route_segments": 43,
        "config_records": 246,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_fill_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")


def assert_cmsis_dfg_sim_evidence_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    sim_evidence = out_dir / "cmsis-sim-evidence"
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--sim-evidence-dir",
            str(sim_evidence),
        ],
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_abs_f32",
        "BasicMathFunctions/arm_abs_f32.c",
        "llvm.intr.fabs",
        {
            "arg4": ["f32:-1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:1", "f32:2", "f32:3.500000", "f32:4.250000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_mult_f32",
        "BasicMathFunctions/arm_mult_f32.c",
        "arith.mulf",
        {
            "arg4": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:2", "f32:-2", "f32:-10.500000", "f32:2.125000"],
            "arg6": ["f32:2", "f32:-1", "f32:3", "f32:0.500000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_add_q15",
        "BasicMathFunctions/arm_add_q15.c",
        "llvm.arm.qadd16",
        {
            "arg4": ["i16:1000", "i16:20000", "i16:-30000", "i16:32760"],
            "arg5": ["i16:3000", "i16:32767", "i16:-32768", "i16:32767"],
            "arg6": ["i16:2000", "i16:15000", "i16:-10000", "i16:1000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_mat_add_f32",
        "MatrixFunctions/arm_mat_add_f32.c",
        "arith.addf",
        {
            "arg4": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:2", "f32:-1", "f32:3", "f32:0.500000"],
            "arg6": ["f32:3", "f32:1", "f32:-0.500000", "f32:4.750000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_copy_f32",
        "SupportFunctions/arm_copy_f32.c",
        "dataflow.load",
        {
            "arg4": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
            "arg5": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
        },
    )
    assert_cmsis_dfg_report(
        sim_evidence,
        "arm_fill_f32",
        "SupportFunctions/arm_fill_f32.c",
        "dataflow.store",
        {
            "arg5": ["f32:3.250000", "f32:3.250000", "f32:3.250000", "f32:3.250000"],
        },
    )

    rows = read_rows(out_dir / "cgra-status-summary.csv")
    data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "cmsis-dsp",
        {
            "total": 16,
            "pass": 7,
            "fail": 0,
            "blocked": 7,
            "unsupported": 2,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "cmsis-nn",
        {
            "total": 18,
            "pass": 1,
            "fail": 0,
            "blocked": 9,
            "unsupported": 8,
            "missing_status": 0,
        },
    )
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_abs_f32.c", "arm_abs_f32")
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_mult_f32.c", "arm_mult_f32")
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c", "arm_add_q15")
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_offset_f32.c", "arm_offset_f32"
    )
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "MatrixFunctions/arm_mat_add_f32.c", "arm_mat_add_f32"
    )
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "SupportFunctions/arm_copy_f32.c", "arm_copy_f32")
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "SupportFunctions/arm_fill_f32.c", "arm_fill_f32")
    assert_cmsis_add_q15_shared_adg_evidence(sim_evidence)
    assert_cmsis_fill_shared_adg_evidence(sim_evidence)
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c", "arm_relu_q15"
    )

    fake_cgra_tool = out_dir / "not-executable-cgra-sim"
    fake_cgra_tool.write_text("#!/bin/sh\nexit 99\n")
    no_cgra_evidence = out_dir / "cmsis-sim-evidence-no-cgra"
    fake_result = run(
        repo,
        [
            sys.executable,
            "test/e2e/run_cmsis_dfg_sim_attempts.py",
            "--cmsis-dsp-dfg-dir",
            str(out_dir / "cmsis-dsp-dfg"),
            "--cmsis-nn-dfg-dir",
            str(out_dir / "cmsis-nn-dfg"),
            "--output-dir",
            str(no_cgra_evidence),
            "--loom-cgra-sim",
            str(fake_cgra_tool),
        ],
        expect_success=False,
    )
    if "CGRA-sim" not in fake_result.stderr and "loom-cgra-sim" not in fake_result.stderr:
        raise AssertionError(f"CMSIS offset should fail at unavailable CGRA-sim: {fake_result.stderr}")
    if not (no_cgra_evidence / "arm_abs_f32.mapping.json").is_file():
        raise AssertionError("CMSIS abs should emit mapping evidence before requiring CGRA-sim")
    if (no_cgra_evidence / "arm_abs_f32.cgra.report.json").exists():
        raise AssertionError("CMSIS abs should not emit CGRA evidence from a failing CGRA-sim tool")


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
        assert_direct_cmsis_dfg_mode(repo, out_dir / "direct-cmsis-dfg", legacy_root)
        assert_cmsis_dfg_sim_evidence_mode(repo, out_dir / "cmsis-dfg-sim-evidence", legacy_root)
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
