#!/usr/bin/env python3
"""Regression test for real CMSIS DFG evidence in CGRA status rollup."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import csv
from pathlib import Path

import artifact_test_common

sys.path.insert(0, str(Path(__file__).resolve().parent))
from default_batch_test_common import default_batch_hardware  # noqa: E402
from test_cgra_status_summary import assert_sha256_file, one_row, read_rows  # noqa: E402


def run(
    repo: Path,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=repo,
        env=env,
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


def assert_cmsis_dfg_only_counts(data: dict[str, object]) -> None:
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


def assert_no_cmsis_pass(rows: list[dict[str, str]]) -> None:
    passed = [row for row in rows if row["suite"] in {"cmsis-dsp", "cmsis-nn"} and row["status"] == "pass"]
    if passed:
        raise AssertionError(f"CMSIS DFG-only rollup must not claim CGRA pass rows: {passed[:3]}")


def assert_no_sim_stage_evidence(row: dict[str, str]) -> None:
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        if row[key]:
            raise AssertionError(f"DFG-only row should not consume stale {key}: {row}")
    for key in ("dfg_status", "mapping_status", "cgra_status", "comparison_status"):
        if row[key] != "not_run":
            raise AssertionError(f"DFG-only row should leave {key}=not_run: {row}")


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
    assert_cmsis_dfg_only_counts(data)
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


def assert_app_cgra_sweep_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-default-batch",
        ],
    )
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "app",
        {
            "total": 109,
            "pass": 37,
            "fail": 0,
            "blocked": 50,
            "unsupported": 0,
            "missing_status": 22,
        },
    )
    expected_hardware = default_batch_hardware(repo)
    for case, hardware in expected_hardware.items():
        assert_app_cgra_pass_row(repo, rows, case, expected_hardware=hardware)
        for suffix in ("dfg.report.json", "mapping.json", "cgra.report.json"):
            artifact = out_dir / "current-sim-cycle" / f"{case}.{suffix}"
            if not artifact.is_file():
                raise AssertionError(f"app CGRA sweep mode should emit {artifact}")

    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-case",
            "vecsum",
        ],
    )
    stale_rows = read_rows(out_dir / "cgra-status-summary.csv")
    stale_data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        stale_data,
        "app",
        {
            "total": 109,
            "pass": 1,
            "fail": 0,
            "blocked": 50,
            "unsupported": 0,
            "missing_status": 58,
        },
    )
    assert_app_cgra_pass_row(repo, stale_rows, "vecsum", expected_hardware="shared_reduction_adg")
    dotproduct = one_row(stale_rows, "app", "dotproduct")
    if dotproduct["status"] == "pass" or dotproduct["dfg_report"]:
        raise AssertionError(f"app sweep mode should not reuse stale dotproduct evidence: {dotproduct}")


def assert_app_attempt_manifest_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-attempt-manifest",
            "test/app/shared-cgra-blocker-batch.json",
        ],
    )
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "app",
        {
            "total": 109,
            "pass": 0,
            "fail": 0,
            "blocked": 57,
            "unsupported": 0,
            "missing_status": 52,
        },
    )
    expected_diagnostics = {
        "crc32": "unsupported op: scf.for",
        "fir_filter": "unsupported op: scf.for",
        "merge": "unsupported op: scf.for",
        "convolve_1d_same": "unsupported op: scf.for",
        "autocorrelation": "unsupported op: scf.for",
        "binary_search": "primary workload graph absent: expected token binary_search_candidate",
        "partition": "unsupported op: scf.for",
    }
    for case, diagnostic in expected_diagnostics.items():
        row = one_row(rows, "app", case)
        if (
            row["status"] != "blocked"
            or row["diagnostic_class"] != "dfg_report_unsupported"
            or row["owner"] != "sim_report"
            or row["blocking_prerequisite"] != "dfg_report"
            or row["dfg_status"] != "unsupported"
            or row["mapping_status"] != "unsupported"
            or row["cgra_status"] != "blocked"
            or row["comparison_status"] != "blocked"
            or row["hardware_system"] != "shared_reduction_adg"
            or row["final_outputs_present"] != "false"
            or row["final_memory_state_present"] != "false"
            or diagnostic not in row["diagnostic"]
        ):
            raise AssertionError(f"attempted app row should expose structured shared-ADG blocker: {row}")
        for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
            assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
            artifact = out_dir / "current-sim-cycle" / Path(row[key]).name
            if not artifact.is_file():
                raise AssertionError(f"attempt manifest should emit {artifact}")


def assert_cmsis_sim_default_mode(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir),
            "--legacy-loombench-root",
            str(legacy_root),
            "--cmsis-sim-default",
        ],
    )
    sim_evidence = out_dir / "current-sim-cycle"
    rows = read_rows(out_dir / "cgra-status-summary.csv")
    data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_counts(
        data,
        "cmsis-dsp",
        {
            "total": 16,
            "pass": 11,
            "fail": 0,
            "blocked": 3,
            "unsupported": 2,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "cmsis-nn",
        {
            "total": 18,
            "pass": 2,
            "fail": 0,
            "blocked": 8,
            "unsupported": 8,
            "missing_status": 0,
        },
    )
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c", "arm_add_q15")
    assert_cmsis_cgra_pass_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-dsp",
        "FilteringFunctions/arm_biquad_cascade_df1_f32.c",
        "arm_biquad_cascade_df1_f32",
    )
    assert_cmsis_biquad_shared_adg_evidence(sim_evidence)
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c", "arm_relu_q15"
    )
    assert_cmsis_relu_q7_pass_row(repo, rows, sim_evidence)
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
    dfg_only_rows = read_rows(out_dir / "cgra-status-summary.csv")
    dfg_only_data = json.loads((out_dir / "cgra-status-summary.json").read_text())
    assert_cmsis_dfg_only_counts(dfg_only_data)
    assert_no_cmsis_pass(dfg_only_rows)
    assert_no_sim_stage_evidence(one_row(dfg_only_rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c"))
    assert_no_sim_stage_evidence(one_row(dfg_only_rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c"))


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


def assert_app_cgra_pass_row(
    repo: Path,
    rows: list[dict[str, str]],
    case: str,
    *,
    expected_hardware: str,
) -> None:
    row = one_row(rows, "app", case)
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or row["hardware_system"] != expected_hardware
        or row["final_outputs_present"] != "true"
        or row["final_memory_state_present"] != "true"
    ):
        raise AssertionError(f"app row should expose real CGRA-sim evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)


def assert_cmsis_dfg_blocker_row(
    repo: Path,
    rows: list[dict[str, str]],
    suite: str,
    case: str,
    *,
    dfg_status: str,
    diagnostic_class: str,
    diagnostic_substring: str,
) -> None:
    row = one_row(rows, suite, case)
    if (
        row["status"] != "blocked"
        or row["diagnostic_class"] != diagnostic_class
        or row["blocking_prerequisite"] != "dfg_report"
        or row["owner"] != "sim_report"
        or row["dfg_status"] != dfg_status
        or row["mapping_status"] != "not_run"
        or row["cgra_status"] != "not_run"
        or diagnostic_substring not in row["diagnostic"]
    ):
        raise AssertionError(f"CMSIS row should expose exact DFG-sim blocker evidence: {row}")
    assert_sha256_file(row["dfg_report"], row["dfg_report_fingerprint"], repo)


def assert_cmsis_dfg_ready_for_mapping_row(
    repo: Path,
    rows: list[dict[str, str]],
    suite: str,
    case: str,
) -> None:
    row = one_row(rows, suite, case)
    if (
        row["status"] != "blocked"
        or row["diagnostic_class"] != "missing_mapping_artifact"
        or row["blocking_prerequisite"] != "mapping_artifact"
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "not_run"
        or row["cgra_status"] != "not_run"
        or row["comparison_status"] != "not_run"
        or row["mapping_artifact"]
        or row["cgra_report"]
        or row["comparison_report"]
        or "PnR mapping artifact is absent" not in row["diagnostic"]
    ):
        raise AssertionError(f"CMSIS row should expose DFG pass evidence and a missing mapping artifact: {row}")
    assert_sha256_file(row["dfg_report"], row["dfg_report_fingerprint"], repo)


def assert_cmsis_relu_q7_shared_adg_evidence(sim_evidence: Path) -> None:
    expected_graphs = [
        "g_t_arm_relu_q7_red_0_0",
        "g_t_arm_relu_q7_red_1_0",
    ]
    red1_memory = {"arg5": ["i8:0", "i8:2", "i8:0"]}
    aggregate_memory = {
        "g_t_arm_relu_q7_red_0_0:arg8": ["i32:0", "i32:2130706433"],
        "g_t_arm_relu_q7_red_1_0:arg5": ["i8:0", "i8:2", "i8:0"],
    }
    red1_dfg = json.loads((sim_evidence / "arm_relu_q7.red1.dfg.report.json").read_text())
    if (
        red1_dfg.get("kind") != "dfg_sim_report"
        or red1_dfg.get("workload") != "ActivationFunctions/arm_relu_q7.c"
        or red1_dfg.get("graph") != "g_t_arm_relu_q7_red_1_0"
        or red1_dfg.get("status") != "pass"
        or red1_dfg.get("optimistic_cycles") != 84
        or red1_dfg.get("dynamic_work_items") != 3
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.load") != 3
        or red1_dfg.get("operation_fire_counts", {}).get("arith.cmpi") != 3
        or red1_dfg.get("operation_fire_counts", {}).get("arith.select") != 3
        or red1_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 3
        or red1_dfg.get("final_outputs") != ["none"]
        or red1_dfg.get("final_memory_state") != red1_memory
    ):
        raise AssertionError(f"unexpected arm_relu_q7 residual DFG report: {red1_dfg}")

    red1_mapping = json.loads((sim_evidence / "arm_relu_q7.red1.mapping.json").read_text())
    expected_red1_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "g_t_arm_relu_q7_red_1_0",
        "placed_records": 13,
        "routed_edges": 18,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 401,
        "status": "pass",
    }
    for key, value in expected_red1_mapping.items():
        if red1_mapping.get(key) != value:
            raise AssertionError(f"arm_relu_q7 red1 mapping {key}={red1_mapping.get(key)!r}, expected {value!r}")
    routes = red1_mapping.get("routes", [])
    if len(routes) != 18:
        raise AssertionError(f"arm_relu_q7 red1 mapping should expose every routed edge: {red1_mapping}")
    routes_by_edge = {route.get("edge_ref"): route for route in routes if isinstance(route, dict)}
    required_routes = {
        "dataflow.invariant#0.result0->arith.select#0.operand1": (
            "shared_reduction_adg::fabric.op#3.result0",
            "shared_reduction_adg::fabric.op#40.operand1",
        ),
        "dataflow.constant#2.result0->dataflow.store#0.operand1": (
            "shared_reduction_adg::fabric.op#21.result0",
            "shared_reduction_adg::mem.store#0.operand0",
        ),
        "arith.select#0.result0->dataflow.store#0.operand2": (
            "shared_reduction_adg::fabric.op#40.result0",
            "shared_reduction_adg::mem.store#0.operand1",
        ),
    }
    for edge_ref, (source_endpoint, sink_endpoint) in required_routes.items():
        route = routes_by_edge.get(edge_ref)
        if route is None:
            raise AssertionError(f"arm_relu_q7 red1 mapping missed route {edge_ref}: {red1_mapping}")
        segments = route.get("segments", [])
        if (
            not segments
            or not isinstance(segments[0], dict)
            or not isinstance(segments[-1], dict)
            or segments[0].get("source_endpoint") != source_endpoint
            or segments[-1].get("sink_endpoint") != sink_endpoint
        ):
            raise AssertionError(f"arm_relu_q7 red1 route endpoints changed for {edge_ref}: {route}")
        if not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in segments):
            raise AssertionError(f"arm_relu_q7 red1 route should traverse Fabric paths: {route}")
    sync_route = routes_by_edge.get("dataflow.store#0.result0->dataflow.sync#0.operand1")
    if sync_route is None:
        raise AssertionError(f"arm_relu_q7 red1 mapping missed store-to-sync route: {red1_mapping}")
    sync_segments = sync_route.get("segments", [])
    if (
        not sync_segments
        or not isinstance(sync_segments[0], dict)
        or sync_segments[0].get("source_endpoint") != "shared_reduction_adg::mem.store#0.result0"
        or not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in sync_segments)
    ):
        raise AssertionError(f"arm_relu_q7 red1 store-to-sync route should leave the store through Fabric paths: {sync_route}")

    aggregate_dfg = json.loads((sim_evidence / "arm_relu_q7.dfg.report.json").read_text())
    if (
        aggregate_dfg.get("kind") != "dfg_sim_report"
        or aggregate_dfg.get("workload") != "ActivationFunctions/arm_relu_q7.c"
        or aggregate_dfg.get("graph") != "workload_graph_set"
        or aggregate_dfg.get("aggregation_kind") != "workload_graph_set"
        or aggregate_dfg.get("component_graphs") != expected_graphs
        or aggregate_dfg.get("status") != "pass"
        or aggregate_dfg.get("optimistic_cycles") != 153
        or aggregate_dfg.get("dynamic_work_items") != 5
        or aggregate_dfg.get("operation_fire_counts", {}).get("dataflow.load") != 5
        or aggregate_dfg.get("operation_fire_counts", {}).get("dataflow.store") != 5
        or aggregate_dfg.get("final_outputs") != ["none", "none"]
        or aggregate_dfg.get("final_memory_state") != aggregate_memory
    ):
        raise AssertionError(f"unexpected arm_relu_q7 aggregate DFG report: {aggregate_dfg}")

    aggregate_mapping = json.loads((sim_evidence / "arm_relu_q7.mapping.json").read_text())
    expected_aggregate_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "workload_graph_set",
        "aggregation_kind": "workload_graph_set",
        "placed_records": 31,
        "routed_edges": 44,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 981,
        "route_segments": 178,
        "status": "pass",
    }
    for key, value in expected_aggregate_mapping.items():
        if aggregate_mapping.get(key) != value:
            raise AssertionError(f"arm_relu_q7 aggregate mapping {key}={aggregate_mapping.get(key)!r}, expected {value!r}")
    if aggregate_mapping.get("component_graphs") != expected_graphs or len(aggregate_mapping.get("routes", [])) != 44:
        raise AssertionError(f"arm_relu_q7 aggregate mapping should preserve component graph routes: {aggregate_mapping}")

    aggregate_cgra = json.loads((sim_evidence / "arm_relu_q7.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "graph": "workload_graph_set",
        "aggregation_kind": "workload_graph_set",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "dfg_cycles": 153,
        "hardware_aware_cycles": 347,
        "performance_delta_cycles": 194,
        "route_segments": 178,
        "config_records": 981,
        "functional_state_source": "component_cgra_sim_reports_carried_from_dfg_sim_reports",
    }
    for key, value in expected_cgra.items():
        if aggregate_cgra.get(key) != value:
            raise AssertionError(f"arm_relu_q7 CGRA report {key}={aggregate_cgra.get(key)!r}, expected {value!r}")
    if (
        aggregate_cgra.get("component_graphs") != expected_graphs
        or aggregate_cgra.get("final_outputs") != ["none", "none"]
        or aggregate_cgra.get("final_memory_state") != aggregate_memory
    ):
        raise AssertionError(f"arm_relu_q7 CGRA report should carry aggregate final state: {aggregate_cgra}")


def assert_cmsis_relu_q7_pass_row(
    repo: Path,
    rows: list[dict[str, str]],
    sim_evidence: Path,
) -> None:
    row = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu_q7.c")
    expected_graphs = {
        "g_t_arm_relu_q7_red_0_0",
        "g_t_arm_relu_q7_red_1_0",
    }
    if set(row["graph_ids"].split(",")) != expected_graphs:
        raise AssertionError(f"arm_relu_q7 row should keep both component graph ids: {row}")
    if (
        row["status"] != "pass"
        or row["diagnostic_class"] != "cgra_sim_pass"
        or row["blocking_prerequisite"] != ""
        or row["owner"] != "sim_report"
        or row["dfg_status"] != "pass"
        or row["mapping_status"] != "pass"
        or row["cgra_status"] != "pass"
        or row["comparison_status"] != "pass"
        or Path(row["dfg_report"]).name != "arm_relu_q7.dfg.report.json"
        or Path(row["mapping_artifact"]).name != "arm_relu_q7.mapping.json"
        or Path(row["cgra_report"]).name != "arm_relu_q7.cgra.report.json"
        or Path(row["comparison_report"]).name != "arm_relu_q7.c.sim-comparison-report.json"
    ):
        raise AssertionError(f"arm_relu_q7 should expose aggregate CGRA pass evidence: {row}")
    for key in ("dfg_report", "mapping_artifact", "cgra_report", "comparison_report"):
        assert_sha256_file(row[key], row[f"{key}_fingerprint"], repo)
    for artifact_name in (
        "arm_relu_q7.red0.dfg.report.json",
        "arm_relu_q7.red0.mapping.json",
        "arm_relu_q7.red0.cgra.report.json",
        "arm_relu_q7.red1.dfg.report.json",
        "arm_relu_q7.red1.mapping.json",
        "arm_relu_q7.red1.cgra.report.json",
        "arm_relu_q7.dfg.report.json",
        "arm_relu_q7.mapping.json",
        "arm_relu_q7.cgra.report.json",
    ):
        artifact = sim_evidence / artifact_name
        if not artifact.is_file():
            raise AssertionError(f"arm_relu_q7 component evidence should emit {artifact}")
    assert_cmsis_relu_q7_shared_adg_evidence(sim_evidence)


def assert_cgra_status_audit_rejects_bad_relu_q7_mapping(
    repo: Path,
    out_dir: Path,
    legacy_root: Path,
) -> None:
    mapping = out_dir / "cmsis-sim-evidence" / "arm_relu_q7.mapping.json"
    original = mapping.read_text()
    data = json.loads(original)
    data["status"] = "fail"
    try:
        mapping.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        result = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(out_dir / "cgra-status-summary.csv"),
                "--json-input",
                str(out_dir / "cgra-status-summary.json"),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
    finally:
        mapping.write_text(original)
    combined = result.stdout + result.stderr
    if "referenced mapping_artifact JSON status is not pass" not in combined:
        raise AssertionError(f"CGRA status audit should reject stale arm_relu_q7 mapping evidence: {combined}")

    try:
        mapping.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        result = run(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "generic-audit-bad-component.json"),
                str(out_dir / "cgra-status-summary.csv"),
            ],
            expect_success=False,
        )
    finally:
        mapping.write_text(original)
    combined = result.stdout + result.stderr
    audit_data = json.loads((out_dir / "generic-audit-bad-component.json").read_text())
    audit_diagnostics = "\n".join(str(item) for item in audit_data.get("diagnostics", []))
    if "referenced mapping_artifact JSON status is not pass" not in combined + audit_diagnostics:
        raise AssertionError(f"generic audit should reject stale arm_relu_q7 mapping evidence: {combined} {audit_data}")


def assert_cgra_status_audit_rejects_bad_aggregate_graphs(
    repo: Path,
    out_dir: Path,
    legacy_root: Path,
) -> None:
    for report_name in ("arm_var_f32.dfg.report.json", "arm_var_f32.cgra.report.json"):
        report = out_dir / "cmsis-sim-evidence" / report_name
        original = report.read_text()
        data = json.loads(original)
        data["component_graphs"] = ["g_t_arm_var_f32_red_0_0"]
        try:
            report.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            result = run(
                repo,
                [
                    "bash",
                    "test/e2e/run_cgra_status_audit.sh",
                    "--input",
                    str(out_dir / "cgra-status-summary.csv"),
                    "--json-input",
                    str(out_dir / "cgra-status-summary.json"),
                    "--legacy-loombench-root",
                    str(legacy_root),
                ],
                expect_success=False,
            )
        finally:
            report.write_text(original)
        combined = result.stdout + result.stderr
        if "aggregate component_graphs do not exactly match row graph_ids" not in combined:
            raise AssertionError(f"CGRA status audit should reject stale aggregate graph coverage: {combined}")

        data = json.loads(original)
        data.pop("aggregation_kind", None)
        try:
            report.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            result = run(
                repo,
                [
                    "bash",
                    "test/e2e/run_cgra_status_audit.sh",
                    "--input",
                    str(out_dir / "cgra-status-summary.csv"),
                    "--json-input",
                    str(out_dir / "cgra-status-summary.json"),
                    "--legacy-loombench-root",
                    str(legacy_root),
                ],
                expect_success=False,
            )
        finally:
            report.write_text(original)
        combined = result.stdout + result.stderr
        if "aggregate lacks workload_graph_set aggregation_kind" not in combined:
            raise AssertionError(f"CGRA status audit should reject pseudo aggregate evidence: {combined}")


def assert_generic_artifact_audit_rejects_bad_aggregate_graphs(repo: Path, out_dir: Path) -> None:
    report = out_dir / "cmsis-sim-evidence" / "arm_var_f32.cgra.report.json"
    original = report.read_text()
    data = json.loads(original)
    data["component_graphs"] = ["g_t_arm_var_f32_red_0_0"]
    try:
        report.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        result = run(
            repo,
            [
                sys.executable,
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "generic-audit-bad-aggregate.json"),
                str(out_dir / "cgra-status-summary.csv"),
            ],
            expect_success=False,
        )
    finally:
        report.write_text(original)
    combined = result.stdout + result.stderr
    audit_data = json.loads((out_dir / "generic-audit-bad-aggregate.json").read_text())
    audit_diagnostics = "\n".join(str(item) for item in audit_data.get("diagnostics", []))
    if "aggregate component_graphs do not exactly match row graph_ids" not in combined + audit_diagnostics:
        raise AssertionError(f"generic audit should reject stale aggregate graph coverage: {combined} {audit_data}")


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


def assert_cmsis_offset_shared_adg_evidence(sim_evidence: Path) -> None:
    mapping_artifact = json.loads((sim_evidence / "arm_offset_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "placed_records": 11,
        "routed_edges": 15,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_offset_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 15:
        raise AssertionError(f"arm_offset_f32 mapping should expose every routed edge: {mapping_artifact}")
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    required_edges = {
        "dataflow.carry#0.result0->arith.addi#0.operand0",
        "arith.addi#0.result0->dataflow.carry#0.operand2",
        "dataflow.carry#0.result0->dataflow.load#0.operand1",
        "dataflow.carry#0.result0->dataflow.store#0.operand1",
    }
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_offset_f32 mapping missed explicit index-carry route evidence: {mapping_artifact}")
    if any("llvm.getelementptr" in str(edge) for edge in actual_edges):
        raise AssertionError(f"arm_offset_f32 mapping must not hide GEP as a routed edge: {mapping_artifact}")
    config_records = mapping_artifact.get("config_records")
    if not isinstance(config_records, int) or config_records <= 0:
        raise AssertionError(f"arm_offset_f32 mapping should emit non-empty configuration evidence: {mapping_artifact}")

    cgra_report = json.loads((sim_evidence / "arm_offset_f32.cgra.report.json").read_text())
    if cgra_report.get("config_records") != config_records or cgra_report.get("route_segments", 0) <= 0:
        raise AssertionError(f"arm_offset_f32 CGRA report should consume mapping evidence: {cgra_report}")


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


def assert_cmsis_mean_shared_adg_evidence(sim_evidence: Path) -> None:
    mapping_artifact = json.loads((sim_evidence / "arm_mean_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "placed_records": 10,
        "routed_edges": 13,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 296,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_mean_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 13:
        raise AssertionError(f"arm_mean_f32 mapping should expose every routed edge: {mapping_artifact}")
    required_edges = {
        "arith.addi#0.result0->dataflow.carry#1.operand2",
        "dataflow.carry#1.result0->arith.addi#0.operand0",
        "dataflow.carry#1.result0->dataflow.load#0.operand1",
        "dataflow.constant#0.result0->dataflow.carry#1.operand1",
    }
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_mean_f32 mapping missed index-carry route evidence: {mapping_artifact}")
    expected_endpoints = {
        "arith.addi#0.result0->dataflow.carry#1.operand2": (
            "shared_reduction_adg::fabric.op#2.result0",
            "shared_reduction_adg::fabric.op#15.operand2",
        ),
        "dataflow.carry#1.result0->arith.addi#0.operand0": (
            "shared_reduction_adg::fabric.op#15.result0",
            "shared_reduction_adg::fabric.op#2.operand0",
        ),
        "dataflow.carry#1.result0->dataflow.load#0.operand1": (
            "shared_reduction_adg::fabric.op#15.result0",
            "shared_reduction_adg::mem.load#0.operand0",
        ),
        "dataflow.constant#0.result0->dataflow.carry#1.operand1": (
            "shared_reduction_adg::fabric.op#19.result0",
            "shared_reduction_adg::fabric.op#15.operand1",
        ),
    }
    routes_by_edge = {route.get("edge_ref"): route for route in routes if isinstance(route, dict)}
    for edge_ref, (source_endpoint, sink_endpoint) in expected_endpoints.items():
        route = routes_by_edge.get(edge_ref)
        if route is None:
            raise AssertionError(f"arm_mean_f32 mapping missed route {edge_ref}: {mapping_artifact}")
        segments = route.get("segments", [])
        if (
            not segments
            or not isinstance(segments[0], dict)
            or not isinstance(segments[-1], dict)
            or segments[0].get("source_endpoint") != source_endpoint
            or segments[-1].get("sink_endpoint") != sink_endpoint
        ):
            raise AssertionError(f"arm_mean_f32 route endpoints changed for {edge_ref}: {route}")
        if not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in segments):
            raise AssertionError(f"arm_mean_f32 index-carry route should traverse Fabric paths: {route}")

    cgra_report = json.loads((sim_evidence / "arm_mean_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "hardware_aware_cycles": 129,
        "performance_delta_cycles": 57,
        "route_segments": 53,
        "config_records": 296,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_mean_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")
    if cgra_report.get("final_outputs") != ["none", "f32:3.750000"]:
        raise AssertionError(f"arm_mean_f32 CGRA report should carry final reduction output: {cgra_report}")


def assert_cmsis_max_shared_adg_evidence(sim_evidence: Path) -> None:
    dfg_report = json.loads((sim_evidence / "arm_max_f32.dfg.report.json").read_text())
    expected_outputs = ["none", "i32:3", "f32:4.250000"]
    expected_memory = {
        "arg7": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
    }
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != "StatisticsFunctions/arm_max_f32.c"
        or dfg_report.get("graph") != "g_t_arm_max_f32_red_0_0"
        or dfg_report.get("status") != "pass"
        or dfg_report.get("optimistic_cycles") != 90
        or dfg_report.get("dynamic_work_items") != 3
        or dfg_report.get("operation_fire_counts", {}).get("dataflow.load") != 3
        or dfg_report.get("operation_fire_counts", {}).get("arith.cmpf") != 3
        or dfg_report.get("final_outputs") != expected_outputs
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected arm_max_f32 DFG report: {dfg_report}")

    mapping_artifact = json.loads((sim_evidence / "arm_max_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "g_t_arm_max_f32_red_0_0",
        "placed_records": 18,
        "routed_edges": 28,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 653,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_max_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 28:
        raise AssertionError(f"arm_max_f32 mapping should expose every routed edge: {mapping_artifact}")
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    required_edges = {
        "dataflow.invariant#2.result0->arith.addi#1.operand1",
        "arith.addi#1.result0->dataflow.load#0.operand1",
        "dataflow.load#0.result0->arith.cmpf#0.operand1",
        "arith.cmpf#0.result0->arith.select#0.operand0",
        "arith.select#0.result0->dataflow.carry#1.operand2",
    }
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_max_f32 mapping missed max-reduction route evidence: {mapping_artifact}")
    for route in routes:
        if route.get("edge_ref") not in required_edges:
            continue
        segments = route.get("segments", [])
        if not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in segments):
            raise AssertionError(f"arm_max_f32 route should traverse Fabric paths: {route}")

    cgra_report = json.loads((sim_evidence / "arm_max_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "dfg_cycles": 90,
        "hardware_aware_cycles": 216,
        "performance_delta_cycles": 126,
        "route_segments": 122,
        "config_records": 653,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_max_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")
    if cgra_report.get("final_outputs") != expected_outputs or cgra_report.get("final_memory_state") != expected_memory:
        raise AssertionError(f"arm_max_f32 CGRA report should carry final max state: {cgra_report}")


def assert_cmsis_biquad_shared_adg_evidence(sim_evidence: Path) -> None:
    dfg_report = json.loads((sim_evidence / "arm_biquad_cascade_df1_f32.dfg.report.json").read_text())
    expected_outputs = ["none", "f32:0.321289", "f32:-0.385681", "f32:-3.500000", "f32:4.250000"]
    expected_memory = {
        "arg9": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
        "arg14": ["f32:0.250000", "f32:1.015625", "f32:0.321289", "f32:-0.385681"],
    }
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != "FilteringFunctions/arm_biquad_cascade_df1_f32.c"
        or dfg_report.get("graph") != "g_t_arm_biquad_cascade_df1_f32_red_0_0"
        or dfg_report.get("status") != "pass"
        or dfg_report.get("optimistic_cycles") != 254
        or dfg_report.get("dynamic_work_items") != 4
        or dfg_report.get("final_outputs") != expected_outputs
        or dfg_report.get("final_memory_state") != expected_memory
    ):
        raise AssertionError(f"unexpected arm_biquad_cascade_df1_f32 DFG report: {dfg_report}")

    mapping_artifact = json.loads((sim_evidence / "arm_biquad_cascade_df1_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "g_t_arm_biquad_cascade_df1_f32_red_0_0",
        "placed_records": 23,
        "routed_edges": 39,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 890,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_biquad_cascade_df1_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    routes = mapping_artifact.get("routes", [])
    if len(routes) != 39:
        raise AssertionError(f"arm_biquad_cascade_df1_f32 mapping should expose every routed edge: {mapping_artifact}")
    actual_edges = {route.get("edge_ref") for route in routes if isinstance(route, dict)}
    required_edges = {
        "dataflow.carry#4.result0->dataflow.load#0.operand1",
        "dataflow.carry#4.result0->dataflow.store#0.operand1",
        "dataflow.invariant#4.result0->arith.mulf#0.operand0",
        "dataflow.carry#3.result0->arith.mulf#0.operand1",
        "arith.mulf#0.result0->llvm.intr.fmuladd#0.operand2",
        "llvm.intr.fmuladd#0.result0->llvm.intr.fmuladd#1.operand2",
        "llvm.intr.fmuladd#1.result0->llvm.intr.fmuladd#2.operand2",
        "llvm.intr.fmuladd#2.result0->llvm.intr.fmuladd#3.operand2",
        "llvm.intr.fmuladd#3.result0->dataflow.store#0.operand2",
    }
    if not required_edges.issubset(actual_edges):
        raise AssertionError(f"arm_biquad_cascade_df1_f32 mapping missed biquad datapath routes: {mapping_artifact}")
    for route in routes:
        if route.get("edge_ref") not in required_edges:
            continue
        segments = route.get("segments", [])
        if not any(isinstance(segment, dict) and segment.get("segment_kind") == "module_path" for segment in segments):
            raise AssertionError(f"arm_biquad_cascade_df1_f32 route should traverse Fabric paths: {route}")

    cgra_report = json.loads((sim_evidence / "arm_biquad_cascade_df1_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "status": "pass",
        "fidelity_level": "mapping_constraint_estimate",
        "dfg_cycles": 254,
        "hardware_aware_cycles": 431,
        "performance_delta_cycles": 177,
        "route_segments": 169,
        "config_records": 890,
        "functional_state_source": "carried_from_dfg_sim_report",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_biquad_cascade_df1_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")
    if cgra_report.get("final_outputs") != expected_outputs or cgra_report.get("final_memory_state") != expected_memory:
        raise AssertionError(f"arm_biquad_cascade_df1_f32 CGRA report should carry final state: {cgra_report}")


def assert_cmsis_var_shared_adg_evidence(sim_evidence: Path) -> None:
    dfg_report = json.loads((sim_evidence / "arm_var_f32.dfg.report.json").read_text())
    expected_graphs = ["g_t_arm_var_f32_red_0_0", "g_t_arm_var_f32_red_1_0"]
    expected_outputs = ["none", "f32:3.750000", "none", "f32:31.796875"]
    if (
        dfg_report.get("kind") != "dfg_sim_report"
        or dfg_report.get("workload") != "StatisticsFunctions/arm_var_f32.c"
        or dfg_report.get("graph") != "workload_graph_set"
        or dfg_report.get("aggregation_kind") != "workload_graph_set"
        or dfg_report.get("component_graphs") != expected_graphs
        or dfg_report.get("status") != "pass"
        or dfg_report.get("optimistic_cycles") != 178
        or dfg_report.get("dynamic_work_items") != 8
        or dfg_report.get("final_outputs") != expected_outputs
    ):
        raise AssertionError(f"unexpected arm_var_f32 aggregate DFG report: {dfg_report}")
    expected_memory = {
        "g_t_arm_var_f32_red_0_0:arg4": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
        "g_t_arm_var_f32_red_1_0:arg5": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
    }
    if dfg_report.get("final_memory_state") != expected_memory:
        raise AssertionError(f"unexpected arm_var_f32 aggregate DFG memory: {dfg_report}")

    mapping_artifact = json.loads((sim_evidence / "arm_var_f32.mapping.json").read_text())
    expected_mapping = {
        "hardware": "shared_reduction_adg",
        "graph": "workload_graph_set",
        "aggregation_kind": "workload_graph_set",
        "placed_records": 22,
        "routed_edges": 30,
        "unrouted_edges": 0,
        "unplaced_records": 0,
        "config_records": 684,
        "route_segments": 124,
        "status": "pass",
    }
    for key, value in expected_mapping.items():
        if mapping_artifact.get(key) != value:
            raise AssertionError(f"arm_var_f32 mapping {key}={mapping_artifact.get(key)!r}, expected {value!r}")
    if mapping_artifact.get("component_graphs") != expected_graphs:
        raise AssertionError(f"arm_var_f32 mapping should preserve component graph coverage: {mapping_artifact}")
    if len(mapping_artifact.get("routes", [])) != 30:
        raise AssertionError(f"arm_var_f32 mapping should expose every component route: {mapping_artifact}")

    cgra_report = json.loads((sim_evidence / "arm_var_f32.cgra.report.json").read_text())
    expected_cgra = {
        "hardware": "shared_reduction_adg",
        "graph": "workload_graph_set",
        "status": "pass",
        "aggregation_kind": "workload_graph_set",
        "fidelity_level": "mapping_constraint_estimate",
        "dfg_cycles": 178,
        "hardware_aware_cycles": 310,
        "performance_delta_cycles": 132,
        "route_segments": 124,
        "config_records": 684,
        "functional_state_source": "component_cgra_sim_reports_carried_from_dfg_sim_reports",
    }
    for key, value in expected_cgra.items():
        if cgra_report.get(key) != value:
            raise AssertionError(f"arm_var_f32 CGRA report {key}={cgra_report.get(key)!r}, expected {value!r}")
    if cgra_report.get("component_graphs") != expected_graphs:
        raise AssertionError(f"arm_var_f32 CGRA report should preserve component graph coverage: {cgra_report}")
    if cgra_report.get("final_outputs") != expected_outputs or cgra_report.get("final_memory_state") != expected_memory:
        raise AssertionError(f"arm_var_f32 CGRA report should carry aggregate final state: {cgra_report}")


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
        "arm_mean_f32",
        "StatisticsFunctions/arm_mean_f32.c",
        "arith.addf",
        {
            "arg5": ["f32:1", "f32:2", "f32:-3.500000", "f32:4.250000"],
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
            "pass": 11,
            "fail": 0,
            "blocked": 3,
            "unsupported": 2,
            "missing_status": 0,
        },
    )
    assert_counts(
        data,
        "cmsis-nn",
        {
            "total": 18,
            "pass": 2,
            "fail": 0,
            "blocked": 8,
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
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "StatisticsFunctions/arm_mean_f32.c", "arm_mean_f32"
    )
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "StatisticsFunctions/arm_var_f32.c", "arm_var_f32"
    )
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-dsp", "StatisticsFunctions/arm_max_f32.c", "arm_max_f32"
    )
    assert_cmsis_cgra_pass_row(
        repo,
        rows,
        sim_evidence,
        "cmsis-dsp",
        "FilteringFunctions/arm_biquad_cascade_df1_f32.c",
        "arm_biquad_cascade_df1_f32",
    )
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "SupportFunctions/arm_copy_f32.c", "arm_copy_f32")
    assert_cmsis_cgra_pass_row(repo, rows, sim_evidence, "cmsis-dsp", "SupportFunctions/arm_fill_f32.c", "arm_fill_f32")
    assert_cmsis_add_q15_shared_adg_evidence(sim_evidence)
    assert_cmsis_offset_shared_adg_evidence(sim_evidence)
    assert_cmsis_fill_shared_adg_evidence(sim_evidence)
    assert_cmsis_mean_shared_adg_evidence(sim_evidence)
    assert_cmsis_max_shared_adg_evidence(sim_evidence)
    assert_cmsis_biquad_shared_adg_evidence(sim_evidence)
    assert_cmsis_var_shared_adg_evidence(sim_evidence)
    assert_cgra_status_audit_rejects_bad_aggregate_graphs(repo, out_dir, legacy_root)
    assert_generic_artifact_audit_rejects_bad_aggregate_graphs(repo, out_dir)
    assert_cgra_status_audit_rejects_bad_relu_q7_mapping(repo, out_dir, legacy_root)
    assert_cmsis_cgra_pass_row(
        repo, rows, sim_evidence, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c", "arm_relu_q15"
    )
    assert_cmsis_relu_q7_pass_row(repo, rows, sim_evidence)
    assert_cmsis_dfg_blocker_row(
        repo,
        rows,
        "cmsis-nn",
        "FullyConnectedFunctions/arm_vector_sum_s8.c",
        dfg_status="unsupported",
        diagnostic_class="dfg_report_unsupported",
        diagnostic_substring="unsupported op: scf.for",
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


def assert_app_default_batch_manifest_fail_fast(repo: Path, out_dir: Path, legacy_root: Path) -> None:
    out_dir.mkdir(parents=True)
    invalid_manifest = out_dir / "invalid-default-cgra-sim-batch.json"
    invalid_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"case": "vecsum", "hardware": "not_a_real_shared_adg"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    result = run(
        repo,
        [
            "bash",
            "test/e2e/run_cmsis_cgra_status_rollup.sh",
            "--output-dir",
            str(out_dir / "invalid-default-batch"),
            "--legacy-loombench-root",
            str(legacy_root),
            "--app-sim-default-batch",
        ],
        env={**os.environ, "LOOM_DEFAULT_CGRA_SIM_BATCH": str(invalid_manifest)},
        expect_success=False,
    )
    if "unsupported hardware" not in result.stderr:
        raise AssertionError(f"default batch manifest failure should be reported by the shell wrapper: {result.stderr}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "cmsis-cgra-status-rollup-") as raw_out_dir:
        out_dir = Path(raw_out_dir)
        assert_no_legacy_mode(repo, out_dir / "no-legacy")
        legacy_root = out_dir / "legacy-loombench"
        write_legacy_case(legacy_root, "legacy_missing")
        write_legacy_case(legacy_root, "vecadd")
        write_legacy_case(legacy_root, "blocked_case", with_header=False)
        assert_app_default_batch_manifest_fail_fast(repo, out_dir / "manifest-fail-fast", legacy_root)
        assert_app_attempt_manifest_mode(repo, out_dir / "app-attempt-manifest", legacy_root)
        assert_direct_cmsis_dfg_mode(repo, out_dir / "direct-cmsis-dfg", legacy_root)
        assert_app_cgra_sweep_mode(repo, out_dir / "app-cgra-sweep", legacy_root)
        assert_cmsis_sim_default_mode(repo, out_dir / "cmsis-sim-default", legacy_root)
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
        assert_cmsis_dfg_only_counts(data)
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
            or loombench_vecadd["diagnostic_class"] != "loombench_workload_identity_bridge_ready"
            or loombench_vecadd["blocking_prerequisite"] != "sim_evidence"
            or loombench_vecadd["manifest_case"] != "vecadd"
        ):
            raise AssertionError(f"LoomBench accepted row should expose explicit evidence bridge: {loombench_vecadd}")
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
