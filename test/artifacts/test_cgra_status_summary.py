#!/usr/bin/env python3
"""Regression test for row-complete CGRA status evidence."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import artifact_test_common


HEADER = [
    "suite",
    "case",
    "source_row",
    "manifest_case",
    "software_root",
    "graph_ids",
    "dfg_mlir",
    "dfg_mlir_fingerprint",
    "required_slice_count",
    "hardware_system",
    "spatialcore_template",
    "mapping_id",
    "dfg_report",
    "dfg_report_fingerprint",
    "dfg_status",
    "mapping_artifact",
    "mapping_artifact_fingerprint",
    "mapping_status",
    "cgra_report",
    "cgra_report_fingerprint",
    "cgra_status",
    "comparison_report",
    "comparison_report_fingerprint",
    "comparison_status",
    "final_outputs_present",
    "final_memory_state_present",
    "status",
    "diagnostic_class",
    "owner",
    "blocking_prerequisite",
    "diagnostic",
]
LEGACY_CASE_COUNT = 127
APP_CASE_COUNT = 115
APP_NO_DFG_TIER_COUNT = 0
REQUIRED_LEGACY_CASE = "breadth_first_search"
CURRENT_SIM_CYCLE_CASES = [
    "axpy",
    "bit_reverse",
    "byte_swap",
    "compare_swap",
    "convolve_1d",
    "correlation",
    "covariance",
    "conv1d",
    "cumsum",
    "dotproduct",
    "dotprod",
    "downsample_avg",
    "gemv",
    "hash_mix",
    "integrate_trapz",
    "mean",
    "matvec",
    "prefix_sum",
    "prefix_sum_inclusive",
    "reduction",
    "relu",
    "rotate_bits",
    "spmv",
    "variance",
    "vecadd",
    "vecmul",
    "vecnorm_l1",
    "vecnorm_l2",
    "vecscale",
    "vecsum",
    "xor_block",
]


def run(repo: Path, argv: list[str], *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
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
        raise AssertionError(f"command unexpectedly passed: {' '.join(argv)}")
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames != HEADER:
            raise AssertionError(f"unexpected header: {reader.fieldnames}")
        return rows


def one_row(rows: list[dict[str, str]], suite: str, case: str) -> dict[str, str]:
    matches = [row for row in rows if row["suite"] == suite and row["case"] == case]
    if len(matches) != 1:
        raise AssertionError(f"expected one {suite}/{case} row, got {matches}")
    return matches[0]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def suite_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        suite = row["suite"]
        suite_counts = counts.setdefault(
            suite,
            {
                "total": 0,
                "pass": 0,
                "fail": 0,
                "blocked": 0,
                "unsupported": 0,
                "missing_status": 0,
            },
        )
        suite_counts["total"] += 1
        status = row["status"]
        if status in ("pass", "fail", "blocked", "unsupported"):
            suite_counts[status] += 1
        if row.get("diagnostic_class") == "missing_status":
            suite_counts["missing_status"] += 1
    return counts


def write_json_projection(path: Path, csv_output: Path, rows: list[dict[str, str]]) -> None:
    data = {
        "schema_version": 1,
        "kind": "cgra_status_summary",
        "csv_projection": str(csv_output),
        "counts": suite_counts(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_legacy_fixture(root: Path) -> None:
    names = [REQUIRED_LEGACY_CASE]
    names.extend(f"legacy_case_{index:03d}" for index in range(LEGACY_CASE_COUNT - 1))
    for name in names:
        (root / name).mkdir(parents=True)


def write_ready_legacy_case(root: Path, case: str) -> None:
    case_root = root / case
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "main.cpp").write_text("int main() { return 0; }\n")
    (case_root / f"{case}.cpp").write_text(f"#include \"{case}.h\"\n")
    (case_root / f"{case}.h").write_text("#pragma once\n")


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def dfg_cycle_fixture_fields(
    cycles: int,
    *,
    operation_mix_cycles: int = 0,
    memory_address_setup_cycles: int = 0,
    evidence: str = "fixture DFG report",
) -> dict[str, object]:
    pipeline_cycles = cycles - operation_mix_cycles - memory_address_setup_cycles
    if pipeline_cycles < 0:
        raise AssertionError("fixture DFG cycle components exceed optimistic_cycles")
    return {
        "pipeline_latency_throughput_cycles": pipeline_cycles,
        "operation_mix_cycles": operation_mix_cycles,
        "memory_address_setup_cycles": memory_address_setup_cycles,
        "cycle_breakdown": [
            {
                "category": "pipeline_latency_throughput",
                "cycles": pipeline_cycles,
                "evidence": evidence,
                "modeled": True,
            },
            {
                "category": "operation_mix",
                "cycles": operation_mix_cycles,
                "evidence": evidence,
                "modeled": True,
            },
            {
                "category": "memory_address_setup",
                "cycles": memory_address_setup_cycles,
                "evidence": evidence,
                "modeled": True,
            },
        ],
    }


def write_loombench_manifest(path: Path, cases: list[dict[str, object]]) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "kind": "loombench_manifest",
            "csv_projection": "",
            "case_count": len(cases),
            "cases": cases,
        },
    )


def write_sim_evidence_case(
    evidence_dir: Path,
    case: str,
    *,
    cgra_final_state: bool,
    workload_identity: str | None = None,
    functional_state_source: str = "carried_from_dfg_sim_report",
) -> None:
    workload = workload_identity or case
    graph = f"g_{workload}_0"
    mapping_id = f"{workload}__shared_reduction_adg"
    final_outputs = ["i32:7"]
    final_memory_state = {"arg0": ["i32:7"]}
    write_json(
        evidence_dir / f"{case}.dfg.report.json",
        {
            "schema_version": 1,
            "kind": "dfg_sim_report",
            "workload": workload,
            "graph": graph,
            "status": "pass",
            "optimistic_cycles": 10,
            **dfg_cycle_fixture_fields(10),
            "final_outputs": final_outputs,
            "final_memory_state": final_memory_state,
            "metric_definition": "fixture",
        },
    )
    write_json(
        evidence_dir / f"{case}.mapping.json",
        {
            "schema_version": 1,
            "kind": "pnr_mapping",
            "workload": workload,
            "graph": graph,
            "hardware": "shared_reduction_adg",
            "mapping_id": mapping_id,
            "status": "pass",
            "placed_records": 1,
            "routed_edges": 1,
            "unrouted_edges": 0,
            "unplaced_records": 0,
            "config_records": 0,
            "placements": [],
            "routes": [],
            "config_bitstream": [],
            "diagnostics": [],
        },
    )
    cgra_report = {
        "schema_version": 1,
        "kind": "cgra_sim_report",
        "workload": workload,
        "hardware": "shared_reduction_adg",
        "hardware_artifact": "test/pnr/shared_reduction_adg.mlir",
        "mapping_id": mapping_id,
        "status": "pass",
        "dfg_cycles": 10,
        "hardware_aware_cycles": 12,
        "performance_delta_cycles": 2,
        "difference_classification": "expected_hardware_constraint",
        "metric_definition": "fixture",
        "cycle_breakdown": [],
        "diagnostics": [],
    }
    if cgra_final_state:
        cgra_report["final_outputs"] = final_outputs
        cgra_report["final_memory_state"] = final_memory_state
        cgra_report["functional_state_source"] = functional_state_source
    write_json(evidence_dir / f"{case}.cgra.report.json", cgra_report)


def write_chain_style_sim_evidence_case(evidence_dir: Path, case: str) -> Path:
    chain_dir = evidence_dir / "_chains" / case
    mapping_id = f"{case}__shared_reduction_adg"
    final_outputs = ["i32:7"]
    final_memory_state = {"arg0": ["i32:7"]}
    write_json(
        chain_dir / f"{case}-dfg-sim-report.json",
        {
            "schema_version": 1,
            "kind": "dfg_sim_report",
            "workload": case,
            "graph": f"g_{case}_0",
            "status": "pass",
            "optimistic_cycles": 10,
            **dfg_cycle_fixture_fields(10),
            "final_outputs": final_outputs,
            "final_memory_state": final_memory_state,
            "metric_definition": "fixture",
        },
    )
    write_json(
        chain_dir / "pnr-mapping.json",
        {
            "schema_version": 1,
            "kind": "pnr_mapping",
            "workload": case,
            "graph": f"g_{case}_0",
            "hardware": "shared_reduction_adg",
            "mapping_id": mapping_id,
            "status": "pass",
            "placed_records": 1,
            "routed_edges": 1,
            "unrouted_edges": 0,
            "unplaced_records": 0,
            "config_records": 0,
            "placements": [],
            "routes": [],
            "config_bitstream": [],
            "diagnostics": [],
        },
    )
    write_json(
        chain_dir / f"{case}-cgra-sim-report.json",
        {
            "schema_version": 1,
            "kind": "cgra_sim_report",
            "workload": case,
            "hardware": "shared_reduction_adg",
            "hardware_artifact": "test/pnr/shared_reduction_adg.mlir",
            "mapping_id": mapping_id,
            "status": "pass",
            "dfg_cycles": 10,
            "hardware_aware_cycles": 12,
            "performance_delta_cycles": 2,
            "difference_classification": "expected_hardware_constraint",
            "metric_definition": "fixture",
            "final_outputs": final_outputs,
            "final_memory_state": final_memory_state,
            "functional_state_source": "carried_from_dfg_sim_report",
            "cycle_breakdown": [],
            "diagnostics": [],
        },
    )
    return chain_dir


def write_cmsis_dfg_mlir(path: Path, *, symbol: str, graph: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if graph:
        body = f"""
module {{
  func.func @{symbol}() {{
    return
  }}
  dataflow.thread private @t_{symbol}_0() ctrl (%arg0: none) iv (%arg1: index) {{
    %done = dataflow.graph.launch @g_{symbol}_0(%arg0) : (none) -> none
    dataflow.thread.yield
  }}
  dataflow.graph.func private @g_{symbol}_0(%arg0: none) -> none {{
    dataflow.graph.return %arg0 : none
  }}
}}
"""
    else:
        body = f"""
module {{
  func.func @{symbol}() {{
    return
  }}
}}
"""
    path.write_text(body.strip() + "\n")


def import_cgra_status_summary(repo: Path):
    module_path = repo / "test" / "e2e" / "cgra_status_summary.py"
    spec = importlib.util.spec_from_file_location("cgra_status_summary_under_test", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_component_only_evidence(evidence_dir: Path, case: str) -> None:
    write_json(
        evidence_dir / f"{case}.dfg.report.json",
        {
            "schema_version": 1,
            "kind": "dfg_sim_report",
            "workload": case,
            "graph": f"g_{case}_aggregate",
            "status": "pass",
            "optimistic_cycles": 11,
            **dfg_cycle_fixture_fields(11),
            "final_outputs": ["i32:11"],
            "final_memory_state": {"arg0": ["i32:11"]},
            "metric_definition": "fixture",
        },
    )
    write_json(
        evidence_dir / f"{case}.cgra.report.json",
        {
            "schema_version": 1,
            "kind": "cgra_sim_report",
            "workload": case,
            "hardware": "shared_reduction_adg",
            "mapping_id": f"{case}__aggregate__shared_reduction_adg",
            "status": "pass",
            "hardware_aware_cycles": 13,
            "dfg_cycles": 11,
        },
    )
    write_json(
        evidence_dir / f"{case}.core.mapping.json",
        {
            "schema_version": 1,
            "kind": "pnr_mapping",
            "workload": case,
            "graph": f"g_{case}_core",
            "hardware": "shared_reduction_adg",
            "mapping_id": f"{case}__core__shared_reduction_adg",
            "status": "pass",
        },
    )
    write_json(
        evidence_dir / f"{case}.core.cgra.report.json",
        {
            "schema_version": 1,
            "kind": "cgra_sim_report",
            "workload": case,
            "hardware": "shared_reduction_adg",
            "mapping_id": f"{case}__core__shared_reduction_adg",
            "status": "pass",
            "hardware_aware_cycles": 3,
            "dfg_cycles": 2,
        },
    )


def assert_sha256_file(path_text: str, fingerprint: str, repo: Path) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo / path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if fingerprint != digest:
        raise AssertionError(f"fingerprint mismatch for {path}: {fingerprint} != {digest}")


def artifact_exists(path_text: str, repo: Path) -> bool:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo / path
    return path.is_file()


def assert_counts(rows: list[dict[str, str]], data: dict[str, object]) -> None:
    expected_totals = {
        "app": APP_CASE_COUNT,
        "cmsis-dsp": 16,
        "cmsis-nn": 18,
        "loombench": LEGACY_CASE_COUNT,
    }
    by_suite = {suite: 0 for suite in expected_totals}
    for row in rows:
        by_suite[row["suite"]] = by_suite.get(row["suite"], 0) + 1
    if by_suite != expected_totals:
        raise AssertionError(f"unexpected suite totals: {by_suite}")

    counts = data.get("counts")
    if not isinstance(counts, dict):
        raise AssertionError(f"JSON SSOT lacks counts: {data}")
    for suite, total in expected_totals.items():
        suite_counts = counts.get(suite)
        if not isinstance(suite_counts, dict):
            raise AssertionError(f"missing counts for {suite}: {counts}")
        if suite_counts.get("total") != total:
            raise AssertionError(f"{suite} total={suite_counts.get('total')}, expected {total}")
        if suite == "loombench":
            expected = {
                "total": total,
                "pass": 0,
                "fail": 0,
                "blocked": 0,
                "unsupported": total,
                "missing_status": 0,
            }
            if suite_counts != expected:
                raise AssertionError(f"LoomBench baseline should consume generated manifest: {suite_counts}")
        elif suite == "app":
            expected = {
                "total": total,
                "pass": 0,
                "fail": 0,
                "blocked": total,
                "unsupported": 0,
                "missing_status": 0,
            }
            if suite_counts != expected:
                raise AssertionError(f"app baseline should structure every app row as non-missing: {suite_counts}")
        elif suite == "cmsis-dsp":
            expected = {
                "total": total,
                "pass": 0,
                "fail": 0,
                "blocked": 14,
                "unsupported": 2,
                "missing_status": 0,
            }
            if suite_counts != expected:
                raise AssertionError(f"{suite} baseline should consume default DFG MLIR evidence: {suite_counts}")
        elif suite == "cmsis-nn":
            expected = {
                "total": total,
                "pass": 0,
                "fail": 0,
                "blocked": 13,
                "unsupported": 5,
                "missing_status": 0,
            }
            if suite_counts != expected:
                raise AssertionError(f"{suite} baseline should consume default DFG MLIR evidence: {suite_counts}")
        else:
            if suite_counts.get("missing_status") != total:
                raise AssertionError(f"{suite} missing_status should equal total in baseline: {suite_counts}")
            for key in ("pass", "fail", "blocked", "unsupported"):
                if suite_counts.get(key) != 0:
                    raise AssertionError(f"{suite} {key} should be zero in baseline: {suite_counts}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    cgra_status_summary = import_cgra_status_summary(repo)
    with artifact_test_common.repo_temp_dir(repo, "loom-cgra-status-") as tmp:
        out_dir = Path(tmp)
        csv_output = out_dir / "cgra-status-summary.csv"
        json_output = out_dir / "cgra-status-summary.json"
        legacy_root = out_dir / "legacy-loombench"
        write_legacy_fixture(legacy_root)

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
            ],
        )
        rows = read_rows(csv_output)
        data = json.loads(json_output.read_text())
        if data.get("schema_version") != 1 or data.get("kind") != "cgra_status_summary":
            raise AssertionError(f"unexpected JSON header: {data}")
        if data.get("csv_projection") != str(csv_output):
            raise AssertionError(f"JSON should name CSV projection: {data}")
        assert_counts(rows, data)

        def assert_app_not_attempted(row_data: dict[str, str], case: str) -> None:
            if (
                row_data["status"] != "blocked"
                or row_data["diagnostic_class"] != "missing_dfg_report"
                or row_data["owner"] != "sim_report"
                or row_data["blocking_prerequisite"] != "dfg_report"
                or row_data["dfg_status"] != "not_run"
                or row_data["mapping_status"] != "not_run"
                or row_data["cgra_status"] != "not_run"
                or row_data["comparison_status"] != "not_run"
                or row_data["final_outputs_present"] != "false"
                or row_data["final_memory_state_present"] != "false"
                or row_data["dfg_report"]
                or row_data["mapping_artifact"]
                or row_data["cgra_report"]
                or row_data["comparison_report"]
                or f"DFG-sim report is absent for app row {case}" not in row_data["diagnostic"]
            ):
                raise AssertionError(f"{case} baseline should publish structured not-attempted evidence: {row_data}")

        app_vecsum = one_row(rows, "app", "vecsum")
        assert_app_not_attempted(app_vecsum, "vecsum")

        app_batchnorm = one_row(rows, "app", "batchnorm")
        assert_app_not_attempted(app_batchnorm, "batchnorm")
        if app_batchnorm["required_slice_count"] != "1":
            raise AssertionError(f"batchnorm should require one DFG slice after adding dfg_check.sh: {app_batchnorm}")
        app_interpolate = one_row(rows, "app", "interpolate_linear")
        assert_app_not_attempted(app_interpolate, "interpolate_linear")
        if app_interpolate["required_slice_count"] != "1":
            raise AssertionError(f"interpolate_linear should require one DFG slice after adding dfg_check.sh: {app_interpolate}")
        tampered_app_rows = [dict(row) for row in rows]
        tampered_app = one_row(tampered_app_rows, "app", "vecsum")
        tampered_app.update(
            {
                "status": "not_run",
                "diagnostic_class": "missing_status",
                "owner": "implementation",
                "blocking_prerequisite": "mapping_artifact",
                "diagnostic": "CGRA status missing after app dataflow tier; mapping artifact and CGRA-sim report are absent",
            }
        )
        tampered_app_csv = out_dir / "tampered-app-missing-cgra-status-summary.csv"
        tampered_app_json = out_dir / "tampered-app-missing-cgra-status-summary.json"
        write_rows(tampered_app_csv, tampered_app_rows)
        write_json_projection(tampered_app_json, tampered_app_csv, tampered_app_rows)
        failed_app_missing_audit = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(tampered_app_csv),
                "--json-input",
                str(tampered_app_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "app row must not use missing_status" not in failed_app_missing_audit.stderr:
            raise AssertionError(
                f"tampered app missing_status row should fail status audit: {failed_app_missing_audit.stderr}"
            )
        tampered_app_missing_dfg_rows = [dict(row) for row in rows]
        tampered_app_missing_dfg = one_row(tampered_app_missing_dfg_rows, "app", "vecsum")
        tampered_app_missing_dfg.update(
            {
                "owner": "implementation",
                "blocking_prerequisite": "dfg_report",
                "dfg_status": "not_run",
                "diagnostic": "DFG-sim report is absent for app row vecsum",
            }
        )
        tampered_app_missing_dfg_csv = out_dir / "tampered-app-missing-dfg-cgra-status-summary.csv"
        tampered_app_missing_dfg_json = out_dir / "tampered-app-missing-dfg-cgra-status-summary.json"
        write_rows(tampered_app_missing_dfg_csv, tampered_app_missing_dfg_rows)
        write_json_projection(tampered_app_missing_dfg_json, tampered_app_missing_dfg_csv, tampered_app_missing_dfg_rows)
        failed_app_missing_dfg_audit = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(tampered_app_missing_dfg_csv),
                "--json-input",
                str(tampered_app_missing_dfg_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "app missing DFG report row requires owner=sim_report" not in failed_app_missing_dfg_audit.stderr:
            raise AssertionError(
                "tampered app missing_dfg_report row should fail status audit: "
                f"{failed_app_missing_dfg_audit.stderr}"
            )
        tampered_app_missing_dfg_mapping = out_dir / "tampered-app-missing-dfg.mapping.json"
        write_json(
            tampered_app_missing_dfg_mapping,
            {
                "schema_version": 1,
                "kind": "pnr_mapping",
                "workload": "vecsum",
                "graph": "g_vecsum_0",
                "hardware": "shared_reduction_adg",
                "mapping_id": "vecsum__shared_reduction_adg",
                "status": "blocked",
                "diagnostics": ["fixture blocked mapping"],
            },
        )
        tampered_app_missing_dfg_later_rows = [dict(row) for row in rows]
        tampered_app_missing_dfg_later = one_row(tampered_app_missing_dfg_later_rows, "app", "vecsum")
        tampered_app_missing_dfg_later.update(
            {
                "mapping_artifact": str(tampered_app_missing_dfg_mapping),
                "mapping_artifact_fingerprint": hashlib.sha256(
                    tampered_app_missing_dfg_mapping.read_bytes()
                ).hexdigest(),
                "mapping_status": "blocked",
                "final_outputs_present": "true",
                "final_memory_state_present": "true",
            }
        )
        tampered_app_missing_dfg_later_csv = out_dir / "tampered-app-missing-dfg-later-cgra-status-summary.csv"
        tampered_app_missing_dfg_later_json = out_dir / "tampered-app-missing-dfg-later-cgra-status-summary.json"
        write_rows(tampered_app_missing_dfg_later_csv, tampered_app_missing_dfg_later_rows)
        write_json_projection(
            tampered_app_missing_dfg_later_json,
            tampered_app_missing_dfg_later_csv,
            tampered_app_missing_dfg_later_rows,
        )
        failed_app_missing_dfg_later_audit = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(tampered_app_missing_dfg_later_csv),
                "--json-input",
                str(tampered_app_missing_dfg_later_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "app missing DFG report row must not claim final-state evidence" not in failed_app_missing_dfg_later_audit.stderr:
            raise AssertionError(
                "tampered app missing_dfg_report row with later artifacts should fail status audit: "
                f"{failed_app_missing_dfg_later_audit.stderr}"
            )
        no_dfg_rows = [
            row for row in rows
            if row["suite"] == "app" and row["diagnostic_class"] == "app_dataflow_tier_missing"
        ]
        if len(no_dfg_rows) != APP_NO_DFG_TIER_COUNT:
            raise AssertionError(f"unexpected app no-DFG rows: {no_dfg_rows}")
        if no_dfg_rows:
            tampered_no_dfg_rows = [dict(row) for row in rows]
            tampered_no_dfg = one_row(tampered_no_dfg_rows, "app", no_dfg_rows[0]["case"])
            tampered_no_dfg.update(
                {
                    "status": "not_run",
                    "diagnostic_class": "missing_status",
                    "owner": "implementation",
                    "blocking_prerequisite": "dataflow",
                    "diagnostic": "CGRA status missing because app row has no dataflow tier yet",
                }
            )
            tampered_no_dfg_csv = out_dir / "tampered-no-dfg-cgra-status-summary.csv"
            tampered_no_dfg_json = out_dir / "tampered-no-dfg-cgra-status-summary.json"
            write_rows(tampered_no_dfg_csv, tampered_no_dfg_rows)
            write_json_projection(tampered_no_dfg_json, tampered_no_dfg_csv, tampered_no_dfg_rows)
            failed_no_dfg_audit = run(
                repo,
                [
                    "bash",
                    "test/e2e/run_cgra_status_audit.sh",
                    "--input",
                    str(tampered_no_dfg_csv),
                    "--json-input",
                    str(tampered_no_dfg_json),
                    "--legacy-loombench-root",
                    str(legacy_root),
                ],
                expect_success=False,
            )
            if "app row without dfg tier" not in failed_no_dfg_audit.stderr:
                raise AssertionError(f"tampered app no-DFG row should fail status audit: {failed_no_dfg_audit.stderr}")

        cmsis_dsp = one_row(rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        if cmsis_dsp["software_root"] != "externals/cmsis-dsp/Source":
            raise AssertionError(f"unexpected CMSIS-DSP root: {cmsis_dsp}")
        if (
            cmsis_dsp["status"] != "blocked"
            or cmsis_dsp["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or cmsis_dsp["blocking_prerequisite"] != "dfg_sim_report"
            or cmsis_dsp["owner"] != "compiler_pipeline"
            or not cmsis_dsp["dfg_mlir"]
            or not cmsis_dsp["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-DSP baseline row should consume DFG MLIR evidence: {cmsis_dsp}")

        cmsis_nn = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
        if cmsis_nn["software_root"] != "externals/cmsis-nn/Source":
            raise AssertionError(f"unexpected CMSIS-NN root: {cmsis_nn}")
        if (
            cmsis_nn["status"] != "blocked"
            or cmsis_nn["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or cmsis_nn["blocking_prerequisite"] != "dfg_sim_report"
            or cmsis_nn["owner"] != "compiler_pipeline"
            or not cmsis_nn["dfg_mlir"]
            or not cmsis_nn["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-NN baseline row should consume DFG MLIR evidence: {cmsis_nn}")
        cmsis_relu6 = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu6_s8.c")
        if (
            cmsis_relu6["status"] != "blocked"
            or cmsis_relu6["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or cmsis_relu6["blocking_prerequisite"] != "dfg_sim_report"
            or cmsis_relu6["owner"] != "compiler_pipeline"
            or cmsis_relu6["required_slice_count"] != "1"
            or "g_t_arm_relu6_s8_0_0" not in cmsis_relu6["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-NN relu6 baseline row should consume DFG MLIR evidence: {cmsis_relu6}")
        cmsis_dsp_no_graph = one_row(rows, "cmsis-dsp", "FastMathFunctions/arm_sin_f32.c")
        if (
            cmsis_dsp_no_graph["status"] != "unsupported"
            or cmsis_dsp_no_graph["diagnostic_class"] != "cmsis_no_dataflow_graph"
            or cmsis_dsp_no_graph["blocking_prerequisite"] != "dataflow_graph"
            or not cmsis_dsp_no_graph["dfg_mlir"]
        ):
            raise AssertionError(f"CMSIS-DSP no-graph row should be structured unsupported: {cmsis_dsp_no_graph}")
        no_cmsis_auto_csv = out_dir / "no-cmsis-auto-cgra-status-summary.csv"
        no_cmsis_auto_json = out_dir / "no-cmsis-auto-cgra-status-summary.json"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(no_cmsis_auto_csv),
                "--json-output",
                str(no_cmsis_auto_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--no-cmsis-dfg-auto",
            ],
        )
        no_cmsis_auto_rows = read_rows(no_cmsis_auto_csv)
        no_cmsis_auto_dsp = one_row(no_cmsis_auto_rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        if (
            no_cmsis_auto_dsp["status"] != "blocked"
            or no_cmsis_auto_dsp["diagnostic_class"] != "cmsis_dfg_mlir_missing"
            or no_cmsis_auto_dsp["blocking_prerequisite"] != "dfg_mlir"
        ):
            raise AssertionError(f"opt-out CMSIS row should preserve missing-DFG blocker: {no_cmsis_auto_dsp}")

        tampered_cmsis_rows = [dict(row) for row in rows]
        tampered_cmsis = one_row(tampered_cmsis_rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        tampered_cmsis.update(
            {
                "status": "not_run",
                "diagnostic_class": "missing_status",
                "owner": "implementation",
                "blocking_prerequisite": "mapping_artifact",
                "diagnostic": "CGRA status missing after CMSIS dataflow-shape row",
            }
        )
        tampered_cmsis_csv = out_dir / "tampered-cmsis-missing-cgra-status-summary.csv"
        tampered_cmsis_json = out_dir / "tampered-cmsis-missing-cgra-status-summary.json"
        write_rows(tampered_cmsis_csv, tampered_cmsis_rows)
        write_json_projection(tampered_cmsis_json, tampered_cmsis_csv, tampered_cmsis_rows)
        failed_cmsis_missing_audit = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(tampered_cmsis_csv),
                "--json-input",
                str(tampered_cmsis_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "CMSIS row must not use missing_status" not in failed_cmsis_missing_audit.stderr:
            raise AssertionError(
                f"tampered CMSIS missing_status row should fail status audit: {failed_cmsis_missing_audit.stderr}"
            )

        loombench = one_row(rows, "loombench", "breadth_first_search")
        if (
            loombench["status"] != "unsupported"
            or loombench["diagnostic_class"] != "loombench_import_excluded"
            or loombench["blocking_prerequisite"] != "legacy_source"
        ):
            raise AssertionError(f"LoomBench legacy rows should consume generated manifest: {loombench}")

        default_legacy_root = out_dir / "default-legacy-source"
        write_ready_legacy_case(default_legacy_root, "vecadd")
        write_ready_legacy_case(default_legacy_root, "batchnorm")
        write_ready_legacy_case(default_legacy_root, "breadth_first_search")
        (default_legacy_root / "blocked_case").mkdir(parents=True)
        default_legacy_out_dir = out_dir / "default-legacy-root"
        default_legacy_csv = default_legacy_out_dir / "cgra-status-summary.csv"
        default_legacy_json = default_legacy_out_dir / "cgra-status-summary.json"
        run(
            repo,
            [
                "env",
                f"LOOM_LEGACY_LOOMBENCH_ROOT={default_legacy_root}",
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(default_legacy_csv),
                "--json-output",
                str(default_legacy_json),
            ],
        )
        default_legacy_rows = read_rows(default_legacy_csv)
        default_legacy_loombench = [row for row in default_legacy_rows if row["suite"] == "loombench"]
        if not default_legacy_loombench:
            raise AssertionError("default legacy root should emit LoomBench rows when the root exists")
        default_legacy_counts = suite_counts(default_legacy_loombench).get("loombench")
        if default_legacy_counts != {
            "total": 4,
            "pass": 0,
            "fail": 0,
            "blocked": 3,
            "unsupported": 1,
            "missing_status": 0,
        }:
            raise AssertionError(f"default legacy root should produce row-specific LoomBench counts: {default_legacy_counts}")
        expected_default_classes = {
            "loombench_workload_identity_bridge_ready": 2,
            "loombench_import_deferred": 1,
            "loombench_import_excluded": 1,
        }
        observed_default_classes: dict[str, int] = {}
        for row_data in default_legacy_loombench:
            observed_default_classes[row_data["diagnostic_class"]] = (
                observed_default_classes.get(row_data["diagnostic_class"], 0) + 1
            )
        if observed_default_classes != expected_default_classes:
            raise AssertionError(
                "default legacy root should expose accepted, deferred, and excluded row-specific states: "
                f"{observed_default_classes}"
            )
        default_vecadd = one_row(default_legacy_rows, "loombench", "vecadd")
        if (
            default_vecadd["status"] != "blocked"
            or default_vecadd["diagnostic_class"] != "loombench_workload_identity_bridge_ready"
            or default_vecadd["blocking_prerequisite"] != "sim_evidence"
            or default_vecadd["manifest_case"] != "vecadd"
        ):
            raise AssertionError(f"default legacy vecadd should expose bridge-ready status: {default_vecadd}")
        default_batchnorm = one_row(default_legacy_rows, "loombench", "batchnorm")
        if (
            default_batchnorm["status"] != "blocked"
            or default_batchnorm["diagnostic_class"] != "loombench_workload_identity_bridge_ready"
            or default_batchnorm["blocking_prerequisite"] != "sim_evidence"
            or default_batchnorm["manifest_case"] != "batchnorm"
        ):
            raise AssertionError(f"default legacy batchnorm should expose bridge-ready status: {default_batchnorm}")
        default_deferred = one_row(default_legacy_rows, "loombench", "breadth_first_search")
        if (
            default_deferred["status"] != "blocked"
            or default_deferred["diagnostic_class"] != "loombench_import_deferred"
            or default_deferred["blocking_prerequisite"] != "app_import"
        ):
            raise AssertionError(f"default legacy-only row should be deferred: {default_deferred}")
        default_excluded = one_row(default_legacy_rows, "loombench", "blocked_case")
        if (
            default_excluded["status"] != "unsupported"
            or default_excluded["diagnostic_class"] != "loombench_import_excluded"
            or default_excluded["blocking_prerequisite"] != "legacy_source"
        ):
            raise AssertionError(f"default blocked legacy source should be excluded: {default_excluded}")
        missing_manifest_rows = [
            row for row in default_legacy_loombench if row["diagnostic_class"] == "loombench_manifest_missing"
        ]
        if missing_manifest_rows:
            raise AssertionError(
                "default legacy root should generate a LoomBench manifest sidecar instead of manifest-missing rows: "
                f"{missing_manifest_rows[:3]}"
            )
        if not (default_legacy_csv.parent / "loombench-manifest.json").is_file():
            raise AssertionError("default legacy root wrapper should emit a LoomBench manifest JSON sidecar")
        if not (default_legacy_csv.parent / "loombench-manifest.csv").is_file():
            raise AssertionError("default legacy root wrapper should emit a LoomBench manifest CSV sidecar")
        run(
            repo,
            [
                "env",
                f"LOOM_LEGACY_LOOMBENCH_ROOT={out_dir / 'missing-default-root'}",
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(default_legacy_csv),
                "--json-input",
                str(default_legacy_json),
            ],
        )

        explicit_manifest = out_dir / "explicit-loombench-manifest.json"
        write_loombench_manifest(
            explicit_manifest,
            [
                {
                    "case": "breadth_first_search",
                    "source_row": "breadth_first_search",
                    "software_root": "synthetic-explicit-root/breadth_first_search",
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
                    "reason": "explicit manifest should win",
                }
            ],
        )
        explicit_csv = out_dir / "explicit-form-cgra-status-summary.csv"
        explicit_json = out_dir / "explicit-form-cgra-status-summary.json"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(explicit_csv),
                "--json-output",
                str(explicit_json),
                "--legacy-loombench-root",
                str(legacy_root),
                f"--loombench-manifest={explicit_manifest}",
            ],
        )
        explicit_rows = read_rows(explicit_csv)
        explicit_loombench = [row for row in explicit_rows if row["suite"] == "loombench"]
        if len(explicit_loombench) != 1:
            raise AssertionError(f"explicit manifest should define LoomBench row coverage: {explicit_loombench[:3]}")
        explicit_bfs = explicit_loombench[0]
        if (
            explicit_bfs["case"] != "breadth_first_search"
            or explicit_bfs["status"] != "blocked"
            or explicit_bfs["diagnostic_class"] != "loombench_import_deferred"
            or explicit_bfs["diagnostic"] != "explicit manifest should win"
        ):
            raise AssertionError(f"explicit --loombench-manifest= form should not be overridden: {explicit_bfs}")
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(explicit_csv),
                "--json-input",
                str(explicit_json),
                "--legacy-loombench-root",
                str(legacy_root),
                f"--loombench-manifest={explicit_manifest}",
            ],
        )

        direct_no_manifest_csv = out_dir / "direct-no-manifest-cgra-status-summary.csv"
        direct_no_manifest_json = out_dir / "direct-no-manifest-cgra-status-summary.json"
        run(
            repo,
            [
                "python3",
                "test/e2e/cgra_status_summary.py",
                "--output",
                str(direct_no_manifest_csv),
                "--json-output",
                str(direct_no_manifest_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
        )
        direct_no_manifest_rows = read_rows(direct_no_manifest_csv)
        direct_no_manifest_data = json.loads(direct_no_manifest_json.read_text())
        loombench_counts = direct_no_manifest_data.get("counts", {}).get("loombench")
        if loombench_counts != {
            "total": LEGACY_CASE_COUNT,
            "pass": 0,
            "fail": 0,
            "blocked": LEGACY_CASE_COUNT,
            "unsupported": 0,
            "missing_status": 0,
        }:
            raise AssertionError(f"LoomBench rows without a manifest should be structured blockers: {loombench_counts}")
        direct_no_manifest_bfs = one_row(direct_no_manifest_rows, "loombench", "breadth_first_search")
        if (
            direct_no_manifest_bfs["status"] != "blocked"
            or direct_no_manifest_bfs["diagnostic_class"] != "loombench_manifest_missing"
            or direct_no_manifest_bfs["blocking_prerequisite"] != "loombench_manifest"
            or direct_no_manifest_bfs["owner"] != "loombench_manifest"
        ):
            raise AssertionError(
                f"LoomBench row without a manifest should block on manifest reconciliation: {direct_no_manifest_bfs}"
            )
        direct_no_manifest_audit = out_dir / "direct-no-manifest-artifact-audit.json"
        run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(direct_no_manifest_audit),
                str(direct_no_manifest_csv),
            ],
        )
        tampered_loombench_rows = [dict(row) for row in direct_no_manifest_rows]
        tampered_loombench = one_row(tampered_loombench_rows, "loombench", "breadth_first_search")
        tampered_loombench.update(
            {
                "status": "not_run",
                "diagnostic_class": "missing_status",
                "owner": "implementation",
                "blocking_prerequisite": "loombench_manifest",
                "diagnostic": "CGRA status missing because dedicated LoomBench manifest reconciliation is absent",
            }
        )
        tampered_loombench_csv = out_dir / "tampered-loombench-missing-cgra-status-summary.csv"
        tampered_loombench_json = out_dir / "tampered-loombench-missing-cgra-status-summary.json"
        write_rows(tampered_loombench_csv, tampered_loombench_rows)
        write_json_projection(tampered_loombench_json, tampered_loombench_csv, tampered_loombench_rows)
        tampered_loombench_audit = out_dir / "tampered-loombench-artifact-audit.json"
        failed_loombench_missing_audit = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(tampered_loombench_audit),
                str(tampered_loombench_csv),
            ],
            expect_success=False,
        )
        tampered_loombench_audit_data = (
            json.loads(tampered_loombench_audit.read_text())
            if tampered_loombench_audit.is_file()
            else {}
        )
        tampered_loombench_diagnostics = "\n".join(
            str(item) for item in tampered_loombench_audit_data.get("diagnostics", [])
        )
        if "LoomBench row must not use missing_status" not in tampered_loombench_diagnostics:
            raise AssertionError(
                "tampered LoomBench missing_status row should fail generic artifact audit: "
                f"stdout={failed_loombench_missing_audit.stdout} "
                f"stderr={failed_loombench_missing_audit.stderr} audit={tampered_loombench_audit_data}"
            )
        forged_loombench_pass_rows = [dict(row) for row in direct_no_manifest_rows]
        forged_loombench_pass = one_row(forged_loombench_pass_rows, "loombench", "breadth_first_search")
        forged_loombench_pass.update(
            {
                "dfg_status": "pass",
                "mapping_status": "pass",
                "cgra_status": "pass",
                "comparison_status": "pass",
                "status": "pass",
                "diagnostic_class": "loombench_manifest_missing",
                "final_outputs_present": "true",
                "final_memory_state_present": "true",
            }
        )
        for artifact_column, fingerprint_column in (
            ("dfg_report", "dfg_report_fingerprint"),
            ("mapping_artifact", "mapping_artifact_fingerprint"),
            ("cgra_report", "cgra_report_fingerprint"),
            ("comparison_report", "comparison_report_fingerprint"),
        ):
            forged_loombench_pass[artifact_column] = str(out_dir / f"missing-loombench-{artifact_column}.json")
            forged_loombench_pass[fingerprint_column] = "0" * 64
        forged_loombench_pass_csv = out_dir / "forged-loombench-pass-cgra-status-summary.csv"
        forged_loombench_pass_json = out_dir / "forged-loombench-pass-cgra-status-summary.json"
        write_rows(forged_loombench_pass_csv, forged_loombench_pass_rows)
        write_json_projection(forged_loombench_pass_json, forged_loombench_pass_csv, forged_loombench_pass_rows)
        forged_loombench_pass_audit = out_dir / "forged-loombench-pass-generic-audit.json"
        failed_loombench_pass = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_loombench_pass_audit),
                str(forged_loombench_pass_csv),
            ],
            expect_success=False,
        )
        forged_loombench_pass_data = json.loads(forged_loombench_pass_audit.read_text())
        forged_loombench_pass_diagnostics = "\n".join(
            str(item) for item in forged_loombench_pass_data.get("diagnostics", [])
        )
        if "LoomBench row without manifest requires status=blocked" not in forged_loombench_pass_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged LoomBench no-manifest pass rows: "
                f"stdout={failed_loombench_pass.stdout} stderr={failed_loombench_pass.stderr} "
                f"audit={forged_loombench_pass_data}"
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

        ledger = out_dir / "unsupported-scope-ledger.csv"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_unsupported_scope_ledger.sh",
                "--artifact",
                str(csv_output),
                "--output",
                str(ledger),
            ],
        )
        with ledger.open(newline="") as handle:
            ledger_rows = list(csv.DictReader(handle))
        vecsum_gaps = [
            row for row in ledger_rows
            if row["artifact"] == "cgra_status"
            and row["case"] == "app:vecsum:vecsum"
            and row["stage"] == "status"
        ]
        if len(vecsum_gaps) != 1:
            raise AssertionError(f"expected one vecsum CGRA status gap, got {ledger_rows[:10]}")
        if (
            "status=blocked" not in vecsum_gaps[0]["reason"]
            or "DFG-sim report is absent for app row vecsum" not in vecsum_gaps[0]["reason"]
        ):
            raise AssertionError(f"ledger row should preserve structured blocked status: {vecsum_gaps[0]}")

        basename_dfg_dir = out_dir / "basename-dfg"
        write_cmsis_dfg_mlir(basename_dfg_dir / "arm_source_name.dfg.mlir", symbol="different_exported_symbol", graph=True)
        basename_row = cgra_status_summary.row(
            suite="cmsis-dsp",
            case="Synthetic/arm_source_name.c",
            source_row="Synthetic/arm_source_name.c",
            software_root="externals/cmsis-dsp/Source",
            required_slice_count="1",
            blocking_prerequisite="mapping_artifact",
            diagnostic="synthetic baseline",
        )
        cgra_status_summary.apply_cmsis_dfg_mlir_evidence(
            basename_row,
            [
                "Synthetic/arm_source_name.c",
                "thumbv7em-none-eabi",
                "cortex-m4",
                "thumbv7em-unknown-none-eabi",
                "different_exported_symbol",
                "",
                "1",
                "1",
                "1",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
            ],
            basename_dfg_dir,
        )
        if (
            basename_row["status"] != "blocked"
            or not basename_row["dfg_mlir"].endswith("arm_source_name.dfg.mlir")
            or "g_different_exported_symbol_0" not in basename_row["graph_ids"]
        ):
            raise AssertionError(f"CMSIS DFG evidence filename should follow source basename: {basename_row}")

        stale_default_dir = out_dir / "stale-default"
        stale_default_dfg = stale_default_dir / "test-runs" / "cmsis-dsp" / "dfg"
        write_cmsis_dfg_mlir(stale_default_dfg / "arm_add_q15.dfg.mlir", symbol="arm_add_q15", graph=True)
        stale_default_csv = stale_default_dir / "cgra-status-summary.csv"
        stale_default_json = stale_default_dir / "cgra-status-summary.json"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(stale_default_csv),
                "--json-output",
                str(stale_default_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--no-cmsis-dfg-auto",
            ],
        )
        stale_default_rows = read_rows(stale_default_csv)
        stale_default_add = one_row(stale_default_rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        if stale_default_add["status"] != "blocked" or stale_default_add["diagnostic_class"] != "cmsis_dfg_mlir_missing":
            raise AssertionError(f"opt-out CGRA status must not consume stale CMSIS DFG evidence: {stale_default_add}")

        cmsis_dsp_dfg_dir = out_dir / "cmsis-dsp-fixture-dfg"
        cmsis_nn_dfg_dir = out_dir / "cmsis-nn-fixture-dfg"
        write_cmsis_dfg_mlir(cmsis_dsp_dfg_dir / "arm_add_q15.dfg.mlir", symbol="arm_add_q15", graph=True)
        write_cmsis_dfg_mlir(cmsis_dsp_dfg_dir / "arm_mult_f32.dfg.mlir", symbol="wrong_symbol", graph=True)
        write_cmsis_dfg_mlir(cmsis_dsp_dfg_dir / "arm_sin_f32.dfg.mlir", symbol="arm_sin_f32", graph=False)
        write_cmsis_dfg_mlir(cmsis_nn_dfg_dir / "arm_relu_q15.dfg.mlir", symbol="arm_relu_q15", graph=True)
        write_cmsis_dfg_mlir(cmsis_nn_dfg_dir / "arm_reshape_s8.dfg.mlir", symbol="arm_reshape_s8", graph=False)
        cmsis_sim_evidence = out_dir / "cmsis-sim-evidence"
        write_json(
            cmsis_sim_evidence / "arm_add_q15.dfg.report.json",
            {
                "schema_version": 1,
                "kind": "dfg_sim_report",
                "workload": "BasicMathFunctions/arm_add_q15.c",
                "graph": "g_arm_add_q15_0",
                "status": "unsupported",
                "optimistic_cycles": 0,
                **dfg_cycle_fixture_fields(0),
                "wavefront_steps": 0,
                "event_count": 0,
                "dynamic_work_items": 0,
                "operation_fire_counts": {},
                "final_outputs": [],
                "final_memory_state": {},
                "diagnostics": ["unsupported op: llvm.getelementptr"],
                "input_artifact_fingerprints": {
                    "arm_add_q15.dfg": hashlib.sha256(
                        (cmsis_dsp_dfg_dir / "arm_add_q15.dfg.mlir").read_bytes()
                    ).hexdigest(),
                },
                "metric_definition": "fixture",
                "operation_semantics_source": "fixture",
                "operation_cost_model_source": "fixture",
            },
        )
        cmsis_evidence_csv = out_dir / "cmsis-evidence-cgra-status-summary.csv"
        cmsis_evidence_json = out_dir / "cmsis-evidence-cgra-status-summary.json"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(cmsis_evidence_csv),
                "--json-output",
                str(cmsis_evidence_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--cmsis-dsp-dfg-dir",
                str(cmsis_dsp_dfg_dir),
                "--cmsis-nn-dfg-dir",
                str(cmsis_nn_dfg_dir),
                "--sim-evidence-dir",
                str(cmsis_sim_evidence),
            ],
        )
        cmsis_evidence_rows = read_rows(cmsis_evidence_csv)
        cmsis_evidence_data = json.loads(cmsis_evidence_json.read_text())
        cmsis_counts = cmsis_evidence_data.get("counts", {})
        if cmsis_counts.get("cmsis-dsp") != {
            "total": 16,
            "pass": 0,
            "fail": 1,
            "blocked": 13,
            "unsupported": 2,
            "missing_status": 0,
        }:
            raise AssertionError(f"unexpected CMSIS-DSP evidence counts: {cmsis_counts.get('cmsis-dsp')}")
        if cmsis_counts.get("cmsis-nn") != {
            "total": 18,
            "pass": 0,
            "fail": 0,
            "blocked": 17,
            "unsupported": 1,
            "missing_status": 0,
        }:
            raise AssertionError(f"unexpected CMSIS-NN evidence counts: {cmsis_counts.get('cmsis-nn')}")
        cmsis_add = one_row(cmsis_evidence_rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        if (
            cmsis_add["status"] != "unsupported"
            or cmsis_add["diagnostic_class"] != "dfg_report_unsupported"
            or cmsis_add["blocking_prerequisite"] != "dfg_report"
            or cmsis_add["owner"] != "sim_report"
            or cmsis_add["dfg_status"] != "unsupported"
            or "unsupported op: llvm.getelementptr" not in cmsis_add["diagnostic"]
            or "g_arm_add_q15_0" not in cmsis_add["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-DSP DFG-sim evidence should become an exact report blocker: {cmsis_add}")
        assert_sha256_file(cmsis_add["dfg_mlir"], cmsis_add["dfg_mlir_fingerprint"], repo)
        assert_sha256_file(cmsis_add["dfg_report"], cmsis_add["dfg_report_fingerprint"], repo)
        forged_cmsis_unsupported_status_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_cmsis_unsupported_status = one_row(
            forged_cmsis_unsupported_status_rows,
            "cmsis-dsp",
            "BasicMathFunctions/arm_add_q15.c",
        )
        forged_cmsis_unsupported_status["status"] = "blocked"
        forged_cmsis_unsupported_status_csv = out_dir / "forged-cmsis-unsupported-status-cgra-status-summary.csv"
        forged_cmsis_unsupported_status_json = out_dir / "forged-cmsis-unsupported-status-cgra-status-summary.json"
        write_rows(forged_cmsis_unsupported_status_csv, forged_cmsis_unsupported_status_rows)
        write_json_projection(
            forged_cmsis_unsupported_status_json,
            forged_cmsis_unsupported_status_csv,
            forged_cmsis_unsupported_status_rows,
        )
        failed_forged_cmsis_unsupported_status = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_cmsis_unsupported_status_csv),
                "--json-input",
                str(forged_cmsis_unsupported_status_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "DFG unsupported report row requires status=unsupported" not in failed_forged_cmsis_unsupported_status.stderr:
            raise AssertionError(
                "forged CMSIS unsupported DFG row status should fail CGRA status audit: "
                f"{failed_forged_cmsis_unsupported_status.stderr}"
            )
        forged_cmsis_unsupported_status_generic = out_dir / "forged-cmsis-unsupported-status-generic-audit.json"
        failed_forged_cmsis_unsupported_status_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_cmsis_unsupported_status_generic),
                str(forged_cmsis_unsupported_status_csv),
            ],
            expect_success=False,
        )
        forged_cmsis_unsupported_status_generic_data = json.loads(
            forged_cmsis_unsupported_status_generic.read_text()
        )
        forged_cmsis_unsupported_status_generic_diagnostics = "\n".join(
            str(item) for item in forged_cmsis_unsupported_status_generic_data.get("diagnostics", [])
        )
        if "DFG unsupported report row requires status=unsupported" not in forged_cmsis_unsupported_status_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CMSIS unsupported DFG row status: "
                f"stdout={failed_forged_cmsis_unsupported_status_generic.stdout} "
                f"stderr={failed_forged_cmsis_unsupported_status_generic.stderr} "
                f"audit={forged_cmsis_unsupported_status_generic_data}"
            )
        forged_cmsis_report_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_cmsis_report = one_row(
            forged_cmsis_report_rows,
            "cmsis-dsp",
            "BasicMathFunctions/arm_add_q15.c",
        )
        forged_cmsis_report["dfg_report_fingerprint"] = "0" * 64
        forged_cmsis_report_csv = out_dir / "forged-cmsis-dfg-report-cgra-status-summary.csv"
        forged_cmsis_report_json = out_dir / "forged-cmsis-dfg-report-cgra-status-summary.json"
        write_rows(forged_cmsis_report_csv, forged_cmsis_report_rows)
        write_json_projection(forged_cmsis_report_json, forged_cmsis_report_csv, forged_cmsis_report_rows)
        failed_forged_cmsis_report = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_cmsis_report_csv),
                "--json-input",
                str(forged_cmsis_report_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "dfg_report_fingerprint" not in failed_forged_cmsis_report.stderr:
            raise AssertionError(
                "forged CMSIS DFG-sim report fingerprint should fail CGRA status audit: "
                f"{failed_forged_cmsis_report.stderr}"
            )
        forged_cmsis_report_generic = out_dir / "forged-cmsis-dfg-report-generic-audit.json"
        failed_forged_cmsis_report_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_cmsis_report_generic),
                str(forged_cmsis_report_csv),
            ],
            expect_success=False,
        )
        forged_cmsis_report_generic_data = json.loads(forged_cmsis_report_generic.read_text())
        forged_cmsis_report_generic_diagnostics = "\n".join(
            str(item) for item in forged_cmsis_report_generic_data.get("diagnostics", [])
        )
        if "dfg_report_fingerprint" not in forged_cmsis_report_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CMSIS DFG-sim report fingerprint: "
                f"stdout={failed_forged_cmsis_report_generic.stdout} "
                f"stderr={failed_forged_cmsis_report_generic.stderr} "
                f"audit={forged_cmsis_report_generic_data}"
            )
        forged_cmsis_bad_json_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_cmsis_bad_json = one_row(
            forged_cmsis_bad_json_rows,
            "cmsis-dsp",
            "BasicMathFunctions/arm_add_q15.c",
        )
        bad_json_report = out_dir / "cmsis-bad-json.dfg.report.json"
        bad_json_report.write_text("{bad-json\n")
        forged_cmsis_bad_json["dfg_report"] = str(bad_json_report)
        forged_cmsis_bad_json["dfg_report_fingerprint"] = hashlib.sha256(bad_json_report.read_bytes()).hexdigest()
        forged_cmsis_bad_json_csv = out_dir / "forged-cmsis-dfg-report-bad-json-cgra-status-summary.csv"
        forged_cmsis_bad_json_json = out_dir / "forged-cmsis-dfg-report-bad-json-cgra-status-summary.json"
        write_rows(forged_cmsis_bad_json_csv, forged_cmsis_bad_json_rows)
        write_json_projection(forged_cmsis_bad_json_json, forged_cmsis_bad_json_csv, forged_cmsis_bad_json_rows)
        failed_cmsis_bad_json = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_cmsis_bad_json_csv),
                "--json-input",
                str(forged_cmsis_bad_json_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "referenced dfg_report JSON is not parseable" not in failed_cmsis_bad_json.stderr:
            raise AssertionError(
                "malformed CMSIS DFG-sim report JSON should fail CGRA status audit: "
                f"{failed_cmsis_bad_json.stderr}"
            )
        forged_cmsis_bad_json_generic = out_dir / "forged-cmsis-dfg-report-bad-json-generic-audit.json"
        failed_cmsis_bad_json_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_cmsis_bad_json_generic),
                str(forged_cmsis_bad_json_csv),
            ],
            expect_success=False,
        )
        forged_cmsis_bad_json_generic_data = json.loads(forged_cmsis_bad_json_generic.read_text())
        forged_cmsis_bad_json_generic_diagnostics = "\n".join(
            str(item) for item in forged_cmsis_bad_json_generic_data.get("diagnostics", [])
        )
        if "referenced dfg_report JSON is not parseable" not in forged_cmsis_bad_json_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject malformed CMSIS DFG-sim report JSON: "
                f"stdout={failed_cmsis_bad_json_generic.stdout} "
                f"stderr={failed_cmsis_bad_json_generic.stderr} "
                f"audit={forged_cmsis_bad_json_generic_data}"
            )
        forged_cmsis_wrong_graph_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_cmsis_wrong_graph = one_row(
            forged_cmsis_wrong_graph_rows,
            "cmsis-dsp",
            "BasicMathFunctions/arm_add_q15.c",
        )
        wrong_graph_report = out_dir / "cmsis-wrong-graph.dfg.report.json"
        wrong_graph_data = json.loads((cmsis_sim_evidence / "arm_add_q15.dfg.report.json").read_text())
        wrong_graph_data["graph"] = "g_not_in_cmsis_dfg_mlir"
        write_json(wrong_graph_report, wrong_graph_data)
        forged_cmsis_wrong_graph["dfg_report"] = str(wrong_graph_report)
        forged_cmsis_wrong_graph["dfg_report_fingerprint"] = hashlib.sha256(wrong_graph_report.read_bytes()).hexdigest()
        forged_cmsis_wrong_graph_csv = out_dir / "forged-cmsis-dfg-report-wrong-graph-cgra-status-summary.csv"
        forged_cmsis_wrong_graph_json = out_dir / "forged-cmsis-dfg-report-wrong-graph-cgra-status-summary.json"
        write_rows(forged_cmsis_wrong_graph_csv, forged_cmsis_wrong_graph_rows)
        write_json_projection(forged_cmsis_wrong_graph_json, forged_cmsis_wrong_graph_csv, forged_cmsis_wrong_graph_rows)
        failed_cmsis_wrong_graph = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_cmsis_wrong_graph_csv),
                "--json-input",
                str(forged_cmsis_wrong_graph_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "referenced dfg_report graph is not listed in row graph_ids" not in failed_cmsis_wrong_graph.stderr:
            raise AssertionError(
                "wrong-graph CMSIS DFG-sim report should fail CGRA status audit: "
                f"{failed_cmsis_wrong_graph.stderr}"
            )
        forged_cmsis_wrong_graph_generic = out_dir / "forged-cmsis-dfg-report-wrong-graph-generic-audit.json"
        failed_cmsis_wrong_graph_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_cmsis_wrong_graph_generic),
                str(forged_cmsis_wrong_graph_csv),
            ],
            expect_success=False,
        )
        forged_cmsis_wrong_graph_generic_data = json.loads(forged_cmsis_wrong_graph_generic.read_text())
        forged_cmsis_wrong_graph_generic_diagnostics = "\n".join(
            str(item) for item in forged_cmsis_wrong_graph_generic_data.get("diagnostics", [])
        )
        if "referenced dfg_report graph is not listed in row graph_ids" not in forged_cmsis_wrong_graph_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject wrong-graph CMSIS DFG-sim report: "
                f"stdout={failed_cmsis_wrong_graph_generic.stdout} "
                f"stderr={failed_cmsis_wrong_graph_generic.stderr} "
                f"audit={forged_cmsis_wrong_graph_generic_data}"
            )
        cmsis_sin = one_row(cmsis_evidence_rows, "cmsis-dsp", "FastMathFunctions/arm_sin_f32.c")
        if (
            cmsis_sin["status"] != "unsupported"
            or cmsis_sin["diagnostic_class"] != "cmsis_no_dataflow_graph"
            or cmsis_sin["blocking_prerequisite"] != "dataflow_graph"
            or cmsis_sin["required_slice_count"] != "0"
        ):
            raise AssertionError(f"CMSIS-DSP no-graph DFG MLIR should become structured unsupported: {cmsis_sin}")
        assert_sha256_file(cmsis_sin["dfg_mlir"], cmsis_sin["dfg_mlir_fingerprint"], repo)
        cmsis_mult = one_row(cmsis_evidence_rows, "cmsis-dsp", "BasicMathFunctions/arm_mult_f32.c")
        if (
            cmsis_mult["status"] != "fail"
            or cmsis_mult["diagnostic_class"] != "cmsis_dfg_mlir_identity_mismatch"
            or cmsis_mult["blocking_prerequisite"] != "dataflow_graph_identity"
        ):
            raise AssertionError(f"CMSIS DFG MLIR with wrong symbol should become a row-specific failure: {cmsis_mult}")
        cmsis_relu = one_row(cmsis_evidence_rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
        if (
            cmsis_relu["status"] != "blocked"
            or cmsis_relu["diagnostic_class"] != "cmsis_dfg_mlir_ready_for_dfg_sim"
            or "g_arm_relu_q15_0" not in cmsis_relu["graph_ids"]
        ):
            raise AssertionError(f"CMSIS-NN DFG MLIR evidence should become an exact DFG-sim blocker: {cmsis_relu}")
        cmsis_reshape = one_row(cmsis_evidence_rows, "cmsis-nn", "ReshapeFunctions/arm_reshape_s8.c")
        if cmsis_reshape["status"] != "unsupported" or cmsis_reshape["diagnostic_class"] != "cmsis_no_dataflow_graph":
            raise AssertionError(f"CMSIS-NN no-graph DFG MLIR should become structured unsupported: {cmsis_reshape}")
        forged_cmsis_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_cmsis = one_row(forged_cmsis_rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        forged_cmsis["dfg_mlir"] = str(out_dir / "missing-cmsis-dfg.mlir")
        forged_cmsis["dfg_mlir_fingerprint"] = "0" * 64
        forged_cmsis_csv = out_dir / "forged-cmsis-cgra-status-summary.csv"
        forged_cmsis_json = out_dir / "forged-cmsis-cgra-status-summary.json"
        write_rows(forged_cmsis_csv, forged_cmsis_rows)
        write_json_projection(forged_cmsis_json, forged_cmsis_csv, forged_cmsis_rows)
        failed_forged_cmsis = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_cmsis_csv),
                "--json-input",
                str(forged_cmsis_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "dfg_mlir" not in failed_forged_cmsis.stderr:
            raise AssertionError(f"forged CMSIS DFG evidence should fail CGRA status audit: {failed_forged_cmsis.stderr}")
        forged_cmsis_generic_audit = out_dir / "forged-cmsis-generic-audit.json"
        failed_forged_cmsis_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_cmsis_generic_audit),
                str(forged_cmsis_csv),
            ],
            expect_success=False,
        )
        forged_cmsis_generic_data = json.loads(forged_cmsis_generic_audit.read_text())
        forged_cmsis_generic_diagnostics = "\n".join(
            str(item) for item in forged_cmsis_generic_data.get("diagnostics", [])
        )
        if "dfg_mlir" not in forged_cmsis_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CMSIS DFG evidence: "
                f"stdout={failed_forged_cmsis_generic.stdout} stderr={failed_forged_cmsis_generic.stderr} "
                f"audit={forged_cmsis_generic_data}"
            )
        forged_binding_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_binding = one_row(forged_binding_rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
        forged_binding["dfg_mlir"] = cmsis_reshape["dfg_mlir"]
        forged_binding["dfg_mlir_fingerprint"] = cmsis_reshape["dfg_mlir_fingerprint"]
        forged_binding["required_slice_count"] = "99"
        forged_binding_csv = out_dir / "forged-cmsis-binding-cgra-status-summary.csv"
        forged_binding_json = out_dir / "forged-cmsis-binding-cgra-status-summary.json"
        write_rows(forged_binding_csv, forged_binding_rows)
        write_json_projection(forged_binding_json, forged_binding_csv, forged_binding_rows)
        failed_forged_binding = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_binding_csv),
                "--json-input",
                str(forged_binding_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "dfg_mlir basename" not in failed_forged_binding.stderr or "required_slice_count" not in failed_forged_binding.stderr:
            raise AssertionError(
                "forged CMSIS binding should fail CGRA status audit on basename and slice count: "
                f"{failed_forged_binding.stderr}"
            )
        forged_binding_generic_audit = out_dir / "forged-cmsis-binding-generic-audit.json"
        run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_binding_generic_audit),
                str(forged_binding_csv),
            ],
            expect_success=False,
        )
        forged_binding_generic_data = json.loads(forged_binding_generic_audit.read_text())
        forged_binding_generic_diagnostics = "\n".join(
            str(item) for item in forged_binding_generic_data.get("diagnostics", [])
        )
        if "dfg_mlir basename" not in forged_binding_generic_diagnostics or "required_slice_count" not in forged_binding_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CMSIS DFG binding: "
                f"{forged_binding_generic_data}"
            )
        forged_ready_no_graph_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_ready_no_graph = one_row(forged_ready_no_graph_rows, "cmsis-dsp", "FastMathFunctions/arm_sin_f32.c")
        forged_ready_no_graph.update(
            {
                "status": "blocked",
                "diagnostic_class": "cmsis_dfg_mlir_ready_for_dfg_sim",
                "blocking_prerequisite": "dfg_sim_report",
                "graph_ids": "g_forged",
                "required_slice_count": "1",
            }
        )
        forged_ready_no_graph_csv = out_dir / "forged-cmsis-ready-no-graph-cgra-status-summary.csv"
        forged_ready_no_graph_json = out_dir / "forged-cmsis-ready-no-graph-cgra-status-summary.json"
        write_rows(forged_ready_no_graph_csv, forged_ready_no_graph_rows)
        write_json_projection(forged_ready_no_graph_json, forged_ready_no_graph_csv, forged_ready_no_graph_rows)
        failed_ready_no_graph = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_ready_no_graph_csv),
                "--json-input",
                str(forged_ready_no_graph_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "ready row requires dfg_mlir content graph_ids" not in failed_ready_no_graph.stderr:
            raise AssertionError(
                "forged CMSIS ready row over no-graph MLIR should fail CGRA status audit: "
                f"{failed_ready_no_graph.stderr}"
            )
        forged_ready_no_graph_generic_audit = out_dir / "forged-cmsis-ready-no-graph-generic-audit.json"
        run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_ready_no_graph_generic_audit),
                str(forged_ready_no_graph_csv),
            ],
            expect_success=False,
        )
        forged_ready_no_graph_generic_data = json.loads(forged_ready_no_graph_generic_audit.read_text())
        forged_ready_no_graph_generic_diagnostics = "\n".join(
            str(item) for item in forged_ready_no_graph_generic_data.get("diagnostics", [])
        )
        if "ready row requires dfg_mlir content graph_ids" not in forged_ready_no_graph_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CMSIS ready row over no-graph MLIR: "
                f"{forged_ready_no_graph_generic_data}"
            )
        forged_semantic_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_semantic = one_row(forged_semantic_rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
        forged_semantic["dfg_status"] = "pass"
        forged_semantic["blocking_prerequisite"] = ""
        forged_semantic_csv = out_dir / "forged-cmsis-semantic-cgra-status-summary.csv"
        forged_semantic_json = out_dir / "forged-cmsis-semantic-cgra-status-summary.json"
        write_rows(forged_semantic_csv, forged_semantic_rows)
        write_json_projection(forged_semantic_json, forged_semantic_csv, forged_semantic_rows)
        failed_forged_semantic = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_semantic_csv),
                "--json-input",
                str(forged_semantic_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "CMSIS DFG MLIR ready row" not in failed_forged_semantic.stderr:
            raise AssertionError(
                "forged CMSIS semantic status should fail CGRA status audit: "
                f"{failed_forged_semantic.stderr}"
            )
        forged_semantic_generic_audit = out_dir / "forged-cmsis-semantic-generic-audit.json"
        run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_semantic_generic_audit),
                str(forged_semantic_csv),
            ],
            expect_success=False,
        )
        forged_semantic_generic_data = json.loads(forged_semantic_generic_audit.read_text())
        forged_semantic_generic_diagnostics = "\n".join(
            str(item) for item in forged_semantic_generic_data.get("diagnostics", [])
        )
        if "CMSIS DFG MLIR ready row" not in forged_semantic_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CMSIS semantic status: "
                f"{forged_semantic_generic_data}"
            )
        forged_missing_blocker_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_missing_blocker = one_row(
            forged_missing_blocker_rows,
            "cmsis-dsp",
            "BasicMathFunctions/arm_offset_f32.c",
        )
        forged_missing_blocker.update(
            {
                "diagnostic_class": "bogus_structured_blocker",
                "blocking_prerequisite": "bogus_prerequisite",
                "diagnostic": "bogus CMSIS structured blocker",
            }
        )
        forged_missing_blocker_csv = out_dir / "forged-cmsis-missing-blocker-cgra-status-summary.csv"
        forged_missing_blocker_json = out_dir / "forged-cmsis-missing-blocker-cgra-status-summary.json"
        write_rows(forged_missing_blocker_csv, forged_missing_blocker_rows)
        write_json_projection(forged_missing_blocker_json, forged_missing_blocker_csv, forged_missing_blocker_rows)
        failed_missing_blocker = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_missing_blocker_csv),
                "--json-input",
                str(forged_missing_blocker_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "CMSIS row without DFG MLIR evidence must use cmsis_dfg_mlir_missing" not in failed_missing_blocker.stderr:
            raise AssertionError(
                "forged CMSIS missing-DFG blocker should fail CGRA status audit: "
                f"{failed_missing_blocker.stderr}"
            )
        forged_missing_blocker_generic = out_dir / "forged-cmsis-missing-blocker-generic-audit.json"
        failed_missing_blocker_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_missing_blocker_generic),
                str(forged_missing_blocker_csv),
            ],
            expect_success=False,
        )
        forged_missing_blocker_generic_data = json.loads(forged_missing_blocker_generic.read_text())
        forged_missing_blocker_generic_diagnostics = "\n".join(
            str(item) for item in forged_missing_blocker_generic_data.get("diagnostics", [])
        )
        if "CMSIS row without DFG MLIR evidence must use cmsis_dfg_mlir_missing" not in forged_missing_blocker_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CMSIS missing-DFG blocker: "
                f"stdout={failed_missing_blocker_generic.stdout} stderr={failed_missing_blocker_generic.stderr} "
                f"audit={forged_missing_blocker_generic_data}"
            )
        forged_missing_pass_rows = [dict(row) for row in cmsis_evidence_rows]
        forged_missing_pass = one_row(
            forged_missing_pass_rows,
            "cmsis-dsp",
            "BasicMathFunctions/arm_offset_f32.c",
        )
        forged_missing_case = forged_missing_pass["case"]
        forged_missing_artifact_dir = out_dir / "forged-cmsis-missing-pass-artifacts"
        final_outputs = ["i32:17"]
        final_memory_state = {"arg0": ["i32:17"]}
        forged_missing_artifacts = {
            "dfg_report": forged_missing_artifact_dir / "arm_offset_f32.dfg.report.json",
            "mapping_artifact": forged_missing_artifact_dir / "arm_offset_f32.mapping.json",
            "cgra_report": forged_missing_artifact_dir / "arm_offset_f32.cgra.report.json",
            "comparison_report": forged_missing_artifact_dir / "arm_offset_f32.sim-comparison-report.json",
        }
        write_json(
            forged_missing_artifacts["dfg_report"],
            {
                "schema_version": 1,
                "kind": "dfg_sim_report",
                "workload": forged_missing_case,
                "graph": "g_arm_offset_f32_forged",
                "status": "pass",
                "metric_definition": "fixture",
                "optimistic_cycles": 17,
                **dfg_cycle_fixture_fields(17),
                "final_outputs": final_outputs,
                "final_memory_state": final_memory_state,
            },
        )
        write_json(
            forged_missing_artifacts["mapping_artifact"],
            {
                "schema_version": 1,
                "kind": "pnr_mapping",
                "workload": forged_missing_case,
                "graph": "g_arm_offset_f32_forged",
                "hardware": "shared_reduction_adg",
                "mapping_id": "arm_offset_f32__forged",
                "status": "pass",
                "placed_records": 1,
                "routed_edges": 1,
                "unrouted_edges": 0,
                "unplaced_records": 0,
                "config_records": 0,
                "placements": [],
                "routes": [],
                "config_bitstream": [],
                "diagnostics": [],
            },
        )
        write_json(
            forged_missing_artifacts["cgra_report"],
            {
                "schema_version": 1,
                "kind": "cgra_sim_report",
                "workload": forged_missing_case,
                "hardware": "shared_reduction_adg",
                "mapping_id": "arm_offset_f32__forged",
                "status": "pass",
                "dfg_cycles": 17,
                "hardware_aware_cycles": 19,
                "performance_delta_cycles": 2,
                "difference_classification": "expected_hardware_constraint",
                "metric_definition": "fixture",
                "cycle_breakdown": [],
                "diagnostics": [],
                "final_outputs": final_outputs,
                "final_memory_state": final_memory_state,
                "functional_state_source": "carried_from_dfg_sim_report",
            },
        )
        write_json(
            forged_missing_artifacts["comparison_report"],
            {
                "schema_version": 1,
                "kind": "sim_comparison_report",
                "workload": forged_missing_case,
                "status": "pass",
                "functional_comparison_status": "pass",
                "memory_comparison_status": "pass",
                "performance_comparison_status": "pass",
                "diagnostics": [],
            },
        )
        forged_missing_pass.update(
            {
                "dfg_status": "pass",
                "mapping_status": "pass",
                "cgra_status": "pass",
                "comparison_status": "pass",
                "status": "pass",
                "diagnostic_class": "",
                "owner": "",
                "blocking_prerequisite": "",
                "diagnostic": "",
                "final_outputs_present": "true",
                "final_memory_state_present": "true",
            }
        )
        for artifact_column, fingerprint_column in (
            ("dfg_report", "dfg_report_fingerprint"),
            ("mapping_artifact", "mapping_artifact_fingerprint"),
            ("cgra_report", "cgra_report_fingerprint"),
            ("comparison_report", "comparison_report_fingerprint"),
        ):
            artifact_path = forged_missing_artifacts[artifact_column]
            forged_missing_pass[artifact_column] = str(artifact_path)
            forged_missing_pass[fingerprint_column] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        forged_missing_pass_csv = out_dir / "forged-cmsis-missing-pass-cgra-status-summary.csv"
        forged_missing_pass_json = out_dir / "forged-cmsis-missing-pass-cgra-status-summary.json"
        write_rows(forged_missing_pass_csv, forged_missing_pass_rows)
        write_json_projection(forged_missing_pass_json, forged_missing_pass_csv, forged_missing_pass_rows)
        failed_missing_pass = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_missing_pass_csv),
                "--json-input",
                str(forged_missing_pass_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "CMSIS pass row requires DFG MLIR evidence" not in failed_missing_pass.stderr:
            raise AssertionError(
                "forged CMSIS pass without DFG MLIR should fail CGRA status audit: "
                f"{failed_missing_pass.stderr}"
            )
        forged_missing_pass_generic = out_dir / "forged-cmsis-missing-pass-generic-audit.json"
        failed_missing_pass_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_missing_pass_generic),
                str(forged_missing_pass_csv),
            ],
            expect_success=False,
        )
        forged_missing_pass_generic_data = json.loads(forged_missing_pass_generic.read_text())
        forged_missing_pass_generic_diagnostics = "\n".join(
            str(item) for item in forged_missing_pass_generic_data.get("diagnostics", [])
        )
        if "CMSIS pass row requires DFG MLIR evidence" not in forged_missing_pass_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CMSIS pass without DFG MLIR: "
                f"stdout={failed_missing_pass_generic.stdout} stderr={failed_missing_pass_generic.stderr} "
                f"audit={forged_missing_pass_generic_data}"
            )
        forged_missing_pass_with_bad_dfg_rows = [dict(row) for row in forged_missing_pass_rows]
        forged_missing_pass_with_bad_dfg = one_row(
            forged_missing_pass_with_bad_dfg_rows,
            "cmsis-dsp",
            "BasicMathFunctions/arm_offset_f32.c",
        )
        forged_missing_pass_with_bad_dfg["dfg_mlir"] = str(out_dir / "missing-arm-offset-f32.dfg.mlir")
        forged_missing_pass_with_bad_dfg["dfg_mlir_fingerprint"] = "0" * 64
        forged_missing_pass_with_bad_dfg_csv = out_dir / "forged-cmsis-missing-pass-bad-dfg-cgra-status-summary.csv"
        forged_missing_pass_with_bad_dfg_json = out_dir / "forged-cmsis-missing-pass-bad-dfg-cgra-status-summary.json"
        write_rows(forged_missing_pass_with_bad_dfg_csv, forged_missing_pass_with_bad_dfg_rows)
        write_json_projection(forged_missing_pass_with_bad_dfg_json, forged_missing_pass_with_bad_dfg_csv, forged_missing_pass_with_bad_dfg_rows)
        failed_missing_pass_with_bad_dfg = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_missing_pass_with_bad_dfg_csv),
                "--json-input",
                str(forged_missing_pass_with_bad_dfg_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "dfg_mlir" not in failed_missing_pass_with_bad_dfg.stderr:
            raise AssertionError(
                "forged CMSIS pass with bad DFG MLIR should fail CGRA status audit: "
                f"{failed_missing_pass_with_bad_dfg.stderr}"
            )
        forged_missing_pass_with_bad_dfg_generic = out_dir / "forged-cmsis-missing-pass-bad-dfg-generic-audit.json"
        failed_missing_pass_with_bad_dfg_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_missing_pass_with_bad_dfg_generic),
                str(forged_missing_pass_with_bad_dfg_csv),
            ],
            expect_success=False,
        )
        forged_missing_pass_with_bad_dfg_generic_data = json.loads(forged_missing_pass_with_bad_dfg_generic.read_text())
        forged_missing_pass_with_bad_dfg_generic_diagnostics = "\n".join(
            str(item) for item in forged_missing_pass_with_bad_dfg_generic_data.get("diagnostics", [])
        )
        if "dfg_mlir" not in forged_missing_pass_with_bad_dfg_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CMSIS pass with bad DFG MLIR: "
                f"stdout={failed_missing_pass_with_bad_dfg_generic.stdout} "
                f"stderr={failed_missing_pass_with_bad_dfg_generic.stderr} "
                f"audit={forged_missing_pass_with_bad_dfg_generic_data}"
            )
        forged_missing_pass_status_rows = [dict(row) for row in forged_missing_pass_rows]
        forged_missing_pass_status = one_row(
            forged_missing_pass_status_rows,
            "cmsis-dsp",
            "BasicMathFunctions/arm_offset_f32.c",
        )
        forged_missing_pass_status["diagnostic_class"] = "missing_status"
        forged_missing_pass_status_csv = out_dir / "forged-cmsis-pass-missing-status-cgra-status-summary.csv"
        forged_missing_pass_status_json = out_dir / "forged-cmsis-pass-missing-status-cgra-status-summary.json"
        write_rows(forged_missing_pass_status_csv, forged_missing_pass_status_rows)
        write_json_projection(forged_missing_pass_status_json, forged_missing_pass_status_csv, forged_missing_pass_status_rows)
        failed_missing_pass_status = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_missing_pass_status_csv),
                "--json-input",
                str(forged_missing_pass_status_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "CMSIS row must not use missing_status" not in failed_missing_pass_status.stderr:
            raise AssertionError(
                "CMSIS pass row with missing_status diagnostic should fail CGRA status audit: "
                f"{failed_missing_pass_status.stderr}"
            )
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(cmsis_evidence_csv),
                "--json-input",
                str(cmsis_evidence_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
        )

        sim_evidence = out_dir / "sim-evidence"
        write_sim_evidence_case(sim_evidence, "vecsum", cgra_final_state=True)
        write_sim_evidence_case(sim_evidence, "axpy", cgra_final_state=False)
        write_sim_evidence_case(
            sim_evidence,
            "reduction",
            cgra_final_state=True,
            functional_state_source="component_cgra_sim_reports_carried_from_dfg_sim_reports",
        )
        write_sim_evidence_case(sim_evidence, "mean", cgra_final_state=True)
        write_sim_evidence_case(sim_evidence, "string_compare", cgra_final_state=False)
        write_json(
            sim_evidence / "mean.dfg.report.json",
            {
                "schema_version": 1,
                "kind": "dfg_sim_report",
                "workload": "mean",
                "graph": "g_mean_0",
                "status": "fail",
                "optimistic_cycles": 0,
                **dfg_cycle_fixture_fields(0),
                "final_outputs": [],
                "final_memory_state": {},
                "diagnostics": ["fixture DFG-sim failure"],
                "metric_definition": "fixture",
            },
        )
        promoted_csv = out_dir / "promoted-cgra-status-summary.csv"
        promoted_json = out_dir / "promoted-cgra-status-summary.json"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(promoted_csv),
                "--json-output",
                str(promoted_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--sim-evidence-dir",
                str(sim_evidence),
            ],
        )
        if list(sim_evidence.glob("*sim-comparison-report.json")):
            raise AssertionError("CGRA status summary must not write generated comparisons into the input evidence dir")
        promoted_rows = read_rows(promoted_csv)
        promoted_data = json.loads(promoted_json.read_text())
        promoted_counts = promoted_data.get("counts", {})
        app_counts = promoted_counts.get("app") if isinstance(promoted_counts, dict) else None
        if app_counts != {
            "total": APP_CASE_COUNT,
            "pass": 2,
            "fail": 1,
            "blocked": APP_CASE_COUNT - 2 - 1,
            "unsupported": 0,
            "missing_status": 0,
        }:
            raise AssertionError(f"unexpected promoted app counts: {app_counts}")
        vecsum = one_row(promoted_rows, "app", "vecsum")
        if vecsum["status"] != "pass":
            raise AssertionError(f"vecsum should be promoted to pass: {vecsum}")
        for column in ("dfg_status", "mapping_status", "cgra_status", "comparison_status"):
            if vecsum[column] != "pass":
                raise AssertionError(f"vecsum pass row should have {column}=pass: {vecsum}")
        if vecsum["final_outputs_present"] != "true" or vecsum["final_memory_state_present"] != "true":
            raise AssertionError(f"vecsum pass row should preserve final-state evidence: {vecsum}")
        promoted_string_compare = one_row(promoted_rows, "app", "string_compare")
        if (
            promoted_string_compare["status"] != "blocked"
            or promoted_string_compare["diagnostic_class"] != "sim_comparison_blocked"
        ):
            raise AssertionError(
                f"string_compare should consume non-pass comparison evidence: {promoted_string_compare}"
            )
        if promoted_string_compare["blocking_prerequisite"] != "sim_comparison_report":
            raise AssertionError(f"string_compare should block on comparison evidence: {promoted_string_compare}")
        for column in ("dfg_status", "mapping_status", "cgra_status"):
            if promoted_string_compare[column] != "pass":
                raise AssertionError(
                    f"string_compare should preserve earlier pass stage evidence: {promoted_string_compare}"
                )
        if promoted_string_compare["comparison_status"] != "blocked":
            raise AssertionError(
                f"string_compare should preserve blocked comparison status: {promoted_string_compare}"
            )
        for artifact_column, fingerprint_column in (
            ("dfg_report", "dfg_report_fingerprint"),
            ("mapping_artifact", "mapping_artifact_fingerprint"),
            ("cgra_report", "cgra_report_fingerprint"),
            ("comparison_report", "comparison_report_fingerprint"),
        ):
            assert_sha256_file(vecsum[artifact_column], vecsum[fingerprint_column], repo)
        axpy = one_row(promoted_rows, "app", "axpy")
        if axpy["status"] != "blocked" or axpy["comparison_status"] != "blocked":
            raise AssertionError(f"axpy should become row-specific blocked: {axpy}")
        if axpy["diagnostic_class"] != "sim_comparison_blocked":
            raise AssertionError(f"axpy should name comparison blocker: {axpy}")
        if axpy["blocking_prerequisite"] != "sim_comparison_report":
            raise AssertionError(f"axpy should block on comparison evidence: {axpy}")
        if not artifact_exists(axpy["comparison_report"], repo):
            raise AssertionError(f"axpy should have a structured comparison report: {axpy}")
        reduction = one_row(promoted_rows, "app", "reduction")
        if reduction["status"] != "pass" or reduction["comparison_status"] != "pass":
            raise AssertionError(f"aggregate final-state provenance should be accepted: {reduction}")
        mean = one_row(promoted_rows, "app", "mean")
        if mean["status"] != "fail" or mean["diagnostic_class"] != "dfg_report_failed":
            raise AssertionError(f"malformed DFG evidence should fail one row without aborting: {mean}")

        tampered_rows = [dict(row) for row in promoted_rows]
        tampered_axpy = one_row(tampered_rows, "app", "axpy")
        tampered_axpy.update(
            {
                "dfg_status": "pass",
                "mapping_status": "pass",
                "cgra_status": "pass",
                "comparison_status": "pass",
                "status": "pass",
                "final_outputs_present": "true",
                "final_memory_state_present": "true",
                "diagnostic_class": "",
                "owner": "",
                "blocking_prerequisite": "",
                "diagnostic": "",
            }
        )
        tampered_csv = out_dir / "tampered-existing-blocked-cgra-status-summary.csv"
        tampered_json = out_dir / "tampered-existing-blocked-cgra-status-summary.json"
        write_rows(tampered_csv, tampered_rows)
        write_json_projection(tampered_json, tampered_csv, tampered_rows)
        failed_tampered = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(tampered_csv),
                "--json-input",
                str(tampered_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "referenced comparison_report JSON status is not pass" not in failed_tampered.stderr:
            raise AssertionError(f"tampered pass should fail on referenced JSON content: {failed_tampered.stderr}")

        promoted_no_dfg_rows = [
            row for row in promoted_rows
            if row["suite"] == "app" and row["diagnostic_class"] == "app_dataflow_tier_missing"
        ]
        if len(promoted_no_dfg_rows) != APP_NO_DFG_TIER_COUNT:
            raise AssertionError(f"unexpected promoted app no-DFG rows: {promoted_no_dfg_rows}")
        if promoted_no_dfg_rows:
            forged_no_dfg_failure_rows = [dict(row) for row in promoted_rows]
            forged_no_dfg_failure = one_row(forged_no_dfg_failure_rows, "app", promoted_no_dfg_rows[0]["case"])
            forged_no_dfg_dfg = out_dir / "forged-no-dfg-failed-dfg.report.json"
            write_json(
                forged_no_dfg_dfg,
                {
                    "schema_version": 1,
                    "kind": "dfg_sim_report",
                    "workload": promoted_no_dfg_rows[0]["case"],
                    "status": "fail",
                    "optimistic_cycles": 0,
                    **dfg_cycle_fixture_fields(0),
                    "final_outputs": [],
                    "final_memory_state": {},
                    "diagnostics": ["fixture DFG-sim failure"],
                    "metric_definition": "fixture",
                },
            )
            forged_no_dfg_failure.update(
                {
                    "dfg_report": str(forged_no_dfg_dfg),
                    "dfg_report_fingerprint": hashlib.sha256(forged_no_dfg_dfg.read_bytes()).hexdigest(),
                    "dfg_status": "fail",
                    "status": "blocked",
                    "diagnostic_class": "dfg_report_failed",
                    "owner": "sim_report",
                    "blocking_prerequisite": "dfg_report",
                    "diagnostic": "fixture DFG-sim failure",
                }
            )
            forged_no_dfg_failure_csv = out_dir / "forged-no-dfg-failure-downgrade-cgra-status-summary.csv"
            forged_no_dfg_failure_json = out_dir / "forged-no-dfg-failure-downgrade-cgra-status-summary.json"
            write_rows(forged_no_dfg_failure_csv, forged_no_dfg_failure_rows)
            write_json_projection(forged_no_dfg_failure_json, forged_no_dfg_failure_csv, forged_no_dfg_failure_rows)
            failed_no_dfg_failure = run(
                repo,
                [
                    "bash",
                    "test/e2e/run_cgra_status_audit.sh",
                    "--input",
                    str(forged_no_dfg_failure_csv),
                    "--json-input",
                    str(forged_no_dfg_failure_json),
                    "--legacy-loombench-root",
                    str(legacy_root),
                ],
                expect_success=False,
            )
            if "evidenced app row without dfg tier failed stage requires status=fail" not in failed_no_dfg_failure.stderr:
                raise AssertionError(
                    "forged no-DFG failure downgrade should fail CGRA status audit: "
                    f"{failed_no_dfg_failure.stderr}"
                )

        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(promoted_csv),
                "--json-input",
                str(promoted_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
        )
        promoted_generic_audit = out_dir / "promoted-generic-audit.json"
        run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(promoted_generic_audit),
                str(promoted_csv),
            ],
        )

        stale_cgra = json.loads((sim_evidence / "vecsum.cgra.report.json").read_text())
        stale_cgra["hardware_aware_cycles"] = 5
        stale_cgra["performance_delta_cycles"] = -5
        write_json(sim_evidence / "vecsum.cgra.report.json", stale_cgra)
        stale_csv = out_dir / "stale-comparison-cgra-status-summary.csv"
        stale_json = out_dir / "stale-comparison-cgra-status-summary.json"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(stale_csv),
                "--json-output",
                str(stale_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--sim-evidence-dir",
                str(sim_evidence),
            ],
        )
        stale_vecsum = one_row(read_rows(stale_csv), "app", "vecsum")
        if stale_vecsum["status"] != "fail" or stale_vecsum["comparison_status"] != "fail":
            raise AssertionError(f"stale comparison reports must be regenerated from current inputs: {stale_vecsum}")
        stale_comparison_path = Path(stale_vecsum["comparison_report"])
        if not stale_comparison_path.is_absolute():
            stale_comparison_path = repo / stale_comparison_path
        stale_comparison = json.loads(stale_comparison_path.read_text())
        if stale_comparison.get("cgra_sim_cycles") != 5:
            raise AssertionError(f"comparison report should reflect the mutated CGRA input: {stale_comparison}")

        identity_mismatch_evidence = out_dir / "identity-mismatch-evidence"
        write_sim_evidence_case(
            identity_mismatch_evidence,
            "vecsum",
            cgra_final_state=True,
            workload_identity="axpy",
        )
        identity_mismatch_csv = out_dir / "identity-mismatch-cgra-status-summary.csv"
        identity_mismatch_json = out_dir / "identity-mismatch-cgra-status-summary.json"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(identity_mismatch_csv),
                "--json-output",
                str(identity_mismatch_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--sim-evidence-dir",
                str(identity_mismatch_evidence),
            ],
        )
        identity_mismatch_rows = read_rows(identity_mismatch_csv)
        identity_mismatch_vecsum = one_row(identity_mismatch_rows, "app", "vecsum")
        if (
            identity_mismatch_vecsum["status"] != "fail"
            or identity_mismatch_vecsum["diagnostic_class"] != "evidence_identity_mismatch"
        ):
            raise AssertionError(f"row case must be checked against referenced JSON workload: {identity_mismatch_vecsum}")

        forged_identity_rows = [dict(row) for row in identity_mismatch_rows]
        forged_identity = one_row(forged_identity_rows, "app", "vecsum")
        forged_identity.update(
            {
                "dfg_status": "pass",
                "mapping_status": "pass",
                "cgra_status": "pass",
                "comparison_status": "pass",
                "status": "pass",
                "final_outputs_present": "true",
                "final_memory_state_present": "true",
                "diagnostic_class": "",
                "owner": "",
                "blocking_prerequisite": "",
                "diagnostic": "",
            }
        )
        forged_identity_csv = out_dir / "forged-identity-cgra-status-summary.csv"
        forged_identity_json = out_dir / "forged-identity-cgra-status-summary.json"
        write_rows(forged_identity_csv, forged_identity_rows)
        write_json_projection(forged_identity_json, forged_identity_csv, forged_identity_rows)
        failed_identity_audit = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_identity_csv),
                "--json-input",
                str(forged_identity_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "workload identity" not in failed_identity_audit.stderr:
            raise AssertionError(f"forged pass should fail on JSON workload identity: {failed_identity_audit.stderr}")
        forged_identity_generic_audit = out_dir / "forged-identity-generic-audit.json"
        failed_identity_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(forged_identity_generic_audit),
                str(forged_identity_csv),
            ],
            expect_success=False,
        )
        forged_identity_generic_data = (
            json.loads(forged_identity_generic_audit.read_text())
            if forged_identity_generic_audit.is_file()
            else {}
        )
        forged_identity_generic_diagnostics = "\n".join(
            str(item) for item in forged_identity_generic_data.get("diagnostics", [])
        )
        if "workload identity" not in forged_identity_generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject forged CGRA status row identity: "
                f"stdout={failed_identity_generic.stdout} stderr={failed_identity_generic.stderr} "
                f"audit={forged_identity_generic_data}"
            )

        current_like = out_dir / "current-like-evidence"
        for case in CURRENT_SIM_CYCLE_CASES:
            if case in {"gemv", "matvec", "relu", "variance", "vecadd"}:
                write_component_only_evidence(current_like, case)
            else:
                write_sim_evidence_case(current_like, case, cgra_final_state=False)
        current_like_csv = out_dir / "current-like-cgra-status-summary.csv"
        current_like_json = out_dir / "current-like-cgra-status-summary.json"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(current_like_csv),
                "--json-output",
                str(current_like_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--sim-evidence-dir",
                str(current_like),
            ],
        )
        current_like_rows = read_rows(current_like_csv)
        current_like_counts = json.loads(current_like_json.read_text())["counts"]["app"]
        if current_like_counts != {
            "total": APP_CASE_COUNT,
            "pass": 0,
            "fail": 0,
            "blocked": APP_CASE_COUNT,
            "unsupported": 0,
            "missing_status": 0,
        }:
            raise AssertionError(
                f"current-like evidence should keep every app row non-missing: {current_like_counts}"
            )
        vecadd_like = one_row(current_like_rows, "app", "vecadd")
        if vecadd_like["diagnostic_class"] != "missing_aggregate_cgra_status_evidence":
            raise AssertionError(f"component-only vecadd should require aggregate artifacts: {vecadd_like}")
        axpy_like = one_row(current_like_rows, "app", "axpy")
        if axpy_like["diagnostic_class"] != "sim_comparison_blocked":
            raise AssertionError(f"single-slice axpy should block on final-state comparison: {axpy_like}")

        chain_style_evidence = out_dir / "chain-style-current-sim-cycle"
        chain_vecsum_dir = write_chain_style_sim_evidence_case(chain_style_evidence, "vecsum")
        chain_style_csv = out_dir / "chain-style-cgra-status-summary.csv"
        chain_style_json = out_dir / "chain-style-cgra-status-summary.json"
        chain_style_comparison_dir = out_dir / "chain-style-comparisons"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(chain_style_csv),
                "--json-output",
                str(chain_style_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--sim-evidence-dir",
                str(chain_style_evidence),
                "--comparison-output-dir",
                str(chain_style_comparison_dir),
            ],
        )
        chain_style_rows = read_rows(chain_style_csv)
        chain_style_counts = json.loads(chain_style_json.read_text())["counts"]["app"]
        expected_chain_style_counts = {
            "total": APP_CASE_COUNT,
            "pass": 1,
            "fail": 0,
            "blocked": APP_CASE_COUNT - 1,
            "unsupported": 0,
            "missing_status": 0,
        }
        if chain_style_counts != expected_chain_style_counts:
            raise AssertionError(f"chain-style evidence should promote one app row: {chain_style_counts}")
        chain_vecsum = one_row(chain_style_rows, "app", "vecsum")
        if (
            chain_vecsum["status"] != "pass"
            or chain_vecsum["diagnostic_class"] != "cgra_sim_pass"
            or chain_vecsum["dfg_status"] != "pass"
            or chain_vecsum["mapping_status"] != "pass"
            or chain_vecsum["cgra_status"] != "pass"
            or chain_vecsum["comparison_status"] != "pass"
            or chain_vecsum["final_outputs_present"] != "true"
            or chain_vecsum["final_memory_state_present"] != "true"
        ):
            raise AssertionError(f"chain-style vecsum should become a complete CGRA-sim pass row: {chain_vecsum}")
        expected_chain_paths = {
            "dfg_report": chain_vecsum_dir / "vecsum-dfg-sim-report.json",
            "mapping_artifact": chain_vecsum_dir / "pnr-mapping.json",
            "cgra_report": chain_vecsum_dir / "vecsum-cgra-sim-report.json",
            "comparison_report": chain_style_comparison_dir / "vecsum.sim-comparison-report.json",
        }
        for column, expected_path in expected_chain_paths.items():
            observed = Path(chain_vecsum[column])
            if not observed.is_absolute():
                observed = repo / observed
            if observed.resolve() != expected_path.resolve():
                raise AssertionError(f"chain-style vecsum {column} path mismatch: {chain_vecsum}")
            assert_sha256_file(chain_vecsum[column], chain_vecsum[f"{column}_fingerprint"], repo)
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(chain_style_csv),
                "--json-input",
                str(chain_style_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
        )

        direct_chain_csv = out_dir / "direct-chain-cgra-status-summary.csv"
        direct_chain_json = out_dir / "direct-chain-cgra-status-summary.json"
        run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_summary.sh",
                "--output",
                str(direct_chain_csv),
                "--json-output",
                str(direct_chain_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--sim-evidence-dir",
                str(chain_vecsum_dir),
                "--comparison-output-dir",
                str(out_dir / "direct-chain-comparisons"),
            ],
        )
        direct_chain_vecsum = one_row(read_rows(direct_chain_csv), "app", "vecsum")
        direct_chain_counts = json.loads(direct_chain_json.read_text())["counts"]["app"]
        if direct_chain_counts != expected_chain_style_counts:
            raise AssertionError(
                "direct chain output dir should not let root pnr-mapping.json poison unrelated rows: "
                f"{direct_chain_counts}"
            )
        if direct_chain_vecsum["status"] != "pass":
            raise AssertionError(f"direct chain output dir should promote vecsum: {direct_chain_vecsum}")
        observed_direct_mapping = Path(direct_chain_vecsum["mapping_artifact"])
        if not observed_direct_mapping.is_absolute():
            observed_direct_mapping = repo / observed_direct_mapping
        if observed_direct_mapping.resolve() != (chain_vecsum_dir / "pnr-mapping.json").resolve():
            raise AssertionError(f"direct chain vecsum should use root pnr-mapping.json: {direct_chain_vecsum}")

        fake_pass_rows = [dict(row) for row in rows]
        fake_pass = one_row(fake_pass_rows, "app", "vecsum")
        fake_pass.update(
            {
                "dfg_status": "pass",
                "mapping_status": "pass",
                "cgra_status": "pass",
                "comparison_status": "pass",
                "status": "pass",
                "diagnostic_class": "",
                "owner": "",
                "blocking_prerequisite": "",
                "diagnostic": "",
                "final_outputs_present": "false",
                "final_memory_state_present": "false",
            }
        )
        for artifact_column, fingerprint_column in (
            ("dfg_report", "dfg_report_fingerprint"),
            ("mapping_artifact", "mapping_artifact_fingerprint"),
            ("cgra_report", "cgra_report_fingerprint"),
            ("comparison_report", "comparison_report_fingerprint"),
        ):
            fake_pass[artifact_column] = str(out_dir / f"missing-{artifact_column}.json")
            fake_pass[fingerprint_column] = "not-a-sha256"
        fake_pass_csv = out_dir / "fake-pass-cgra-status-summary.csv"
        fake_pass_json = out_dir / "fake-pass-cgra-status-summary.json"
        write_rows(fake_pass_csv, fake_pass_rows)
        write_json_projection(fake_pass_json, fake_pass_csv, fake_pass_rows)
        failed_pass = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(fake_pass_csv),
                "--json-input",
                str(fake_pass_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "pass row artifact path does not exist" not in failed_pass.stderr:
            raise AssertionError(f"fake pass failure should name missing artifact evidence: {failed_pass.stderr}")
        generic_audit = out_dir / "fake-pass-generic-audit.json"
        failed_generic = run(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(generic_audit),
                str(fake_pass_csv),
            ],
            expect_success=False,
        )
        generic_data = json.loads(generic_audit.read_text()) if generic_audit.is_file() else {}
        generic_diagnostics = "\n".join(str(item) for item in generic_data.get("diagnostics", []))
        if "pass row artifact path does not exist" not in generic_diagnostics:
            raise AssertionError(
                "generic artifact audit should reject fake CGRA status pass rows: "
                f"stdout={failed_generic.stdout} stderr={failed_generic.stderr} audit={generic_data}"
            )

        diverged_json = out_dir / "diverged-cgra-status-summary.json"
        diverged_rows = [dict(row) for row in rows]
        one_row(diverged_rows, "app", "vecsum")["status"] = "fail"
        write_json_projection(diverged_json, csv_output, diverged_rows)
        failed_json = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(csv_output),
                "--json-input",
                str(diverged_json),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "CGRA status JSON row content does not match CSV row" not in failed_json.stderr:
            raise AssertionError(f"JSON divergence failure should name row content mismatch: {failed_json.stderr}")

        missing_row = out_dir / "missing-row-cgra-status-summary.csv"
        with csv_output.open(newline="") as handle:
            reader = csv.DictReader(handle)
            kept = [row for row in reader if not (row["suite"] == "loombench" and row["case"] == "breadth_first_search")]
        with missing_row.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(kept)
        failed = run(
            repo,
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(missing_row),
                "--json-input",
                str(json_output),
                "--legacy-loombench-root",
                str(legacy_root),
            ],
            expect_success=False,
        )
        if "row coverage mismatch" not in failed.stderr:
            raise AssertionError(f"audit failure should name row coverage mismatch: {failed.stderr}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
