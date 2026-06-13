#!/usr/bin/env python3
"""Regression test for row-complete CGRA status evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import artifact_test_common


HEADER = [
    "suite",
    "case",
    "source_row",
    "software_root",
    "graph_ids",
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
REQUIRED_LEGACY_CASE = "breadth_first_search"
CURRENT_SIM_CYCLE_CASES = [
    "axpy",
    "bit_reverse",
    "byte_swap",
    "compare_swap",
    "convolve_1d",
    "correlation",
    "conv1d",
    "cumsum",
    "dotproduct",
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


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


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
        "app": 109,
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
        if suite_counts.get("missing_status") != total:
            raise AssertionError(f"{suite} missing_status should equal total in baseline: {suite_counts}")
        for key in ("pass", "fail", "blocked", "unsupported"):
            if suite_counts.get(key) != 0:
                raise AssertionError(f"{suite} {key} should be zero in baseline: {suite_counts}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
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

        app_vecsum = one_row(rows, "app", "vecsum")
        if app_vecsum["status"] != "not_run" or app_vecsum["diagnostic_class"] != "missing_status":
            raise AssertionError(f"vecsum baseline must not claim pass: {app_vecsum}")
        if app_vecsum["blocking_prerequisite"] != "mapping_artifact":
            raise AssertionError(f"vecsum should be blocked on mapping artifact: {app_vecsum}")

        app_batchnorm = one_row(rows, "app", "batchnorm")
        if app_batchnorm["blocking_prerequisite"] != "dataflow":
            raise AssertionError(f"batchnorm should be blocked before CGRA mapping: {app_batchnorm}")

        cmsis_dsp = one_row(rows, "cmsis-dsp", "BasicMathFunctions/arm_add_q15.c")
        if cmsis_dsp["software_root"] != "externals/cmsis-dsp/Source":
            raise AssertionError(f"unexpected CMSIS-DSP root: {cmsis_dsp}")

        cmsis_nn = one_row(rows, "cmsis-nn", "ActivationFunctions/arm_relu_q15.c")
        if cmsis_nn["software_root"] != "externals/cmsis-nn/Source":
            raise AssertionError(f"unexpected CMSIS-NN root: {cmsis_nn}")

        loombench = one_row(rows, "loombench", "breadth_first_search")
        if loombench["blocking_prerequisite"] != "loombench_manifest":
            raise AssertionError(f"LoomBench legacy rows should require manifest reconciliation: {loombench}")

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
        if "not_run" not in vecsum_gaps[0]["reason"]:
            raise AssertionError(f"ledger row should preserve not_run status: {vecsum_gaps[0]}")

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
        (sim_evidence / "mean.dfg.report.json").write_text("{invalid-json\n")
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
            "total": 109,
            "pass": 2,
            "fail": 1,
            "blocked": 1,
            "unsupported": 0,
            "missing_status": 105,
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
            "total": 109,
            "pass": 0,
            "fail": 0,
            "blocked": len(CURRENT_SIM_CYCLE_CASES),
            "unsupported": 0,
            "missing_status": 109 - len(CURRENT_SIM_CYCLE_CASES),
        }:
            raise AssertionError(f"current-like evidence should produce 29 blocked app rows: {current_like_counts}")
        vecadd_like = one_row(current_like_rows, "app", "vecadd")
        if vecadd_like["diagnostic_class"] != "missing_aggregate_cgra_status_evidence":
            raise AssertionError(f"component-only vecadd should require aggregate artifacts: {vecadd_like}")
        axpy_like = one_row(current_like_rows, "app", "axpy")
        if axpy_like["diagnostic_class"] != "sim_comparison_blocked":
            raise AssertionError(f"single-slice axpy should block on final-state comparison: {axpy_like}")

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
        one_row(diverged_rows, "app", "vecsum")["status"] = "blocked"
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
