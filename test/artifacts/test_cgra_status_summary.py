#!/usr/bin/env python3
"""Regression test for row-complete CGRA status evidence."""

from __future__ import annotations

import csv
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
