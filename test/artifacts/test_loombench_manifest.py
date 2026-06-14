#!/usr/bin/env python3
"""Regression test for LoomBench manifest evidence."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import artifact_test_common
import test_cgra_status_summary


MANIFEST_CSV_HEADER = [
    "case",
    "source_row",
    "software_root",
    "source_fingerprint",
    "main_source",
    "implementation_sources",
    "headers",
    "feature_tags",
    "import_state",
    "manifest_case",
    "oracle",
    "input_profile",
    "tier_states",
    "owner",
    "reason",
]


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


def write_case(root: Path, name: str, *, with_header: bool = True) -> None:
    case_dir = root / name
    case_dir.mkdir(parents=True)
    (case_dir / "main.cpp").write_text("int main() { return 0; }\n")
    (case_dir / f"{name}.cpp").write_text(f'#include "{name}.h"\n')
    if with_header:
        (case_dir / f"{name}.h").write_text("#pragma once\n")


def read_manifest_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames != MANIFEST_CSV_HEADER:
            raise AssertionError(f"unexpected LoomBench manifest CSV header: {reader.fieldnames}")
        return rows


def one_row(rows: list[dict[str, str]], suite: str, case: str) -> dict[str, str]:
    return test_cgra_status_summary.one_row(rows, suite, case)


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-loombench-manifest-") as tmp:
        temp_root = Path(tmp)
        legacy_root = temp_root / "legacy"
        write_case(legacy_root, "legacy_missing")
        write_case(legacy_root, "vecadd")
        write_case(legacy_root, "blocked_case", with_header=False)

        inventory = temp_root / "old-app-inventory.csv"
        import_status = temp_root / "app-import-status.csv"
        manifest_json = temp_root / "loombench-manifest.json"
        manifest_csv = temp_root / "loombench-manifest.csv"

        run(
            repo,
            [
                "python3",
                "test/app/old_app_corpus_inventory.py",
                "--source-root",
                str(legacy_root),
                "--output",
                str(inventory),
            ],
        )
        run(
            repo,
            [
                "python3",
                "test/app/app_import_status.py",
                "--inventory",
                str(inventory),
                "--manifest",
                "test/app/manifest.json",
                "--output",
                str(import_status),
            ],
        )
        run(
            repo,
            [
                "python3",
                "test/loombench/loombench_manifest.py",
                "--inventory",
                str(inventory),
                "--import-status",
                str(import_status),
                "--source-root",
                str(legacy_root),
                "--output",
                str(manifest_json),
                "--csv-output",
                str(manifest_csv),
            ],
        )

        manifest = json.loads(manifest_json.read_text())
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "loombench_manifest":
            raise AssertionError(f"unexpected manifest header: {manifest}")
        if manifest.get("csv_projection") != str(manifest_csv):
            raise AssertionError(f"manifest must name its CSV projection: {manifest}")
        manifest_rows = read_manifest_csv(manifest_csv)
        if [row["case"] for row in manifest_rows] != ["blocked_case", "legacy_missing", "vecadd"]:
            raise AssertionError(f"manifest rows should preserve inventory order: {manifest_rows}")
        by_case = {row["case"]: row for row in manifest_rows}
        if by_case["vecadd"]["import_state"] != "accepted" or by_case["vecadd"]["manifest_case"] != "vecadd":
            raise AssertionError(f"vecadd should be accepted into LoomBench manifest: {by_case['vecadd']}")
        if len(by_case["vecadd"]["source_fingerprint"]) != 64:
            raise AssertionError(f"vecadd should carry a source fingerprint: {by_case['vecadd']}")
        if by_case["legacy_missing"]["import_state"] != "deferred":
            raise AssertionError(f"legacy_missing should remain deferred: {by_case['legacy_missing']}")
        if by_case["blocked_case"]["import_state"] != "excluded":
            raise AssertionError(f"blocked inventory rows should be excluded: {by_case['blocked_case']}")

        evidence_dir = temp_root / "sim-evidence"
        test_cgra_status_summary.write_sim_evidence_case(
            evidence_dir,
            "vecadd",
            cgra_final_state=True,
        )
        csv_output = temp_root / "cgra-status-summary.csv"
        json_output = temp_root / "cgra-status-summary.json"
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
                "--loombench-manifest",
                str(manifest_json),
                "--sim-evidence-dir",
                str(evidence_dir),
                "--comparison-output-dir",
                str(temp_root / "comparisons"),
            ],
        )
        rows = test_cgra_status_summary.read_rows(csv_output)
        loombench_vecadd = one_row(rows, "loombench", "vecadd")
        if (
            loombench_vecadd["status"] != "blocked"
            or loombench_vecadd["diagnostic_class"] != "loombench_workload_identity_fingerprint_missing"
        ):
            raise AssertionError(f"accepted LoomBench vecadd should require fingerprint bridge: {loombench_vecadd}")
        legacy_missing = one_row(rows, "loombench", "legacy_missing")
        if legacy_missing["status"] != "blocked" or legacy_missing["diagnostic_class"] != "loombench_import_deferred":
            raise AssertionError(f"deferred LoomBench row should be structured blocked: {legacy_missing}")
        blocked_case = one_row(rows, "loombench", "blocked_case")
        if blocked_case["status"] != "unsupported" or blocked_case["diagnostic_class"] != "loombench_import_excluded":
            raise AssertionError(f"excluded LoomBench row should be structured unsupported: {blocked_case}")

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
                "--loombench-manifest",
                str(manifest_json),
            ],
        )

        forged_rows = [dict(row) for row in rows]
        forged_vecadd = one_row(forged_rows, "loombench", "vecadd")
        app_vecadd = one_row(forged_rows, "app", "vecadd")
        for key, value in app_vecadd.items():
            if key not in {"suite", "case", "source_row", "software_root"}:
                forged_vecadd[key] = value
        forged_csv = temp_root / "forged-loombench-pass.csv"
        forged_json = temp_root / "forged-loombench-pass.json"
        test_cgra_status_summary.write_rows(forged_csv, forged_rows)
        test_cgra_status_summary.write_json_projection(forged_json, forged_csv, forged_rows)
        forged_result = subprocess.run(
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_csv),
                "--json-input",
                str(forged_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--loombench-manifest",
                str(manifest_json),
            ],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if forged_result.returncode == 0:
            raise AssertionError("forged LoomBench name-only pass unexpectedly passed audit")
        if "cannot pass without explicit workload identity bridge" not in forged_result.stderr:
            raise AssertionError(f"forged pass diagnostic missing: {forged_result.stderr}")

        forged_deferred = [dict(row) for row in rows]
        deferred_row = one_row(forged_deferred, "loombench", "legacy_missing")
        deferred_row["status"] = "unsupported"
        deferred_row["diagnostic_class"] = "loombench_import_excluded"
        deferred_row["blocking_prerequisite"] = "legacy_source"
        deferred_row["diagnostic"] = "forged excluded state"
        forged_deferred_csv = temp_root / "forged-loombench-deferred.csv"
        forged_deferred_json = temp_root / "forged-loombench-deferred.json"
        test_cgra_status_summary.write_rows(forged_deferred_csv, forged_deferred)
        test_cgra_status_summary.write_json_projection(
            forged_deferred_json,
            forged_deferred_csv,
            forged_deferred,
        )
        forged_deferred_result = subprocess.run(
            [
                "bash",
                "test/e2e/run_cgra_status_audit.sh",
                "--input",
                str(forged_deferred_csv),
                "--json-input",
                str(forged_deferred_json),
                "--legacy-loombench-root",
                str(legacy_root),
                "--loombench-manifest",
                str(manifest_json),
            ],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if forged_deferred_result.returncode == 0:
            raise AssertionError("forged LoomBench deferred state unexpectedly passed audit")
        if "deferred row legacy_missing must stay blocked" not in forged_deferred_result.stderr:
            raise AssertionError(f"forged deferred diagnostic missing: {forged_deferred_result.stderr}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
