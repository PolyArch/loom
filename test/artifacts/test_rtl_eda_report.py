#!/usr/bin/env python3
"""Regression test for RTL lint EDA report artifacts."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import artifact_test_common


REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "report_id",
    "capability_class",
    "rtl_manifest_identity",
    "tool_profile_id",
    "tool_name",
    "tool_version",
    "command_role",
    "checked_top_modules",
    "checked_source_files",
    "input_artifact_fingerprints",
    "source_file_fingerprints",
    "returncode",
    "diagnostic_records",
    "diagnostics",
    "status",
}


def prepare_manifest(repo: Path, out_dir: Path) -> Path:
    _, hardware = artifact_test_common.prepare_candidate_inputs(repo, out_dir)
    manifest = out_dir / "rtl-manifest.json"
    artifact_test_common.require_success(
        repo,
        [
            "bash",
            "test/rtl/run_rtl_manifest.sh",
            "--hardware-summary",
            str(hardware),
            "--output",
            str(manifest),
        ],
        "RTL manifest",
    )
    return manifest


def require_audit_pass(repo: Path, artifact: Path, output: Path, label: str) -> dict[str, object]:
    artifact_test_common.require_success(
        repo,
        [
            "python3",
            "test/e2e/audit_intermediate_artifacts.py",
            "--output",
            str(output),
            str(artifact),
        ],
        label,
    )
    return json.loads(output.read_text())


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-rtl-eda-") as tmp:
        out_dir = Path(tmp)
        manifest = prepare_manifest(repo, out_dir)

        blocked = out_dir / "rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                "definitely-missing-verilator",
                "--output",
                str(blocked),
            ],
            "blocked RTL EDA report",
        )
        blocked_data = json.loads(blocked.read_text())
        missing = REQUIRED_KEYS - set(blocked_data)
        if missing:
            raise AssertionError(f"RTL EDA report missing keys: {sorted(missing)}")
        if blocked_data.get("kind") != "eda_report":
            raise AssertionError(f"unexpected RTL EDA report kind: {blocked_data}")
        if blocked_data.get("capability_class") != "rtl_lint":
            raise AssertionError(f"unexpected RTL EDA capability class: {blocked_data}")
        if blocked_data.get("rtl_manifest_identity") != "rtl-manifest":
            raise AssertionError(f"unexpected RTL manifest identity: {blocked_data}")
        if blocked_data.get("status") != "blocked":
            raise AssertionError(f"missing lint tool should produce blocked report: {blocked_data}")
        if blocked_data.get("tool_name") != "definitely-missing-verilator":
            raise AssertionError(f"blocked report should preserve requested tool name: {blocked_data}")
        if blocked_data.get("input_artifact_fingerprints") != {
            "rtl-manifest": artifact_test_common.fingerprint(manifest)
        }:
            raise AssertionError(f"blocked report should fingerprint RTL manifest input: {blocked_data}")
        records = blocked_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "tool_unavailable"
            and record.get("component") == "rtl_eda_report"
            for record in records
        ):
            raise AssertionError(f"blocked report needs structured tool diagnostic: {blocked_data}")
        require_audit_pass(
            repo,
            blocked,
            out_dir / "blocked-rtl-eda-audit-summary.json",
            "blocked RTL EDA report audit",
        )

        stale = out_dir / "stale-rtl-eda-report.json"
        stale_data = dict(blocked_data)
        stale_data["input_artifact_fingerprints"] = {"rtl-manifest": "0" * 64}
        stale.write_text(json.dumps(stale_data, indent=2, sort_keys=True) + "\n")
        result = artifact_test_common.run_command(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(out_dir / "stale-rtl-eda-audit-summary.json"),
                str(stale),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("RTL EDA report with stale manifest fingerprint unexpectedly passed audit")

        verilator = shutil.which("verilator")
        if verilator is None:
            return 0

        passed = out_dir / "passing-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                verilator,
                "--output",
                str(passed),
            ],
            "passing RTL EDA report",
        )
        passed_data = json.loads(passed.read_text())
        if passed_data.get("status") != "pass":
            raise AssertionError(f"available Verilator should produce passing report: {passed_data}")
        if passed_data.get("tool_name") != "verilator" or not passed_data.get("tool_version"):
            raise AssertionError(f"passing report should identify Verilator: {passed_data}")
        if passed_data.get("checked_top_modules") != ["pe_2x2"]:
            raise AssertionError(f"unexpected checked top modules: {passed_data}")
        if passed_data.get("checked_source_files") != ["rtl/pe_2x2.sv"]:
            raise AssertionError(f"unexpected checked source files: {passed_data}")
        if passed_data.get("returncode") != 0:
            raise AssertionError(f"passing report should record zero return code: {passed_data}")
        expected_sources = {"rtl/pe_2x2.sv": artifact_test_common.fingerprint(out_dir / "rtl/pe_2x2.sv")}
        if passed_data.get("source_file_fingerprints") != expected_sources:
            raise AssertionError(f"passing report should fingerprint checked sources: {passed_data}")
        if passed_data.get("diagnostic_records") != [] or passed_data.get("diagnostics") != []:
            raise AssertionError(f"passing report should not carry diagnostics: {passed_data}")
        require_audit_pass(
            repo,
            passed,
            out_dir / "passing-rtl-eda-audit-summary.json",
            "passing RTL EDA report audit",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
