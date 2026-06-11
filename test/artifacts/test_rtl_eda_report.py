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

        non_executable_tool = out_dir / "not-executable-tool"
        non_executable_tool.write_text("#!/bin/sh\nexit 0\n")
        non_executable = out_dir / "non-executable-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(non_executable_tool),
                "--output",
                str(non_executable),
            ],
            "non-executable RTL lint tool report",
        )
        non_executable_data = json.loads(non_executable.read_text())
        if non_executable_data.get("status") != "blocked":
            raise AssertionError(f"non-executable tool should produce blocked report: {non_executable_data}")
        non_executable_records = non_executable_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "tool_unavailable"
            for record in non_executable_records
        ):
            raise AssertionError(
                f"non-executable tool should produce structured diagnostic: {non_executable_data}"
            )
        require_audit_pass(
            repo,
            non_executable,
            out_dir / "non-executable-rtl-eda-audit-summary.json",
            "non-executable RTL EDA report audit",
        )

        missing_source_manifest = out_dir / "missing-source-rtl-manifest.json"
        missing_source_data = json.loads(manifest.read_text())
        missing_source_data["emitted_source_files"][0]["path"] = "rtl/missing.sv"
        missing_source_manifest.write_text(json.dumps(missing_source_data, indent=2, sort_keys=True) + "\n")
        missing_source_report = out_dir / "missing-source-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(missing_source_manifest),
                "--tool",
                "definitely-missing-verilator",
                "--output",
                str(missing_source_report),
            ],
            "missing-source RTL EDA report",
        )
        missing_source_report_data = json.loads(missing_source_report.read_text())
        if missing_source_report_data.get("status") != "blocked":
            raise AssertionError(f"missing source should produce blocked report: {missing_source_report_data}")
        if missing_source_report_data.get("source_file_fingerprints") != {}:
            raise AssertionError(
                f"missing source should not claim source fingerprints: {missing_source_report_data}"
            )
        require_audit_pass(
            repo,
            missing_source_report,
            out_dir / "missing-source-rtl-eda-audit-summary.json",
            "missing-source RTL EDA report audit",
        )

        failing_version_tool = out_dir / "failing-version-tool"
        failing_version_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'activation failed: review smoke' >&2\n"
            "  exit 9\n"
            "fi\n"
            "exit 0\n"
        )
        failing_version_tool.chmod(failing_version_tool.stat().st_mode | 0o111)
        failing_version = out_dir / "failing-version-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(failing_version_tool),
                "--output",
                str(failing_version),
            ],
            "failing-version RTL EDA report",
        )
        failing_version_data = json.loads(failing_version.read_text())
        if failing_version_data.get("status") != "blocked":
            raise AssertionError(
                f"failing tool version probe should produce blocked report: {failing_version_data}"
            )
        failing_version_records = failing_version_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "tool_activation_failed"
            for record in failing_version_records
        ):
            raise AssertionError(
                f"failing tool version probe should produce activation diagnostic: {failing_version_data}"
            )
        require_audit_pass(
            repo,
            failing_version,
            out_dir / "failing-version-rtl-eda-audit-summary.json",
            "failing-version RTL EDA report audit",
        )

        env_tool = out_dir / "env-verilator"
        env_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test env'\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        )
        env_tool.chmod(env_tool.stat().st_mode | 0o111)
        env_selected = out_dir / "env-selected-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                f"LOOM_RTL_LINT_TOOL={env_tool}",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--output",
                str(env_selected),
            ],
            "environment-selected RTL lint tool report",
        )
        env_selected_data = json.loads(env_selected.read_text())
        if env_selected_data.get("status") != "pass":
            raise AssertionError(
                f"environment-selected RTL lint tool should pass: {env_selected_data}"
            )
        if (
            env_selected_data.get("tool_name") != "env-verilator"
            or env_selected_data.get("tool_version") != "Verilator 5.test env"
        ):
            raise AssertionError(
                f"environment-selected RTL lint tool was not recorded: {env_selected_data}"
            )
        require_audit_pass(
            repo,
            env_selected,
            out_dir / "env-selected-rtl-eda-audit-summary.json",
            "environment-selected RTL EDA report audit",
        )

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

        top_a = out_dir / "rtl/top_a.sv"
        top_b = out_dir / "rtl/top_b.sv"
        top_a.write_text(
            "`timescale 1ns/1ps\n"
            "module top_a(input logic clk);\n"
            "endmodule\n"
        )
        top_b.write_text(
            "`timescale 1ns/1ps\n"
            "module top_b(input logic clk);\n"
            "  missing_child u_missing();\n"
            "endmodule\n"
        )
        multi_top_manifest = out_dir / "multi-top-rtl-manifest.json"
        multi_top_data = json.loads(manifest.read_text())
        multi_top_data["manifest_id"] = "rtl-manifest::multi_top"
        multi_top_data["top_level_modules"] = ["top_a", "top_b"]
        multi_top_data["emitted_source_files"] = [
            {
                "path": "rtl/top_a.sv",
                "language": "systemverilog",
                "fingerprint": artifact_test_common.fingerprint(top_a),
            },
            {
                "path": "rtl/top_b.sv",
                "language": "systemverilog",
                "fingerprint": artifact_test_common.fingerprint(top_b),
            },
        ]
        multi_top_manifest.write_text(json.dumps(multi_top_data, indent=2, sort_keys=True) + "\n")
        multi_top_report = out_dir / "multi-top-rtl-eda-report.json"
        result = artifact_test_common.run_command(
            repo,
            [
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(multi_top_manifest),
                "--tool",
                verilator,
                "--output",
                str(multi_top_report),
            ],
        )
        if result.returncode == 0:
            raise AssertionError("multi-top RTL lint with failing second top unexpectedly passed")
        multi_top_report_data = json.loads(multi_top_report.read_text())
        if multi_top_report_data.get("status") != "fail":
            raise AssertionError(f"multi-top RTL lint should fail on second top: {multi_top_report_data}")
        if multi_top_report_data.get("returncode") == 0:
            raise AssertionError(f"multi-top RTL lint should record non-zero returncode: {multi_top_report_data}")
        require_audit_pass(
            repo,
            multi_top_report,
            out_dir / "multi-top-rtl-eda-audit-summary.json",
            "multi-top RTL EDA report audit",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
