#!/usr/bin/env python3
"""Regression test for RTL lint EDA report artifacts."""

from __future__ import annotations

import json
import shlex
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
    "fidelity_level",
    "command_role",
    "command_timeout_seconds",
    "checked_top_modules",
    "checked_source_files",
    "input_artifact_fingerprints",
    "source_file_fingerprints",
    "returncode",
    "eda_retry_count",
    "eda_parallel_jobs",
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


def rtl_eda_env(*argv: str) -> list[str]:
    return [
        "env",
        "-u",
        "LOOM_RTL_EDA_ENV_FILE",
        "-u",
        "LOOM_RTL_EDA_DEFAULT_ENV_FILE",
        "-u",
        "LOOM_RTL_EDA_PROFILE_ERROR",
        "-u",
        "LOOM_RTL_EDA_PROFILE_ERROR_CLASS",
        *argv,
    ]


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-rtl-eda-") as tmp:
        out_dir = Path(tmp)
        manifest = prepare_manifest(repo, out_dir)

        blocked = out_dir / "rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                "definitely-missing-verilator",
                "--output",
                str(blocked),
            ),
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
        if blocked_data.get("fidelity_level") != "rtl_structural":
            raise AssertionError(f"RTL lint report should declare structural fidelity: {blocked_data}")
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
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(non_executable_tool),
                "--output",
                str(non_executable),
            ),
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
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(missing_source_manifest),
                "--tool",
                "definitely-missing-verilator",
                "--output",
                str(missing_source_report),
            ),
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
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(failing_version_tool),
                "--output",
                str(failing_version),
            ),
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

        version_timeout_tool = out_dir / "version-timeout-tool"
        version_timeout_tool.write_text("#!/bin/sh\nsleep 5\n")
        version_timeout_tool.chmod(version_timeout_tool.stat().st_mode | 0o111)
        version_timeout_report = out_dir / "version-timeout-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(version_timeout_tool),
                "--timeout-seconds",
                "1",
                "--output",
                str(version_timeout_report),
            ),
            "version-timeout RTL EDA report",
        )
        version_timeout_data = json.loads(version_timeout_report.read_text())
        if version_timeout_data.get("status") != "blocked":
            raise AssertionError(
                f"timeout version probe should produce blocked report: {version_timeout_data}"
            )
        version_timeout_records = version_timeout_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "tool_timeout"
            for record in version_timeout_records
        ):
            raise AssertionError(
                f"timeout version probe should produce timeout diagnostic: {version_timeout_data}"
            )
        require_audit_pass(
            repo,
            version_timeout_report,
            out_dir / "version-timeout-rtl-eda-audit-summary.json",
            "version-timeout RTL EDA report audit",
        )

        timeout_tool = out_dir / "timeout-tool"
        timeout_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.timeout'\n"
            "  exit 0\n"
            "fi\n"
            "sleep 5\n"
        )
        timeout_tool.chmod(timeout_tool.stat().st_mode | 0o111)
        timeout_report = out_dir / "timeout-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(timeout_tool),
                "--timeout-seconds",
                "1",
                "--output",
                str(timeout_report),
            ),
            "timeout RTL EDA report",
        )
        timeout_data = json.loads(timeout_report.read_text())
        if timeout_data.get("status") != "blocked":
            raise AssertionError(f"timeout lint run should produce blocked report: {timeout_data}")
        if timeout_data.get("command_timeout_seconds") != 1:
            raise AssertionError(f"timeout report should record command timeout: {timeout_data}")
        timeout_records = timeout_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "tool_timeout"
            for record in timeout_records
        ):
            raise AssertionError(f"timeout lint run should produce timeout diagnostic: {timeout_data}")
        require_audit_pass(
            repo,
            timeout_report,
            out_dir / "timeout-rtl-eda-audit-summary.json",
            "timeout RTL EDA report audit",
        )

        failing_profile = out_dir / "failing-profile.sh"
        failing_profile.write_text("echo 'profile activation failed for test' >&2\nreturn 9\n")
        profile_failure_report = out_dir / "profile-failure-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                "-u",
                "LOOM_RTL_EDA_DEFAULT_ENV_FILE",
                f"LOOM_RTL_EDA_ENV_FILE={failing_profile}",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--output",
                str(profile_failure_report),
            ],
            "profile-failure RTL EDA report",
        )
        profile_failure_data = json.loads(profile_failure_report.read_text())
        if profile_failure_data.get("status") != "blocked":
            raise AssertionError(
                f"profile activation failure should produce blocked report: {profile_failure_data}"
            )
        profile_failure_records = profile_failure_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "tool_activation_failed"
            and "profile activation failed for test" in record.get("message", "")
            for record in profile_failure_records
        ):
            raise AssertionError(
                f"profile activation failure should produce structured diagnostic: {profile_failure_data}"
            )
        require_audit_pass(
            repo,
            profile_failure_report,
            out_dir / "profile-failure-rtl-eda-audit-summary.json",
            "profile-failure RTL EDA report audit",
        )

        nounset_profile = out_dir / "nounset-profile.sh"
        nounset_profile.write_text("set -u\necho \"${LOOM_RTL_EDA_TEST_UNSET}\"\n")
        nounset_profile_report = out_dir / "nounset-profile-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                "-u",
                "LOOM_RTL_EDA_TEST_UNSET",
                "-u",
                "LOOM_RTL_EDA_DEFAULT_ENV_FILE",
                f"LOOM_RTL_EDA_ENV_FILE={nounset_profile}",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--output",
                str(nounset_profile_report),
            ],
            "nounset-profile RTL EDA report",
        )
        nounset_profile_data = json.loads(nounset_profile_report.read_text())
        if nounset_profile_data.get("status") != "blocked":
            raise AssertionError(
                f"nounset profile failure should produce blocked report: {nounset_profile_data}"
            )
        nounset_records = nounset_profile_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "tool_activation_failed"
            and "LOOM_RTL_EDA_TEST_UNSET" in record.get("message", "")
            for record in nounset_records
        ):
            raise AssertionError(
                f"nounset profile failure should produce structured diagnostic: {nounset_profile_data}"
            )
        require_audit_pass(
            repo,
            nounset_profile_report,
            out_dir / "nounset-profile-rtl-eda-audit-summary.json",
            "nounset-profile RTL EDA report audit",
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
                "-u",
                "LOOM_RTL_EDA_ENV_FILE",
                "-u",
                "LOOM_RTL_EDA_DEFAULT_ENV_FILE",
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
        if env_selected_data.get("fidelity_level") != "rtl_structural":
            raise AssertionError(
                f"environment-selected RTL lint should record structural fidelity: {env_selected_data}"
            )
        require_audit_pass(
            repo,
            env_selected,
            out_dir / "env-selected-rtl-eda-audit-summary.json",
            "environment-selected RTL EDA report audit",
        )

        profile_tool = out_dir / "profile-verilator"
        profile_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test profile'\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        )
        profile_tool.chmod(profile_tool.stat().st_mode | 0o111)
        profile_env = out_dir / "rtl-eda-profile.sh"
        profile_env.write_text(
            f"export LOOM_RTL_LINT_TOOL={shlex.quote(str(profile_tool))}\n"
            "export LOOM_RTL_EDA_TIMEOUT_SECONDS=7\n"
        )
        profile_selected = out_dir / "profile-selected-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                "-u",
                "LOOM_RTL_LINT_TOOL",
                "-u",
                "LOOM_RTL_EDA_TIMEOUT_SECONDS",
                "-u",
                "LOOM_RTL_EDA_DEFAULT_ENV_FILE",
                f"LOOM_RTL_EDA_ENV_FILE={profile_env}",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--output",
                str(profile_selected),
            ],
            "profile-selected RTL lint tool report",
        )
        profile_selected_data = json.loads(profile_selected.read_text())
        if profile_selected_data.get("status") != "pass":
            raise AssertionError(
                f"profile-selected RTL lint tool should pass: {profile_selected_data}"
            )
        if (
            profile_selected_data.get("tool_name") != "profile-verilator"
            or profile_selected_data.get("tool_version") != "Verilator 5.test profile"
        ):
            raise AssertionError(
                f"profile-selected RTL lint tool was not recorded: {profile_selected_data}"
            )
        if profile_selected_data.get("command_timeout_seconds") != 7:
            raise AssertionError(
                f"profile-selected RTL lint report should record local timeout: {profile_selected_data}"
            )
        require_audit_pass(
            repo,
            profile_selected,
            out_dir / "profile-selected-rtl-eda-audit-summary.json",
            "profile-selected RTL EDA report audit",
        )

        profile_env_tool = out_dir / "profile-env-verilator"
        profile_env_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test profile env'\n"
            "  exit 0\n"
            "fi\n"
            "if [ \"${VCS_TARGET_ARCH:-}\" != \"linux64\" ]; then\n"
            "  echo 'missing VCS_TARGET_ARCH' >&2\n"
            "  exit 9\n"
            "fi\n"
            "if [ \"${VCSMX_HOME:-}\" != \"/fake/vcsmx\" ]; then\n"
            "  echo 'missing VCSMX_HOME' >&2\n"
            "  exit 9\n"
            "fi\n"
            "if [ \"${SNPSLMD_LICENSE_FILE:-}\" != \"forwarding-sentinel\" ]; then\n"
            "  echo 'missing SNPSLMD_LICENSE_FILE' >&2\n"
            "  exit 9\n"
            "fi\n"
            "exit 0\n"
        )
        profile_env_tool.chmod(profile_env_tool.stat().st_mode | 0o111)
        profile_env_file = out_dir / "rtl-eda-profile-env.sh"
        profile_env_file.write_text(
            f"export LOOM_RTL_LINT_TOOL={shlex.quote(str(profile_env_tool))}\n"
            "export VCS_TARGET_ARCH=linux64\n"
            "export VCSMX_HOME=/fake/vcsmx\n"
            "export SNPSLMD_LICENSE_FILE=forwarding-sentinel\n"
        )
        profile_env_selected = out_dir / "profile-env-selected-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                "-u",
                "LOOM_RTL_LINT_TOOL",
                "-u",
                "VCS_TARGET_ARCH",
                "-u",
                "VCSMX_HOME",
                "-u",
                "SNPSLMD_LICENSE_FILE",
                "-u",
                "LOOM_RTL_EDA_DEFAULT_ENV_FILE",
                f"LOOM_RTL_EDA_ENV_FILE={profile_env_file}",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--output",
                str(profile_env_selected),
            ],
            "profile-selected RTL lint tool environment report",
        )
        profile_env_selected_data = json.loads(profile_env_selected.read_text())
        if profile_env_selected_data.get("status") != "pass":
            raise AssertionError(
                f"profile-selected RTL lint tool environment should pass: {profile_env_selected_data}"
            )
        require_audit_pass(
            repo,
            profile_env_selected,
            out_dir / "profile-env-selected-rtl-eda-audit-summary.json",
            "profile-selected RTL EDA environment report audit",
        )

        retry_state = out_dir / "retry-state"
        retry_tool = out_dir / "retry-verilator"
        retry_tool.write_text(
            "#!/bin/sh\n"
            f"state={shlex.quote(str(retry_state))}\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test retry'\n"
            "  exit 0\n"
            "fi\n"
            "if [ ! -f \"$state\" ]; then\n"
            "  echo first > \"$state\"\n"
            "  echo 'Unable to checkout license feature VCSRuntime_Net' >&2\n"
            "  exit 17\n"
            "fi\n"
            "exit 0\n"
        )
        retry_tool.chmod(retry_tool.stat().st_mode | 0o111)
        retry_report = out_dir / "retry-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            rtl_eda_env(
                f"LOOM_RTL_EDA_RETRIES=1",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(retry_tool),
                "--output",
                str(retry_report),
            ),
            "retrying RTL lint tool report",
        )
        retry_data = json.loads(retry_report.read_text())
        if retry_data.get("status") != "pass":
            raise AssertionError(f"retrying license checkout should pass: {retry_data}")
        if retry_data.get("tool_version") != "Verilator 5.test retry":
            raise AssertionError(f"retrying tool version should be preserved: {retry_data}")
        if retry_data.get("eda_retry_count", 0) < 1:
            raise AssertionError(f"retrying report should record retry count: {retry_data}")
        require_audit_pass(
            repo,
            retry_report,
            out_dir / "retry-rtl-eda-audit-summary.json",
            "retrying RTL EDA report audit",
        )

        persistent_license_tool = out_dir / "persistent-license-verilator"
        persistent_license_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test persistent license'\n"
            "  exit 0\n"
            "fi\n"
            "echo 'Unable to checkout license feature VCSRuntime_Net' >&2\n"
            "exit 17\n"
        )
        persistent_license_tool.chmod(persistent_license_tool.stat().st_mode | 0o111)
        persistent_license_report = out_dir / "persistent-license-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            rtl_eda_env(
                "LOOM_RTL_EDA_RETRIES=1",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(persistent_license_tool),
                "--output",
                str(persistent_license_report),
            ),
            "persistent-license RTL lint tool report",
        )
        persistent_license_data = json.loads(persistent_license_report.read_text())
        if persistent_license_data.get("status") != "blocked":
            raise AssertionError(
                f"persistent license checkout failure should be blocked: {persistent_license_data}"
            )
        if persistent_license_data.get("returncode") is not None:
            raise AssertionError(
                f"persistent license checkout should not claim RTL returncode: {persistent_license_data}"
            )
        persistent_records = persistent_license_data.get("diagnostic_records", [])
        if not any(
            isinstance(record, dict)
            and record.get("diagnostic_class") == "tool_license_unavailable"
            for record in persistent_records
        ):
            raise AssertionError(
                f"persistent license checkout should produce license diagnostic: {persistent_license_data}"
            )
        if persistent_license_data.get("eda_retry_count") != 1:
            raise AssertionError(
                f"persistent license checkout should record retry count: {persistent_license_data}"
            )
        require_audit_pass(
            repo,
            persistent_license_report,
            out_dir / "persistent-license-rtl-eda-audit-summary.json",
            "persistent-license RTL EDA report audit",
        )

        non_license_counter = out_dir / "non-license-counter"
        non_license_tool = out_dir / "non-license-verilator"
        non_license_tool.write_text(
            "#!/bin/sh\n"
            f"counter={shlex.quote(str(non_license_counter))}\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test non-license'\n"
            "  exit 0\n"
            "fi\n"
            "count=0\n"
            "if [ -f \"$counter\" ]; then count=$(cat \"$counter\"); fi\n"
            "count=$((count + 1))\n"
            "echo \"$count\" > \"$counter\"\n"
            "echo 'unsupported design feature in test fixture' >&2\n"
            "exit 19\n"
        )
        non_license_tool.chmod(non_license_tool.stat().st_mode | 0o111)
        non_license_report = out_dir / "non-license-rtl-eda-report.json"
        non_license_result = artifact_test_common.run_command(
            repo,
            rtl_eda_env(
                "LOOM_RTL_EDA_RETRIES=2",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                str(non_license_tool),
                "--output",
                str(non_license_report),
            ),
        )
        if non_license_result.returncode == 0:
            raise AssertionError("non-license lint failure unexpectedly passed")
        non_license_data = json.loads(non_license_report.read_text())
        if non_license_data.get("status") != "fail":
            raise AssertionError(f"non-license lint failure should remain fail: {non_license_data}")
        if non_license_data.get("eda_retry_count") != 0:
            raise AssertionError(f"non-license lint failure should not retry: {non_license_data}")
        if non_license_counter.read_text().strip() != "1":
            raise AssertionError("non-license lint failure was retried")
        require_audit_pass(
            repo,
            non_license_report,
            out_dir / "non-license-rtl-eda-audit-summary.json",
            "non-license RTL EDA report audit",
        )

        equals_tool = out_dir / "equals-verilator"
        equals_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test equals'\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        )
        equals_tool.chmod(equals_tool.stat().st_mode | 0o111)
        equals_default_profile = out_dir / "equals-default-profile.sh"
        equals_default_profile.write_text("echo 'equals default profile should not load' >&2\nreturn 7\n")
        equals_selected = out_dir / "equals-selected-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                "-u",
                "LOOM_RTL_EDA_ENV_FILE",
                "-u",
                "LOOM_RTL_LINT_TOOL",
                "-u",
                "LOOM_RTL_EDA_TIMEOUT_SECONDS",
                f"LOOM_RTL_EDA_DEFAULT_ENV_FILE={equals_default_profile}",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                f"--tool={equals_tool}",
                "--output",
                str(equals_selected),
            ],
            "equals-selected RTL lint tool report",
        )
        equals_selected_data = json.loads(equals_selected.read_text())
        if equals_selected_data.get("status") != "pass":
            raise AssertionError(
                f"equals-form RTL lint tool should bypass default profile: {equals_selected_data}"
            )
        if equals_selected_data.get("tool_name") != "equals-verilator":
            raise AssertionError(
                f"equals-form RTL lint tool was not recorded: {equals_selected_data}"
            )
        require_audit_pass(
            repo,
            equals_selected,
            out_dir / "equals-selected-rtl-eda-audit-summary.json",
            "equals-selected RTL EDA report audit",
        )

        readonly_profile_tool = out_dir / "readonly-profile-verilator"
        readonly_profile_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ]; then\n"
            "  echo 'Verilator 5.test readonly-profile'\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n"
        )
        readonly_profile_tool.chmod(readonly_profile_tool.stat().st_mode | 0o111)
        readonly_profile = out_dir / "readonly-profile.sh"
        readonly_profile.write_text(
            f"export LOOM_RTL_LINT_TOOL={shlex.quote(str(readonly_profile_tool))}\n"
        )
        readonly_selected = out_dir / "readonly-profile-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            [
                "env",
                "-u",
                "LOOM_RTL_EDA_ENV_FILE",
                "-u",
                "LOOM_RTL_LINT_TOOL",
                f"LOOM_RTL_EDA_DEFAULT_ENV_FILE={readonly_profile}",
                "SHELLOPTS=braceexpand:hashall:interactive-comments",
                "BASHOPTS=checkwinsize:cmdhist",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--output",
                str(readonly_selected),
            ],
            "readonly-profile RTL lint tool report",
        )
        readonly_selected_data = json.loads(readonly_selected.read_text())
        if readonly_selected_data.get("status") != "pass":
            raise AssertionError(
                f"readonly inherited vars should not break profile import: {readonly_selected_data}"
            )
        if readonly_selected_data.get("tool_name") != "readonly-profile-verilator":
            raise AssertionError(
                f"readonly profile-selected RTL lint tool was not recorded: {readonly_selected_data}"
            )
        require_audit_pass(
            repo,
            readonly_selected,
            out_dir / "readonly-profile-rtl-eda-audit-summary.json",
            "readonly-profile RTL EDA report audit",
        )

        rtl_sim_tool = out_dir / "rtl-sim-tool"
        rtl_sim_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ] || [ \"$1\" = \"-ID\" ]; then\n"
            "  echo 'VCS X.test sim'\n"
            "  exit 0\n"
            "fi\n"
            "out=''\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  if [ \"$1\" = \"-o\" ]; then\n"
            "    shift\n"
            "    out=\"$1\"\n"
            "  fi\n"
            "  shift || break\n"
            "done\n"
            "if [ -n \"$out\" ]; then\n"
            "  printf '%s\\n' '#!/bin/sh' 'echo RTL sim smoke passed' 'exit 0' > \"$out\"\n"
            "  chmod +x \"$out\"\n"
            "fi\n"
            "exit 0\n"
        )
        rtl_sim_tool.chmod(rtl_sim_tool.stat().st_mode | 0o111)
        rtl_sim = out_dir / "rtl-sim-eda-report.json"
        artifact_test_common.require_success(
            repo,
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--capability-class",
                "rtl_sim",
                "--tool",
                str(rtl_sim_tool),
                "--output",
                str(rtl_sim),
            ),
            "passing RTL sim EDA report",
        )
        rtl_sim_data = json.loads(rtl_sim.read_text())
        if rtl_sim_data.get("status") != "pass":
            raise AssertionError(f"RTL sim smoke should produce passing report: {rtl_sim_data}")
        if rtl_sim_data.get("capability_class") != "rtl_sim":
            raise AssertionError(f"RTL sim report should declare rtl_sim capability: {rtl_sim_data}")
        if rtl_sim_data.get("command_role") != "rtl sim":
            raise AssertionError(f"RTL sim report should declare command role: {rtl_sim_data}")
        if rtl_sim_data.get("tool_version") != "VCS X.test sim":
            raise AssertionError(f"RTL sim report should record tool version: {rtl_sim_data}")
        if rtl_sim_data.get("returncode") != 0:
            raise AssertionError(f"RTL sim report should record zero return code: {rtl_sim_data}")
        require_audit_pass(
            repo,
            rtl_sim,
            out_dir / "rtl-sim-eda-audit-summary.json",
            "RTL sim EDA report audit",
        )

        sim_top_a = out_dir / "rtl/sim_top_a.sv"
        sim_top_b = out_dir / "rtl/sim_top_b.sv"
        sim_top_a.write_text(
            "`timescale 1ns/1ps\n"
            "module sim_top_a(input logic clk, input logic rst_n, input logic a);\n"
            "endmodule\n"
        )
        sim_top_b.write_text(
            "`timescale 1ns/1ps\n"
            "module sim_top_b(input logic clk, input logic rst_n);\n"
            "endmodule\n"
        )
        multi_sim_manifest = out_dir / "multi-sim-rtl-manifest.json"
        multi_sim_data = json.loads(manifest.read_text())
        multi_sim_data["manifest_id"] = "rtl-manifest::multi_sim"
        multi_sim_data["top_level_modules"] = ["sim_top_a", "sim_top_b"]
        multi_sim_data["emitted_source_files"] = [
            {
                "path": "rtl/sim_top_a.sv",
                "language": "systemverilog",
                "fingerprint": artifact_test_common.fingerprint(sim_top_a),
            },
            {
                "path": "rtl/sim_top_b.sv",
                "language": "systemverilog",
                "fingerprint": artifact_test_common.fingerprint(sim_top_b),
            },
        ]
        multi_sim_data["generated_interfaces"] = [
            {
                "interface_id": "interface::unscoped::scalar_bits_top_ports",
                "interface_kind": "scalar_bits_top_ports",
                "ports": [
                    {
                        "name": "a",
                        "direction": "input",
                        "fabric_type": "!fabric.bits<1>",
                        "systemverilog_type": "logic",
                    }
                ],
            }
        ]
        multi_sim_manifest.write_text(json.dumps(multi_sim_data, indent=2, sort_keys=True) + "\n")
        multi_sim_tool = out_dir / "multi-sim-tool"
        multi_sim_tool.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"--version\" ] || [ \"$1\" = \"-ID\" ]; then\n"
            "  echo 'VCS X.test sim'\n"
            "  exit 0\n"
            "fi\n"
            "out=''\n"
            "for arg in \"$@\"; do\n"
            "  case \"$arg\" in\n"
            "    *sim_top_b_smoke_tb.sv)\n"
            "      if grep -q '\\.a(a)' \"$arg\"; then\n"
            "        echo 'sim_top_b testbench reused unscoped interface' >&2\n"
            "        exit 19\n"
            "      fi\n"
            "      ;;\n"
            "  esac\n"
            "done\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  if [ \"$1\" = \"-o\" ]; then\n"
            "    shift\n"
            "    out=\"$1\"\n"
            "  fi\n"
            "  shift || break\n"
            "done\n"
            "if [ -n \"$out\" ]; then\n"
            "  printf '%s\\n' '#!/bin/sh' 'echo RTL sim smoke passed' 'exit 0' > \"$out\"\n"
            "  chmod +x \"$out\"\n"
            "fi\n"
            "exit 0\n"
        )
        multi_sim_tool.chmod(multi_sim_tool.stat().st_mode | 0o111)
        multi_sim = out_dir / "multi-sim-eda-report.json"
        artifact_test_common.require_success(
            repo,
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(multi_sim_manifest),
                "--capability-class",
                "rtl_sim",
                "--tool",
                str(multi_sim_tool),
                "--output",
                str(multi_sim),
            ),
            "multi-top RTL sim EDA report",
        )
        multi_sim_report = json.loads(multi_sim.read_text())
        if multi_sim_report.get("status") != "pass":
            raise AssertionError(f"multi-top RTL sim should not reuse unscoped interface: {multi_sim_report}")

        parallel_marker = out_dir / "parallel-sim-seen"
        parallel_lock = out_dir / "parallel-sim-lock"
        parallel_sim_tool = out_dir / "parallel-sim-tool"
        parallel_sim_tool.write_text(
            "#!/bin/sh\n"
            f"marker={shlex.quote(str(parallel_marker))}\n"
            f"lockdir={shlex.quote(str(parallel_lock))}\n"
            "if [ \"$1\" = \"--version\" ] || [ \"$1\" = \"-ID\" ]; then\n"
            "  echo 'VCS X.test parallel sim'\n"
            "  exit 0\n"
            "fi\n"
            "out=''\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  if [ \"$1\" = \"-o\" ]; then\n"
            "    shift\n"
            "    out=\"$1\"\n"
            "  fi\n"
            "  shift || break\n"
            "done\n"
            "if mkdir \"$lockdir\" 2>/dev/null; then\n"
            "  sleep 1\n"
            "  rmdir \"$lockdir\"\n"
            "else\n"
            "  echo parallel > \"$marker\"\n"
            "fi\n"
            "if [ -n \"$out\" ]; then\n"
            "  printf '%s\\n' '#!/bin/sh' 'echo RTL sim smoke passed' 'exit 0' > \"$out\"\n"
            "  chmod +x \"$out\"\n"
            "fi\n"
            "exit 0\n"
        )
        parallel_sim_tool.chmod(parallel_sim_tool.stat().st_mode | 0o111)
        parallel_sim = out_dir / "parallel-sim-eda-report.json"
        artifact_test_common.require_success(
            repo,
            rtl_eda_env(
                "LOOM_RTL_EDA_JOBS=2",
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(multi_sim_manifest),
                "--capability-class",
                "rtl_sim",
                "--tool",
                str(parallel_sim_tool),
                "--output",
                str(parallel_sim),
            ),
            "parallel multi-top RTL sim EDA report",
        )
        parallel_sim_report = json.loads(parallel_sim.read_text())
        if parallel_sim_report.get("status") != "pass":
            raise AssertionError(f"parallel multi-top RTL sim should pass: {parallel_sim_report}")
        if not parallel_marker.is_file():
            raise AssertionError("multi-top RTL sim did not run independent top simulations concurrently")
        if parallel_sim_report.get("eda_parallel_jobs") != 2:
            raise AssertionError(f"parallel RTL sim report should record worker budget: {parallel_sim_report}")

        verilator = shutil.which("verilator")
        if verilator is None:
            return 0

        passed = out_dir / "passing-rtl-eda-report.json"
        artifact_test_common.require_success(
            repo,
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(manifest),
                "--tool",
                verilator,
                "--output",
                str(passed),
            ),
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
            rtl_eda_env(
                "bash",
                "test/rtl/run_rtl_eda_report.sh",
                "--manifest",
                str(multi_top_manifest),
                "--tool",
                verilator,
                "--output",
                str(multi_top_report),
            ),
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
