#!/usr/bin/env python3
"""Emit RTL lint EDA report artifacts from RTL manifests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


@dataclass(frozen=True)
class ToolResolution:
    name: str
    executable: str | None
    diagnostic_class: str
    diagnostic_message: str


def positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tool", default=os.environ.get("LOOM_RTL_LINT_TOOL", ""))
    parser.add_argument(
        "--timeout-seconds",
        type=positive_int,
        default=positive_int(os.environ.get("LOOM_RTL_EDA_TIMEOUT_SECONDS", "7200")),
        help="timeout for each EDA tool invocation",
    )
    return parser.parse_args(argv)


def artifact_id(path: Path | None) -> str:
    return intermediate_artifacts.artifact_id_for_path(path)


def diagnostic_records(messages: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index, (diagnostic_class, severity, message) in enumerate(messages, start=1):
        records.append(
            {
                "diagnostic_id": f"rtl-eda-report::{index}",
                "diagnostic_class": diagnostic_class,
                "component": "rtl_eda_report",
                "severity": severity,
                "message": message,
            }
        )
    return records


def source_records(manifest: dict[str, object]) -> list[dict[str, object]]:
    records = manifest.get("emitted_source_files")
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def source_paths(manifest_path: Path, manifest: dict[str, object]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for record in source_records(manifest):
        raw_path = record.get("path")
        language = record.get("language")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        if language not in {"systemverilog", "verilog"}:
            continue
        path = Path(raw_path)
        resolved = path if path.is_absolute() else manifest_path.parent / path
        paths.append((raw_path, resolved))
    return paths


def resolve_tool(raw_tool: str) -> ToolResolution:
    requested = raw_tool or "verilator"
    if Path(requested).is_absolute() or "/" in requested:
        path = Path(requested)
        if path.is_file() and os.access(path, os.X_OK):
            return ToolResolution(path.name, str(path), "", "")
        return ToolResolution(
            path.name,
            None,
            "tool_unavailable",
            f"RTL lint tool is unavailable or not executable: {path.name}",
        )
    resolved = shutil.which(requested)
    if resolved is None:
        return ToolResolution(
            requested,
            None,
            "tool_unavailable",
            f"RTL lint tool is unavailable: {requested}",
        )
    return ToolResolution(requested, resolved, "", "")


def tool_version(tool_path: str, timeout_seconds: int) -> tuple[str, str, str]:
    try:
        result = subprocess.run(
            [tool_path, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return "", f"RTL lint tool version probe timed out after {timeout_seconds}s", "tool_timeout"
    except OSError as exc:
        return "", f"RTL lint tool version probe failed: {exc}", "tool_activation_failed"
    text = result.stdout.strip() or result.stderr.strip()
    if result.returncode != 0:
        detail = text.splitlines()[0] if text else f"exit code {result.returncode}"
        return "", f"RTL lint tool version probe failed: {detail}", "tool_activation_failed"
    return (text.splitlines()[0] if text else "", "", "")


def base_report(
    manifest_path: Path,
    manifest: dict[str, object],
    tool_name: str,
    timeout_seconds: int,
) -> dict[str, object]:
    tops = manifest.get("top_level_modules")
    top_modules = [top for top in tops if isinstance(top, str) and top] if isinstance(tops, list) else []
    sources = source_paths(manifest_path, manifest)
    input_fingerprints = intermediate_artifacts.input_artifact_fingerprints([manifest_path])
    return {
        "schema_version": 1,
        "kind": "eda_report",
        "report_id": f"eda-report::rtl-lint::{artifact_id(manifest_path)}",
        "capability_class": "rtl_lint",
        "rtl_manifest_identity": artifact_id(manifest_path),
        "tool_profile_id": f"tool::{tool_name}::rtl_lint",
        "tool_name": tool_name,
        "tool_version": "",
        "command_role": "rtl lint",
        "command_timeout_seconds": timeout_seconds,
        "checked_top_modules": top_modules,
        "checked_source_files": [logical for logical, _ in sources],
        "input_artifact_fingerprints": input_fingerprints,
        "source_file_fingerprints": {
            logical: intermediate_artifacts.artifact_fingerprint(path)
            for logical, path in sources
            if path.is_file()
        },
        "returncode": None,
        "diagnostic_records": [],
        "diagnostics": [],
        "status": "blocked",
    }


def blocked_report(
    manifest_path: Path,
    manifest: dict[str, object],
    tool_name: str,
    diagnostic_class: str,
    message: str,
    timeout_seconds: int,
) -> dict[str, object]:
    report = base_report(manifest_path, manifest, tool_name, timeout_seconds)
    report["diagnostic_records"] = diagnostic_records([(diagnostic_class, "error", message)])
    report["diagnostics"] = [message]
    return report


def restrict_source_claims(report: dict[str, object], sources: list[tuple[str, Path]]) -> None:
    report["checked_source_files"] = [logical for logical, path in sources if path.is_file()]
    report["source_file_fingerprints"] = {
        logical: intermediate_artifacts.artifact_fingerprint(path)
        for logical, path in sources
        if path.is_file()
    }


def lint_once(
    tool_path: str,
    top_module: str,
    sources: list[tuple[str, Path]],
    timeout_seconds: int,
) -> tuple[int | None, str, str]:
    command = [tool_path, "--lint-only", "--sv"]
    if top_module:
        command.extend(["--top-module", top_module])
    command.extend(str(path) for _, path in sources)
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return None, f"RTL lint timed out after {timeout_seconds}s", "tool_timeout"
    except OSError as exc:
        return None, f"RTL lint execution failed: {exc}", "tool_execution_failed"
    detail = (result.stderr.strip() or result.stdout.strip() or "RTL lint failed").splitlines()[0]
    if top_module and result.returncode != 0:
        detail = f"{top_module}: {detail}"
    return result.returncode, detail, ""


def run_lint(
    report: dict[str, object],
    tool_path: str,
    source_paths: list[tuple[str, Path]],
    timeout_seconds: int,
) -> None:
    tops = report.get("checked_top_modules")
    lint_targets = [top for top in tops if isinstance(top, str) and top] if isinstance(tops, list) else []
    if not lint_targets:
        lint_targets = [""]
    failures: list[tuple[str, str, str]] = []
    returncodes: list[int] = []
    for top_module in lint_targets:
        returncode, detail, diagnostic_class = lint_once(
            tool_path,
            top_module,
            source_paths,
            timeout_seconds,
        )
        if returncode is None:
            report["returncode"] = None
            report["status"] = "blocked"
            report["diagnostic_records"] = diagnostic_records(
                [(diagnostic_class or "tool_execution_failed", "error", detail)]
            )
            report["diagnostics"] = [detail]
            return
        returncodes.append(returncode)
        if returncode != 0:
            failures.append(("rtl_lint_failed", "error", detail))
    report["returncode"] = next((code for code in returncodes if code != 0), 0)
    if not failures:
        report["status"] = "pass"
        report["diagnostic_records"] = []
        report["diagnostics"] = []
        return
    report["status"] = "fail"
    report["diagnostic_records"] = diagnostic_records(failures)
    report["diagnostics"] = [message for _, _, message in failures]


def load_manifest(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_report(manifest_path: Path, raw_tool: str, timeout_seconds: int) -> dict[str, object]:
    tool = resolve_tool(raw_tool)
    manifest = load_manifest(manifest_path)
    if manifest.get("kind") != "rtl_manifest":
        return blocked_report(
            manifest_path,
            manifest,
            tool.name,
            "rtl_manifest_invalid",
            "input is not an RTL manifest",
            timeout_seconds,
        )
    if manifest.get("status") != "pass":
        return blocked_report(
            manifest_path,
            manifest,
            tool.name,
            "rtl_manifest_not_passing",
            "RTL manifest is not passing",
            timeout_seconds,
        )
    sources = source_paths(manifest_path, manifest)
    if not sources:
        return blocked_report(
            manifest_path,
            manifest,
            tool.name,
            "rtl_source_missing",
            "RTL manifest has no SystemVerilog source files",
            timeout_seconds,
        )
    missing_sources = [logical for logical, path in sources if not path.is_file()]
    if missing_sources:
        report = blocked_report(
            manifest_path,
            manifest,
            tool.name,
            "rtl_source_missing",
            f"RTL source files are missing: {', '.join(missing_sources)}",
            timeout_seconds,
        )
        restrict_source_claims(report, sources)
        return report
    if tool.executable is None:
        return blocked_report(
            manifest_path,
            manifest,
            tool.name,
            tool.diagnostic_class,
            tool.diagnostic_message,
            timeout_seconds,
        )

    report = base_report(manifest_path, manifest, tool.name, timeout_seconds)
    version, version_error, version_error_class = tool_version(
        tool.executable,
        timeout_seconds,
    )
    if version_error:
        report["status"] = "blocked"
        report["diagnostic_records"] = diagnostic_records(
            [(version_error_class or "tool_activation_failed", "error", version_error)]
        )
        report["diagnostics"] = [version_error]
        return report
    report["tool_version"] = version
    run_lint(report, tool.executable, sources, timeout_seconds)
    return report


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(Path(args.manifest), args.tool, args.timeout_seconds)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
