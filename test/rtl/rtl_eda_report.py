#!/usr/bin/env python3
"""Emit RTL lint EDA report artifacts from RTL manifests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tool", default="")
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


def tool_command(raw_tool: str) -> tuple[str, str | None]:
    requested = raw_tool or "verilator"
    if Path(requested).is_absolute() or "/" in requested:
        path = Path(requested)
        return path.name, str(path) if path.is_file() else None
    resolved = shutil.which(requested)
    return requested, resolved


def tool_version(tool_path: str) -> str:
    result = subprocess.run(
        [tool_path, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = result.stdout.strip() or result.stderr.strip()
    return text.splitlines()[0] if text else ""


def base_report(manifest_path: Path, manifest: dict[str, object], tool_name: str) -> dict[str, object]:
    tops = manifest.get("top_level_modules")
    top_modules = [top for top in tops if isinstance(top, str) and top] if isinstance(tops, list) else []
    sources = source_paths(manifest_path, manifest)
    input_fingerprints = {}
    if manifest_path.is_file():
        input_fingerprints[artifact_id(manifest_path)] = intermediate_artifacts.artifact_fingerprint(manifest_path)
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
) -> dict[str, object]:
    report = base_report(manifest_path, manifest, tool_name)
    report["diagnostic_records"] = diagnostic_records([(diagnostic_class, "error", message)])
    report["diagnostics"] = [message]
    return report


def run_lint(report: dict[str, object], tool_path: str, source_paths: list[tuple[str, Path]]) -> None:
    tops = report.get("checked_top_modules")
    top_module = tops[0] if isinstance(tops, list) and tops else ""
    command = [tool_path, "--lint-only", "--sv"]
    if isinstance(top_module, str) and top_module:
        command.extend(["--top-module", top_module])
    command.extend(str(path) for _, path in source_paths)
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    report["returncode"] = result.returncode
    if result.returncode == 0:
        report["status"] = "pass"
        report["diagnostic_records"] = []
        report["diagnostics"] = []
        return
    detail = (result.stderr.strip() or result.stdout.strip() or "RTL lint failed").splitlines()[0]
    report["status"] = "fail"
    report["diagnostic_records"] = diagnostic_records([("rtl_lint_failed", "error", detail)])
    report["diagnostics"] = [detail]


def load_manifest(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_report(manifest_path: Path, raw_tool: str) -> dict[str, object]:
    tool_name, resolved_tool = tool_command(raw_tool)
    manifest = load_manifest(manifest_path)
    if manifest.get("kind") != "rtl_manifest":
        return blocked_report(manifest_path, manifest, tool_name, "rtl_manifest_invalid", "input is not an RTL manifest")
    if manifest.get("status") != "pass":
        return blocked_report(
            manifest_path,
            manifest,
            tool_name,
            "rtl_manifest_not_passing",
            "RTL manifest is not passing",
        )
    sources = source_paths(manifest_path, manifest)
    if not sources:
        return blocked_report(
            manifest_path,
            manifest,
            tool_name,
            "rtl_source_missing",
            "RTL manifest has no SystemVerilog source files",
        )
    missing_sources = [logical for logical, path in sources if not path.is_file()]
    if missing_sources:
        return blocked_report(
            manifest_path,
            manifest,
            tool_name,
            "rtl_source_missing",
            f"RTL source files are missing: {', '.join(missing_sources)}",
        )
    if resolved_tool is None:
        return blocked_report(
            manifest_path,
            manifest,
            tool_name,
            "tool_unavailable",
            f"RTL lint tool is unavailable: {tool_name}",
        )

    report = base_report(manifest_path, manifest, tool_name)
    report["tool_version"] = tool_version(resolved_tool)
    run_lint(report, resolved_tool, sources)
    return report


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(Path(args.manifest), args.tool)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
