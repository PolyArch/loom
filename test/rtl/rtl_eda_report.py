#!/usr/bin/env python3
"""Emit RTL lint EDA report artifacts from RTL manifests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
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


@dataclass(frozen=True)
class CapabilityConfig:
    capability_class: str
    command_role: str
    default_tool: str
    tool_env_var: str


CAPABILITIES = {
    "rtl_lint": CapabilityConfig(
        capability_class="rtl_lint",
        command_role="rtl lint",
        default_tool="verilator",
        tool_env_var="LOOM_RTL_LINT_TOOL",
    ),
    "rtl_sim": CapabilityConfig(
        capability_class="rtl_sim",
        command_role="rtl sim",
        default_tool="vcs",
        tool_env_var="LOOM_RTL_SIM_TOOL",
    ),
}

FIDELITY_BY_CAPABILITY = {
    "rtl_lint": "rtl_structural",
    "rtl_sim": "rtl_functional",
}


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
    parser.add_argument("--capability-class", choices=sorted(CAPABILITIES), default="rtl_lint")
    parser.add_argument("--tool")
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


def diagnostic_detail(stdout: str, stderr: str, fallback: str) -> str:
    lines = [line.strip() for line in (stderr + "\n" + stdout).splitlines() if line.strip()]
    low_value_fragments = (
        "grep: warning:",
        "egrep is obsolescent",
    )
    for line in lines:
        lowered = line.lower()
        if any(fragment in lowered for fragment in low_value_fragments):
            continue
        if (
            "cannot execute" in lowered
            or "error" in lowered
            or "failed" in lowered
            or "sigsegv" in lowered
            or "segmentation fault" in lowered
            or "unexpected termination" in lowered
        ):
            return line
    for line in lines:
        lowered = line.lower()
        if any(fragment in lowered for fragment in low_value_fragments):
            continue
        return line
    return fallback


def capability_config(capability_class: str) -> CapabilityConfig:
    return CAPABILITIES[capability_class]


def selected_tool(args: argparse.Namespace, config: CapabilityConfig) -> str:
    if args.tool:
        return args.tool
    return os.environ.get(config.tool_env_var, "")


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
        resolved = resolved.resolve()
        paths.append((raw_path, resolved))
    return paths


def resolve_tool(raw_tool: str, config: CapabilityConfig) -> ToolResolution:
    requested = raw_tool or config.default_tool
    label = "RTL lint" if config.capability_class == "rtl_lint" else "RTL sim"
    if Path(requested).is_absolute() or "/" in requested:
        path = Path(requested)
        if path.is_file() and os.access(path, os.X_OK):
            return ToolResolution(path.name, str(path), "", "")
        return ToolResolution(
            path.name,
            None,
            "tool_unavailable",
            f"{label} tool is unavailable or not executable: {path.name}",
        )
    resolved = shutil.which(requested)
    if resolved is None:
        return ToolResolution(
            requested,
            None,
            "tool_unavailable",
            f"{label} tool is unavailable: {requested}",
        )
    return ToolResolution(requested, resolved, "", "")


def run_version_probe(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError as exc:
        raise exc


def tool_version(
    tool_path: str,
    timeout_seconds: int,
    config: CapabilityConfig,
) -> tuple[str, str, str]:
    probes = [[tool_path, "--version"]]
    if config.capability_class == "rtl_sim":
        probes.append([tool_path, "-ID"])
    timeout_seen = False
    last_detail = ""
    try:
        for command in probes:
            result = run_version_probe(command, timeout_seconds)
            if result is None:
                timeout_seen = True
                continue
            text = result.stdout.strip() or result.stderr.strip()
            if result.returncode == 0:
                return (text.splitlines()[0] if text else "", "", "")
            last_detail = diagnostic_detail(
                result.stdout,
                result.stderr,
                f"exit code {result.returncode}",
            )
    except OSError as exc:
        return "", f"RTL EDA tool version probe failed: {exc}", "tool_activation_failed"
    if timeout_seen and not last_detail:
        return "", f"RTL EDA tool version probe timed out after {timeout_seconds}s", "tool_timeout"
    detail = last_detail or "no version probe succeeded"
    return "", f"RTL EDA tool version probe failed: {detail}", "tool_activation_failed"


def base_report(
    manifest_path: Path,
    manifest: dict[str, object],
    tool_name: str,
    timeout_seconds: int,
    config: CapabilityConfig,
) -> dict[str, object]:
    tops = manifest.get("top_level_modules")
    top_modules = [top for top in tops if isinstance(top, str) and top] if isinstance(tops, list) else []
    sources = source_paths(manifest_path, manifest)
    input_fingerprints = intermediate_artifacts.input_artifact_fingerprints([manifest_path])
    return {
        "schema_version": 1,
        "kind": "eda_report",
        "report_id": f"eda-report::{config.capability_class.replace('_', '-')}::{artifact_id(manifest_path)}",
        "capability_class": config.capability_class,
        "rtl_manifest_identity": artifact_id(manifest_path),
        "tool_profile_id": f"tool::{tool_name}::{config.capability_class}",
        "tool_name": tool_name,
        "tool_version": "",
        "fidelity_level": FIDELITY_BY_CAPABILITY[config.capability_class],
        "command_role": config.command_role,
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
    config: CapabilityConfig,
) -> dict[str, object]:
    report = base_report(manifest_path, manifest, tool_name, timeout_seconds, config)
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
    detail = diagnostic_detail(result.stdout, result.stderr, "RTL lint failed")
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


def interface_ports(manifest: dict[str, object], top_module: str) -> list[dict[str, str]]:
    interfaces = manifest.get("generated_interfaces")
    if not isinstance(interfaces, list):
        return []
    tops = manifest.get("top_level_modules")
    top_modules = [top for top in tops if isinstance(top, str) and top] if isinstance(tops, list) else []
    allow_unscoped_interface = len(top_modules) <= 1
    for interface in interfaces:
        if not isinstance(interface, dict):
            continue
        scoped_top = interface.get("top_level_module")
        if scoped_top != top_module and not (scoped_top is None and allow_unscoped_interface):
            continue
        ports = interface.get("ports")
        if not isinstance(ports, list):
            continue
        result: list[dict[str, str]] = []
        for port in ports:
            if not isinstance(port, dict):
                continue
            name = port.get("name")
            direction = port.get("direction")
            sv_type = port.get("systemverilog_type")
            if isinstance(name, str) and isinstance(direction, str) and isinstance(sv_type, str):
                result.append(
                    {
                        "name": name,
                        "direction": direction,
                        "systemverilog_type": sv_type,
                    }
                )
        return result
    return []


def sim_testbench_source(top_module: str, ports: list[dict[str, str]]) -> str:
    declarations = ["  logic clk;", "  logic rst_n;"]
    connections = ["    .clk(clk)", "    .rst_n(rst_n)"]
    initial_assignments = ["    clk = 1'b0;", "    rst_n = 1'b0;"]
    for port in ports:
        name = port["name"]
        if name in {"clk", "rst_n"}:
            continue
        sv_type = port["systemverilog_type"]
        declarations.append(f"  {sv_type} {name};")
        connections.append(f"    .{name}({name})")
        if port["direction"] == "input":
            initial_assignments.append(f"    {name} = '0;")
    rendered_connections = ",\n".join(connections)
    rendered_declarations = "\n".join(declarations)
    rendered_initial_assignments = "\n".join(initial_assignments)
    return (
        "`timescale 1ns/1ps\n"
        "module loom_rtl_smoke_tb;\n"
        f"{rendered_declarations}\n"
        f"  {top_module} dut(\n"
        f"{rendered_connections}\n"
        "  );\n"
        "  always #1 clk = ~clk;\n"
        "  initial begin\n"
        f"{rendered_initial_assignments}\n"
        "    #2 rst_n = 1'b1;\n"
        "    #8 $finish;\n"
        "  end\n"
        "endmodule\n"
    )


def run_command(
    command: list[str],
    timeout_seconds: int,
    *,
    cwd: Path | None = None,
) -> tuple[int | None, str, str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return None, f"RTL sim timed out after {timeout_seconds}s", "tool_timeout"
    except OSError as exc:
        return None, f"RTL sim execution failed: {exc}", "tool_execution_failed"
    detail = diagnostic_detail(result.stdout, result.stderr, "RTL sim failed")
    return result.returncode, detail, ""


def run_sim(
    report: dict[str, object],
    manifest: dict[str, object],
    tool_path: str,
    source_paths: list[tuple[str, Path]],
    output_path: Path,
    timeout_seconds: int,
) -> None:
    tops = report.get("checked_top_modules")
    sim_targets = [top for top in tops if isinstance(top, str) and top] if isinstance(tops, list) else []
    if not sim_targets:
        report["returncode"] = None
        report["status"] = "blocked"
        detail = "RTL sim requires at least one top module"
        report["diagnostic_records"] = diagnostic_records([("rtl_sim_input_missing", "error", detail)])
        report["diagnostics"] = [detail]
        return
    failures: list[tuple[str, str, str]] = []
    returncodes: list[int] = []
    with tempfile.TemporaryDirectory(prefix=f"{output_path.stem}-", dir=output_path.parent) as tmp:
        work_dir = Path(tmp)
        for top_module in sim_targets:
            tb_path = work_dir / f"{top_module}_smoke_tb.sv"
            tb_path.write_text(sim_testbench_source(top_module, interface_ports(manifest, top_module)))
            simv = work_dir / f"{top_module}_simv"
            compile_command = [
                tool_path,
                "-full64",
                "-sverilog",
                "-timescale=1ns/1ps",
                "-top",
                "loom_rtl_smoke_tb",
                "-o",
                str(simv),
                *(str(path) for _, path in source_paths),
                str(tb_path),
            ]
            returncode, detail, diagnostic_class = run_command(
                compile_command,
                timeout_seconds,
                cwd=work_dir,
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
                failures.append(("rtl_sim_compile_failed", "error", f"{top_module}: {detail}"))
                continue
            run_code, run_detail, run_class = run_command(
                [str(simv)],
                timeout_seconds,
                cwd=work_dir,
            )
            if run_code is None:
                report["returncode"] = None
                report["status"] = "blocked"
                report["diagnostic_records"] = diagnostic_records(
                    [(run_class or "tool_execution_failed", "error", run_detail)]
                )
                report["diagnostics"] = [run_detail]
                return
            returncodes.append(run_code)
            if run_code != 0:
                failures.append(("rtl_sim_run_failed", "error", f"{top_module}: {run_detail}"))
    report["returncode"] = next((code for code in returncodes if code != 0), 0)
    if failures:
        report["status"] = "fail"
        report["diagnostic_records"] = diagnostic_records(failures)
        report["diagnostics"] = [message for _, _, message in failures]
        return
    report["status"] = "pass"
    report["diagnostic_records"] = []
    report["diagnostics"] = []


def load_manifest(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_report(
    manifest_path: Path,
    raw_tool: str,
    timeout_seconds: int,
    config: CapabilityConfig,
    output_path: Path,
) -> dict[str, object]:
    tool = resolve_tool(raw_tool or config.default_tool, config)
    manifest = load_manifest(manifest_path)
    if manifest.get("kind") != "rtl_manifest":
        return blocked_report(
            manifest_path,
            manifest,
            tool.name,
            "rtl_manifest_invalid",
            "input is not an RTL manifest",
            timeout_seconds,
            config,
        )
    if manifest.get("status") != "pass":
        return blocked_report(
            manifest_path,
            manifest,
            tool.name,
            "rtl_manifest_not_passing",
            "RTL manifest is not passing",
            timeout_seconds,
            config,
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
            config,
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
            config,
        )
        restrict_source_claims(report, sources)
        return report
    profile_error = os.environ.get("LOOM_RTL_EDA_PROFILE_ERROR", "").strip()
    if profile_error:
        return blocked_report(
            manifest_path,
            manifest,
            tool.name,
            os.environ.get("LOOM_RTL_EDA_PROFILE_ERROR_CLASS", "tool_activation_failed"),
            profile_error,
            timeout_seconds,
            config,
        )
    if tool.executable is None:
        return blocked_report(
            manifest_path,
            manifest,
            tool.name,
            tool.diagnostic_class,
            tool.diagnostic_message,
            timeout_seconds,
            config,
        )

    report = base_report(manifest_path, manifest, tool.name, timeout_seconds, config)
    version, version_error, version_error_class = tool_version(
        tool.executable,
        timeout_seconds,
        config,
    )
    if version_error:
        report["status"] = "blocked"
        report["diagnostic_records"] = diagnostic_records(
            [(version_error_class or "tool_activation_failed", "error", version_error)]
        )
        report["diagnostics"] = [version_error]
        return report
    report["tool_version"] = version
    if config.capability_class == "rtl_lint":
        run_lint(report, tool.executable, sources, timeout_seconds)
    else:
        run_sim(report, manifest, tool.executable, sources, output_path, timeout_seconds)
    return report


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = capability_config(args.capability_class)
    report = build_report(
        Path(args.manifest),
        selected_tool(args, config),
        args.timeout_seconds,
        config,
        output,
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
