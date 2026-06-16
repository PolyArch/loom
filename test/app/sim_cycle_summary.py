#!/usr/bin/env python3
"""Emit simulator cycle summary rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "artifacts"))

import intermediate_artifacts  # noqa: E402


DEFAULT_SIM_CYCLE_CASES = (
    "vecsum",
    "dotproduct",
    "vecadd",
    "axpy",
    "byte_swap",
    "vecmul",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primitive-coverage")
    parser.add_argument("--dfg-report", action="append", default=[])
    parser.add_argument("--cgra-report", action="append", default=[])
    parser.add_argument(
        "--equivalence-groups",
        default=str(ROOT / "test/app/sim-cycle-equivalence-groups.json"),
        help="JSON file describing simulator cycle equivalence groups",
    )
    return parser.parse_args(argv)


def workloads_from_primitive_coverage(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return sorted({row["workload"] for row in csv.DictReader(handle) if row.get("workload")})


def tool_candidates() -> list[Path]:
    env_tool = os.environ.get("LOOM_SIM_CYCLE_SUMMARY")
    candidates = []
    if env_tool:
        candidates.append(Path(env_tool))
    candidates.extend(
        [
            ROOT / "build/tools/loom-sim-cycle-summary/loom-sim-cycle-summary",
            ROOT / "build/bin/loom-sim-cycle-summary",
        ]
    )
    return candidates


def find_existing_tool(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_tool() -> Path | None:
    return find_existing_tool(tool_candidates())


def classify_report(path: Path) -> str:
    filename_kind = report_kind_from_filename(path)
    if filename_kind is not None:
        return filename_kind
    with path.open() as handle:
        data = json.load(handle)
    kind = data.get("kind")
    if kind not in {"dfg_sim_report", "cgra_sim_report"}:
        raise ValueError(f"{path} has unsupported simulator report kind {kind!r}")
    return kind


def report_kind_from_filename(path: Path) -> str | None:
    name = path.name
    if name.endswith("-dfg-sim-report.json") or ".dfg." in name:
        return "dfg_sim_report"
    if name.endswith("-cgra-sim-report.json") or ".cgra." in name:
        return "cgra_sim_report"
    return None


def read_report(path: Path) -> dict[str, object]:
    try:
        with path.open() as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def workload_from_report_path(path: Path) -> str:
    name = path.name
    for suffix in ("-dfg-sim-report.json", "-cgra-sim-report.json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    for marker in (".dfg.", ".cgra."):
        if marker in name:
            return name.split(marker, 1)[0]
    if name.endswith(".report.json"):
        return name[: -len(".report.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def is_discovered_summary_report(path: Path) -> bool:
    name = path.name
    for suffix in (".dfg.report.json", ".cgra.report.json"):
        if name.endswith(suffix):
            prefix = name[: -len(suffix)]
            return "." not in prefix
    if name.endswith(".report.json"):
        return "." not in name[: -len(".report.json")]
    return True


def discover_report_inputs(evidence_dir: Path) -> tuple[list[Path], list[Path]]:
    dfg_reports: list[Path] = []
    cgra_reports: list[Path] = []
    if not evidence_dir.is_dir():
        return dfg_reports, cgra_reports
    reports = sorted(
        {
            *evidence_dir.glob("*.report.json"),
            *evidence_dir.glob("*-dfg-sim-report.json"),
            *evidence_dir.glob("*-cgra-sim-report.json"),
        }
    )
    for report in reports:
        if not is_discovered_summary_report(report):
            continue
        kind = classify_report(report)
        if kind == "dfg_sim_report":
            dfg_reports.append(report)
        else:
            cgra_reports.append(report)
    return dfg_reports, cgra_reports


def first_audit_diagnostic(audit: dict[str, object]) -> str:
    diagnostics = audit.get("diagnostics")
    if isinstance(diagnostics, list) and diagnostics:
        return str(diagnostics[0])
    return "artifact audit failed"


def write_blocked_discovered_evidence(
    output: Path,
    dfg_reports: list[Path],
    cgra_reports: list[Path],
    diagnostic: str,
) -> None:
    workloads: dict[str, dict[str, str]] = {}
    for report in dfg_reports:
        data = read_report(report)
        workload = data.get("workload")
        if not isinstance(workload, str) or not workload:
            workload = workload_from_report_path(report)
        row = workloads.setdefault(
            workload,
            {
                "kernel": workload,
                "dfg_sim_cycles": "",
                "cgra_sim_cycles": "",
                "status": "blocked",
                "diagnostic": diagnostic,
            },
        )
    for report in cgra_reports:
        data = read_report(report)
        workload = data.get("workload")
        if not isinstance(workload, str) or not workload:
            workload = workload_from_report_path(report)
        workloads.setdefault(
            workload,
            {
                "kernel": workload,
                "dfg_sim_cycles": "",
                "cgra_sim_cycles": "",
                "status": "blocked",
                "diagnostic": diagnostic,
            },
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows(
        "sim_cycle",
        output,
        [workloads[name] for name in sorted(workloads)],
    )


def load_cycle_equivalence_groups(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    groups = data.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError(f"{path} groups must be a list")
    parsed: list[dict[str, object]] = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError(f"{path} contains a non-object equivalence group")
        name = group.get("group")
        members = group.get("members")
        evidence = group.get("evidence")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path} equivalence group is missing group")
        if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
            raise ValueError(f"{path} equivalence group {name} has invalid members")
        if len(set(members)) < 2:
            raise ValueError(f"{path} equivalence group {name} needs at least two members")
        if not isinstance(evidence, str) or not evidence:
            raise ValueError(f"{path} equivalence group {name} is missing evidence")
        parsed.append({"group": name, "members": sorted(set(members)), "evidence": evidence})
    return parsed


def annotate_cycle_equivalence(output: Path, groups_path: Path) -> None:
    groups = load_cycle_equivalence_groups(groups_path)
    if not groups or not output.is_file():
        return
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return
    by_kernel = {row.get("kernel", ""): row for row in rows}
    annotated = False
    for group in groups:
        members = group["members"]
        assert isinstance(members, list)
        matching = [by_kernel.get(member) for member in members]
        if any(row is None or row.get("status") != "pass" for row in matching):
            continue
        dfg_values = {row.get("dfg_sim_cycles", "") for row in matching if row is not None}
        cgra_values = {row.get("cgra_sim_cycles", "") for row in matching if row is not None}
        if len(dfg_values) != 1 or len(cgra_values) != 1:
            continue
        if "" in dfg_values or "" in cgra_values:
            continue
        member_text = ";".join(members)
        for row in matching:
            assert row is not None
            row["cycle_equivalence_group"] = str(group["group"])
            row["cycle_equivalence_members"] = member_text
            row["cycle_equivalence_evidence"] = str(group["evidence"])
            annotated = True
    if not annotated:
        return
    for column in (
        "cycle_equivalence_group",
        "cycle_equivalence_members",
        "cycle_equivalence_evidence",
    ):
        if column not in fieldnames:
            fieldnames.append(column)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_discovered_report_inputs(output: Path, dfg_reports: list[Path], cgra_reports: list[Path]) -> bool:
    audit = intermediate_artifacts.audit([*dfg_reports, *cgra_reports])
    if audit.get("verdict") == "pass":
        return True
    diagnostic = (
        "discovered simulator evidence failed artifact audit: "
        + first_audit_diagnostic(audit)
    )
    write_blocked_discovered_evidence(output, dfg_reports, cgra_reports, diagnostic)
    return False


def discovered_reports_lack_dfg(output: Path, dfg_reports: list[Path], cgra_reports: list[Path]) -> bool:
    if dfg_reports or not cgra_reports:
        return False
    write_blocked_discovered_evidence(
        output,
        dfg_reports,
        cgra_reports,
        "discovered CGRA-sim report lacks matching DFG-sim report evidence",
    )
    return True


def write_blocked_default(output: Path, diagnostic: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_artifacts.write_csv_rows(
        "sim_cycle",
        output,
        [
            {
                "kernel": case,
                "dfg_sim_cycles": "",
                "cgra_sim_cycles": "",
                "status": "blocked",
                "diagnostic": diagnostic,
            }
            for case in DEFAULT_SIM_CYCLE_CASES
        ],
    )


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    return result


def command_failure_diagnostic(result: subprocess.CompletedProcess[str]) -> str:
    for stream in (result.stderr, result.stdout):
        for line in stream.splitlines():
            stripped = line.strip()
            if stripped:
                return f"default app CGRA evidence sweep failed: {stripped}"
    return f"default app CGRA evidence sweep failed with exit code {result.returncode}"


def summarize_reports(
    output: Path,
    dfg_reports: list[Path],
    cgra_reports: list[Path],
    equivalence_groups: Path,
) -> int:
    tool = find_tool()
    if tool is None:
        write_blocked_default(output, "missing loom-sim-cycle-summary tool")
        return 0
    command = [str(tool)]
    for report in dfg_reports:
        command.extend(["--dfg-report", str(report)])
    for report in cgra_reports:
        command.extend(["--cgra-report", str(report)])
    command.extend(["--output", str(output)])
    result = run_command(command)
    if result.returncode == 0:
        annotate_cycle_equivalence(output, equivalence_groups)
    return result.returncode


def emit_default_batch_summary(output: Path, equivalence_groups: Path) -> int:
    default_root = output.parent / f"{output.stem}-default-evidence"
    evidence_dir = default_root / "current-sim-cycle"
    shutil.rmtree(default_root, ignore_errors=True)
    default_root.mkdir(parents=True, exist_ok=True)
    command = [
        "bash",
        str(ROOT / "test/e2e/run_cgra_sim_evidence_sweep.sh"),
        "--output-dir",
        str(evidence_dir),
    ]
    for case in DEFAULT_SIM_CYCLE_CASES:
        command.extend(["--case", case])
    result = run_command(command)
    if result.returncode != 0:
        write_blocked_default(output, command_failure_diagnostic(result))
        return 0

    dfg_reports, cgra_reports = discover_report_inputs(evidence_dir)
    if not dfg_reports:
        write_blocked_default(output, "default app CGRA evidence sweep produced no DFG-sim reports")
        return 0
    if not audit_discovered_report_inputs(output, dfg_reports, cgra_reports):
        return 0
    if discovered_reports_lack_dfg(output, dfg_reports, cgra_reports):
        return 0
    return summarize_reports(output, dfg_reports, cgra_reports, equivalence_groups)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    equivalence_groups = Path(args.equivalence_groups)
    dfg_reports = [Path(path) for path in args.dfg_report]
    valid_dfg_reports = [path for path in dfg_reports if path.is_file()]
    cgra_reports = [Path(path) for path in args.cgra_report]
    valid_cgra_reports = [path for path in cgra_reports if path.is_file()]
    if not args.primitive_coverage and not valid_dfg_reports:
        discovered_dfg_reports, discovered_cgra_reports = discover_report_inputs(
            output.parent / "current-sim-cycle"
        )
        if discovered_dfg_reports or discovered_cgra_reports:
            if not audit_discovered_report_inputs(output, discovered_dfg_reports, discovered_cgra_reports):
                return 0
            if discovered_reports_lack_dfg(output, discovered_dfg_reports, discovered_cgra_reports):
                return 0
            tool = find_tool()
            if tool is not None:
                return summarize_reports(
                    output,
                    discovered_dfg_reports,
                    discovered_cgra_reports,
                    equivalence_groups,
                )
            intermediate_artifacts.write_csv("sim_cycle", output)
            return 0
        return emit_default_batch_summary(output, equivalence_groups)
    if valid_dfg_reports:
        tool = find_tool()
        if tool is not None:
            return summarize_reports(output, valid_dfg_reports, valid_cgra_reports, equivalence_groups)
        intermediate_artifacts.write_csv("sim_cycle", output)
        return 0
    primitive_path = Path(args.primitive_coverage)
    if not primitive_path.is_file():
        intermediate_artifacts.write_csv("sim_cycle", intermediate_artifacts.output_path(args.output))
        return 0

    tool = find_tool()
    if tool is not None:
        result = subprocess.run(
            [
                str(tool),
                "--primitive-coverage",
                str(primitive_path),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
        return result.returncode

    rows = [
        {
            "kernel": workload,
            "dfg_sim_cycles": "",
            "cgra_sim_cycles": "",
            "status": "blocked",
            "diagnostic": "DFG-sim and CGRA-sim cycle evidence is not available yet",
        }
        for workload in workloads_from_primitive_coverage(primitive_path)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        intermediate_artifacts.write_csv_rows("sim_cycle", output, rows)
    else:
        intermediate_artifacts.write_csv("sim_cycle", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
