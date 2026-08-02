#!/usr/bin/env python3
"""Run exact TechMapping coverage over a completed DFG semantic gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

from simulation_conformance import (  # noqa: E402
    MAX_OUTER_WORKERS,
    outer_worker_limit,
)


BUILTIN_PRESETS = ("small", "default", "large")
TARGET_PROFILE_UNSUPPORTED = "target-profile-unsupported"
MAX_PEAK_RESIDENT_BYTES = 1024 * 1024 * 1024
SMALL_GRAPH_MAX_ACTORS = 256
P95_GENERATION_CPU_TARGET_SECONDS = 0.5
MAX_GENERATION_WALL_SECONDS = 10.0
PROCESS_SETUP_ALLOWANCE_SECONDS = 5.0
DEFAULT_CASE_TIMEOUT_SECONDS = (
    len(BUILTIN_PRESETS) * MAX_GENERATION_WALL_SECONDS + PROCESS_SETUP_ALLOWANCE_SECONDS
)
DEFAULT_BUILTIN_TIMEOUT_SECONDS = 120.0


class CoverageError(ValueError):
    """Raised when an input or result cannot support an exact coverage claim."""


@dataclass(frozen=True)
class WorkloadInput:
    identity: str
    suite: str
    case: str
    graphs: int
    actors: int
    program: Path


@dataclass(frozen=True)
class SourceGate:
    summary_path: Path
    workloads: tuple[WorkloadInput, ...]
    unsupported: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    wall_seconds: float
    peak_resident_bytes: int
    timed_out: bool


def default_worker_count(cpu_count: int | None = None) -> int:
    return outer_worker_limit(
        cpu_count=cpu_count,
        memory_derived_limit=MAX_OUTER_WORKERS,
    )


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CoverageError(f"{field} must be an integer")
    if positive and value < 1:
        raise CoverageError(f"{field} must be positive")
    return value


def _identity(value: Any, field: str, digits: int = 64) -> str:
    if not isinstance(value, str) or len(value) != digits:
        raise CoverageError(f"{field} must contain {digits} lowercase hex digits")
    if any(character not in "0123456789abcdef" for character in value):
        raise CoverageError(f"{field} must contain {digits} lowercase hex digits")
    return value


def _component(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CoverageError(f"{field} must be a nonempty string")
    if Path(value).name != value or value in {".", ".."}:
        raise CoverageError(f"{field} is not one path component")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CoverageError(f"JSON root is not an object: {path}")
    return value


def load_source_gate(summary_path: Path) -> SourceGate:
    summary_path = summary_path.expanduser().resolve()
    summary = _load_json(summary_path)
    if summary.get("stage") != "dfg-sim":
        raise CoverageError("source summary is not a dfg-sim gate")
    if _integer(summary.get("failed"), "failed") != 0:
        raise CoverageError("source DFG semantic gate contains failed rows")
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise CoverageError("source summary cases must be an array")
    if _integer(summary.get("case_count"), "case_count") != len(cases):
        raise CoverageError("source summary case_count does not match cases")

    workloads: list[WorkloadInput] = []
    unsupported: list[dict[str, str]] = []
    seen: set[str] = set()
    for ordinal, case_value in enumerate(cases):
        if not isinstance(case_value, dict):
            raise CoverageError(f"source case {ordinal} is not an object")
        identity = case_value.get("identity")
        if not isinstance(identity, str) or not identity:
            raise CoverageError(f"source case {ordinal} has no identity")
        if identity in seen:
            raise CoverageError(f"duplicate source identity: {identity}")
        seen.add(identity)
        suite = _component(case_value.get("suite"), f"{identity}.suite")
        case = _component(case_value.get("case"), f"{identity}.case")
        status = case_value.get("status")
        if status == "pass":
            if case_value.get("category") is not None:
                raise CoverageError(f"passing source row has a category: {identity}")
            digest = identity.rpartition(":")[2]
            if not digest or any(c not in "0123456789abcdef" for c in digest):
                raise CoverageError(
                    f"source identity has no hex operator digest: {identity}"
                )
            program = summary_path.parent / suite / case / digest / "program.dfg.mlir"
            if not program.is_file():
                raise CoverageError(f"missing Canonical Dataflow program: {program}")
            workloads.append(
                WorkloadInput(
                    identity=identity,
                    suite=suite,
                    case=case,
                    graphs=_integer(
                        case_value.get("graphs"), f"{identity}.graphs", positive=True
                    ),
                    actors=_integer(
                        case_value.get("actors"), f"{identity}.actors", positive=True
                    ),
                    program=program,
                )
            )
            continue
        if status != "unsupported":
            raise CoverageError(f"source row did not terminate cleanly: {identity}")
        category = case_value.get("category")
        if category != TARGET_PROFILE_UNSUPPORTED:
            raise CoverageError(
                f"unsupported category is not exact target-profile-unsupported: "
                f"{identity} ({category})"
            )
        detail = case_value.get("detail")
        if not isinstance(detail, str) or not detail:
            raise CoverageError(f"unsupported source row has no detail: {identity}")
        unsupported.append(
            {
                "case": case,
                "category": category,
                "detail": detail,
                "identity": identity,
                "suite": suite,
            }
        )

    if _integer(summary.get("passed"), "passed") != len(workloads):
        raise CoverageError("source summary passed count does not match pass rows")
    if _integer(summary.get("unsupported"), "unsupported") != len(unsupported):
        raise CoverageError(
            "source summary unsupported count does not match unsupported rows"
        )
    categories = summary.get("unsupported_categories")
    if categories != {TARGET_PROFILE_UNSUPPORTED: len(unsupported)}:
        raise CoverageError("source unsupported category accounting is not exact")

    expected_programs = {row.program.resolve() for row in workloads}
    actual_programs = {
        path.resolve() for path in summary_path.parent.rglob("program.dfg.mlir")
    }
    missing = sorted(expected_programs - actual_programs)
    extra = sorted(actual_programs - expected_programs)
    if missing:
        raise CoverageError(f"missing Canonical Dataflow program: {missing[0]}")
    if extra:
        raise CoverageError(f"unexpected Canonical Dataflow program: {extra[0]}")
    return SourceGate(summary_path, tuple(workloads), tuple(unsupported))


def validate_tool_report(
    workload_identity: str,
    expected_graphs: int,
    expected_actors: int,
    report: dict[str, Any],
    root_identities: dict[str, str],
) -> dict[str, Any]:
    if tuple(root_identities) != BUILTIN_PRESETS:
        raise CoverageError("builtin root order is not small/default/large")
    root_to_preset: dict[str, str] = {}
    for preset, identity in root_identities.items():
        identity = _identity(identity, f"{preset} root identity")
        if identity in root_to_preset:
            raise CoverageError("builtin roots do not have distinct identities")
        root_to_preset[identity] = preset
    if report.get("kind") != "tech_mapping_coverage":
        raise CoverageError(f"invalid tool report kind for {workload_identity}")
    if _integer(report.get("graph_count"), "graph_count") != expected_graphs:
        raise CoverageError(f"graph count changed for {workload_identity}")
    if _integer(report.get("actor_count"), "actor_count") != expected_actors:
        raise CoverageError(f"actor count changed for {workload_identity}")
    canonical_dataflow = _identity(
        report.get("canonical_dataflow"), "canonical_dataflow"
    )
    fabrics = report.get("fabrics")
    if not isinstance(fabrics, list) or len(fabrics) != len(BUILTIN_PRESETS):
        raise CoverageError(
            f"tool report does not cover every builtin: {workload_identity}"
        )

    normalized_fabrics: dict[str, dict[str, Any]] = {}
    for fabric_value in fabrics:
        if not isinstance(fabric_value, dict):
            raise CoverageError(f"malformed Fabric report for {workload_identity}")
        input_root = _identity(
            fabric_value.get("input_fabric_root"), "input_fabric_root"
        )
        preset = root_to_preset.get(input_root)
        if preset is None:
            raise CoverageError(f"foreign builtin root for {workload_identity}")
        if preset in normalized_fabrics:
            raise CoverageError(f"duplicate {preset} result for {workload_identity}")
        for field in ("generation_cpu_seconds", "generation_wall_seconds"):
            duration = fabric_value.get(field)
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise CoverageError(f"invalid {field} for {workload_identity}")
        status = fabric_value.get("status")
        classification = fabric_value.get("classification")
        _identity(fabric_value.get("fabric"), "exact Fabric Module identity")
        if status == "generated":
            if classification != "pending-spatial-capacity":
                raise CoverageError(
                    f"generated result has wrong classification: {workload_identity}"
                )
            _integer(
                fabric_value.get("candidate_count"),
                "candidate_count",
                positive=True,
            )
            if (
                _integer(fabric_value.get("covered_actor_count"), "actor cover")
                != expected_actors
            ):
                raise CoverageError(f"incomplete actor cover for {workload_identity}")
        elif status == "proven-infeasible":
            if (
                classification != "capability-rejected"
                or fabric_value.get("reason") != "no-complete-exact-cover"
            ):
                raise CoverageError(
                    f"infeasible result lacks exact proof class: {workload_identity}"
                )
        elif status == "incomplete":
            if (
                classification != "incomplete"
                or fabric_value.get("reason") != "proof-not-established"
            ):
                raise CoverageError(
                    f"incomplete result lacks exact reason: {workload_identity}"
                )
        elif status in {"invalid", "internal"}:
            diagnostic = fabric_value.get("diagnostic")
            if (
                classification != status
                or not isinstance(diagnostic, str)
                or not diagnostic
            ):
                raise CoverageError(
                    f"{status} result lacks its diagnostic: {workload_identity}"
                )
        else:
            raise CoverageError(
                f"TechMapping did not terminate for {workload_identity}: {status}"
            )
        normalized_fabrics[preset] = dict(fabric_value)
    if tuple(normalized_fabrics) != BUILTIN_PRESETS:
        raise CoverageError(f"builtin result order drifted for {workload_identity}")
    return {
        "actors": expected_actors,
        "canonical_dataflow": canonical_dataflow,
        "fabrics": normalized_fabrics,
        "graphs": expected_graphs,
        "identity": workload_identity,
        "status": "reported",
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_results(
    *,
    expected_identities: Sequence[str],
    results: dict[str, dict[str, Any]],
    root_identities: dict[str, str],
) -> dict[str, Any]:
    expected = set(expected_identities)
    actual = set(results)
    anti_join = {
        "extra": sorted(actual - expected),
        "missing": sorted(expected - actual),
    }
    invalid = sorted(
        identity
        for identity in expected & actual
        if results[identity].get("status") != "reported"
    )
    builtins: dict[str, dict[str, Any]] = {}
    all_generated = not invalid and not anti_join["extra"] and not anti_join["missing"]
    for preset in BUILTIN_PRESETS:
        outcome_counts = {
            "capability_admitted": 0,
            "capability_rejected": 0,
            "incomplete": 0,
            "internal": 0,
            "invalid": 0,
        }
        cpu_timings: list[float] = []
        small_graph_cpu_timings: list[float] = []
        wall_timings: list[float] = []
        for identity in expected_identities:
            result = results.get(identity)
            if not result or result.get("status") != "reported":
                continue
            fabric = result["fabrics"][preset]
            cpu_timings.append(float(fabric["generation_cpu_seconds"]))
            if int(result["actors"]) <= SMALL_GRAPH_MAX_ACTORS:
                small_graph_cpu_timings.append(float(fabric["generation_cpu_seconds"]))
            wall_timings.append(float(fabric["generation_wall_seconds"]))
            if fabric["status"] == "generated":
                outcome_counts["capability_admitted"] += 1
            elif fabric["status"] == "proven-infeasible":
                outcome_counts["capability_rejected"] += 1
            else:
                outcome_counts[fabric["status"]] += 1
        all_generated = all_generated and outcome_counts["capability_admitted"] == len(
            expected_identities
        )
        builtins[preset] = {
            **outcome_counts,
            "generation_cpu_seconds": {
                "max": max(cpu_timings, default=0.0),
                "p50": _percentile(cpu_timings, 0.50),
                "p95": _percentile(cpu_timings, 0.95),
            },
            "generation_wall_seconds": {
                "max": max(wall_timings, default=0.0),
                "p50": _percentile(wall_timings, 0.50),
                "p95": _percentile(wall_timings, 0.95),
            },
            "input_fabric_root": root_identities[preset],
            "small_graph_generation_cpu_seconds": {
                "actor_limit": SMALL_GRAPH_MAX_ACTORS,
                "count": len(small_graph_cpu_timings),
                "max": max(small_graph_cpu_timings, default=0.0),
                "p50": _percentile(small_graph_cpu_timings, 0.50),
                "p95": _percentile(small_graph_cpu_timings, 0.95),
            },
        }
    peak_resident_bytes = max(
        (int(result.get("peak_resident_bytes", 0)) for result in results.values()),
        default=0,
    )
    p95_within_target = all(
        value["small_graph_generation_cpu_seconds"]["count"] > 0
        and value["small_graph_generation_cpu_seconds"]["p95"]
        <= P95_GENERATION_CPU_TARGET_SECONDS
        for value in builtins.values()
    )
    max_within_limit = all(
        value["generation_wall_seconds"]["max"] <= MAX_GENERATION_WALL_SECONDS
        for value in builtins.values()
    )
    performance_passed = (
        p95_within_target
        and max_within_limit
        and peak_resident_bytes <= MAX_PEAK_RESIDENT_BYTES
    )
    return {
        "anti_join": anti_join,
        "builtins": builtins,
        "expected": len(expected),
        "invalid": invalid,
        "reported": len(expected) - len(anti_join["missing"]) - len(invalid),
        "passed": all_generated and performance_passed,
        "performance": {
            "max_generation_wall_seconds": MAX_GENERATION_WALL_SECONDS,
            "max_peak_resident_bytes": MAX_PEAK_RESIDENT_BYTES,
            "p95_generation_cpu_target_seconds": (P95_GENERATION_CPU_TARGET_SECONDS),
            "small_graph_max_actors": SMALL_GRAPH_MAX_ACTORS,
            "passed": performance_passed,
            "peak_resident_bytes": peak_resident_bytes,
        },
    }


def _terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline:
        try:
            reaped, _, _ = os.wait4(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if reaped == pid:
            return
        time.sleep(0.01)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_process(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> ProcessOutcome:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        deadline = started + timeout_seconds
        timed_out = False
        status = 0
        usage = None
        while True:
            reaped, status, usage = os.wait4(process.pid, os.WNOHANG)
            if reaped == process.pid:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_process_group(process.pid)
                try:
                    _, status, usage = os.wait4(process.pid, 0)
                except ChildProcessError:
                    usage = None
                break
            time.sleep(0.01)
        process.returncode = os.waitstatus_to_exitcode(status)
    peak_resident_bytes = 0 if usage is None else int(usage.ru_maxrss) * 1024
    return ProcessOutcome(
        returncode=process.returncode,
        wall_seconds=time.monotonic() - started,
        peak_resident_bytes=peak_resident_bytes,
        timed_out=timed_out,
    )


def build_builtin_roots(
    *,
    loom_adg: Path,
    artifact_store: Path,
    output_root: Path,
    timeout_seconds: float,
) -> tuple[dict[str, str], dict[str, Path]]:
    roots: dict[str, str] = {}
    references: dict[str, Path] = {}
    hardware_root = output_root / "hardware"
    hardware_root.mkdir(parents=True, exist_ok=True)
    for preset in BUILTIN_PRESETS:
        output_base = hardware_root / preset
        reference_path = hardware_root / f"{preset}.ref"
        outcome = run_process(
            [
                str(loom_adg),
                f"--builtin={preset}",
                f"--artifact-store={artifact_store}",
                f"--output={output_base}",
            ],
            stdout_path=reference_path,
            stderr_path=hardware_root / f"{preset}.log",
            timeout_seconds=timeout_seconds,
        )
        if outcome.timed_out:
            raise CoverageError(f"{preset} builtin finalization timed out")
        if outcome.returncode != 0:
            raise CoverageError(f"{preset} builtin finalization failed")
        try:
            identity = reference_path.read_text().strip()
        except OSError as exc:
            raise CoverageError(f"cannot read {preset} Fabric identity: {exc}") from exc
        roots[preset] = _identity(identity, f"{preset} Fabric identity")
        for suffix in (".mlir", ".html"):
            output = output_base.with_suffix(suffix)
            if not output.is_file() or output.stat().st_size == 0:
                raise CoverageError(f"{preset} builtin export is missing: {output}")
        references[preset] = reference_path
    return roots, references


def run_workload(
    row: WorkloadInput,
    *,
    loom_tech_map: Path,
    artifact_store: Path,
    output_root: Path,
    reference_paths: dict[str, Path],
    root_identities: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    digest = row.identity.rpartition(":")[2]
    case_root = output_root / "cases" / row.suite / row.case / digest
    report_path = case_root / "tech-mapping.json"
    command = [
        str(loom_tech_map),
        f"--artifact-store={artifact_store}",
        *(
            f"--fabric-reference-file={reference_paths[preset]}"
            for preset in BUILTIN_PRESETS
        ),
        f"--report={report_path}",
        str(row.program),
    ]
    try:
        report_path.unlink(missing_ok=True)
        outcome = run_process(
            command,
            stdout_path=case_root / "stdout.log",
            stderr_path=case_root / "stderr.log",
            timeout_seconds=timeout_seconds,
        )
        if outcome.timed_out:
            raise CoverageError("TechMapping process timed out")
        if outcome.returncode not in {0, 2}:
            raise CoverageError(f"TechMapping process exited {outcome.returncode}")
        normalized = validate_tool_report(
            row.identity,
            row.graphs,
            row.actors,
            _load_json(report_path),
            root_identities,
        )
        normalized.update(
            {
                "peak_resident_bytes": outcome.peak_resident_bytes,
                "process_wall_seconds": outcome.wall_seconds,
                "report": str(report_path),
                "suite": row.suite,
                "case": row.case,
                "tool_returncode": outcome.returncode,
            }
        )
        return normalized
    except (CoverageError, OSError, subprocess.SubprocessError) as exc:
        return {
            "case": row.case,
            "detail": str(exc),
            "identity": row.identity,
            "status": "failed",
            "suite": row.suite,
        }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _executable(path: str, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise CoverageError(f"{role} is not executable: {resolved}")
    return resolved


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfg-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--loom-adg",
        default=os.environ.get(
            "LOOM_ADG", str(ROOT / "build" / "tools" / "loom-adg" / "loom-adg")
        ),
    )
    parser.add_argument(
        "--loom-tech-map",
        default=os.environ.get(
            "LOOM_TECH_MAP", str(ROOT / "build" / "bin" / "loom-tech-map")
        ),
    )
    parser.add_argument("--jobs", type=int, default=default_worker_count())
    parser.add_argument(
        "--case-timeout", type=float, default=DEFAULT_CASE_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--builtin-timeout", type=float, default=DEFAULT_BUILTIN_TIMEOUT_SECONDS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    try:
        if args.jobs < 1 or args.jobs > default_worker_count():
            raise CoverageError(
                f"jobs must be between 1 and {default_worker_count()} on this host"
            )
        if args.case_timeout <= 0 or args.builtin_timeout <= 0:
            raise CoverageError("timeouts must be positive")
        gate = load_source_gate(args.dfg_summary)
        loom_adg = _executable(args.loom_adg, "loom-adg")
        loom_tech_map = _executable(args.loom_tech_map, "loom-tech-map")
        output_root = args.out_dir.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        artifact_store = output_root / "store"
        artifact_store.mkdir(parents=True, exist_ok=True)
        roots, reference_paths = build_builtin_roots(
            loom_adg=loom_adg,
            artifact_store=artifact_store,
            output_root=output_root,
            timeout_seconds=args.builtin_timeout,
        )
    except (CoverageError, OSError) as exc:
        print(f"[tech-mapping-corpus] configuration error: {exc}", file=sys.stderr)
        return 2

    started = time.monotonic()
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_workload,
                row,
                loom_tech_map=loom_tech_map,
                artifact_store=artifact_store,
                output_root=output_root,
                reference_paths=reference_paths,
                root_identities=roots,
                timeout_seconds=args.case_timeout,
            ): row
            for row in gate.workloads
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            row = futures[future]
            try:
                results[row.identity] = future.result()
            except Exception as exc:  # preserve an honest row on worker failure
                results[row.identity] = {
                    "case": row.case,
                    "detail": f"worker raised {type(exc).__name__}: {exc}",
                    "identity": row.identity,
                    "status": "failed",
                    "suite": row.suite,
                }
            if completed % 25 == 0 or completed == len(gate.workloads):
                print(
                    f"[tech-mapping-corpus] completed {completed}/"
                    f"{len(gate.workloads)}",
                    file=sys.stderr,
                )

    aggregate = summarize_results(
        expected_identities=[row.identity for row in gate.workloads],
        results=results,
        root_identities=roots,
    )
    report = {
        "builtins": aggregate["builtins"],
        "case_count": len(gate.workloads) + len(gate.unsupported),
        "cases": [results[row.identity] for row in gate.workloads],
        "coverage": aggregate,
        "duration_seconds": time.monotonic() - started,
        "jobs": args.jobs,
        "kind": "tech_mapping_corpus_coverage",
        "source_summary": str(gate.summary_path),
        "target_profile_unsupported": list(gate.unsupported),
    }
    try:
        _write_json_atomic(output_root / "summary.json", report)
    except OSError as exc:
        print(f"[tech-mapping-corpus] cannot write summary: {exc}", file=sys.stderr)
        return 2
    print(
        f"[tech-mapping-corpus] compatible={len(gate.workloads)} "
        f"unsupported={len(gate.unsupported)} passed={aggregate['passed']} "
        f"duration={report['duration_seconds']:.3f}s",
        file=sys.stderr,
    )
    return 0 if aggregate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
