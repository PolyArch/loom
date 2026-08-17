#!/usr/bin/env python3
"""Run and validate Loom's reproducible semantic baseline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import corpus_gate  # noqa: E402
import corpus_inventory  # noqa: E402
import corpus_target_profile  # noqa: E402
from corpus_simulation_report import parse_dse_execution_projection  # noqa: E402


EXPECTED_WORKLOAD_COUNT = 892
CORPUS_WALL_LIMIT_SECONDS = 600.0
CASE_WALL_LIMIT_SECONDS = 30.0
DFG_WALL_LIMIT_SECONDS = 15.0
REALTIME_LAYOUT_MARKERS = ("forceSimulation", "dagre.layout", "elk.layout")
ARTIFACT_FIELDS = {
    "canonical_dataflow",
    "simulation_runtime_input",
    "simulation_workload",
}
HARDWARE_ANCHORS = (
    "regular-topology",
    "irregular-directed-topology",
    "heterogeneous-multi-acc-core",
    "temporal-resource-grant",
    "memory-service-forwarding",
)


class BaselineError(RuntimeError):
    """Raised when an execution does not satisfy the semantic baseline."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise BaselineError(f"{context} must be an object")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BaselineError(f"{context} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError(f"{context} must be a finite number")
    result = float(value)
    if not result < float("inf") or result < minimum:
        raise BaselineError(f"{context} must be finite and >= {minimum}")
    return result


def _artifact_identity(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaselineError(f"{context} is not a canonical artifact identity")
    return value


def _validate_dfg_projection(value: object, identity: str) -> Mapping[str, object]:
    projection = _mapping(value, f"{identity} DFG projection")
    artifacts = _mapping(projection.get("artifacts"), f"{identity} artifacts")
    if set(artifacts) != ARTIFACT_FIELDS:
        raise BaselineError(f"{identity} has incomplete artifact identities")
    for field in sorted(ARTIFACT_FIELDS):
        _artifact_identity(artifacts[field], f"{identity} {field}")
    if projection.get("execution_terminal") != "retired":
        raise BaselineError(f"{identity} did not retire")
    try:
        parse_dse_execution_projection(projection.get("dse_execution"))
    except ValueError as exc:
        raise BaselineError(f"{identity} has invalid DSE execution: {exc}") from exc
    for field in ("dynamic_calls", "event_count", "wavefront_steps"):
        _integer(projection.get(field), f"{identity} {field}", minimum=1)
    value_lanes = _integer(
        projection.get("value_lanes_compared"),
        f"{identity} value_lanes_compared",
    )
    memory_bytes = _integer(
        projection.get("memory_bytes_compared"),
        f"{identity} memory_bytes_compared",
    )
    if value_lanes == 0 and memory_bytes == 0:
        raise BaselineError(f"{identity} has no compared observation")
    firings = _mapping(projection.get("operation_firings"), f"{identity} firings")
    if not firings:
        raise BaselineError(f"{identity} has no operation firings")
    for operation, count in firings.items():
        if not isinstance(operation, str) or not operation:
            raise BaselineError(f"{identity} has an invalid operation firing key")
        _integer(count, f"{identity} firing {operation}", minimum=1)
    simulation_seconds = _finite_number(
        projection.get("simulation_seconds"),
        f"{identity} simulation_seconds",
        minimum=0.0,
    )
    if simulation_seconds <= 0.0 or simulation_seconds > DFG_WALL_LIMIT_SECONDS:
        raise BaselineError(f"{identity} exceeded the DFG simulation wall budget")
    return projection


def validate_corpus_summary(
    summary: Mapping[str, object],
    workloads: Sequence[corpus_inventory.ProgramWorkload],
) -> dict[str, object]:
    """Validate one complete exact-target DFG conformance summary."""
    if len(workloads) != EXPECTED_WORKLOAD_COUNT:
        raise BaselineError(
            f"workload manifest contains {len(workloads)}, expected "
            f"{EXPECTED_WORKLOAD_COUNT}"
        )
    if summary.get("stage") != "dfg-sim":
        raise BaselineError("corpus summary is not a DFG simulation gate")
    target = _mapping(summary.get("target"), "corpus target")
    if target.get("triple") != corpus_gate.TARGET_TRIPLE:
        raise BaselineError("corpus summary has the wrong exact target triple")
    if summary.get("candidate_jobs") != 1:
        raise BaselineError("corpus summary used nested candidate parallelism")
    jobs = _integer(summary.get("jobs"), "corpus jobs", minimum=1)
    if jobs > corpus_gate.default_jobs():
        raise BaselineError("corpus jobs exceed the host worker cap")
    if summary.get("case_timeout_seconds") != CASE_WALL_LIMIT_SECONDS:
        raise BaselineError("corpus case wall budget changed")
    if summary.get("dfg_simulation_timeout_seconds") != DFG_WALL_LIMIT_SECONDS:
        raise BaselineError("corpus DFG wall budget changed")
    if _finite_number(summary.get("duration_seconds"), "corpus wall time") > (
        CORPUS_WALL_LIMIT_SECONDS
    ):
        raise BaselineError("corpus execution exceeded the ten-minute gate")
    _finite_number(summary.get("cpu_seconds"), "corpus CPU time")
    _integer(summary.get("peak_resident_bytes"), "corpus peak RSS")

    raw_cases = summary.get("cases")
    if not isinstance(raw_cases, list):
        raise BaselineError("corpus cases must be an array")
    expected_identities = [workload.identity for workload in workloads]
    actual_identities = [
        _mapping(case, "corpus case").get("identity") for case in raw_cases
    ]
    if actual_identities != expected_identities:
        raise BaselineError("corpus case identity order differs from the manifest")

    passed = 0
    unsupported = 0
    unsupported_rows: list[dict[str, str]] = []
    suite_counts: dict[str, dict[str, int]] = {}
    for workload, raw_case in zip(workloads, raw_cases, strict=True):
        case = _mapping(raw_case, workload.identity)
        if case.get("suite") != workload.suite or case.get("case") != workload.case:
            raise BaselineError(f"{workload.identity} changed suite or case identity")
        _finite_number(case.get("duration_seconds"), f"{workload.identity} wall time")
        if float(case["duration_seconds"]) > CASE_WALL_LIMIT_SECONDS:
            raise BaselineError(f"{workload.identity} exceeded its case wall budget")
        _finite_number(case.get("cpu_seconds"), f"{workload.identity} CPU time")
        _integer(case.get("peak_resident_bytes"), f"{workload.identity} peak RSS")

        profile = corpus_target_profile.resolve_target_profile(
            workload.suite,
            workload.target_profile,
            corpus_gate.TARGET_TRIPLE,
        )
        status = case.get("status")
        if status not in {"pass", "unsupported", "fail"}:
            raise BaselineError(f"{workload.identity} has an invalid outcome")
        if (
            profile.disposition
            is corpus_target_profile.TargetProfileDisposition.RUNNABLE
        ):
            if status != "pass" or case.get("category") is not None:
                raise BaselineError(
                    f"{workload.identity} violates its exact target disposition"
                )
            _integer(case.get("graphs"), f"{workload.identity} graphs", minimum=1)
            _integer(case.get("actors"), f"{workload.identity} actors", minimum=1)
            selected_sources = case.get("selected_sources")
            if not isinstance(selected_sources, list) or not selected_sources:
                raise BaselineError(f"{workload.identity} has no source coverage")
            _validate_dfg_projection(case.get("dfg_simulation"), workload.identity)
            passed += 1
        elif (
            profile.disposition
            is corpus_target_profile.TargetProfileDisposition.INCOMPATIBLE_ISA
        ):
            if (
                status != "unsupported"
                or case.get("category")
                != corpus_gate.CATEGORY_TARGET_PROFILE_UNSUPPORTED
                or case.get("detail") != profile.detail
            ):
                raise BaselineError(
                    f"{workload.identity} violates its exact target disposition"
                )
            unsupported += 1
            unsupported_rows.append(
                {"identity": workload.identity, "reason": profile.detail}
            )
        else:
            raise BaselineError(
                f"{workload.identity} lacks a compatible target-profile provider"
            )
        counts = suite_counts.setdefault(
            workload.suite, {"pass": 0, "unsupported": 0, "fail": 0}
        )
        counts[status] += 1

    expected_counts = {
        "case_count": len(workloads),
        "failed": 0,
        "passed": passed,
        "unsupported": unsupported,
    }
    for field, expected in expected_counts.items():
        if summary.get(field) != expected:
            raise BaselineError(f"corpus {field} does not match exact case outcomes")
    if summary.get("failure_categories") != {}:
        raise BaselineError("corpus contains failure categories")
    expected_unsupported_categories = {
        corpus_gate.CATEGORY_TARGET_PROFILE_UNSUPPORTED: unsupported
    }
    if summary.get("unsupported_categories") != expected_unsupported_categories:
        raise BaselineError("corpus unsupported categories changed")
    if summary.get("suite_counts") != suite_counts:
        raise BaselineError("corpus suite counts do not match exact outcomes")
    return {
        "failed": 0,
        "passed": passed,
        "unsupported": unsupported,
        "unsupported_rows": unsupported_rows,
        "workloads": len(workloads),
    }


def _semantic_case_projection(case: Mapping[str, object]) -> dict[str, object]:
    projection: dict[str, object] = {
        "category": case.get("category"),
        "detail": case.get("detail"),
        "identity": case.get("identity"),
        "status": case.get("status"),
    }
    if case.get("status") != "pass":
        return projection
    dfg = _validate_dfg_projection(
        case.get("dfg_simulation"), str(case.get("identity"))
    )
    projection.update(
        {
            "actors": case.get("actors"),
            "dfg_simulation": {
                field: dfg.get(field)
                for field in (
                    "artifacts",
                    "dse_execution",
                    "dynamic_calls",
                    "event_count",
                    "execution_terminal",
                    "floating_variance_bytes",
                    "floating_variance_kind",
                    "memory_bytes_compared",
                    "operation_firings",
                    "selected_source_files",
                    "value_lanes_compared",
                    "wavefront_steps",
                )
            },
            "graphs": case.get("graphs"),
            "selected_sources": case.get("selected_sources"),
        }
    )
    return projection


def stable_replay_projection(
    summary: Mapping[str, object], selected_identities: Sequence[str]
) -> list[dict[str, object]]:
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise BaselineError("replay cases must be an array")
    by_identity = {
        str(_mapping(case, "replay case").get("identity")): _mapping(
            case, "replay case"
        )
        for case in cases
    }
    if set(by_identity) != set(selected_identities) or len(cases) != len(
        selected_identities
    ):
        raise BaselineError("replay identities differ from the selected set")
    return [
        _semantic_case_projection(by_identity[identity])
        for identity in selected_identities
    ]


def compare_replays(
    first: Mapping[str, object],
    second: Mapping[str, object],
    selected_identities: Sequence[str],
) -> None:
    first_projection = stable_replay_projection(first, selected_identities)
    second_projection = stable_replay_projection(second, selected_identities)
    if first_projection != second_projection:
        raise BaselineError("semantic replay differs between identical executions")


def validate_builtin_export(identity_text: str, output_base: Path) -> dict[str, object]:
    identity = _artifact_identity(identity_text.strip(), "builtin Fabric identity")
    mlir_path = output_base.with_suffix(".mlir")
    html_path = output_base.with_suffix(".html")
    try:
        mlir = mlir_path.read_text(encoding="utf-8")
        html = html_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BaselineError(f"cannot read builtin Fabric export: {exc}") from exc
    if "fabric.system" not in mlir or "fabric.system.acc_core" not in mlir:
        raise BaselineError("builtin Fabric MLIR has no nonempty System topology")
    required_html = (
        'data-layout-engine="loom-layered-v1"',
        'data-view-kind="system"',
        'data-view-kind="spatial-core"',
        'data-entity-kind="fabric.acc_core_occurrence"',
    )
    if any(marker not in html for marker in required_html):
        raise BaselineError("builtin Fabric HTML lacks a static architecture view")
    if any(marker in html for marker in REALTIME_LAYOUT_MARKERS):
        raise BaselineError("builtin Fabric HTML contains browser-side layout")
    return {
        "html_bytes": html_path.stat().st_size,
        "identity": identity,
        "mlir_bytes": mlir_path.stat().st_size,
    }


def validate_hardware_anchor_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    if set(report) != {"anchors"}:
        raise BaselineError("hardware anchor report has unknown fields")
    anchors = report.get("anchors")
    if anchors != list(HARDWARE_ANCHORS):
        raise BaselineError("hardware anchor inventory is not exact")
    return {"anchors": list(HARDWARE_ANCHORS)}


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        return _mapping(json.loads(path.read_text()), str(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot read JSON result {path}: {exc}") from exc


def _git_fact(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=source_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise BaselineError(completed.stderr.strip() or "cannot inspect git state")
    return completed.stdout.strip()


def _run_step(
    name: str,
    command: Sequence[str],
    output_root: Path,
    timeout_seconds: float,
    *,
    cwd: Path,
) -> dict[str, object]:
    usage = corpus_gate.CaseResourceUsage()
    started = time.monotonic()
    log_path = output_root / f"{name}.log"
    failure = corpus_gate.run_step(
        command,
        log_path,
        started + timeout_seconds,
        name,
        cwd=cwd,
        resource_usage=usage,
    )
    wall_seconds = time.monotonic() - started
    if failure is not None:
        raise BaselineError(f"{name}: {failure.detail}")
    return {
        "command": shlex.join(command),
        "cpu_seconds": usage.cpu_seconds,
        "log": str(log_path),
        "peak_resident_bytes": usage.peak_resident_bytes,
        "wall_seconds": wall_seconds,
    }


def _find_representative(
    workloads: Sequence[corpus_inventory.ProgramWorkload], case_name: str
) -> str:
    matches = [
        workload.identity
        for workload in workloads
        if workload.suite == "loombench" and workload.case == case_name
    ]
    if len(matches) != 1:
        raise BaselineError(f"expected one LoomBench workload named {case_name}")
    return matches[0]


def _corpus_command(
    source_root: Path,
    output_dir: Path,
    jobs: int,
    args: argparse.Namespace,
    selected_identities: Sequence[str] = (),
) -> list[str]:
    command = [
        sys.executable,
        str(source_root / "test" / "corpus_gate.py"),
        "--stage",
        "dfg-sim",
        "--jobs",
        str(jobs),
        "--candidate-jobs",
        "1",
        "--case-timeout",
        str(CASE_WALL_LIMIT_SECONDS),
        "--dfg-simulation-timeout",
        str(DFG_WALL_LIMIT_SECONDS),
        "--out-dir",
        str(output_dir),
        "--json",
        str(output_dir / "summary.json"),
    ]
    for identity in selected_identities:
        command.extend(("--case", identity))
    for option, value in (
        ("--sysroot", args.sysroot),
        ("--gcc-toolchain", args.gcc_toolchain),
        ("--riscv-gcc", args.riscv_gcc),
    ):
        if value:
            command.extend((option, value))
    return command


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--build-root", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "build" / "test-runs" / "semantic-baseline",
    )
    parser.add_argument("--jobs", type=int, default=corpus_gate.default_jobs())
    parser.add_argument("--sysroot")
    parser.add_argument("--gcc-toolchain")
    parser.add_argument("--riscv-gcc")
    return parser.parse_args(argv)


def run_baseline(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve()
    build_root = args.build_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.jobs < 1 or args.jobs > corpus_gate.default_jobs():
        raise BaselineError("jobs must be within the host worker cap")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    started = time.monotonic()
    steps: dict[str, object] = {}
    lit = source_root / "externals" / "llvm" / "build" / "bin" / "llvm-lit"
    steps["lit"] = _run_step(
        "lit",
        [
            str(lit),
            "-sv",
            "--time-tests",
            "-j",
            str(args.jobs),
            str(build_root / "test"),
        ],
        output_root,
        1200.0,
        cwd=source_root,
    )
    steps["adg_builder"] = _run_step(
        "adg-builder",
        [
            str(build_root / "test" / "adg" / "loom-adg-builder-api-test"),
            "--conformance-anchors",
        ],
        output_root,
        180.0,
        cwd=source_root,
    )
    steps["adg_builder"]["coverage"] = validate_hardware_anchor_report(
        _load_json(output_root / "adg-builder.log")
    )

    builtins: dict[str, object] = {}
    loom_adg = build_root / "tools" / "loom-adg" / "loom-adg"
    for preset in ("small", "coverage", "large"):
        preset_root = output_root / "fabric" / preset
        store = preset_root / "store"
        store.mkdir(parents=True)
        output_base = preset_root / "fabric"
        step_name = f"fabric-{preset}"
        steps[step_name] = _run_step(
            step_name,
            [
                str(loom_adg),
                f"--builtin={preset}",
                f"--artifact-store={store}",
                f"--output={output_base}",
            ],
            output_root,
            180.0,
            cwd=source_root,
        )
        identity_text = (output_root / f"{step_name}.log").read_text()
        builtins[preset] = validate_builtin_export(identity_text, output_base)

    workloads = corpus_inventory.load_workload_inventory(source_root)
    representative_identities = tuple(
        _find_representative(workloads, name)
        for name in ("crc32", "vecadd", "gather", "stream-nested")
    )
    replay_summaries: list[Mapping[str, object]] = []
    for ordinal in range(2):
        replay_root = output_root / f"replay-{ordinal}"
        step_name = f"replay-{ordinal}"
        steps[step_name] = _run_step(
            step_name,
            _corpus_command(
                source_root,
                replay_root,
                min(args.jobs, len(representative_identities)),
                args,
                representative_identities,
            ),
            output_root,
            len(representative_identities) * CASE_WALL_LIMIT_SECONDS + 60.0,
            cwd=source_root,
        )
        replay_summaries.append(_load_json(replay_root / "summary.json"))
    compare_replays(replay_summaries[0], replay_summaries[1], representative_identities)

    corpus_root = output_root / "corpus"
    steps["corpus"] = _run_step(
        "corpus",
        _corpus_command(source_root, corpus_root, args.jobs, args),
        output_root,
        CORPUS_WALL_LIMIT_SECONDS + 30.0,
        cwd=source_root,
    )
    corpus_summary = _load_json(corpus_root / "summary.json")
    corpus_validation = validate_corpus_summary(corpus_summary, workloads)

    result: dict[str, object] = {
        "builtins": builtins,
        "commit": _git_fact(source_root, "rev-parse", "HEAD"),
        "corpus": {
            **corpus_validation,
            "summary": str(corpus_root / "summary.json"),
        },
        "git_dirty": bool(_git_fact(source_root, "status", "--short")),
        "host": {
            "logical_cpus": os.cpu_count(),
            "worker_cap": corpus_gate.default_jobs(),
            "workers": args.jobs,
        },
        "replay": {
            "identities": list(representative_identities),
            "projection": stable_replay_projection(
                replay_summaries[0], representative_identities
            ),
        },
        "schema_version": 1,
        "steps": steps,
        "wall_seconds": time.monotonic() - started,
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main(argv: Sequence[str]) -> int:
    try:
        result = run_baseline(parse_args(argv))
    except (BaselineError, corpus_inventory.InventoryError, OSError) as exc:
        print(f"[semantic-baseline] error: {exc}", file=sys.stderr)
        return 1
    corpus = result["corpus"]
    print(
        f"[semantic-baseline] PASS: {corpus['passed']} pass, "
        f"{corpus['unsupported']} unsupported, {corpus['failed']} fail"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
