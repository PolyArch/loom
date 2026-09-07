#!/usr/bin/env python3
"""Qualify the tracked suite-wide CGRA Spatial execution budget."""

from __future__ import annotations

import json
import os
import tempfile
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config.timeout_budgets import Tier, seconds as timeout_seconds  # noqa: E402
import simulation_conformance  # noqa: E402
import cgra_qualification  # noqa: E402


MINIMAL_RUNTIME = REPOSITORY_ROOT / "test" / "frontend" / "Inputs" / "minimal-c-runtime"
SOURCE_PREPARATION_WORKERS = 2
SOURCE_CANDIDATE_JOBS = 4
COMPILATION_TIMEOUT_SECONDS = 120.0
SOURCE_PIPELINE_TIMEOUT_SECONDS = 900.0
SPATIAL_PNR_TIMEOUT_SECONDS = float(timeout_seconds(Tier.FAST))
PROFILE_TIMEOUT_SECONDS = float(timeout_seconds(Tier.XLONG))
PROFILE_TIMEOUT_MARGIN_SECONDS = PROFILE_TIMEOUT_SECONDS - SPATIAL_PNR_TIMEOUT_SECONDS
if PROFILE_TIMEOUT_MARGIN_SECONDS <= 0:
    raise ValueError("CGRA profile wrapper has no deadline margin")


@dataclass(frozen=True)
class ResolvedSourceWorkload:
    name: str
    source: Path
    operator_id: str
    protocol_symbol: str
    compiler_flags: tuple[str, ...]


class QualificationDisposition(Enum):
    COMPLETED_EMPTY = "completed_empty"
    INCOMPLETE = "incomplete"
    PROVEN_INFEASIBLE = "proven_infeasible"


class QualificationStopped(RuntimeError):
    def __init__(
        self,
        disposition: QualificationDisposition,
        reason: str | None,
        diagnostic: str,
    ) -> None:
        super().__init__(diagnostic)
        self.disposition = disposition
        self.reason = reason
        self.diagnostic = diagnostic


def resolve_workloads() -> tuple[str, tuple[ResolvedSourceWorkload, ...]]:
    digest, operator_rows = cgra_qualification.load_cgra_representative_operators()
    rows_by_workload = {row.workload: row for row in operator_rows}
    resolved = tuple(
        ResolvedSourceWorkload(
            workload,
            Path(rows_by_workload[workload].source),
            rows_by_workload[workload].operator_id,
            rows_by_workload[workload].protocol_symbol,
            rows_by_workload[workload].compiler_flags,
        )
        for workload in cgra_qualification.CGRA_REPRESENTATIVE_WORKLOADS
    )
    for workload in resolved:
        if workload.source.suffix not in {".c", ".cpp"}:
            raise RuntimeError("qualification source has an unsupported language")
        if not (REPOSITORY_ROOT / workload.source).is_file():
            raise RuntimeError("qualification source is absent")
    return digest, resolved


def run(
    command: Sequence[str],
    timeout_seconds: float,
    environment: dict[str, str],
    trace: Path,
) -> simulation_conformance.ProcessExecution:
    completed = simulation_conformance.execute_process(
        command,
        timeout_seconds,
        environment=environment,
    )
    # Preserve completed work even when a later workload prevents publication
    # of the suite gate. These are diagnostics, never a partial gate authority.
    trace.with_suffix(".stdout").write_text(completed.stdout, encoding="utf-8")
    trace.with_suffix(".stderr").write_text(completed.stderr, encoding="utf-8")
    trace.with_suffix(".execution.json").write_text(
        json.dumps(
            {
                "command": completed.command,
                "budget_seconds": timeout_seconds,
                "disposition": completed.disposition.value,
                "return_code": completed.return_code,
                "elapsed_seconds": completed.elapsed_seconds,
                "process_group_terminated": completed.process_group_terminated,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    if completed.disposition is not simulation_conformance.ProcessDisposition.COMPLETED:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        if completed.disposition in {
            simulation_conformance.ProcessDisposition.TIMED_OUT,
            simulation_conformance.ProcessDisposition.CLEANUP_FAILED,
        }:
            raise QualificationStopped(
                QualificationDisposition.INCOMPLETE,
                completed.disposition.value,
                diagnostic,
            )
        raise RuntimeError(
            f"command failed with disposition {completed.disposition.value}: "
            f"{' '.join(command)}\n{diagnostic}"
        )
    return completed


def compile_command(
    loom_cc: Path, workload: ResolvedSourceWorkload, output: Path
) -> list[str]:
    command = [
        str(loom_cc),
        "--target=riscv64-unknown-elf",
        "-march=rv64imafdc_zicsr_zifencei",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-mcpu=generic-rv64",
        "-isystem",
        str(MINIMAL_RUNTIME),
    ]
    if workload.source.suffix == ".cpp":
        command.append("-std=c++17")
    command.extend(("-emit-llvm", "-S", "-O1", "-gline-tables-only"))
    command.extend(workload.compiler_flags)
    command.extend((str(REPOSITORY_ROOT / workload.source), "-o", str(output)))
    return command


def prepare_workload(
    workload: ResolvedSourceWorkload,
    loom_cc: Path,
    loom_dfg_run: Path,
    environment: dict[str, str],
    qualification_root: Path,
    store: Path,
) -> Path:
    root = qualification_root / workload.name
    root.mkdir(parents=True)
    llvm_ir = root / "input.ll"
    report = root / "source-report.json"
    canonical = root / "dataflow.mlir"
    run(
        compile_command(loom_cc, workload, llvm_ir),
        COMPILATION_TIMEOUT_SECONDS,
        environment,
        root / "compile",
    )
    run(
        (
            str(loom_dfg_run),
            f"--artifact-store={store}",
            f"--candidate-jobs={SOURCE_CANDIDATE_JOBS}",
            f"--operator-protocol-symbol={workload.protocol_symbol}",
            "--expected-entry-result=0",
            f"--canonical-output={canonical}",
            f"--output={report}",
            str(llvm_ir),
        ),
        SOURCE_PIPELINE_TIMEOUT_SECONDS,
        environment,
        root / "source",
    )
    return report


def qualify_workload(
    workload: ResolvedSourceWorkload,
    cgra_profile: Path,
    environment: dict[str, str],
    qualification_root: Path,
    store: Path,
    hardware_report: Path,
) -> dict[str, object]:
    root = qualification_root / workload.name
    report = root / "source-report.json"
    profiled = run(
        (
            str(cgra_profile),
            str(store),
            str(report),
            workload.name,
            workload.operator_id,
            workload.protocol_symbol,
            str(hardware_report),
        ),
        PROFILE_TIMEOUT_SECONDS,
        environment,
        root / "profile",
    )
    parsed = json.loads(profiled.stdout)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"CGRA profile for {workload.name} is not an object")
    if parsed.get("schema") == cgra_qualification.CGRA_PROFILE_OUTCOME_SCHEMA:
        if (
            parsed.get("workload") != workload.name
            or parsed.get("operator_id") != workload.operator_id
            or parsed.get("protocol_symbol") != workload.protocol_symbol
        ):
            raise RuntimeError("CGRA profile outcome has a foreign workload")
        outcome, reason = cgra_qualification.validate_cgra_profile_outcome(parsed)
        if outcome == "incomplete" and reason is not None:
            raise QualificationStopped(
                QualificationDisposition.INCOMPLETE,
                reason,
                f"{workload.name}: {reason}",
            )
        if outcome == "proven_infeasible" and reason is None:
            raise QualificationStopped(
                QualificationDisposition.PROVEN_INFEASIBLE,
                None,
                f"{workload.name}: proven_infeasible",
            )
        if outcome == "completed" and reason is None:
            raise QualificationStopped(
                QualificationDisposition.COMPLETED_EMPTY,
                None,
                f"{workload.name}: completed_empty",
            )
        raise RuntimeError("CGRA PnR outcome has an invalid disposition")
    if parsed.get("schema") != cgra_qualification.CGRA_PROFILE_SCHEMA:
        raise RuntimeError("CGRA profile has a foreign schema")
    return parsed


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--dfg-run", required=True, type=Path)
    parser.add_argument("--cgra-profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.unlink(missing_ok=True)
    loom_cc = arguments.compiler.resolve(strict=True)
    loom_dfg_run = arguments.dfg_run.resolve(strict=True)
    cgra_profile = arguments.cgra_profile.resolve(strict=True)
    operator_gate_sha256, workloads = resolve_workloads()
    if tuple(workload.name for workload in workloads) != (
        cgra_qualification.CGRA_REPRESENTATIVE_WORKLOADS
    ):
        raise RuntimeError("qualification workload inventory drifted from the gate")
    temporary_root = REPOSITORY_ROOT / "temp"
    temporary_root.mkdir(exist_ok=True)
    qualification_root = Path(tempfile.mkdtemp(
        prefix="cgra-budget-qualification-", dir=temporary_root
    ))
    store = qualification_root / "store"
    store.mkdir()
    environment = dict(os.environ)
    environment["TMPDIR"] = str(qualification_root)
    print(f"qualification evidence: {qualification_root}", file=sys.stderr, flush=True)
    profiles: list[dict[str, object]] = []
    try:
        requests = []
        # Each source pipeline already bounds its candidate work to four jobs.
        # Prepare two independent sources at a time, retaining canonical report
        # order and finishing all preparation before hardware/profile execution.
        with ThreadPoolExecutor(max_workers=SOURCE_PREPARATION_WORKERS) as pool:
            preparations = [
                pool.submit(
                    prepare_workload, workload, loom_cc, loom_dfg_run,
                    environment, qualification_root, store,
                )
                for workload in workloads
            ]
            for workload, preparation in zip(workloads, preparations):
                report = preparation.result()
                print(f"prepared {workload.name}", file=sys.stderr, flush=True)
                requests.append({
                    "workload": workload.name,
                    "operator_id": workload.operator_id,
                    "protocol_symbol": workload.protocol_symbol,
                    "source_report": str(report),
                })
        request_path = qualification_root / "source-requests.json"
        request_path.write_text(json.dumps(requests, indent=2) + "\n", encoding="ascii")
        selected = run(
            (str(cgra_profile), "--hardware", str(store), str(request_path)),
            SPATIAL_PNR_TIMEOUT_SECONDS, environment, qualification_root / "hardware",
        )
        hardware = json.loads(selected.stdout)
        hardware_report = qualification_root / "hardware-search.json"
        hardware_report.write_text(
            json.dumps(hardware, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        if not cgra_qualification.validate_cgra_hardware_search(hardware):
            raise QualificationStopped(
                QualificationDisposition.INCOMPLETE,
                "hardware_search_incomplete",
                "shared hardware search did not admit every source case",
            )
        for workload in workloads:
            print(f"qualifying {workload.name}", file=sys.stderr, flush=True)
            profiles.append(qualify_workload(
                workload, cgra_profile, environment, qualification_root, store,
                hardware_report,
            ))
    except QualificationStopped as stopped:
        print(
            json.dumps(
                {
                    "schema": "loom.cgra_budget_qualification_outcome.2",
                    "disposition": stopped.disposition.value,
                    "reason": stopped.reason,
                    "diagnostic": stopped.diagnostic,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    budget = cgra_qualification.derive_cgra_spatial_budget_nanoseconds(profiles)
    output = {
        "schema": cgra_qualification.CGRA_GATE_SCHEMA,
        "policy": {
            "qualification_limit_nanoseconds": (
                cgra_qualification.CGRA_QUALIFICATION_LIMIT_NANOSECONDS
            ),
            "warmup_runs": cgra_qualification.CGRA_QUALIFICATION_WARMUP_RUNS,
            "measurement_runs": (
                cgra_qualification.CGRA_QUALIFICATION_MEASUREMENT_RUNS
            ),
            "reference_rate_target_cycles_per_second": (
                cgra_qualification.REFERENCE_RATE_TARGET_CYCLES_PER_SECOND
            ),
        },
        "operator_gate": {
            "path": cgra_qualification.CGRA_OPERATOR_GATE_RELATIVE_PATH,
            "sha256": operator_gate_sha256,
        },
        "spatial_absolute_budget_nanoseconds": budget,
        "profiles": profiles,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
