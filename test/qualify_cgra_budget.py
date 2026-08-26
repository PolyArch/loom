#!/usr/bin/env python3
"""Qualify the tracked suite-wide CGRA Spatial execution budget."""

from __future__ import annotations

import json
import os
import shutil
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import simulation_conformance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MINIMAL_RUNTIME = REPOSITORY_ROOT / "test" / "frontend" / "Inputs" / "minimal-c-runtime"
QUALIFICATION_ROOT = REPOSITORY_ROOT / "temp" / "cgra-budget-qualification"
COMPILATION_TIMEOUT_SECONDS = 120.0
SOURCE_PIPELINE_TIMEOUT_SECONDS = 900.0
PROFILE_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class SourceWorkload:
    name: str
    operator: str


@dataclass(frozen=True)
class ResolvedSourceWorkload:
    name: str
    source: Path
    operator: str
    operator_id: str
    compiler_flags: tuple[str, ...]


WORKLOADS = (
    SourceWorkload("vecadd", "vecadd"),
    SourceWorkload("vector_pack", "vector_pack_kernel"),
    SourceWorkload("matmul", "matmul_kernel"),
    SourceWorkload("spmm", "spmm_kernel"),
    SourceWorkload("gather", "gather"),
    SourceWorkload("edge_update", "edge_update_kernel"),
    SourceWorkload("fir_filter", "fir_filter_kernel"),
    SourceWorkload("conv2d", "conv2d_kernel"),
    SourceWorkload("stencil3d", "stencil3d_kernel"),
    SourceWorkload("attention", "attention_kernel"),
)


def resolve_workloads() -> tuple[str, tuple[ResolvedSourceWorkload, ...]]:
    digest, operator_rows = simulation_conformance.load_cgra_representative_operators()
    rows_by_workload = {row.workload: row for row in operator_rows}
    resolved = tuple(
        ResolvedSourceWorkload(
            workload.name,
            Path(rows_by_workload[workload.name].source),
            workload.operator,
            rows_by_workload[workload.name].operator_id,
            rows_by_workload[workload.name].compiler_flags,
        )
        for workload in WORKLOADS
    )
    for workload in resolved:
        if workload.source.suffix not in {".c", ".cpp"}:
            raise RuntimeError("qualification source has an unsupported language")
        if not (REPOSITORY_ROOT / workload.source).is_file():
            raise RuntimeError("qualification source is absent")
    return digest, resolved


def run(
    command: Sequence[str], timeout_seconds: float, environment: dict[str, str]
) -> simulation_conformance.ProcessExecution:
    completed = simulation_conformance.execute_process(
        command,
        timeout_seconds,
        environment=environment,
        working_directory=REPOSITORY_ROOT,
    )
    if completed.disposition is not simulation_conformance.ProcessDisposition.COMPLETED:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
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


def qualify_workload(
    workload: ResolvedSourceWorkload,
    loom_cc: Path,
    loom_dfg_run: Path,
    cgra_profile: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    root = QUALIFICATION_ROOT / workload.name
    root.mkdir(parents=True)
    llvm_ir = root / "input.ll"
    report = root / "source-report.json"
    canonical = root / "dataflow.mlir"
    store = root / "store"
    store.mkdir()
    run(
        compile_command(loom_cc, workload, llvm_ir),
        COMPILATION_TIMEOUT_SECONDS,
        environment,
    )
    run(
        (
            str(loom_dfg_run),
            f"--artifact-store={store}",
            "--candidate-jobs=4",
            f"--operator-protocol-symbol={workload.operator}",
            "--expected-entry-result=0",
            f"--canonical-output={canonical}",
            f"--output={report}",
            str(llvm_ir),
        ),
        SOURCE_PIPELINE_TIMEOUT_SECONDS,
        environment,
    )
    profiled = run(
        (
            str(cgra_profile),
            str(store),
            str(report),
            workload.name,
            workload.operator_id,
        ),
        PROFILE_TIMEOUT_SECONDS,
        environment,
    )
    parsed = json.loads(profiled.stdout)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"CGRA profile for {workload.name} is not an object")
    return parsed


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--loom-cc", required=True, type=Path)
    parser.add_argument("--loom-dfg-run", required=True, type=Path)
    parser.add_argument("--cgra-profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    loom_cc = arguments.loom_cc.resolve(strict=True)
    loom_dfg_run = arguments.loom_dfg_run.resolve(strict=True)
    cgra_profile = arguments.cgra_profile.resolve(strict=True)
    operator_gate_sha256, workloads = resolve_workloads()
    if tuple(workload.name for workload in workloads) != (
        simulation_conformance.CGRA_REPRESENTATIVE_WORKLOADS
    ):
        raise RuntimeError("qualification workload inventory drifted from the gate")
    shutil.rmtree(QUALIFICATION_ROOT, ignore_errors=True)
    QUALIFICATION_ROOT.mkdir(parents=True)
    environment = dict(os.environ)
    environment["TMPDIR"] = str(QUALIFICATION_ROOT)
    profiles: list[dict[str, object]] = []
    for workload in workloads:
        print(f"qualifying {workload.name}", file=sys.stderr, flush=True)
        profiles.append(
            qualify_workload(
                workload,
                loom_cc,
                loom_dfg_run,
                cgra_profile,
                environment,
            )
        )
    budget = simulation_conformance.derive_cgra_spatial_budget_nanoseconds(profiles)
    output = {
        "schema": "loom.cgra_simulation_gate.2",
        "policy": {
            "qualification_limit_nanoseconds": (
                simulation_conformance.CGRA_QUALIFICATION_LIMIT_NANOSECONDS
            ),
            "warmup_runs": simulation_conformance.CGRA_QUALIFICATION_WARMUP_RUNS,
            "measurement_runs": (
                simulation_conformance.CGRA_QUALIFICATION_MEASUREMENT_RUNS
            ),
            "reference_rate_target_cycles_per_second": (
                simulation_conformance.REFERENCE_RATE_TARGET_CYCLES_PER_SECOND
            ),
        },
        "operator_gate": {
            "path": simulation_conformance.CGRA_OPERATOR_GATE_RELATIVE_PATH,
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
