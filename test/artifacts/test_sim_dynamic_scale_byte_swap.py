#!/usr/bin/env python3
"""Regression test for byte_swap simulator scaling evidence."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import artifact_test_common


GRAPH = "g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0"
HARDWARE = "shared_reduction_adg"
HARDWARE_MLIR = Path("test/pnr/shared_reduction_adg.mlir")
SCALE_CASES = {
    "byte_swap_small": {"count": 8, "dfg_cycles": 80, "cgra_cycles": 92},
    "byte_swap_large": {"count": 32, "dfg_cycles": 320, "cgra_cycles": 332},
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object in {path.name}: {data}")
    return data


def assert_fields(
    record: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"unexpected {label} {key}: {record}")


def single_row(
    rows: list[dict[str, str]],
    *,
    key: str,
    value: str,
    label: str,
) -> dict[str, str]:
    matches = [row for row in rows if row[key] == value]
    if len(matches) != 1:
        raise AssertionError(f"expected one {label} row, got {rows}")
    return matches[0]


def run_command(
    repo: Path,
    argv: list[str],
    label: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed with {result.returncode}\n"
            f"command: {' '.join(argv)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def find_tool(repo: Path, tool: str) -> Path:
    candidates = [
        repo / "build" / "tools" / tool / tool,
        repo / "build" / "bin" / tool,
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise AssertionError(f"missing required tool {tool}: {candidates}")


def to_i32_literal(value: int) -> str:
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        return str(value - 0x100000000)
    return str(value)


def byte_swap_input_values(count: int) -> list[str]:
    seeds = [
        0,
        0xFFFFFFFF,
        0x12345678,
        0x11223344,
        0xFF000000,
        0x000000FF,
        0xABCDEF01,
        0x01020304,
    ]
    values = []
    for index in range(count):
        value = seeds[index] if index < len(seeds) else index * 0x01020304
        values.append(to_i32_literal(value))
    return values


def generate_byte_swap_dfg(repo: Path, out_dir: Path) -> Path:
    dfg_dir = out_dir / "byte-swap-dfg"
    env = os.environ.copy()
    env.update(
        {
            "BUILD_DIR": str(dfg_dir),
            "LOOM_CC": str(repo / "build/bin/loom-cc"),
            "LOOM_RAISE": str(repo / "build/bin/loom-raise"),
            "LOOM_LOWER": str(repo / "build/bin/loom-lower"),
            "LOOM_RAISE_OPT": str(repo / "build/bin/loom-raise-opt"),
        }
    )
    run_command(
        repo,
        ["bash", "test/app/byte_swap/dfg_check.sh"],
        "byte_swap DFG generation",
        env=env,
    )
    dfg_mlir = dfg_dir / "main_func.dfg.mlir"
    if not dfg_mlir.is_file():
        raise AssertionError(f"byte_swap DFG generation missed {dfg_mlir}")
    return dfg_mlir


def run_dfg_sim(
    repo: Path,
    dfg_mlir: Path,
    out_dir: Path,
    workload: str,
    count: int,
) -> Path:
    dfg_tool = find_tool(repo, "loom-dfg-sim")
    report = out_dir / f"{workload}-dfg-sim-report.json"
    args = [str(dfg_tool), str(dfg_mlir)]
    for _ in range(count):
        args.extend(["--arg", "0=none"])
    args.extend(["--memref", f"1={','.join(byte_swap_input_values(count))}"])
    args.extend(["--memref", f"2={','.join('0' for _ in range(count))}"])
    for index in range(count):
        args.extend(["--arg", f"3={index}"])
    args.extend(["--graph", GRAPH, "--workload", workload, "--output", str(report)])
    run_command(repo, args, f"{workload} DFG simulation")
    return report


def run_mapping(
    repo: Path,
    dfg_mlir: Path,
    out_dir: Path,
    workload: str,
) -> tuple[Path, Path]:
    mapping_summary = out_dir / f"{workload}-pnr-mapping-summary.csv"
    mapping_artifact = out_dir / f"{workload}-pnr-mapping.json"
    run_command(
        repo,
        [
            "bash",
            "test/pnr/run_mapping_summary.sh",
            "--dfg-mlir",
            str(dfg_mlir),
            "--graph",
            GRAPH,
            "--hardware-mlir",
            str(HARDWARE_MLIR),
            "--hardware",
            HARDWARE,
            "--workload",
            workload,
            "--artifact",
            str(mapping_artifact),
            "--output",
            str(mapping_summary),
        ],
        f"{workload} mapping summary",
    )
    return mapping_summary, mapping_artifact


def run_cgra_sim(
    repo: Path,
    out_dir: Path,
    workload: str,
    dfg_report: Path,
    mapping_artifact: Path,
) -> Path:
    cgra_tool = find_tool(repo, "loom-cgra-sim")
    cgra_report = out_dir / f"{workload}-cgra-sim-report.json"
    run_command(
        repo,
        [
            str(cgra_tool),
            "--dfg-report",
            str(dfg_report),
            "--mapping-artifact",
            str(mapping_artifact),
            "--hardware-mlir",
            str(HARDWARE_MLIR),
            "--output",
            str(cgra_report),
        ],
        f"{workload} CGRA simulation",
    )
    return cgra_report


def run_comparison(
    repo: Path,
    out_dir: Path,
    workload: str,
    dfg_report: Path,
    cgra_report: Path,
    mapping_artifact: Path,
) -> Path:
    comparison = out_dir / f"{workload}-sim-comparison-report.json"
    artifact_test_common.require_success(
        repo,
        [
            "bash",
            "test/simulator/run_sim_comparison_report.sh",
            "--dfg-report",
            str(dfg_report),
            "--cgra-report",
            str(cgra_report),
            "--mapping-artifact",
            str(mapping_artifact),
            "--output",
            str(comparison),
        ],
        f"{workload} simulation comparison report",
    )
    return comparison


def assert_scale_artifacts(
    out_dir: Path,
    workload: str,
    count: int,
    expected_dfg_cycles: int,
    expected_cgra_cycles: int,
) -> None:
    mapping_id = f"{workload}__{GRAPH}__{HARDWARE}"
    mapping_summary = single_row(
        read_csv_rows(out_dir / f"{workload}-pnr-mapping-summary.csv"),
        key="workload",
        value=workload,
        label=f"{workload} mapping",
    )
    assert_fields(
        mapping_summary,
        {
            "hardware": HARDWARE,
            "mapping_id": mapping_id,
            "placed_records": "4",
            "routed_edges": "4",
            "unrouted_edges": "0",
            "unplaced_records": "0",
            "status": "pass",
        },
        label=f"{workload} mapping row",
    )
    mapping = read_json_object(out_dir / f"{workload}-pnr-mapping.json")
    assert_fields(
        mapping,
        {
            "workload": workload,
            "graph": GRAPH,
            "hardware": HARDWARE,
            "mapping_id": mapping_id,
            "placed_records": 4,
            "routed_edges": 4,
            "config_records": 45,
            "status": "pass",
        },
        label=f"{workload} mapping artifact",
    )

    dfg_report = read_json_object(out_dir / f"{workload}-dfg-sim-report.json")
    assert_fields(
        dfg_report,
        {
            "status": "pass",
            "workload": workload,
            "graph": GRAPH,
            "optimistic_cycles": expected_dfg_cycles,
            "dynamic_work_items": count,
        },
        label=f"{workload} DFG report",
    )
    fire_counts = dfg_report.get("operation_fire_counts", {})
    if not isinstance(fire_counts, dict):
        raise AssertionError(f"{workload} DFG report lacks operation fire counts: {dfg_report}")
    assert_fields(
        fire_counts,
        {
            "llvm.intr.bswap": count,
            "dataflow.load": count,
            "dataflow.store": count,
            "dataflow.sync": count,
        },
        label=f"{workload} fire counts",
    )

    cgra_report = read_json_object(out_dir / f"{workload}-cgra-sim-report.json")
    assert_fields(
        cgra_report,
        {
            "status": "pass",
            "workload": workload,
            "mapping_id": mapping_id,
            "dfg_cycles": expected_dfg_cycles,
            "hardware_aware_cycles": expected_cgra_cycles,
            "difference_classification": "expected_hardware_constraint",
            "performance_delta_cycles": 12,
        },
        label=f"{workload} CGRA report",
    )
    if cgra_report["hardware_aware_cycles"] < dfg_report["optimistic_cycles"]:
        raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra_report}")

    comparison = read_json_object(out_dir / f"{workload}-sim-comparison-report.json")
    assert_fields(
        comparison,
        {
            "status": "pass",
            "workload": workload,
            "dfg_sim_cycles": expected_dfg_cycles,
            "cgra_sim_cycles": expected_cgra_cycles,
            "performance_delta_cycles": 12,
            "difference_classification": "expected_hardware_constraint",
        },
        label=f"{workload} simulation comparison",
    )


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-byte-swap-scale-") as tmp:
        out_dir = Path(tmp)
        dfg_mlir = generate_byte_swap_dfg(repo, out_dir)
        artifacts: list[Path] = []

        for workload, expected in SCALE_CASES.items():
            count = int(expected["count"])
            dfg_report = run_dfg_sim(repo, dfg_mlir, out_dir, workload, count)
            mapping_summary, mapping_artifact = run_mapping(repo, dfg_mlir, out_dir, workload)
            cgra_report = run_cgra_sim(repo, out_dir, workload, dfg_report, mapping_artifact)
            comparison = run_comparison(repo, out_dir, workload, dfg_report, cgra_report, mapping_artifact)
            artifacts.extend([mapping_summary, mapping_artifact, dfg_report, cgra_report, comparison])

        sim_cycle = out_dir / "sim-cycle-summary.csv"
        sim_args = ["bash", "test/app/run_sim_cycle_summary.sh"]
        for workload in SCALE_CASES:
            sim_args.extend(["--dfg-report", str(out_dir / f"{workload}-dfg-sim-report.json")])
        for workload in SCALE_CASES:
            sim_args.extend(["--cgra-report", str(out_dir / f"{workload}-cgra-sim-report.json")])
        sim_args.extend(["--output", str(sim_cycle)])
        artifact_test_common.require_success(repo, sim_args, "byte_swap scale simulator cycle summary")
        artifacts.append(sim_cycle)

        for workload, expected in SCALE_CASES.items():
            assert_scale_artifacts(
                out_dir,
                workload,
                int(expected["count"]),
                int(expected["dfg_cycles"]),
                int(expected["cgra_cycles"]),
            )

        rows = read_csv_rows(sim_cycle)
        small = single_row(rows, key="kernel", value="byte_swap_small", label="small sim cycle")
        large = single_row(rows, key="kernel", value="byte_swap_large", label="large sim cycle")
        assert_fields(
            small,
            {"dfg_sim_cycles": "80", "cgra_sim_cycles": "92", "status": "pass"},
            label="small sim cycle row",
        )
        assert_fields(
            large,
            {"dfg_sim_cycles": "320", "cgra_sim_cycles": "332", "status": "pass"},
            label="large sim cycle row",
        )
        if int(large["dfg_sim_cycles"]) <= int(small["dfg_sim_cycles"]):
            raise AssertionError(f"larger byte_swap input should cost more DFG cycles: {rows}")
        if int(large["cgra_sim_cycles"]) <= int(small["cgra_sim_cycles"]):
            raise AssertionError(f"larger byte_swap input should cost more CGRA cycles: {rows}")
        if int(small["dfg_sim_cycles"]) in {448, 579, 1027}:
            raise AssertionError(f"small byte_swap scale should add distinct cycle evidence: {small}")
        if int(large["dfg_sim_cycles"]) in {448, 579, 1027}:
            raise AssertionError(f"large byte_swap scale should add distinct cycle evidence: {large}")

        audit = out_dir / "artifact-audit-summary.json"
        artifact_test_common.require_success(
            repo,
            [
                "python3",
                "test/e2e/audit_intermediate_artifacts.py",
                "--output",
                str(audit),
                *[str(path) for path in artifacts],
            ],
            "byte_swap scale artifact audit",
        )
        audit_data = read_json_object(audit)
        if audit_data.get("verdict") != "pass":
            raise AssertionError(f"expected byte_swap scale artifact audit pass: {audit_data}")
        checks = {
            check.get("rule")
            for check in audit_data.get("cross_artifact_checks", [])
            if isinstance(check, dict)
        }
        expected_checks = {
            "sim_cycle_dfg_report_evidence",
            "sim_cycle_report_mapping_evidence",
        }
        if not expected_checks.issubset(checks):
            raise AssertionError(f"audit missed byte_swap scale cross checks {checks}: {audit_data}")
        if audit_data.get("cross_artifact_findings"):
            raise AssertionError(f"byte_swap scale audit should not have cross findings: {audit_data}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
