#!/usr/bin/env python3
"""Regression test for the modexp full-stack artifact chain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


CASE = "modexp"
GRAPH = "g_t_modexp_kernel_0_0"
HARDWARE = "shared_memory_reduction_adg"
EXPECTED_OUTPUT = [
    "i32:8",
    "i32:81",
    "i32:25",
    "i32:593996258",
    "i32:586778098",
    "i32:1000000006",
    "i32:154996558",
    "i32:89848317",
]


def read_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise AssertionError(f"expected JSON object in {path}: {data}")
    return data


def require_status(path: Path, expected: str) -> dict[str, object]:
    data = read_json(path)
    if data.get("status") != expected:
        raise AssertionError(f"unexpected status in {path.name}: {data}")
    return data


def require_output_memory(report: dict[str, object]) -> None:
    final_memory = report.get("final_memory_state")
    if not isinstance(final_memory, dict):
        raise AssertionError(f"modexp report should carry final memory state: {report}")
    if final_memory.get("arg9") != EXPECTED_OUTPUT:
        raise AssertionError(f"modexp output memory should match source constants: {report}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <repo>")
    repo = Path(argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-modexp-chain-") as tmp:
        out_dir = Path(tmp)
        artifact_test_common.require_success(
            repo,
            [
                "bash",
                "test/e2e/run_intermediate_artifact_chain.sh",
                "--output-dir",
                str(out_dir),
                "--case",
                CASE,
            ],
            "modexp intermediate artifact chain",
        )

        dfg = require_status(out_dir / f"{CASE}-dfg-sim-report.json", "pass")
        mapping = require_status(out_dir / "pnr-mapping.json", "pass")
        cgra = require_status(out_dir / f"{CASE}-cgra-sim-report.json", "pass")
        comparison = require_status(out_dir / "sim-comparison-report.json", "pass")
        runtime = require_status(out_dir / "runtime-package.json", "pass")

        if dfg.get("workload") != CASE or dfg.get("graph") != GRAPH:
            raise AssertionError(f"modexp DFG report should use the kernel graph: {dfg}")
        if mapping.get("workload") != CASE or mapping.get("graph") != GRAPH:
            raise AssertionError(f"modexp mapping should target the kernel graph: {mapping}")
        if mapping.get("hardware") != HARDWARE:
            raise AssertionError(f"modexp should map on the shared memory fabric: {mapping}")
        if cgra.get("hardware_aware_cycles", 0) < dfg.get("optimistic_cycles", 0):
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra}")
        if comparison.get("difference_classification") != "expected_hardware_constraint":
            raise AssertionError(f"unexpected modexp comparison classification: {comparison}")
        require_output_memory(dfg)
        require_output_memory(cgra)
        if runtime.get("workload") != CASE:
            raise AssertionError(f"runtime package should name modexp: {runtime}")

        audit = read_json(out_dir / "artifact-audit-summary.json")
        if audit.get("verdict") != "pass" or audit.get("cross_artifact_findings"):
            raise AssertionError(f"modexp artifact audit must pass: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
