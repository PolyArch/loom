#!/usr/bin/env python3
"""Regression test for the stream_update full-stack artifact chain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import artifact_test_common


CASE = "stream_update"
GRAPH = "g_t_stream_update_kernel_red_0_0"
HARDWARE = "shared_memory_reduction_adg"


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


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    with artifact_test_common.repo_temp_dir(repo, "loom-stream-update-chain-") as tmp:
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
            "stream_update intermediate artifact chain",
        )

        dfg = require_status(out_dir / f"{CASE}-dfg-sim-report.json", "pass")
        mapping = require_status(out_dir / "pnr-mapping.json", "pass")
        cgra = require_status(out_dir / f"{CASE}-cgra-sim-report.json", "pass")
        comparison = require_status(out_dir / "sim-comparison-report.json", "pass")
        runtime = require_status(out_dir / "runtime-package.json", "pass")

        if dfg.get("workload") != CASE or dfg.get("graph") != GRAPH:
            raise AssertionError(f"stream_update DFG report should use the kernel graph: {dfg}")
        if mapping.get("workload") != CASE or mapping.get("graph") != GRAPH:
            raise AssertionError(f"stream_update mapping should target the kernel graph: {mapping}")
        if mapping.get("hardware") != HARDWARE:
            raise AssertionError(f"stream_update should map on the shared memory fabric: {mapping}")
        if cgra.get("hardware_aware_cycles", 0) < dfg.get("optimistic_cycles", 0):
            raise AssertionError(f"CGRA-sim must not be more optimistic than DFG-sim: {cgra}")
        if comparison.get("difference_classification") != "expected_hardware_constraint":
            raise AssertionError(f"unexpected stream_update comparison classification: {comparison}")
        if dfg.get("final_outputs") != ["none", "i32:1976", "i32:30"]:
            raise AssertionError(f"stream_update must expose source-derived final outputs: {dfg}")
        if cgra.get("final_outputs") != dfg.get("final_outputs"):
            raise AssertionError(f"stream_update CGRA report should preserve final outputs: {cgra}")
        if runtime.get("workload") != CASE:
            raise AssertionError(f"runtime package should name stream_update: {runtime}")

        audit = read_json(out_dir / "artifact-audit-summary.json")
        if audit.get("verdict") != "pass" or audit.get("cross_artifact_findings"):
            raise AssertionError(f"stream_update artifact audit must pass: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
