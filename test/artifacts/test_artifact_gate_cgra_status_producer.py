#!/usr/bin/env python3
"""Regression test for artifact gate CGRA status producer selection."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path


def load_gate_module(repo: Path):
    module_path = repo / "test" / "artifacts" / "test_intermediate_artifacts.py"
    spec = importlib.util.spec_from_file_location("test_intermediate_artifacts", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_contains(command: list[str], *items: str) -> None:
    missing = [item for item in items if item not in command]
    if missing:
        raise AssertionError(f"command missed {missing}: {command}")


def main() -> int:
    repo = Path(sys.argv[1]).resolve()
    gate = load_gate_module(repo)
    previous_jobs = os.environ.get("LOOM_TEST_JOBS")
    previous_inner_jobs = os.environ.get("LOOM_ARTIFACT_GATE_INNER_JOBS")
    os.environ["LOOM_TEST_JOBS"] = "23"
    try:
        temp_root = repo / "temp" / "test-runs"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            out_dir = Path(tmp)
            output, command = gate.csv_producer_command(
                out_dir,
                "test/e2e/run_cgra_status_summary.sh",
                "cgra-status-summary.csv",
            )
            if output != out_dir / "cgra-status-rollup" / "cgra-status-summary.csv":
                raise AssertionError(f"unexpected CGRA status output path: {output}")
            if command[:2] != ["bash", "test/e2e/run_cmsis_cgra_status_rollup.sh"]:
                raise AssertionError(
                    f"CGRA status gate should use the CMSIS rollup producer: {command}"
                )
            require_contains(command, "--output-dir", str(output.parent), "--cmsis-sim-default", "--jobs", "4")

            os.environ["LOOM_ARTIFACT_GATE_INNER_JOBS"] = "7"
            _override_output, override_command = gate.csv_producer_command(
                out_dir,
                "test/e2e/run_cgra_status_summary.sh",
                "cgra-status-summary.csv",
            )
            require_contains(override_command, "--jobs", "7")

            normal_output, normal_command = gate.csv_producer_command(
                out_dir,
                "test/app/run_source_compat_summary.sh",
                "source-compat-summary.csv",
            )
            if normal_output != out_dir / "source-compat-summary.csv":
                raise AssertionError(f"unexpected normal output path: {normal_output}")
            expected_normal = [
                "bash",
                "test/app/run_source_compat_summary.sh",
                "--output",
                str(normal_output),
            ]
            if normal_command != expected_normal:
                raise AssertionError(f"normal CSV command changed unexpectedly: {normal_command}")
    finally:
        if previous_jobs is None:
            os.environ.pop("LOOM_TEST_JOBS", None)
        else:
            os.environ["LOOM_TEST_JOBS"] = previous_jobs
        if previous_inner_jobs is None:
            os.environ.pop("LOOM_ARTIFACT_GATE_INNER_JOBS", None)
        else:
            os.environ["LOOM_ARTIFACT_GATE_INNER_JOBS"] = previous_inner_jobs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
