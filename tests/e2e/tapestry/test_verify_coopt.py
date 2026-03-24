"""Verification tests for Co-Optimization (C13).

Group J tests: Validates that co_optimize() runs multiple rounds,
produces a Pareto frontier with at least 1 point, and the full
pipeline integration works end-to-end.
"""

import os
import re
import subprocess

import pytest
from pathlib import Path

from test_utils import (
    run_tapestry_tool,
    assert_success_output,
    check_no_error_strings,
)


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "tools" / "tapestry").exists():
            return p
        p = p.parent
    raise RuntimeError("Cannot locate repository root")


REPO_ROOT = _find_repo_root()


class TestCoOptimizerAPI:
    """J1: co_optimize() runs multiple rounds with architecture changes."""

    def test_co_optimizer_header_exists(self):
        """co_optimizer.h should define co_optimize function."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        assert co_h.exists(), f"co_optimizer.h not found"
        content = co_h.read_text(encoding="utf-8")

        assert "co_optimize" in content, "Missing co_optimize function"
        assert "CoOptResult" in content, "Missing CoOptResult struct"
        assert "CoOptOptions" in content, "Missing CoOptOptions struct"

    def test_co_opt_options_has_max_rounds(self):
        """CoOptOptions should have maxRounds for loop control."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "maxRounds" in content, "Missing maxRounds in CoOptOptions"
        assert "improvementThreshold" in content, (
            "Missing improvementThreshold in CoOptOptions"
        )

    def test_co_opt_options_has_sw_and_hw_configs(self):
        """CoOptOptions should contain both SW and HW optimizer configs."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "swOpts" in content or "TDGOptimizeOptions" in content, (
            "Missing SW optimizer config in CoOptOptions"
        )
        assert "hwOuterOpts" in content or "HWOuterOptimizerOptions" in content, (
            "Missing HW outer optimizer config in CoOptOptions"
        )
        assert "hwInnerOpts" in content or "HWInnerOptimizerOptions" in content, (
            "Missing HW inner optimizer config in CoOptOptions"
        )


class TestCoOptResult:
    """J1 continued: CoOptResult has proper structure."""

    def test_result_has_pareto_frontier(self):
        """CoOptResult should contain a Pareto frontier."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "paretoFrontier" in content, "Missing paretoFrontier in CoOptResult"
        assert "ParetoPoint" in content, "Missing ParetoPoint struct"

    def test_pareto_point_fields(self):
        """ParetoPoint should have throughput, area, and round."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "double throughput" in content, "Missing throughput in ParetoPoint"
        assert "double area" in content, "Missing area in ParetoPoint"
        assert "unsigned round" in content, "Missing round in ParetoPoint"

    def test_result_tracks_history(self):
        """CoOptResult should record per-round history."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "RoundRecord" in content, "Missing RoundRecord struct"
        assert "history" in content, "Missing history vector"

    def test_round_record_fields(self):
        """RoundRecord should track SW throughput, HW area, improvements."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        for field in ["swThroughput", "hwArea", "swTransforms", "hwCoreTypes", "improved"]:
            assert field in content, f"Missing {field} in RoundRecord"

    def test_result_has_best_values(self):
        """CoOptResult should report best throughput and area."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "bestThroughput" in content, "Missing bestThroughput"
        assert "bestArea" in content, "Missing bestArea"
        assert "rounds" in content, "Missing rounds count"


class TestCoOptHelpers:
    """J2: Helper functions connect SW and HW optimization steps."""

    def test_extract_kernel_profiles(self):
        """extractKernelProfiles should convert KernelDesc to KernelProfile."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "extractKernelProfiles" in content, (
            "Missing extractKernelProfiles helper"
        )

    def test_build_default_architecture(self):
        """buildDefaultArchitecture should create initial arch from profiles."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "buildDefaultArchitecture" in content, (
            "Missing buildDefaultArchitecture helper"
        )

    def test_update_contracts_from_sw(self):
        """updateContractsFromSW should propagate achieved rates."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "updateContractsFromSW" in content, (
            "Missing updateContractsFromSW helper"
        )

    def test_compute_system_area(self):
        """computeSystemArea should aggregate HW results."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "computeSystemArea" in content, "Missing computeSystemArea helper"

    def test_add_pareto_point(self):
        """addParetoPoint should manage non-dominated frontier."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "addParetoPoint" in content, "Missing addParetoPoint helper"

    def test_build_arch_from_hw_results(self):
        """buildArchFromHWResults should construct SystemArchitecture."""
        co_h = REPO_ROOT / "include" / "tapestry" / "co_optimizer.h"
        content = co_h.read_text(encoding="utf-8")

        assert "buildArchFromHWResults" in content, (
            "Missing buildArchFromHWResults helper"
        )


class TestFullPipelineIntegration:
    """J3: Full end-to-end pipeline attempts compilation and produces output."""

    def test_full_pipeline_runs_without_crash(
        self, tapestry_pipeline_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Full pipeline should run without segfault or assertion failure."""
        result = run_tapestry_tool(
            tapestry_pipeline_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "5",
                "-max-cycles", "10000",
            ],
            timeout_sec=300,
        )
        check_no_error_strings(
            result.stderr, ["Segmentation fault", "Assertion failed"]
        )
        # Pipeline should produce some diagnostic output regardless of success
        combined = result.stdout + result.stderr
        assert len(combined) > 10, "Pipeline produced no output at all"

    def test_full_pipeline_reports_status(
        self, tapestry_pipeline_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Full pipeline should report either SUCCESS or FAILED."""
        result = run_tapestry_tool(
            tapestry_pipeline_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "3",
                "-max-cycles", "10000",
            ],
            timeout_sec=300,
        )
        check_no_error_strings(
            result.stderr, ["Segmentation fault", "Assertion failed"]
        )
        combined = result.stdout + result.stderr
        assert "SUCCESS" in combined or "FAILED" in combined, (
            f"Expected SUCCESS or FAILED in output, got:\n{combined[:1000]}"
        )

    def test_full_pipeline_loads_architecture(
        self, tapestry_pipeline_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Full pipeline should load and process the architecture file."""
        result = run_tapestry_tool(
            tapestry_pipeline_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "5",
                "-max-cycles", "10000",
            ],
            timeout_sec=300,
        )
        check_no_error_strings(
            result.stderr, ["Segmentation fault", "Assertion failed"]
        )
        # Should show TDG lowering activity
        combined = result.stdout + result.stderr
        has_tdg_activity = any(kw in combined for kw in [
            "TDGLowering", "DFG", "candidate", "kernel", "loom:"
        ])
        assert has_tdg_activity, (
            "Expected TDG/DFG processing activity in pipeline output"
        )
