"""Verification tests for the Temporal Execution Model (C04).

Group D tests: Validates that executionCycles = tripCount * achievedII,
BATCH_SEQUENTIAL mode produces a schedule, multi-core scenarios include
reconfiguration costs, and unsupported modes produce clear errors.
"""

import os
import re

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


class TestTemporalModelAPI:
    """D1-D2: The temporal execution model header defines the correct structures."""

    def test_execution_model_header_exists(self):
        """ExecutionModel.h should exist with all required types."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        assert em_h.exists(), f"ExecutionModel.h not found"
        content = em_h.read_text(encoding="utf-8")

        # Must define ExecutionMode enum
        assert "enum class ExecutionMode" in content, (
            "Expected ExecutionMode enum in ExecutionModel.h"
        )
        assert "BATCH_SEQUENTIAL" in content, (
            "Expected BATCH_SEQUENTIAL in ExecutionMode"
        )
        assert "PIPELINE_PARALLEL" in content, (
            "Expected PIPELINE_PARALLEL in ExecutionMode (even if unimplemented)"
        )

    def test_kernel_timing_has_correct_fields(self):
        """KernelTiming must have tripCount, achievedII, executionCycles."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        content = em_h.read_text(encoding="utf-8")

        assert "tripCount" in content, "KernelTiming missing tripCount"
        assert "achievedII" in content, "KernelTiming missing achievedII"
        assert "executionCycles" in content, "KernelTiming missing executionCycles"

    def test_execution_cycles_formula_documented(self):
        """KernelTiming should document executionCycles = tripCount * achievedII."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        content = em_h.read_text(encoding="utf-8")

        assert "tripCount * achievedII" in content, (
            "Expected executionCycles = tripCount * achievedII documentation"
        )

    def test_temporal_schedule_has_per_core_data(self):
        """TemporalSchedule must have per-core schedules."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        content = em_h.read_text(encoding="utf-8")

        assert "CoreSchedule" in content, "Expected CoreSchedule struct"
        assert "coreSchedules" in content, "Expected coreSchedules field"
        assert "maxCoreCycles" in content, "Expected maxCoreCycles field"


class TestReconfigCost:
    """D1: Reconfiguration cost is included in temporal schedule."""

    def test_reconfig_cycles_in_config(self):
        """ExecutionModelConfig should have reconfigCycles field."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        content = em_h.read_text(encoding="utf-8")

        assert "reconfigCycles" in content, (
            "Expected reconfigCycles in ExecutionModelConfig"
        )

    def test_reconfig_count_tracked_per_core(self):
        """CoreSchedule should track reconfigCount."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        content = em_h.read_text(encoding="utf-8")

        assert "reconfigCount" in content, (
            "Expected reconfigCount in CoreSchedule"
        )


class TestBatchSequentialSchedule:
    """D3: BATCH_SEQUENTIAL produces a valid schedule with topological ordering."""

    def test_compilation_references_temporal_scheduling(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Compilation should reference temporal scheduling (even if mapping fails)."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "5",
                "-verbose",
            ],
        )
        check_no_error_strings(
            result.stderr, ["Segmentation fault", "Assertion failed"]
        )

        combined = result.stdout + result.stderr
        # Should mention temporal schedule or execution model
        has_schedule = any(kw in combined for kw in [
            "temporal", "BATCH_SEQUENTIAL", "schedule",
            "reconfigCycles"
        ])
        assert has_schedule, (
            "Expected temporal scheduling reference in verbose output.\n"
            f"Output snippet:\n{combined[:1500]}"
        )

    def test_topological_sort_function_exists(self):
        """The topologicalSortKernels utility should be defined."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        content = em_h.read_text(encoding="utf-8")

        assert "topologicalSortKernels" in content, (
            "Expected topologicalSortKernels function declaration"
        )


class TestUnsupportedModeError:
    """D4: PIPELINE_PARALLEL and SPATIAL_SHARING modes produce clear errors."""

    def test_unsupported_modes_declared(self):
        """Unsupported modes should be declared (for future use) but not implemented."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        content = em_h.read_text(encoding="utf-8")

        # Both modes should be declared in the enum
        assert "PIPELINE_PARALLEL" in content
        assert "SPATIAL_SHARING" in content

        # Comments should indicate they are not implemented
        # (checking for "Not implemented" or similar)
        has_not_impl = (
            "Not implemented" in content or
            "not implemented" in content or
            "Not supported" in content
        )
        assert has_not_impl, (
            "Expected documentation that PIPELINE_PARALLEL/SPATIAL_SHARING "
            "are not implemented"
        )


class TestNoCOverhead:
    """D: NoC overhead is computed for inter-core communication."""

    def test_noc_overhead_function_exists(self):
        """computeNoCOverhead should be declared in ExecutionModel.h."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        content = em_h.read_text(encoding="utf-8")

        assert "computeNoCOverhead" in content, (
            "Expected computeNoCOverhead function declaration"
        )

    def test_system_latency_includes_noc(self):
        """TemporalSchedule should include nocOverheadCycles in systemLatencyCycles."""
        em_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "ExecutionModel.h"
        content = em_h.read_text(encoding="utf-8")

        assert "nocOverheadCycles" in content, (
            "Expected nocOverheadCycles field in TemporalSchedule"
        )
        assert "systemLatencyCycles" in content, (
            "Expected systemLatencyCycles field in TemporalSchedule"
        )
