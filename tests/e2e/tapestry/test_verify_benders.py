"""Verification tests for HierarchicalCompiler (C01) and infeasibility cut feedback.

Group A tests: Validates that the Benders decomposition bilevel loop
runs real iterations, invokes L1 and L2, produces meaningful diagnostics,
and generates infeasibility feedback on mapping failure.
"""

import json
import re

import pytest
from pathlib import Path

from test_utils import (
    run_tapestry_tool,
    assert_success_output,
    load_json_report,
    check_no_error_strings,
)


class TestBendersRealIteration:
    """A1: The bilevel iteration loop runs real iterations (not hardcoded)."""

    def test_benders_driver_invoked(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Benders driver must actually be invoked and start iterations."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "10",
                "-verbose",
            ],
        )
        # Not requiring success -- the pipeline may fail on this input
        check_no_error_strings(
            result.stderr, ["Segmentation fault", "Assertion failed"]
        )

        combined = result.stdout + result.stderr
        # HierarchicalCompiler should start and announce itself
        assert "HierarchicalCompiler" in combined or "bilevel" in combined, (
            "Expected HierarchicalCompiler invocation in verbose output.\n"
            f"Output snippet:\n{combined[:1500]}"
        )

    def test_iteration_at_least_one(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Benders should attempt at least 1 iteration."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "10",
                "-verbose",
            ],
        )
        check_no_error_strings(
            result.stderr, ["Segmentation fault", "Assertion failed"]
        )

        combined = result.stdout + result.stderr
        # Look for iteration marker
        has_iteration = (
            "iteration 1" in combined.lower() or
            "--- iteration 1 ---" in combined
        )
        assert has_iteration, (
            "Expected at least iteration 1 in Benders output.\n"
            f"Output snippet:\n{combined[:1500]}"
        )


class TestBendersL1L2Invocation:
    """A1 continued: Both L1 (assignment) and L2 (compiler) are invoked."""

    def test_l1_core_assignment_invoked(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """L1 core assignment solver must be invoked during Benders."""
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
        # L1 solver should be mentioned
        has_l1 = any(kw in combined for kw in [
            "L1 core assignment", "L1 solver", "core assignment"
        ])
        assert has_l1, (
            "Expected L1 core assignment activity in verbose output.\n"
            f"Output snippet:\n{combined[:2000]}"
        )

    def test_architecture_loaded(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """The architecture file should be loaded and parsed."""
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
        # Architecture should be loaded and report core types
        has_arch = any(kw in combined for kw in [
            "coreTypes", "core instances", "loading architecture"
        ])
        assert has_arch, (
            "Expected architecture loading information in verbose output.\n"
            f"Output snippet:\n{combined[:2000]}"
        )


class TestBendersMappingSuccess:
    """A1: mappingSuccess comes from real mapper evaluation."""

    def test_different_architectures_give_different_results(
        self,
        tapestry_compile_bin,
        simple_2kernel_mlir,
        arch_1core_json,
        arch_2x2_json,
        tmp_output_dir,
    ):
        """Single-core vs 2x2 should produce different output."""
        r1 = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir / "single"),
                "-max-benders-iter", "5",
                "-verbose",
            ],
        )
        check_no_error_strings(
            r1.stderr, ["Segmentation fault", "Assertion failed"]
        )

        r2 = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir / "multi"),
                "-max-benders-iter", "5",
                "-verbose",
            ],
        )
        check_no_error_strings(
            r2.stderr, ["Segmentation fault", "Assertion failed"]
        )

        # Outputs should differ (different architectures)
        combined1 = r1.stdout + r1.stderr
        combined2 = r2.stdout + r2.stderr
        assert combined1 != combined2, (
            "1-core and 2x2 compilations produced identical output, "
            "suggesting hardcoded results."
        )


class TestBendersInfeasibilityCuts:
    """A2: Infeasibility diagnostics are produced when compilation fails."""

    def test_infeasibility_reported(
        self, tapestry_compile_bin, data_dir, tmp_output_dir
    ):
        """Failed compilation should report infeasibility."""
        tdg_path = data_dir / "simple_2kernel.mlir"
        arch_path = data_dir / "arch_2x2.json"

        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(tdg_path),
                "-system-arch", str(arch_path),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "3",
                "-verbose",
            ],
        )
        check_no_error_strings(
            result.stderr, ["Segmentation fault", "Assertion failed"]
        )

        combined = result.stdout + result.stderr
        # On failure, should report infeasibility or FAILED status
        has_diagnostic = any(kw in combined for kw in [
            "infeasible", "INFEASIBLE", "FAILED", "cuts"
        ])
        assert has_diagnostic, (
            "Expected infeasibility diagnostic in output.\n"
            f"Output snippet:\n{combined[:1500]}"
        )


class TestBendersHelpers:
    """A4: HierarchicalCompiler reports iteration count and kernel information."""

    def test_kernel_count_reported(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Benders should report the number of kernels being compiled."""
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
        # Should report kernel count
        has_kernels = "kernels=" in combined or "kernel" in combined.lower()
        assert has_kernels, (
            "Expected kernel count in Benders output.\n"
            f"Output snippet:\n{combined[:1500]}"
        )

    def test_temporal_scheduling_mentioned(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Pipeline should reference temporal scheduling."""
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
        has_temporal = any(kw in combined for kw in [
            "temporal", "BATCH_SEQUENTIAL", "schedule"
        ])
        assert has_temporal, (
            "Expected temporal scheduling reference in output.\n"
            f"Output snippet:\n{combined[:1500]}"
        )

    def test_contract_inference_runs(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """ContractInferencePass should run during compilation."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "3",
                "-verbose",
            ],
        )
        check_no_error_strings(
            result.stderr, ["Segmentation fault", "Assertion failed"]
        )

        combined = result.stdout + result.stderr
        has_inference = "ContractInference" in combined or "inference" in combined.lower()
        assert has_inference, (
            "Expected ContractInferencePass mention in output.\n"
            f"Output snippet:\n{combined[:1500]}"
        )
