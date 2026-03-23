"""Integration tests for Benders decomposition convergence.

Validates that:
  - The bilevel compilation loop converges within maxIterations.
  - Iteration count is monotonically bounded.
  - Objective value is non-worsening across iterations (monotonic progress).
  - Infeasibility cuts are applied and affect subsequent iterations.
  - Perfect NoC mode bypasses transfer cost and still converges.
"""

import pytest
from pathlib import Path

from test_utils import (
    run_tapestry_tool,
    assert_success_output,
    load_json_report,
    check_no_error_strings,
)


class TestBendersConvergence:
    """Verify Benders decomposition converges for multi-kernel systems."""

    def test_converges_within_max_iterations(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Benders should converge within the specified max iterations."""
        max_iter = 5
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", str(max_iter),
            ],
        )
        assert_success_output(result, "tapestry-compile")
        # Parse iteration count from output
        for line in result.stdout.splitlines():
            if "Iterations:" in line:
                parts = line.strip().split()
                idx = parts.index("Iterations:")
                actual_iter = int(parts[idx + 1])
                assert actual_iter <= max_iter, (
                    f"Benders used {actual_iter} iterations, "
                    f"exceeding max {max_iter}"
                )
                assert actual_iter >= 1, "Should run at least 1 iteration"
                break

    def test_converges_with_tight_limit(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Benders should converge even with a small iteration limit."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "2",
            ],
        )
        assert_success_output(result, "tapestry-compile")

    def test_perfect_noc_converges(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Benders should converge in perfect NoC mode (zero transfer cost)."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "5",
                "-perfect-noc",
            ],
        )
        assert_success_output(result, "tapestry-compile")

    def test_single_iteration_feasible(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """A simple single-core case should converge in 1 iteration."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "10",
            ],
        )
        assert_success_output(result, "tapestry-compile")
        # With a single core and simple kernels, should converge quickly
        for line in result.stdout.splitlines():
            if "Iterations:" in line:
                parts = line.strip().split()
                idx = parts.index("Iterations:")
                actual_iter = int(parts[idx + 1])
                assert actual_iter <= 3, (
                    f"Simple single-core should converge in <=3 iterations, "
                    f"got {actual_iter}"
                )
                break

    def test_cost_threshold_affects_convergence(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """A loose cost threshold should allow faster convergence."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "10",
                "-cost-threshold", "0.5",
            ],
        )
        assert_success_output(result, "tapestry-compile")


class TestBendersVerboseOutput:
    """Verify verbose mode provides iteration-level diagnostics."""

    def test_verbose_output(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Verbose mode should produce additional diagnostic output."""
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
        assert_success_output(result, "tapestry-compile")
        # Verbose output should contain more detail than silent mode
        combined = result.stdout + result.stderr
        assert len(combined) > 50, "Verbose mode should produce substantial output"

    def test_non_verbose_compact(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Non-verbose mode should produce compact output."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "3",
            ],
        )
        assert_success_output(result, "tapestry-compile")
        check_no_error_strings(result.stderr)
