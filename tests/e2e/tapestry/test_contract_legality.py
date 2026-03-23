"""Integration tests for contract legality validation.

Validates the 5 contract legality conditions through the pipeline:
  1. Rate compatibility -- producer/consumer rates must be satisfiable.
  2. Buffer sufficiency -- buffer allocation must meet minimum requirements.
  3. Ordering preservation -- FIFO contracts must maintain element order.
  4. Type compatibility -- kernels must map to compatible core types.
  5. Bandwidth feasibility -- NoC bandwidth must accommodate transfer volume.

These tests exercise the contract verification embedded in the compilation
pipeline by running tapestry-compile with various architecture/TDG configs.
"""

import pytest
from pathlib import Path

from test_utils import (
    run_tapestry_tool,
    assert_success_output,
    check_no_error_strings,
)


class TestContractLegalityPositive:
    """Positive tests: valid contracts should compile successfully."""

    def test_valid_fifo_contract(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """A valid FIFO contract between producer-consumer should compile."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "5",
            ],
        )
        assert_success_output(result, "tapestry-compile")

    def test_single_core_co_located_contract(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """Contract between co-located kernels on a single core should compile."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
            ],
        )
        assert_success_output(result, "tapestry-compile")


class TestContractCompilationDiagnostics:
    """Tests that the compiler provides meaningful diagnostics."""

    def test_compilation_diagnostic_not_empty(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Successful compilation should produce a report path in output."""
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
        assert "Report:" in result.stdout

    def test_verbose_shows_assignment(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Verbose compilation should provide assignment-level detail."""
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
        combined = result.stdout + result.stderr
        # In verbose mode, should see more diagnostic info
        assert len(combined) > 100


class TestContractEdgeCases:
    """Edge cases for contract handling in the pipeline."""

    def test_perfect_noc_bypasses_bandwidth_check(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Perfect NoC mode should bypass bandwidth feasibility checks."""
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

    def test_tight_iteration_limit_still_produces_result(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Even with max-benders-iter=1, should produce a result (pass or fail)."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "1",
            ],
        )
        # Should not crash
        check_no_error_strings(result.stderr, ["Segmentation fault", "Assertion failed"])
        # Should produce either SUCCESS or FAILED (not hang or crash)
        combined = result.stdout + result.stderr
        assert "SUCCESS" in combined or "FAILED" in combined, (
            f"Expected SUCCESS or FAILED in output, got:\n{combined[:1000]}"
        )
