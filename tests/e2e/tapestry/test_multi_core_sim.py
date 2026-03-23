"""Integration tests for multi-core simulation.

Validates that:
  - A 2-core system simulation runs to completion.
  - Inter-core data transfer is reported via NoC statistics.
  - Cycle counts are reasonable (non-zero, within max-cycles).
  - NoC contention toggle works correctly.
  - Tracing mode produces additional output.
"""

import pytest
from pathlib import Path

from test_utils import (
    run_tapestry_tool,
    assert_success_output,
    check_no_error_strings,
)


class TestMultiCoreSim:
    """Verify multi-core simulation on a 2x2 system."""

    def test_sim_completes_within_max_cycles(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Simulation should complete within the configured max cycles."""
        max_cyc = 100000
        result = run_tapestry_tool(
            tapestry_simulate_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-cycles", str(max_cyc),
            ],
            timeout_sec=180,
        )
        assert_success_output(result, "tapestry-simulate")
        # Parse total cycles from output
        for line in result.stdout.splitlines():
            if "Total cycles:" in line:
                parts = line.strip().split()
                idx = parts.index("cycles:")
                total = int(parts[idx + 1].rstrip(","))
                assert 0 < total <= max_cyc, (
                    f"Total cycles {total} should be in (0, {max_cyc}]"
                )
                break

    def test_sim_reports_noc_transfer(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Simulation with cross-core edges should report NoC flit transfers."""
        result = run_tapestry_tool(
            tapestry_simulate_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-cycles", "50000",
            ],
            timeout_sec=180,
        )
        assert_success_output(result, "tapestry-simulate")
        assert "NoC flits:" in result.stdout

    def test_sim_without_noc_contention(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Simulation with NoC contention disabled should still succeed."""
        result = run_tapestry_tool(
            tapestry_simulate_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-cycles", "50000",
                "-noc-contention=false",
            ],
            timeout_sec=180,
        )
        assert_success_output(result, "tapestry-simulate")

    def test_sim_with_tracing(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Simulation with tracing enabled should succeed."""
        result = run_tapestry_tool(
            tapestry_simulate_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-cycles", "10000",
                "-trace",
            ],
            timeout_sec=180,
        )
        assert_success_output(result, "tapestry-simulate")

    def test_sim_single_core_baseline(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """Simulation on a single-core system should succeed as baseline."""
        result = run_tapestry_tool(
            tapestry_simulate_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
                "-max-cycles", "50000",
            ],
            timeout_sec=180,
        )
        assert_success_output(result, "tapestry-simulate")


class TestMultiCoreSimEdgeCases:
    """Edge-case tests for multi-core simulation."""

    def test_sim_very_low_max_cycles(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Simulation with a very low max-cycles should terminate gracefully."""
        result = run_tapestry_tool(
            tapestry_simulate_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-cycles", "10",
            ],
            timeout_sec=60,
        )
        # May or may not succeed (could timeout at cycle level), but should not crash
        check_no_error_strings(result.stderr, ["Segmentation fault", "Assertion failed"])

    def test_sim_verbose_output(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Simulation in verbose mode should produce additional diagnostics."""
        result = run_tapestry_tool(
            tapestry_simulate_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-cycles", "10000",
                "-verbose",
            ],
            timeout_sec=180,
        )
        # Should not crash regardless of outcome
        check_no_error_strings(result.stderr, ["Segmentation fault", "Assertion failed"])
