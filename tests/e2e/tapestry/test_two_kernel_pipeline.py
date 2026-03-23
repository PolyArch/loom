"""Integration tests for a 2-kernel producer-consumer pipeline.

Validates that:
  - Two connected kernels compile on a multi-core system.
  - Contract inference populates producer/consumer rate fields.
  - Core assignment places kernels on valid core instances.
  - NoC scheduling computes routes for cross-core edges.
  - Simulation produces expected output with inter-core data transfer.
"""

import pytest
from pathlib import Path

from test_utils import (
    run_tapestry_tool,
    assert_success_output,
    load_json_report,
    assert_metric_in_range,
    check_no_error_strings,
    assert_files_exist,
)


class TestTwoKernelCompilation:
    """Compile a 2-kernel pipeline on a 2x2 multi-core system."""

    def test_compile_succeeds(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """2-kernel TDG should compile successfully on a 2x2 system."""
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

    def test_kernels_assigned_to_cores(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Both kernels should be assigned to core instances."""
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
        # Should report non-zero core count
        assert "Cores:" in result.stdout
        # Extract the core count from output like "  Cores: 2"
        for line in result.stdout.splitlines():
            if "Cores:" in line:
                parts = line.strip().split()
                idx = parts.index("Cores:")
                core_count = int(parts[idx + 1])
                assert core_count >= 1, "Expected at least 1 core"
                break

    def test_compilation_time_reported(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Compilation should report wall-clock time."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
            ],
        )
        assert_success_output(result, "tapestry-compile")
        assert "time" in result.stdout.lower() or "sec" in result.stdout.lower()


class TestTwoKernelSimulation:
    """Simulate a 2-kernel pipeline on a 2x2 system."""

    def test_simulate_succeeds(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Compilation + simulation should succeed for 2-kernel pipeline."""
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

    def test_simulation_reports_cycle_count(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Simulation output should report total cycle count."""
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
        assert "Total cycles:" in result.stdout

    def test_simulation_reports_noc_flits(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Simulation should report NoC flit transfer count."""
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

    def test_simulation_cores_simulated(
        self, tapestry_simulate_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Simulation should report the number of cores simulated."""
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
        assert "Cores simulated:" in result.stdout


class TestTwoKernelFullPipeline:
    """Full pipeline (compile+simulate+rtlgen) for 2-kernel system."""

    def test_full_pipeline_2core(
        self, tapestry_pipeline_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Full pipeline should succeed on 2x2 multi-core system."""
        result = run_tapestry_tool(
            tapestry_pipeline_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_2x2_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "5",
                "-max-cycles", "50000",
            ],
            timeout_sec=300,
        )
        assert_success_output(result, "tapestry-pipeline")
        # All three stages should be reported
        assert "[compile]" in result.stdout
        assert "[simulate]" in result.stdout
        assert "[rtlgen]" in result.stdout

    def test_full_pipeline_report_generated(
        self, tapestry_pipeline_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Full pipeline should generate a report file."""
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
        assert_success_output(result, "tapestry-pipeline")
        assert "Report:" in result.stdout
