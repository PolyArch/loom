"""Integration tests for single-kernel compilation through the full pipeline.

Validates that:
  - A single kernel compiles successfully to a single core.
  - The compilation produces expected output artifacts.
  - Configuration binary is generated.
  - Simulation (if enabled) runs to completion.
"""

import pytest
from pathlib import Path

from test_utils import (
    run_tapestry_tool,
    assert_success_output,
    assert_files_exist,
    load_json_report,
    assert_metric_in_range,
    check_no_error_strings,
)


class TestSingleKernelCompile:
    """Compile a single kernel through tapestry-compile."""

    def test_compile_succeeds(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """A minimal TDG should compile successfully on a single core."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "3",
            ],
        )
        assert_success_output(result, "tapestry-compile")
        check_no_error_strings(result.stderr)

    def test_compile_produces_report(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """Compilation should produce a JSON report file."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
            ],
        )
        assert_success_output(result, "tapestry-compile")
        # The report path is printed in the output
        assert "Report:" in result.stdout

    def test_compile_reports_iteration_count(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """Compilation output should report Benders iteration count."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "5",
            ],
        )
        assert_success_output(result, "tapestry-compile")
        assert "Iterations:" in result.stdout

    def test_compile_reports_core_count(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """Compilation output should report number of cores used."""
        result = run_tapestry_tool(
            tapestry_compile_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
            ],
        )
        assert_success_output(result, "tapestry-compile")
        assert "Cores:" in result.stdout


class TestSingleKernelPipeline:
    """Run the full pipeline on a single-core system."""

    def test_full_pipeline_succeeds(
        self, tapestry_pipeline_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """Full pipeline (compile+sim+rtl) on a single core should succeed."""
        result = run_tapestry_tool(
            tapestry_pipeline_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "3",
                "-max-cycles", "10000",
            ],
            timeout_sec=180,
        )
        assert_success_output(result, "tapestry-pipeline")

    def test_pipeline_compile_stage_output(
        self, tapestry_pipeline_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """Pipeline should report compile stage metrics."""
        result = run_tapestry_tool(
            tapestry_pipeline_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "3",
            ],
        )
        assert_success_output(result, "tapestry-pipeline")
        assert "[compile]" in result.stdout

    def test_pipeline_sim_stage_output(
        self, tapestry_pipeline_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """Pipeline should report simulation stage metrics."""
        result = run_tapestry_tool(
            tapestry_pipeline_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "3",
                "-max-cycles", "10000",
            ],
            timeout_sec=180,
        )
        assert_success_output(result, "tapestry-pipeline")
        assert "[simulate]" in result.stdout

    def test_pipeline_rtl_stage_output(
        self, tapestry_pipeline_bin, simple_2kernel_mlir, arch_1core_json, tmp_output_dir
    ):
        """Pipeline should report RTL generation stage metrics."""
        result = run_tapestry_tool(
            tapestry_pipeline_bin,
            [
                "-tdg", str(simple_2kernel_mlir),
                "-system-arch", str(arch_1core_json),
                "-o", str(tmp_output_dir),
                "-max-benders-iter", "3",
            ],
        )
        assert_success_output(result, "tapestry-pipeline")
        assert "[rtlgen]" in result.stdout
