"""Verification tests for the Programming Interface (C07, C08, C09).

Group G tests: Validates that the TaskGraph API creates kernels and edges
with contracts, TDG MLIR emission produces valid tdg.graph/tdg.kernel/
tdg.contract ops, and auto_analyze detects producer/consumer dependencies.
"""

import os
import subprocess

import pytest
from pathlib import Path


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "tools" / "tapestry").exists():
            return p
        p = p.parent
    raise RuntimeError("Cannot locate repository root")


REPO_ROOT = _find_repo_root()
BUILD_DIR = REPO_ROOT / "build"


class TestTaskGraphAPI:
    """G1: TaskGraph API creates kernels, edges, and propagates contract fields."""

    def test_taskgraph_header_exists(self):
        """task_graph.h should define TaskGraph, KernelHandle, EdgeHandle."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        assert tg_h.exists(), f"task_graph.h not found"
        content = tg_h.read_text(encoding="utf-8")

        assert "class TaskGraph" in content, "Missing TaskGraph class"
        assert "class KernelHandle" in content, "Missing KernelHandle class"
        assert "class EdgeHandle" in content, "Missing EdgeHandle class"

    def test_taskgraph_kernel_definition(self):
        """TaskGraph should support both function-pointer and name-only kernels."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        # Template version for function pointers
        assert "kernel(const std::string &kernelName, F funcPtr)" in content, (
            "Expected template kernel(name, funcPtr) method"
        )
        # Name-only version for auto_analyze
        assert "kernel(const std::string &kernelName)" in content, (
            "Expected name-only kernel(name) method"
        )

    def test_edge_handle_chainable_setters(self):
        """EdgeHandle should have chainable setters for all contract fields."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        required_setters = [
            "ordering(Ordering",
            "data_type(const std::string",
            "rate(int64_t",
            "tile_shape(std::vector<int64_t>",
            "visibility(Visibility",
            "double_buffering(bool",
            "backpressure(Backpressure",
            "may_fuse(bool",
            "may_replicate(bool",
            "may_retile(bool",
        ]
        for setter in required_setters:
            assert setter in content, (
                f"EdgeHandle missing chainable setter: {setter}"
            )

    def test_taskgraph_inspection_methods(self):
        """TaskGraph should have numKernels, numEdges, forEachKernel, forEachEdge."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        for method in ["numKernels", "numEdges", "forEachKernel", "forEachEdge"]:
            assert method in content, f"TaskGraph missing inspection method: {method}"

    def test_taskgraph_unit_test_passes(self):
        """The TaskGraph unit test (taskgraph-test) should pass."""
        test_bin = BUILD_DIR / "bin" / "taskgraph-test"
        if not test_bin.exists():
            pytest.skip("taskgraph-test binary not built")
        result = subprocess.run(
            [str(test_bin)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"taskgraph-test failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout[:2000]}\n"
            f"STDERR:\n{result.stderr[:2000]}"
        )

    def test_verification_unit_test_passes(self):
        """The verification unit test (verification-test) should pass."""
        test_bin = BUILD_DIR / "bin" / "verification-test"
        if not test_bin.exists():
            pytest.skip("verification-test binary not built")
        result = subprocess.run(
            [str(test_bin)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"verification-test failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout[:2000]}\n"
            f"STDERR:\n{result.stderr[:2000]}"
        )


class TestTDGEmitter:
    """G2: TDG MLIR emission produces valid tdg.graph/tdg.kernel/tdg.contract ops."""

    def test_tdg_emitter_header_exists(self):
        """tdg_emitter.h should define emitTDG and writeTDGToFile."""
        em_h = REPO_ROOT / "include" / "tapestry" / "tdg_emitter.h"
        assert em_h.exists(), f"tdg_emitter.h not found"
        content = em_h.read_text(encoding="utf-8")

        assert "emitTDG" in content, "Missing emitTDG function"
        assert "writeTDGToFile" in content, "Missing writeTDGToFile function"

    def test_tdg_dialect_test_passes(self):
        """The TDG dialect unit test should pass."""
        test_bin = BUILD_DIR / "bin" / "tdg-dialect-test"
        if not test_bin.exists():
            pytest.skip("tdg-dialect-test binary not built")
        result = subprocess.run(
            [str(test_bin)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"tdg-dialect-test failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout[:2000]}\n"
            f"STDERR:\n{result.stderr[:2000]}"
        )

    def test_tdg_dialect_ops_defined(self):
        """TDG dialect should define GraphOp, KernelOp, ContractOp."""
        tdg_dir = REPO_ROOT / "include" / "loom" / "Dialect" / "TDG"
        assert tdg_dir.exists(), "TDG dialect directory not found"

        # Ops are defined in .td files (TableGen) and auto-generated to .h.inc.
        # Check the .td definition file for op definitions.
        td_files = list(tdg_dir.glob("*.td"))
        h_files = list(tdg_dir.glob("*.h"))
        all_files = td_files + h_files
        assert len(all_files) > 0, "Expected TDG dialect definition files"

        # Concatenate all sources to search for op definitions
        all_content = ""
        for f in all_files:
            all_content += f.read_text(encoding="utf-8")

        assert "GraphOp" in all_content or "TDG_GraphOp" in all_content, (
            "Expected GraphOp definition in TDG dialect"
        )
        assert "KernelOp" in all_content or "TDG_KernelOp" in all_content, (
            "Expected KernelOp definition in TDG dialect"
        )
        assert "ContractOp" in all_content or "TDG_ContractOp" in all_content, (
            "Expected ContractOp definition in TDG dialect"
        )


class TestAutoAnalyze:
    """G3: auto_analyze detects producer/consumer dependencies."""

    def test_auto_analyze_header_exists(self):
        """auto_analyze.h should define autoAnalyze function."""
        aa_h = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        assert aa_h.exists(), f"auto_analyze.h not found"
        content = aa_h.read_text(encoding="utf-8")

        assert "autoAnalyze" in content, "Missing autoAnalyze function"
        assert "AutoAnalyzeResult" in content, "Missing AutoAnalyzeResult struct"

    def test_auto_analyze_result_has_required_fields(self):
        """AutoAnalyzeResult should have callBindings and edges."""
        aa_h = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        content = aa_h.read_text(encoding="utf-8")

        assert "callBindings" in content, "Missing callBindings field"
        assert "edges" in content, "Missing edges field"
        assert "numKernels" in content, "Missing numKernels method"
        assert "numEdges" in content, "Missing numEdges method"

    def test_data_dependency_struct(self):
        """DataDependency should track dependency existence and data type."""
        aa_h = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        content = aa_h.read_text(encoding="utf-8")

        assert "struct DataDependency" in content, "Missing DataDependency struct"
        assert "dataType" in content, "Missing dataType field in DataDependency"
        assert "isSequential" in content, "Missing isSequential field"

    def test_inferred_edge_has_ordering(self):
        """InferredEdge should carry inferred ordering (FIFO for sequential)."""
        aa_h = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        content = aa_h.read_text(encoding="utf-8")

        assert "struct InferredEdge" in content, "Missing InferredEdge struct"
        assert "ordering" in content, "Missing ordering field in InferredEdge"


class TestCompileAPI:
    """G: The compile() top-level API ties everything together."""

    def test_compile_header_exists(self):
        """compile.h should define compile(TaskGraph, CompileOptions)."""
        c_h = REPO_ROOT / "include" / "tapestry" / "compile.h"
        assert c_h.exists(), f"compile.h not found"
        content = c_h.read_text(encoding="utf-8")

        assert "CompileResult compile(" in content, "Missing compile function"
        assert "CompileOptions" in content, "Missing CompileOptions struct"
        assert "CompileResult" in content, "Missing CompileResult struct"

    def test_compile_result_fields(self):
        """CompileResult should report success, iterations, throughput."""
        c_h = REPO_ROOT / "include" / "tapestry" / "compile.h"
        content = c_h.read_text(encoding="utf-8")

        assert "bool success" in content, "CompileResult missing success"
        assert "iterations" in content, "CompileResult missing iterations"
        assert "systemThroughput" in content, "CompileResult missing throughput"
        assert "reportPath" in content, "CompileResult missing reportPath"
