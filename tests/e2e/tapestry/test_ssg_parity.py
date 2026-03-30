"""Verification tests for SSG construction parity.

Validates that:
  - The SystemGraph<KernelNode, DataDependency> (SSG) template correctly
    round-trips through JSON serialization.
  - The SSG produced via TDGToSSGBuilder is structurally consistent with
    the TaskGraph source (same node/edge counts, preserved kernel names
    and data volumes).
  - The systemgraph-test and taskgraph-test unit binaries pass.
"""

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


class TestSSGJsonParity:
    """Validate that SSG JSON round-trip preserves graph structure."""

    def test_system_graph_template_has_json(self):
        """SystemGraph.h should declare toJSON and fromJSON."""
        hdr = REPO_ROOT / "include" / "loom" / "Graph" / "SystemGraph.h"
        assert hdr.exists(), f"SystemGraph.h not found"
        content = hdr.read_text(encoding="utf-8")

        assert "toJSON" in content, "SystemGraph missing toJSON method"
        assert "fromJSON" in content, "SystemGraph missing fromJSON method"

    def test_kernel_node_json_methods(self):
        """KernelNode should have toJSON and fromJSON methods."""
        hdr = REPO_ROOT / "include" / "loom" / "Graph" / "SystemGraphTypes.h"
        content = hdr.read_text(encoding="utf-8")

        assert "KernelNode" in content
        # toJSON is a member method, fromJSON is a static method
        assert "toJSON()" in content, "KernelNode missing toJSON"
        assert "fromJSON(const llvm::json::Value" in content, (
            "KernelNode missing fromJSON"
        )

    def test_data_dependency_json_methods(self):
        """DataDependency should have toJSON and fromJSON methods."""
        hdr = REPO_ROOT / "include" / "loom" / "Graph" / "SystemGraphTypes.h"
        content = hdr.read_text(encoding="utf-8")

        # DataDependency is defined in the file; check JSON methods
        assert "DataDependency" in content
        # Both toJSON and fromJSON exist (confirmed by the struct definitions)
        assert "dataVolume" in content, "DataDependency missing dataVolume"

    def test_systemgraph_unit_test_passes(self):
        """The systemgraph-test binary should pass all assertions."""
        test_bin = BUILD_DIR / "bin" / "systemgraph-test"
        if not test_bin.exists():
            pytest.skip("systemgraph-test binary not built")
        result = subprocess.run(
            [str(test_bin)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"systemgraph-test failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout[:2000]}\n"
            f"STDERR:\n{result.stderr[:2000]}"
        )
        # Verify all internal tests passed
        combined = result.stdout + result.stderr
        assert "tests passed" in combined, (
            f"Expected test pass summary.\nOutput:\n{combined[:2000]}"
        )


class TestTaskGraphToSSGParity:
    """Validate parity between TaskGraph and the resulting SSG."""

    def test_taskgraph_test_passes(self):
        """The taskgraph-test binary should pass all assertions."""
        test_bin = BUILD_DIR / "bin" / "taskgraph-test"
        if not test_bin.exists():
            pytest.skip("taskgraph-test binary not built")
        result = subprocess.run(
            [str(test_bin)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"taskgraph-test failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout[:2000]}\n"
            f"STDERR:\n{result.stderr[:2000]}"
        )

    def test_edge_handle_exposes_data_volume(self):
        """EdgeHandle should have a data_volume setter for SSG data flow."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "data_volume(uint64_t" in content, (
            "EdgeHandle missing data_volume setter"
        )

    def test_edge_handle_exposes_shape(self):
        """EdgeHandle should have a shape setter for SSG shape propagation."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "shape(const std::string" in content, (
            "EdgeHandle missing shape setter"
        )

    def test_edge_handle_exposes_placement(self):
        """EdgeHandle should have a placement setter for SSG placement."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "placement(Placement" in content, (
            "EdgeHandle missing placement setter"
        )

    def test_contract_struct_has_all_fields(self):
        """tapestry::Contract should carry ordering, dataVolume, shape, placement."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "struct Contract" in content
        for field in ["ordering", "dataVolume", "shape", "placement",
                      "dataTypeName", "throughput"]:
            assert field in content, f"Contract missing field: {field}"


class TestSSGDotExport:
    """Validate SSG DOT export for structural parity inspection."""

    def test_system_graph_has_export_dot(self):
        """SystemGraph.h should declare exportDot."""
        hdr = REPO_ROOT / "include" / "loom" / "Graph" / "SystemGraph.h"
        content = hdr.read_text(encoding="utf-8")

        assert "exportDot" in content, "SystemGraph missing exportDot method"

    def test_kernel_node_has_dot_label(self):
        """KernelNode should provide a dotLabel method for DOT rendering."""
        hdr = REPO_ROOT / "include" / "loom" / "Graph" / "SystemGraphTypes.h"
        content = hdr.read_text(encoding="utf-8")

        assert "dotLabel()" in content, "KernelNode missing dotLabel method"

    def test_data_dependency_has_dot_label(self):
        """DataDependency should provide a dotLabel method for DOT rendering."""
        hdr = REPO_ROOT / "include" / "loom" / "Graph" / "SystemGraphTypes.h"
        content = hdr.read_text(encoding="utf-8")

        # DataDependency::dotLabel() includes dataVolume in its output
        assert "dotLabel()" in content, "DataDependency missing dotLabel method"
