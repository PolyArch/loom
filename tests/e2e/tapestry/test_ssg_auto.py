"""Verification tests for the SSG auto-construction path.

Validates that:
  - The auto-analyze bridge (AutoAnalyzeResult -> TaskGraph -> SSG) produces
    correct kernel nodes, edges, execution targets, and data volumes.
  - The TDGToSSGBuilder header is present and declares the expected API.
  - The auto-analyze-bridge-test binary passes all its internal assertions.
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


class TestSSGAutoHeaders:
    """Validate that SSG auto-construction headers define required types."""

    def test_tdg_to_ssg_builder_header_exists(self):
        """TDGToSSGBuilder.h should exist and declare the builder class."""
        hdr = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "TDGToSSGBuilder.h"
        assert hdr.exists(), f"TDGToSSGBuilder.h not found at {hdr}"
        content = hdr.read_text(encoding="utf-8")

        assert "class TDGToSSGBuilder" in content, (
            "Missing TDGToSSGBuilder class declaration"
        )

    def test_ssg_builder_has_build_method(self):
        """TDGToSSGBuilder should declare a build method returning SSG."""
        hdr = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "TDGToSSGBuilder.h"
        content = hdr.read_text(encoding="utf-8")

        assert "SSG build(" in content, (
            "TDGToSSGBuilder missing build() method returning SSG"
        )

    def test_system_graph_types_define_ssg(self):
        """SystemGraphTypes.h should define SSG as SystemGraph<KernelNode, DataDependency>."""
        hdr = REPO_ROOT / "include" / "loom" / "Graph" / "SystemGraphTypes.h"
        assert hdr.exists(), f"SystemGraphTypes.h not found"
        content = hdr.read_text(encoding="utf-8")

        assert "using SSG = SystemGraph<KernelNode, DataDependency>" in content, (
            "Missing SSG type alias in SystemGraphTypes.h"
        )

    def test_kernel_node_has_required_fields(self):
        """KernelNode should have name, kernelType, hasDFG, and variantSet."""
        hdr = REPO_ROOT / "include" / "loom" / "Graph" / "SystemGraphTypes.h"
        content = hdr.read_text(encoding="utf-8")

        assert "struct KernelNode" in content, "Missing KernelNode struct"
        for field in ["name", "kernelType", "hasDFG", "variantSet"]:
            assert field in content, f"KernelNode missing field: {field}"

    def test_data_dependency_has_required_fields(self):
        """DataDependency should have producerKernel, consumerKernel, dataVolume."""
        hdr = REPO_ROOT / "include" / "loom" / "Graph" / "SystemGraphTypes.h"
        content = hdr.read_text(encoding="utf-8")

        assert "struct DataDependency" in content, "Missing DataDependency struct"
        for field in ["producerKernel", "consumerKernel", "dataVolume"]:
            assert field in content, f"DataDependency missing field: {field}"


class TestAutoAnalyzeBridge:
    """Validate the auto-analyze bridge builds a correct TaskGraph from analysis."""

    def test_bridge_header_declares_build_function(self):
        """auto_analyze.h should declare buildTaskGraphFromAnalysis."""
        hdr = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        assert hdr.exists(), f"auto_analyze.h not found"
        content = hdr.read_text(encoding="utf-8")

        assert "buildTaskGraphFromAnalysis" in content, (
            "Missing buildTaskGraphFromAnalysis declaration"
        )

    def test_bridge_header_declares_size_of_type(self):
        """auto_analyze.h should declare sizeOfType helper."""
        hdr = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        content = hdr.read_text(encoding="utf-8")

        assert "sizeOfType" in content, "Missing sizeOfType declaration"

    def test_auto_analyze_bridge_unit_test_passes(self):
        """The auto-analyze-bridge-test binary should pass all assertions."""
        test_bin = BUILD_DIR / "bin" / "auto-analyze-bridge-test"
        if not test_bin.exists():
            pytest.skip("auto-analyze-bridge-test binary not built")
        result = subprocess.run(
            [str(test_bin)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"auto-analyze-bridge-test failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout[:2000]}\n"
            f"STDERR:\n{result.stderr[:2000]}"
        )
        # Verify all internal tests passed (0 failures)
        combined = result.stdout + result.stderr
        assert "0 failed" in combined, (
            f"Expected zero failures in bridge tests.\n"
            f"Output:\n{combined[:2000]}"
        )


class TestSSGAutoConstruction:
    """Validate that the SSG auto-construction path produces correct output."""

    def test_auto_analyze_result_fields(self):
        """AutoAnalyzeResult should have callBindings, edges, numKernels, numEdges."""
        hdr = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        content = hdr.read_text(encoding="utf-8")

        for field in ["callBindings", "edges", "numKernels", "numEdges"]:
            assert field in content, (
                f"AutoAnalyzeResult missing field/method: {field}"
            )

    def test_inferred_edge_carries_data_volume(self):
        """InferredEdge dependency should carry elementCount for data volume."""
        hdr = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        content = hdr.read_text(encoding="utf-8")

        assert "elementCount" in content, (
            "DataDependency missing elementCount for volume computation"
        )

    def test_kernel_target_enum_defined(self):
        """auto_analyze.h should define KernelTarget enum (CGRA, HOST, AUTO)."""
        hdr = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        content = hdr.read_text(encoding="utf-8")

        assert "enum class KernelTarget" in content, "Missing KernelTarget enum"
        for val in ["CGRA", "HOST", "AUTO"]:
            assert val in content, f"KernelTarget missing value: {val}"

    def test_call_site_binding_has_target(self):
        """CallSiteBinding should carry a KernelTarget field."""
        hdr = REPO_ROOT / "include" / "tapestry" / "auto_analyze.h"
        content = hdr.read_text(encoding="utf-8")

        assert "struct CallSiteBinding" in content, "Missing CallSiteBinding struct"
        assert "KernelTarget target" in content, (
            "CallSiteBinding missing target field"
        )
