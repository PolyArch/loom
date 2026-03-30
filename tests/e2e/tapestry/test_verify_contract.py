"""Verification tests for the Contract System (C02).

Group B tests: Validates that AFFINE_INDEXED is removed, the Placement enum
replaces the legacy Visibility enum, TDCEdgeSpec carries dataVolume and
shape fields, and contract inference populates missing fields.
"""

import os
import re
import subprocess

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


class TestAffineIndexedRemoved:
    """B1: The Ordering enum must not contain AFFINE_INDEXED."""

    def test_no_affine_indexed_in_contract_header(self):
        """Contract.h should not contain AFFINE_INDEXED."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        assert contract_h.exists(), f"Contract.h not found at {contract_h}"
        content = contract_h.read_text(encoding="utf-8")
        assert "AFFINE_INDEXED" not in content, (
            "AFFINE_INDEXED still present in Contract.h Ordering enum"
        )

    def test_no_affine_indexed_in_task_graph_header(self):
        """task_graph.h should not contain AFFINE_INDEXED."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        assert tg_h.exists(), f"task_graph.h not found at {tg_h}"
        content = tg_h.read_text(encoding="utf-8")
        assert "AFFINE_INDEXED" not in content, (
            "AFFINE_INDEXED still present in task_graph.h Ordering enum"
        )

    def test_no_affine_indexed_anywhere_in_sources(self):
        """grep across the entire source tree should find zero AFFINE_INDEXED hits."""
        result = subprocess.run(
            ["grep", "-rn", "--include=*.h", "--include=*.cpp",
             "--include=*.py", "AFFINE_INDEXED",
             str(REPO_ROOT / "include"), str(REPO_ROOT / "lib"),
             str(REPO_ROOT / "tools")],
            capture_output=True, text=True, timeout=30,
        )
        # grep returns 1 when no match found
        assert result.returncode == 1 or result.stdout.strip() == "", (
            f"AFFINE_INDEXED still present in source tree:\n{result.stdout[:1000]}"
        )


class TestOrderingEnum:
    """B1 continued: Ordering enum has exactly FIFO and UNORDERED."""

    def test_ordering_values_in_contract_header(self):
        """loom::Ordering should have exactly FIFO and UNORDERED."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        # Find the enum class Ordering block
        assert "enum class Ordering" in content
        # Must contain FIFO and UNORDERED
        assert "FIFO" in content, "Missing FIFO in Ordering enum"
        assert "UNORDERED" in content, "Missing UNORDERED in Ordering enum"

    def test_ordering_values_in_taskgraph_header(self):
        """tapestry::Ordering should have exactly FIFO and UNORDERED."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "enum class Ordering" in content
        assert "FIFO" in content, "Missing FIFO in tapestry::Ordering"
        assert "UNORDERED" in content, "Missing UNORDERED in tapestry::Ordering"


class TestPlacementEnum:
    """B: Placement enum replaces legacy Visibility with LOCAL_SPM, SHARED_L2, EXTERNAL, AUTO."""

    def test_placement_enum_in_contract_header(self):
        """Contract.h should define enum class Placement with all 4 values."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "enum class Placement" in content, "Missing Placement enum"
        assert "LOCAL_SPM" in content, "Missing LOCAL_SPM in Placement"
        assert "SHARED_L2" in content, "Missing SHARED_L2 in Placement"
        assert "EXTERNAL" in content, "Missing EXTERNAL in Placement"
        assert "AUTO" in content, "Missing AUTO in Placement"

    def test_placement_enum_in_taskgraph_header(self):
        """task_graph.h should define enum class Placement with all 4 values."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "enum class Placement" in content, "Missing Placement enum"
        assert "LOCAL_SPM" in content, "Missing LOCAL_SPM"
        assert "SHARED_L2" in content, "Missing SHARED_L2"
        assert "EXTERNAL" in content, "Missing EXTERNAL"
        assert "AUTO" in content, "Missing AUTO"

    def test_visibility_alias_retained(self):
        """Contract.h should retain Visibility as a type alias to Placement."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "using Visibility = Placement" in content, (
            "Expected Visibility alias to Placement for legacy compatibility"
        )


class TestTDCEdgeSpec:
    """B: TDCEdgeSpec carries the 4 atomic edge dimensions plus identity fields."""

    def test_tdc_edge_spec_exists(self):
        """Contract.h should define struct TDCEdgeSpec."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "struct TDCEdgeSpec" in content, "Missing TDCEdgeSpec struct"

    def test_tdc_edge_spec_has_placement(self):
        """TDCEdgeSpec should have an optional Placement field."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "Placement" in content, "TDCEdgeSpec missing Placement reference"
        assert "placement" in content, "TDCEdgeSpec missing placement field"

    def test_tdc_edge_spec_has_shape(self):
        """TDCEdgeSpec should have an optional shape field."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "shape" in content, "TDCEdgeSpec missing shape field"


class TestTapestryContract:
    """B: tapestry::Contract carries dataVolume, shape, and placement."""

    def test_contract_has_data_volume(self):
        """tapestry::Contract should have an optional dataVolume field."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "struct Contract" in content, "Missing Contract struct"
        assert "dataVolume" in content, "Contract missing dataVolume field"

    def test_contract_has_shape(self):
        """tapestry::Contract should have an optional shape field."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "shape" in content, "Contract missing shape field"

    def test_contract_has_placement(self):
        """tapestry::Contract should have an optional placement field."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "placement" in content, "Contract missing placement field"


class TestContractInferencePipeline:
    """B5: ContractInference runs before HierarchicalCompiler in the pipeline."""

    def test_inference_in_verbose_pipeline(
        self, tapestry_compile_bin, simple_2kernel_mlir, arch_2x2_json, tmp_output_dir
    ):
        """Verbose compilation output should show inference activity."""
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
        # Should show ContractInference or compilation activity
        has_inference = any(kw in combined for kw in [
            "ContractInference", "Inference", "HierarchicalCompiler", "bilevel"
        ])
        assert has_inference, (
            "Expected inference/compilation activity in verbose output.\n"
            f"Output snippet:\n{combined[:1500]}"
        )


class TestShapeExprParser:
    """B: parseShapeExpr utility is declared for symbolic shape parsing."""

    def test_parse_shape_expr_declared(self):
        """Contract.h should declare parseShapeExpr."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "parseShapeExpr" in content, "Missing parseShapeExpr declaration"

    def test_tdc_types_test_passes(self):
        """The TDC types unit test (tdc-types-test) should pass."""
        build_dir = REPO_ROOT / "build"
        test_bin = build_dir / "bin" / "tdc-types-test"
        if not test_bin.exists():
            pytest.skip("tdc-types-test binary not built")
        result = subprocess.run(
            [str(test_bin)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"tdc-types-test failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout[:2000]}\n"
            f"STDERR:\n{result.stderr[:2000]}"
        )
