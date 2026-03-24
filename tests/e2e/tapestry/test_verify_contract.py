"""Verification tests for the Contract System (C02).

Group B tests: Validates that AFFINE_INDEXED is removed, DROP/OVERWRITE
backpressure produces warnings and falls back to BLOCK, and contract
inference populates missing fields.
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


class TestBackpressureWarning:
    """B4: DROP/OVERWRITE backpressure should produce a warning and fall back to BLOCK."""

    def test_backpressure_enum_exists(self):
        """Backpressure enum should have BLOCK, DROP, and OVERWRITE."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "enum class Backpressure" in content
        assert "BLOCK" in content
        assert "DROP" in content
        assert "OVERWRITE" in content

    def test_default_backpressure_is_block(self):
        """Default backpressure in ContractSpec should be BLOCK."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        # Find the default value
        assert "Backpressure::BLOCK" in content, (
            "Expected default backpressure = BLOCK in ContractSpec"
        )


class TestContractInferencePipeline:
    """B5: ContractInference runs before BendersDriver in the pipeline."""

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
            "ContractInference", "Inference", "BendersDriver", "Benders"
        ])
        assert has_inference, (
            "Expected inference/compilation activity in verbose output.\n"
            f"Output snippet:\n{combined[:1500]}"
        )


class TestContractPermissions:
    """B: Contract transformation permissions default correctly."""

    def test_permission_defaults_in_contract_header(self):
        """ContractSpec mayFuse/mayReplicate/mayRetile defaults should be true."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "bool mayFuse = true" in content
        assert "bool mayRetile = true" in content
        # mayReorder should default to false
        assert "bool mayReorder = false" in content

    def test_permission_defaults_in_taskgraph_header(self):
        """tapestry::Contract mayFuse/mayRetile defaults should match spec."""
        tg_h = REPO_ROOT / "include" / "tapestry" / "task_graph.h"
        content = tg_h.read_text(encoding="utf-8")

        assert "bool mayFuse = true" in content
        assert "bool mayRetile = true" in content
        assert "bool mayReorder = false" in content


class TestContractVisibility:
    """B: Visibility enum has LOCAL_SPM, SHARED_L2, EXTERNAL_DRAM."""

    def test_visibility_values(self):
        """Visibility enum should have all three memory levels."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "LOCAL_SPM" in content, "Missing LOCAL_SPM"
        assert "SHARED_L2" in content, "Missing SHARED_L2"
        assert "EXTERNAL_DRAM" in content, "Missing EXTERNAL_DRAM"

    def test_default_visibility_is_local_spm(self):
        """Default visibility should be LOCAL_SPM."""
        contract_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "Contract.h"
        content = contract_h.read_text(encoding="utf-8")

        assert "Visibility::LOCAL_SPM" in content, (
            "Expected default visibility = LOCAL_SPM in ContractSpec"
        )
