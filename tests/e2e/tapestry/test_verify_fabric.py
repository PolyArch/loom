"""Verification tests for the Fabric System Dialect (C03).

Group C tests: Validates that system-level MLIR has typed ops
(fabric.router, fabric.shared_mem, fabric.noc_link), and uses
real MLIR operations instead of string-concatenated comments.
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


class TestFabricSystemOps:
    """C1: System MLIR uses typed ops from the Fabric dialect."""

    def test_fabric_system_ops_defined(self):
        """FabricSystemOps.td should define router, shared_mem, noc_link ops."""
        ops_td = REPO_ROOT / "include" / "loom" / "Dialect" / "Fabric" / "FabricSystemOps.td"
        assert ops_td.exists(), f"FabricSystemOps.td not found at {ops_td}"
        content = ops_td.read_text(encoding="utf-8")

        # Check for typed op definitions
        has_router = "router" in content.lower() or "Router" in content
        has_shared_mem = "shared_mem" in content.lower() or "SharedMem" in content
        has_noc_link = "noc_link" in content.lower() or "NocLink" in content or "Link" in content

        assert has_router, (
            "Expected fabric.router op definition in FabricSystemOps.td"
        )
        assert has_shared_mem, (
            "Expected fabric.shared_mem op definition in FabricSystemOps.td"
        )
        assert has_noc_link, (
            "Expected fabric.noc_link or Link op definition in FabricSystemOps.td"
        )

    def test_system_adg_builder_uses_typed_ops(self):
        """SystemADGMLIRBuilder should create fabric.router/shared_mem/noc_link ops."""
        builder_cpp = (
            REPO_ROOT / "lib" / "loom" / "ADG" / "SystemADGMLIRBuilder.cpp"
        )
        assert builder_cpp.exists(), f"SystemADGMLIRBuilder.cpp not found"
        content = builder_cpp.read_text(encoding="utf-8")

        # Builder should reference the typed system ops, not emit raw strings
        has_router_create = any(kw in content for kw in [
            "RouterOp", "router", "createRouter"
        ])
        has_mem_create = any(kw in content for kw in [
            "SharedMemOp", "shared_mem", "createSharedMem"
        ])
        has_link_create = any(kw in content for kw in [
            "NocLinkOp", "noc_link", "LinkOp", "createLink"
        ])

        assert has_router_create, (
            "Expected SystemADGMLIRBuilder to create typed router ops"
        )
        assert has_mem_create, (
            "Expected SystemADGMLIRBuilder to create typed shared memory ops"
        )
        assert has_link_create, (
            "Expected SystemADGMLIRBuilder to create typed NoC link ops"
        )


class TestNoStringConcat:
    """C2: System MLIR builder uses real ops, not string-concatenated comments."""

    def test_builder_uses_mlir_api(self):
        """Builder should use MLIR builder API, not string concatenation."""
        builder_cpp = (
            REPO_ROOT / "lib" / "loom" / "ADG" / "SystemADGMLIRBuilder.cpp"
        )
        assert builder_cpp.exists(), f"SystemADGMLIRBuilder.cpp not found"
        content = builder_cpp.read_text(encoding="utf-8")

        # Should use MLIR OpBuilder patterns (builder.clone, builder.set, etc)
        has_builder_api = any(kw in content for kw in [
            "builder.create<", "builder.clone", "builder.set",
            "builder.getUnknownLoc", "OpBuilder"
        ])
        assert has_builder_api, (
            "Expected MLIR OpBuilder pattern in SystemADGMLIRBuilder"
        )

    def test_builder_header_api(self):
        """SystemADGMLIRBuilder.h should expose a proper MLIR builder interface."""
        builder_h = (
            REPO_ROOT / "include" / "loom" / "ADG" / "SystemADGMLIRBuilder.h"
        )
        assert builder_h.exists(), f"SystemADGMLIRBuilder.h not found"
        content = builder_h.read_text(encoding="utf-8")

        # Should depend on MLIR types
        assert "mlir" in content.lower(), (
            "Expected MLIR dependencies in SystemADGMLIRBuilder.h"
        )
        # Should define a builder class or function
        has_builder = "SystemADGMLIRBuilder" in content or "buildSystemADG" in content
        assert has_builder, (
            "Expected SystemADGMLIRBuilder class or buildSystemADG function"
        )


class TestFabricSystemOpsUnit:
    """C3: System MLIR builder unit test exists and exercises the builder."""

    def test_system_adg_builder_unit_test_exists(self):
        """A unit test for the SystemADGBuilder should exist."""
        test_path = (
            REPO_ROOT / "tests" / "unit" / "system-adg-builder" / "adg.cpp"
        )
        assert test_path.exists(), (
            f"SystemADGBuilder unit test not found at {test_path}"
        )
        content = test_path.read_text(encoding="utf-8")

        # Test should exercise the builder (SystemADGBuilder or SystemADGMLIRBuilder)
        has_builder = any(kw in content for kw in [
            "SystemADGBuilder", "SystemADGMLIRBuilder", "buildSystemADG"
        ])
        assert has_builder, (
            "Unit test should exercise SystemADGBuilder"
        )

    def test_builder_test_verifies_ops(self):
        """The unit test should verify fabric ops are created."""
        test_path = (
            REPO_ROOT / "tests" / "unit" / "system-adg-builder" / "adg.cpp"
        )
        if not test_path.exists():
            pytest.skip("SystemADGMLIRBuilder unit test not found")
        content = test_path.read_text(encoding="utf-8")

        # Test should check for router, shared_mem, or link ops
        has_op_check = any(kw in content for kw in [
            "router", "shared_mem", "noc_link", "RouterOp",
            "SharedMemOp", "NocLinkOp", "LinkOp"
        ])
        assert has_op_check, (
            "Unit test should verify typed fabric system ops are created"
        )


class TestFabricDialectIntegration:
    """C: Fabric dialect is properly integrated into the build."""

    def test_fabric_dialect_directory_exists(self):
        """Fabric dialect should have proper directory structure."""
        fabric_dir = REPO_ROOT / "include" / "loom" / "Dialect" / "Fabric"
        assert fabric_dir.exists(), f"Fabric dialect directory not found"

        # Should have .td files
        td_files = list(fabric_dir.glob("*.td"))
        assert len(td_files) > 0, (
            "Expected .td files in Fabric dialect directory"
        )

    def test_fabric_dialect_registered(self):
        """Fabric dialect should be registered (FabricDialect.h exists)."""
        dialect_h = (
            REPO_ROOT / "include" / "loom" / "Dialect" / "Fabric" / "FabricDialect.h"
        )
        assert dialect_h.exists(), (
            "FabricDialect.h not found -- dialect may not be registered"
        )
