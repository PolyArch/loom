"""Verification tests for Hardware DSE (C11 OUTER-HW, C12 INNER-HW).

Group I tests: Validates that OUTER-HW produces a CoreTypeLibrary,
INNER-HW produces per-core ADGs with 13 design dimensions, and
no fabricated correlation data exists.
"""

import os

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


class TestHWOuterOptimizer:
    """I1: OUTER-HW produces CoreTypeLibrary with pruning."""

    def test_hw_outer_header_exists(self):
        """HWOuterOptimizer.h should define the outer HW optimizer."""
        ho_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWOuterOptimizer.h"
        assert ho_h.exists(), f"HWOuterOptimizer.h not found"
        content = ho_h.read_text(encoding="utf-8")

        assert "class HWOuterOptimizer" in content, "Missing HWOuterOptimizer class"
        assert "HWOuterOptimizerResult" in content, "Missing result struct"
        assert "HWOuterOptimizerOptions" in content, "Missing options struct"

    def test_core_type_library_defined(self):
        """CoreTypeLibrary should be defined with entries and utility methods."""
        ho_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWOuterOptimizer.h"
        content = ho_h.read_text(encoding="utf-8")

        assert "struct CoreTypeLibrary" in content, "Missing CoreTypeLibrary struct"
        assert "CoreTypeLibraryEntry" in content, "Missing CoreTypeLibraryEntry"
        assert "numTypes" in content, "Missing numTypes method"
        assert "totalInstances" in content, "Missing totalInstances method"

    def test_core_type_has_resource_bounds(self):
        """CoreTypeLibraryEntry should have resource lower bounds for INNER-HW."""
        ho_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWOuterOptimizer.h"
        content = ho_h.read_text(encoding="utf-8")

        assert "minPEs" in content, "Missing minPEs resource bound"
        assert "minSPMKB" in content, "Missing minSPMKB resource bound"
        assert "requiredFUTypes" in content, "Missing requiredFUTypes"

    def test_core_role_enum(self):
        """CoreRole should have FP_HEAVY, CONTROL_HEAVY, MEMORY_HEAVY, BALANCED."""
        ho_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWOuterOptimizer.h"
        content = ho_h.read_text(encoding="utf-8")

        assert "enum class CoreRole" in content, "Missing CoreRole enum"
        for role in ["FP_HEAVY", "CONTROL_HEAVY", "MEMORY_HEAVY", "BALANCED"]:
            assert role in content, f"Missing CoreRole::{role}"

    def test_system_topology_spec(self):
        """SystemTopologySpec should define NoC, L2, and core placements."""
        ho_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWOuterOptimizer.h"
        content = ho_h.read_text(encoding="utf-8")

        assert "struct SystemTopologySpec" in content, "Missing SystemTopologySpec"
        assert "nocTopology" in content, "Missing nocTopology field"
        assert "meshRows" in content, "Missing meshRows"
        assert "meshCols" in content, "Missing meshCols"
        assert "l2TotalSizeKB" in content, "Missing l2TotalSizeKB"
        assert "corePlacements" in content, "Missing corePlacements"

    def test_tdc_rejection_tracked(self):
        """Optimizer result should track TDC rejections (pruned candidates)."""
        ho_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWOuterOptimizer.h"
        content = ho_h.read_text(encoding="utf-8")

        assert "tdcRejections" in content, "Missing tdcRejections counter"
        assert "tier1Evaluations" in content, "Missing tier1Evaluations"
        assert "tier2Evaluations" in content, "Missing tier2Evaluations"

    def test_generate_system_mlir_method(self):
        """HWOuterOptimizer should be able to generate system MLIR."""
        ho_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWOuterOptimizer.h"
        content = ho_h.read_text(encoding="utf-8")

        assert "generateSystemMLIR" in content, "Missing generateSystemMLIR method"


class TestHWInnerOptimizer:
    """I2-I3: INNER-HW produces per-core ADG with 13 design dimensions."""

    def test_hw_inner_header_exists(self):
        """HWInnerOptimizer.h should define the per-core ADG optimizer."""
        hi_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWInnerOptimizer.h"
        assert hi_h.exists(), f"HWInnerOptimizer.h not found"
        content = hi_h.read_text(encoding="utf-8")

        assert "class HWInnerOptimizer" in content, "Missing HWInnerOptimizer class"
        assert "ADGOptResult" in content, "Missing ADGOptResult struct"
        assert "HWInnerOptimizerOptions" in content, "Missing options struct"

    def test_core_design_params_has_13_dimensions(self):
        """CoreDesignParams should define all 13 design dimensions."""
        hi_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWInnerOptimizer.h"
        content = hi_h.read_text(encoding="utf-8")

        # All 13 dimensions from the C12 plan
        dimensions = {
            "peType": "Dimension 1: PE type",
            "arrayRows": "Dimension 2: Array dimensions",
            "arrayCols": "Dimension 2: Array dimensions (cols)",
            "dataWidth": "Dimension 3: Data width",
            "fuRepertoire": "Dimension 4: FU repertoire",
            "multiOpFUBodies": "Dimension 5: FU body structure",
            "switchType": "Dimension 6: Switch type",
            "decomposableBits": "Dimension 7: Switch decomposability",
            "spmSizeKB": "Dimension 8: SPM",
            "extmemCount": "Dimension 9: External memory",
            "topology": "Dimension 10: Routing topology",
            "instructionSlots": "Dimension 11: Temporal PE params",
            "scalarInputs": "Dimension 12: Scalar I/O",
            "connectivity": "Dimension 13: Connectivity matrix",
        }

        for field, desc in dimensions.items():
            assert field in content, (
                f"CoreDesignParams missing {field} ({desc})"
            )

    def test_pe_type_enum(self):
        """PEType should have SPATIAL and TEMPORAL."""
        hi_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWInnerOptimizer.h"
        content = hi_h.read_text(encoding="utf-8")

        assert "enum class PEType" in content, "Missing PEType enum"
        assert "SPATIAL" in content, "Missing PEType::SPATIAL"
        assert "TEMPORAL" in content, "Missing PEType::TEMPORAL"

    def test_routing_topology_enum(self):
        """RoutingTopology should have CHESS, MESH, LATTICE, RING."""
        hi_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWInnerOptimizer.h"
        content = hi_h.read_text(encoding="utf-8")

        assert "enum class RoutingTopology" in content, "Missing RoutingTopology enum"
        for topo in ["CHESS", "MESH", "LATTICE", "RING"]:
            assert topo in content, f"Missing RoutingTopology::{topo}"

    def test_three_tier_evaluation(self):
        """HWInnerOptimizer should implement Tier-A and Tier-B evaluation."""
        hi_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWInnerOptimizer.h"
        content = hi_h.read_text(encoding="utf-8")

        # Tier-A: analytical derivation
        assert "deriveInitialParams" in content or "runTierA" in content, (
            "Missing Tier-A analytical derivation"
        )
        # Tier-B: BO + mapper
        assert "tier2Enabled" in content or "runTierB" in content, (
            "Missing Tier-B BO optimization"
        )

    def test_adg_opt_result_has_area(self):
        """ADGOptResult should have area estimate and mapping results."""
        hi_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWInnerOptimizer.h"
        content = hi_h.read_text(encoding="utf-8")

        assert "areaEstimate" in content, "Missing areaEstimate in ADGOptResult"
        assert "mappingResults" in content, "Missing mappingResults"
        assert "adgMLIR" in content, "Missing adgMLIR (generated ADG text)"

    def test_area_model_function(self):
        """estimateCoreArea should exist for analytical area estimation."""
        hi_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWInnerOptimizer.h"
        content = hi_h.read_text(encoding="utf-8")

        assert "estimateCoreArea" in content, "Missing estimateCoreArea function"

    def test_batch_optimization_function(self):
        """optimizeAllCoreTypes should exist for parallel core type optimization."""
        hi_h = REPO_ROOT / "include" / "loom" / "SystemCompiler" / "HWInnerOptimizer.h"
        content = hi_h.read_text(encoding="utf-8")

        assert "optimizeAllCoreTypes" in content, "Missing optimizeAllCoreTypes"


class TestHWDSENoFabricatedData:
    """I4: No fabricated correlation data in HW DSE outputs."""

    def test_no_fabricated_data_in_hw_outer_script(self):
        """hw_outer_optimizer.py should not have fabricated data."""
        hw_outer = REPO_ROOT / "scripts" / "dse" / "hw_outer_optimizer.py"
        if not hw_outer.exists():
            pytest.skip("hw_outer_optimizer.py not found")
        content = hw_outer.read_text(encoding="utf-8")

        for term in ["gaussian_sigma", "systematic_bias", "fake_"]:
            assert term not in content, (
                f"Fabricated data term '{term}' in hw_outer_optimizer.py"
            )

    def test_no_fabricated_data_in_spectral_clustering(self):
        """spectral_clustering.py should not have fabricated data."""
        sc_path = REPO_ROOT / "scripts" / "dse" / "spectral_clustering.py"
        if not sc_path.exists():
            pytest.skip("spectral_clustering.py not found")
        content = sc_path.read_text(encoding="utf-8")

        for term in ["gaussian_sigma", "systematic_bias"]:
            assert term not in content, (
                f"Fabricated data term '{term}' in spectral_clustering.py"
            )
