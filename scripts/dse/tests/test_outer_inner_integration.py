"""Integration tests for outer-to-inner DSE wiring (Group E2-E3).

Tests cover:
  IT1: _library_to_inner_params correctly converts library params
  IT2: HWOuterOptimizer._refine_with_inner_dse runs inner DSE on selected types
  IT3: Inner DSE results appear in OuterHWResult
  IT4: write_dse_results includes inner_dse section
  IT5: to_system_mlir docstring warns about pseudo-MLIR
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from scripts.dse.core_type_library import (
    CoreDesignParams as LibraryCoreDesignParams,
    get_core_design_params,
)
from scripts.dse.hw_outer_optimizer import (
    HWOuterOptimizer,
    OuterHWResult,
    _library_to_inner_params,
    write_dse_results,
)
from scripts.dse.inner_optimizer import (
    CoreDesignParams as InnerCoreDesignParams,
    InnerDSEConfig,
    InnerResult,
    KernelMappingResult,
)
from scripts.dse.proxy_model import (
    ContractEdge,
    KernelProfile,
    WorkloadProfile,
)
from scripts.dse.system_graph_generator import to_system_mlir, SystemTopologySpec


def _make_workload() -> WorkloadProfile:
    """Create a minimal workload for testing."""
    k1 = KernelProfile()
    k1.name = "kernel_fp"
    k1.op_histogram = {"arith.addf": 10, "arith.mulf": 5, "handshake.load": 3}
    k1.loads_per_iter = 3
    k1.dfg_node_count = 18

    k2 = KernelProfile()
    k2.name = "kernel_int"
    k2.op_histogram = {"arith.addi": 8, "arith.muli": 4, "handshake.load": 2}
    k2.loads_per_iter = 2
    k2.dfg_node_count = 14

    contract = ContractEdge(
        producer="kernel_fp",
        consumer="kernel_int",
        production_rate=1.0,
        element_size_bytes=4,
    )

    return WorkloadProfile(
        kernels=[k1, k2],
        contracts=[contract],
        critical_path=["kernel_fp", "kernel_int"],
    )


class TestLibraryToInnerParams(unittest.TestCase):
    """IT1: _library_to_inner_params correctly converts library params."""

    def test_domain_specific_d1(self):
        lib_params = get_core_design_params("D1")
        inner = _library_to_inner_params(lib_params, "D1")

        self.assertIsInstance(inner, InnerCoreDesignParams)
        self.assertEqual(inner.array_rows, lib_params.array_rows)
        self.assertEqual(inner.array_cols, lib_params.array_cols)
        self.assertEqual(inner.data_width, lib_params.data_width)

        # D1 has FP, so inner should have FP ops
        has_addf = "arith.addf" in inner.fu_repertoire
        self.assertTrue(has_addf, "D1 conversion must include FP ops")

        # Must have load/store
        has_load = "handshake.load" in inner.fu_repertoire
        self.assertTrue(has_load, "Conversion must include memory ops")

    def test_combinatorial_khg(self):
        lib_params = get_core_design_params("CISY8")
        inner = _library_to_inner_params(lib_params, "CISY8")

        self.assertEqual(inner.pe_type, "spatial")
        self.assertEqual(inner.array_rows, 8)
        self.assertEqual(inner.array_cols, 8)

        # INT_HEAVY should have ALU ops
        has_addi = "arith.addi" in inner.fu_repertoire
        self.assertTrue(has_addi)

    def test_temporal_type(self):
        lib_params = get_core_design_params("CITY8")
        inner = _library_to_inner_params(lib_params, "CITY8")

        self.assertEqual(inner.pe_type, "temporal")
        self.assertTrue(inner.instruction_slots > 0 or inner.num_registers > 0)

    def test_no_duplicate_ops(self):
        lib_params = get_core_design_params("D1")
        inner = _library_to_inner_params(lib_params, "D1")

        # FU repertoire should have no duplicates
        self.assertEqual(
            len(inner.fu_repertoire),
            len(set(inner.fu_repertoire)),
            "FU repertoire has duplicates",
        )


class TestInnerDSERefinement(unittest.TestCase):
    """IT2: HWOuterOptimizer._refine_with_inner_dse runs inner DSE."""

    def test_refinement_runs_for_selected_types(self):
        workload = _make_workload()

        def mock_compile(p, k):
            return KernelMappingResult(
                kernel_name=k.name, success=True, achieved_ii=3,
            )

        optimizer = HWOuterOptimizer(
            workload=workload,
            inner_dse_config=InnerDSEConfig(
                max_inner_iter=3,
                top_k=2,
                tier_b_enabled=True,
            ),
            compile_fn=mock_compile,
        )

        # Manually set best state as if BO loop ran
        optimizer._best_selected_ids = ["D1", "CISY8"]
        optimizer._best_assignment = {
            "kernel_fp": 0,
            "kernel_int": 1,
        }

        results = optimizer._refine_with_inner_dse()

        self.assertIsInstance(results, dict)
        # Should have results for types with assigned kernels
        self.assertIn("D1", results)
        self.assertIn("CISY8", results)

        for type_id, result in results.items():
            self.assertIsInstance(result, InnerResult)
            self.assertTrue(
                result.success,
                f"Inner DSE for {type_id} should succeed with mock compile",
            )
            self.assertGreater(result.tier_a_evaluations, 0)

    def test_refinement_empty_when_no_selection(self):
        workload = _make_workload()
        optimizer = HWOuterOptimizer(workload=workload)

        # No best selected IDs
        optimizer._best_selected_ids = []
        optimizer._best_assignment = {}

        results = optimizer._refine_with_inner_dse()
        self.assertEqual(results, {})


class TestOuterHWResultIntegration(unittest.TestCase):
    """IT3: Inner DSE results appear in OuterHWResult."""

    def test_inner_dse_results_in_summary(self):
        result = OuterHWResult()
        result.selected_type_ids = ["D1"]
        result.inner_dse_results = {
            "D1": InnerResult(
                type_id="D1",
                success=True,
                tier_a_score=0.5,
                tier_b_score=0.8,
            ),
        }

        summary = result.summary()
        self.assertIn("Inner DSE", summary)
        self.assertIn("1/1", summary)

    def test_inner_dse_results_empty_summary(self):
        result = OuterHWResult()
        summary = result.summary()
        # Should not mention inner DSE if no results
        self.assertNotIn("Inner DSE", summary)


class TestWriteDSEResults(unittest.TestCase):
    """IT4: write_dse_results includes inner_dse section."""

    def test_output_contains_inner_dse(self):
        result = OuterHWResult()
        result.selected_type_ids = ["CISY8"]
        result.inner_dse_results = {
            "CISY8": InnerResult(
                type_id="CISY8",
                success=True,
                tier_a_score=0.42,
                area_estimate=1234.0,
            ),
        }

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            path = f.name

        try:
            write_dse_results(result, path)

            with open(path, "r") as f:
                data = json.load(f)

            self.assertIn("inner_dse", data)
            self.assertIn("CISY8", data["inner_dse"])
            self.assertEqual(data["inner_dse"]["CISY8"]["type_id"], "CISY8")
            self.assertTrue(data["inner_dse"]["CISY8"]["success"])
            self.assertAlmostEqual(
                data["inner_dse"]["CISY8"]["tier_a_score"], 0.42
            )
        finally:
            os.unlink(path)


class TestSystemMLIRDocstring(unittest.TestCase):
    """IT5: to_system_mlir docstring warns about pseudo-MLIR."""

    def test_docstring_warns_pseudo_mlir(self):
        docstring = to_system_mlir.__doc__
        self.assertIn("pseudo-MLIR", docstring)
        self.assertIn("SystemADGBuilder", docstring)
        self.assertIn("WARNING", docstring)

    def test_output_is_structural_description(self):
        spec = SystemTopologySpec()
        spec.mesh_rows = 2
        spec.mesh_cols = 2
        output = to_system_mlir(spec)

        # Should contain fabric.module wrapper
        self.assertIn("fabric.module", output)
        # Should contain router descriptions
        self.assertIn("fabric.router", output)


if __name__ == "__main__":
    unittest.main()
