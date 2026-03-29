"""Unit tests for the inner DSE per-core optimizer (Plan E3).

Tests cover:
  UT1: Parameter sweep generates valid candidates
  UT2: Tier-A proxy scoring (basic case)
  UT3: Tier-A infeasibility detection
  UT4: Tier-B compilation invocation (mock)
  UT5: Tier-B partial failure handling
  UT6: Tier ranking correlation
  UT7: Domain-specific preset construction
  UT8: Combinatorial preset construction
  UT9: Inner DSE end-to-end smoke test
  UT10: Caching across outer DSE iterations
"""

from __future__ import annotations

import math
import unittest

from scripts.dse.inner_optimizer import (
    ComputeMix,
    CoreDesignParams,
    FreedomMask,
    InnerDSEConfig,
    InnerDSEDriver,
    InnerResult,
    KernelMappingResult,
    PEType,
    TierBResult,
    clear_cache,
    create_combinatorial_preset,
    create_domain_preset,
    evaluate_candidate_tier_b,
    generate_sweep_candidates,
    generate_type_id,
    optimize_single_type,
    score_core_design,
)
from scripts.dse.proxy_model import KernelProfile


def _make_fp_kernel() -> KernelProfile:
    """Create an FP-heavy kernel profile."""
    kp = KernelProfile()
    kp.name = "kernel_fp"
    kp.op_histogram = {"arith.addf": 20, "arith.mulf": 10, "handshake.load": 5}
    kp.loads_per_iter = 5
    kp.dfg_node_count = 35
    return kp


def _make_int_kernel() -> KernelProfile:
    """Create an INT-heavy kernel profile."""
    kp = KernelProfile()
    kp.name = "kernel_int"
    kp.op_histogram = {"arith.addi": 15, "arith.muli": 8, "handshake.load": 3}
    kp.loads_per_iter = 3
    kp.dfg_node_count = 26
    return kp


def _make_test_params() -> CoreDesignParams:
    """Create a test CoreDesignParams covering both FP and INT ops."""
    params = CoreDesignParams()
    params.array_rows = 8
    params.array_cols = 8
    params.data_width = 32
    params.spm_size_kb = 16
    params.spm_ld_ports = 2
    params.spm_st_ports = 2
    params.topology = "chess"
    params.fu_repertoire = [
        "arith.addi", "arith.muli", "arith.addf", "arith.mulf",
        "arith.cmpi", "arith.select",
        "handshake.load", "handshake.store",
    ]
    return params


class TestParameterSweep(unittest.TestCase):
    """UT1: Parameter sweep generates valid candidates."""

    def test_sweep_generates_candidates(self):
        baseline = create_combinatorial_preset(
            ComputeMix.FP_HEAVY, PEType.SPATIAL, True, 8
        )
        mask = FreedomMask.combinatorial(is_temporal=False)

        candidates = generate_sweep_candidates(baseline, mask, 10, seed=42)

        # All candidates should be non-empty
        self.assertTrue(len(candidates) > 0)
        self.assertLessEqual(len(candidates), 10)

        # Fixed dimensions must be preserved
        for cand in candidates:
            self.assertEqual(cand.array_rows, 8)
            self.assertEqual(cand.array_cols, 8)
            self.assertEqual(cand.pe_type, "spatial")
            self.assertEqual(cand.spm_size_kb, 16)

        # Free dimensions should vary: check topology diversity
        topologies = {c.topology for c in candidates}
        self.assertGreaterEqual(
            len(topologies), 2,
            f"Expected at least 2 distinct topologies, got {topologies}"
        )

    def test_sweep_with_no_free_dims(self):
        baseline = _make_test_params()
        mask = FreedomMask()  # all fixed

        candidates = generate_sweep_candidates(baseline, mask, 10, seed=42)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].array_rows, baseline.array_rows)


class TestTierAScoring(unittest.TestCase):
    """UT2: Tier-A proxy scoring (basic case)."""

    def test_feasible_scoring(self):
        params = _make_test_params()
        kernel_fp = _make_fp_kernel()
        kernel_int = _make_int_kernel()

        result = score_core_design(params, [kernel_fp, kernel_int])

        self.assertTrue(result.feasible)
        self.assertIn("kernel_fp", result.per_kernel_ii)
        self.assertIn("kernel_int", result.per_kernel_ii)
        self.assertGreater(result.area_estimate, 0)
        self.assertGreater(result.composite_score, 0)

        # FP kernel: 30 FP ops (addf=20 + mulf=10), FP FUs = 2 per PE * 64 PEs
        # fu_bound for FP = ceil(30 / (2*64)) = 1
        # But the FU category counting is per-type-in-repertoire * PE count
        # arith.addf -> "fp", arith.mulf -> "fp" => 2 FP FU types * 64 PEs
        fp_ii = result.per_kernel_ii["kernel_fp"]
        self.assertGreaterEqual(fp_ii, 1.0)

        int_ii = result.per_kernel_ii["kernel_int"]
        self.assertGreaterEqual(int_ii, 1.0)


class TestTierAInfeasibility(unittest.TestCase):
    """UT3: Tier-A infeasibility detection."""

    def test_missing_fp_ops(self):
        # Remove FP ops from repertoire
        params = _make_test_params()
        params.fu_repertoire = [
            "arith.addi", "arith.muli", "arith.cmpi", "arith.select",
            "handshake.load", "handshake.store",
        ]

        kernel_fp = _make_fp_kernel()

        result = score_core_design(params, [kernel_fp])

        self.assertFalse(result.feasible)
        self.assertEqual(result.composite_score, 0)


class TestTierBCompilation(unittest.TestCase):
    """UT4: Tier-B compilation invocation (mock)."""

    def test_all_kernels_map(self):
        params = _make_test_params()
        kernel_fp = _make_fp_kernel()
        kernel_int = _make_int_kernel()

        def mock_compile(p, k):
            return KernelMappingResult(
                kernel_name=k.name, success=True, achieved_ii=3,
            )

        result = evaluate_candidate_tier_b(
            params, [kernel_fp, kernel_int],
            compile_fn=mock_compile,
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.mapping_success_rate, 1.0)
        self.assertAlmostEqual(result.mean_achieved_ii, 3.0)
        self.assertEqual(len(result.mapping_results), 2)
        for mr in result.mapping_results:
            self.assertTrue(mr.success)


class TestTierBPartialFailure(unittest.TestCase):
    """UT5: Tier-B partial failure handling."""

    def test_one_kernel_fails(self):
        params = _make_test_params()
        kernel_fp = _make_fp_kernel()
        kernel_int = _make_int_kernel()

        call_count = [0]

        def mock_compile(p, k):
            call_count[0] += 1
            if k.name == "kernel_int":
                return KernelMappingResult(
                    kernel_name=k.name, success=False, achieved_ii=0,
                )
            return KernelMappingResult(
                kernel_name=k.name, success=True, achieved_ii=3,
            )

        result = evaluate_candidate_tier_b(
            params, [kernel_fp, kernel_int],
            compile_fn=mock_compile,
        )

        self.assertFalse(result.success)
        self.assertAlmostEqual(result.mapping_success_rate, 0.5)
        # Geometric mean from 1 successful kernel
        self.assertAlmostEqual(result.mean_achieved_ii, 3.0)
        self.assertIn("kernel_int", result.diagnostics)


class TestTierRankingCorrelation(unittest.TestCase):
    """UT6: Tier ranking correlation."""

    def test_rank_correlation(self):
        """Tier-A and Tier-B rankings should be correlated."""
        base = _make_test_params()
        kernels = [_make_fp_kernel(), _make_int_kernel()]

        # Generate 5 variants with different topologies and repertoires
        variants = []

        # Variant 0: chess, large repertoire
        v0 = CoreDesignParams.from_dict(base.to_dict())
        v0.topology = "chess"
        variants.append(v0)

        # Variant 1: mesh
        v1 = CoreDesignParams.from_dict(base.to_dict())
        v1.topology = "mesh"
        variants.append(v1)

        # Variant 2: ring (worse routing)
        v2 = CoreDesignParams.from_dict(base.to_dict())
        v2.topology = "ring"
        variants.append(v2)

        # Variant 3: smaller array (worse for large kernels)
        v3 = CoreDesignParams.from_dict(base.to_dict())
        v3.array_rows = 4
        v3.array_cols = 4
        variants.append(v3)

        # Variant 4: larger array (better but more area)
        v4 = CoreDesignParams.from_dict(base.to_dict())
        v4.array_rows = 10
        v4.array_cols = 10
        variants.append(v4)

        # Compute Tier-A scores
        tier_a_scores = []
        for v in variants:
            ta = score_core_design(v, kernels)
            tier_a_scores.append(ta.composite_score)

        # Compute Tier-B scores using mock compile (II = total_ops / PE_count)
        tier_b_scores = []
        for v in variants:
            def mock_compile(p, k, v=v):
                pe_count = p.total_pes()
                total_ops = sum(k.op_histogram.values())
                ii = max(1, math.ceil(total_ops / max(1, pe_count)))
                return KernelMappingResult(
                    kernel_name=k.name, success=True, achieved_ii=ii,
                )

            tb = evaluate_candidate_tier_b(v, kernels, compile_fn=mock_compile)
            # Score: success_rate / (mean_ii * area)
            if tb.mean_achieved_ii > 0:
                tb_score = tb.mapping_success_rate / (
                    tb.mean_achieved_ii * max(1, tb.area)
                )
            else:
                tb_score = 0
            tier_b_scores.append(tb_score)

        # Compute Spearman rank correlation
        def rank(values):
            sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
            ranks = [0.0] * len(values)
            for r, idx in enumerate(sorted_idx):
                ranks[idx] = float(r + 1)
            return ranks

        rank_a = rank(tier_a_scores)
        rank_b = rank(tier_b_scores)

        n = len(rank_a)
        d_sq_sum = sum((rank_a[i] - rank_b[i]) ** 2 for i in range(n))
        rho = 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))

        # We expect at least moderate correlation
        self.assertGreaterEqual(
            rho, 0.3,
            f"Spearman rho={rho:.3f}, Tier-A ranks={rank_a}, "
            f"Tier-B ranks={rank_b}",
        )


class TestDomainPresetConstruction(unittest.TestCase):
    """UT7: Domain-specific preset construction."""

    def test_d1_llm_preset(self):
        params = create_domain_preset(1)

        self.assertEqual(params.array_rows, 12)
        self.assertEqual(params.array_cols, 12)
        self.assertEqual(params.pe_type, "spatial")
        self.assertEqual(params.spm_size_kb, 64)
        self.assertEqual(params.topology, "mesh")
        self.assertEqual(params.data_width, 32)

        # Must have FP ops
        has_addf = any("addf" in op for op in params.fu_repertoire)
        has_mulf = any("mulf" in op for op in params.fu_repertoire)
        self.assertTrue(has_addf, "D1 preset must include arith.addf")
        self.assertTrue(has_mulf, "D1 preset must include arith.mulf")

    def test_all_domain_presets_valid(self):
        for idx in range(1, 7):
            params = create_domain_preset(idx)
            self.assertGreater(params.array_rows, 0)
            self.assertGreater(params.array_cols, 0)
            self.assertGreater(len(params.fu_repertoire), 0)


class TestCombinatorialPresetConstruction(unittest.TestCase):
    """UT8: Combinatorial preset construction."""

    def test_int_heavy_temporal_with_spm(self):
        params = create_combinatorial_preset(
            ComputeMix.INT_HEAVY, PEType.TEMPORAL, True, 8
        )

        self.assertEqual(params.array_rows, 8)
        self.assertEqual(params.array_cols, 8)
        self.assertEqual(params.pe_type, "temporal")
        self.assertEqual(params.switch_type, "temporal")
        self.assertEqual(params.spm_size_kb, 16)
        self.assertEqual(params.spm_ld_ports, 2)
        self.assertEqual(params.spm_st_ports, 2)
        self.assertEqual(params.instruction_slots, 8)
        self.assertEqual(params.num_registers, 8)

        # INT-heavy should have ALU and shift ops
        has_addi = any("addi" in op for op in params.fu_repertoire)
        has_muli = any("muli" in op for op in params.fu_repertoire)
        self.assertTrue(has_addi)
        self.assertTrue(has_muli)

    def test_type_id_generation(self):
        tid = generate_type_id(
            ComputeMix.INT_HEAVY, PEType.TEMPORAL, True, 8
        )
        self.assertEqual(tid, "CITY8")

        tid2 = generate_type_id(
            ComputeMix.FP_HEAVY, PEType.SPATIAL, True, 8
        )
        self.assertEqual(tid2, "CFSY8")

    def test_no_spm_variant(self):
        params = create_combinatorial_preset(
            ComputeMix.MIXED, PEType.SPATIAL, False, 4
        )
        self.assertEqual(params.spm_size_kb, 0)
        self.assertEqual(params.spm_ld_ports, 0)


class TestInnerDSEEndToEnd(unittest.TestCase):
    """UT9: Inner DSE end-to-end smoke test."""

    def test_smoke(self):
        baseline = create_combinatorial_preset(
            ComputeMix.MIXED, PEType.SPATIAL, True, 8
        )
        mask = FreedomMask.combinatorial(is_temporal=False)

        kernels = [_make_fp_kernel(), _make_int_kernel()]
        # Add a third kernel
        k3 = KernelProfile()
        k3.name = "kernel_ctrl"
        k3.op_histogram = {"arith.addi": 5, "arith.cmpi": 3, "arith.select": 2}
        k3.dfg_node_count = 10
        kernels.append(k3)

        # Mock compiler: all succeed with II=4
        def mock_compile(p, k):
            return KernelMappingResult(
                kernel_name=k.name, success=True, achieved_ii=4,
            )

        config = InnerDSEConfig(
            max_inner_iter=5,
            top_k=3,
            tier_b_enabled=True,
            seed=42,
        )

        driver = InnerDSEDriver(
            type_id="CMSY8",
            baseline=baseline,
            freedom_mask=mask,
            assigned_kernels=kernels,
            config=config,
            compile_fn=mock_compile,
        )

        result = driver.optimize()

        self.assertTrue(result.success)
        self.assertIsNotNone(result.best_params)
        self.assertGreater(result.area_estimate, 0)
        self.assertGreater(result.tier_a_score, 0)
        # At least 2 distinct candidates evaluated at Tier-A
        self.assertGreaterEqual(result.tier_a_evaluations, 2)
        # At least 1 candidate at Tier-B
        self.assertGreaterEqual(result.tier_b_evaluations, 1)

    def test_json_round_trip(self):
        """InnerResult JSON serialization and deserialization."""
        result = InnerResult(
            type_id="test_type",
            success=True,
            best_params=_make_test_params(),
            tier_a_score=0.5,
            tier_b_score=0.8,
            area_estimate=1000.0,
            tier_a_evaluations=10,
            tier_b_evaluations=3,
            wall_time_sec=1.5,
            mapping_results=[
                KernelMappingResult(
                    kernel_name="k1", success=True,
                    achieved_ii=3, mapping_time_sec=0.1,
                ),
            ],
        )

        json_str = result.to_json()
        restored = InnerResult.from_json(json_str)

        self.assertEqual(restored.type_id, "test_type")
        self.assertTrue(restored.success)
        self.assertAlmostEqual(restored.tier_a_score, 0.5)
        self.assertAlmostEqual(restored.area_estimate, 1000.0)
        self.assertEqual(len(restored.mapping_results), 1)
        self.assertEqual(restored.mapping_results[0].kernel_name, "k1")


class TestCaching(unittest.TestCase):
    """UT10: Caching across outer DSE iterations."""

    def setUp(self):
        clear_cache()

    def test_cache_hit(self):
        baseline = create_combinatorial_preset(
            ComputeMix.FP_HEAVY, PEType.SPATIAL, True, 8
        )
        mask = FreedomMask.combinatorial()
        kernels = [_make_fp_kernel(), _make_int_kernel()]

        call_count = [0]

        def mock_compile(p, k):
            call_count[0] += 1
            return KernelMappingResult(
                kernel_name=k.name, success=True, achieved_ii=3,
            )

        config = InnerDSEConfig(max_inner_iter=5, top_k=3)

        # First call
        result1 = optimize_single_type(
            "CFSY8", baseline, mask, kernels,
            config=config, compile_fn=mock_compile,
        )
        calls_after_first = call_count[0]

        # Second call with same inputs - should use cache
        result2 = optimize_single_type(
            "CFSY8", baseline, mask, kernels,
            config=config, compile_fn=mock_compile,
        )
        calls_after_second = call_count[0]

        # Second call should not invoke compile_fn again
        self.assertEqual(calls_after_first, calls_after_second)

        # Results should be identical
        self.assertEqual(result1.type_id, result2.type_id)
        self.assertEqual(result1.success, result2.success)

    def tearDown(self):
        clear_cache()


if __name__ == "__main__":
    unittest.main()
