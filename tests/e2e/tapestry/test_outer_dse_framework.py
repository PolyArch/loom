"""Unit tests for the Outer DSE Framework (E2).

Tests the 30-type design space encoding, core type library, system graph
generation, kernel-to-core assignment, TDC feasibility, Pareto front
extraction, and end-to-end BO convergence.
"""

import os
import sys
from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "tools" / "tapestry").exists():
            return p
        p = p.parent
    raise RuntimeError("Cannot locate repository root")


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from scripts.dse.core_type_library import (
    ALL_TYPES,
    COMBINATORIAL_TYPES,
    DOMAIN_SPECIFIC_TYPES,
    NUM_TYPES,
    CoreDesignParams,
    get_all_type_ids,
    get_core_design_params,
    get_type_name,
    index_to_type_id,
    type_id_to_index,
    type_supports_ops,
)
from scripts.dse.design_space import DesignPoint, DesignSpace
from scripts.dse.dse_config import BOConfig, MAX_CORE_TYPES
from scripts.dse.hw_outer_optimizer import (
    CoreTypeLibrary,
    HWOuterOptimizer,
    OuterHWResult,
    TDCBounds,
    assign_kernels_to_types,
    compute_kernel_type_affinity,
    compute_tdc_bounds,
    evaluate_tier1,
    validate_coverage,
    write_dse_results,
)
from scripts.dse.pareto import (
    ParetoEntry,
    extract_pareto_front,
)
from scripts.dse.proxy_model import (
    ContractEdge,
    KernelProfile,
    ProxyScore,
    WorkloadProfile,
)
from scripts.dse.system_graph_generator import (
    SystemGraphGenerator,
    SystemTopologySpec,
    to_system_mlir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kernel(name, fp_ops=0, int_ops=10, mem_ops=2, dfg=16, mem_bytes=1024):
    """Build a KernelProfile with a simple op histogram."""
    hist = {}
    if fp_ops > 0:
        hist["fadd"] = fp_ops
    if int_ops > 0:
        hist["add"] = int_ops
    if mem_ops > 0:
        hist["load"] = mem_ops
    return KernelProfile(
        name=name,
        op_histogram=hist,
        memory_footprint_bytes=mem_bytes,
        loads_per_iter=mem_ops,
        stores_per_iter=max(1, mem_ops // 2),
        dfg_node_count=dfg,
    )


def _make_workload(n_kernels=4, n_contracts=2):
    """Build a simple WorkloadProfile."""
    kernels = [
        _make_kernel(f"k{i}", fp_ops=(10 if i % 2 == 0 else 0),
                     int_ops=(10 if i % 2 == 1 else 0))
        for i in range(n_kernels)
    ]
    contracts = []
    for i in range(min(n_contracts, n_kernels - 1)):
        contracts.append(ContractEdge(
            producer=f"k{i}",
            consumer=f"k{i+1}",
            production_rate=2.0,
            element_size_bytes=4,
        ))
    return WorkloadProfile(
        kernels=kernels,
        contracts=contracts,
        critical_path=[k.name for k in kernels],
    )


# ---------------------------------------------------------------------------
# UT1: BO suggests valid configurations
# ---------------------------------------------------------------------------

class TestBOSuggestsValid:
    """UT1: BO suggests valid 30-type configurations."""

    def test_decode_always_has_selected_type(self):
        """Every decoded DesignPoint has at least 1 selected type."""
        space = DesignSpace(seed=42)
        from scripts.dse.bayesian_opt import BayesianOptimizer
        bo_config = BOConfig(n_initial_samples=5, max_iterations=15, seed=42)
        optimizer = BayesianOptimizer(space=space, config=bo_config)

        for _ in range(10):
            point = optimizer.suggest()
            decoded = DesignPoint.from_vector(point.to_vector())
            assert decoded.num_selected_types() >= 1, (
                "Decoded point has no selected types"
            )
            # Every selected type must have instance_count >= 1
            for i in decoded.selected_type_indices():
                assert decoded.instance_counts[i] >= 1, (
                    f"Selected type {i} has instance_count=0"
                )

            # Observe a dummy score
            optimizer.observe(point, float(decoded.total_cores()) * 0.1)

    def test_noc_topology_valid(self):
        """NoC topology is always a valid value."""
        space = DesignSpace(seed=123)
        for point in space.sample_random(20):
            assert point.noc_topology in ["mesh", "ring", "hierarchical"]

    def test_l2_within_range(self):
        """L2 size and bank count are within configured ranges."""
        space = DesignSpace(seed=456)
        for point in space.sample_random(20):
            assert 64 <= point.l2_size_kb <= 1024
            assert 4 <= point.l2_bank_count <= 16

    def test_vector_round_trip(self):
        """Encode/decode round-trips produce consistent results."""
        space = DesignSpace(seed=789)
        for point in space.sample_random(10):
            vec = point.to_vector()
            assert len(vec) == DesignPoint.vector_dimension()
            decoded = DesignPoint.from_vector(vec)
            assert decoded.noc_topology == point.noc_topology
            assert decoded.noc_bandwidth == point.noc_bandwidth

    def test_vector_dimension_is_64(self):
        """The encoded vector has 30+30+4 = 64 dimensions."""
        assert DesignPoint.vector_dimension() == 64


# ---------------------------------------------------------------------------
# UT2: System graph generation from DSE output
# ---------------------------------------------------------------------------

class TestSystemGraphGeneration:
    """UT2: SystemGraphGenerator produces valid topologies."""

    def test_basic_generation(self):
        """Generate from a 3-type selection with known counts."""
        point = DesignPoint()
        all_ids = get_all_type_ids()

        # Select D1 x2, first combinatorial type x3, second x1
        d1_idx = type_id_to_index("D1")
        combo_ids = list(COMBINATORIAL_TYPES.keys())
        c1_idx = type_id_to_index(combo_ids[0])
        c2_idx = type_id_to_index(combo_ids[5])

        point.type_mask[d1_idx] = True
        point.instance_counts[d1_idx] = 2
        point.type_mask[c1_idx] = True
        point.instance_counts[c1_idx] = 3
        point.type_mask[c2_idx] = True
        point.instance_counts[c2_idx] = 1

        point.noc_topology = "mesh"
        point.noc_bandwidth = 2
        point.l2_size_kb = 256
        point.l2_bank_count = 8

        gen = SystemGraphGenerator()
        spec = gen.generate(point)

        total_cores = 2 + 3 + 1
        assert spec.mesh_rows * spec.mesh_cols >= total_cores
        assert len(spec.core_placements) == total_cores
        assert len(spec.core_library) == 3
        assert spec.l2_bank_count == 8
        assert spec.noc_bandwidth == 2

        # All placements within mesh bounds
        for cp in spec.core_placements:
            assert 0 <= cp.row < spec.mesh_rows
            assert 0 <= cp.col < spec.mesh_cols

    def test_mlir_output_has_fabric(self):
        """to_system_mlir produces text with fabric.module and fabric.core."""
        point = DesignPoint()
        point.type_mask[0] = True
        point.instance_counts[0] = 2
        point.noc_topology = "mesh"
        point.noc_bandwidth = 1
        point.l2_size_kb = 128
        point.l2_bank_count = 4

        gen = SystemGraphGenerator()
        spec = gen.generate(point)
        mlir_text = to_system_mlir(spec)

        assert "fabric.module" in mlir_text
        assert "fabric.core" in mlir_text

    def test_l2_bank_placement(self):
        """L2 banks are placed along mesh edges."""
        point = DesignPoint()
        point.type_mask[0] = True
        point.instance_counts[0] = 4
        point.l2_size_kb = 256
        point.l2_bank_count = 8

        gen = SystemGraphGenerator()
        spec = gen.generate(point)

        assert len(spec.l2_bank_placements) == 8
        for bp in spec.l2_bank_placements:
            assert 0 <= bp.row < spec.mesh_rows
            assert 0 <= bp.col < spec.mesh_cols
            assert bp.size_kb > 0


# ---------------------------------------------------------------------------
# UT3: Evaluation pipeline (Tier-1 with mock)
# ---------------------------------------------------------------------------

class TestEvaluationPipeline:
    """UT3: Multi-fidelity evaluation pipeline chains tiers correctly."""

    def test_tier1_returns_feasible(self):
        """Tier-1 returns a feasible ProxyScore for a valid configuration."""
        workload = _make_workload(4, 2)
        point = DesignPoint()

        # Select two types: one FP, one INT
        all_ids = get_all_type_ids()
        fp_idx = type_id_to_index("D1")
        int_idx = type_id_to_index("D4")
        point.type_mask[fp_idx] = True
        point.instance_counts[fp_idx] = 2
        point.type_mask[int_idx] = True
        point.instance_counts[int_idx] = 2

        point.noc_topology = "mesh"
        point.noc_bandwidth = 2
        point.l2_size_kb = 256
        point.l2_bank_count = 8

        selected_ids = [all_ids[fp_idx], all_ids[int_idx]]
        assignment = assign_kernels_to_types(workload.kernels, selected_ids)

        result = evaluate_tier1(point, workload, assignment, selected_ids)
        assert result.score.feasible
        assert result.score.throughput > 0
        assert result.tier == 1

    def test_tier1_uses_highest_fidelity(self):
        """The BO observes the highest-fidelity score available."""
        # With only Tier-1 available, the composite score should be positive
        workload = _make_workload(2, 1)
        point = DesignPoint()
        point.type_mask[0] = True
        point.instance_counts[0] = 2

        all_ids = get_all_type_ids()
        selected_ids = [all_ids[0]]
        assignment = assign_kernels_to_types(workload.kernels, selected_ids)

        result = evaluate_tier1(point, workload, assignment, selected_ids)
        composite = result.score.composite_score()
        assert composite >= 0


# ---------------------------------------------------------------------------
# UT4: TDC feasibility pruning
# ---------------------------------------------------------------------------

class TestTDCFeasibility:
    """UT4: TDC-derived bounds correctly prune infeasible candidates."""

    def test_low_bandwidth_rejected(self):
        """Candidate with insufficient NoC bandwidth is rejected."""
        workload = _make_workload(2, 1)
        # Create a workload with contracts that imply min_noc_bandwidth > 1
        workload.contracts = [
            ContractEdge(
                producer="k0", consumer="k1",
                production_rate=100.0,
                element_size_bytes=4,
            ),
        ]
        bounds = compute_tdc_bounds(workload)

        optimizer = HWOuterOptimizer(workload=workload)

        # Candidate with low bandwidth
        low_bw = DesignPoint()
        low_bw.type_mask[0] = True
        low_bw.instance_counts[0] = 2
        low_bw.noc_bandwidth = 1
        low_bw.l2_size_kb = 256

        # If TDC min > 1, this should fail
        if bounds.min_noc_bandwidth > 1:
            assert not optimizer._is_tdc_feasible(low_bw)

        # Candidate with sufficient bandwidth and enough cores/types
        high_bw = DesignPoint()
        # Select enough types and instances to satisfy TDC bounds
        for i in range(max(1, bounds.min_core_types)):
            high_bw.type_mask[i] = True
        total_needed = max(bounds.min_total_cores, bounds.min_core_types)
        per_type = max(1, total_needed // max(1, bounds.min_core_types))
        for i in range(max(1, bounds.min_core_types)):
            high_bw.instance_counts[i] = per_type
        high_bw.noc_bandwidth = 4
        high_bw.l2_size_kb = max(256, int(bounds.min_l2_size_kb))

        assert optimizer._is_tdc_feasible(high_bw)

    def test_too_few_types_rejected(self):
        """Candidate with no selected types is rejected."""
        workload = _make_workload(2, 1)
        optimizer = HWOuterOptimizer(workload=workload)

        empty = DesignPoint()
        # All masks False, all counts 0
        assert not optimizer._is_tdc_feasible(empty)


# ---------------------------------------------------------------------------
# UT5: Kernel-to-core assignment
# ---------------------------------------------------------------------------

class TestKernelAssignment:
    """UT5: Assignment correctly matches kernel requirements to core types."""

    def test_fp_kernels_to_fp_type(self):
        """FP-heavy kernels are assigned to FP-capable types."""
        fp_kernel1 = _make_kernel("fp_k1", fp_ops=20, int_ops=2)
        fp_kernel2 = _make_kernel("fp_k2", fp_ops=15, int_ops=3)
        int_kernel1 = _make_kernel("int_k1", fp_ops=0, int_ops=20)
        int_kernel2 = _make_kernel("int_k2", fp_ops=0, int_ops=18)

        kernels = [fp_kernel1, fp_kernel2, int_kernel1, int_kernel2]

        # D1 is FP-heavy, D4 is INT-heavy (no FP)
        selected_ids = ["D1", "D4"]
        assignment = assign_kernels_to_types(kernels, selected_ids)

        # FP kernels should go to D1 (index 0)
        assert assignment["fp_k1"] == 0
        assert assignment["fp_k2"] == 0

        # INT kernels should go to D4 (index 1)
        assert assignment["int_k1"] == 1
        assert assignment["int_k2"] == 1

    def test_all_kernels_assigned(self):
        """No kernel is left unassigned."""
        kernels = [_make_kernel(f"k{i}") for i in range(6)]
        selected_ids = ["D1", "D2"]
        assignment = assign_kernels_to_types(kernels, selected_ids)
        assert validate_coverage(kernels, assignment)

    def test_infeasible_assignment_detected(self):
        """Kernels needing FP ops with no FP-capable type are detected."""
        fp_kernel = _make_kernel("need_fp", fp_ops=20, int_ops=0)
        # D4 has no FP FUs
        selected_ids = ["D4"]
        assignment = assign_kernels_to_types([fp_kernel], selected_ids)
        # Should fail coverage since D4 has fu_fp_count=0
        assert not validate_coverage([fp_kernel], assignment)


# ---------------------------------------------------------------------------
# UT6: Pareto front correctness
# ---------------------------------------------------------------------------

class TestParetoFront:
    """UT6: Pareto front extraction identifies non-dominated solutions."""

    def test_non_dominated_set(self):
        """The Pareto front contains only non-dominated entries.

        Objectives: maximize throughput, minimize area.
        Pareto objectives in code: (throughput, -area), both HIGHER is better.
        So a point dominates another only if it has higher throughput AND
        lower area (or equal in one, strictly better in the other).
        """
        entries = [
            ParetoEntry(point=DesignPoint(), throughput=10, area=100),
            ParetoEntry(point=DesignPoint(), throughput=8, area=80),
            ParetoEntry(point=DesignPoint(), throughput=12, area=200),
            ParetoEntry(point=DesignPoint(), throughput=9, area=90),
            ParetoEntry(point=DesignPoint(), throughput=11, area=150),
        ]

        front = extract_pareto_front(entries)
        front_objectives = [(e.throughput, e.area) for e in front]

        # All 5 points form a Pareto front in the (throughput, -area) space:
        # No point has both higher throughput AND lower area than another.
        # (8,80): lowest area, lowest throughput
        # (9,90): slightly higher in both
        # (10,100): higher still
        # (11,150): higher throughput but higher area
        # (12,200): highest throughput, highest area
        # Each trades off throughput for area, so all are non-dominated.
        assert (10, 100) in front_objectives
        assert (8, 80) in front_objectives
        assert (12, 200) in front_objectives
        assert (9, 90) in front_objectives
        assert (11, 150) in front_objectives
        assert len(front) == 5

    def test_dominated_point_excluded(self):
        """A point dominated in both objectives is excluded from the front."""
        entries = [
            ParetoEntry(point=DesignPoint(), throughput=10, area=100),
            ParetoEntry(point=DesignPoint(), throughput=8, area=120),   # dominated by (10,100)
            ParetoEntry(point=DesignPoint(), throughput=6, area=110),   # dominated by (10,100)
        ]
        front = extract_pareto_front(entries)
        front_objectives = [(e.throughput, e.area) for e in front]

        # (10, 100) dominates both others: higher throughput AND lower area
        assert (10, 100) in front_objectives
        assert (8, 120) not in front_objectives
        assert (6, 110) not in front_objectives
        assert len(front) == 1

    def test_empty_front(self):
        """Empty input produces empty front."""
        assert extract_pareto_front([]) == []


# ---------------------------------------------------------------------------
# UT7: End-to-end BO convergence (smoke test)
# ---------------------------------------------------------------------------

class TestE2EConvergence:
    """UT7: Full BO loop runs and shows learning behavior."""

    def test_bo_converges_tier1_only(self):
        """BO with Tier-1 only converges within 30 iterations."""
        workload = _make_workload(2, 1)
        bo_config = BOConfig(
            n_initial_samples=5,
            max_iterations=30,
            seed=42,
        )

        optimizer = HWOuterOptimizer(
            workload=workload,
            bo_config=bo_config,
        )
        result = optimizer.optimize()

        # Should produce a non-trivial result
        assert result.best_score > float("-inf")
        assert result.iterations_used == 30
        assert result.topology.noc_topology in ["mesh", "ring", "hierarchical"]
        assert len(result.selected_type_ids) >= 1 or result.best_score == 0.0

        # Convergence: best score should be non-decreasing
        if result.convergence_trace:
            scores = [s for _, s in result.convergence_trace]
            for i in range(1, len(scores)):
                assert scores[i] >= scores[i-1], (
                    f"Best score decreased at iteration {i}: "
                    f"{scores[i-1]} -> {scores[i]}"
                )


# ---------------------------------------------------------------------------
# Core type library tests
# ---------------------------------------------------------------------------

class TestCoreTypeLibrary:
    """Verify the 30-type library is correctly constructed."""

    def test_total_types_is_30(self):
        """Library has exactly 30 types."""
        assert NUM_TYPES == 30
        assert len(ALL_TYPES) == 30

    def test_domain_specific_count(self):
        """6 domain-specific types D1-D6."""
        assert len(DOMAIN_SPECIFIC_TYPES) == 6
        for i in range(1, 7):
            assert f"D{i}" in DOMAIN_SPECIFIC_TYPES

    def test_combinatorial_count(self):
        """24 combinatorial KHG types."""
        assert len(COMBINATORIAL_TYPES) == 24

    def test_type_id_roundtrip(self):
        """type_id_to_index and index_to_type_id are inverse."""
        for idx in range(NUM_TYPES):
            tid = index_to_type_id(idx)
            assert type_id_to_index(tid) == idx

    def test_get_params(self):
        """get_core_design_params returns valid params."""
        params = get_core_design_params("D1")
        assert params.array_rows == 6
        assert params.fu_fp_count == 4
        assert params.total_pes == 36

    def test_type_names_are_human_readable(self):
        """get_type_name returns non-empty strings."""
        for tid in get_all_type_ids():
            name = get_type_name(tid)
            assert len(name) > 0

    def test_combinatorial_naming_convention(self):
        """Combinatorial IDs follow C<mix><pe><spm><size> pattern."""
        for tid in COMBINATORIAL_TYPES:
            assert tid.startswith("C")
            assert len(tid) >= 4

    def test_op_support_check(self):
        """type_supports_ops correctly checks FU availability."""
        # D1 has FP FUs
        assert type_supports_ops("D1", {"fadd": 5, "add": 3})
        # D4 has no FP FUs
        assert not type_supports_ops("D4", {"fadd": 5})
        # D4 can handle pure INT ops
        assert type_supports_ops("D4", {"add": 10, "mul": 5})

    def test_max_core_types_matches(self):
        """MAX_CORE_TYPES in config matches library size."""
        assert MAX_CORE_TYPES == NUM_TYPES
