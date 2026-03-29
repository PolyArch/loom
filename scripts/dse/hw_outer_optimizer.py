"""OUTER-HW System-Level Hardware Optimizer.

Bayesian Optimization over the system-level hardware design space using a
30-type core library (6 domain-specific + 24 combinatorial KHG types).

The BO loop suggests DesignPoints with a binary selection mask over the
30 fixed types, per-type instance counts, NoC topology/bandwidth, and
L2 sizing. Candidates are evaluated through a multi-fidelity pipeline:
  Tier-1: Analytical proxy (AnalyticalResourceModel)
  Tier-2: Real compile (tapestry_compile) for representative kernels
  Tier-3: Full simulation (tapestry_simulate) for the complete workload

Uses TDC contract-derived constraints to prune infeasible candidates before
any evaluation tier.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bayesian_opt import AcquisitionType, BayesianOptimizer
from .core_type_library import (
    ALL_TYPES,
    NUM_TYPES,
    CoreDesignParams,
    get_all_type_ids,
    get_core_design_params,
    get_fu_capability_vector,
    index_to_type_id,
    type_id_to_index,
    type_supports_ops,
)
from .design_space import CoreTypeConfig, DesignPoint, DesignSpace
from .dse_config import (
    BOConfig,
    DSEConfig,
    MAX_CORE_TYPES,
    NOC_TOPOLOGIES,
    TIER2_TIMEOUT_SEC,
    TIER3_TIMEOUT_SEC,
    TierThresholds,
)
from .pareto import ParetoEntry, extract_pareto_front, merge_fronts
from .proxy_model import (
    AnalyticalResourceModel,
    ContractEdge,
    KernelProfile,
    ProxyScore,
    WorkloadProfile,
)
from .spectral_clustering import CoreTypeDiscovery
from .system_graph_generator import (
    SystemGraphGenerator,
    SystemTopologySpec,
    to_system_mlir,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core-type role classification
# ---------------------------------------------------------------------------

class CoreRole(Enum):
    """High-level classification of a core type's role in the system."""

    FP_HEAVY = "fp_heavy"
    CONTROL_HEAVY = "control_heavy"
    MEMORY_HEAVY = "memory_heavy"
    BALANCED = "balanced"


# ---------------------------------------------------------------------------
# Core type library specification (output of OUTER-HW)
# ---------------------------------------------------------------------------

@dataclass
class CoreTypeEntry:
    """Specification for one core type in the library."""

    type_index: int = 0
    type_id: str = ""
    role: CoreRole = CoreRole.BALANCED
    instance_count: int = 1

    min_pes: int = 4
    min_spm_kb: int = 4
    required_fu_types: List[str] = field(default_factory=list)
    assigned_kernels: List[str] = field(default_factory=list)

    core_config: Optional[CoreTypeConfig] = None
    design_params: Optional[CoreDesignParams] = None


@dataclass
class CoreTypeLibrary:
    """The complete core type library produced by OUTER-HW."""

    entries: List[CoreTypeEntry] = field(default_factory=list)

    @property
    def num_types(self) -> int:
        return len(self.entries)

    @property
    def total_instances(self) -> int:
        return sum(e.instance_count for e in self.entries)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"core_types": []}
        for entry in self.entries:
            d: Dict[str, Any] = {
                "type_index": entry.type_index,
                "type_id": entry.type_id,
                "role": entry.role.value,
                "instance_count": entry.instance_count,
                "min_pes": entry.min_pes,
                "min_spm_kb": entry.min_spm_kb,
                "required_fu_types": entry.required_fu_types,
                "assigned_kernels": entry.assigned_kernels,
            }
            if entry.design_params is not None:
                d["design_params"] = entry.design_params.to_dict()
            result["core_types"].append(d)
        return result


# ---------------------------------------------------------------------------
# TDC constraint bounds (derived from contracts)
# ---------------------------------------------------------------------------

@dataclass
class TDCBounds:
    """Hard lower bounds derived from TDC contracts."""

    min_noc_bandwidth: float = 0.0
    min_l2_size_kb: float = 0.0
    min_core_types: int = 1
    min_total_cores: int = 1

    def summary(self) -> str:
        return (
            f"TDC bounds: NoC BW >= {self.min_noc_bandwidth:.1f}, "
            f"L2 >= {self.min_l2_size_kb:.0f} KB, "
            f"core types >= {self.min_core_types}, "
            f"total cores >= {self.min_total_cores}"
        )


def compute_tdc_bounds(
    workload: WorkloadProfile,
    element_size_bytes: int = 4,
    base_flit_width_bytes: int = 32,
) -> TDCBounds:
    """Derive hard lower bounds from TDC contracts."""
    bounds = TDCBounds()

    total_bandwidth_bytes = 0.0
    for contract in workload.contracts:
        volume = contract.production_rate * contract.element_size_bytes
        total_bandwidth_bytes += volume
    bw_multiplier = total_bandwidth_bytes / max(1, base_flit_width_bytes)
    bounds.min_noc_bandwidth = max(1.0, min(4.0, bw_multiplier))

    total_shared_volume = 0.0
    for contract in workload.contracts:
        volume = (
            contract.production_rate
            * contract.element_size_bytes
            * 2
        )
        total_shared_volume += volume
    bounds.min_l2_size_kb = max(64.0, total_shared_volume / 1024.0)

    fu_categories: set = set()
    for kernel in workload.kernels:
        cat = _classify_kernel_fu_needs(kernel)
        fu_categories.add(cat)
    bounds.min_core_types = max(1, len(fu_categories))
    bounds.min_total_cores = max(1, len(workload.kernels))

    return bounds


def _classify_kernel_fu_needs(kernel: KernelProfile) -> str:
    """Classify a kernel by its dominant FU requirement."""
    total_ops = max(1, sum(kernel.op_histogram.values()))

    fp_frac = 0.0
    mem_frac = 0.0
    control_frac = 0.0

    for op_name, count in kernel.op_histogram.items():
        op_lower = op_name.lower()
        if any(k in op_lower for k in ("fp", "float", "fadd", "fmul", "fdiv")):
            fp_frac += count
        elif any(k in op_lower for k in ("load", "store", "mem")):
            mem_frac += count
        elif any(k in op_lower for k in ("cmp", "select", "br", "mux")):
            control_frac += count

    fp_frac /= total_ops
    mem_frac /= total_ops
    control_frac /= total_ops

    if fp_frac > 0.3:
        return "fp_heavy"
    if mem_frac > 0.4:
        return "memory_heavy"
    if control_frac > 0.3:
        return "control_heavy"
    return "balanced"


# ---------------------------------------------------------------------------
# Kernel-to-core assignment (T5)
# ---------------------------------------------------------------------------

def compute_kernel_type_affinity(
    kernels: List[KernelProfile],
    selected_type_ids: List[str],
) -> np.ndarray:
    """Compute affinity matrix: (n_kernels x n_selected_types).

    Higher affinity means the kernel matches the core type better.
    """
    n_kernels = len(kernels)
    n_types = len(selected_type_ids)
    affinity = np.zeros((n_kernels, n_types))

    for ki, kernel in enumerate(kernels):
        total_ops = max(1, sum(kernel.op_histogram.values()))
        for ti, type_id in enumerate(selected_type_ids):
            if not type_supports_ops(type_id, kernel.op_histogram):
                affinity[ki, ti] = -1.0
                continue

            params = get_core_design_params(type_id)
            fu_mix = params.fu_mix
            score = 0.0

            # FU match: ratio of kernel demand to core supply
            for op_name, count in kernel.op_histogram.items():
                if count == 0:
                    continue
                fu_key = _op_to_fu_key(op_name)
                available = fu_mix.get(fu_key, 0)
                if available > 0:
                    score += min(1.0, available / max(1, count / 4.0))

            # PE capacity bonus
            if params.total_pes >= kernel.dfg_node_count:
                score += 1.0
            else:
                score += params.total_pes / max(1, kernel.dfg_node_count)

            # SPM capacity bonus
            spm_bytes = params.spm_size_kb * 1024
            if spm_bytes >= kernel.memory_footprint_bytes:
                score += 0.5
            elif kernel.memory_footprint_bytes > 0:
                score += 0.5 * spm_bytes / kernel.memory_footprint_bytes

            affinity[ki, ti] = score

    return affinity


def assign_kernels_to_types(
    kernels: List[KernelProfile],
    selected_type_ids: List[str],
) -> Dict[str, int]:
    """Greedy kernel-to-type assignment based on affinity.

    Returns a dict mapping kernel name -> index into selected_type_ids.
    If a kernel cannot be assigned (no feasible type), it maps to -1.
    """
    if not kernels or not selected_type_ids:
        return {}

    affinity = compute_kernel_type_affinity(kernels, selected_type_ids)
    n_kernels = len(kernels)
    n_types = len(selected_type_ids)

    # Track load per type for tie-breaking
    type_load = [0] * n_types
    assignment: Dict[str, int] = {}

    for ki in range(n_kernels):
        best_ti = -1
        best_score = -float("inf")

        for ti in range(n_types):
            score = affinity[ki, ti]
            if score < 0:
                continue
            # Tie-break: prefer types with fewer assigned kernels
            adjusted = score - 0.01 * type_load[ti]
            if adjusted > best_score:
                best_score = adjusted
                best_ti = ti

        assignment[kernels[ki].name] = best_ti
        if best_ti >= 0:
            type_load[best_ti] += 1

    return assignment


def validate_coverage(
    kernels: List[KernelProfile],
    assignment: Dict[str, int],
) -> bool:
    """Check that every kernel has at least one feasible assignment."""
    for kernel in kernels:
        if assignment.get(kernel.name, -1) < 0:
            return False
    return True


def _op_to_fu_key(op_type: str) -> str:
    """Map operation type names to FU category keys."""
    op_lower = op_type.lower()
    if any(k in op_lower for k in ("mul", "div", "rem")):
        return "mul"
    if any(k in op_lower for k in ("fp", "float", "fadd", "fmul", "fdiv")):
        return "fp"
    if any(k in op_lower for k in ("load", "store", "mem")):
        return "mem"
    return "alu"


# ---------------------------------------------------------------------------
# Multi-fidelity evaluation (T7)
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result from one tier of evaluation."""

    tier: int = 1
    score: ProxyScore = field(default_factory=ProxyScore)
    mapping_success_rate: float = 0.0
    mean_achieved_ii: float = 0.0


def evaluate_tier1(
    design_point: DesignPoint,
    workload: WorkloadProfile,
    assignment: Dict[str, int],
    selected_type_ids: List[str],
) -> EvalResult:
    """Tier-1: analytical system evaluation using per-kernel-per-type scoring.

    Aggregates via geometric mean of per-kernel throughput/area.
    """
    model = AnalyticalResourceModel()

    # Build a legacy DesignPoint with CoreTypeConfig for the proxy model
    legacy = _to_legacy_design_point(design_point, selected_type_ids, assignment, workload)
    proxy_score = model.evaluate(legacy, workload)

    return EvalResult(
        tier=1,
        score=proxy_score,
    )


def evaluate_tier2(
    design_point: DesignPoint,
    workload: WorkloadProfile,
    assignment: Dict[str, int],
    selected_type_ids: List[str],
    topology_spec: SystemTopologySpec,
    pipeline_bin: str = "tapestry-pipeline",
    timeout_sec: int = TIER2_TIMEOUT_SEC,
) -> Optional[EvalResult]:
    """Tier-2: invoke tapestry_compile for representative kernels.

    Picks top-3 kernels by DFG size per core type, compiles them,
    and aggregates mapping success rate and mean achieved II.
    """
    # Select representative kernels: top 3 by DFG size per type
    type_kernels: Dict[int, List[KernelProfile]] = {}
    for kernel in workload.kernels:
        ti = assignment.get(kernel.name, -1)
        if ti < 0:
            continue
        type_kernels.setdefault(ti, []).append(kernel)

    representative_kernels: List[KernelProfile] = []
    for ti, kerns in type_kernels.items():
        sorted_kerns = sorted(kerns, key=lambda k: k.dfg_node_count, reverse=True)
        representative_kernels.extend(sorted_kerns[:3])

    if not representative_kernels:
        return None

    with tempfile.TemporaryDirectory(prefix="outer_hw_t2_") as tmpdir:
        # Write system topology spec
        spec_path = os.path.join(tmpdir, "topology_spec.json")
        with open(spec_path, "w") as f:
            f.write(topology_spec.to_json())

        # Write arch JSON
        arch_path = os.path.join(tmpdir, "arch.json")
        with open(arch_path, "w") as f:
            json.dump(design_point.to_arch_json(), f)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            pipeline_bin,
            "--system-arch", arch_path,
            "--o", output_dir,
            "--max-benders-iter", "1",
            "--enable-sim", "false",
            "--enable-rtl", "false",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("Tier-2 compile unavailable or timed out")
            return None

        if result.returncode != 0:
            return None

        return _parse_tier2_output(output_dir, representative_kernels)


def evaluate_tier3(
    design_point: DesignPoint,
    workload: WorkloadProfile,
    topology_spec: SystemTopologySpec,
    pipeline_bin: str = "tapestry-pipeline",
    timeout_sec: int = TIER3_TIMEOUT_SEC,
) -> Optional[EvalResult]:
    """Tier-3: invoke tapestry_simulate for the full workload.

    Returns throughput = iterations / cycles, normalized by area.
    """
    with tempfile.TemporaryDirectory(prefix="outer_hw_t3_") as tmpdir:
        spec_path = os.path.join(tmpdir, "topology_spec.json")
        with open(spec_path, "w") as f:
            f.write(topology_spec.to_json())

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            pipeline_bin,
            "--system-arch", spec_path,
            "--o", output_dir,
            "--enable-sim", "true",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("Tier-3 simulation unavailable or timed out")
            return None

        if result.returncode != 0:
            return None

        return _parse_tier3_output(output_dir)


def _parse_tier2_output(
    output_dir: str,
    representative_kernels: List[KernelProfile],
) -> Optional[EvalResult]:
    """Parse compile output for Tier-2 score."""
    report_path = os.path.join(output_dir, "report.json")
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    successes = report.get("mapping_successes", 0)
    total = report.get("mapping_total", len(representative_kernels))
    mean_ii = report.get("mean_achieved_ii", 0.0)
    throughput = report.get("throughput", 0.0)
    area = report.get("area_um2", 1.0)

    success_rate = successes / max(1, total)

    return EvalResult(
        tier=2,
        score=ProxyScore(
            throughput=throughput,
            area_um2=area,
            feasible=success_rate > 0,
        ),
        mapping_success_rate=success_rate,
        mean_achieved_ii=mean_ii,
    )


def _parse_tier3_output(output_dir: str) -> Optional[EvalResult]:
    """Parse simulation output for Tier-3 score."""
    report_path = os.path.join(output_dir, "report.json")
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    cycles = report.get("total_cycles", 0)
    iterations = report.get("total_iterations", 1)
    area = report.get("area_um2", 1.0)

    throughput = iterations / max(1, cycles)

    return EvalResult(
        tier=3,
        score=ProxyScore(
            throughput=throughput,
            area_um2=area,
            feasible=True,
        ),
    )


def _to_legacy_design_point(
    design_point: DesignPoint,
    selected_type_ids: List[str],
    assignment: Dict[str, int],
    workload: WorkloadProfile,
) -> DesignPoint:
    """Convert 30-type DesignPoint to legacy format for AnalyticalResourceModel.

    Maps each selected type to a CoreTypeConfig, and updates kernel assignments.
    """
    legacy = DesignPoint()
    legacy.noc_topology = design_point.noc_topology
    legacy.noc_bandwidth = design_point.noc_bandwidth
    legacy.l2_size_kb = design_point.l2_size_kb
    legacy.l2_bank_count = design_point.l2_bank_count
    legacy.type_mask = list(design_point.type_mask)
    legacy.instance_counts = list(design_point.instance_counts)

    core_types = []
    type_idx_map: Dict[int, int] = {}

    for ti, type_id in enumerate(selected_type_ids):
        params = get_core_design_params(type_id)
        ct = CoreTypeConfig(
            pe_grid_rows=params.array_rows,
            pe_grid_cols=params.array_cols,
            fu_alu_count=params.fu_alu_count,
            fu_mul_count=params.fu_mul_count,
            fu_fp_count=params.fu_fp_count,
            fu_mem_count=params.fu_mem_count,
            spm_size_kb=params.spm_size_kb,
            instance_count=design_point.instance_counts[
                type_id_to_index(type_id)
            ] if type_id_to_index(type_id) < len(design_point.instance_counts) else 1,
        )
        type_idx_map[ti] = len(core_types)
        core_types.append(ct)

    legacy.core_types = core_types

    # Update kernel assignments to use legacy core_types indices
    for kernel in workload.kernels:
        ti = assignment.get(kernel.name, 0)
        legacy_idx = type_idx_map.get(ti, 0)
        kernel.assigned_core_type_idx = legacy_idx

    return legacy


# ---------------------------------------------------------------------------
# OUTER-HW Optimizer
# ---------------------------------------------------------------------------

@dataclass
class OuterHWResult:
    """Result of the OUTER-HW optimization."""

    topology: SystemTopologySpec = field(default_factory=SystemTopologySpec)
    best_score: float = 0.0
    iterations_used: int = 0
    tdc_rejections: int = 0
    tier1_evaluations: int = 0
    tier2_evaluations: int = 0
    tier3_evaluations: int = 0
    wall_time_sec: float = 0.0

    # Pareto front entries
    pareto_front: List[ParetoEntry] = field(default_factory=list)

    # Convergence trace: (iteration, best_score_so_far)
    convergence_trace: List[Tuple[int, float]] = field(default_factory=list)

    # Kernel-to-type assignment
    kernel_assignment: Dict[str, int] = field(default_factory=dict)

    # Type selection mask as type IDs
    selected_type_ids: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "OUTER-HW Optimization Result:",
            f"  Best score: {self.best_score:.6f}",
            f"  Iterations: {self.iterations_used}",
            f"  TDC rejections: {self.tdc_rejections}",
            f"  Tier-1 evals: {self.tier1_evaluations}",
            f"  Tier-2 evals: {self.tier2_evaluations}",
            f"  Tier-3 evals: {self.tier3_evaluations}",
            f"  Wall time: {self.wall_time_sec:.2f}s",
            f"  Types selected: {len(self.selected_type_ids)}",
            f"  NoC: {self.topology.noc_topology} "
            f"BW={self.topology.noc_bandwidth}",
            f"  L2: {self.topology.l2_total_size_kb} KB "
            f"({self.topology.l2_bank_count} banks)",
        ]
        if self.pareto_front:
            lines.append(f"  Pareto front size: {len(self.pareto_front)}")
        return "\n".join(lines)


class HWOuterOptimizer:
    """System-level hardware optimizer using Bayesian Optimization.

    Explores the 30-type design space (binary selection mask, instance counts,
    NoC topology, shared memory) with TDC-derived constraint pruning and
    multi-fidelity evaluation.
    """

    def __init__(
        self,
        workload: WorkloadProfile,
        config: Optional[DSEConfig] = None,
        bo_config: Optional[BOConfig] = None,
        tier_thresholds: Optional[TierThresholds] = None,
    ):
        self.workload = workload
        self.dse_config = config or DSEConfig()
        self.bo_config = bo_config or self.dse_config.bo
        self.thresholds = tier_thresholds or self.dse_config.tier_thresholds

        # Compute TDC bounds from contracts
        self.tdc_bounds = compute_tdc_bounds(workload)
        logger.info(self.tdc_bounds.summary())

        # Tier-1 proxy model
        self.proxy = AnalyticalResourceModel()

        # System graph generator
        self.graph_gen = SystemGraphGenerator()

        # Track the best result
        self._best_design: Optional[DesignPoint] = None
        self._best_score: float = float("-inf")
        self._best_eval: Optional[EvalResult] = None
        self._best_assignment: Dict[str, int] = {}
        self._best_selected_ids: List[str] = []

        # Pareto front
        self._pareto_entries: List[ParetoEntry] = []
        self._convergence_trace: List[Tuple[int, float]] = []

    def optimize(
        self,
        max_iterations: Optional[int] = None,
        seed_type_indices: Optional[List[int]] = None,
    ) -> OuterHWResult:
        """Run the OUTER-HW optimization loop.

        Args:
            max_iterations: Override BO iteration budget.
            seed_type_indices: Initial type indices from clustering.
                If not provided, runs spectral-clustering-informed selection.

        Returns:
            OuterHWResult with the best system topology.
        """
        wall_start = time.time()
        max_iter = max_iterations or self.bo_config.max_iterations

        # Discover initial type selection if not provided
        if seed_type_indices is None:
            seed_type_indices = self._discover_initial_types()

        # Set up the design space with TDC-aware bounds
        space = self._create_constrained_space()
        optimizer = BayesianOptimizer(
            space=space,
            config=self.bo_config,
            acquisition=AcquisitionType.EI,
        )

        tdc_rejections = 0
        tier1_evals = 0
        tier2_evals = 0
        tier3_evals = 0

        all_type_ids = get_all_type_ids()

        for iteration in range(max_iter):
            candidate = optimizer.suggest()

            # Inject seed type selection into initial candidates
            if seed_type_indices and iteration < self.bo_config.n_initial_samples:
                candidate = self._inject_seed_selection(
                    candidate, seed_type_indices
                )

            # TDC feasibility pruning
            if not self._is_tdc_feasible(candidate):
                optimizer.observe(candidate, float("-inf"))
                tdc_rejections += 1
                self._convergence_trace.append(
                    (iteration, self._best_score)
                )
                continue

            # Get selected type IDs
            selected_ids = []
            for i in candidate.selected_type_indices():
                if i < len(all_type_ids):
                    selected_ids.append(all_type_ids[i])

            if not selected_ids:
                optimizer.observe(candidate, float("-inf"))
                tdc_rejections += 1
                self._convergence_trace.append(
                    (iteration, self._best_score)
                )
                continue

            # Kernel-to-core assignment
            assignment = assign_kernels_to_types(
                self.workload.kernels, selected_ids
            )

            # Validate coverage
            if not validate_coverage(self.workload.kernels, assignment):
                optimizer.observe(candidate, 0.0)
                self._convergence_trace.append(
                    (iteration, self._best_score)
                )
                continue

            # Tier-1: analytical system evaluation
            eval_result = evaluate_tier1(
                candidate, self.workload, assignment, selected_ids
            )
            tier1_evals += 1

            if not eval_result.score.feasible:
                optimizer.observe(candidate, 0.0)
                self._convergence_trace.append(
                    (iteration, self._best_score)
                )
                continue

            composite = eval_result.score.composite_score()
            best_tier = 1

            # Tier-2: real compile for promising candidates
            if composite >= self.thresholds.tier2_promotion:
                topo_spec = self.graph_gen.generate(candidate)
                tier2_result = evaluate_tier2(
                    candidate, self.workload, assignment, selected_ids,
                    topo_spec,
                    pipeline_bin=self.dse_config.tapestry_pipeline_bin,
                )
                tier2_evals += 1

                if tier2_result is not None and tier2_result.score.feasible:
                    composite = tier2_result.score.composite_score()
                    eval_result = tier2_result
                    best_tier = 2

                    # Tier-3: full simulation for top candidates
                    if composite >= self.thresholds.tier3_promotion:
                        tier3_result = evaluate_tier3(
                            candidate, self.workload, topo_spec,
                            pipeline_bin=self.dse_config.tapestry_pipeline_bin,
                        )
                        tier3_evals += 1

                        if tier3_result is not None and tier3_result.score.feasible:
                            composite = tier3_result.score.composite_score()
                            eval_result = tier3_result
                            best_tier = 3

            optimizer.observe(candidate, composite)

            # Track Pareto front
            if eval_result.score.feasible:
                entry = ParetoEntry(
                    point=candidate,
                    throughput=eval_result.score.throughput,
                    area=eval_result.score.area_um2,
                    score=eval_result.score,
                    tier=best_tier,
                )
                self._pareto_entries = merge_fronts(
                    self._pareto_entries, [entry]
                )

            # Track best
            if composite > self._best_score:
                self._best_score = composite
                self._best_design = copy.deepcopy(candidate)
                self._best_eval = eval_result
                self._best_assignment = dict(assignment)
                self._best_selected_ids = list(selected_ids)

            self._convergence_trace.append(
                (iteration, self._best_score)
            )

        wall_time = time.time() - wall_start

        # Build the output topology spec
        topology = self._build_topology_spec()

        return OuterHWResult(
            topology=topology,
            best_score=self._best_score,
            iterations_used=max_iter,
            tdc_rejections=tdc_rejections,
            tier1_evaluations=tier1_evals,
            tier2_evaluations=tier2_evals,
            tier3_evaluations=tier3_evals,
            wall_time_sec=wall_time,
            pareto_front=list(self._pareto_entries),
            convergence_trace=list(self._convergence_trace),
            kernel_assignment=dict(self._best_assignment),
            selected_type_ids=list(self._best_selected_ids),
        )

    # -------------------------------------------------------------------
    # TDC constraint checking
    # -------------------------------------------------------------------

    def _is_tdc_feasible(self, candidate: DesignPoint) -> bool:
        """Check whether a candidate satisfies TDC-derived hard constraints."""
        if candidate.noc_bandwidth < self.tdc_bounds.min_noc_bandwidth:
            return False
        if candidate.l2_size_kb < self.tdc_bounds.min_l2_size_kb:
            return False
        if candidate.num_selected_types() < self.tdc_bounds.min_core_types:
            return False
        if candidate.total_cores() < self.tdc_bounds.min_total_cores:
            return False
        return True

    # -------------------------------------------------------------------
    # Initial type discovery via spectral clustering
    # -------------------------------------------------------------------

    def _discover_initial_types(self) -> List[int]:
        """Map spectral clustering output to nearest types from the library.

        Uses FU mix and array size similarity to match each cluster centroid
        to the best-matching type from the 30-type library.
        """
        kernels = self.workload.kernels
        if len(kernels) < 2:
            return [0]

        discovery = CoreTypeDiscovery(
            k_range=(
                max(1, self.tdc_bounds.min_core_types),
                min(5, len(kernels)),
            ),
            seed=self.bo_config.seed,
        )
        cluster_result = discovery.discover(kernels)

        logger.info(
            "Initial clustering: %d core types "
            "(silhouette=%.3f, feasibility=%.3f)",
            cluster_result.k,
            cluster_result.silhouette_score,
            cluster_result.feasibility_score,
        )

        # Map each cluster's derived config to the nearest library type
        all_type_ids = get_all_type_ids()
        selected_indices: List[int] = []

        for ct in cluster_result.core_types:
            best_idx = 0
            best_dist = float("inf")

            for idx, type_id in enumerate(all_type_ids):
                params = get_core_design_params(type_id)
                dist = _config_distance(ct, params)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx not in selected_indices:
                selected_indices.append(best_idx)

        logger.info(
            "Mapped clusters to library types: %s",
            [all_type_ids[i] for i in selected_indices],
        )

        return selected_indices

    # -------------------------------------------------------------------
    # Design space construction
    # -------------------------------------------------------------------

    def _create_constrained_space(self) -> DesignSpace:
        """Create a design space with TDC-aware lower bounds."""
        from .dse_config import DEFAULT_PARAM_RANGES

        ranges = dict(DEFAULT_PARAM_RANGES)

        noc_lo, noc_hi = ranges["noc_bandwidth"]
        ranges["noc_bandwidth"] = (
            max(noc_lo, int(math.ceil(self.tdc_bounds.min_noc_bandwidth))),
            noc_hi,
        )

        l2_lo, l2_hi = ranges["l2_size_kb"]
        ranges["l2_size_kb"] = (
            max(l2_lo, int(math.ceil(self.tdc_bounds.min_l2_size_kb))),
            l2_hi,
        )

        return DesignSpace(
            param_ranges=ranges,
            seed=self.bo_config.seed,
        )

    # -------------------------------------------------------------------
    # Seed injection
    # -------------------------------------------------------------------

    @staticmethod
    def _inject_seed_selection(
        candidate: DesignPoint,
        seed_indices: List[int],
    ) -> DesignPoint:
        """Ensure seed types are selected in the candidate."""
        result = copy.deepcopy(candidate)
        for idx in seed_indices:
            if idx < MAX_CORE_TYPES:
                result.type_mask[idx] = True
                if result.instance_counts[idx] < 1:
                    result.instance_counts[idx] = 1
        return result

    # -------------------------------------------------------------------
    # Build output topology spec
    # -------------------------------------------------------------------

    def _build_topology_spec(self) -> SystemTopologySpec:
        """Convert the best DesignPoint into a SystemTopologySpec."""
        if self._best_design is None:
            return SystemTopologySpec()

        return self.graph_gen.generate(self._best_design)


# ---------------------------------------------------------------------------
# Helper: distance between CoreTypeConfig and CoreDesignParams
# ---------------------------------------------------------------------------

def _config_distance(ct: CoreTypeConfig, params: CoreDesignParams) -> float:
    """Euclidean distance in FU-mix + array-size space."""
    d = 0.0
    d += (ct.fu_alu_count - params.fu_alu_count) ** 2
    d += (ct.fu_mul_count - params.fu_mul_count) ** 2
    d += (ct.fu_fp_count - params.fu_fp_count) ** 2
    d += (ct.fu_mem_count - params.fu_mem_count) ** 2
    d += (ct.pe_grid_rows - params.array_rows) ** 2
    d += (ct.pe_grid_cols - params.array_cols) ** 2
    d += ((ct.spm_size_kb - params.spm_size_kb) / 8.0) ** 2
    return math.sqrt(d)


# ---------------------------------------------------------------------------
# CLI entry point for standalone use
# ---------------------------------------------------------------------------

def run_outer_hw(
    workload: WorkloadProfile,
    max_iterations: int = 100,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> OuterHWResult:
    """Convenience function for running OUTER-HW standalone."""
    bo_config = BOConfig(
        max_iterations=max_iterations,
        seed=seed,
    )

    optimizer = HWOuterOptimizer(
        workload=workload,
        bo_config=bo_config,
    )

    result = optimizer.optimize()
    logger.info(result.summary())

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(result.topology.to_json())
        logger.info("Topology spec written to %s", output_path)

    return result


# ---------------------------------------------------------------------------
# DSE results output (T8)
# ---------------------------------------------------------------------------

def write_dse_results(
    result: OuterHWResult,
    output_path: str,
) -> None:
    """Write full DSE results to JSON for post-hoc analysis."""
    data: Dict[str, Any] = {
        "best_score": result.best_score,
        "iterations_used": result.iterations_used,
        "tdc_rejections": result.tdc_rejections,
        "tier1_evaluations": result.tier1_evaluations,
        "tier2_evaluations": result.tier2_evaluations,
        "tier3_evaluations": result.tier3_evaluations,
        "wall_time_sec": result.wall_time_sec,
        "selected_type_ids": result.selected_type_ids,
        "kernel_assignment": result.kernel_assignment,
        "convergence_trace": [
            {"iteration": it, "best_score": sc}
            for it, sc in result.convergence_trace
        ],
        "pareto_front": [
            {
                "throughput": e.throughput,
                "area": e.area,
                "tier": e.tier,
            }
            for e in result.pareto_front
        ],
        "topology": result.topology.to_dict(),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
