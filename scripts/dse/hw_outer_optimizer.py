"""OUTER-HW System-Level Hardware Optimizer.

Bayesian Optimization over the system-level hardware design space:
  - Number of core types and instances per type
  - NoC topology and bandwidth
  - L2 sizing (bank count, total size)

Uses TDC contract-derived constraints to prune infeasible candidates before
Tier-1 analytical evaluation. Produces a CoreTypeLibrary and
SystemTopologySpec that INNER-HW (C12) refines into concrete ADGs.
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
from .design_space import CoreTypeConfig, DesignPoint, DesignSpace
from .dse_config import BOConfig, DSEConfig, NOC_TOPOLOGIES, TierThresholds
from .pareto import ParetoEntry, extract_pareto_front, merge_fronts
from .proxy_model import (
    ContractEdge,
    ContractProxy,
    KernelProfile,
    ProxyScore,
    WorkloadProfile,
)
from .spectral_clustering import CoreTypeDiscovery

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
    """Specification for one core type in the library.

    OUTER-HW decides role, instance count, and resource hints.
    INNER-HW (C12) fills in the concrete microarchitecture.
    """

    type_index: int = 0
    role: CoreRole = CoreRole.BALANCED
    instance_count: int = 1

    # Resource hints derived from assigned kernels
    min_pes: int = 4
    min_spm_kb: int = 4
    required_fu_types: List[str] = field(default_factory=list)

    # Kernel names assigned to this core type
    assigned_kernels: List[str] = field(default_factory=list)

    # Underlying per-core DSE config seed
    core_config: Optional[CoreTypeConfig] = None


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
            result["core_types"].append({
                "type_index": entry.type_index,
                "role": entry.role.value,
                "instance_count": entry.instance_count,
                "min_pes": entry.min_pes,
                "min_spm_kb": entry.min_spm_kb,
                "required_fu_types": entry.required_fu_types,
                "assigned_kernels": entry.assigned_kernels,
            })
        return result


# ---------------------------------------------------------------------------
# System topology specification (output of OUTER-HW)
# ---------------------------------------------------------------------------

@dataclass
class SystemTopologySpec:
    """System-level topology specification produced by OUTER-HW.

    Describes NoC structure, shared memory configuration, and core placement.
    """

    # NoC parameters
    noc_topology: str = "mesh"
    noc_bandwidth: int = 1
    mesh_rows: int = 2
    mesh_cols: int = 2

    # Shared L2 memory
    l2_total_size_kb: int = 256
    l2_bank_count: int = 4

    # Core type library
    core_library: CoreTypeLibrary = field(default_factory=CoreTypeLibrary)

    # Core placement: list of (type_index, instance_id, row, col)
    core_placement: List[Tuple[int, int, int, int]] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "noc": {
                "topology": self.noc_topology,
                "bandwidth": self.noc_bandwidth,
                "mesh_rows": self.mesh_rows,
                "mesh_cols": self.mesh_cols,
            },
            "shared_memory": {
                "l2_total_size_kb": self.l2_total_size_kb,
                "l2_bank_count": self.l2_bank_count,
            },
            "core_library": self.core_library.to_dict(),
            "core_placement": [
                {"type": t, "instance": i, "row": r, "col": c}
                for t, i, r, c in self.core_placement
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# TDC constraint bounds (derived from contracts)
# ---------------------------------------------------------------------------

@dataclass
class TDCBounds:
    """Hard lower bounds derived from TDC contracts.

    Any candidate violating these bounds is rejected before Tier-1 evaluation.
    """

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
    """Derive hard lower bounds from TDC contracts.

    - Sum of cross-core contract bandwidths -> NoC bandwidth multiplier
    - Sum of SHARED_L2 contract data volumes -> L2 size lower bound
    - Kernel FU type diversity -> core type count lower bound
    - Total kernel count -> core instance count lower bound

    The NoC bandwidth bound is expressed as a flit-width multiplier (1-4),
    matching the design space parameter range. Total bytes/cycle requirement
    is divided by the base flit width and capped at the design space max.
    """
    bounds = TDCBounds()

    # NoC bandwidth: sum of inter-kernel data rates (bytes/cycle)
    # divided by base flit width to get the multiplier (1-4)
    total_bandwidth_bytes = 0.0
    for contract in workload.contracts:
        volume = contract.production_rate * contract.element_size_bytes
        total_bandwidth_bytes += volume
    # Convert to multiplier; cap at 1 minimum (the design space floor)
    bw_multiplier = total_bandwidth_bytes / max(1, base_flit_width_bytes)
    bounds.min_noc_bandwidth = max(1.0, min(4.0, bw_multiplier))

    # L2 size: total shared data volume across all contracts
    total_shared_volume = 0.0
    for contract in workload.contracts:
        volume = (
            contract.production_rate
            * contract.element_size_bytes
            * 2  # double-buffering headroom
        )
        total_shared_volume += volume
    bounds.min_l2_size_kb = max(64.0, total_shared_volume / 1024.0)

    # Core type diversity: count distinct FU requirement categories
    fu_categories: set = set()
    for kernel in workload.kernels:
        cat = _classify_kernel_fu_needs(kernel)
        fu_categories.add(cat)
    bounds.min_core_types = max(1, len(fu_categories))

    # Total core instances: at least one per kernel
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
# OUTER-HW Optimizer
# ---------------------------------------------------------------------------

@dataclass
class OuterHWResult:
    """Result of the OUTER-HW optimization."""

    topology: SystemTopologySpec
    best_score: float = 0.0
    iterations_used: int = 0
    tdc_rejections: int = 0
    tier1_evaluations: int = 0
    tier2_evaluations: int = 0
    wall_time_sec: float = 0.0

    def summary(self) -> str:
        lines = [
            "OUTER-HW Optimization Result:",
            f"  Best score: {self.best_score:.6f}",
            f"  Iterations: {self.iterations_used}",
            f"  TDC rejections: {self.tdc_rejections}",
            f"  Tier-1 evals: {self.tier1_evaluations}",
            f"  Tier-2 evals: {self.tier2_evaluations}",
            f"  Wall time: {self.wall_time_sec:.2f}s",
            f"  Core types: {self.topology.core_library.num_types}",
            f"  Total cores: {self.topology.core_library.total_instances}",
            f"  NoC: {self.topology.noc_topology} "
            f"BW={self.topology.noc_bandwidth}",
            f"  L2: {self.topology.l2_total_size_kb} KB "
            f"({self.topology.l2_bank_count} banks)",
        ]
        return "\n".join(lines)


class HWOuterOptimizer:
    """System-level hardware optimizer using Bayesian Optimization.

    Explores the system-level design space (core type library, NoC topology,
    shared memory) with TDC-derived constraint pruning and multi-fidelity
    evaluation (Tier-1 analytical, Tier-2 real compile).
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
        self.proxy = ContractProxy()

        # Track the best result
        self._best_design: Optional[DesignPoint] = None
        self._best_score: float = float("-inf")
        self._best_proxy: Optional[ProxyScore] = None

    def optimize(
        self,
        max_iterations: Optional[int] = None,
        seed_core_types: Optional[List[CoreTypeConfig]] = None,
    ) -> OuterHWResult:
        """Run the OUTER-HW optimization loop.

        Args:
            max_iterations: Override BO iteration budget.
            seed_core_types: Initial core types from spectral clustering.
                If not provided, runs clustering internally.

        Returns:
            OuterHWResult with the best system topology.
        """
        wall_start = time.time()
        max_iter = max_iterations or self.bo_config.max_iterations

        # Discover initial core types if not provided
        if seed_core_types is None:
            seed_core_types = self._discover_initial_core_types()

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

        for iteration in range(max_iter):
            candidate = optimizer.suggest()

            # Inject seed core type structure
            if seed_core_types:
                candidate = self._inject_seed_types(
                    candidate, seed_core_types
                )

            # TDC feasibility pruning (microseconds)
            if not self._is_tdc_feasible(candidate):
                optimizer.observe(candidate, float("-inf"))
                tdc_rejections += 1
                continue

            # Tier-1: analytical system evaluation
            proxy_score = self.proxy.evaluate(candidate, self.workload)
            tier1_evals += 1

            if not proxy_score.feasible:
                optimizer.observe(candidate, 0.0)
                continue

            composite = proxy_score.composite_score()

            # Tier-2: real compile for promising candidates
            if composite >= self.thresholds.tier2_promotion:
                tier2_score = self._evaluate_tier2(candidate)
                tier2_evals += 1

                if tier2_score is not None:
                    composite = tier2_score.composite_score()
                    proxy_score = tier2_score

            optimizer.observe(candidate, composite)

            # Track best
            if composite > self._best_score:
                self._best_score = composite
                self._best_design = copy.deepcopy(candidate)
                self._best_proxy = proxy_score

        wall_time = time.time() - wall_start

        # Build the output topology spec
        topology = self._build_topology_spec(
            self._best_design, self._best_proxy
        )

        return OuterHWResult(
            topology=topology,
            best_score=self._best_score,
            iterations_used=max_iter,
            tdc_rejections=tdc_rejections,
            tier1_evaluations=tier1_evals,
            tier2_evaluations=tier2_evals,
            wall_time_sec=wall_time,
        )

    # -------------------------------------------------------------------
    # TDC constraint checking
    # -------------------------------------------------------------------

    def _is_tdc_feasible(self, candidate: DesignPoint) -> bool:
        """Check whether a candidate satisfies TDC-derived hard constraints.

        Runs in microseconds -- called before any evaluation tier.
        """
        if candidate.noc_bandwidth < self.tdc_bounds.min_noc_bandwidth:
            return False
        if candidate.l2_size_kb < self.tdc_bounds.min_l2_size_kb:
            return False
        if len(candidate.core_types) < self.tdc_bounds.min_core_types:
            return False
        if candidate.total_cores() < self.tdc_bounds.min_total_cores:
            return False
        return True

    # -------------------------------------------------------------------
    # Core type discovery
    # -------------------------------------------------------------------

    def _discover_initial_core_types(self) -> List[CoreTypeConfig]:
        """Use spectral clustering to derive an initial core type library."""
        kernels = self.workload.kernels
        if len(kernels) < 2:
            return []

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

        # Update kernel assignments
        for i, kernel in enumerate(kernels):
            kernel.assigned_core_type_idx = int(cluster_result.labels[i])

        return cluster_result.core_types

    # -------------------------------------------------------------------
    # Design space construction
    # -------------------------------------------------------------------

    def _create_constrained_space(self) -> DesignSpace:
        """Create a design space with TDC-aware lower bounds."""
        from .dse_config import DEFAULT_PARAM_RANGES

        ranges = dict(DEFAULT_PARAM_RANGES)

        # Tighten lower bounds based on TDC constraints
        ct_lo, ct_hi = ranges["core_type_count"]
        ranges["core_type_count"] = (
            max(ct_lo, self.tdc_bounds.min_core_types),
            ct_hi,
        )

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
    def _inject_seed_types(
        candidate: DesignPoint,
        seed_types: List[CoreTypeConfig],
    ) -> DesignPoint:
        """Overlay discovered core-type configs onto the candidate.

        Preserves the candidate's instance_count but uses the seed FU mix,
        grid size, and SPM size as lower bounds.
        """
        result = copy.deepcopy(candidate)
        n = min(len(result.core_types), len(seed_types))
        for i in range(n):
            seed = seed_types[i]
            ct = result.core_types[i]
            ct.fu_alu_count = max(ct.fu_alu_count, seed.fu_alu_count)
            ct.fu_mul_count = max(ct.fu_mul_count, seed.fu_mul_count)
            ct.fu_fp_count = max(ct.fu_fp_count, seed.fu_fp_count)
            ct.fu_mem_count = max(ct.fu_mem_count, seed.fu_mem_count)
            ct.pe_grid_rows = max(ct.pe_grid_rows, seed.pe_grid_rows)
            ct.pe_grid_cols = max(ct.pe_grid_cols, seed.pe_grid_cols)
            ct.spm_size_kb = max(ct.spm_size_kb, seed.spm_size_kb)
        return result

    # -------------------------------------------------------------------
    # Tier-2 evaluation
    # -------------------------------------------------------------------

    def _evaluate_tier2(
        self, candidate: DesignPoint
    ) -> Optional[ProxyScore]:
        """Tier-2: invoke tapestry-pipeline subprocess for real compile.

        Uses a single Benders iteration for a quick feasibility check.
        """
        arch_json = candidate.to_arch_json()

        with tempfile.TemporaryDirectory(prefix="outer_hw_") as tmpdir:
            arch_path = os.path.join(tmpdir, "arch.json")
            with open(arch_path, "w") as f:
                json.dump(arch_json, f)

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                self.dse_config.tapestry_pipeline_bin,
                "--system-arch", arch_path,
                "--o", output_dir,
                "--max-benders-iter", "1",
                "--enable-sim", "false",
                "--enable-rtl", "false",
            ]

            if self.dse_config.workload_tdgs:
                cmd.extend(["--tdg", self.dse_config.workload_tdgs[0]])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except FileNotFoundError:
                logger.debug(
                    "tapestry-pipeline not found; skipping Tier-2"
                )
                return None
            except subprocess.TimeoutExpired:
                logger.debug("Tier-2 evaluation timed out")
                return None

            if result.returncode != 0:
                return None

            return self._parse_compile_output(output_dir, candidate)

    def _parse_compile_output(
        self, output_dir: str, candidate: DesignPoint
    ) -> Optional[ProxyScore]:
        """Parse tapestry-pipeline output JSON for score extraction."""
        report_path = os.path.join(output_dir, "report.json")
        if not os.path.exists(report_path):
            return None

        try:
            with open(report_path, "r") as f:
                report = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        return ProxyScore(
            throughput=report.get("throughput", 0.0),
            area_um2=report.get(
                "area_um2", self.proxy._estimate_area(candidate)
            ),
            communication_cost=report.get("communication_cost", 0.0),
            utilization=report.get("utilization", 0.0),
            feasible=True,
        )

    # -------------------------------------------------------------------
    # Build output topology spec
    # -------------------------------------------------------------------

    def _build_topology_spec(
        self,
        design: Optional[DesignPoint],
        proxy_score: Optional[ProxyScore],
    ) -> SystemTopologySpec:
        """Convert the best DesignPoint into a SystemTopologySpec."""
        if design is None:
            return SystemTopologySpec()

        # Build core type library
        library = CoreTypeLibrary()
        kernels = self.workload.kernels

        for i, ct in enumerate(design.core_types):
            role = self._classify_core_role(ct)
            assigned = [
                k.name
                for k in kernels
                if k.assigned_core_type_idx == i
            ]
            fu_types = []
            if ct.fu_alu_count > 0:
                fu_types.append("alu")
            if ct.fu_mul_count > 0:
                fu_types.append("mul")
            if ct.fu_fp_count > 0:
                fu_types.append("fp")
            if ct.fu_mem_count > 0:
                fu_types.append("mem")

            entry = CoreTypeEntry(
                type_index=i,
                role=role,
                instance_count=ct.instance_count,
                min_pes=ct.num_pes,
                min_spm_kb=ct.spm_size_kb,
                required_fu_types=fu_types,
                assigned_kernels=assigned,
                core_config=ct,
            )
            library.entries.append(entry)

        # Compute mesh dimensions
        total_cores = design.total_cores()
        mesh_side = max(2, math.isqrt(total_cores))
        if mesh_side * mesh_side < total_cores:
            mesh_side += 1
        mesh_rows = mesh_side
        mesh_cols = mesh_side

        # Generate core placement
        placement = self._generate_placement(design, mesh_rows, mesh_cols)

        return SystemTopologySpec(
            noc_topology=design.noc_topology,
            noc_bandwidth=design.noc_bandwidth,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            l2_total_size_kb=design.l2_size_kb,
            l2_bank_count=max(1, design.l2_size_kb // 64),
            core_library=library,
            core_placement=placement,
        )

    @staticmethod
    def _classify_core_role(ct: CoreTypeConfig) -> CoreRole:
        """Classify a core type's role from its FU mix."""
        total_fus = (
            ct.fu_alu_count
            + ct.fu_mul_count
            + ct.fu_fp_count
            + ct.fu_mem_count
        )
        if total_fus == 0:
            return CoreRole.BALANCED

        fp_ratio = ct.fu_fp_count / total_fus
        mem_ratio = ct.fu_mem_count / total_fus

        if fp_ratio > 0.3:
            return CoreRole.FP_HEAVY
        if mem_ratio > 0.4:
            return CoreRole.MEMORY_HEAVY
        return CoreRole.BALANCED

    @staticmethod
    def _generate_placement(
        design: DesignPoint,
        mesh_rows: int,
        mesh_cols: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate core placement coordinates on the mesh.

        Places cores in row-major order by type.
        """
        placement: List[Tuple[int, int, int, int]] = []
        pos = 0
        for type_idx, ct in enumerate(design.core_types):
            for inst in range(ct.instance_count):
                row = pos // mesh_cols
                col = pos % mesh_cols
                if row >= mesh_rows:
                    row = mesh_rows - 1
                    col = min(col, mesh_cols - 1)
                placement.append((type_idx, inst, row, col))
                pos += 1
        return placement


# ---------------------------------------------------------------------------
# CLI entry point for standalone use
# ---------------------------------------------------------------------------

def run_outer_hw(
    workload: WorkloadProfile,
    max_iterations: int = 100,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> OuterHWResult:
    """Convenience function for running OUTER-HW standalone.

    Args:
        workload: Workload profile with kernels and contracts.
        max_iterations: BO iteration budget.
        seed: Random seed.
        output_path: Optional JSON output path for the topology spec.

    Returns:
        OuterHWResult with the best system topology found.
    """
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
