"""Analytical resource model for Tier-1 DSE evaluation.

Provides fast (~1ms) architecture quality estimates from resource constraints
(operation histograms, FU counts, area formulas) and contract-informed
estimations (cache stall via visibility, pipeline stall via ordering).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .design_space import CoreTypeConfig, DesignPoint
from .dse_config import (
    CACHE_MISS_RATES,
    FU_AREA_TABLE,
    PE_AREA_UM2,
    SRAM_AREA_PER_BYTE_UM2,
    SW_AREA_UM2,
)


# ---------------------------------------------------------------------------
# Workload description (simplified TDG summary)
# ---------------------------------------------------------------------------

@dataclass
class KernelProfile:
    """Profile of a single kernel extracted from a TDG."""

    name: str = ""

    # Operation histogram: op_type -> count per iteration.
    op_histogram: Dict[str, int] = field(default_factory=dict)

    # Memory footprint in bytes per iteration.
    memory_footprint_bytes: int = 0

    # Number of memory load operations per iteration.
    loads_per_iter: int = 0

    # Number of memory store operations per iteration.
    stores_per_iter: int = 0

    # Estimated DFG node count.
    dfg_node_count: int = 0

    # Assigned core-type index (filled by DSE).
    assigned_core_type_idx: int = 0


# ---------------------------------------------------------------------------
# Contract visibility levels
# ---------------------------------------------------------------------------

VISIBILITY_LOCAL_SPM = "LOCAL_SPM"
VISIBILITY_SHARED_L2 = "SHARED_L2"
VISIBILITY_EXTERNAL_DRAM = "EXTERNAL_DRAM"

# ---------------------------------------------------------------------------
# Contract ordering modes
# ---------------------------------------------------------------------------

ORDERING_FIFO = "FIFO"
ORDERING_UNORDERED = "UNORDERED"


@dataclass
class ContractEdge:
    """A data-flow contract between two kernels."""

    producer: str = ""
    consumer: str = ""
    production_rate: float = 1.0
    consumption_rate: float = 1.0
    element_size_bytes: int = 4

    # Contract visibility: where the data buffer resides in the memory
    # hierarchy. Affects cache hit rate estimation.
    visibility: str = VISIBILITY_LOCAL_SPM

    # Contract ordering: whether data must be consumed in production order
    # (FIFO) or can be consumed out-of-order (UNORDERED). Affects pipeline
    # stall estimation.
    ordering: str = ORDERING_FIFO


@dataclass
class WorkloadProfile:
    """Simplified workload profile summarizing a TDG for proxy evaluation."""

    kernels: List[KernelProfile] = field(default_factory=list)
    contracts: List[ContractEdge] = field(default_factory=list)

    # Critical path as a list of kernel names (topological order).
    critical_path: List[str] = field(default_factory=list)

    def kernel_by_name(self, name: str) -> Optional[KernelProfile]:
        for k in self.kernels:
            if k.name == name:
                return k
        return None


# ---------------------------------------------------------------------------
# Proxy evaluation result
# ---------------------------------------------------------------------------

@dataclass
class ProxyScore:
    """Result of a Tier-1 proxy evaluation."""

    throughput: float = 0.0           # iterations per cycle (higher is better)
    area_um2: float = 0.0            # estimated total area in um^2
    communication_cost: float = 0.0  # estimated inter-core data transfer cost
    utilization: float = 0.0         # average PE utilization (0-1)
    cache_stall: float = 0.0         # estimated cache miss stall cycles
    pipeline_stall: float = 0.0      # estimated pipeline stall cycles
    feasible: bool = True            # whether mapping is feasible at all

    def composite_score(self, area_weight: float = 0.5) -> float:
        """Combined objective: throughput / area (with weighting).

        Higher is better. Returns 0 for infeasible designs.
        """
        if not self.feasible or self.area_um2 <= 0:
            return 0.0
        # Normalize throughput by area, penalize communication cost
        efficiency = self.throughput / (self.area_um2 * 1e-6)
        comm_penalty = 1.0 / (1.0 + self.communication_cost)
        return efficiency * (1.0 - area_weight) + comm_penalty * area_weight


# ---------------------------------------------------------------------------
# Analytical resource model
# ---------------------------------------------------------------------------

class AnalyticalResourceModel:
    """Tier-1 analytical resource model.

    Estimates throughput, area, and communication cost from architecture
    parameters and kernel profiles alone -- no compilation needed.

    The model combines resource-counting (FU/PE area, operation histograms)
    with contract-informed estimations:
    - Cache stall: uses contract visibility to model memory hierarchy penalty.
    - Pipeline stall: uses contract ordering and rate mismatch to model
      producer/consumer synchronization overhead.
    """

    def evaluate(
        self,
        design: DesignPoint,
        workload: WorkloadProfile,
    ) -> ProxyScore:
        """Evaluate a design point against a workload profile.

        Runs in ~1ms for typical workloads.
        """
        if not design.core_types or not workload.kernels:
            return ProxyScore(feasible=False)

        # Estimate per-kernel initiation intervals.
        kernel_iis: Dict[str, float] = {}
        total_pe_demand = 0.0
        total_pe_supply = 0.0

        for kernel in workload.kernels:
            ct_idx = kernel.assigned_core_type_idx
            if ct_idx >= len(design.core_types):
                return ProxyScore(feasible=False)
            ct = design.core_types[ct_idx]

            compute_ii = self._estimate_compute_ii(kernel, ct)
            memory_ii = self._estimate_memory_ii(kernel, ct)

            if math.isinf(compute_ii) or math.isinf(memory_ii):
                return ProxyScore(feasible=False)

            ii = max(compute_ii, memory_ii)
            kernel_iis[kernel.name] = ii

            # Track utilization
            total_pe_demand += kernel.dfg_node_count
            total_pe_supply += ct.num_pes * ii

        # Cache stall estimation (contract visibility)
        cache_stall = self._estimate_cache_stalls(design, workload)

        # Pipeline stall estimation (contract ordering + rate)
        pipeline_stall = self._estimate_pipeline_stalls(workload, kernel_iis)

        # Effective II includes stall penalties
        effective_iis: Dict[str, float] = {}
        for name, ii in kernel_iis.items():
            effective_iis[name] = ii + cache_stall + pipeline_stall

        # Critical-path throughput (using effective IIs)
        throughput = self._estimate_throughput(
            workload.critical_path, effective_iis
        )

        # Area estimate
        area = self._estimate_area(design)

        # Communication cost
        comm_cost = self._estimate_comm_cost(design, workload)

        # Utilization
        utilization = 0.0
        if total_pe_supply > 0:
            utilization = min(1.0, total_pe_demand / total_pe_supply)

        return ProxyScore(
            throughput=throughput,
            area_um2=area,
            communication_cost=comm_cost,
            utilization=utilization,
            cache_stall=cache_stall,
            pipeline_stall=pipeline_stall,
            feasible=True,
        )

    def evaluate_batch(
        self,
        designs: Sequence[DesignPoint],
        workload: WorkloadProfile,
    ) -> List[ProxyScore]:
        """Evaluate multiple designs against the same workload."""
        return [self.evaluate(d, workload) for d in designs]

    # -------------------------------------------------------------------
    # Internal estimation methods
    # -------------------------------------------------------------------

    def _estimate_compute_ii(
        self, kernel: KernelProfile, ct: CoreTypeConfig
    ) -> float:
        """Estimate initiation interval from operation histogram and FU counts.

        II = max over all op types of ceil(op_count / fu_count).
        """
        fu_counts = ct.fu_mix
        max_ii = 1.0

        for op_type, count in kernel.op_histogram.items():
            fu_key = self._op_to_fu_key(op_type)
            available = fu_counts.get(fu_key, 0)
            if available <= 0:
                if count > 0:
                    return float("inf")
                continue
            ii = math.ceil(count / available)
            max_ii = max(max_ii, ii)

        return max_ii

    def _estimate_memory_ii(
        self, kernel: KernelProfile, ct: CoreTypeConfig
    ) -> float:
        """Estimate memory-bound II from load/store counts and memory FU count."""
        mem_ops = kernel.loads_per_iter + kernel.stores_per_iter
        if mem_ops == 0:
            return 1.0
        mem_fus = ct.fu_mix.get("mem", 0)
        if mem_fus <= 0:
            return float("inf")
        return math.ceil(mem_ops / mem_fus)

    def _estimate_cache_stalls(
        self,
        design: DesignPoint,
        workload: WorkloadProfile,
    ) -> float:
        """Estimate cache miss stall cycles using contract visibility.

        Each contract's visibility field indicates where the data buffer
        resides in the memory hierarchy:
        - LOCAL_SPM: data in scratchpad, 100% hit rate, no stall.
        - SHARED_L2: data in shared L2 cache, configurable miss rate.
        - EXTERNAL_DRAM: data in off-chip DRAM, high miss rate.

        The stall is proportional to the miss rate times the number of
        memory accesses on each contract, weighted by the miss penalty
        (which depends on where in the hierarchy the miss occurs).
        """
        if not workload.contracts:
            return 0.0

        miss_rates = CACHE_MISS_RATES
        total_stall = 0.0

        for contract in workload.contracts:
            vis = contract.visibility

            miss_rate = miss_rates.get(vis, 0.0)
            if miss_rate <= 0.0:
                continue

            # Penalty in cycles per miss depends on hierarchy level
            if vis == VISIBILITY_SHARED_L2:
                miss_penalty = 10.0   # L2 miss -> DRAM fetch
            elif vis == VISIBILITY_EXTERNAL_DRAM:
                miss_penalty = 100.0  # DRAM access latency
            else:
                miss_penalty = 0.0

            # Volume of data accessed on this contract per iteration
            accesses = contract.production_rate
            stall = accesses * miss_rate * miss_penalty
            total_stall += stall

        return total_stall

    def _estimate_pipeline_stalls(
        self,
        workload: WorkloadProfile,
        kernel_iis: Dict[str, float],
    ) -> float:
        """Estimate pipeline stalls from contract ordering and rate mismatch.

        When a producer and consumer operate at different rates, the faster
        one must stall waiting for the slower one. The ordering mode affects
        how severe this synchronization overhead is:
        - FIFO: strict ordering requires the consumer to wait for each
          element in sequence. Rate mismatch causes bubble cycles.
        - UNORDERED: out-of-order consumption allows better overlap,
          reducing the effective stall penalty.
        """
        if not workload.contracts or not kernel_iis:
            return 0.0

        total_stall = 0.0

        for contract in workload.contracts:
            prod_ii = kernel_iis.get(contract.producer, 1.0)
            cons_ii = kernel_iis.get(contract.consumer, 1.0)

            # Rate mismatch: ratio of the slower to faster kernel II
            if prod_ii <= 0 or cons_ii <= 0:
                continue

            rate_ratio = max(prod_ii, cons_ii) / min(prod_ii, cons_ii)
            if rate_ratio <= 1.0:
                continue

            # Mismatch penalty: proportional to how far from 1:1 the rates are
            mismatch_penalty = rate_ratio - 1.0

            # Ordering mode scales the penalty
            if contract.ordering == ORDERING_FIFO:
                # FIFO: full penalty from rate mismatch (blocking synchronous)
                ordering_factor = 1.0
            elif contract.ordering == ORDERING_UNORDERED:
                # UNORDERED: reduced penalty (can overlap partial consumption)
                ordering_factor = 0.3
            else:
                ordering_factor = 1.0

            stall = mismatch_penalty * ordering_factor
            total_stall += stall

        return total_stall

    def _estimate_throughput(
        self,
        critical_path: List[str],
        kernel_iis: Dict[str, float],
    ) -> float:
        """Estimate throughput from critical-path kernel IIs.

        Throughput = 1 / (sum of IIs on critical path).
        If critical path is empty, use the maximum II across all kernels.
        """
        if critical_path:
            total_ii = sum(
                kernel_iis.get(k, 1.0) for k in critical_path
            )
        elif kernel_iis:
            total_ii = max(kernel_iis.values())
        else:
            return 0.0

        if total_ii <= 0:
            return 0.0
        return 1.0 / total_ii

    def _estimate_area(self, design: DesignPoint) -> float:
        """Analytical area model based on component counts."""
        area = 0.0

        for ct in design.core_types:
            per_core = (
                ct.num_pes * PE_AREA_UM2
                + sum(
                    FU_AREA_TABLE.get(ft, ALU_FU_AREA_UM2) * count
                    for ft, count in ct.fu_mix.items()
                )
                + ct.spm_bytes * SRAM_AREA_PER_BYTE_UM2
                + ct.num_switches * SW_AREA_UM2
            )
            area += per_core * ct.instance_count

        # NoC area (rough model: proportional to total cores and bandwidth)
        total_cores = design.total_cores()
        noc_area = self._estimate_noc_area(
            total_cores, design.noc_topology, design.noc_bandwidth
        )
        area += noc_area

        # L2 cache
        area += design.l2_size_kb * 1024 * SRAM_AREA_PER_BYTE_UM2

        return area

    def _estimate_noc_area(
        self, total_cores: int, topology: str, bandwidth: int
    ) -> float:
        """Rough NoC area model."""
        base_link_area = 2000.0  # per link
        if topology == "mesh":
            # Mesh: ~2*(rows+cols) links per core
            side = max(1, math.isqrt(total_cores))
            links = 2 * total_cores * 2
        elif topology == "ring":
            links = total_cores * 2
        else:
            # Hierarchical: fewer top-level links
            links = total_cores * 3
        return links * base_link_area * bandwidth

    def _estimate_comm_cost(
        self,
        design: DesignPoint,
        workload: WorkloadProfile,
    ) -> float:
        """Estimate inter-core communication cost from contracts."""
        cost = 0.0
        noc_bw = max(1, design.noc_bandwidth)

        for contract in workload.contracts:
            prod_kernel = workload.kernel_by_name(contract.producer)
            cons_kernel = workload.kernel_by_name(contract.consumer)
            if prod_kernel is None or cons_kernel is None:
                continue

            # If on different core types, communication crosses the NoC
            if (
                prod_kernel.assigned_core_type_idx
                != cons_kernel.assigned_core_type_idx
            ):
                hops = self._estimate_hops(design, contract)
                volume = contract.production_rate * contract.element_size_bytes
                cost += hops * volume / noc_bw

        return cost

    def _estimate_hops(
        self, design: DesignPoint, contract: ContractEdge
    ) -> float:
        """Estimate NoC hop count for an inter-core transfer."""
        total_cores = design.total_cores()
        if design.noc_topology == "mesh":
            side = max(1, math.isqrt(total_cores))
            return float(side)  # average Manhattan distance
        elif design.noc_topology == "ring":
            return total_cores / 4.0  # average ring distance
        else:
            return 2.0  # hierarchical: 2-level hop

    @staticmethod
    def _op_to_fu_key(op_type: str) -> str:
        """Map operation type names to FU category keys."""
        op_lower = op_type.lower()
        if any(k in op_lower for k in ("mul", "div", "rem")):
            return "mul"
        if any(k in op_lower for k in ("fp", "float", "fadd", "fmul")):
            return "fp"
        if any(k in op_lower for k in ("load", "store", "mem")):
            return "mem"
        return "alu"


# Needed in _estimate_area fallback
ALU_FU_AREA_UM2 = FU_AREA_TABLE["alu"]
