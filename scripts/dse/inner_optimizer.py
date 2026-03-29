"""Inner DSE per-core optimizer.

Optimizes the concrete microarchitectural parameters (free dimensions) for
a single core type against its assigned kernel set. Uses a three-tier
evaluation pipeline:

  Tier A: Analytical proxy scoring (FU coverage, resource-bound II, area)
  Tier B: Compilation-based evaluation (tapestry_compile)
  Tier C: Simulation (deferred, returns Tier B results)

Provides JSON I/O for C++ interop and caching across outer DSE iterations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dse_config import (
    FU_AREA_TABLE,
    PE_AREA_UM2,
    SRAM_AREA_PER_BYTE_UM2,
    SW_AREA_UM2,
    TIER2_TIMEOUT_SEC,
)
from .proxy_model import KernelProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Design parameter types mirroring the C++ CoreDesignParams
# ---------------------------------------------------------------------------

class PEType(Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


class RoutingTopology(Enum):
    CHESS = "chess"
    MESH = "mesh"
    LATTICE = "lattice"
    RING = "ring"


class ComputeMix(Enum):
    FP_HEAVY = "fp_heavy"
    INT_HEAVY = "int_heavy"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# CoreDesignParams (Python mirror of C++ struct)
# ---------------------------------------------------------------------------

@dataclass
class CoreDesignParams:
    """Python mirror of C++ CoreDesignParams with all 13 dimensions."""

    pe_type: str = "spatial"
    array_rows: int = 2
    array_cols: int = 2
    data_width: int = 32
    fu_repertoire: List[str] = field(default_factory=list)
    multi_op_fu_bodies: bool = False
    switch_type: str = "spatial"
    decomposable_bits: int = -1
    spm_size_kb: int = 4
    spm_ld_ports: int = 1
    spm_st_ports: int = 1
    extmem_count: int = 2
    extmem_ld_ports: int = 1
    extmem_st_ports: int = 1
    topology: str = "chess"
    instruction_slots: int = 4
    num_registers: int = 4
    reg_fifo_depth: int = 0
    share_operand_buffer: bool = False
    operand_buffer_size: int = 0
    scalar_inputs: int = 3
    scalar_outputs: int = 1

    def total_pes(self) -> int:
        return self.array_rows * self.array_cols

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CoreDesignParams":
        return CoreDesignParams(**{
            k: v for k, v in d.items()
            if k in CoreDesignParams.__dataclass_fields__
        })


# ---------------------------------------------------------------------------
# FreedomMask
# ---------------------------------------------------------------------------

@dataclass
class FreedomMask:
    """Boolean mask over 13 dimensions: True = free for optimization."""

    pe_type: bool = False
    array_dims: bool = False
    data_width: bool = False
    fu_repertoire: bool = False
    fu_body_structure: bool = False
    switch_type: bool = False
    decomposability: bool = False
    spm: bool = False
    ext_mem: bool = False
    topology: bool = False
    temporal_params: bool = False
    scalar_io: bool = False
    connectivity: bool = False

    def count_free(self) -> int:
        return sum(1 for v in asdict(self).values() if v)

    @staticmethod
    def domain_specific() -> "FreedomMask":
        """Mask for domain-specific types (D1-D6)."""
        return FreedomMask(
            topology=True,
            connectivity=True,
            fu_repertoire=True,
        )

    @staticmethod
    def combinatorial(is_temporal: bool = False) -> "FreedomMask":
        """Mask for combinatorial types."""
        return FreedomMask(
            data_width=True,
            fu_repertoire=True,
            decomposability=True,
            ext_mem=True,
            topology=True,
            temporal_params=is_temporal,
            scalar_io=True,
            connectivity=True,
        )


# ---------------------------------------------------------------------------
# Tier-A Analytical Scoring
# ---------------------------------------------------------------------------

@dataclass
class TierAKernelII:
    """Per-kernel resource-bound II from Tier-A analysis."""
    kernel_name: str = ""
    fu_bound: float = 1.0
    mem_bound: float = 1.0
    routing_bound: float = 1.0
    effective_ii: float = 1.0


@dataclass
class TierAScore:
    """Full result of Tier-A analytical scoring."""
    feasible: bool = False
    area_estimate: float = 0.0
    per_kernel_ii: Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0


# Routing overhead factors by topology
_ROUTING_OVERHEAD = {
    "chess": 1.0,
    "mesh": 1.0,
    "lattice": 1.2,
    "ring": 1.5,
}


def _op_to_fu_category(op_name: str) -> str:
    """Map operation name to FU category key."""
    op = op_name.lower()

    fp_keys = (
        "addf", "subf", "mulf", "divf", "negf", "cmpf",
        "sqrt", "exp", "log", "sin", "cos", "fma", "absf",
        "sitofp", "uitofp", "fptosi", "fptoui",
    )
    if any(k in op for k in fp_keys):
        return "fp"

    if any(k in op for k in ("load", "store")):
        return "mem"

    if any(k in op for k in ("muli", "divsi", "divui", "remsi", "remui")):
        return "mul"

    return "alu"


def _canonicalize_op(op: str) -> str:
    """Canonicalize short op names to fully-qualified MLIR names."""
    if "." in op:
        return op
    _CANON_MAP = {
        "addi": "arith.addi", "subi": "arith.subi",
        "muli": "arith.muli", "addf": "arith.addf",
        "subf": "arith.subf", "mulf": "arith.mulf",
        "divf": "arith.divf", "cmpi": "arith.cmpi",
        "cmpf": "arith.cmpf", "select": "arith.select",
        "andi": "arith.andi", "ori": "arith.ori",
        "xori": "arith.xori", "shli": "arith.shli",
        "shrsi": "arith.shrsi", "shrui": "arith.shrui",
        "fadd": "arith.addf", "fmul": "arith.mulf",
        "load": "handshake.load", "store": "handshake.store",
    }
    return _CANON_MAP.get(op, op)


def _estimate_area(params: CoreDesignParams) -> float:
    """Analytical area model matching C++ estimateCoreArea."""
    pe_count = params.total_pes()

    # FU area per PE
    fu_area_per_pe = 0.0
    for op in params.fu_repertoire:
        cat = _op_to_fu_category(op)
        fu_area_per_pe += FU_AREA_TABLE.get(cat, FU_AREA_TABLE["alu"])

    pe_overhead = PE_AREA_UM2
    if params.pe_type == "temporal":
        pe_overhead += 3.0 * params.instruction_slots
        pe_overhead += 2.0 * params.num_registers

    total_pe_area = pe_count * (fu_area_per_pe + pe_overhead)

    # Switch area
    num_switches = pe_count
    if params.topology == "chess":
        num_switches = (params.array_rows + 1) * (params.array_cols + 1)
    sw_area = num_switches * SW_AREA_UM2

    # SPM area
    spm_area = params.spm_size_kb * 1024 * SRAM_AREA_PER_BYTE_UM2

    return total_pe_area + sw_area + spm_area


def score_core_design(
    params: CoreDesignParams,
    profiles: List[KernelProfile],
) -> TierAScore:
    """Score a CoreDesignParams against assigned kernel profiles.

    Computes FU coverage, resource-bound II per kernel, area, and a
    composite score (geomean(1/II) / area).
    """
    result = TierAScore()
    result.feasible = True

    if not profiles:
        result.feasible = False
        return result

    # Count FUs per category from repertoire, scaled by PE count
    fu_counts: Dict[str, int] = {}
    for op in params.fu_repertoire:
        cat = _op_to_fu_category(op)
        fu_counts[cat] = fu_counts.get(cat, 0) + 1

    pe_count = params.total_pes()
    fu_counts = {cat: count * pe_count for cat, count in fu_counts.items()}

    geo_product = 1.0
    geo_count = 0

    for profile in profiles:
        # Check FU coverage
        fu_repertoire_set = set(params.fu_repertoire)
        ops_per_category: Dict[str, int] = {}

        for op_name, count in profile.op_histogram.items():
            canon = _canonicalize_op(op_name)
            if canon not in fu_repertoire_set:
                result.feasible = False
                result.composite_score = 0.0
                return result
            cat = _op_to_fu_category(canon)
            ops_per_category[cat] = ops_per_category.get(cat, 0) + count

        # FU-bound II
        fu_bound = 1.0
        for cat, ops in ops_per_category.items():
            available = fu_counts.get(cat, 0)
            if available == 0 and ops > 0:
                result.feasible = False
                result.composite_score = 0.0
                return result
            if available > 0:
                bound = math.ceil(ops / available)
                fu_bound = max(fu_bound, bound)

        # Memory-bound II
        load_ops = sum(
            count for op_name, count in profile.op_histogram.items()
            if "load" in op_name.lower()
        )
        mem_bound = 1.0
        if load_ops > 0 and params.spm_ld_ports > 0:
            mem_bound = math.ceil(load_ops / params.spm_ld_ports)

        # Routing bound
        routing_bound = _ROUTING_OVERHEAD.get(params.topology, 1.0)

        # Effective II
        effective_ii = max(fu_bound, mem_bound, routing_bound)

        result.per_kernel_ii[profile.name] = effective_ii

        if effective_ii > 0:
            geo_product *= (1.0 / effective_ii)
            geo_count += 1

    # Area estimate
    result.area_estimate = _estimate_area(params)

    # Composite score
    if geo_count > 0 and result.area_estimate > 0:
        geo_mean = geo_product ** (1.0 / geo_count)
        result.composite_score = geo_mean / result.area_estimate
    else:
        result.composite_score = 0.0

    return result


# ---------------------------------------------------------------------------
# Preset Constructors
# ---------------------------------------------------------------------------

def create_domain_preset(domain_index: int) -> CoreDesignParams:
    """Create a domain-specific preset for D1 through D6."""
    presets = {
        1: CoreDesignParams(  # LLM
            pe_type="spatial", array_rows=12, array_cols=12,
            data_width=32, spm_size_kb=64, spm_ld_ports=2, spm_st_ports=2,
            topology="mesh", scalar_inputs=4, scalar_outputs=2,
            extmem_count=4, extmem_ld_ports=2, extmem_st_ports=1,
            fu_repertoire=[
                "arith.addf", "arith.mulf", "arith.addi", "arith.muli",
                "arith.cmpi", "arith.select", "handshake.load",
                "handshake.store", "math.exp", "math.sqrt",
            ],
        ),
        2: CoreDesignParams(  # CV
            pe_type="spatial", array_rows=8, array_cols=8,
            data_width=32, spm_size_kb=32, spm_ld_ports=2, spm_st_ports=2,
            topology="chess",
            fu_repertoire=[
                "arith.addf", "arith.mulf", "arith.addi", "arith.muli",
                "arith.cmpi", "arith.select", "arith.shli", "arith.shrsi",
                "handshake.load", "handshake.store",
            ],
        ),
        3: CoreDesignParams(  # Signal
            pe_type="temporal", switch_type="temporal",
            array_rows=6, array_cols=6,
            data_width=32, spm_size_kb=16, spm_ld_ports=2, spm_st_ports=1,
            topology="chess", instruction_slots=8, num_registers=8,
            reg_fifo_depth=2,
            fu_repertoire=[
                "arith.addi", "arith.muli", "arith.addf", "arith.mulf",
                "arith.cmpi", "arith.select", "handshake.load",
                "handshake.store", "math.fma",
            ],
        ),
        4: CoreDesignParams(  # Crypto
            pe_type="spatial", array_rows=4, array_cols=4,
            data_width=64, spm_size_kb=8,
            fu_repertoire=[
                "arith.addi", "arith.muli", "arith.andi", "arith.ori",
                "arith.xori", "arith.shli", "arith.shrsi", "arith.shrui",
                "arith.cmpi", "arith.select", "handshake.load",
                "handshake.store",
            ],
        ),
        5: CoreDesignParams(  # Sensor
            pe_type="temporal", switch_type="temporal",
            array_rows=4, array_cols=4,
            data_width=32, spm_size_kb=8,
            instruction_slots=16, num_registers=8, reg_fifo_depth=4,
            fu_repertoire=[
                "arith.addi", "arith.muli", "arith.cmpi", "arith.select",
                "arith.addf", "arith.cmpf", "handshake.load",
                "handshake.store", "handshake.cond_br",
            ],
        ),
        6: CoreDesignParams(  # Control
            pe_type="spatial", array_rows=4, array_cols=4,
            data_width=32, spm_size_kb=4,
            fu_repertoire=[
                "arith.addi", "arith.muli", "arith.cmpi", "arith.select",
                "arith.andi", "arith.ori", "handshake.load",
                "handshake.store", "handshake.cond_br", "handshake.mux",
            ],
        ),
    }
    return presets.get(domain_index, presets[6])


def create_combinatorial_preset(
    mix: ComputeMix,
    pe: PEType,
    has_spm: bool,
    array_size: int,
) -> CoreDesignParams:
    """Create a combinatorial preset from the four axes."""
    params = CoreDesignParams()
    params.pe_type = pe.value
    params.array_rows = array_size
    params.array_cols = array_size

    # Switch type matches PE type
    params.switch_type = pe.value

    if mix == ComputeMix.FP_HEAVY:
        params.fu_repertoire = [
            "arith.addf", "arith.mulf", "arith.subf", "arith.divf",
            "arith.cmpf", "arith.addi", "arith.muli",
            "arith.cmpi", "arith.select",
            "handshake.load", "handshake.store",
        ]
    elif mix == ComputeMix.INT_HEAVY:
        params.fu_repertoire = [
            "arith.addi", "arith.subi", "arith.muli",
            "arith.andi", "arith.ori", "arith.xori",
            "arith.shli", "arith.shrsi",
            "arith.cmpi", "arith.select",
            "handshake.load", "handshake.store",
        ]
    else:  # MIXED
        params.fu_repertoire = [
            "arith.addi", "arith.muli", "arith.addf", "arith.mulf",
            "arith.cmpi", "arith.select",
            "handshake.load", "handshake.store",
        ]

    if has_spm:
        params.spm_size_kb = 16
        params.spm_ld_ports = 2
        params.spm_st_ports = 2
    else:
        params.spm_size_kb = 0
        params.spm_ld_ports = 0
        params.spm_st_ports = 0

    if pe == PEType.TEMPORAL:
        params.instruction_slots = 8
        params.num_registers = 8

    return params


def generate_type_id(
    mix: ComputeMix,
    pe: PEType,
    has_spm: bool,
    array_size: int,
) -> str:
    """Generate the type ID string for a combinatorial type."""
    mix_char = {"fp_heavy": "F", "int_heavy": "I", "mixed": "M"}[mix.value]
    pe_char = "S" if pe == PEType.SPATIAL else "T"
    spm_char = "Y" if has_spm else "N"
    return f"C{mix_char}{pe_char}{spm_char}{array_size}"


# ---------------------------------------------------------------------------
# Parameter Sweep Generation
# ---------------------------------------------------------------------------

def generate_sweep_candidates(
    baseline: CoreDesignParams,
    mask: FreedomMask,
    max_candidates: int = 30,
    seed: int = 42,
) -> List[CoreDesignParams]:
    """Generate candidates via constrained Latin Hypercube Sampling.

    Fixed dimensions retain baseline values; free dimensions are sampled
    from discrete value sets.
    """
    import random as rnd
    rng = rnd.Random(seed)

    # Define discrete value sets for each free dimension
    topo_values = ["chess", "mesh", "lattice", "ring"]
    data_width_values = [32, 64]
    decomp_values = [-1, 8, 16]
    extmem_configs = [(1, 1, 1), (2, 1, 1), (2, 2, 1)]
    scalar_io_configs = [(2, 1), (3, 1), (4, 2)]
    temporal_configs = [(4, 4, 0), (8, 8, 2), (16, 8, 4)]

    # Collect free dimension choice counts
    free_dims: List[List[Any]] = []
    dim_names: List[str] = []

    if mask.topology:
        free_dims.append(topo_values)
        dim_names.append("topology")
    if mask.data_width:
        free_dims.append(data_width_values)
        dim_names.append("data_width")
    if mask.decomposability:
        free_dims.append(decomp_values)
        dim_names.append("decomposability")
    if mask.ext_mem:
        free_dims.append(extmem_configs)
        dim_names.append("ext_mem")
    if mask.scalar_io:
        free_dims.append(scalar_io_configs)
        dim_names.append("scalar_io")
    if mask.temporal_params and baseline.pe_type == "temporal":
        free_dims.append(temporal_configs)
        dim_names.append("temporal_params")
    if mask.fu_repertoire:
        free_dims.append(["base", "extended", "pruned"])
        dim_names.append("fu_repertoire")

    if not free_dims:
        return [baseline]

    # Compute total combinations
    total_combos = 1
    for dim_vals in free_dims:
        total_combos *= len(dim_vals)
        if total_combos > max_candidates:
            break

    n = min(max_candidates, total_combos)

    # Latin Hypercube Sampling: for each dimension, create a shuffled
    # index permutation
    dim_samples: List[List[int]] = []
    for dim_vals in free_dims:
        nc = len(dim_vals)
        indices = [i % nc for i in range(n)]
        rng.shuffle(indices)
        dim_samples.append(indices)

    candidates: List[CoreDesignParams] = []
    for i in range(n):
        cand = CoreDesignParams.from_dict(baseline.to_dict())
        dim_idx = 0

        if mask.topology:
            val_idx = dim_samples[dim_idx][i]
            cand.topology = topo_values[val_idx]
            dim_idx += 1
        if mask.data_width:
            val_idx = dim_samples[dim_idx][i]
            cand.data_width = data_width_values[val_idx]
            dim_idx += 1
        if mask.decomposability:
            val_idx = dim_samples[dim_idx][i]
            cand.decomposable_bits = decomp_values[val_idx]
            dim_idx += 1
        if mask.ext_mem:
            val_idx = dim_samples[dim_idx][i]
            cfg = extmem_configs[val_idx]
            cand.extmem_count = cfg[0]
            cand.extmem_ld_ports = cfg[1]
            cand.extmem_st_ports = cfg[2]
            dim_idx += 1
        if mask.scalar_io:
            val_idx = dim_samples[dim_idx][i]
            cfg = scalar_io_configs[val_idx]
            cand.scalar_inputs = cfg[0]
            cand.scalar_outputs = cfg[1]
            dim_idx += 1
        if mask.temporal_params and baseline.pe_type == "temporal":
            val_idx = dim_samples[dim_idx][i]
            cfg = temporal_configs[val_idx]
            cand.instruction_slots = cfg[0]
            cand.num_registers = cfg[1]
            cand.reg_fifo_depth = cfg[2]
            dim_idx += 1
        if mask.fu_repertoire:
            val_idx = dim_samples[dim_idx][i]
            choice = free_dims[dim_idx][val_idx]
            if choice == "extended":
                extras = ["arith.cmpi", "arith.select", "arith.addi"]
                for e in extras:
                    if e not in cand.fu_repertoire:
                        cand.fu_repertoire.append(e)
            elif choice == "pruned":
                essential = {
                    "arith.addi", "arith.muli",
                    "handshake.load", "handshake.store",
                }
                pruned = [op for op in cand.fu_repertoire if op in essential]
                if len(pruned) >= 2:
                    cand.fu_repertoire = pruned
            dim_idx += 1

        candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------------
# Tier-B Compilation Evaluation
# ---------------------------------------------------------------------------

@dataclass
class KernelMappingResult:
    """Result of mapping a single kernel to a candidate ADG."""
    kernel_name: str = ""
    success: bool = False
    achieved_ii: int = 0
    mapping_time_sec: float = 0.0


@dataclass
class TierBResult:
    """Result of Tier-B compilation evaluation."""
    success: bool = False
    mapping_results: List[KernelMappingResult] = field(default_factory=list)
    mapping_success_rate: float = 0.0
    mean_achieved_ii: float = 0.0
    area: float = 0.0
    diagnostics: str = ""


def evaluate_candidate_tier_b(
    params: CoreDesignParams,
    assigned_kernels: List[KernelProfile],
    compile_fn=None,
    timeout_sec: float = TIER2_TIMEOUT_SEC,
) -> TierBResult:
    """Evaluate a candidate via compilation (Tier B).

    If compile_fn is provided, it is called for each kernel as:
        compile_fn(params, kernel) -> KernelMappingResult

    Otherwise, falls back to resource feasibility check as a proxy.
    The proxy checks FU coverage and estimates II from op count / PE count,
    which is a reasonable lower bound but does not capture routing
    constraints or real mapping failures.

    For real compilation, compile_fn should invoke the tapestry_compile
    binary (or tapestry-pipeline with --enable-sim=false). This requires:
      1. A concrete ADG MLIR file for the candidate (generated via the C++
         ADGBuilder API or HWInnerADGGen from CoreDesignParams)
      2. A TDG MLIR file for each kernel
      3. The tapestry-pipeline binary on PATH
    The compile_fn is responsible for generating the ADG, writing temp
    files, invoking the subprocess, and parsing the output into a
    KernelMappingResult with success/achieved_ii fields.
    """
    result = TierBResult()
    result.area = _estimate_area(params)

    if not assigned_kernels:
        result.success = True
        result.mapping_success_rate = 1.0
        result.mean_achieved_ii = 1.0
        return result

    success_count = 0
    ii_product = 1.0
    ii_count = 0
    failed_kernels: List[str] = []

    for kernel in assigned_kernels:
        if compile_fn is not None:
            mr = compile_fn(params, kernel)
        else:
            # Fallback: use Tier-A feasibility as proxy
            mr = _resource_feasibility_check(params, kernel)

        result.mapping_results.append(mr)

        if mr.success:
            success_count += 1
            if mr.achieved_ii > 0:
                ii_product *= mr.achieved_ii
                ii_count += 1
        else:
            failed_kernels.append(mr.kernel_name)

    total = len(assigned_kernels)
    result.mapping_success_rate = success_count / total if total > 0 else 0.0
    result.success = (success_count == total)

    if ii_count > 0:
        result.mean_achieved_ii = ii_product ** (1.0 / ii_count)
    else:
        result.mean_achieved_ii = 0.0

    if failed_kernels:
        result.diagnostics = (
            f"Failed kernels: {', '.join(failed_kernels)}"
        )

    return result


def _resource_feasibility_check(
    params: CoreDesignParams,
    kernel: KernelProfile,
) -> KernelMappingResult:
    """Resource feasibility check as Tier-B proxy."""
    mr = KernelMappingResult(kernel_name=kernel.name)

    fu_set = set(params.fu_repertoire)
    for op_name in kernel.op_histogram:
        canon = _canonicalize_op(op_name)
        if canon not in fu_set:
            mr.success = False
            return mr

    mr.success = True
    # Estimate II from op histogram
    pe_count = params.total_pes()
    total_ops = sum(kernel.op_histogram.values())
    mr.achieved_ii = max(1, math.ceil(total_ops / max(1, pe_count)))
    return mr


# ---------------------------------------------------------------------------
# Inner DSE Result
# ---------------------------------------------------------------------------

@dataclass
class InnerResult:
    """Complete result of inner DSE for one core type."""

    type_id: str = ""
    success: bool = False
    best_params: Optional[CoreDesignParams] = None
    adg_mlir_path: str = ""
    tier_a_score: float = 0.0
    tier_b_score: float = 0.0
    tier_c_score: float = 0.0
    mapping_results: List[KernelMappingResult] = field(default_factory=list)
    area_estimate: float = 0.0
    tier_a_evaluations: int = 0
    tier_b_evaluations: int = 0
    wall_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type_id": self.type_id,
            "success": self.success,
            "best_params": self.best_params.to_dict() if self.best_params else None,
            "adg_mlir_path": self.adg_mlir_path,
            "tier_a_score": self.tier_a_score,
            "tier_b_score": self.tier_b_score,
            "tier_c_score": self.tier_c_score,
            "mapping_results": [
                {
                    "kernel_name": mr.kernel_name,
                    "success": mr.success,
                    "achieved_ii": mr.achieved_ii,
                    "mapping_time_sec": mr.mapping_time_sec,
                }
                for mr in self.mapping_results
            ],
            "area_estimate": self.area_estimate,
            "tier_a_evaluations": self.tier_a_evaluations,
            "tier_b_evaluations": self.tier_b_evaluations,
            "wall_time_sec": self.wall_time_sec,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(text: str) -> "InnerResult":
        d = json.loads(text)
        return InnerResult.from_dict(d)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InnerResult":
        result = InnerResult()
        result.type_id = d.get("type_id", "")
        result.success = d.get("success", False)
        if d.get("best_params"):
            result.best_params = CoreDesignParams.from_dict(d["best_params"])
        result.adg_mlir_path = d.get("adg_mlir_path", "")
        result.tier_a_score = d.get("tier_a_score", 0.0)
        result.tier_b_score = d.get("tier_b_score", 0.0)
        result.tier_c_score = d.get("tier_c_score", 0.0)
        result.area_estimate = d.get("area_estimate", 0.0)
        result.tier_a_evaluations = d.get("tier_a_evaluations", 0)
        result.tier_b_evaluations = d.get("tier_b_evaluations", 0)
        result.wall_time_sec = d.get("wall_time_sec", 0.0)
        for mr_dict in d.get("mapping_results", []):
            result.mapping_results.append(KernelMappingResult(
                kernel_name=mr_dict.get("kernel_name", ""),
                success=mr_dict.get("success", False),
                achieved_ii=mr_dict.get("achieved_ii", 0),
                mapping_time_sec=mr_dict.get("mapping_time_sec", 0.0),
            ))
        return result


# ---------------------------------------------------------------------------
# Inner DSE Driver
# ---------------------------------------------------------------------------

@dataclass
class InnerDSEConfig:
    """Configuration for the inner DSE driver."""
    max_inner_iter: int = 30
    top_k: int = 10
    seed: int = 42
    tier_b_enabled: bool = True
    tier_c_enabled: bool = False
    mapper_timeout_sec: float = 10.0
    output_dir: str = ""
    verbose: bool = False


class InnerDSEDriver:
    """Inner DSE optimization driver for one core type.

    Runs the three-tier evaluation pipeline:
      1. Generate sweep candidates from free dimensions
      2. Tier-A: score all candidates analytically, keep top-K
      3. Tier-B: compile representative kernels for top-K, re-rank
      4. Optionally run Tier-C on winner
      5. Output best params, ADG MLIR, and result JSON
    """

    def __init__(
        self,
        type_id: str,
        baseline: CoreDesignParams,
        freedom_mask: FreedomMask,
        assigned_kernels: List[KernelProfile],
        config: Optional[InnerDSEConfig] = None,
        compile_fn=None,
    ):
        self.type_id = type_id
        self.baseline = baseline
        self.mask = freedom_mask
        self.kernels = assigned_kernels
        self.config = config or InnerDSEConfig()
        self.compile_fn = compile_fn

    def optimize(self) -> InnerResult:
        """Run the full inner DSE optimization loop."""
        wall_start = time.time()
        result = InnerResult(type_id=self.type_id)

        # Generate sweep candidates
        candidates = generate_sweep_candidates(
            self.baseline,
            self.mask,
            max_candidates=self.config.max_inner_iter,
            seed=self.config.seed,
        )
        # Include the baseline
        all_candidates = [self.baseline] + candidates

        # Tier-A: score all candidates analytically
        scored: List[Tuple[CoreDesignParams, TierAScore]] = []
        for cand in all_candidates:
            ta = score_core_design(cand, self.kernels)
            scored.append((cand, ta))
            result.tier_a_evaluations += 1

        # Filter feasible and sort by composite score
        feasible = [(p, s) for p, s in scored if s.feasible]
        feasible.sort(key=lambda x: x[1].composite_score, reverse=True)

        if not feasible:
            result.success = False
            result.wall_time_sec = time.time() - wall_start
            return result

        # Keep top-K
        top_k = feasible[: self.config.top_k]
        result.tier_a_score = top_k[0][1].composite_score

        # Tier-B: compile evaluation for top-K
        best_params = top_k[0][0]
        best_tier_b: Optional[TierBResult] = None

        if self.config.tier_b_enabled:
            for cand_params, ta_score in top_k:
                tier_b = evaluate_candidate_tier_b(
                    cand_params,
                    self.kernels,
                    compile_fn=self.compile_fn,
                    timeout_sec=self.config.mapper_timeout_sec,
                )
                result.tier_b_evaluations += 1

                if best_tier_b is None or (
                    tier_b.mapping_success_rate > best_tier_b.mapping_success_rate
                    or (
                        tier_b.mapping_success_rate == best_tier_b.mapping_success_rate
                        and tier_b.area < best_tier_b.area
                    )
                ):
                    best_tier_b = tier_b
                    best_params = cand_params

            if best_tier_b:
                result.tier_b_score = best_tier_b.mapping_success_rate
                result.mapping_results = best_tier_b.mapping_results
        else:
            result.tier_b_score = 0.0

        result.success = True
        result.best_params = best_params
        result.area_estimate = _estimate_area(best_params)

        # Write ADG MLIR if output_dir is specified
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)
            adg_path = os.path.join(
                self.config.output_dir, f"{self.type_id}_adg.mlir"
            )
            result.adg_mlir_path = adg_path
            # The actual MLIR generation is done by the C++ side

            # Write result JSON
            json_path = os.path.join(
                self.config.output_dir, f"{self.type_id}_result.json"
            )
            with open(json_path, "w") as f:
                f.write(result.to_json())

        result.wall_time_sec = time.time() - wall_start
        return result


# ---------------------------------------------------------------------------
# Batch optimization with caching
# ---------------------------------------------------------------------------

# Module-level cache for inner DSE results
_inner_dse_cache: Dict[str, InnerResult] = {}


def _cache_key(type_id: str, kernel_names: List[str]) -> str:
    """Compute a cache key from type ID and sorted kernel names."""
    sorted_names = sorted(kernel_names)
    raw = f"{type_id}:{','.join(sorted_names)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def optimize_single_type(
    type_id: str,
    baseline: CoreDesignParams,
    mask: FreedomMask,
    kernels: List[KernelProfile],
    config: Optional[InnerDSEConfig] = None,
    compile_fn=None,
    use_cache: bool = True,
) -> InnerResult:
    """Optimize a single core type with caching.

    Returns cached result if available for the same (type_id, kernel set).
    """
    kernel_names = [k.name for k in kernels]
    key = _cache_key(type_id, kernel_names)

    if use_cache and key in _inner_dse_cache:
        logger.debug("Cache hit for type %s (key=%s)", type_id, key)
        return _inner_dse_cache[key]

    driver = InnerDSEDriver(
        type_id=type_id,
        baseline=baseline,
        freedom_mask=mask,
        assigned_kernels=kernels,
        config=config,
        compile_fn=compile_fn,
    )
    result = driver.optimize()

    if use_cache:
        _inner_dse_cache[key] = result

    return result


def clear_cache() -> None:
    """Clear the inner DSE result cache."""
    _inner_dse_cache.clear()


def optimize_all_types(
    type_configs: List[Tuple[str, CoreDesignParams, FreedomMask, List[KernelProfile]]],
    config: Optional[InnerDSEConfig] = None,
    compile_fn=None,
    parallel: bool = False,
    max_workers: int = 4,
) -> Dict[str, InnerResult]:
    """Optimize multiple core types.

    Args:
        type_configs: List of (type_id, baseline, mask, kernels) tuples.
        config: Shared configuration.
        compile_fn: Optional compilation function.
        parallel: If True, use ProcessPoolExecutor.
        max_workers: Number of parallel workers.

    Returns:
        Dict mapping type_id to InnerResult.
    """
    results: Dict[str, InnerResult] = {}

    if parallel and len(type_configs) > 1:
        # Parallel execution (compile_fn must be picklable)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for type_id, baseline, mask, kernels in type_configs:
                future = pool.submit(
                    optimize_single_type,
                    type_id, baseline, mask, kernels,
                    config, compile_fn, False,
                )
                futures[future] = type_id

            for future in as_completed(futures):
                type_id = futures[future]
                try:
                    results[type_id] = future.result()
                except Exception as exc:
                    logger.error(
                        "Inner DSE failed for type %s: %s", type_id, exc
                    )
                    results[type_id] = InnerResult(
                        type_id=type_id, success=False
                    )
    else:
        # Sequential execution
        for type_id, baseline, mask, kernels in type_configs:
            results[type_id] = optimize_single_type(
                type_id, baseline, mask, kernels,
                config, compile_fn,
            )

    return results
