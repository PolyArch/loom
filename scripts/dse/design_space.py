"""Design space definition for LOOM DSE.

Defines the DesignPoint representation, parameter encoding/decoding,
and sampling strategies (random, Latin Hypercube, grid).
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dse_config import (
    DEFAULT_PARAM_RANGES,
    MAX_CORE_TYPES,
    NOC_TOPOLOGIES,
)


# ---------------------------------------------------------------------------
# Core-type configuration
# ---------------------------------------------------------------------------

@dataclass
class CoreTypeConfig:
    """Configuration of a single core type."""

    pe_grid_rows: int = 4
    pe_grid_cols: int = 4
    fu_alu_count: int = 2
    fu_mul_count: int = 2
    fu_fp_count: int = 0
    fu_mem_count: int = 2
    spm_size_kb: int = 16
    instance_count: int = 1

    @property
    def num_pes(self) -> int:
        return self.pe_grid_rows * self.pe_grid_cols

    @property
    def num_switches(self) -> int:
        """Estimate switch count: one per PE row + column boundary."""
        return self.pe_grid_rows + self.pe_grid_cols

    @property
    def spm_bytes(self) -> int:
        return self.spm_size_kb * 1024

    @property
    def fu_mix(self) -> Dict[str, int]:
        return {
            "alu": self.fu_alu_count,
            "mul": self.fu_mul_count,
            "fp": self.fu_fp_count,
            "mem": self.fu_mem_count,
        }


# ---------------------------------------------------------------------------
# Design point
# ---------------------------------------------------------------------------

@dataclass
class DesignPoint:
    """Complete architecture configuration for one DSE candidate."""

    core_types: List[CoreTypeConfig] = field(default_factory=list)
    noc_topology: str = "mesh"
    noc_bandwidth: int = 1
    l2_size_kb: int = 256

    def total_cores(self) -> int:
        return sum(ct.instance_count for ct in self.core_types)

    def to_vector(self) -> np.ndarray:
        """Encode this design point as a flat numeric vector for the GP."""
        vec: List[float] = []
        vec.append(float(len(self.core_types)))
        for i in range(MAX_CORE_TYPES):
            if i < len(self.core_types):
                ct = self.core_types[i]
                vec.extend([
                    float(ct.pe_grid_rows),
                    float(ct.pe_grid_cols),
                    float(ct.fu_alu_count),
                    float(ct.fu_mul_count),
                    float(ct.fu_fp_count),
                    float(ct.fu_mem_count),
                    float(ct.spm_size_kb),
                    float(ct.instance_count),
                ])
            else:
                vec.extend([0.0] * 8)
        vec.append(float(NOC_TOPOLOGIES.index(self.noc_topology)))
        vec.append(float(self.noc_bandwidth))
        vec.append(float(self.l2_size_kb))
        return np.array(vec, dtype=np.float64)

    @staticmethod
    def from_vector(vec: np.ndarray) -> "DesignPoint":
        """Decode a flat numeric vector back to a DesignPoint."""
        idx = 0
        n_types = max(1, min(MAX_CORE_TYPES, int(round(vec[idx]))))
        idx += 1

        core_types: List[CoreTypeConfig] = []
        for i in range(MAX_CORE_TYPES):
            vals = vec[idx: idx + 8]
            idx += 8
            if i < n_types:
                ct = CoreTypeConfig(
                    pe_grid_rows=_clamp_int(vals[0], 2, 8),
                    pe_grid_cols=_clamp_int(vals[1], 2, 8),
                    fu_alu_count=_clamp_int(vals[2], 1, 8),
                    fu_mul_count=_clamp_int(vals[3], 0, 8),
                    fu_fp_count=_clamp_int(vals[4], 0, 4),
                    fu_mem_count=_clamp_int(vals[5], 1, 4),
                    spm_size_kb=_nearest_power_of_2(vals[6], 4, 64),
                    instance_count=_clamp_int(vals[7], 1, 8),
                )
                core_types.append(ct)

        topo_idx = _clamp_int(vec[idx], 0, len(NOC_TOPOLOGIES) - 1)
        idx += 1
        noc_bw = _clamp_int(vec[idx], 1, 4)
        idx += 1
        l2_kb = _nearest_power_of_2(vec[idx], 64, 1024)
        idx += 1

        return DesignPoint(
            core_types=core_types,
            noc_topology=NOC_TOPOLOGIES[topo_idx],
            noc_bandwidth=noc_bw,
            l2_size_kb=l2_kb,
        )

    @staticmethod
    def vector_dimension() -> int:
        """Return the dimensionality of the encoded vector."""
        # 1 (n_types) + MAX_CORE_TYPES*8 + 3 (topo, bw, l2)
        return 1 + MAX_CORE_TYPES * 8 + 3

    def to_arch_json(self) -> Dict[str, Any]:
        """Export to the JSON schema consumed by tapestry-pipeline."""
        cores = []
        for ct in self.core_types:
            cores.append({
                "pe_grid": [ct.pe_grid_rows, ct.pe_grid_cols],
                "fu_mix": ct.fu_mix,
                "spm_size_kb": ct.spm_size_kb,
                "instance_count": ct.instance_count,
            })
        return {
            "core_types": cores,
            "noc": {
                "topology": self.noc_topology,
                "bandwidth": self.noc_bandwidth,
            },
            "l2_size_kb": self.l2_size_kb,
        }

    def summary(self) -> str:
        parts = [f"{len(self.core_types)} core types"]
        for i, ct in enumerate(self.core_types):
            parts.append(
                f"  T{i}: {ct.pe_grid_rows}x{ct.pe_grid_cols} PEs, "
                f"ALU={ct.fu_alu_count} MUL={ct.fu_mul_count} "
                f"FP={ct.fu_fp_count} MEM={ct.fu_mem_count}, "
                f"SPM={ct.spm_size_kb}KB x{ct.instance_count}"
            )
        parts.append(
            f"  NoC: {self.noc_topology} BW={self.noc_bandwidth}, "
            f"L2={self.l2_size_kb}KB"
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Search space and sampling
# ---------------------------------------------------------------------------

class DesignSpace:
    """Defines bounds and provides sampling methods for the design space.

    Supports optional TDC-derived constraint bounds that tighten the search
    space lower limits based on contract analysis.
    """

    def __init__(
        self,
        param_ranges: Optional[Dict[str, Tuple]] = None,
        noc_topologies: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.ranges = param_ranges or dict(DEFAULT_PARAM_RANGES)
        self.topologies = noc_topologies or list(NOC_TOPOLOGIES)
        self.rng = np.random.RandomState(seed)

        # TDC-derived constraint lower bounds (None = no TDC constraints)
        self._tdc_min_noc_bw: Optional[float] = None
        self._tdc_min_l2_kb: Optional[float] = None
        self._tdc_min_core_types: Optional[int] = None
        self._tdc_min_total_cores: Optional[int] = None

    def set_tdc_bounds(
        self,
        min_noc_bw: float = 0.0,
        min_l2_kb: float = 0.0,
        min_core_types: int = 1,
        min_total_cores: int = 1,
    ) -> None:
        """Set TDC-derived constraint lower bounds.

        Tightens the design space bounds so that sampling avoids clearly
        infeasible regions. Call before bounds() or any sampling method.
        """
        self._tdc_min_noc_bw = min_noc_bw
        self._tdc_min_l2_kb = min_l2_kb
        self._tdc_min_core_types = min_core_types
        self._tdc_min_total_cores = min_total_cores

        # Apply to ranges
        if min_noc_bw > 0:
            lo, hi = self.ranges["noc_bandwidth"]
            self.ranges["noc_bandwidth"] = (
                max(lo, int(math.ceil(min_noc_bw))),
                hi,
            )
        if min_l2_kb > 0:
            lo, hi = self.ranges["l2_size_kb"]
            self.ranges["l2_size_kb"] = (
                max(lo, int(math.ceil(min_l2_kb))),
                hi,
            )
        if min_core_types > 1:
            lo, hi = self.ranges["core_type_count"]
            self.ranges["core_type_count"] = (
                max(lo, min_core_types),
                hi,
            )

    def is_feasible(self, point: DesignPoint) -> bool:
        """Check if a design point satisfies TDC-derived constraints.

        Returns True if no TDC bounds are set or all constraints are met.
        """
        if self._tdc_min_noc_bw is not None:
            if point.noc_bandwidth < self._tdc_min_noc_bw:
                return False
        if self._tdc_min_l2_kb is not None:
            if point.l2_size_kb < self._tdc_min_l2_kb:
                return False
        if self._tdc_min_core_types is not None:
            if len(point.core_types) < self._tdc_min_core_types:
                return False
        if self._tdc_min_total_cores is not None:
            if point.total_cores() < self._tdc_min_total_cores:
                return False
        return True

    @property
    def dimension(self) -> int:
        return DesignPoint.vector_dimension()

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bound arrays for the encoded vector."""
        r = self.ranges
        lo: List[float] = []
        hi: List[float] = []
        # core_type_count
        lo.append(float(r["core_type_count"][0]))
        hi.append(float(r["core_type_count"][1]))
        # per core-type params (repeated MAX_CORE_TYPES times)
        for _ in range(MAX_CORE_TYPES):
            lo.extend([
                float(r["pe_grid_rows"][0]),
                float(r["pe_grid_cols"][0]),
                float(r["fu_alu_count"][0]),
                float(r["fu_mul_count"][0]),
                float(r["fu_fp_count"][0]),
                float(r["fu_mem_count"][0]),
                float(r["spm_size_kb"][0]),
                float(r["cores_per_type"][0]),
            ])
            hi.extend([
                float(r["pe_grid_rows"][1]),
                float(r["pe_grid_cols"][1]),
                float(r["fu_alu_count"][1]),
                float(r["fu_mul_count"][1]),
                float(r["fu_fp_count"][1]),
                float(r["fu_mem_count"][1]),
                float(r["spm_size_kb"][1]),
                float(r["cores_per_type"][1]),
            ])
        # noc_topology (categorical encoded as integer index)
        lo.append(0.0)
        hi.append(float(len(self.topologies) - 1))
        # noc_bandwidth
        lo.append(float(r["noc_bandwidth"][0]))
        hi.append(float(r["noc_bandwidth"][1]))
        # l2_size_kb
        lo.append(float(r["l2_size_kb"][0]))
        hi.append(float(r["l2_size_kb"][1]))
        return np.array(lo), np.array(hi)

    def sample_random(self, n: int = 1) -> List[DesignPoint]:
        """Uniform random sampling within bounds."""
        lo, hi = self.bounds()
        points: List[DesignPoint] = []
        for _ in range(n):
            vec = self.rng.uniform(lo, hi)
            points.append(DesignPoint.from_vector(vec))
        return points

    def sample_latin_hypercube(self, n: int) -> List[DesignPoint]:
        """Latin Hypercube Sampling for better space coverage."""
        lo, hi = self.bounds()
        d = len(lo)
        # Generate LHS intervals
        result = np.zeros((n, d))
        for j in range(d):
            perm = self.rng.permutation(n)
            for i in range(n):
                u = (perm[i] + self.rng.uniform()) / n
                result[i, j] = lo[j] + u * (hi[j] - lo[j])
        return [DesignPoint.from_vector(result[i]) for i in range(n)]

    def sample_grid(
        self, steps_per_dim: int = 3, max_samples: int = 1000
    ) -> List[DesignPoint]:
        """Grid sampling on a coarse lattice (may be truncated)."""
        lo, hi = self.bounds()
        d = len(lo)
        # Generate axis values
        axes = []
        for j in range(d):
            vals = np.linspace(lo[j], hi[j], steps_per_dim)
            axes.append(vals)
        # Full grid would be steps^d; we subsample if too large
        total = steps_per_dim ** d
        if total <= max_samples:
            grid = np.array(np.meshgrid(*axes)).T.reshape(-1, d)
        else:
            # Random subset of grid points
            grid = np.zeros((max_samples, d))
            for i in range(max_samples):
                for j in range(d):
                    grid[i, j] = self.rng.choice(axes[j])
        return [DesignPoint.from_vector(grid[i]) for i in range(len(grid))]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp_int(val: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(val))))


def _nearest_power_of_2(val: float, lo: int, hi: int) -> int:
    """Snap to the nearest power of 2 within [lo, hi]."""
    val = max(lo, min(hi, val))
    if val <= 0:
        return lo
    log2_val = math.log2(val)
    low_p2 = 2 ** int(math.floor(log2_val))
    high_p2 = 2 ** int(math.ceil(log2_val))
    low_p2 = max(lo, low_p2)
    high_p2 = min(hi, high_p2)
    if abs(val - low_p2) <= abs(val - high_p2):
        return low_p2
    return high_p2
