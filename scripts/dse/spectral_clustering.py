"""Spectral clustering for core-type discovery.

Analyzes kernel resource requirements to discover natural groupings
that map to distinct core types. Uses kernel similarity based on
operation histograms, memory profiles, and control characteristics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .design_space import CoreTypeConfig
from .proxy_model import KernelProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Canonical feature order for the kernel feature matrix.
FEATURE_NAMES = [
    "alu_frac",
    "mul_frac",
    "fp_frac",
    "mem_frac",
    "memory_ratio",
    "load_store_ratio",
    "dfg_size",
    "control_density",
]


@dataclass
class KernelFeatures:
    """Extracted feature vector for one kernel."""

    kernel_name: str = ""
    features: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))

    def as_dict(self) -> Dict[str, float]:
        return {
            name: float(self.features[i])
            for i, name in enumerate(FEATURE_NAMES)
        }


def extract_features(kernel: KernelProfile) -> KernelFeatures:
    """Build a normalized feature vector from a KernelProfile."""
    total_ops = max(1, sum(kernel.op_histogram.values()))
    mem_ops = kernel.loads_per_iter + kernel.stores_per_iter

    def _op_fraction(keys: Sequence[str]) -> float:
        count = 0
        for op_name, op_count in kernel.op_histogram.items():
            op_lower = op_name.lower()
            if any(k in op_lower for k in keys):
                count += op_count
        return count / total_ops

    alu_frac = _op_fraction(["add", "sub", "and", "or", "xor", "shift", "cmp"])
    mul_frac = _op_fraction(["mul", "div", "rem"])
    fp_frac = _op_fraction(["fp", "float", "fadd", "fmul", "fdiv"])
    mem_frac = _op_fraction(["load", "store", "mem"])

    memory_ratio = kernel.memory_footprint_bytes / max(1, total_ops)
    ls_ratio = kernel.loads_per_iter / max(1, kernel.stores_per_iter + kernel.loads_per_iter)
    dfg_size = float(kernel.dfg_node_count)

    # Control density: fraction of comparison / branch-like ops.
    control_density = _op_fraction(["cmp", "select", "br", "mux"])

    vec = np.array([
        alu_frac,
        mul_frac,
        fp_frac,
        mem_frac,
        memory_ratio,
        ls_ratio,
        dfg_size,
        control_density,
    ])

    return KernelFeatures(kernel_name=kernel.name, features=vec)


def build_feature_matrix(kernels: Sequence[KernelProfile]) -> Tuple[np.ndarray, List[str]]:
    """Build the (n_kernels x n_features) matrix.

    Returns:
        X: feature matrix (n_kernels, n_features).
        names: list of kernel names in the same order.
    """
    features_list = [extract_features(k) for k in kernels]
    X = np.vstack([f.features for f in features_list])
    names = [f.kernel_name for f in features_list]
    return X, names


# ---------------------------------------------------------------------------
# Core-type discovery via spectral clustering
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Result of the core-type discovery process."""

    # Number of clusters chosen.
    k: int = 0

    # Cluster labels per kernel (same order as input).
    labels: np.ndarray = field(default_factory=lambda: np.array([]))

    # Derived core-type configs, one per cluster.
    core_types: List[CoreTypeConfig] = field(default_factory=list)

    # Silhouette score of the chosen clustering.
    silhouette_score: float = 0.0

    # Mapping feasibility score (fraction of kernels that fit their cluster type).
    feasibility_score: float = 0.0


class CoreTypeDiscovery:
    """Discover core types by clustering kernels based on resource needs."""

    def __init__(
        self,
        k_range: Tuple[int, int] = (2, 5),
        gamma: Optional[float] = None,
        seed: int = 42,
    ):
        self.k_range = k_range
        self.gamma = gamma
        self.seed = seed

    def discover(
        self,
        kernels: Sequence[KernelProfile],
    ) -> ClusterResult:
        """Run spectral clustering and derive core-type configs.

        Tries cluster counts in k_range and picks the best by a combined
        silhouette + feasibility metric.
        """
        if len(kernels) < 2:
            ct = self._derive_single_core_type(kernels)
            return ClusterResult(
                k=1,
                labels=np.zeros(len(kernels), dtype=int),
                core_types=[ct],
                silhouette_score=1.0,
                feasibility_score=1.0,
            )

        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        X, names = build_feature_matrix(kernels)
        X_scaled = StandardScaler().fit_transform(X)

        best_result: Optional[ClusterResult] = None
        best_combined = -1.0

        k_lo, k_hi = self.k_range
        # Cannot have more clusters than samples
        k_hi = min(k_hi, len(kernels))
        k_lo = min(k_lo, k_hi)

        for k in range(k_lo, k_hi + 1):
            try:
                clustering = SpectralClustering(
                    n_clusters=k,
                    affinity="rbf",
                    gamma=self.gamma,
                    random_state=self.seed,
                    n_init=10,
                )
                labels = clustering.fit_predict(X_scaled)
            except Exception as exc:
                logger.warning("Spectral clustering failed for k=%d: %s", k, exc)
                continue

            if k > 1 and len(set(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)
            else:
                sil = 0.0

            core_types = self._derive_core_types(kernels, labels, k)
            feas = self._validate_feasibility(kernels, core_types, labels)

            combined = 0.4 * sil + 0.6 * feas

            logger.info(
                "k=%d  silhouette=%.3f  feasibility=%.3f  combined=%.3f",
                k, sil, feas, combined,
            )

            if combined > best_combined:
                best_combined = combined
                best_result = ClusterResult(
                    k=k,
                    labels=labels,
                    core_types=core_types,
                    silhouette_score=sil,
                    feasibility_score=feas,
                )

        if best_result is None:
            # Fallback: single cluster
            ct = self._derive_single_core_type(kernels)
            return ClusterResult(
                k=1,
                labels=np.zeros(len(kernels), dtype=int),
                core_types=[ct],
            )

        return best_result

    # -------------------------------------------------------------------
    # Core-type derivation
    # -------------------------------------------------------------------

    def _derive_core_types(
        self,
        kernels: Sequence[KernelProfile],
        labels: np.ndarray,
        k: int,
    ) -> List[CoreTypeConfig]:
        """Derive one CoreTypeConfig per cluster from member kernels."""
        core_types: List[CoreTypeConfig] = []
        for cluster_id in range(k):
            members = [
                kernels[i] for i in range(len(kernels)) if labels[i] == cluster_id
            ]
            if not members:
                core_types.append(CoreTypeConfig())
                continue
            core_types.append(self._derive_one_core_type(members))
        return core_types

    def _derive_one_core_type(
        self, members: Sequence[KernelProfile]
    ) -> CoreTypeConfig:
        """Derive a core-type config from a cluster of kernels.

        Uses the maximum resource demand across the cluster so that every
        member kernel is guaranteed to fit.
        """
        max_alu = 1
        max_mul = 0
        max_fp = 0
        max_mem = 1
        max_dfg = 1
        max_spm = 4

        for k in members:
            alu, mul, fp, mem = self._count_fu_demand(k)
            max_alu = max(max_alu, alu)
            max_mul = max(max_mul, mul)
            max_fp = max(max_fp, fp)
            max_mem = max(max_mem, mem)
            max_dfg = max(max_dfg, k.dfg_node_count)
            spm_kb = max(4, (k.memory_footprint_bytes + 1023) // 1024)
            max_spm = max(max_spm, spm_kb)

        grid_size = self._estimate_grid_size(max_dfg)

        # Snap SPM to power of 2
        import math
        max_spm = 2 ** math.ceil(math.log2(max(4, max_spm)))
        max_spm = min(64, max_spm)

        return CoreTypeConfig(
            pe_grid_rows=grid_size,
            pe_grid_cols=grid_size,
            fu_alu_count=max_alu,
            fu_mul_count=max_mul,
            fu_fp_count=max_fp,
            fu_mem_count=max_mem,
            spm_size_kb=max_spm,
            instance_count=len(members),
        )

    def _derive_single_core_type(
        self, kernels: Sequence[KernelProfile]
    ) -> CoreTypeConfig:
        return self._derive_one_core_type(list(kernels))

    @staticmethod
    def _count_fu_demand(kernel: KernelProfile) -> Tuple[int, int, int, int]:
        """Count per-type FU demand from operation histogram."""
        alu = 0
        mul = 0
        fp = 0
        mem = 0
        for op_name, count in kernel.op_histogram.items():
            op_lower = op_name.lower()
            if any(k in op_lower for k in ("mul", "div", "rem")):
                mul += count
            elif any(k in op_lower for k in ("fp", "float", "fadd", "fmul")):
                fp += count
            elif any(k in op_lower for k in ("load", "store", "mem")):
                mem += count
            else:
                alu += count
        # Convert from op counts to FU counts: at least ceil(count/target_II)
        # where target II ~ 4 for a balanced design.
        import math
        target_ii = 4
        return (
            max(1, math.ceil(alu / target_ii)),
            math.ceil(mul / target_ii) if mul > 0 else 0,
            math.ceil(fp / target_ii) if fp > 0 else 0,
            max(1, math.ceil(mem / target_ii)),
        )

    @staticmethod
    def _estimate_grid_size(dfg_node_count: int) -> int:
        """Estimate PE grid side length from DFG node count."""
        import math
        side = max(2, math.ceil(math.sqrt(dfg_node_count)))
        return min(8, side)

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    def _validate_feasibility(
        self,
        kernels: Sequence[KernelProfile],
        core_types: List[CoreTypeConfig],
        labels: np.ndarray,
    ) -> float:
        """Fraction of kernels whose demands fit within their assigned core type."""
        if not kernels:
            return 1.0
        fits = 0
        for i, kernel in enumerate(kernels):
            ct = core_types[labels[i]]
            if self._kernel_fits(kernel, ct):
                fits += 1
        return fits / len(kernels)

    @staticmethod
    def _kernel_fits(kernel: KernelProfile, ct: CoreTypeConfig) -> bool:
        """Check if a kernel's resource demands can be satisfied by a core type."""
        if kernel.dfg_node_count > ct.num_pes:
            return False
        if kernel.memory_footprint_bytes > ct.spm_bytes:
            return False
        # Check FU coverage
        for op_name, count in kernel.op_histogram.items():
            if count == 0:
                continue
            op_lower = op_name.lower()
            if any(k in op_lower for k in ("mul", "div", "rem")):
                if ct.fu_mul_count == 0:
                    return False
            elif any(k in op_lower for k in ("fp", "float", "fadd", "fmul")):
                if ct.fu_fp_count == 0:
                    return False
            elif any(k in op_lower for k in ("load", "store", "mem")):
                if ct.fu_mem_count == 0:
                    return False
        return True
