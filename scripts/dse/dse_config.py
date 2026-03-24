"""DSE configuration: default parameter ranges, tier thresholds, BO settings."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Area constants (um^2, placeholder values calibrated to a 32nm SAED process)
# ---------------------------------------------------------------------------

PE_AREA_UM2 = 12000.0
SW_AREA_UM2 = 3000.0
SRAM_AREA_PER_BYTE_UM2 = 0.5
MUL_FU_AREA_UM2 = 4500.0
FP_FU_AREA_UM2 = 8000.0
ALU_FU_AREA_UM2 = 2000.0
MEM_FU_AREA_UM2 = 3500.0

FU_AREA_TABLE: Dict[str, float] = {
    "alu": ALU_FU_AREA_UM2,
    "mul": MUL_FU_AREA_UM2,
    "fp": FP_FU_AREA_UM2,
    "mem": MEM_FU_AREA_UM2,
}

# ---------------------------------------------------------------------------
# Cache miss rates per contract visibility level
# ---------------------------------------------------------------------------

CACHE_MISS_RATES: Dict[str, float] = {
    "LOCAL_SPM": 0.0,        # scratchpad: guaranteed hit
    "SHARED_L2": 0.1,        # shared L2 cache: configurable miss rate
    "EXTERNAL_DRAM": 0.7,    # off-chip DRAM: high miss rate
}

# ---------------------------------------------------------------------------
# Subprocess timeout for Tier-2/3 compile evaluations (seconds)
# ---------------------------------------------------------------------------

TIER2_TIMEOUT_SEC = 60
TIER3_TIMEOUT_SEC = 600

# ---------------------------------------------------------------------------
# Default search-space bounds
# ---------------------------------------------------------------------------

MAX_CORE_TYPES = 5

DEFAULT_PARAM_RANGES: Dict[str, Tuple] = {
    "core_type_count": (2, 5),
    "pe_grid_rows": (2, 8),
    "pe_grid_cols": (2, 8),
    "fu_alu_count": (1, 8),
    "fu_mul_count": (0, 8),
    "fu_fp_count": (0, 4),
    "fu_mem_count": (1, 4),
    "cores_per_type": (1, 8),
    "spm_size_kb": (4, 64),
    "l2_size_kb": (64, 1024),
    "noc_bandwidth": (1, 4),
}

NOC_TOPOLOGIES = ["mesh", "ring", "hierarchical"]


# ---------------------------------------------------------------------------
# Multi-fidelity tier thresholds
# ---------------------------------------------------------------------------

@dataclass
class TierThresholds:
    """Thresholds that gate promotion from one evaluation tier to the next."""

    # Minimum Tier-1 proxy score (normalized 0-1) to proceed to Tier 2.
    tier2_promotion: float = 0.3

    # Minimum Tier-2 partial-compile score to proceed to Tier 3.
    tier3_promotion: float = 0.5


# ---------------------------------------------------------------------------
# Bayesian-optimization hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class BOConfig:
    """Bayesian Optimization hyperparameters."""

    # Number of initial random samples before fitting the GP.
    n_initial_samples: int = 20

    # Total BO iteration budget (including initial samples).
    max_iterations: int = 200

    # Exploration-exploitation trade-off for UCB acquisition.
    kappa: float = 2.576

    # GP kernel length-scale bounds.
    length_scale_bounds: Tuple[float, float] = (1e-2, 1e3)

    # Random seed for reproducibility.
    seed: int = 42


# ---------------------------------------------------------------------------
# Clustering hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class ClusteringConfig:
    """Spectral clustering hyperparameters for core-type discovery."""

    # Range of cluster counts to evaluate.
    k_range: Tuple[int, int] = (2, 5)

    # RBF kernel gamma (None = auto = 1/n_features).
    gamma: Optional[float] = None

    # Random seed.
    seed: int = 42


# ---------------------------------------------------------------------------
# Top-level DSE configuration
# ---------------------------------------------------------------------------

@dataclass
class DSEConfig:
    """Top-level configuration for a DSE run."""

    # Workload TDG paths.
    workload_tdgs: List[str] = field(default_factory=list)

    # Path to base architecture template JSON.
    arch_template_path: str = ""

    # Output directory for results, checkpoints, plots.
    output_dir: str = "dse-output"

    # Multi-fidelity thresholds.
    tier_thresholds: TierThresholds = field(default_factory=TierThresholds)

    # BO settings.
    bo: BOConfig = field(default_factory=BOConfig)

    # Clustering settings.
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)

    # Path to tapestry-pipeline binary for Tier 2/3 evaluations.
    tapestry_pipeline_bin: str = "tapestry-pipeline"

    # Maximum number of parallel Tier-3 subprocess evaluations.
    max_parallel_evals: int = 4

    # Enable checkpointing for resumable runs.
    enable_checkpointing: bool = True

    # Checkpoint interval (number of iterations between saves).
    checkpoint_interval: int = 20

    # Verbosity level (0=quiet, 1=normal, 2=debug).
    verbosity: int = 1
