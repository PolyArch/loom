"""Shared utilities for E11-E16 Hardware Bilevel DSE experiments.

Provides:
- Benchmark domain definitions with kernel profiles (op histograms, etc.)
- Workload profile construction from TDG files and C source analysis
- Area estimation using the AnalyticalResourceModel constants
- Common provenance helpers (git hash, timestamp)
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Add repo root to path so scripts.dse is importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dse.dse_config import (
    FU_AREA_TABLE,
    PE_AREA_UM2,
    SRAM_AREA_PER_BYTE_UM2,
    SW_AREA_UM2,
)
from scripts.dse.design_space import CoreTypeConfig, DesignPoint
from scripts.dse.proxy_model import (
    AnalyticalResourceModel,
    ContractEdge,
    KernelProfile,
    ProxyScore,
    WorkloadProfile,
)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def git_hash() -> str:
    """Get current git short hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Op histogram extraction from C source files
# ---------------------------------------------------------------------------

# Patterns for different operation categories in C kernel source
_OP_PATTERNS = {
    "fadd": [r'\bsum\s*\+=', r'\+\s*=\s*[a-zA-Z]', r'\b\w+\s*=\s*\w+\s*\+\s*\w+'],
    "fmul": [r'\*\s*[a-zA-Z]\w*\[', r'\bm31_mul\b', r'\bw\s*\*\s*'],
    "load": [r'\w+\[.*?\](?!\s*=)', r'\bdata\[', r'\bnbrs\['],
    "store": [r'\w+\[.*?\]\s*=', r'\bdata\[.*?\]\s*=', r'\blevel\[.*?\]\s*='],
    "cmp": [r'\bif\s*\(', r'==\s*-1', r'>\s*\d', r'<\s*\d'],
    "add": [r'\+\+', r'\+=\s*1', r'\bcur_level\s*\+\+'],
    "shift": [r'>>\s*\d', r'<<\s*\d', r'\brev\s*<<'],
    "mul_int": [r'\*\s*stride', r'\*\s*N\b', r'\*\s*K\b', r'\bi\s*\*\s*\d'],
    "sub": [r'\b\w+\s*-\s*\w+', r'\bm31_sub\b'],
}


def _count_op_occurrences(content: str, patterns: List[str]) -> int:
    """Count approximate operation occurrences from regex patterns."""
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, content))
    return total


def extract_op_histogram_from_c(filepath: str) -> Dict[str, int]:
    """Extract an approximate operation histogram from a C kernel file.

    Uses regex-based pattern matching on the source code to estimate
    the operation mix. This is intentionally approximate -- a real
    implementation would use LLVM IR analysis.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except (FileNotFoundError, IOError):
        return {"add": 4, "load": 2, "store": 1}

    # Only look at the tiled kernel function body (not reference or main)
    # Find functions that aren't _ref and aren't main
    func_bodies = []
    for m in re.finditer(
        r'(?:void|int|float|m31_t)\s+(\w+)\s*\([^)]*\)\s*\{',
        content,
    ):
        fname = m.group(1)
        if fname == "main" or fname.endswith("_ref"):
            continue
        start = m.end() - 1
        depth = 1
        pos = start + 1
        while pos < len(content) and depth > 0:
            if content[pos] == '{':
                depth += 1
            elif content[pos] == '}':
                depth -= 1
            pos += 1
        func_bodies.append(content[start:pos])

    if not func_bodies:
        func_bodies = [content]

    body = "\n".join(func_bodies)

    histogram: Dict[str, int] = {}
    for op_name, patterns in _OP_PATTERNS.items():
        count = _count_op_occurrences(body, patterns)
        if count > 0:
            histogram[op_name] = count

    # Ensure at least minimal histogram
    if not histogram:
        histogram = {"add": 4, "load": 2, "store": 1}

    return histogram


# ---------------------------------------------------------------------------
# Kernel profile database (all 33 kernels across 6 domains)
# ---------------------------------------------------------------------------

# Kernel type -> approximate characteristics.
# These are derived from domain knowledge and C source structure.
KERNEL_PROFILES: Dict[str, Dict[str, Any]] = {
    # ai_llm domain
    "qkv_proj": {
        "domain": "ai_llm",
        "type": "matmul",
        "op_histogram": {"fmul": 32, "fadd": 32, "load": 16, "store": 8},
        "dfg_node_count": 48,
        "memory_footprint_bytes": 32768,
        "loads_per_iter": 16,
        "stores_per_iter": 8,
        "source": "ai_llm/qkv_proj.c",
    },
    "attn_score": {
        "domain": "ai_llm",
        "type": "batched_matmul",
        "op_histogram": {"fmul": 24, "fadd": 24, "load": 12, "store": 6},
        "dfg_node_count": 36,
        "memory_footprint_bytes": 16384,
        "loads_per_iter": 12,
        "stores_per_iter": 6,
        "source": "ai_llm/attn_score.c",
    },
    "softmax": {
        "domain": "ai_llm",
        "type": "elementwise",
        "op_histogram": {"fadd": 8, "fmul": 4, "load": 4, "store": 2, "cmp": 2},
        "dfg_node_count": 16,
        "memory_footprint_bytes": 4096,
        "loads_per_iter": 4,
        "stores_per_iter": 2,
        "source": "ai_llm/softmax.c",
    },
    "attn_output": {
        "domain": "ai_llm",
        "type": "batched_matmul",
        "op_histogram": {"fmul": 24, "fadd": 24, "load": 12, "store": 6},
        "dfg_node_count": 36,
        "memory_footprint_bytes": 16384,
        "loads_per_iter": 12,
        "stores_per_iter": 6,
        "source": "ai_llm/attn_output.c",
    },
    "ffn1": {
        "domain": "ai_llm",
        "type": "matmul",
        "op_histogram": {"fmul": 32, "fadd": 32, "load": 16, "store": 8},
        "dfg_node_count": 48,
        "memory_footprint_bytes": 65536,
        "loads_per_iter": 16,
        "stores_per_iter": 8,
        "source": "ai_llm/ffn1.c",
    },
    "gelu": {
        "domain": "ai_llm",
        "type": "elementwise",
        "op_histogram": {"fmul": 6, "fadd": 4, "load": 2, "store": 2},
        "dfg_node_count": 12,
        "memory_footprint_bytes": 4096,
        "loads_per_iter": 2,
        "stores_per_iter": 2,
        "source": "ai_llm/gelu.c",
    },
    "ffn2": {
        "domain": "ai_llm",
        "type": "matmul",
        "op_histogram": {"fmul": 32, "fadd": 32, "load": 16, "store": 8},
        "dfg_node_count": 48,
        "memory_footprint_bytes": 65536,
        "loads_per_iter": 16,
        "stores_per_iter": 8,
        "source": "ai_llm/ffn2.c",
    },
    "layernorm": {
        "domain": "ai_llm",
        "type": "reduction",
        "op_histogram": {"fadd": 12, "fmul": 8, "load": 6, "store": 2, "cmp": 1},
        "dfg_node_count": 20,
        "memory_footprint_bytes": 8192,
        "loads_per_iter": 6,
        "stores_per_iter": 2,
        "source": "ai_llm/layernorm.c",
    },
    # dsp_ofdm domain
    "fft_butterfly": {
        "domain": "dsp_ofdm",
        "type": "fft",
        "op_histogram": {"fmul": 16, "fadd": 16, "load": 8, "store": 4, "shift": 4},
        "dfg_node_count": 32,
        "memory_footprint_bytes": 16384,
        "loads_per_iter": 8,
        "stores_per_iter": 4,
        "source": "dsp_ofdm/fft_butterfly.c",
    },
    "channel_est": {
        "domain": "dsp_ofdm",
        "type": "interpolation",
        "op_histogram": {"fmul": 12, "fadd": 8, "load": 6, "store": 3},
        "dfg_node_count": 24,
        "memory_footprint_bytes": 9600,
        "loads_per_iter": 6,
        "stores_per_iter": 3,
        "source": "dsp_ofdm/channel_est.c",
    },
    "equalizer": {
        "domain": "dsp_ofdm",
        "type": "elementwise",
        "op_histogram": {"fmul": 8, "fadd": 4, "load": 4, "store": 2},
        "dfg_node_count": 14,
        "memory_footprint_bytes": 4800,
        "loads_per_iter": 4,
        "stores_per_iter": 2,
        "source": "dsp_ofdm/equalizer.c",
    },
    "qam_demod": {
        "domain": "dsp_ofdm",
        "type": "demapping",
        "op_histogram": {"cmp": 8, "add": 6, "load": 4, "store": 2, "mul_int": 2},
        "dfg_node_count": 18,
        "memory_footprint_bytes": 4800,
        "loads_per_iter": 4,
        "stores_per_iter": 2,
        "source": "dsp_ofdm/qam_demod.c",
    },
    "viterbi": {
        "domain": "dsp_ofdm",
        "type": "decoder",
        "op_histogram": {"add": 16, "cmp": 12, "load": 8, "store": 4},
        "dfg_node_count": 28,
        "memory_footprint_bytes": 28800,
        "loads_per_iter": 8,
        "stores_per_iter": 4,
        "source": "dsp_ofdm/viterbi.c",
    },
    "crc_check": {
        "domain": "dsp_ofdm",
        "type": "check",
        "op_histogram": {"shift": 8, "add": 4, "load": 2, "store": 1, "cmp": 2},
        "dfg_node_count": 12,
        "memory_footprint_bytes": 7200,
        "loads_per_iter": 2,
        "stores_per_iter": 1,
        "source": "dsp_ofdm/crc_check.c",
    },
    # arvr_stereo domain
    "harris_corner": {
        "domain": "arvr_stereo",
        "type": "feature_detect",
        "op_histogram": {"fmul": 12, "fadd": 8, "load": 6, "store": 2, "cmp": 4},
        "dfg_node_count": 24,
        "memory_footprint_bytes": 16384,
        "loads_per_iter": 6,
        "stores_per_iter": 2,
        "source": "arvr_stereo/harris_corner.c",
    },
    "sad_matching": {
        "domain": "arvr_stereo",
        "type": "matching",
        "op_histogram": {"sub": 16, "add": 12, "load": 8, "store": 2, "cmp": 4},
        "dfg_node_count": 28,
        "memory_footprint_bytes": 1048576,
        "loads_per_iter": 8,
        "stores_per_iter": 2,
        "source": "arvr_stereo/sad_matching.c",
    },
    "stereo_disparity": {
        "domain": "arvr_stereo",
        "type": "optimization",
        "op_histogram": {"fadd": 10, "fmul": 8, "load": 6, "store": 3, "cmp": 4},
        "dfg_node_count": 22,
        "memory_footprint_bytes": 16384,
        "loads_per_iter": 6,
        "stores_per_iter": 3,
        "source": "arvr_stereo/stereo_disparity.c",
    },
    "image_warp": {
        "domain": "arvr_stereo",
        "type": "interpolation",
        "op_histogram": {"fmul": 8, "fadd": 6, "load": 4, "store": 2},
        "dfg_node_count": 16,
        "memory_footprint_bytes": 16384,
        "loads_per_iter": 4,
        "stores_per_iter": 2,
        "source": "arvr_stereo/image_warp.c",
    },
    "post_filter": {
        "domain": "arvr_stereo",
        "type": "filter",
        "op_histogram": {"fmul": 9, "fadd": 9, "load": 5, "store": 2},
        "dfg_node_count": 18,
        "memory_footprint_bytes": 16384,
        "loads_per_iter": 5,
        "stores_per_iter": 2,
        "source": "arvr_stereo/post_filter.c",
    },
    # robotics_vio domain
    "imu_integration": {
        "domain": "robotics_vio",
        "type": "sequential_accum",
        "op_histogram": {"fadd": 8, "fmul": 6, "load": 4, "store": 2},
        "dfg_node_count": 16,
        "memory_footprint_bytes": 2400,
        "loads_per_iter": 4,
        "stores_per_iter": 2,
        "source": "robotics_vio/imu_integration.c",
    },
    "fast_detect": {
        "domain": "robotics_vio",
        "type": "stencil_2d",
        "op_histogram": {"cmp": 12, "add": 8, "load": 6, "store": 2},
        "dfg_node_count": 20,
        "memory_footprint_bytes": 4000,
        "loads_per_iter": 6,
        "stores_per_iter": 2,
        "source": "robotics_vio/fast_detect.c",
    },
    "orb_descriptor": {
        "domain": "robotics_vio",
        "type": "patch_compute",
        "op_histogram": {"cmp": 8, "shift": 4, "add": 6, "load": 4, "store": 2},
        "dfg_node_count": 18,
        "memory_footprint_bytes": 16000,
        "loads_per_iter": 4,
        "stores_per_iter": 2,
        "source": "robotics_vio/orb_descriptor.c",
    },
    "feature_match": {
        "domain": "robotics_vio",
        "type": "brute_force_search",
        "op_histogram": {"sub": 8, "add": 6, "cmp": 8, "load": 4, "store": 1},
        "dfg_node_count": 22,
        "memory_footprint_bytes": 16000,
        "loads_per_iter": 4,
        "stores_per_iter": 1,
        "source": "robotics_vio/feature_match.c",
    },
    "pose_estimate": {
        "domain": "robotics_vio",
        "type": "linear_algebra",
        "op_histogram": {"fmul": 16, "fadd": 12, "load": 8, "store": 4},
        "dfg_node_count": 28,
        "memory_footprint_bytes": 3200,
        "loads_per_iter": 8,
        "stores_per_iter": 4,
        "source": "robotics_vio/pose_estimate.c",
    },
    # graph_analytics domain
    "bfs_traversal": {
        "domain": "graph_analytics",
        "type": "frontier_based",
        "op_histogram": {"cmp": 12, "add": 8, "load": 6, "store": 4},
        "dfg_node_count": 20,
        "memory_footprint_bytes": 4096,
        "loads_per_iter": 6,
        "stores_per_iter": 4,
        "source": "graph_analytics/bfs_traversal.c",
    },
    "pagerank_spmv": {
        "domain": "graph_analytics",
        "type": "spmv_iterative",
        "op_histogram": {"fmul": 8, "fadd": 8, "load": 6, "store": 2, "cmp": 2},
        "dfg_node_count": 18,
        "memory_footprint_bytes": 4096,
        "loads_per_iter": 6,
        "stores_per_iter": 2,
        "source": "graph_analytics/pagerank_spmv.c",
    },
    "triangle_count": {
        "domain": "graph_analytics",
        "type": "set_intersection",
        "op_histogram": {"cmp": 16, "add": 6, "load": 8, "store": 2},
        "dfg_node_count": 22,
        "memory_footprint_bytes": 4096,
        "loads_per_iter": 8,
        "stores_per_iter": 2,
        "source": "graph_analytics/triangle_count.c",
    },
    "label_prop": {
        "domain": "graph_analytics",
        "type": "neighbor_vote",
        "op_histogram": {"cmp": 10, "add": 6, "load": 6, "store": 3},
        "dfg_node_count": 18,
        "memory_footprint_bytes": 4096,
        "loads_per_iter": 6,
        "stores_per_iter": 3,
        "source": "graph_analytics/label_prop.c",
    },
    # zk_stark domain
    "ntt": {
        "domain": "zk_stark",
        "type": "butterfly_transform",
        "op_histogram": {"mul_int": 16, "add": 12, "sub": 8, "load": 8, "store": 4, "shift": 6},
        "dfg_node_count": 32,
        "memory_footprint_bytes": 4096,
        "loads_per_iter": 8,
        "stores_per_iter": 4,
        "source": "zk_stark/ntt.c",
    },
    "msm": {
        "domain": "zk_stark",
        "type": "bucket_accumulate",
        "op_histogram": {"mul_int": 12, "add": 10, "load": 6, "store": 3, "cmp": 4},
        "dfg_node_count": 24,
        "memory_footprint_bytes": 8192,
        "loads_per_iter": 6,
        "stores_per_iter": 3,
        "source": "zk_stark/msm.c",
    },
    "poseidon_hash": {
        "domain": "zk_stark",
        "type": "permutation_sponge",
        "op_histogram": {"mul_int": 20, "add": 16, "load": 4, "store": 2},
        "dfg_node_count": 28,
        "memory_footprint_bytes": 2048,
        "loads_per_iter": 4,
        "stores_per_iter": 2,
        "source": "zk_stark/poseidon_hash.c",
    },
    "poly_eval": {
        "domain": "zk_stark",
        "type": "horner_batch",
        "op_histogram": {"mul_int": 14, "add": 10, "load": 6, "store": 2},
        "dfg_node_count": 22,
        "memory_footprint_bytes": 4096,
        "loads_per_iter": 6,
        "stores_per_iter": 2,
        "source": "zk_stark/poly_eval.c",
    },
    "proof_compose": {
        "domain": "zk_stark",
        "type": "linear_combination",
        "op_histogram": {"mul_int": 8, "add": 12, "load": 6, "store": 3},
        "dfg_node_count": 20,
        "memory_footprint_bytes": 4096,
        "loads_per_iter": 6,
        "stores_per_iter": 3,
        "source": "zk_stark/proof_compose.c",
    },
}

DOMAIN_NAMES = [
    "ai_llm", "dsp_ofdm", "arvr_stereo",
    "robotics_vio", "graph_analytics", "zk_stark",
]


# ---------------------------------------------------------------------------
# Build KernelProfile objects from the database
# ---------------------------------------------------------------------------

def build_kernel_profile(name: str) -> KernelProfile:
    """Build a KernelProfile from the KERNEL_PROFILES database."""
    info = KERNEL_PROFILES[name]
    return KernelProfile(
        name=name,
        op_histogram=dict(info["op_histogram"]),
        memory_footprint_bytes=info["memory_footprint_bytes"],
        loads_per_iter=info["loads_per_iter"],
        stores_per_iter=info["stores_per_iter"],
        dfg_node_count=info["dfg_node_count"],
    )


def get_domain_kernels(domain: str) -> List[KernelProfile]:
    """Return all kernel profiles for a given domain."""
    profiles = []
    for name, info in KERNEL_PROFILES.items():
        if info["domain"] == domain:
            profiles.append(build_kernel_profile(name))
    return profiles


def get_all_kernels() -> List[KernelProfile]:
    """Return all 33 kernel profiles."""
    return [build_kernel_profile(name) for name in KERNEL_PROFILES]


# ---------------------------------------------------------------------------
# Area estimation using shared constants
# ---------------------------------------------------------------------------

def estimate_core_area(ct: CoreTypeConfig) -> float:
    """Estimate area of a single core type instance in um^2."""
    pe_area = ct.num_pes * PE_AREA_UM2
    fu_area = (
        ct.fu_alu_count * FU_AREA_TABLE["alu"]
        + ct.fu_mul_count * FU_AREA_TABLE["mul"]
        + ct.fu_fp_count * FU_AREA_TABLE["fp"]
        + ct.fu_mem_count * FU_AREA_TABLE["mem"]
    )
    spm_area = ct.spm_bytes * SRAM_AREA_PER_BYTE_UM2
    sw_area = ct.num_switches * SW_AREA_UM2
    return pe_area + fu_area + spm_area + sw_area


def estimate_design_area(design: DesignPoint) -> float:
    """Estimate total area of a design point."""
    proxy = AnalyticalResourceModel()
    return proxy._estimate_area(design)


# ---------------------------------------------------------------------------
# Core type role classification
# ---------------------------------------------------------------------------

CORE_ROLE_PROFILES = {
    "ctrl": {
        "label": "Control-heavy",
        "fu_alu_count": 4, "fu_mul_count": 1,
        "fu_fp_count": 0, "fu_mem_count": 2,
        "pe_grid_rows": 4, "pe_grid_cols": 4,
        "spm_size_kb": 8,
    },
    "gp": {
        "label": "General-purpose",
        "fu_alu_count": 3, "fu_mul_count": 2,
        "fu_fp_count": 1, "fu_mem_count": 2,
        "pe_grid_rows": 6, "pe_grid_cols": 6,
        "spm_size_kb": 16,
    },
    "dsp": {
        "label": "DSP-oriented",
        "fu_alu_count": 2, "fu_mul_count": 3,
        "fu_fp_count": 2, "fu_mem_count": 2,
        "pe_grid_rows": 6, "pe_grid_cols": 6,
        "spm_size_kb": 16,
    },
    "ai": {
        "label": "AI/FP-heavy",
        "fu_alu_count": 2, "fu_mul_count": 2,
        "fu_fp_count": 4, "fu_mem_count": 2,
        "pe_grid_rows": 8, "pe_grid_cols": 8,
        "spm_size_kb": 32,
    },
}


def make_core_type(role: str, instance_count: int = 1) -> CoreTypeConfig:
    """Create a CoreTypeConfig from a role profile name."""
    prof = CORE_ROLE_PROFILES[role]
    return CoreTypeConfig(
        pe_grid_rows=prof["pe_grid_rows"],
        pe_grid_cols=prof["pe_grid_cols"],
        fu_alu_count=prof["fu_alu_count"],
        fu_mul_count=prof["fu_mul_count"],
        fu_fp_count=prof["fu_fp_count"],
        fu_mem_count=prof["fu_mem_count"],
        spm_size_kb=prof["spm_size_kb"],
        instance_count=instance_count,
    )


def classify_kernel_to_role(kp: KernelProfile) -> str:
    """Classify a kernel to a core role based on its op histogram."""
    total = max(1, sum(kp.op_histogram.values()))
    fp_ops = 0
    ctrl_ops = 0
    mul_ops = 0
    for op, count in kp.op_histogram.items():
        op_lower = op.lower()
        if any(k in op_lower for k in ("fp", "float", "fadd", "fmul", "fdiv")):
            fp_ops += count
        elif any(k in op_lower for k in ("cmp", "select", "br", "mux")):
            ctrl_ops += count
        elif any(k in op_lower for k in ("mul", "div", "rem")):
            mul_ops += count

    fp_frac = fp_ops / total
    ctrl_frac = ctrl_ops / total
    mul_frac = mul_ops / total

    if fp_frac > 0.3:
        return "ai"
    if ctrl_frac > 0.25:
        return "ctrl"
    if mul_frac > 0.25:
        return "dsp"
    return "gp"


# ---------------------------------------------------------------------------
# Build workload profiles from TDG definitions
# ---------------------------------------------------------------------------

# TDG contract definitions per domain
TDG_CONTRACTS = {
    "ai_llm": [
        ("qkv_proj", "attn_score", 2048.0, "LOCAL_SPM", "FIFO"),
        ("attn_score", "softmax", 4096.0, "LOCAL_SPM", "FIFO"),
        ("softmax", "attn_output", 4096.0, "LOCAL_SPM", "FIFO"),
        ("attn_output", "ffn1", 16384.0, "LOCAL_SPM", "FIFO"),
        ("ffn1", "gelu", 65536.0, "LOCAL_SPM", "FIFO"),
        ("gelu", "ffn2", 65536.0, "LOCAL_SPM", "FIFO"),
        ("ffn2", "layernorm", 16384.0, "LOCAL_SPM", "FIFO"),
    ],
    "dsp_ofdm": [
        ("fft_butterfly", "channel_est", 4096.0, "LOCAL_SPM", "FIFO"),
        ("channel_est", "equalizer", 1200.0, "LOCAL_SPM", "FIFO"),
        ("equalizer", "qam_demod", 1200.0, "LOCAL_SPM", "FIFO"),
        ("qam_demod", "viterbi", 7200.0, "LOCAL_SPM", "FIFO"),
        ("viterbi", "crc_check", 1800.0, "LOCAL_SPM", "FIFO"),
    ],
    "arvr_stereo": [
        ("harris_corner", "sad_matching", 4096.0, "LOCAL_SPM", "FIFO"),
        ("sad_matching", "stereo_disparity", 262144.0, "LOCAL_SPM", "FIFO"),
        ("stereo_disparity", "image_warp", 4096.0, "LOCAL_SPM", "FIFO"),
        ("image_warp", "post_filter", 4096.0, "LOCAL_SPM", "FIFO"),
    ],
    "robotics_vio": [
        ("imu_integration", "pose_estimate", 600.0, "LOCAL_SPM", "FIFO"),
        ("fast_detect", "orb_descriptor", 1000.0, "LOCAL_SPM", "FIFO"),
        ("orb_descriptor", "feature_match", 4000.0, "LOCAL_SPM", "FIFO"),
        ("feature_match", "pose_estimate", 400.0, "LOCAL_SPM", "FIFO"),
    ],
    "graph_analytics": [
        ("bfs_traversal", "pagerank_spmv", 1024.0, "EXTERNAL_DRAM", "FIFO"),
        ("pagerank_spmv", "label_prop", 1024.0, "EXTERNAL_DRAM", "FIFO"),
        ("bfs_traversal", "triangle_count", 1024.0, "EXTERNAL_DRAM", "FIFO"),
    ],
    "zk_stark": [
        ("ntt", "poly_eval", 1024.0, "LOCAL_SPM", "FIFO"),
        ("poly_eval", "proof_compose", 256.0, "LOCAL_SPM", "FIFO"),
        ("poseidon_hash", "proof_compose", 4.0, "LOCAL_SPM", "FIFO"),
        ("msm", "proof_compose", 3.0, "LOCAL_SPM", "FIFO"),
        ("ntt", "poseidon_hash", 8.0, "LOCAL_SPM", "FIFO"),
    ],
}


def build_workload_for_domain(domain: str) -> WorkloadProfile:
    """Build a WorkloadProfile for a single domain."""
    kernels = get_domain_kernels(domain)
    contracts_def = TDG_CONTRACTS.get(domain, [])
    contracts = []
    for prod, cons, rate, vis, ordering in contracts_def:
        contracts.append(ContractEdge(
            producer=prod,
            consumer=cons,
            production_rate=rate,
            element_size_bytes=4,
            visibility=vis,
            ordering=ordering,
        ))
    critical_path = [k.name for k in kernels]
    return WorkloadProfile(
        kernels=kernels,
        contracts=contracts,
        critical_path=critical_path,
    )


def build_combined_workload() -> WorkloadProfile:
    """Build a combined workload with all 33 kernels and all contracts."""
    all_kernels = get_all_kernels()
    all_contracts = []
    for domain in DOMAIN_NAMES:
        for prod, cons, rate, vis, ordering in TDG_CONTRACTS.get(domain, []):
            all_contracts.append(ContractEdge(
                producer=prod,
                consumer=cons,
                production_rate=rate,
                element_size_bytes=4,
                visibility=vis,
                ordering=ordering,
            ))
    critical_path = [k.name for k in all_kernels]
    return WorkloadProfile(
        kernels=all_kernels,
        contracts=all_contracts,
        critical_path=critical_path,
    )
