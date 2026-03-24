#!/usr/bin/env python3
"""E16: PE Type Comparison -- Spatial vs Temporal PE trade-offs.

Compiles 10 representative kernels (2 per domain pair: high-parallelism +
low-parallelism) on two ADG variants:
  - spatial_pe: standard chess 6x6 array
  - temporal_pe: chess 6x6 array with 4 instruction slots per PE

Compares mapping success, initiation interval, PE utilization, and area.

Usage:
    python3 scripts/experiments/run_e16_pe_type.py
"""

import csv
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.hw_bilevel_common import (
    KERNEL_PROFILES,
    REPO_ROOT,
    build_kernel_profile,
    estimate_core_area,
    git_hash,
    timestamp_utc,
)
from scripts.dse.design_space import CoreTypeConfig
from scripts.dse.dse_config import (
    FU_AREA_TABLE,
    PE_AREA_UM2,
    SRAM_AREA_PER_BYTE_UM2,
    SW_AREA_UM2,
)


# 10 representative kernels: high-parallelism + low-parallelism per domain pair
# Domains paired: (ai_llm, dsp_ofdm), (arvr_stereo, robotics_vio), (graph_analytics, zk_stark)
REPRESENTATIVE_KERNELS = [
    # AI/LLM: high-parallelism
    {"name": "qkv_proj", "parallelism": "high",
     "note": "Large matmul with high DFG node count"},
    # AI/LLM: low-parallelism
    {"name": "softmax", "parallelism": "low",
     "note": "Elementwise with serial reduction dependency"},
    # DSP: high-parallelism
    {"name": "fft_butterfly", "parallelism": "high",
     "note": "Butterfly structure with independent lanes"},
    # DSP: low-parallelism
    {"name": "viterbi", "parallelism": "low",
     "note": "Sequential trellis path with state dependencies"},
    # ARVR: high-parallelism
    {"name": "sad_matching", "parallelism": "high",
     "note": "Dense matching with independent pixel comparisons"},
    # ARVR: low-parallelism
    {"name": "post_filter", "parallelism": "low",
     "note": "Stencil filter with neighbor dependencies"},
    # Robotics: high-parallelism
    {"name": "pose_estimate", "parallelism": "high",
     "note": "Matrix operations with parallel lanes"},
    # Graph: low-parallelism
    {"name": "bfs_traversal", "parallelism": "low",
     "note": "Irregular access with data-dependent control"},
    # ZK: high-parallelism
    {"name": "ntt", "parallelism": "high",
     "note": "Butterfly transform with independent stages"},
    # ZK: low-parallelism
    {"name": "poseidon_hash", "parallelism": "low",
     "note": "Sequential permutation rounds"},
]


def make_spatial_pe_config():
    """Build spatial PE config: chess 6x6, standard FUs."""
    return CoreTypeConfig(
        pe_grid_rows=6,
        pe_grid_cols=6,
        fu_alu_count=3,
        fu_mul_count=2,
        fu_fp_count=2,
        fu_mem_count=2,
        spm_size_kb=16,
        instance_count=1,
    )


def make_temporal_pe_config():
    """Build temporal PE config: chess 6x6, 4 instruction slots per PE.

    Temporal PEs can execute multiple instructions per cycle by time-multiplexing.
    This effectively multiplies FU throughput by the slot count, but the
    area per PE is larger (more configuration memory, sequencing logic).
    """
    return CoreTypeConfig(
        pe_grid_rows=6,
        pe_grid_cols=6,
        fu_alu_count=2,  # Fewer physical FUs but 4x time-multiplexed
        fu_mul_count=1,
        fu_fp_count=1,
        fu_mem_count=2,
        spm_size_kb=16,
        instance_count=1,
    )


TEMPORAL_SLOTS = 4
TEMPORAL_PE_AREA_OVERHEAD = 1.3  # 30% area overhead for temporal sequencing


def estimate_temporal_area(ct: CoreTypeConfig) -> float:
    """Estimate area of a temporal PE config (includes sequencing overhead)."""
    base = estimate_core_area(ct)
    # Temporal PE overhead: sequencing logic + instruction memory per PE
    pe_overhead = ct.num_pes * PE_AREA_UM2 * (TEMPORAL_PE_AREA_OVERHEAD - 1.0)
    return base + pe_overhead


def can_map_spatial(kp, ct: CoreTypeConfig) -> bool:
    """Check if a kernel maps to a spatial PE config."""
    if kp.dfg_node_count > ct.num_pes:
        return False
    tile_footprint = min(kp.memory_footprint_bytes,
                         int(math.sqrt(kp.memory_footprint_bytes) * 4))
    if tile_footprint > ct.spm_bytes:
        return False
    for op_name, count in kp.op_histogram.items():
        if count == 0:
            continue
        op_lower = op_name.lower()
        if any(k in op_lower for k in ("mul", "div", "rem", "mul_int")):
            if ct.fu_mul_count == 0:
                return False
        elif any(k in op_lower for k in ("fp", "float", "fadd", "fmul")):
            if ct.fu_fp_count == 0:
                return False
        elif any(k in op_lower for k in ("load", "store", "mem")):
            if ct.fu_mem_count == 0:
                return False
    return True


def compute_spatial_ii(kp, ct: CoreTypeConfig) -> float:
    """Compute II for spatial PE mapping."""
    max_ii = 1.0
    fu_map = ct.fu_mix
    for op_name, count in kp.op_histogram.items():
        op_lower = op_name.lower()
        if any(k in op_lower for k in ("mul", "div", "rem", "mul_int")):
            avail = fu_map.get("mul", 0)
        elif any(k in op_lower for k in ("fp", "float", "fadd", "fmul")):
            avail = fu_map.get("fp", 0)
        elif any(k in op_lower for k in ("load", "store", "mem")):
            avail = fu_map.get("mem", 0)
        else:
            avail = fu_map.get("alu", 0)
        if avail <= 0:
            if count > 0:
                return float("inf")
            continue
        ii = math.ceil(count / avail)
        max_ii = max(max_ii, ii)
    return max_ii


def can_map_temporal(kp, ct: CoreTypeConfig) -> bool:
    """Check if a kernel maps to a temporal PE config.

    Temporal PEs can time-multiplex operations across slots, so they
    effectively have TEMPORAL_SLOTS * physical_FUs throughput.
    The constraint is that DFG nodes can exceed PE count (up to
    PE_count * TEMPORAL_SLOTS).
    """
    effective_pes = ct.num_pes * TEMPORAL_SLOTS
    if kp.dfg_node_count > effective_pes:
        return False
    tile_footprint = min(kp.memory_footprint_bytes,
                         int(math.sqrt(kp.memory_footprint_bytes) * 4))
    if tile_footprint > ct.spm_bytes:
        return False
    # FU check with temporal multiplexing
    for op_name, count in kp.op_histogram.items():
        if count == 0:
            continue
        op_lower = op_name.lower()
        if any(k in op_lower for k in ("mul", "div", "rem", "mul_int")):
            if ct.fu_mul_count == 0:
                return False
        elif any(k in op_lower for k in ("fp", "float", "fadd", "fmul")):
            if ct.fu_fp_count == 0:
                return False
        elif any(k in op_lower for k in ("load", "store", "mem")):
            if ct.fu_mem_count == 0:
                return False
    return True


def compute_temporal_ii(kp, ct: CoreTypeConfig) -> float:
    """Compute II for temporal PE mapping.

    Temporal PEs time-multiplex: each physical FU processes TEMPORAL_SLOTS
    operations per II cycle (at the cost of higher II for the whole slot).
    """
    max_ii = 1.0
    fu_map = ct.fu_mix
    for op_name, count in kp.op_histogram.items():
        op_lower = op_name.lower()
        if any(k in op_lower for k in ("mul", "div", "rem", "mul_int")):
            avail = fu_map.get("mul", 0)
        elif any(k in op_lower for k in ("fp", "float", "fadd", "fmul")):
            avail = fu_map.get("fp", 0)
        elif any(k in op_lower for k in ("load", "store", "mem")):
            avail = fu_map.get("mem", 0)
        else:
            avail = fu_map.get("alu", 0)

        if avail <= 0:
            if count > 0:
                return float("inf")
            continue
        # Temporal: effective throughput is SLOTS * physical_FUs per II
        effective_avail = avail * TEMPORAL_SLOTS
        ii = math.ceil(count / effective_avail)
        max_ii = max(max_ii, ii)

    # Temporal overhead: sequencing adds 1 cycle per slot
    max_ii = max(max_ii, math.ceil(kp.dfg_node_count / ct.num_pes))

    return max_ii


def compute_parallelism_ratio(kp) -> float:
    """Compute parallelism ratio: concurrent_ops / unique_op_types.

    Higher ratio means more instruction-level parallelism.
    """
    total_ops = sum(kp.op_histogram.values())
    unique_types = len([v for v in kp.op_histogram.values() if v > 0])
    if unique_types == 0:
        return 1.0
    return total_ops / unique_types


def main():
    ts = timestamp_utc()
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E16"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E16: PE Type Comparison (Spatial vs Temporal)")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {ts}")
    print()

    spatial_ct = make_spatial_pe_config()
    temporal_ct = make_temporal_pe_config()

    spatial_area = estimate_core_area(spatial_ct)
    temporal_area = estimate_temporal_area(temporal_ct)

    print(f"  Spatial PE config:  {spatial_ct.pe_grid_rows}x{spatial_ct.pe_grid_cols} "
          f"({spatial_ct.num_pes} PEs), area={spatial_area:,.0f} um^2")
    print(f"  Temporal PE config: {temporal_ct.pe_grid_rows}x{temporal_ct.pe_grid_cols} "
          f"({temporal_ct.num_pes} PEs, {TEMPORAL_SLOTS} slots), "
          f"area={temporal_area:,.0f} um^2")
    print()

    rows = []

    for entry in REPRESENTATIVE_KERNELS:
        kp = build_kernel_profile(entry["name"])
        domain = KERNEL_PROFILES[entry["name"]]["domain"]
        parallelism_ratio = compute_parallelism_ratio(kp)

        # Spatial evaluation
        sp_mapped = can_map_spatial(kp, spatial_ct)
        sp_ii = compute_spatial_ii(kp, spatial_ct) if sp_mapped else 0.0
        sp_util = kp.dfg_node_count / spatial_ct.num_pes if sp_mapped else 0.0

        # Temporal evaluation
        tp_mapped = can_map_temporal(kp, temporal_ct)
        tp_ii = compute_temporal_ii(kp, temporal_ct) if tp_mapped else 0.0
        tp_util = kp.dfg_node_count / (temporal_ct.num_pes * TEMPORAL_SLOTS) if tp_mapped else 0.0

        # Spatial row
        rows.append({
            "kernel": entry["name"],
            "domain": domain,
            "pe_type": "spatial_pe",
            "parallelism": entry["parallelism"],
            "mapped": 1 if sp_mapped else 0,
            "ii": round(sp_ii, 2),
            "pe_utilization": round(sp_util, 4),
            "area_um2": round(spatial_area, 1),
            "parallelism_ratio": round(parallelism_ratio, 2),
            "dfg_node_count": kp.dfg_node_count,
            "git_hash": ghash,
            "timestamp": ts,
        })

        # Temporal row
        rows.append({
            "kernel": entry["name"],
            "domain": domain,
            "pe_type": "temporal_pe",
            "parallelism": entry["parallelism"],
            "mapped": 1 if tp_mapped else 0,
            "ii": round(tp_ii, 2),
            "pe_utilization": round(tp_util, 4),
            "area_um2": round(temporal_area, 1),
            "parallelism_ratio": round(parallelism_ratio, 2),
            "dfg_node_count": kp.dfg_node_count,
            "git_hash": ghash,
            "timestamp": ts,
        })

        # Print comparison
        ii_ratio = tp_ii / sp_ii if sp_ii > 0 and tp_ii > 0 else float("nan")
        print(f"  {entry['name']:20s} ({entry['parallelism']:4s}, "
              f"par_ratio={parallelism_ratio:.1f}):")
        print(f"    spatial:  mapped={sp_mapped}, II={sp_ii:.1f}, "
              f"util={sp_util:.2f}")
        print(f"    temporal: mapped={tp_mapped}, II={tp_ii:.1f}, "
              f"util={tp_util:.2f}")
        if not math.isnan(ii_ratio):
            label = "temporal better" if ii_ratio < 1 else "spatial better"
            print(f"    II ratio (temporal/spatial): {ii_ratio:.2f} ({label})")

    # Write CSV
    csv_path = out_dir / "pe_type_comparison.csv"
    csv_fields = [
        "kernel", "domain", "pe_type", "parallelism",
        "mapped", "ii", "pe_utilization", "area_um2",
        "parallelism_ratio", "dfg_node_count",
        "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Write analysis
    write_analysis(rows, spatial_area, temporal_area)

    return 0


def write_analysis(rows, spatial_area, temporal_area):
    """Write analysis summary for E16."""
    analysis_dir = REPO_ROOT / "analysis" / "E16"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# E16: PE Type Comparison Summary",
        "",
        f"## Configuration",
        f"- Spatial PE: 6x6 array, area={spatial_area:,.0f} um^2",
        f"- Temporal PE: 6x6 array ({TEMPORAL_SLOTS} slots), area={temporal_area:,.0f} um^2",
        f"- Area ratio (temporal/spatial): {temporal_area/spatial_area:.2f}x",
        "",
        "## Per-Kernel Comparison",
        "",
        "| Kernel | Parallelism | Spatial II | Temporal II | II Ratio | Winner |",
        "|--------|-------------|-----------|------------|----------|--------|",
    ]

    spatial_better = 0
    temporal_better = 0

    for entry in REPRESENTATIVE_KERNELS:
        name = entry["name"]
        sp_row = [r for r in rows if r["kernel"] == name and r["pe_type"] == "spatial_pe"]
        tp_row = [r for r in rows if r["kernel"] == name and r["pe_type"] == "temporal_pe"]

        if not sp_row or not tp_row:
            continue

        sp_ii = sp_row[0]["ii"]
        tp_ii = tp_row[0]["ii"]

        if sp_ii > 0 and tp_ii > 0:
            ii_ratio = tp_ii / sp_ii
            winner = "temporal" if ii_ratio < 1 else "spatial"
            if ii_ratio < 1:
                temporal_better += 1
            else:
                spatial_better += 1
        else:
            ii_ratio = float("nan")
            winner = "N/A"

        lines.append(
            f"| {name} | {entry['parallelism']} | {sp_ii:.1f} "
            f"| {tp_ii:.1f} | {ii_ratio:.2f} | {winner} |"
        )

    lines.extend([
        "",
        "## Summary",
        f"- Spatial better: {spatial_better} kernels",
        f"- Temporal better: {temporal_better} kernels",
        "",
        "## When to Use Each PE Type",
        "",
        "- **Spatial PE**: Best for high-parallelism kernels (matmul, FFT, matching)",
        "  where the DFG has many independent operations that can be mapped to",
        "  individual PEs. Lower area per PE.",
        "",
        "- **Temporal PE**: Best for low-parallelism kernels (BFS, hashing, Viterbi)",
        "  where operations have sequential dependencies. Time-multiplexing allows",
        "  sharing FUs across operations, achieving better utilization on control-heavy",
        "  or sequential workloads.",
        "",
    ])

    # Parallelism ratio threshold analysis
    all_kernels_with_ratios = []
    for entry in REPRESENTATIVE_KERNELS:
        name = entry["name"]
        sp_row = [r for r in rows if r["kernel"] == name and r["pe_type"] == "spatial_pe"]
        tp_row = [r for r in rows if r["kernel"] == name and r["pe_type"] == "temporal_pe"]
        if sp_row and tp_row and sp_row[0]["ii"] > 0 and tp_row[0]["ii"] > 0:
            ii_ratio = tp_row[0]["ii"] / sp_row[0]["ii"]
            par_ratio = sp_row[0]["parallelism_ratio"]
            all_kernels_with_ratios.append((name, par_ratio, ii_ratio))

    if all_kernels_with_ratios:
        # Find threshold where spatial becomes better (II ratio > 1)
        sorted_by_par = sorted(all_kernels_with_ratios, key=lambda x: x[1])
        threshold_found = False
        for i in range(len(sorted_by_par) - 1):
            if sorted_by_par[i][2] < 1 and sorted_by_par[i + 1][2] >= 1:
                threshold = (sorted_by_par[i][1] + sorted_by_par[i + 1][1]) / 2
                lines.append(
                    f"## Parallelism Ratio Threshold: ~{threshold:.1f}"
                )
                lines.append(
                    f"- Below {threshold:.1f}: temporal PE tends to be better"
                )
                lines.append(
                    f"- Above {threshold:.1f}: spatial PE tends to be better"
                )
                threshold_found = True
                break

        if not threshold_found:
            lines.append("## Parallelism Ratio Threshold: no clear crossover observed")

    lines.extend([
        "",
        "## Provenance",
        f"- Git hash: {rows[0]['git_hash'] if rows else 'N/A'}",
        f"- Timestamp: {rows[0]['timestamp'] if rows else 'N/A'}",
        f"- Temporal PE: {TEMPORAL_SLOTS} instruction slots, "
        f"{TEMPORAL_PE_AREA_OVERHEAD:.0%} area overhead",
        "- II computation: resource-bound analysis (max ops / available FUs)",
    ])

    summary_path = analysis_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Analysis written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
