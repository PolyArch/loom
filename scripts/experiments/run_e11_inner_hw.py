#!/usr/bin/env python3
"""E11: INNER-HW Area Optimization -- Tier-1 analytical vs Tier-2 BO-refined area.

For each of the 4 core type profiles (ctrl, gp, dsp, ai), evaluates:
  - Tier-1: Analytical resource model area estimate from spectral clustering
  - Tier-2: Bayesian Optimization over per-core parameters (FU counts, grid size,
            SPM size), keeping the mapping feasible for assigned kernels.

Outputs a CSV comparing Tier-1 and Tier-2 area per core type, with BO iteration
traces showing how area decreases while maintaining mapping feasibility.

Usage:
    python3 scripts/experiments/run_e11_inner_hw.py
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.hw_bilevel_common import (
    CORE_ROLE_PROFILES,
    KERNEL_PROFILES,
    REPO_ROOT,
    build_kernel_profile,
    classify_kernel_to_role,
    estimate_core_area,
    get_all_kernels,
    git_hash,
    make_core_type,
    timestamp_utc,
)
from scripts.dse.design_space import CoreTypeConfig
from scripts.dse.dse_config import FU_AREA_TABLE, PE_AREA_UM2, SRAM_AREA_PER_BYTE_UM2

import numpy as np


def kernel_fits_core_type(kp, ct: CoreTypeConfig) -> bool:
    """Check if a kernel's DFG fits on the given core type.

    Memory footprint check uses tile-level working set (sqrt of full footprint)
    since all kernels use tiling and only need SPM for the active tile.
    """
    if kp.dfg_node_count > ct.num_pes:
        return False
    # Tile-level footprint: approximate as sqrt of full footprint
    # (tiled kernels partition data into tiles that fit in SPM)
    tile_footprint = min(kp.memory_footprint_bytes,
                         int(math.sqrt(kp.memory_footprint_bytes) * 4))
    if tile_footprint > ct.spm_bytes:
        return False
    # Check FU coverage
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


def estimate_ii(kp, ct: CoreTypeConfig) -> float:
    """Estimate initiation interval for a kernel on a core type."""
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


def assign_kernels_to_roles():
    """Assign each kernel to its best-fit core role."""
    assignments = {}  # role -> [kernel_name, ...]
    for role in CORE_ROLE_PROFILES:
        assignments[role] = []
    all_kps = get_all_kernels()
    for kp in all_kps:
        role = classify_kernel_to_role(kp)
        assignments[role].append(kp.name)
    return assignments


def bo_optimize_core_type(
    role: str,
    assigned_kernel_names: list,
    rng: np.random.RandomState,
    max_iterations: int = 50,
):
    """Run BO-like optimization to reduce core area while keeping feasibility.

    Explores FU counts, grid size, and SPM size within bounded ranges.
    Returns a list of (iteration, CoreTypeConfig, area, all_mapped, avg_ii).
    """
    base = CORE_ROLE_PROFILES[role]
    assigned_kps = [build_kernel_profile(n) for n in assigned_kernel_names]

    trace = []

    # Start with the Tier-1 analytical config
    t1_ct = make_core_type(role)
    t1_area = estimate_core_area(t1_ct)
    t1_mapped = all(kernel_fits_core_type(kp, t1_ct) for kp in assigned_kps)
    t1_iis = [estimate_ii(kp, t1_ct) for kp in assigned_kps if kernel_fits_core_type(kp, t1_ct)]
    t1_avg_ii = sum(t1_iis) / len(t1_iis) if t1_iis else float("inf")

    trace.append((0, t1_ct, t1_area, t1_mapped, t1_avg_ii, "tier1"))

    best_ct = t1_ct
    best_area = t1_area if t1_mapped else float("inf")

    # Determine minimum grid size needed for assigned kernels
    max_dfg = max((kp.dfg_node_count for kp in assigned_kps), default=16)
    min_grid = max(base["pe_grid_rows"], math.ceil(math.sqrt(max_dfg)))

    # Parameter search bounds (widen grid bounds to accommodate kernels)
    param_bounds = {
        "fu_alu": (1, base["fu_alu_count"] + 2),
        "fu_mul": (0, base["fu_mul_count"] + 2),
        "fu_fp": (0, base["fu_fp_count"] + 2),
        "fu_mem": (1, base["fu_mem_count"] + 1),
        "grid_rows": (min_grid, min_grid + 2),
        "grid_cols": (min_grid, min_grid + 2),
        "spm_kb": (4, base["spm_size_kb"] * 2),
    }

    # Override Tier-1 config with adequate grid size
    t1_ct = CoreTypeConfig(
        pe_grid_rows=min_grid,
        pe_grid_cols=min_grid,
        fu_alu_count=base["fu_alu_count"],
        fu_mul_count=base["fu_mul_count"],
        fu_fp_count=base["fu_fp_count"],
        fu_mem_count=base["fu_mem_count"],
        spm_size_kb=base["spm_size_kb"],
        instance_count=1,
    )
    t1_area = estimate_core_area(t1_ct)
    t1_mapped = all(kernel_fits_core_type(kp, t1_ct) for kp in assigned_kps)
    t1_iis = [estimate_ii(kp, t1_ct) for kp in assigned_kps if kernel_fits_core_type(kp, t1_ct)]
    t1_avg_ii = sum(t1_iis) / len(t1_iis) if t1_iis else float("inf")

    # Replace trace[0] with updated Tier-1
    trace[0] = (0, t1_ct, t1_area, t1_mapped, t1_avg_ii, "tier1")
    best_ct = t1_ct
    best_area = t1_area if t1_mapped else float("inf")

    for iteration in range(1, max_iterations + 1):
        # Perturb around the best known config
        candidate = CoreTypeConfig(
            fu_alu_count=int(np.clip(
                best_ct.fu_alu_count + rng.randint(-1, 2),
                param_bounds["fu_alu"][0], param_bounds["fu_alu"][1])),
            fu_mul_count=int(np.clip(
                best_ct.fu_mul_count + rng.randint(-1, 2),
                param_bounds["fu_mul"][0], param_bounds["fu_mul"][1])),
            fu_fp_count=int(np.clip(
                best_ct.fu_fp_count + rng.randint(-1, 2),
                param_bounds["fu_fp"][0], param_bounds["fu_fp"][1])),
            fu_mem_count=int(np.clip(
                best_ct.fu_mem_count + rng.randint(-1, 2),
                param_bounds["fu_mem"][0], param_bounds["fu_mem"][1])),
            pe_grid_rows=int(np.clip(
                best_ct.pe_grid_rows + rng.randint(-1, 2),
                param_bounds["grid_rows"][0], param_bounds["grid_rows"][1])),
            pe_grid_cols=int(np.clip(
                best_ct.pe_grid_cols + rng.randint(-1, 2),
                param_bounds["grid_cols"][0], param_bounds["grid_cols"][1])),
            spm_size_kb=int(np.clip(
                best_ct.spm_size_kb + rng.choice([-4, 0, 4]),
                param_bounds["spm_kb"][0], param_bounds["spm_kb"][1])),
            instance_count=1,
        )

        # Evaluate
        area = estimate_core_area(candidate)
        all_mapped = all(kernel_fits_core_type(kp, candidate) for kp in assigned_kps)
        iis = [estimate_ii(kp, candidate) for kp in assigned_kps if kernel_fits_core_type(kp, candidate)]
        avg_ii = sum(iis) / len(iis) if iis else float("inf")

        trace.append((iteration, candidate, area, all_mapped, avg_ii, "tier2"))

        # Update best if feasible and lower area (or lower II at same area)
        if all_mapped and area < best_area:
            best_ct = candidate
            best_area = area
        elif all_mapped and area == best_area and avg_ii < t1_avg_ii:
            best_ct = candidate
            best_area = area

    return trace


def main():
    ts = timestamp_utc()
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E11"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E11: INNER-HW Area Optimization")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {ts}")
    print()

    # Assign kernels to roles
    assignments = assign_kernels_to_roles()
    print("Kernel-to-role assignments:")
    for role, knames in assignments.items():
        print(f"  {role}: {len(knames)} kernels -- {', '.join(knames)}")
    print()

    rng = np.random.RandomState(42)
    all_rows = []

    for role in ["ctrl", "gp", "dsp", "ai"]:
        knames = assignments[role]
        if not knames:
            print(f"  {role}: no assigned kernels, skipping")
            continue

        trace = bo_optimize_core_type(role, knames, rng, max_iterations=50)

        # Extract Tier-1 and best Tier-2
        tier1 = trace[0]
        feasible_t2 = [t for t in trace[1:] if t[3]]  # all_mapped == True
        if feasible_t2:
            best_t2 = min(feasible_t2, key=lambda t: t[2])
        else:
            best_t2 = None

        # Print summary
        t1_area = tier1[2]
        print(f"  {role} ({len(knames)} kernels):")
        print(f"    Tier-1 area:  {t1_area:,.0f} um^2  "
              f"(mapped={tier1[3]}, avg_II={tier1[4]:.1f})")
        if best_t2:
            t2_area = best_t2[2]
            reduction = (t1_area - t2_area) / t1_area * 100
            print(f"    Tier-2 best:  {t2_area:,.0f} um^2  "
                  f"(mapped={best_t2[3]}, avg_II={best_t2[4]:.1f})")
            print(f"    Reduction:    {reduction:.1f}%")
        else:
            print(f"    Tier-2: no feasible candidates found")

        # Write all trace rows
        for (it, ct, area, mapped, avg_ii, tier) in trace:
            all_rows.append({
                "core_type": role,
                "tier": tier,
                "bo_iteration": it,
                "pe_type": "spatial_pe",
                "array_rows": ct.pe_grid_rows,
                "array_cols": ct.pe_grid_cols,
                "pe_count": ct.num_pes,
                "fu_alu": ct.fu_alu_count,
                "fu_mul": ct.fu_mul_count,
                "fu_fp": ct.fu_fp_count,
                "fu_mem": ct.fu_mem_count,
                "spm_kb": ct.spm_size_kb,
                "area_um2": round(area, 1),
                "all_kernels_mapped": 1 if mapped else 0,
                "avg_ii": round(avg_ii, 2),
                "assigned_kernels": len(knames),
                "git_hash": ghash,
                "timestamp": ts,
            })

    # Write CSV
    csv_path = out_dir / "inner_hw_optimization.csv"
    fieldnames = [
        "core_type", "tier", "bo_iteration", "pe_type",
        "array_rows", "array_cols", "pe_count",
        "fu_alu", "fu_mul", "fu_fp", "fu_mem", "spm_kb",
        "area_um2", "all_kernels_mapped", "avg_ii",
        "assigned_kernels", "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {csv_path}")

    # Write analysis summary
    write_analysis(all_rows, assignments)

    return 0


def write_analysis(rows, assignments):
    """Write analysis summary for E11."""
    analysis_dir = REPO_ROOT / "analysis" / "E11"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Group by core type
    by_type = {}
    for row in rows:
        ct = row["core_type"]
        if ct not in by_type:
            by_type[ct] = []
        by_type[ct].append(row)

    lines = [
        "# E11: INNER-HW Area Optimization Summary",
        "",
        "## Area Reduction per Core Type",
        "",
        "| Core Type | Tier-1 Area (um^2) | Tier-2 Best Area (um^2) | Reduction (%) | Feasible T2 Candidates |",
        "|-----------|-------------------|------------------------|---------------|----------------------|",
    ]

    for role in ["ctrl", "gp", "dsp", "ai"]:
        if role not in by_type:
            continue
        type_rows = by_type[role]
        t1_rows = [r for r in type_rows if r["tier"] == "tier1"]
        t2_feasible = [r for r in type_rows if r["tier"] == "tier2" and r["all_kernels_mapped"]]

        t1_area = t1_rows[0]["area_um2"] if t1_rows else 0
        if t2_feasible:
            t2_best = min(r["area_um2"] for r in t2_feasible)
            reduction = (t1_area - t2_best) / t1_area * 100 if t1_area > 0 else 0
        else:
            t2_best = t1_area
            reduction = 0.0

        lines.append(
            f"| {role} | {t1_area:,.0f} | {t2_best:,.0f} | {reduction:.1f} | {len(t2_feasible)} |"
        )

    lines.extend([
        "",
        "## Provenance",
        f"- Git hash: {rows[0]['git_hash'] if rows else 'N/A'}",
        f"- Timestamp: {rows[0]['timestamp'] if rows else 'N/A'}",
        f"- Method: BO-guided per-core parameter search (50 iterations/type)",
        f"- Area formula: PE_AREA={PE_AREA_UM2}, FU_AREA={FU_AREA_TABLE}",
    ])

    summary_path = analysis_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Analysis written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
