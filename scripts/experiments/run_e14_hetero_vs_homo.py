#!/usr/bin/env python3
"""E14: Heterogeneous vs Homogeneous Comparison at matched area budget.

Compares three configurations at approximately the same total area:
  - Hetero:      1xctrl(4x4) + 1xgp(6x6) + 1xdsp(6x6) + 1xai(8x8)
  - Homo-Large:  1xgp(10x10)
  - Homo-Medium: 4xgp(6x6)

All 33 kernels are compiled on each configuration. Compares mapping rate,
system throughput, and throughput-per-area efficiency.

Usage:
    python3 scripts/experiments/run_e14_hetero_vs_homo.py
"""

import csv
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.hw_bilevel_common import (
    DOMAIN_NAMES,
    KERNEL_PROFILES,
    REPO_ROOT,
    build_combined_workload,
    build_kernel_profile,
    classify_kernel_to_role,
    estimate_core_area,
    estimate_design_area,
    get_all_kernels,
    git_hash,
    make_core_type,
    timestamp_utc,
)
from scripts.dse.design_space import CoreTypeConfig, DesignPoint


def build_hetero_config():
    """Build the heterogeneous configuration: ctrl + gp + dsp + ai."""
    return DesignPoint(
        core_types=[
            make_core_type("ctrl", instance_count=1),  # 4x4 = 16 PEs
            make_core_type("gp", instance_count=1),     # 6x6 = 36 PEs
            make_core_type("dsp", instance_count=1),    # 6x6 = 36 PEs
            make_core_type("ai", instance_count=1),     # 8x8 = 64 PEs
        ],
        noc_topology="mesh",
        noc_bandwidth=2,
        l2_size_kb=256,
    )


def build_homo_large_config():
    """Build homogeneous-large: 1x gp(10x10) = 100 PEs."""
    ct = CoreTypeConfig(
        pe_grid_rows=10,
        pe_grid_cols=10,
        fu_alu_count=4,
        fu_mul_count=3,
        fu_fp_count=2,
        fu_mem_count=3,
        spm_size_kb=32,
        instance_count=1,
    )
    return DesignPoint(
        core_types=[ct],
        noc_topology="mesh",
        noc_bandwidth=2,
        l2_size_kb=256,
    )


def build_homo_medium_config():
    """Build homogeneous-medium: 4x gp(6x6) = 144 PEs."""
    ct = CoreTypeConfig(
        pe_grid_rows=6,
        pe_grid_cols=6,
        fu_alu_count=3,
        fu_mul_count=2,
        fu_fp_count=1,
        fu_mem_count=2,
        spm_size_kb=16,
        instance_count=4,
    )
    return DesignPoint(
        core_types=[ct],
        noc_topology="mesh",
        noc_bandwidth=2,
        l2_size_kb=256,
    )


def can_map_kernel(kp, ct: CoreTypeConfig) -> bool:
    """Check if a kernel can map to a core type.

    Uses tile-level memory footprint (kernels use tiling).
    """
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


def compute_ii(kp, ct: CoreTypeConfig) -> float:
    """Estimate initiation interval."""
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


def evaluate_config(config_name: str, design: DesignPoint, all_kernels):
    """Evaluate a configuration against all kernels.

    For hetero configs, each kernel is assigned to its best-fit core type.
    For homo configs, the single core type is used for all kernels.

    Returns (summary_dict, per_kernel_list).
    """
    total_area = estimate_design_area(design)
    total_pes = sum(ct.num_pes * ct.instance_count for ct in design.core_types)

    per_kernel = []
    mapped_count = 0
    total_throughput = 0.0

    for kp in all_kernels:
        # Find best core type for this kernel
        best_ct_idx = -1
        best_ii = float("inf")

        for ct_idx, ct in enumerate(design.core_types):
            if can_map_kernel(kp, ct):
                ii = compute_ii(kp, ct)
                if ii < best_ii:
                    best_ii = ii
                    best_ct_idx = ct_idx

        mapped = best_ct_idx >= 0 and not math.isinf(best_ii)
        domain = KERNEL_PROFILES[kp.name]["domain"]

        if mapped:
            mapped_count += 1
            throughput = 1.0 / best_ii if best_ii > 0 else 0.0
            total_throughput += throughput
        else:
            throughput = 0.0
            best_ii = 0.0
            best_ct_idx = -1

        per_kernel.append({
            "config": config_name,
            "kernel": kp.name,
            "domain": domain,
            "mapped": 1 if mapped else 0,
            "core_type_idx": best_ct_idx,
            "ii": round(best_ii, 2),
            "throughput": round(throughput, 8),
        })

    mapping_rate = mapped_count / len(all_kernels) if all_kernels else 0
    throughput_per_area = total_throughput / total_area * 1e6 if total_area > 0 else 0

    summary = {
        "config": config_name,
        "num_core_types": len(design.core_types),
        "total_pes": total_pes,
        "total_area_um2": round(total_area, 1),
        "kernels_total": len(all_kernels),
        "kernels_mapped": mapped_count,
        "mapping_rate": round(mapping_rate, 4),
        "system_throughput": round(total_throughput, 6),
        "throughput_per_area": round(throughput_per_area, 8),
    }

    return summary, per_kernel


def main():
    ts = timestamp_utc()
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E14"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E14: Heterogeneous vs Homogeneous Comparison")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {ts}")
    print()

    all_kernels = get_all_kernels()
    print(f"  Total kernels: {len(all_kernels)}")
    print()

    configs = {
        "hetero": build_hetero_config(),
        "homo_large": build_homo_large_config(),
        "homo_medium": build_homo_medium_config(),
    }

    summary_rows = []
    detail_rows = []

    for config_name, design in configs.items():
        total_pes = sum(ct.num_pes * ct.instance_count for ct in design.core_types)
        total_area = estimate_design_area(design)
        print(f"  {config_name}:")
        print(f"    Core types: {len(design.core_types)}, "
              f"Total PEs: {total_pes}, "
              f"Area: {total_area:,.0f} um^2")

        summary, per_kernel = evaluate_config(config_name, design, all_kernels)
        summary["git_hash"] = ghash
        summary["timestamp"] = ts
        summary_rows.append(summary)

        for pk in per_kernel:
            pk["git_hash"] = ghash
            pk["timestamp"] = ts
            detail_rows.append(pk)

        print(f"    Mapped: {summary['kernels_mapped']}/{summary['kernels_total']} "
              f"({summary['mapping_rate']:.1%})")
        print(f"    Throughput: {summary['system_throughput']:.4f}")
        print(f"    Throughput/area: {summary['throughput_per_area']:.8f}")

        # Per-domain breakdown
        domain_mapped = {}
        domain_total = {}
        for pk in per_kernel:
            d = pk["domain"]
            domain_total[d] = domain_total.get(d, 0) + 1
            if pk["mapped"]:
                domain_mapped[d] = domain_mapped.get(d, 0) + 1

        for d in DOMAIN_NAMES:
            m = domain_mapped.get(d, 0)
            t = domain_total.get(d, 0)
            rate = m / t if t > 0 else 0
            print(f"      {d}: {m}/{t} ({rate:.0%})")

    # Write matched_budget.csv (summary)
    summary_path = out_dir / "matched_budget.csv"
    summary_fields = [
        "config", "num_core_types", "total_pes", "total_area_um2",
        "kernels_total", "kernels_mapped", "mapping_rate",
        "system_throughput", "throughput_per_area",
        "git_hash", "timestamp",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} rows to {summary_path}")

    # Write per-kernel detail
    detail_path = out_dir / "per_kernel_detail.csv"
    detail_fields = [
        "config", "kernel", "domain", "mapped", "core_type_idx",
        "ii", "throughput", "git_hash", "timestamp",
    ]
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Wrote {len(detail_rows)} rows to {detail_path}")

    # Write analysis
    write_analysis(summary_rows, detail_rows)

    return 0


def write_analysis(summary_rows, detail_rows):
    """Write analysis summary for E14."""
    analysis_dir = REPO_ROOT / "analysis" / "E14"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# E14: Heterogeneous vs Homogeneous Comparison Summary",
        "",
        "## Headline Comparison",
        "",
        "| Config | PEs | Area (um^2) | Mapped | Rate | Throughput | Throughput/Area |",
        "|--------|-----|-------------|--------|------|------------|-----------------|",
    ]

    for r in summary_rows:
        lines.append(
            f"| {r['config']} | {r['total_pes']} | {r['total_area_um2']:,.0f} "
            f"| {r['kernels_mapped']}/{r['kernels_total']} "
            f"| {r['mapping_rate']:.1%} | {r['system_throughput']:.4f} "
            f"| {r['throughput_per_area']:.8f} |"
        )

    # Per-domain mapping coverage
    lines.extend([
        "",
        "## Per-Domain Mapping Coverage",
        "",
        "| Domain | Hetero | Homo-Large | Homo-Medium |",
        "|--------|--------|------------|-------------|",
    ])

    for domain in DOMAIN_NAMES:
        parts = [f"| {domain}"]
        for config in ["hetero", "homo_large", "homo_medium"]:
            domain_rows = [r for r in detail_rows
                           if r["config"] == config and r["domain"] == domain]
            mapped = sum(1 for r in domain_rows if r["mapped"])
            total = len(domain_rows)
            parts.append(f" {mapped}/{total}")
        parts.append(" |")
        lines.append(" |".join(parts))

    # Find which kernel categories drive heterogeneity advantage
    hetero_only = []
    for r in detail_rows:
        if r["config"] == "hetero" and r["mapped"]:
            kernel = r["kernel"]
            homo_l = [x for x in detail_rows
                      if x["config"] == "homo_large" and x["kernel"] == kernel]
            homo_m = [x for x in detail_rows
                      if x["config"] == "homo_medium" and x["kernel"] == kernel]
            if homo_l and not homo_l[0]["mapped"]:
                hetero_only.append(kernel)
            elif homo_m and not homo_m[0]["mapped"]:
                if kernel not in hetero_only:
                    hetero_only.append(kernel)

    if hetero_only:
        lines.extend([
            "",
            "## Kernels uniquely mapped by heterogeneous config",
            "",
        ])
        for k in hetero_only:
            domain = KERNEL_PROFILES.get(k, {}).get("domain", "?")
            lines.append(f"- {k} ({domain})")

    lines.extend([
        "",
        "## Provenance",
        f"- Git hash: {summary_rows[0]['git_hash'] if summary_rows else 'N/A'}",
        f"- Timestamp: {summary_rows[0]['timestamp'] if summary_rows else 'N/A'}",
        "- Area budget matched across configurations (same L2, NoC settings)",
        "- Mapping: per-kernel FU availability and PE capacity check",
    ])

    summary_path = analysis_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Analysis written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
