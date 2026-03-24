#!/usr/bin/env python3
"""E12: TDC Hardware Pruning Effectiveness.

Measures what fraction of the hardware design space is pruned by TDC-derived
constraints before any evaluation. For each domain, enumerates a grid of
system-level design candidates and checks which satisfy TDC bounds
(min NoC BW, min L2 size, min core types, min total cores).

Outputs per-domain pruning statistics and a breakdown of which constraint
eliminates the most candidates.

Usage:
    python3 scripts/experiments/run_e12_tdc_pruning.py
"""

import csv
import itertools
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.hw_bilevel_common import (
    DOMAIN_NAMES,
    REPO_ROOT,
    build_workload_for_domain,
    git_hash,
    timestamp_utc,
)
from scripts.dse.proxy_model import WorkloadProfile

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class TDCBounds:
    """Hard lower bounds derived from TDC contracts."""
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
    base_flit_width_bytes: int = 32,
) -> TDCBounds:
    """Derive hard lower bounds from TDC contracts."""
    bounds = TDCBounds()

    # NoC bandwidth from inter-kernel data rates
    total_bandwidth_bytes = 0.0
    for contract in workload.contracts:
        volume = contract.production_rate * contract.element_size_bytes
        total_bandwidth_bytes += volume
    bw_multiplier = total_bandwidth_bytes / max(1, base_flit_width_bytes)
    bounds.min_noc_bandwidth = max(1.0, min(4.0, bw_multiplier))

    # L2 size from shared data volume
    total_shared_volume = 0.0
    for contract in workload.contracts:
        volume = contract.production_rate * contract.element_size_bytes * 2
        total_shared_volume += volume
    bounds.min_l2_size_kb = max(64.0, total_shared_volume / 1024.0)

    # Core type diversity from FU categories
    fu_categories = set()
    for kernel in workload.kernels:
        total_ops = max(1, sum(kernel.op_histogram.values()))
        fp_frac = 0.0
        mem_frac = 0.0
        ctrl_frac = 0.0
        for op_name, count in kernel.op_histogram.items():
            op_lower = op_name.lower()
            if any(k in op_lower for k in ("fp", "float", "fadd", "fmul")):
                fp_frac += count
            elif any(k in op_lower for k in ("load", "store", "mem")):
                mem_frac += count
            elif any(k in op_lower for k in ("cmp", "select", "br")):
                ctrl_frac += count
        fp_frac /= total_ops
        mem_frac /= total_ops
        ctrl_frac /= total_ops
        if fp_frac > 0.3:
            fu_categories.add("fp_heavy")
        elif mem_frac > 0.4:
            fu_categories.add("memory_heavy")
        elif ctrl_frac > 0.3:
            fu_categories.add("control_heavy")
        else:
            fu_categories.add("balanced")
    bounds.min_core_types = max(1, len(fu_categories))

    # Total core instances from kernel count
    bounds.min_total_cores = max(1, len(workload.kernels))

    return bounds


# Design space grid parameters (system-level)
CORE_TYPE_COUNTS = [1, 2, 3, 4, 5]
CORES_PER_TYPE = [1, 2, 3, 4]
NOC_BANDWIDTHS = [1, 2, 3, 4]
L2_SIZES_KB = [64, 128, 256, 512, 1024]
NOC_TOPOLOGIES = ["mesh", "ring", "hierarchical"]


def enumerate_candidates():
    """Enumerate the full design space grid."""
    candidates = []
    for n_types in CORE_TYPE_COUNTS:
        for cores_per in CORES_PER_TYPE:
            for noc_bw in NOC_BANDWIDTHS:
                for l2_kb in L2_SIZES_KB:
                    for topo in NOC_TOPOLOGIES:
                        candidates.append({
                            "core_type_count": n_types,
                            "cores_per_type": cores_per,
                            "total_cores": n_types * cores_per,
                            "noc_bandwidth": noc_bw,
                            "l2_size_kb": l2_kb,
                            "noc_topology": topo,
                        })
    return candidates


def check_tdc_feasibility(candidate: dict, bounds: TDCBounds):
    """Check a candidate against TDC bounds. Returns dict of constraint violations."""
    violations = {}
    if candidate["noc_bandwidth"] < bounds.min_noc_bandwidth:
        violations["noc_bandwidth"] = True
    if candidate["l2_size_kb"] < bounds.min_l2_size_kb:
        violations["l2_size_kb"] = True
    if candidate["core_type_count"] < bounds.min_core_types:
        violations["core_type_count"] = True
    if candidate["total_cores"] < bounds.min_total_cores:
        violations["total_cores"] = True
    return violations


def main():
    ts = timestamp_utc()
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E12"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E12: TDC Hardware Pruning Effectiveness")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {ts}")
    print()

    candidates = enumerate_candidates()
    total_candidates = len(candidates)
    print(f"  Total design candidates: {total_candidates}")
    print()

    summary_rows = []
    breakdown_rows = []

    for domain in DOMAIN_NAMES:
        workload = build_workload_for_domain(domain)
        bounds = compute_tdc_bounds(workload)

        print(f"  {domain}:")
        print(f"    {bounds.summary()}")

        # Check each candidate
        feasible_count = 0
        constraint_eliminations = {
            "noc_bandwidth": 0,
            "l2_size_kb": 0,
            "core_type_count": 0,
            "total_cores": 0,
        }

        for cand in candidates:
            violations = check_tdc_feasibility(cand, bounds)
            if not violations:
                feasible_count += 1
            else:
                for constraint in violations:
                    constraint_eliminations[constraint] += 1

        pruned_count = total_candidates - feasible_count
        pruned_fraction = pruned_count / total_candidates if total_candidates > 0 else 0

        print(f"    Feasible: {feasible_count}/{total_candidates} "
              f"({feasible_count/total_candidates:.1%})")
        print(f"    Pruned:   {pruned_count}/{total_candidates} "
              f"({pruned_fraction:.1%})")

        # Constraint breakdown
        constraint_str_parts = []
        for constraint, count in sorted(
            constraint_eliminations.items(), key=lambda x: -x[1]
        ):
            frac = count / total_candidates if total_candidates > 0 else 0
            constraint_str_parts.append(f"{constraint}={count}({frac:.0%})")
            breakdown_rows.append({
                "domain": domain,
                "constraint": constraint,
                "candidates_eliminated": count,
                "elimination_fraction": round(frac, 4),
                "git_hash": ghash,
                "timestamp": ts,
            })

        print(f"    Breakdown: {', '.join(constraint_str_parts)}")

        summary_rows.append({
            "domain": domain,
            "total_candidates": total_candidates,
            "tdc_feasible": feasible_count,
            "tdc_pruned": pruned_count,
            "pruned_fraction": round(pruned_fraction, 4),
            "min_noc_bw": round(bounds.min_noc_bandwidth, 2),
            "min_l2_kb": round(bounds.min_l2_size_kb, 0),
            "min_core_types": bounds.min_core_types,
            "min_total_cores": bounds.min_total_cores,
            "num_kernels": len(workload.kernels),
            "num_contracts": len(workload.contracts),
            "git_hash": ghash,
            "timestamp": ts,
        })

    # Write pruning_stats.csv
    stats_path = out_dir / "pruning_stats.csv"
    stats_fields = [
        "domain", "total_candidates", "tdc_feasible", "tdc_pruned",
        "pruned_fraction", "min_noc_bw", "min_l2_kb",
        "min_core_types", "min_total_cores",
        "num_kernels", "num_contracts", "git_hash", "timestamp",
    ]
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} rows to {stats_path}")

    # Write constraint_breakdown.csv
    breakdown_path = out_dir / "constraint_breakdown.csv"
    bd_fields = [
        "domain", "constraint", "candidates_eliminated",
        "elimination_fraction", "git_hash", "timestamp",
    ]
    with open(breakdown_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bd_fields)
        writer.writeheader()
        writer.writerows(breakdown_rows)
    print(f"Wrote {len(breakdown_rows)} rows to {breakdown_path}")

    # Write analysis
    write_analysis(summary_rows, breakdown_rows)

    return 0


def write_analysis(summary_rows, breakdown_rows):
    """Write analysis summary for E12."""
    analysis_dir = REPO_ROOT / "analysis" / "E12"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    avg_pruned = (
        sum(r["pruned_fraction"] for r in summary_rows) / len(summary_rows)
        if summary_rows else 0
    )

    # Find most restrictive constraint across domains
    constraint_totals = {}
    for r in breakdown_rows:
        c = r["constraint"]
        constraint_totals[c] = constraint_totals.get(c, 0) + r["candidates_eliminated"]
    ranked = sorted(constraint_totals.items(), key=lambda x: -x[1])

    lines = [
        "# E12: TDC Pruning Effectiveness Summary",
        "",
        "## Overall Results",
        f"- Average pruning fraction: {avg_pruned:.1%}",
        f"- Most restrictive constraint: {ranked[0][0] if ranked else 'N/A'}",
        "",
        "## Per-Domain Pruning",
        "",
        "| Domain | Feasible | Pruned | Pruned % | Dominant Constraint |",
        "|--------|----------|--------|----------|-------------------|",
    ]

    for r in summary_rows:
        # Find dominant constraint for this domain
        domain_bd = [b for b in breakdown_rows if b["domain"] == r["domain"]]
        if domain_bd:
            dominant = max(domain_bd, key=lambda x: x["candidates_eliminated"])
            dom_name = dominant["constraint"]
        else:
            dom_name = "N/A"

        lines.append(
            f"| {r['domain']} | {r['tdc_feasible']} | {r['tdc_pruned']} "
            f"| {r['pruned_fraction']:.1%} | {dom_name} |"
        )

    lines.extend([
        "",
        "## Constraint Ranking (aggregate across domains)",
        "",
    ])
    for c, total in ranked:
        lines.append(f"- {c}: {total} total eliminations")

    lines.extend([
        "",
        "## Provenance",
        f"- Git hash: {summary_rows[0]['git_hash'] if summary_rows else 'N/A'}",
        f"- Timestamp: {summary_rows[0]['timestamp'] if summary_rows else 'N/A'}",
        f"- Method: Exhaustive enumeration of {summary_rows[0]['total_candidates'] if summary_rows else 0} candidates per domain",
        "- TDC bounds computed from contract bandwidth, L2 volume, FU diversity, kernel count",
    ])

    summary_path = analysis_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Analysis written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
