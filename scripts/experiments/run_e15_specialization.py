#!/usr/bin/env python3
"""E15: Core Type Specialization Analysis.

Analyzes how FU demand varies across workload domains to justify
heterogeneous core specialization. Extracts per-kernel op histograms,
computes per-domain FU demand profiles, and determines the optimal
FU mix for each core type role.

Usage:
    python3 scripts/experiments/run_e15_specialization.py
"""

import csv
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.hw_bilevel_common import (
    CORE_ROLE_PROFILES,
    DOMAIN_NAMES,
    KERNEL_PROFILES,
    REPO_ROOT,
    build_kernel_profile,
    classify_kernel_to_role,
    estimate_core_area,
    get_all_kernels,
    get_domain_kernels,
    git_hash,
    make_core_type,
    timestamp_utc,
)
from scripts.dse.dse_config import FU_AREA_TABLE


# Canonical FU/op type mapping for the specialization analysis
OP_TYPE_CATEGORIES = {
    "alu": ["add", "sub", "shift"],
    "mul": ["mul_int", "mul"],
    "fp": ["fmul", "fadd"],
    "mem": ["load", "store"],
    "control": ["cmp"],
}


def categorize_op(op_name: str) -> str:
    """Map an op name to its FU category."""
    op_lower = op_name.lower()
    if any(k in op_lower for k in ("fp", "float", "fadd", "fmul", "fdiv")):
        return "fp"
    if any(k in op_lower for k in ("mul", "div", "rem")):
        return "mul"
    if any(k in op_lower for k in ("load", "store", "mem")):
        return "mem"
    if any(k in op_lower for k in ("cmp", "select", "br", "mux")):
        return "control"
    return "alu"


def main():
    ts = timestamp_utc()
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E15"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E15: Core Type Specialization Analysis")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {ts}")
    print()

    # Part 1: Per-kernel op distribution
    spec_rows = []
    all_kernels = get_all_kernels()

    for kp in all_kernels:
        domain = KERNEL_PROFILES[kp.name]["domain"]
        total_ops = max(1, sum(kp.op_histogram.values()))

        for op_name, count in kp.op_histogram.items():
            category = categorize_op(op_name)
            spec_rows.append({
                "domain": domain,
                "kernel": kp.name,
                "op_type": op_name,
                "op_category": category,
                "op_count": count,
                "op_fraction": round(count / total_ops, 4),
                "git_hash": ghash,
                "timestamp": ts,
            })

    spec_path = out_dir / "specialization.csv"
    spec_fields = [
        "domain", "kernel", "op_type", "op_category",
        "op_count", "op_fraction", "git_hash", "timestamp",
    ]
    with open(spec_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=spec_fields)
        writer.writeheader()
        writer.writerows(spec_rows)
    print(f"  Wrote {len(spec_rows)} rows to {spec_path}")

    # Part 2: Per-domain aggregate FU demand profile
    print("\n  Per-domain FU demand profiles:")
    domain_profiles = {}
    for domain in DOMAIN_NAMES:
        kernels = get_domain_kernels(domain)
        cat_totals = {"alu": 0, "mul": 0, "fp": 0, "mem": 0, "control": 0}
        grand_total = 0

        for kp in kernels:
            for op_name, count in kp.op_histogram.items():
                cat = categorize_op(op_name)
                cat_totals[cat] = cat_totals.get(cat, 0) + count
                grand_total += count

        cat_fracs = {}
        for cat, total in cat_totals.items():
            cat_fracs[cat] = total / max(1, grand_total)

        domain_profiles[domain] = cat_fracs

        fracs_str = ", ".join(f"{c}={v:.1%}" for c, v in cat_fracs.items())
        print(f"    {domain}: {fracs_str}")

    # Part 3: Optimal FU mix per core type role
    print("\n  Optimal FU mix per core type:")
    fu_mix_rows = []

    # Assign kernels to roles
    role_assignments = {"ctrl": [], "gp": [], "dsp": [], "ai": []}
    for kp in all_kernels:
        role = classify_kernel_to_role(kp)
        role_assignments[role].append(kp)

    for role, kernels in role_assignments.items():
        if not kernels:
            continue

        # Compute aggregate FU demand for this role
        cat_demand = {"alu": 0, "mul": 0, "fp": 0, "mem": 0, "control": 0}
        for kp in kernels:
            for op_name, count in kp.op_histogram.items():
                cat = categorize_op(op_name)
                cat_demand[cat] = cat_demand.get(cat, 0) + count

        # Convert to FU counts (target II = 4)
        target_ii = 4
        total_demand = max(1, sum(cat_demand.values()))

        # Map categories to FU types
        fu_types = {"alu": "alu", "mul": "mul", "fp": "fp", "mem": "mem", "control": "alu"}
        fu_demand = {}
        for cat, count in cat_demand.items():
            fu_type = fu_types[cat]
            fu_demand[fu_type] = fu_demand.get(fu_type, 0) + count

        print(f"    {role} ({len(kernels)} kernels):")
        for fu_type in ["alu", "mul", "fp", "mem"]:
            demand = fu_demand.get(fu_type, 0)
            optimal_count = max(1 if fu_type in ("alu", "mem") else 0,
                              math.ceil(demand / target_ii / len(kernels)))
            actual_count = getattr(make_core_type(role),
                                  f"fu_{fu_type}_count")
            area_contribution = optimal_count * FU_AREA_TABLE.get(fu_type, 2000)

            # List which kernels need this FU type
            kernel_names = [
                kp.name for kp in kernels
                if any(categorize_op(op) == fu_type or
                       (fu_type == "alu" and categorize_op(op) == "control")
                       for op, c in kp.op_histogram.items() if c > 0)
            ]

            fu_mix_rows.append({
                "core_type": role,
                "fu_type": fu_type,
                "optimal_count": optimal_count,
                "default_count": actual_count,
                "demand_ops": demand,
                "used_by_kernels": len(kernel_names),
                "kernel_names": ";".join(kernel_names[:5]),
                "area_contribution": round(area_contribution, 1),
                "git_hash": ghash,
                "timestamp": ts,
            })

            print(f"      {fu_type}: optimal={optimal_count}, "
                  f"default={actual_count}, "
                  f"area={area_contribution:,.0f} um^2, "
                  f"used_by={len(kernel_names)} kernels")

    fu_mix_path = out_dir / "optimal_fu_mix.csv"
    fu_fields = [
        "core_type", "fu_type", "optimal_count", "default_count",
        "demand_ops", "used_by_kernels", "kernel_names",
        "area_contribution", "git_hash", "timestamp",
    ]
    with open(fu_mix_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fu_fields)
        writer.writeheader()
        writer.writerows(fu_mix_rows)
    print(f"\n  Wrote {len(fu_mix_rows)} rows to {fu_mix_path}")

    # Write analysis
    write_analysis(spec_rows, fu_mix_rows, domain_profiles, role_assignments)

    return 0


def write_analysis(spec_rows, fu_mix_rows, domain_profiles, role_assignments):
    """Write analysis summary for E15."""
    analysis_dir = REPO_ROOT / "analysis" / "E15"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# E15: Core Type Specialization Summary",
        "",
        "## Per-Domain FU Demand Profile",
        "",
        "| Domain | ALU | MUL | FP | MEM | Control |",
        "|--------|-----|-----|-----|-----|---------|",
    ]

    for domain in DOMAIN_NAMES:
        p = domain_profiles.get(domain, {})
        lines.append(
            f"| {domain} | {p.get('alu', 0):.1%} | {p.get('mul', 0):.1%} "
            f"| {p.get('fp', 0):.1%} | {p.get('mem', 0):.1%} "
            f"| {p.get('control', 0):.1%} |"
        )

    # Observations
    lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    # Identify domain characteristics
    for domain in DOMAIN_NAMES:
        p = domain_profiles.get(domain, {})
        dominant = max(p, key=p.get)
        frac = p[dominant]
        lines.append(f"- **{domain}**: Dominated by {dominant} ({frac:.0%})")

    lines.extend([
        "",
        "## Core Type Role Assignment",
        "",
    ])

    for role, kernels in role_assignments.items():
        knames = [kp.name for kp in kernels]
        if knames:
            lines.append(f"- **{role}** ({len(knames)} kernels): {', '.join(knames)}")

    lines.extend([
        "",
        "## Specialization Axis Analysis",
        "",
        "The primary axes of specialization are:",
        "",
    ])

    # Check if data-type (INT vs FP) is the right axis
    fp_domains = [d for d in DOMAIN_NAMES if domain_profiles.get(d, {}).get("fp", 0) > 0.2]
    ctrl_domains = [d for d in DOMAIN_NAMES if domain_profiles.get(d, {}).get("control", 0) > 0.15]
    mul_domains = [d for d in DOMAIN_NAMES if domain_profiles.get(d, {}).get("mul", 0) > 0.15]

    lines.append(f"1. **FP-intensive**: {', '.join(fp_domains) if fp_domains else 'None'}")
    lines.append(f"2. **Control-heavy**: {', '.join(ctrl_domains) if ctrl_domains else 'None'}")
    lines.append(f"3. **MUL-intensive**: {', '.join(mul_domains) if mul_domains else 'None'}")
    lines.append(f"4. **Balanced**: domains not in above categories")

    lines.extend([
        "",
        "## Provenance",
        f"- Git hash: {spec_rows[0]['git_hash'] if spec_rows else 'N/A'}",
        f"- Timestamp: {spec_rows[0]['timestamp'] if spec_rows else 'N/A'}",
        "- Op histograms from kernel profile database (derived from C source structure)",
        "- FU counts computed with target II=4 per kernel",
    ])

    summary_path = analysis_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Analysis written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
