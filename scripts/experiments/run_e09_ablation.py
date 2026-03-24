#!/usr/bin/env python3
"""E09: Contract Ablation -- measure how removing contract field categories
affects compilation quality.

Five ablation levels:
  (a) Full TDC (all fields)
  (b) No permissions (may_fuse/replicate/retile forced false)
  (c) No visibility (visibility removed, inferred as EXTERNAL_DRAM)
  (d) Dependency-only (only producer/consumer names, no contract fields)
  (e) No contracts (greedy assignment, no TDC information)

Usage:
    python3 scripts/experiments/run_e09_ablation.py
"""

import csv
import copy
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DOMAINS = {
    "ai_llm": {
        "tdg": "benchmarks/tapestry/ai_llm/tdg_transformer.py",
        "label": "Transformer Layer",
    },
    "dsp_ofdm": {
        "tdg": "benchmarks/tapestry/dsp_ofdm/tdg_ofdm.py",
        "label": "OFDM Receiver",
    },
    "arvr_stereo": {
        "tdg": "benchmarks/tapestry/arvr_stereo/tdg_stereo.py",
        "label": "Stereo Vision",
    },
    "robotics_vio": {
        "tdg": "benchmarks/tapestry/robotics_vio/tdg_vio.py",
        "label": "VIO Pipeline",
    },
    "graph_analytics": {
        "tdg": "benchmarks/tapestry/graph_analytics/tdg_graph.py",
        "label": "Graph Analytics",
    },
    "zk_stark": {
        "tdg": "benchmarks/tapestry/zk_stark/tdg_stark.py",
        "label": "STARK Proof",
    },
}

ARCH_CONFIG = {
    "label": "2x2 Heterogeneous (2 GP + 2 DSP)",
    "core_types": [
        {"name": "gp", "instances": 2, "mesh": "2x2",
         "has_mul": True, "has_cmp": True, "has_mem": True,
         "fu_budget": 16, "spm_budget": 4096},
        {"name": "dsp", "instances": 2, "mesh": "2x2",
         "has_mul": True, "has_cmp": False, "has_mem": True,
         "fu_budget": 16, "spm_budget": 4096},
    ],
}

ABLATION_LEVELS = [
    "full_tdc",
    "no_permissions",
    "no_visibility",
    "dependency_only",
    "no_contracts",
]


def git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_tdg(tdg_path):
    """Load kernels and contracts from TDG Python descriptor."""
    full_path = REPO_ROOT / tdg_path
    content = full_path.read_text()

    kernels = []
    kernels_match = re.search(r'kernels\s*=\s*\[(.+?)\]', content, re.DOTALL)
    if kernels_match:
        block = kernels_match.group(1)
        for dm in re.finditer(r'\{([^}]+)\}', block, re.DOTALL):
            kernel = {}
            for kv in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', dm.group(1)):
                kernel[kv.group(1)] = kv.group(2)
            kernels.append(kernel)

    contracts = []
    contracts_match = re.search(
        r'contracts\s*=\s*\[(.+?)\]\s*$', content, re.DOTALL | re.MULTILINE)
    if contracts_match:
        block = contracts_match.group(1)
        for dm in re.finditer(r'\{([^}]+)\}', block, re.DOTALL):
            contract = {}
            for kv in re.finditer(
                    r'"(\w+)"\s*:\s*("([^"]*)"|([\d.]+)|(True|False)|'
                    r'\[([^\]]*)\])', dm.group(1)):
                key = kv.group(1)
                if kv.group(3) is not None:
                    contract[key] = kv.group(3)
                elif kv.group(4) is not None:
                    val = kv.group(4)
                    contract[key] = int(val) if '.' not in val else float(val)
                elif kv.group(5) is not None:
                    contract[key] = kv.group(5) == "True"
                elif kv.group(6) is not None:
                    items = [int(x.strip()) for x in kv.group(6).split(',')
                             if x.strip()]
                    contract[key] = items
            contracts.append(contract)

    return kernels, contracts


def ablate_contracts(contracts, level):
    """Apply ablation to contracts at the specified level."""
    if level == "full_tdc":
        return copy.deepcopy(contracts)

    if level == "no_permissions":
        ablated = copy.deepcopy(contracts)
        for c in ablated:
            c["may_fuse"] = False
            c["may_replicate"] = False
            c["may_retile"] = False
            c["may_pipeline"] = False
            c["may_reorder"] = False
        return ablated

    if level == "no_visibility":
        ablated = copy.deepcopy(contracts)
        for c in ablated:
            c["visibility"] = "EXTERNAL_DRAM"
            c.pop("double_buffering", None)
        return ablated

    if level == "dependency_only":
        ablated = []
        for c in contracts:
            ablated.append({
                "producer": c.get("producer", ""),
                "consumer": c.get("consumer", ""),
            })
        return ablated

    if level == "no_contracts":
        return []

    return copy.deepcopy(contracts)


def build_cores():
    """Build core instance list from ARCH_CONFIG."""
    cores = []
    for ct in ARCH_CONFIG["core_types"]:
        for i in range(ct["instances"]):
            cores.append({
                "name": f"{ct['name']}_{i}",
                "type": ct["name"],
                "has_mul": ct["has_mul"],
                "has_cmp": ct["has_cmp"],
                "fu_budget": ct["fu_budget"],
                "spm_budget": ct["spm_budget"],
            })
    return cores


def kernel_type_cycles(ktype):
    type_cycles = {
        "matmul": 32768, "batched_matmul": 65536, "elementwise": 4096,
        "reduction": 8192, "fft": 16384, "interpolation": 8192,
        "demapping": 4096, "decoder": 32768, "check": 2048,
        "feature_detect": 16384, "matching": 32768, "optimization": 16384,
        "filter": 8192, "sequential_accum": 4096, "stencil_2d": 16384,
        "patch_compute": 8192, "brute_force_search": 32768,
        "linear_algebra": 8192, "frontier_based": 16384,
        "spmv_iterative": 32768, "set_intersection": 16384,
        "neighbor_vote": 8192, "butterfly_transform": 32768,
        "bucket_accumulate": 65536, "permutation_sponge": 16384,
        "horner_batch": 8192, "linear_combination": 4096,
    }
    return type_cycles.get(ktype, 8192)


def needs_mul(ktype):
    return ktype in {"matmul", "batched_matmul", "fft",
                     "butterfly_transform", "bucket_accumulate",
                     "spmv_iterative"}


def needs_cmp(ktype):
    return ktype in {"feature_detect", "matching", "brute_force_search",
                     "set_intersection", "frontier_based", "check"}


def kernel_compatible(ktype, core):
    if needs_mul(ktype) and not core["has_mul"]:
        return False
    if needs_cmp(ktype) and not core["has_cmp"]:
        return False
    return True


def compute_objective(assignment, kernels, contracts, cores):
    """Compute mapping objective with contract-aware communication cost."""
    core_load = {c["name"]: 0 for c in cores}
    for k in kernels:
        if k["name"] in assignment:
            core_name = assignment[k["name"]]["name"]
            core_load[core_name] += kernel_type_cycles(k.get("type", ""))

    max_load = max(core_load.values()) if core_load else 0

    # Communication penalty depends on contract fields.
    comm_penalty = 0
    for c in contracts:
        prod = c.get("producer", "")
        cons = c.get("consumer", "")
        if prod in assignment and cons in assignment:
            if assignment[prod]["name"] != assignment[cons]["name"]:
                rate = c.get("production_rate", 1024)
                visibility = c.get("visibility", "EXTERNAL_DRAM")

                # Visibility affects transfer cost.
                vis_multiplier = 1.0
                if visibility == "LOCAL_SPM":
                    vis_multiplier = 0.5
                elif visibility == "SHARED_L2":
                    vis_multiplier = 1.0
                elif visibility in ("EXTERNAL_DRAM", "GLOBAL_MEM"):
                    vis_multiplier = 4.0

                # Double buffering reduces effective latency.
                db = c.get("double_buffering", False)
                db_factor = 0.6 if db else 1.0

                comm_penalty += rate * 4 * vis_multiplier * db_factor / 8.0

    return max_load + 0.5 * comm_penalty


def compile_with_contracts(kernels, contracts, cores, level, max_iter=10):
    """Run compilation simulation with ablated contracts."""
    start = time.time()
    accumulated_cuts = set()
    best_assignment = None
    best_objective = float("inf")
    iterations = 0

    # For "no_contracts", skip iteration loop and do greedy.
    if level == "no_contracts":
        assignment = {}
        core_idx = 0
        for k in kernels:
            ktype = k.get("type", "")
            assigned = False
            for attempt in range(len(cores)):
                ci = (core_idx + attempt) % len(cores)
                if kernel_compatible(ktype, cores[ci]):
                    assignment[k["name"]] = cores[ci]
                    core_idx = (ci + 1) % len(cores)
                    assigned = True
                    break
            if not assigned:
                assignment[k["name"]] = cores[core_idx % len(cores)]
                core_idx = (core_idx + 1) % len(cores)

        elapsed_ms = (time.time() - start) * 1000
        mapped = len(assignment)
        obj = compute_objective(assignment, kernels, contracts, cores)
        return {
            "success_rate": round(mapped / len(kernels), 4) if kernels else 0,
            "avg_ii": round(
                sum(max(1, kernel_type_cycles(k.get("type", "")) // 1024)
                    for k in kernels) / len(kernels), 2) if kernels else 0,
            "total_cost": round(obj, 2),
            "compile_time_ms": round(elapsed_ms, 1),
        }

    # Bilevel compilation with ablated contracts.
    for iteration in range(1, max_iter + 1):
        iterations = iteration

        sorted_kernels = sorted(
            kernels, key=lambda k: kernel_type_cycles(k.get("type", "")),
            reverse=True)

        core_load = {c["name"]: 0 for c in cores}
        assignment = {}
        feasible = True

        for k in sorted_kernels:
            ktype = k.get("type", "")
            best_core = None
            best_load = float("inf")

            for c in cores:
                if (k["name"], c["type"]) in accumulated_cuts:
                    continue
                if not kernel_compatible(ktype, c):
                    continue

                # Use contract info for assignment quality.
                locality_bonus = 0
                if level not in ("dependency_only", "no_contracts"):
                    for con in contracts:
                        if con.get("producer") == k["name"]:
                            vis = con.get("visibility", "EXTERNAL_DRAM")
                            if vis == "LOCAL_SPM":
                                locality_bonus -= 500
                        if con.get("consumer") == k["name"]:
                            vis = con.get("visibility", "EXTERNAL_DRAM")
                            if vis == "LOCAL_SPM":
                                locality_bonus -= 500

                adjusted_load = core_load[c["name"]] + locality_bonus
                if adjusted_load < best_load:
                    best_load = adjusted_load
                    best_core = c

            if best_core is None:
                feasible = False
                break

            assignment[k["name"]] = best_core
            core_load[best_core["name"]] += kernel_type_cycles(ktype)

        if not feasible:
            break

        # L2 check.
        core_kernels = {}
        for kname, core in assignment.items():
            cn = core["name"]
            if cn not in core_kernels:
                core_kernels[cn] = []
            core_kernels[cn].append(kname)

        new_cuts = []
        all_mapped = True
        for cn, knames in core_kernels.items():
            core_info = next(c for c in cores if c["name"] == cn)
            mesh_pes = core_info["fu_budget"]
            config_mem_cap = mesh_pes * 8
            config_demand = sum(
                max(2, kernel_type_cycles(
                    next(k.get("type", "") for k in kernels
                         if k["name"] == kn)) // 8192) * 4
                for kn in knames
            )
            type_fail = False
            for kn in knames:
                ktype = next(k.get("type", "") for k in kernels
                             if k["name"] == kn)
                if not kernel_compatible(ktype, core_info):
                    type_fail = True
                    break
            if type_fail or config_demand > config_mem_cap:
                all_mapped = False
                worst = max(knames, key=lambda kn: kernel_type_cycles(
                    next(k.get("type", "") for k in kernels
                         if k["name"] == kn)))
                new_cuts.append((worst, core_info["type"]))
                accumulated_cuts.add((worst, core_info["type"]))

        if all_mapped:
            obj = compute_objective(assignment, kernels, contracts, cores)
            if obj < best_objective:
                best_objective = obj
                best_assignment = assignment.copy()
            if not new_cuts:
                break

    elapsed_ms = (time.time() - start) * 1000
    mapped = len(best_assignment) if best_assignment else 0
    success_rate = mapped / len(kernels) if kernels else 0

    avg_ii = 0
    if kernels:
        avg_ii = sum(
            max(1, kernel_type_cycles(k.get("type", "")) // 1024)
            for k in kernels) / len(kernels)

    return {
        "success_rate": round(success_rate, 4),
        "avg_ii": round(avg_ii, 2),
        "total_cost": round(best_objective, 2) if best_assignment else -1,
        "compile_time_ms": round(elapsed_ms, 1),
    }


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E09"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation.csv"

    print("E09: Contract Ablation")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print(f"  arch: {ARCH_CONFIG['label']}")
    print()

    cores = build_cores()
    rows = []

    for domain_name, domain_info in DOMAINS.items():
        print(f"  Domain: {domain_name} ({domain_info['label']})")
        kernels, full_contracts = load_tdg(domain_info["tdg"])

        for level in ABLATION_LEVELS:
            ablated = ablate_contracts(full_contracts, level)
            result = compile_with_contracts(
                kernels, ablated, cores, level)

            print(f"    {level:20s}: cost={result['total_cost']:>10.2f}, "
                  f"success={result['success_rate']:.2f}, "
                  f"time={result['compile_time_ms']:.1f}ms")

            rows.append({
                "domain": domain_name,
                "ablation_level": level,
                "mapping_success_rate": result["success_rate"],
                "avg_ii": result["avg_ii"],
                "total_cost": result["total_cost"],
                "compile_time_ms": result["compile_time_ms"],
                "git_hash": ghash,
                "timestamp": timestamp,
            })

    # Write CSV.
    fieldnames = [
        "domain", "ablation_level", "mapping_success_rate", "avg_ii",
        "total_cost", "compile_time_ms",
        "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Normalized cost analysis.
    print("\n--- Normalized Cost (vs full_tdc) ---")
    for domain_name in DOMAINS:
        full_row = next(
            (r for r in rows
             if r["domain"] == domain_name
             and r["ablation_level"] == "full_tdc"), None)
        if not full_row or full_row["total_cost"] <= 0:
            continue
        base_cost = full_row["total_cost"]
        print(f"  {domain_name}:")
        for level in ABLATION_LEVELS:
            row = next(
                (r for r in rows
                 if r["domain"] == domain_name
                 and r["ablation_level"] == level), None)
            if row and row["total_cost"] > 0:
                normalized = row["total_cost"] / base_cost
                print(f"    {level:20s}: {normalized:.3f}x")
            else:
                print(f"    {level:20s}: FAILED")

    # Per-field contribution ranking.
    print("\n--- Field Category Contribution Ranking ---")
    contribution = {}
    for level in ABLATION_LEVELS:
        if level == "full_tdc":
            continue
        deltas = []
        for domain_name in DOMAINS:
            full_row = next(
                (r for r in rows
                 if r["domain"] == domain_name
                 and r["ablation_level"] == "full_tdc"), None)
            level_row = next(
                (r for r in rows
                 if r["domain"] == domain_name
                 and r["ablation_level"] == level), None)
            if full_row and level_row:
                if full_row["total_cost"] > 0 and level_row["total_cost"] > 0:
                    delta = (level_row["total_cost"] - full_row["total_cost"]
                             ) / full_row["total_cost"]
                    deltas.append(delta)
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            contribution[level] = avg_delta

    for level, delta in sorted(contribution.items(), key=lambda x: -x[1]):
        print(f"  {level:20s}: avg cost increase = {delta:+.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
