#!/usr/bin/env python3
"""E08: Hierarchical vs Flat Compiler -- compare bilevel BendersDriver
against greedy round-robin and random assignment baselines.

Usage:
    python3 scripts/experiments/run_e08_compiler.py
"""

import csv
import math
import os
import random
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

# 2x2 heterogeneous architecture for fair comparison.
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

RANDOM_SEEDS = [42, 137, 256, 512, 1024]


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
                    r'"(\w+)"\s*:\s*("([^"]*)"|([\d.]+)|(True|False))',
                    dm.group(1)):
                key = kv.group(1)
                if kv.group(3) is not None:
                    contract[key] = kv.group(3)
                elif kv.group(4) is not None:
                    val = kv.group(4)
                    contract[key] = int(val) if '.' not in val else float(val)
                elif kv.group(5) is not None:
                    contract[key] = kv.group(5) == "True"
            contracts.append(contract)

    return kernels, contracts


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
    """Estimate cycles from kernel type."""
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
    """Check if a kernel type is compatible with a core."""
    if needs_mul(ktype) and not core["has_mul"]:
        return False
    if needs_cmp(ktype) and not core["has_cmp"]:
        return False
    return True


def compute_objective(assignment, kernels, contracts, cores):
    """Compute mapping objective: max core load + communication penalty."""
    core_load = {c["name"]: 0 for c in cores}
    for k in kernels:
        if k["name"] in assignment:
            core_name = assignment[k["name"]]["name"]
            core_load[core_name] += kernel_type_cycles(k.get("type", ""))

    max_load = max(core_load.values()) if core_load else 0

    comm_penalty = 0
    for c in contracts:
        prod = c.get("producer", "")
        cons = c.get("consumer", "")
        if prod in assignment and cons in assignment:
            if assignment[prod]["name"] != assignment[cons]["name"]:
                rate = c.get("production_rate", 1024)
                comm_penalty += rate * 4 / 8.0

    return max_load + 0.5 * comm_penalty


def compute_avg_ii(assignment, kernels):
    """Estimate average II across mapped kernels."""
    ii_sum = 0
    count = 0
    for k in kernels:
        if k["name"] in assignment:
            cycles = kernel_type_cycles(k.get("type", ""))
            ii = max(1, cycles // 1024)
            ii_sum += ii
            count += 1
    return ii_sum / count if count > 0 else 0


def compile_hierarchical(kernels, contracts, cores, max_iter=10):
    """Simulate bilevel BendersDriver compilation."""
    start = time.time()
    accumulated_cuts = set()
    best_assignment = None
    best_objective = float("inf")
    iterations = 0
    success_rate = 0.0

    for iteration in range(1, max_iter + 1):
        iterations = iteration

        # L1: greedy assignment respecting cuts.
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
                if core_load[c["name"]] < best_load:
                    best_load = core_load[c["name"]]
                    best_core = c

            if best_core is None:
                feasible = False
                break

            assignment[k["name"]] = best_core
            core_load[best_core["name"]] += kernel_type_cycles(ktype)

        if not feasible:
            break

        # L2: check per-core feasibility.
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
            # L2 feasibility: in BATCH_SEQUENTIAL, each kernel runs
            # alone on the mesh. The limit is config memory (must store
            # configs for all kernels) and individual kernel complexity.
            mesh_pes = core_info["fu_budget"]
            config_mem_cap = mesh_pes * 8  # config words
            config_demand = sum(
                max(2, kernel_type_cycles(
                    next(k.get("type", "") for k in kernels
                         if k["name"] == kn)) // 8192) * 4
                for kn in knames
            )
            # Also check type compatibility per kernel.
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

    return {
        "success_rate": round(success_rate, 4),
        "total_cost": round(best_objective, 2) if best_assignment else -1,
        "iterations": iterations,
        "compile_time_ms": round(elapsed_ms, 1),
        "assignment": best_assignment,
        "avg_ii": round(compute_avg_ii(best_assignment or {},
                                       kernels), 2),
    }


def compile_greedy(kernels, contracts, cores):
    """Greedy round-robin assignment: modulo assignment by kernel order."""
    start = time.time()
    assignment = {}
    core_idx = 0
    num_cores = len(cores)

    for k in kernels:
        ktype = k.get("type", "")
        # Find next compatible core in round-robin order.
        found = False
        for attempt in range(num_cores):
            ci = (core_idx + attempt) % num_cores
            if kernel_compatible(ktype, cores[ci]):
                assignment[k["name"]] = cores[ci]
                core_idx = (ci + 1) % num_cores
                found = True
                break
        if not found:
            # Try any core (will produce bad quality).
            assignment[k["name"]] = cores[core_idx % num_cores]
            core_idx = (core_idx + 1) % num_cores

    elapsed_ms = (time.time() - start) * 1000
    mapped = len(assignment)
    success_rate = mapped / len(kernels) if kernels else 0
    obj = compute_objective(assignment, kernels, contracts, cores)

    return {
        "success_rate": round(success_rate, 4),
        "total_cost": round(obj, 2),
        "iterations": 1,
        "compile_time_ms": round(elapsed_ms, 1),
        "assignment": assignment,
        "avg_ii": round(compute_avg_ii(assignment, kernels), 2),
    }


def compile_random(kernels, contracts, cores, seed):
    """Random assignment with given seed."""
    start = time.time()
    rng = random.Random(seed)
    assignment = {}

    for k in kernels:
        ktype = k.get("type", "")
        compatible = [c for c in cores if kernel_compatible(ktype, c)]
        if compatible:
            assignment[k["name"]] = rng.choice(compatible)
        else:
            assignment[k["name"]] = rng.choice(cores)

    elapsed_ms = (time.time() - start) * 1000
    mapped = len(assignment)
    success_rate = mapped / len(kernels) if kernels else 0
    obj = compute_objective(assignment, kernels, contracts, cores)

    return {
        "success_rate": round(success_rate, 4),
        "total_cost": round(obj, 2),
        "iterations": 1,
        "compile_time_ms": round(elapsed_ms, 1),
        "assignment": assignment,
        "avg_ii": round(compute_avg_ii(assignment, kernels), 2),
    }


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E08"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "compiler_comparison.csv"

    print("E08: Hierarchical vs Flat Compiler")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print(f"  arch: {ARCH_CONFIG['label']}")
    print()

    cores = build_cores()
    rows = []

    for domain_name, domain_info in DOMAINS.items():
        print(f"  Domain: {domain_name} ({domain_info['label']})")
        kernels, contracts = load_tdg(domain_info["tdg"])
        print(f"    kernels={len(kernels)}, contracts={len(contracts)}")

        # Hierarchical bilevel.
        result_h = compile_hierarchical(kernels, contracts, cores)
        print(f"    hierarchical: cost={result_h['total_cost']}, "
              f"iters={result_h['iterations']}, "
              f"time={result_h['compile_time_ms']:.1f}ms")
        rows.append({
            "domain": domain_name,
            "compiler": "hierarchical",
            "mapping_success_rate": result_h["success_rate"],
            "avg_ii": result_h["avg_ii"],
            "total_cost": result_h["total_cost"],
            "iterations": result_h["iterations"],
            "compile_time_ms": result_h["compile_time_ms"],
            "git_hash": ghash,
            "timestamp": timestamp,
        })

        # Greedy round-robin.
        result_g = compile_greedy(kernels, contracts, cores)
        print(f"    greedy:       cost={result_g['total_cost']}, "
              f"time={result_g['compile_time_ms']:.1f}ms")
        rows.append({
            "domain": domain_name,
            "compiler": "greedy",
            "mapping_success_rate": result_g["success_rate"],
            "avg_ii": result_g["avg_ii"],
            "total_cost": result_g["total_cost"],
            "iterations": result_g["iterations"],
            "compile_time_ms": result_g["compile_time_ms"],
            "git_hash": ghash,
            "timestamp": timestamp,
        })

        # Random assignment (best of 5 seeds).
        best_random = None
        all_random_costs = []
        for seed in RANDOM_SEEDS:
            result_r = compile_random(kernels, contracts, cores, seed)
            all_random_costs.append(result_r["total_cost"])
            if best_random is None or result_r["total_cost"] < best_random["total_cost"]:
                best_random = result_r

        print(f"    random(best): cost={best_random['total_cost']}, "
              f"costs_all={all_random_costs}")
        rows.append({
            "domain": domain_name,
            "compiler": "random",
            "mapping_success_rate": best_random["success_rate"],
            "avg_ii": best_random["avg_ii"],
            "total_cost": best_random["total_cost"],
            "iterations": best_random["iterations"],
            "compile_time_ms": best_random["compile_time_ms"],
            "git_hash": ghash,
            "timestamp": timestamp,
        })

        # Record all individual random runs too.
        for i, seed in enumerate(RANDOM_SEEDS):
            result_r = compile_random(kernels, contracts, cores, seed)
            rows.append({
                "domain": domain_name,
                "compiler": f"random_seed{seed}",
                "mapping_success_rate": result_r["success_rate"],
                "avg_ii": result_r["avg_ii"],
                "total_cost": result_r["total_cost"],
                "iterations": result_r["iterations"],
                "compile_time_ms": result_r["compile_time_ms"],
                "git_hash": ghash,
                "timestamp": timestamp,
            })

    # Write CSV.
    fieldnames = [
        "domain", "compiler", "mapping_success_rate", "avg_ii",
        "total_cost", "iterations", "compile_time_ms",
        "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Win/loss summary.
    print("\n--- Win/Loss Summary ---")
    for domain_name in DOMAINS:
        dr = [r for r in rows if r["domain"] == domain_name
              and r["compiler"] in ("hierarchical", "greedy", "random")]
        if len(dr) == 3:
            h_cost = dr[0]["total_cost"]
            g_cost = dr[1]["total_cost"]
            r_cost = dr[2]["total_cost"]
            winner = "hierarchical"
            if g_cost < h_cost and g_cost < r_cost:
                winner = "greedy"
            elif r_cost < h_cost and r_cost < g_cost:
                winner = "random"
            print(f"  {domain_name}: winner={winner} "
                  f"(h={h_cost}, g={g_cost}, r={r_cost})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
