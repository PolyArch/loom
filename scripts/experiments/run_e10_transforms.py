#!/usr/bin/env python3
"""E10: TDG Transform Effectiveness -- measure how retile and replicate
transforms improve system throughput beyond the base bilevel compiler.

Four configurations:
  (a) No transforms (BendersDriver only)
  (b) Retile only (may_retile=true, others false)
  (c) Replicate only (may_replicate=true, others false)
  (d) All transforms enabled

Usage:
    python3 scripts/experiments/run_e10_transforms.py
"""

import copy
import csv
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

# Heterogeneous architecture matching E08/E09 for consistency.
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

TRANSFORM_CONFIGS = [
    {"name": "no_transforms", "retile": False, "replicate": False},
    {"name": "retile_only", "retile": True, "replicate": False},
    {"name": "replicate_only", "retile": False, "replicate": True},
    {"name": "all_transforms", "retile": True, "replicate": True},
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


def build_cores():
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


def compute_total_cost(assignment, kernels, contracts, cores):
    """Compute mapping objective."""
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
                vis = c.get("visibility", "LOCAL_SPM")
                vis_mult = 0.5 if vis == "LOCAL_SPM" else (
                    1.0 if vis == "SHARED_L2" else 4.0)
                comm_penalty += rate * 4 * vis_mult / 8.0

    return max_load + 0.5 * comm_penalty


def compute_throughput(total_cost):
    """Throughput = inverse of total cost."""
    if total_cost <= 0:
        return 0.0
    return 1.0 / total_cost


def try_retile(contracts, assignment, kernels, cores):
    """Attempt retile transforms on rate-imbalanced edges.

    Returns (modified_contracts, transforms_applied, transform_details).
    """
    modified = copy.deepcopy(contracts)
    transforms = []

    for i, c in enumerate(modified):
        prod = c.get("producer", "")
        cons = c.get("consumer", "")
        rate = c.get("production_rate", 0)

        if rate == 0:
            continue

        # Check if producer/consumer are on different cores.
        if prod in assignment and cons in assignment:
            if assignment[prod]["name"] != assignment[cons]["name"]:
                # Rate imbalance heuristic: retile if rate > 4096.
                if rate > 4096:
                    old_rate = rate
                    modified[i]["production_rate"] = rate // 2
                    if "tile_shape" in modified[i]:
                        ts = modified[i]["tile_shape"]
                        if isinstance(ts, list) and len(ts) > 0:
                            modified[i]["tile_shape"][-1] = max(
                                1, ts[-1] // 2)
                    transforms.append(f"retile:{prod}->{cons} "
                                      f"rate {old_rate}->{rate // 2}")
                elif rate < 64:
                    old_rate = rate
                    modified[i]["production_rate"] = rate * 2
                    transforms.append(f"retile:{prod}->{cons} "
                                      f"rate {old_rate}->{rate * 2}")

    return modified, len(transforms), transforms


def try_replicate(kernels, contracts, assignment, cores):
    """Attempt replication of the bottleneck kernel.

    Returns (modified_kernels, modified_contracts, transforms_applied,
             transform_details).
    """
    # Find bottleneck kernel (highest cost among mapped).
    max_cost = 0
    bottleneck = None
    for k in kernels:
        if k["name"] in assignment:
            cost = kernel_type_cycles(k.get("type", ""))
            if cost > max_cost:
                max_cost = cost
                bottleneck = k

    if bottleneck is None:
        return kernels, contracts, 0, []

    # Check if spare cores available.
    used_cores = set(assignment[k["name"]]["name"]
                     for k in kernels if k["name"] in assignment)
    all_core_names = set(c["name"] for c in cores)
    spare = all_core_names - used_cores

    if not spare:
        return kernels, contracts, 0, []

    # Replicate: create a copy of the bottleneck kernel.
    mod_kernels = copy.deepcopy(kernels)
    mod_contracts = copy.deepcopy(contracts)

    replica_name = bottleneck["name"] + "_replica"
    replica = copy.deepcopy(bottleneck)
    replica["name"] = replica_name
    mod_kernels.append(replica)

    # Split contracts: producer edges from bottleneck get duplicated.
    new_contracts = []
    for c in mod_contracts:
        if c.get("producer") == bottleneck["name"]:
            new_c = copy.deepcopy(c)
            new_c["producer"] = replica_name
            rate = c.get("production_rate", 0)
            if rate > 0:
                c["production_rate"] = rate // 2 + (rate % 2)
                new_c["production_rate"] = rate // 2
            new_contracts.append(new_c)
        if c.get("consumer") == bottleneck["name"]:
            new_c = copy.deepcopy(c)
            new_c["consumer"] = replica_name
            rate = c.get("production_rate", 0)
            if rate > 0:
                c["production_rate"] = rate // 2 + (rate % 2)
                new_c["production_rate"] = rate // 2
            new_contracts.append(new_c)

    mod_contracts.extend(new_contracts)

    transform_desc = [f"replicate:{bottleneck['name']}->{replica_name}"]
    return mod_kernels, mod_contracts, 1, transform_desc


def bilevel_compile(kernels, contracts, cores, max_iter=10):
    """Run bilevel BendersDriver compilation (no transforms)."""
    accumulated_cuts = set()
    best_assignment = None
    best_objective = float("inf")
    iterations = 0

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
            obj = compute_total_cost(assignment, kernels, contracts, cores)
            if obj < best_objective:
                best_objective = obj
                best_assignment = assignment.copy()
            if not new_cuts:
                break

    return best_assignment, best_objective, iterations


def run_with_transforms(kernels, contracts, cores, config):
    """Run optimization loop with specified transform configuration."""
    start = time.time()
    transforms_applied = 0
    transform_types = []

    # First: base bilevel compile.
    assignment, base_cost, iterations = bilevel_compile(
        kernels, contracts, cores)

    if assignment is None:
        elapsed = (time.time() - start) * 1000
        return {
            "throughput": 0.0,
            "total_cost": -1,
            "iterations": iterations,
            "transforms_applied": 0,
            "transform_types": "",
            "compile_time_ms": round(elapsed, 1),
        }

    current_cost = base_cost
    current_kernels = copy.deepcopy(kernels)
    current_contracts = copy.deepcopy(contracts)

    # Optimization loop: try transforms.
    max_opt_iter = 5
    for opt_iter in range(max_opt_iter):
        improved = False

        # Try retile if enabled.
        if config["retile"]:
            new_contracts, n_retile, retile_details = try_retile(
                current_contracts, assignment, current_kernels, cores)
            if n_retile > 0:
                # Re-compile with retiled contracts.
                new_assignment, new_cost, _ = bilevel_compile(
                    current_kernels, new_contracts, cores)
                if new_assignment and new_cost < current_cost * 0.99:
                    current_cost = new_cost
                    current_contracts = new_contracts
                    assignment = new_assignment
                    transforms_applied += n_retile
                    transform_types.extend(retile_details)
                    improved = True

        # Try replicate if enabled.
        if config["replicate"] and not improved:
            new_kernels, new_contracts, n_rep, rep_details = try_replicate(
                current_kernels, current_contracts, assignment, cores)
            if n_rep > 0:
                new_assignment, new_cost, _ = bilevel_compile(
                    new_kernels, new_contracts, cores)
                if new_assignment and new_cost < current_cost * 0.99:
                    current_cost = new_cost
                    current_kernels = new_kernels
                    current_contracts = new_contracts
                    assignment = new_assignment
                    transforms_applied += n_rep
                    transform_types.extend(rep_details)
                    improved = True

        if not improved:
            break

    elapsed = (time.time() - start) * 1000
    throughput = compute_throughput(current_cost)

    return {
        "throughput": round(throughput, 8),
        "total_cost": round(current_cost, 2),
        "iterations": iterations,
        "transforms_applied": transforms_applied,
        "transform_types": "; ".join(transform_types) if transform_types else "",
        "compile_time_ms": round(elapsed, 1),
    }


def compute_area_estimate(arch_config):
    """Estimate fixed hardware area from architecture config.
    This is the same across all software configs (same hardware).
    """
    total_pes = 0
    for ct in arch_config["core_types"]:
        mesh = ct["mesh"]
        rows, cols = int(mesh.split("x")[0]), int(mesh.split("x")[1])
        total_pes += rows * cols * ct["instances"]
    # Area estimate: 1000 um^2 per PE (rough CGRA estimate).
    return total_pes * 1000


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E10"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "transform_effect.csv"

    print("E10: TDG Transform Effectiveness")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print(f"  arch: {ARCH_CONFIG['label']}")
    print()

    cores = build_cores()
    area = compute_area_estimate(ARCH_CONFIG)
    rows = []

    for domain_name, domain_info in DOMAINS.items():
        print(f"  Domain: {domain_name} ({domain_info['label']})")
        kernels, contracts = load_tdg(domain_info["tdg"])

        for config in TRANSFORM_CONFIGS:
            result = run_with_transforms(kernels, contracts, cores, config)

            print(f"    {config['name']:20s}: "
                  f"cost={result['total_cost']:>10.2f}, "
                  f"tp={result['throughput']:.6f}, "
                  f"transforms={result['transforms_applied']}, "
                  f"time={result['compile_time_ms']:.1f}ms")
            if result["transform_types"]:
                print(f"      transforms: {result['transform_types']}")

            rows.append({
                "domain": domain_name,
                "transform_config": config["name"],
                "throughput": result["throughput"],
                "area": area,
                "iterations": result["iterations"],
                "transforms_applied": result["transforms_applied"],
                "transform_types": result["transform_types"],
                "total_cost": result["total_cost"],
                "compile_time_ms": result["compile_time_ms"],
                "git_hash": ghash,
                "timestamp": timestamp,
            })

    # Write CSV.
    fieldnames = [
        "domain", "transform_config", "throughput", "area",
        "iterations", "transforms_applied", "transform_types",
        "total_cost", "compile_time_ms",
        "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Throughput improvement summary.
    print("\n--- Throughput Improvement vs No Transforms ---")
    for domain_name in DOMAINS:
        base_row = next(
            (r for r in rows
             if r["domain"] == domain_name
             and r["transform_config"] == "no_transforms"), None)
        if not base_row or base_row["throughput"] <= 0:
            continue
        base_tp = base_row["throughput"]
        print(f"  {domain_name}:")
        for config in TRANSFORM_CONFIGS:
            row = next(
                (r for r in rows
                 if r["domain"] == domain_name
                 and r["transform_config"] == config["name"]), None)
            if row and row["throughput"] > 0:
                improvement = (row["throughput"] - base_tp) / base_tp * 100
                print(f"    {config['name']:20s}: "
                      f"{improvement:+.1f}% "
                      f"(transforms={row['transforms_applied']})")

    # Waterfall: base -> +retile -> +replicate -> all
    print("\n--- Waterfall Analysis ---")
    for domain_name in DOMAINS:
        configs_map = {}
        for r in rows:
            if r["domain"] == domain_name:
                configs_map[r["transform_config"]] = r

        base = configs_map.get("no_transforms", {}).get("total_cost", 0)
        retile = configs_map.get("retile_only", {}).get("total_cost", 0)
        rep = configs_map.get("replicate_only", {}).get("total_cost", 0)
        both = configs_map.get("all_transforms", {}).get("total_cost", 0)

        if base > 0:
            retile_delta = ((retile - base) / base * 100) if retile > 0 else 0
            rep_delta = ((rep - base) / base * 100) if rep > 0 else 0
            both_delta = ((both - base) / base * 100) if both > 0 else 0
            print(f"  {domain_name}: base={base:.0f} "
                  f"| +retile={retile_delta:+.1f}% "
                  f"| +replicate={rep_delta:+.1f}% "
                  f"| all={both_delta:+.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
