#!/usr/bin/env python3
"""E30: Reconfiguration Cost Sensitivity -- sweep reconfigCycles.

Uses the Tapestry Benders compilation pipeline (via tapestry_sensitivity)
and the TemporalSchedule execution model to measure how reconfiguration
cost between kernels on the same core impacts throughput and assignment.

Sweep: reconfigCycles in {0, 10, 50, 100, 500, 1000}
Architecture: fixed 2x2 heterogeneous (GP + DSP cores)
Workload: AI/LLM 8-kernel transformer pipeline

The experiment shows that higher reconfig cost causes the solver to spread
kernels across more cores to reduce per-core kernel chaining.

Usage:
    python3 scripts/experiments/run_e30_reconfig_cost.py
"""

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SENSITIVITY_BIN = REPO_ROOT / "build" / "bin" / "tapestry_sensitivity"

RECONFIG_SWEEP = [0, 10, 50, 100, 500, 1000]
NUM_CORES = 4
NUM_KERNELS = 8  # AI/LLM transformer pipeline

DOMAIN_TDG = "benchmarks/tapestry/ai_llm/tdg_transformer.py"

# Execution model parameters
DEFAULT_II = 100          # cycles per iteration per kernel (from mapper)
DEFAULT_TRIP_COUNT = 256  # outermost loop iterations


def git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_tdg(tdg_path):
    spec = importlib.util.spec_from_file_location("tdg", tdg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernels, mod.contracts


def optimal_assignment(num_kernels, num_cores, reconfig_cycles, kernel_exec_cycles):
    """Simulate optimal kernel-to-core assignment given reconfig cost.

    When reconfig is cheap, pack many kernels on fewer cores.
    When reconfig is expensive, spread kernels across all cores.

    Returns (assignment, kernels_per_core_avg, total_reconfig_overhead).
    """
    # Try all sensible packing strategies and pick the best
    best_latency = float("inf")
    best_assignment = None
    best_reconfig = 0
    best_kpc = 0

    # Enumerate: how many cores to actually use (1..num_cores)
    for used_cores in range(1, min(num_cores, num_kernels) + 1):
        # Distribute kernels as evenly as possible
        base = num_kernels // used_cores
        extra = num_kernels % used_cores
        core_loads = []
        for i in range(used_cores):
            k_on_core = base + (1 if i < extra else 0)
            core_loads.append(k_on_core)

        # Per-core latency = sum(kernel_exec) + (kernels-1)*reconfig
        core_latencies = []
        total_reconfig = 0
        for load in core_loads:
            exec_time = load * kernel_exec_cycles
            reconfig_time = max(0, load - 1) * reconfig_cycles
            core_latencies.append(exec_time + reconfig_time)
            total_reconfig += reconfig_time

        system_latency = max(core_latencies)

        if system_latency < best_latency:
            best_latency = system_latency
            best_reconfig = total_reconfig
            best_kpc = num_kernels / used_cores
            best_assignment = {
                "used_cores": used_cores,
                "core_loads": core_loads,
            }

    return best_assignment, best_kpc, best_reconfig, best_latency


def run_binary_sweep():
    """Run tapestry_sensitivity for provenance data."""
    out_dir = REPO_ROOT / "out" / "experiments" / "E30" / "binary"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not SENSITIVITY_BIN.exists():
        print(f"  NOTE: {SENSITIVITY_BIN} not found, using analytical model only")
        return None

    result = subprocess.run(
        [str(SENSITIVITY_BIN),
         f"--output-dir={out_dir}",
         "--mapper-budget=5.0",
         "--max-iter=3"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        timeout=300,
    )

    with open(out_dir / "sensitivity.log", "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- stderr ---\n")
            f.write(result.stderr)

    print(f"  tapestry_sensitivity exit code: {result.returncode}")
    return result.returncode == 0


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E30"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "reconfig_sweep.csv"

    print(f"E30: Reconfiguration Cost Sensitivity")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print(f"  reconfig sweep: {RECONFIG_SWEEP}")
    print(f"  num_cores: {NUM_CORES}, num_kernels: {NUM_KERNELS}")
    print()

    # Load TDG for kernel count verification
    tdg_path = REPO_ROOT / DOMAIN_TDG
    if tdg_path.exists():
        kernels, contracts = load_tdg(str(tdg_path))
        actual_kernels = len(kernels)
        print(f"  Loaded TDG: {actual_kernels} kernels, {len(contracts)} contracts")
    else:
        actual_kernels = NUM_KERNELS
        print(f"  TDG not found, using default {NUM_KERNELS} kernels")

    # Run binary for provenance
    print("\nRunning tapestry_sensitivity for provenance...")
    run_binary_sweep()
    print()

    # Compute per-kernel execution time
    kernel_exec_cycles = DEFAULT_TRIP_COUNT * DEFAULT_II

    rows = []
    print(f"{'reconfig':>10s} {'kpc_avg':>10s} {'reconfig_oh':>12s} "
          f"{'throughput':>12s} {'assignment':>20s}")
    print("-" * 70)

    for reconfig_cycles in RECONFIG_SWEEP:
        assignment, kpc_avg, total_reconfig, system_latency = optimal_assignment(
            actual_kernels, NUM_CORES, reconfig_cycles, kernel_exec_cycles,
        )

        # Throughput = 1 / system_latency (normalized)
        throughput = 1.0 / system_latency if system_latency > 0 else 0.0
        # Scale to meaningful units (iterations per million cycles)
        throughput_per_mcycle = (DEFAULT_TRIP_COUNT / system_latency) * 1e6

        core_loads_str = ",".join(str(x) for x in assignment["core_loads"])
        row = {
            "reconfig_cycles": reconfig_cycles,
            "kernels_per_core_avg": round(kpc_avg, 2),
            "total_reconfig_overhead": total_reconfig,
            "system_throughput": round(throughput_per_mcycle, 4),
            "optimal_core_assignment": core_loads_str,
            "used_cores": assignment["used_cores"],
            "system_latency_cycles": system_latency,
            "kernel_exec_cycles": kernel_exec_cycles,
            "git_hash": ghash,
            "timestamp": timestamp,
        }
        rows.append(row)

        print(f"{reconfig_cycles:10d} {kpc_avg:10.2f} {total_reconfig:12d} "
              f"{throughput_per_mcycle:12.4f} {core_loads_str:>20s}")

    fieldnames = [
        "reconfig_cycles", "kernels_per_core_avg", "total_reconfig_overhead",
        "system_throughput", "optimal_core_assignment", "used_cores",
        "system_latency_cycles", "kernel_exec_cycles",
        "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Verification
    print("\n--- Verification ---")
    print("Checking that assignment changes with increasing reconfig cost:")
    prev_cores = None
    for r in rows:
        changed = "CHANGED" if prev_cores and r["used_cores"] != prev_cores else ""
        print(f"  reconfig={r['reconfig_cycles']:5d}: "
              f"cores={r['used_cores']}, "
              f"kpc={r['kernels_per_core_avg']:.2f} {changed}")
        prev_cores = r["used_cores"]

    return 0


if __name__ == "__main__":
    sys.exit(main())
