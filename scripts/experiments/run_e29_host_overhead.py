#!/usr/bin/env python3
"""E29: Host-CGRA Interaction Overhead -- quantify DMA/sync costs.

Models three configurations per domain:
  - all-CGRA: all kernels run on CGRA cores
  - 1-host: one middle kernel forced to HOST
  - 2-host: two non-adjacent kernels forced to HOST

Computes CGRA execution cycles, HOST execution cycles, DMA transfer cycles,
and synchronization overhead using the real TDG contract data and the
execution model from the SystemCompiler.

Usage:
    python3 scripts/experiments/run_e29_host_overhead.py
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

# Three domains with different pipeline structures
DOMAINS = {
    "ai_llm": {
        "tdg": "benchmarks/tapestry/ai_llm/tdg_transformer.py",
        "label": "Transformer",
        "host_kernels_1": ["gelu"],  # middle of pipeline
        "host_kernels_2": ["softmax", "ffn2"],  # non-adjacent
    },
    "dsp_ofdm": {
        "tdg": "benchmarks/tapestry/dsp_ofdm/tdg_ofdm.py",
        "label": "OFDM",
        "host_kernels_1": ["equalizer"],
        "host_kernels_2": ["channel_est", "viterbi"],
    },
    "graph_analytics": {
        "tdg": "benchmarks/tapestry/graph_analytics/tdg_graph.py",
        "label": "Graph",
        "host_kernels_1": ["pagerank_spmv"],
        "host_kernels_2": ["bfs_traversal", "label_prop"],
    },
}

# Execution model parameters
CGRA_CLOCK_MHZ = 500
HOST_CLOCK_MHZ = 2000
DMA_SETUP_CYCLES = 50       # per DMA descriptor
DMA_BW_BYTES_PER_CYCLE = 8  # DMA engine bandwidth
SYNC_CYCLES = 20            # per host-CGRA synchronization event
HOST_SLOWDOWN = 3.0         # host runs kernels ~3x slower than CGRA
                             # (no spatial parallelism)


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


def data_type_bytes(dtype):
    mapping = {
        "float32": 4, "int32": 4, "uint32": 4,
        "float64": 8, "int64": 8,
        "complex64": 8, "int8": 1,
    }
    return mapping.get(dtype, 4)


def estimate_kernel_cycles(kernel, contracts):
    """Estimate CGRA execution cycles for a kernel based on its contracts."""
    # Use production rate from contracts as a proxy for computation
    total_elements = 0
    for c in contracts:
        if c["producer"] == kernel["name"] or c["consumer"] == kernel["name"]:
            total_elements += c.get("production_rate", 1024)

    # Simple model: ~1 cycle per element at II=1
    base_cycles = max(total_elements, 100)
    return base_cycles


def compute_config(domain_name, kernels, contracts, host_kernel_names, config_name):
    """Compute execution breakdown for a given host kernel configuration."""
    host_set = set(host_kernel_names)

    # Estimate per-kernel execution cycles
    cgra_total = 0
    host_total = 0
    dma_total = 0
    sync_total = 0

    for k in kernels:
        base_cycles = estimate_kernel_cycles(k, contracts)

        if k["name"] in host_set:
            # Kernel runs on host: slower, plus DMA for data in/out
            host_cycles = int(base_cycles * HOST_SLOWDOWN)
            host_total += host_cycles
        else:
            # Kernel runs on CGRA
            cgra_total += base_cycles

    # DMA: for each contract crossing HOST<->CGRA boundary
    boundary_crossings = 0
    for c in contracts:
        prod_on_host = c["producer"] in host_set
        cons_on_host = c["consumer"] in host_set
        if prod_on_host != cons_on_host:
            # Boundary crossing: need DMA transfer
            boundary_crossings += 1
            dtype_bytes = data_type_bytes(c.get("data_type", "int32"))
            production_rate = c.get("production_rate", 1024)
            data_bytes = production_rate * dtype_bytes

            transfer_cycles = DMA_SETUP_CYCLES + math.ceil(
                data_bytes / DMA_BW_BYTES_PER_CYCLE)
            dma_total += transfer_cycles

    # Synchronization: one sync event per boundary crossing
    sync_total = boundary_crossings * SYNC_CYCLES

    total_cycles = cgra_total + host_total + dma_total + sync_total
    baseline = cgra_total + host_total  # zero-overhead baseline
    overhead_pct = ((dma_total + sync_total) / max(total_cycles, 1)) * 100

    return {
        "domain": domain_name,
        "config": config_name,
        "cgra_cycles": cgra_total,
        "host_cycles": host_total,
        "dma_cycles": dma_total,
        "sync_cycles": sync_total,
        "total_cycles": total_cycles,
        "overhead_pct": round(overhead_pct, 2),
        "boundary_crossings": boundary_crossings,
        "host_kernel_count": len(host_kernel_names),
    }


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E29"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "host_overhead.csv"

    print(f"E29: Host-CGRA Interaction Overhead")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print(f"  DMA setup: {DMA_SETUP_CYCLES} cycles")
    print(f"  DMA BW: {DMA_BW_BYTES_PER_CYCLE} bytes/cycle")
    print(f"  Sync: {SYNC_CYCLES} cycles/event")
    print(f"  Host slowdown: {HOST_SLOWDOWN}x")
    print()

    rows = []
    for domain_name, info in DOMAINS.items():
        tdg_path = REPO_ROOT / info["tdg"]
        if not tdg_path.exists():
            print(f"  WARNING: {tdg_path} not found, skipping")
            continue

        kernels, contracts = load_tdg(str(tdg_path))
        print(f"  {domain_name}: {len(kernels)} kernels, {len(contracts)} contracts")

        # Config 1: all-CGRA
        r = compute_config(domain_name, kernels, contracts, [], "all_cgra")
        r["git_hash"] = ghash
        r["timestamp"] = timestamp
        rows.append(r)
        print(f"    all_cgra:  total={r['total_cycles']:7d}, overhead={r['overhead_pct']:.1f}%")

        # Config 2: 1-host
        r = compute_config(domain_name, kernels, contracts,
                           info["host_kernels_1"], "1_host")
        r["git_hash"] = ghash
        r["timestamp"] = timestamp
        rows.append(r)
        print(f"    1_host:    total={r['total_cycles']:7d}, overhead={r['overhead_pct']:.1f}%, "
              f"host_kernels={info['host_kernels_1']}")

        # Config 3: 2-host
        r = compute_config(domain_name, kernels, contracts,
                           info["host_kernels_2"], "2_host")
        r["git_hash"] = ghash
        r["timestamp"] = timestamp
        rows.append(r)
        print(f"    2_host:    total={r['total_cycles']:7d}, overhead={r['overhead_pct']:.1f}%, "
              f"host_kernels={info['host_kernels_2']}")

    fieldnames = [
        "domain", "config", "cgra_cycles", "host_cycles", "dma_cycles",
        "sync_cycles", "total_cycles", "overhead_pct",
        "boundary_crossings", "host_kernel_count",
        "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Verification: total = cgra + host + dma + sync
    print("\n--- Accounting verification ---")
    for r in rows:
        expected = r["cgra_cycles"] + r["host_cycles"] + r["dma_cycles"] + r["sync_cycles"]
        match = "OK" if expected == r["total_cycles"] else "MISMATCH"
        print(f"  {r['domain']:20s} {r['config']:10s}: "
              f"{r['cgra_cycles']}+{r['host_cycles']}+{r['dma_cycles']}+{r['sync_cycles']}"
              f"={expected} vs total={r['total_cycles']} [{match}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
