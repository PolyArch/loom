#!/usr/bin/env python3
"""E26: NoC Topology Comparison -- compare mesh, ring, hierarchical topologies.

Uses the NoCScheduler with XY and YX routing, plus ring-style routing via
the HierarchicalCompiler, to measure inter-core transfer costs across 6 benchmark
domains under three topology variants on a 2x2 core grid.

Topology models:
  - mesh: XY dimension-ordered routing on 2x2 grid (standard NoC)
  - ring: unidirectional ring -- max hop count 3 on 4 cores
  - hierarchical: 2 clusters of 2, intra-cluster direct, inter-cluster 2-hop

Usage:
    python3 scripts/experiments/run_e26_noc_topology.py
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

DOMAINS = {
    "ai_llm": {
        "tdg": "benchmarks/tapestry/ai_llm/tdg_transformer.py",
        "label": "Transformer",
    },
    "dsp_ofdm": {
        "tdg": "benchmarks/tapestry/dsp_ofdm/tdg_ofdm.py",
        "label": "OFDM",
    },
    "arvr_stereo": {
        "tdg": "benchmarks/tapestry/arvr_stereo/tdg_stereo.py",
        "label": "Stereo",
    },
    "robotics_vio": {
        "tdg": "benchmarks/tapestry/robotics_vio/tdg_vio.py",
        "label": "VIO",
    },
    "graph_analytics": {
        "tdg": "benchmarks/tapestry/graph_analytics/tdg_graph.py",
        "label": "Graph",
    },
    "zk_stark": {
        "tdg": "benchmarks/tapestry/zk_stark/tdg_stark.py",
        "label": "STARK",
    },
}

# 2x2 mesh core positions
CORE_POSITIONS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
MESH_COLS = 2
FLIT_BYTES = 8
ROUTER_PIPELINE_STAGES = 1
LINK_BW_FLITS = 2  # flits per cycle per link


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
    """Load a TDG Python module and return its kernels and contracts."""
    spec = importlib.util.spec_from_file_location("tdg", tdg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernels, mod.contracts


def data_type_bytes(dtype):
    """Return byte width for a contract data type."""
    mapping = {
        "float32": 4, "int32": 4, "uint32": 4,
        "float64": 8, "int64": 8,
        "complex64": 8, "int8": 1, "uint8": 1,
    }
    return mapping.get(dtype, 4)


def assign_kernels_to_cores(kernels, num_cores=4):
    """Round-robin assignment of kernels to core indices."""
    assignments = {}
    for i, k in enumerate(kernels):
        assignments[k["name"]] = i % num_cores
    return assignments


def xy_route(src, dst):
    """XY dimension-ordered route on a 2D mesh."""
    hops = [src]
    r, c = src
    dr, dc = dst
    while c != dc:
        c += 1 if dc > c else -1
        hops.append((r, c))
    while r != dr:
        r += 1 if dr > r else -1
        hops.append((r, c))
    return hops


def ring_route(src_idx, dst_idx, num_cores=4):
    """Unidirectional ring route (clockwise)."""
    hops = []
    cur = src_idx
    while cur != dst_idx:
        hops.append(cur)
        cur = (cur + 1) % num_cores
    hops.append(dst_idx)
    return hops


def hierarchical_route(src_idx, dst_idx):
    """Hierarchical 2-cluster topology: cores 0,1 in cluster A, 2,3 in cluster B.
    Intra-cluster = 1 hop, inter-cluster = 2 hops (through gateway)."""
    if src_idx == dst_idx:
        return [src_idx]
    src_cluster = src_idx // 2
    dst_cluster = dst_idx // 2
    if src_cluster == dst_cluster:
        return [src_idx, dst_idx]
    else:
        gateway = src_cluster * 2  # cluster gateway is even-indexed core
        return [src_idx, gateway, dst_idx]


def compute_transfer_cost(hops, total_data_bytes):
    """Compute transfer cycles from hop count and data volume."""
    num_hops = len(hops) - 1 if len(hops) > 1 else 0
    pipeline_latency = num_hops * ROUTER_PIPELINE_STAGES
    total_flits = math.ceil(total_data_bytes / FLIT_BYTES)
    serialization = math.ceil(total_flits / LINK_BW_FLITS)
    return pipeline_latency + serialization, num_hops, total_flits


def evaluate_topology(domain_name, kernels, contracts, topology, assignments):
    """Evaluate a topology for a given domain.
    Returns (total_noc_cycles, max_link_util, avg_hop_count, contention_events)."""
    total_cycles = 0
    hop_counts = []
    link_usage = {}
    contention_events = 0

    for c in contracts:
        prod_core = assignments.get(c["producer"])
        cons_core = assignments.get(c["consumer"])
        if prod_core is None or cons_core is None:
            continue
        if prod_core == cons_core:
            continue  # intra-core, no NoC

        dtype_bytes = data_type_bytes(c.get("data_type", "int32"))
        production_rate = c.get("production_rate", 1024)
        total_bytes = production_rate * dtype_bytes

        if topology == "mesh":
            src_pos = CORE_POSITIONS[prod_core]
            dst_pos = CORE_POSITIONS[cons_core]
            hops = xy_route(src_pos, dst_pos)
            hop_key_list = [(hops[i], hops[i + 1]) for i in range(len(hops) - 1)]
        elif topology == "ring":
            ring_hops = ring_route(prod_core, cons_core)
            hop_key_list = [(ring_hops[i], ring_hops[i + 1])
                            for i in range(len(ring_hops) - 1)]
            hops = ring_hops
        elif topology == "hierarchical":
            hier_hops = hierarchical_route(prod_core, cons_core)
            hop_key_list = [(hier_hops[i], hier_hops[i + 1])
                            for i in range(len(hier_hops) - 1)]
            hops = hier_hops
        else:
            raise ValueError(f"Unknown topology: {topology}")

        cost, nhops, nflits = compute_transfer_cost(hops, total_bytes)
        total_cycles += cost
        hop_counts.append(nhops)

        for link in hop_key_list:
            link_usage[link] = link_usage.get(link, 0) + nflits

    # Compute link utilizations
    max_link_util = 0.0
    if link_usage:
        max_flits = max(link_usage.values())
        # Normalize by total available link bandwidth (arbitrary window)
        window_cycles = max(total_cycles, 1)
        max_link_util = max_flits / (LINK_BW_FLITS * window_cycles)
        max_link_util = min(max_link_util, 1.0)

        # Count links with utilization > 0.8 as contention
        for link, flits in link_usage.items():
            util = flits / (LINK_BW_FLITS * window_cycles)
            if util > 0.8:
                contention_events += 1

    avg_hops = sum(hop_counts) / len(hop_counts) if hop_counts else 0.0
    return total_cycles, max_link_util, avg_hops, contention_events


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E26"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "noc_topology.csv"
    topologies = ["mesh", "ring", "hierarchical"]

    print(f"E26: NoC Topology Comparison")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print(f"  topologies: {topologies}")
    print()

    rows = []
    for domain_name, info in DOMAINS.items():
        tdg_path = REPO_ROOT / info["tdg"]
        if not tdg_path.exists():
            print(f"  WARNING: {tdg_path} not found, skipping {domain_name}")
            continue

        kernels, contracts = load_tdg(str(tdg_path))
        assignments = assign_kernels_to_cores(kernels)

        for topo in topologies:
            total_cycles, max_util, avg_hops, contention = evaluate_topology(
                domain_name, kernels, contracts, topo, assignments,
            )

            row = {
                "domain": domain_name,
                "topology": topo,
                "total_noc_cycles": total_cycles,
                "max_link_utilization": round(max_util, 6),
                "avg_hop_count": round(avg_hops, 4),
                "contention_events": contention,
                "num_kernels": len(kernels),
                "num_contracts": len(contracts),
                "num_cross_core": sum(
                    1 for c in contracts
                    if assignments.get(c["producer"]) != assignments.get(c["consumer"])
                ),
                "git_hash": ghash,
                "timestamp": timestamp,
            }
            rows.append(row)
            print(f"  {domain_name:20s} {topo:15s}: "
                  f"cycles={total_cycles:6d}, "
                  f"max_util={max_util:.4f}, "
                  f"avg_hops={avg_hops:.2f}, "
                  f"contention={contention}")

    fieldnames = [
        "domain", "topology", "total_noc_cycles", "max_link_utilization",
        "avg_hop_count", "contention_events", "num_kernels", "num_contracts",
        "num_cross_core", "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # Also write JSON with link utilization details
    json_path = out_dir / "noc_topology_details.json"
    details = {
        "experiment": "E26: NoC Topology Comparison",
        "grid_size": "2x2",
        "topologies": topologies,
        "flit_bytes": FLIT_BYTES,
        "link_bw_flits": LINK_BW_FLITS,
        "router_pipeline_stages": ROUTER_PIPELINE_STAGES,
        "data_points": len(rows),
        "git_hash": ghash,
        "timestamp": timestamp,
    }
    with open(json_path, "w") as f:
        json.dump(details, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
