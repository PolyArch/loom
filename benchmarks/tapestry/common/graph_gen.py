#!/usr/bin/env python3
"""Graph generator for Tapestry graph analytics benchmarks.

Generates CSR-format graphs as binary files for BFS, PageRank,
triangle counting, and label propagation benchmarks.
"""

import struct
import os
import math


def write_int_array(filename, data):
    """Write a list of ints as raw binary (32-bit signed)."""
    with open(filename, "wb") as f:
        for val in data:
            f.write(struct.pack("i", val))


def write_float_array(filename, data):
    """Write a list of floats as raw binary (32-bit float)."""
    with open(filename, "wb") as f:
        for val in data:
            f.write(struct.pack("f", val))


def gen_erdos_renyi(nv, avg_deg, seed=42):
    """Generate an Erdos-Renyi random graph in CSR format.

    Returns (row_ptr, col_idx, values) where the graph is undirected
    and adjacency lists are sorted.
    """
    p = avg_deg / (nv - 1) if nv > 1 else 0.0
    adj = [[] for _ in range(nv)]
    state = seed
    for u in range(nv):
        for v in range(u + 1, nv):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            if (state % 10000) / 10000.0 < p:
                adj[u].append(v)
                adj[v].append(u)
    for u in range(nv):
        adj[u].sort()

    row_ptr = [0]
    col_idx = []
    values = []
    for u in range(nv):
        row_ptr.append(row_ptr[-1] + len(adj[u]))
        col_idx.extend(adj[u])
        values.extend([1.0] * len(adj[u]))

    return row_ptr, col_idx, values


def gen_power_law(nv, avg_deg, seed=42):
    """Generate a power-law (preferential attachment) graph in CSR format."""
    adj = [[] for _ in range(nv)]
    m = max(1, avg_deg // 2)
    state = seed

    # Start with a small clique
    init_size = min(m + 1, nv)
    for u in range(init_size):
        for v in range(u + 1, init_size):
            adj[u].append(v)
            adj[v].append(u)

    deg = [0] * nv
    for u in range(init_size):
        deg[u] = len(adj[u])

    total_deg = sum(deg)
    for new_v in range(init_size, nv):
        targets = set()
        attempts = 0
        while len(targets) < m and attempts < m * 10:
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            # Preferential attachment
            r = state % max(1, total_deg)
            cumul = 0
            for u in range(new_v):
                cumul += deg[u]
                if cumul > r:
                    targets.add(u)
                    break
            attempts += 1
        for t in targets:
            adj[new_v].append(t)
            adj[t].append(new_v)
            deg[new_v] += 1
            deg[t] += 1
            total_deg += 2

    for u in range(nv):
        adj[u].sort()
        # Remove duplicates
        adj[u] = list(dict.fromkeys(adj[u]))

    row_ptr = [0]
    col_idx = []
    values = []
    for u in range(nv):
        row_ptr.append(row_ptr[-1] + len(adj[u]))
        col_idx.extend(adj[u])
        values.extend([1.0] * len(adj[u]))

    return row_ptr, col_idx, values


def save_csr(output_dir, prefix, row_ptr, col_idx, values):
    """Save CSR graph to binary files."""
    os.makedirs(output_dir, exist_ok=True)
    write_int_array(os.path.join(output_dir, f"{prefix}_row_ptr.bin"), row_ptr)
    write_int_array(os.path.join(output_dir, f"{prefix}_col_idx.bin"), col_idx)
    write_float_array(os.path.join(output_dir, f"{prefix}_values.bin"), values)
    nv = len(row_ptr) - 1
    ne = len(col_idx)
    print(f"  {prefix}: {nv} vertices, {ne} edges")


def generate_all(output_dir="generated_inputs"):
    """Generate all graph benchmark inputs."""
    os.makedirs(output_dir, exist_ok=True)
    print("Generating graph benchmark inputs:")

    # Small graph for testing (20 vertices, avg deg 6)
    rp, ci, vals = gen_erdos_renyi(20, 6, seed=100)
    save_csr(output_dir, "graph_small", rp, ci, vals)

    # Medium graph (1000 vertices, avg deg 10)
    rp, ci, vals = gen_erdos_renyi(1000, 10, seed=200)
    save_csr(output_dir, "graph_medium", rp, ci, vals)

    # Power-law graph (500 vertices)
    rp, ci, vals = gen_power_law(500, 8, seed=300)
    save_csr(output_dir, "graph_powerlaw", rp, ci, vals)

    print("Done.")


if __name__ == "__main__":
    generate_all()
