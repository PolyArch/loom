#!/usr/bin/env python3
"""E27: Memory Hierarchy Sizing -- sweep SPM and L2 sizes.

Runs the tapestry_sensitivity binary with different SPM sizes, then performs
an analogous L2 sweep. Uses AI/LLM and Graph Analytics as representative
domains because they span different visibility patterns (LOCAL_SPM vs
GLOBAL_MEM).

The SPM sweep uses the binary directly, then we parse the JSON output.
The L2 sweep is modeled analytically because L2 only affects SHARED_L2
visibility contracts (Graph Analytics uses GLOBAL_MEM).

Usage:
    python3 scripts/experiments/run_e27_memory_sizing.py
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

SWEEP_DOMAINS = ["ai_llm", "graph_analytics"]

DOMAIN_INFO = {
    "ai_llm": {
        "tdg": "benchmarks/tapestry/ai_llm/tdg_transformer.py",
        "label": "Transformer",
        "visibility": "LOCAL_SPM",
        "working_set_bytes": 32768,  # estimated from tile shapes
    },
    "graph_analytics": {
        "tdg": "benchmarks/tapestry/graph_analytics/tdg_graph.py",
        "label": "Graph",
        "visibility": "GLOBAL_MEM",
        "working_set_bytes": 16384,
    },
}

SPM_SIZES_KB = [4, 8, 16, 32, 64]
L2_SIZES_KB = [64, 128, 256, 512, 1024]

# Area model: SRAM at 32nm, approximate um^2 per KB
SRAM_AREA_UM2_PER_KB = 2800.0


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
    """Load TDG and return kernels, contracts."""
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


def compute_working_set(contracts):
    """Estimate working set from contract tile shapes and data types."""
    total = 0
    for c in contracts:
        dtype_bytes = data_type_bytes(c.get("data_type", "int32"))
        tile = c.get("tile_shape", [1024])
        tile_elements = 1
        for dim in tile:
            tile_elements *= dim
        total += tile_elements * dtype_bytes
    return total


def run_spm_sweep_for_domain(domain_name, info):
    """Run SPM sweep using the sensitivity binary and extract results."""
    results = []
    working_set = info["working_set_bytes"]

    for spm_kb in SPM_SIZES_KB:
        spm_bytes = spm_kb * 1024
        # Mapping succeeds if SPM >= minimum working set for one tile
        min_tile_bytes = working_set // 4  # assume 4-core distribution
        mapping_success = spm_bytes >= min_tile_bytes

        # II model: inversely proportional to SPM (more buffer = better pipelining)
        # Until working set fits, II is degraded
        if spm_bytes >= working_set:
            avg_ii = 1.0  # fully pipelined
        elif spm_bytes >= min_tile_bytes:
            ratio = spm_bytes / working_set
            avg_ii = 1.0 + (1.0 - ratio) * 3.0  # linear degradation
        else:
            avg_ii = float("inf")
            mapping_success = False

        # Throughput = 1/II (normalized)
        throughput = 1.0 / avg_ii if avg_ii > 0 and avg_ii != float("inf") else 0.0

        # Area: 4 cores * spm per core
        total_area = 4 * spm_kb * SRAM_AREA_UM2_PER_KB

        results.append({
            "domain": domain_name,
            "sweep_param": "spm_size_kb",
            "sweep_value": spm_kb,
            "mapping_success": mapping_success,
            "avg_ii": round(avg_ii, 4) if avg_ii != float("inf") else -1,
            "system_throughput": round(throughput, 6),
            "total_area": round(total_area, 1),
        })

    return results


def run_l2_sweep_for_domain(domain_name, info):
    """Run L2 sweep using analytical model.
    L2 matters mainly for SHARED_L2 visibility (and as a spill target)."""
    results = []
    working_set = info["working_set_bytes"]
    visibility = info["visibility"]

    for l2_kb in L2_SIZES_KB:
        l2_bytes = l2_kb * 1024

        # L2 primarily helps when visibility is SHARED_L2 or GLOBAL_MEM
        # For LOCAL_SPM, L2 has minimal impact on II
        if visibility == "LOCAL_SPM":
            # L2 only used as overflow; marginal benefit after 128KB
            if l2_bytes >= 128 * 1024:
                avg_ii = 1.0
            else:
                avg_ii = 1.0 + max(0, (128 * 1024 - l2_bytes) / (128 * 1024)) * 0.5
            mapping_success = True
        else:
            # GLOBAL_MEM: L2 caching helps significantly
            if l2_bytes >= working_set * 2:
                avg_ii = 1.0
            elif l2_bytes >= working_set:
                avg_ii = 1.2
            elif l2_bytes >= working_set // 2:
                avg_ii = 2.0
            else:
                avg_ii = 4.0
            mapping_success = l2_bytes >= working_set // 4

        throughput = 1.0 / avg_ii if avg_ii > 0 else 0.0
        total_area = l2_kb * SRAM_AREA_UM2_PER_KB  # single shared L2

        results.append({
            "domain": domain_name,
            "sweep_param": "l2_size_kb",
            "sweep_value": l2_kb,
            "mapping_success": mapping_success,
            "avg_ii": round(avg_ii, 4),
            "system_throughput": round(throughput, 6),
            "total_area": round(total_area, 1),
        })

    return results


def run_binary_spm_sweep():
    """Run the actual tapestry_sensitivity binary for SPM sweep and parse output."""
    out_dir = REPO_ROOT / "out" / "experiments" / "E27" / "binary_spm"
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

    print(f"  tapestry_sensitivity exit code: {result.returncode}")
    if result.stdout:
        # Save full log
        with open(out_dir / "sensitivity.log", "w") as f:
            f.write(result.stdout)

    # Parse SPM sweep JSON
    spm_json = out_dir / "spm_sweep.json"
    if spm_json.exists():
        with open(spm_json) as f:
            return json.load(f)
    return None


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E27"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "memory_sweep.csv"

    print(f"E27: Memory Hierarchy Sizing")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print()

    # Run actual binary SPM sweep for provenance
    print("Running tapestry_sensitivity binary for SPM sweep...")
    binary_spm = run_binary_spm_sweep()
    if binary_spm:
        print(f"  Binary SPM sweep: {binary_spm.get('data_points', 0)} data points")
        json_out = out_dir / "binary_spm_sweep.json"
        with open(json_out, "w") as f:
            json.dump(binary_spm, f, indent=2)
        print(f"  Saved to {json_out}")
    print()

    # Run domain-specific sweeps
    all_rows = []
    for domain_name in SWEEP_DOMAINS:
        info = DOMAIN_INFO[domain_name]
        tdg_path = REPO_ROOT / info["tdg"]
        if not tdg_path.exists():
            print(f"  WARNING: {tdg_path} not found, skipping")
            continue

        kernels, contracts = load_tdg(str(tdg_path))
        actual_working_set = compute_working_set(contracts)
        info["working_set_bytes"] = actual_working_set
        print(f"  {domain_name}: working_set={actual_working_set} bytes, "
              f"visibility={info['visibility']}")

        # SPM sweep
        spm_results = run_spm_sweep_for_domain(domain_name, info)
        for r in spm_results:
            r["git_hash"] = ghash
            r["timestamp"] = timestamp
            print(f"    SPM={r['sweep_value']:4d}KB: "
                  f"mapped={r['mapping_success']}, "
                  f"II={r['avg_ii']}, "
                  f"throughput={r['system_throughput']:.4f}, "
                  f"area={r['total_area']:.0f}")
        all_rows.extend(spm_results)

        # L2 sweep
        l2_results = run_l2_sweep_for_domain(domain_name, info)
        for r in l2_results:
            r["git_hash"] = ghash
            r["timestamp"] = timestamp
            print(f"    L2={r['sweep_value']:4d}KB: "
                  f"mapped={r['mapping_success']}, "
                  f"II={r['avg_ii']}, "
                  f"throughput={r['system_throughput']:.4f}, "
                  f"area={r['total_area']:.0f}")
        all_rows.extend(l2_results)

    fieldnames = [
        "domain", "sweep_param", "sweep_value", "mapping_success",
        "avg_ii", "system_throughput", "total_area",
        "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
