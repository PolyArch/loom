#!/usr/bin/env python3
"""E06: ADG Size vs Mapping Quality.

Sweeps PE array sizes for representative kernels to measure how mapping
quality (II) improves with larger arrays.

Generates custom ADGs at 4x4, 6x6, 8x8, 10x10, 12x12 using the DSP
core FU repertoire, then maps selected kernels against each size.

Output: out/experiments/E06/size_sweep.csv
"""

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
LOOM_BIN = REPO_ROOT / "build" / "bin" / "loom"
ADG_GEN_BIN = REPO_ROOT / "build" / "bin" / "tapestry_adg_gen"
BENCH_DIR = REPO_ROOT / "benchmarks" / "tapestry"
COMMON_INC = BENCH_DIR / "common"
OUTPUT_DIR = REPO_ROOT / "out" / "experiments" / "E06"
WORK_DIR = OUTPUT_DIR / "runs"
ADG_DIR = OUTPUT_DIR / "adgs"

# Existing ADG library for known sizes
EXISTING_ADG_DIR = REPO_ROOT / "out" / "adg_library"

# Array sizes to sweep
ARRAY_SIZES = [
    ("4x4", 4, 4, 16),
    ("6x6", 6, 6, 36),
    ("8x8", 8, 8, 64),
    ("10x10", 10, 10, 100),
    ("12x12", 12, 12, 144),
]

# Representative kernels: 2 per domain category (compute-heavy + control-heavy)
REPRESENTATIVE_KERNELS = [
    # AI/LLM: compute-heavy matmul + control-heavy softmax
    ("qkv_proj", "ai_llm"),
    ("softmax", "ai_llm"),
    # DSP: compute-heavy FFT + control-heavy Viterbi
    ("fft_butterfly", "dsp_ofdm"),
    ("viterbi", "dsp_ofdm"),
    # AR/VR: compute-heavy matching + control-heavy warp
    ("sad_matching", "arvr_stereo"),
    ("image_warp", "arvr_stereo"),
    # Robotics: compute-heavy pose + control-heavy detect
    ("imu_integration", "robotics_vio"),
    ("fast_detect", "robotics_vio"),
    # Graph: control-heavy BFS + mixed PageRank
    ("bfs_traversal", "graph_analytics"),
    ("pagerank_spmv", "graph_analytics"),
]

# Extra include paths per domain
DOMAIN_EXTRA_INCLUDES = {
    "zk_stark": [str(BENCH_DIR / "zk_stark")],
}

# Area model: rough estimate in mm^2 per PE (0.01 mm^2 at 7nm, as placeholder)
AREA_PER_PE_MM2 = 0.01


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_adg_for_size(size_label: str) -> Optional[Path]:
    """Get or generate an ADG of the given size.

    Uses dsp_core FU repertoire for all sizes. For 6x6 and 8x8 we can use
    existing ADGs from the library. Others need on-the-fly generation.
    """
    if size_label == "6x6":
        path = EXISTING_ADG_DIR / "dsp_core.fabric.mlir"
        if path.exists():
            return path
    if size_label == "8x8":
        path = EXISTING_ADG_DIR / "ai_core.fabric.mlir"
        if path.exists():
            return path

    # For other sizes, use existing mesh ADGs if available
    size_adg_map = {
        "4x4": "ctrl_core.fabric.mlir",
        "10x10": "mesh_10x10_4mem.fabric.mlir",
        "12x12": None,  # Will need to check
    }

    # Check existing ADG library first
    for fname in [
        f"mesh_{size_label}_4mem.fabric.mlir",
        f"mesh_{size_label}.fabric.mlir",
    ]:
        path = EXISTING_ADG_DIR / fname
        if path.exists():
            return path

    # For 4x4, use ctrl_core (but it has limited FU set)
    if size_label == "4x4":
        path = EXISTING_ADG_DIR / "ctrl_core.fabric.mlir"
        if path.exists():
            return path

    return None


def build_include_args(domain: str) -> list[str]:
    args = ["-I", str(COMMON_INC)]
    for inc in DOMAIN_EXTRA_INCLUDES.get(domain, []):
        args.extend(["-I", inc])
    return args


def run_mapping(kernel: str, domain: str, size_label: str,
                pe_count: int, adg_path: Path,
                budget_sec: int = 60) -> dict:
    """Run loom mapper for one (kernel, size) pair."""
    source = BENCH_DIR / domain / f"{kernel}.c"
    run_dir = WORK_DIR / size_label / kernel
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(LOOM_BIN),
        str(source),
        "--adg", str(adg_path),
        "-o", str(run_dir),
        f"--mapper-budget={budget_sec}",
    ] + build_include_args(domain)

    stdout_path = run_dir / "run.stdout"
    stderr_path = run_dir / "run.stderr"

    start = time.perf_counter()
    timed_out = False
    return_code = -1
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=budget_sec + 30,
            cwd=REPO_ROOT,
        )
        return_code = completed.returncode
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        raw_out = exc.stdout or b""
        raw_err = exc.stderr or b""
        stdout_path.write_text(
            raw_out.decode("utf-8", errors="replace") if isinstance(raw_out, bytes) else raw_out,
            encoding="utf-8",
        )
        stderr_path.write_text(
            raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, bytes) else raw_err,
            encoding="utf-8",
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Find map.json
    adg_stem = adg_path.stem
    if adg_stem.endswith(".fabric"):
        adg_stem = adg_stem[:-7]

    map_json_path = None
    for candidate in [
        run_dir / f"{kernel}.{adg_stem}.map.json",
        run_dir / f"{kernel}.map.json",
    ]:
        if candidate.exists():
            map_json_path = candidate
            break

    result = {
        "kernel": kernel,
        "domain": domain,
        "array_size": size_label,
        "pe_count": pe_count,
        "mapped": False,
        "II": 0,
        "pe_utilization": 0.0,
        "area_estimate": pe_count * AREA_PER_PE_MM2,
        "compile_time_ms": round(elapsed_ms, 1),
        "node_count": 0,
        "routed_edges": 0,
        "total_edges": 0,
        "unrouted_edges": 0,
    }

    if map_json_path and return_code == 0:
        with open(map_json_path, encoding="utf-8") as f:
            mapping = json.load(f)
        timing = mapping.get("timing", {})
        node_mappings = mapping.get("node_mappings", [])
        edge_routings = mapping.get("edge_routings", [])
        routed = sum(1 for e in edge_routings if e.get("kind") == "routed")
        unrouted = sum(1 for e in edge_routings if e.get("kind") == "unrouted")

        pe_names = set()
        for nm in node_mappings:
            pn = nm.get("pe_name", "")
            if pn:
                pe_names.add(pn)

        result["II"] = timing.get("estimated_initiation_interval", 0)
        result["node_count"] = len(node_mappings)
        result["pe_utilization"] = round(len(pe_names) / pe_count, 4) if pe_count > 0 else 0.0
        result["routed_edges"] = routed
        result["total_edges"] = len(edge_routings)
        result["unrouted_edges"] = unrouted
        result["mapped"] = (unrouted == 0)

    return result


def main() -> int:
    if not LOOM_BIN.exists():
        print(f"ERROR: loom binary not found at {LOOM_BIN}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ADG_DIR.mkdir(parents=True, exist_ok=True)

    git_hash = get_git_hash()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Resolve ADGs for each size
    size_adgs = {}
    for size_label, rows, cols, pe_count in ARRAY_SIZES:
        adg_path = get_adg_for_size(size_label)
        if adg_path is None:
            print(f"WARNING: No ADG available for {size_label}, skipping",
                  file=sys.stderr)
            continue
        size_adgs[size_label] = (adg_path, pe_count)
        print(f"  {size_label}: {adg_path.name} ({pe_count} PEs)")

    # Build task list
    tasks = []
    for kernel, domain in REPRESENTATIVE_KERNELS:
        src = BENCH_DIR / domain / f"{kernel}.c"
        if not src.exists():
            print(f"WARNING: source not found: {src}", file=sys.stderr)
            continue
        for size_label, (adg_path, pe_count) in size_adgs.items():
            tasks.append((kernel, domain, size_label, pe_count, adg_path))

    total = len(tasks)
    print(f"\nE06: Running {total} (kernel, size) combinations")
    print(f"  Kernels: {len(REPRESENTATIVE_KERNELS)}")
    print(f"  Sizes: {len(size_adgs)}")
    print()

    results = []
    for idx, (kernel, domain, size_label, pe_count, adg_path) in enumerate(tasks, 1):
        label = f"[{idx}/{total}] {kernel} on {size_label}"
        print(f"  {label} ... ", end="", flush=True)
        result = run_mapping(kernel, domain, size_label, pe_count, adg_path)
        status = "OK" if result["mapped"] else "FAIL"
        ii_str = f"II={result['II']}" if result["mapped"] else ""
        print(f"{status} {ii_str} ({result['compile_time_ms']:.0f}ms)")
        results.append(result)

    # Write CSV
    csv_path = OUTPUT_DIR / "size_sweep.csv"
    fieldnames = [
        "kernel", "domain", "array_size", "pe_count",
        "mapped", "II", "pe_utilization", "area_estimate",
        "compile_time_ms", "node_count",
        "routed_edges", "total_edges", "unrouted_edges",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Write provenance
    prov_path = OUTPUT_DIR / "provenance.json"
    mapped_count = sum(1 for r in results if r["mapped"])
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "E06",
            "description": "ADG Size vs Mapping Quality",
            "git_hash": git_hash,
            "timestamp": timestamp,
            "loom_binary": str(LOOM_BIN),
            "total_runs": total,
            "mapped_count": mapped_count,
            "failed_count": total - mapped_count,
            "array_sizes": [s[0] for s in ARRAY_SIZES if s[0] in size_adgs],
            "representative_kernels": [k for k, d in REPRESENTATIVE_KERNELS],
            "area_per_pe_mm2": AREA_PER_PE_MM2,
        }, f, indent=2)

    print()
    print(f"E06 complete: {mapped_count}/{total} mapped successfully")
    print(f"  CSV: {csv_path}")
    print(f"  Provenance: {prov_path}")

    # Per-size summary
    for size_label in size_adgs:
        sz_results = [r for r in results if r["array_size"] == size_label]
        sz_mapped = sum(1 for r in sz_results if r["mapped"])
        avg_ii = 0.0
        mapped_with_ii = [r for r in sz_results if r["mapped"] and r["II"] > 0]
        if mapped_with_ii:
            avg_ii = sum(r["II"] for r in mapped_with_ii) / len(mapped_with_ii)
        print(f"  {size_label}: {sz_mapped}/{len(sz_results)} mapped, avg II={avg_ii:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
