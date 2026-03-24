#!/usr/bin/env python3
"""E25: Multi-Kernel Application Pipelines.

Evaluates pipeline parallelism achievable on 6 application TDGs by computing
pipeline schedules from real per-kernel mapping results (IIs). Each domain
defines a directed pipeline graph where kernels execute on separate cores
with inter-core communication via the NoC.

Pipeline throughput = 1 / max(stage_II) (bottleneck-limited).
Sequential throughput = 1 / sum(stage_II).
Pipeline speedup = sum(stage_II) / max(stage_II).

NoC overhead is estimated from the NoCScheduler model: each inter-core
transfer adds latency proportional to hop count and data volume.

Usage:
    python3 scripts/experiments/run_e25_pipeline.py
"""

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CGRA_CLOCK_MHZ = 500

# Pipeline definitions per domain (using TDG descriptions from benchmarks/)
# Each pipeline lists stages in execution order with their properties.
# IIs are from real mapping results where available; estimated otherwise.
PIPELINES = {
    "ai_llm": {
        "label": "Transformer Layer",
        "num_cores": 4,
        "stages": [
            {"name": "qkv_proj",     "ii": 4,  "nodes": 65,  "mapped": True,
             "data_out_bytes": 2048 * 4},
            {"name": "attn_score",   "ii": 8,  "nodes": 97,  "mapped": True,
             "data_out_bytes": 4096 * 4},
            {"name": "softmax",      "ii": 4,  "nodes": 26,  "mapped": True,
             "data_out_bytes": 4096 * 4},
            {"name": "attn_output",  "ii": 3,  "nodes": 64,  "mapped": True,
             "data_out_bytes": 16384 * 4},
            {"name": "ffn1",         "ii": 8,  "nodes": 73,  "mapped": True,
             "data_out_bytes": 65536 * 4},
            {"name": "gelu",         "ii": 29, "nodes": 73,  "mapped": True,
             "data_out_bytes": 65536 * 4},
            {"name": "ffn2",         "ii": 8,  "nodes": 73,  "mapped": True,
             "data_out_bytes": 16384 * 4},
            {"name": "layernorm",    "ii": 2,  "nodes": 42,  "mapped": True,
             "data_out_bytes": 16384 * 4},
        ],
    },
    "dsp_ofdm": {
        "label": "OFDM Receiver Chain",
        "num_cores": 4,
        "stages": [
            {"name": "fft_butterfly", "ii": 0, "nodes": 0,   "mapped": False,
             "data_out_bytes": 4096 * 8},
            {"name": "channel_est",   "ii": 0, "nodes": 0,   "mapped": False,
             "data_out_bytes": 1200 * 8},
            {"name": "equalizer",     "ii": 0, "nodes": 0,   "mapped": False,
             "data_out_bytes": 1200 * 8},
            {"name": "qam_demod",     "ii": 0, "nodes": 0,   "mapped": False,
             "data_out_bytes": 7200 * 4},
            {"name": "viterbi",       "ii": 0, "nodes": 0,   "mapped": False,
             "data_out_bytes": 1800 * 4},
            {"name": "crc_check",     "ii": 2, "nodes": 23,  "mapped": True,
             "data_out_bytes": 1800 * 4},
        ],
    },
    "arvr_stereo": {
        "label": "Stereo Vision Pipeline",
        "num_cores": 4,
        "stages": [
            {"name": "harris_corner",    "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 4096 * 4},
            {"name": "sad_matching",     "ii": 4, "nodes": 95, "mapped": True,
             "data_out_bytes": 262144 * 4},
            {"name": "stereo_disparity", "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 4096 * 4},
            {"name": "image_warp",       "ii": 3, "nodes": 66, "mapped": True,
             "data_out_bytes": 4096 * 4},
            {"name": "post_filter",      "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 4096 * 4},
        ],
    },
    "robotics_vio": {
        "label": "VIO Pipeline",
        "num_cores": 4,
        "stages": [
            {"name": "fast_detect",     "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 1000 * 4},
            {"name": "orb_descriptor",  "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 4000 * 4},
            {"name": "feature_match",   "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 400 * 4},
            {"name": "pose_estimate",   "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 600 * 4},
            {"name": "imu_integration", "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 600 * 4},
        ],
    },
    "graph_analytics": {
        "label": "Graph Analytics Pipeline",
        "num_cores": 4,
        "stages": [
            {"name": "bfs_traversal",   "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 1024 * 4},
            {"name": "pagerank_spmv",   "ii": 0, "nodes": 0,  "mapped": False,
             "data_out_bytes": 1024 * 4},
            {"name": "label_prop",      "ii": 2, "nodes": 38, "mapped": True,
             "data_out_bytes": 1024 * 4},
            {"name": "triangle_count",  "ii": 2, "nodes": 31, "mapped": True,
             "data_out_bytes": 1024 * 4},
        ],
    },
    "zk_stark": {
        "label": "STARK Proof Pipeline",
        "num_cores": 4,
        "stages": [
            {"name": "ntt",            "ii": 0,  "nodes": 0,  "mapped": False,
             "data_out_bytes": 1024 * 4},
            {"name": "poly_eval",      "ii": 15, "nodes": 82, "mapped": True,
             "data_out_bytes": 256 * 4},
            {"name": "poseidon_hash",  "ii": 0,  "nodes": 0,  "mapped": False,
             "data_out_bytes": 4 * 4},
            {"name": "proof_compose",  "ii": 0,  "nodes": 0,  "mapped": False,
             "data_out_bytes": 3 * 4},
            {"name": "msm",           "ii": 0,  "nodes": 0,  "mapped": False,
             "data_out_bytes": 3 * 4},
        ],
    },
}

# NoC parameters
NOC_BANDWIDTH_BYTES_PER_CYCLE = 32  # 256-bit links at 500MHz
NOC_HOP_LATENCY_CYCLES = 3
AVG_HOP_COUNT = 2.0  # Average for 2x2 mesh

# In pipelined execution, each II iteration transfers a small tile chunk,
# not the entire output. The per-iteration transfer size is the SPM tile
# that fits in double-buffered local memory (typically 256-2048 bytes).
# data_out_bytes in the stage definition is the total tile production;
# the per-II transfer is data_out_bytes / (total_iterations_per_tile).
# For simplicity, we model the per-II transfer as a fixed fraction.
PER_II_TRANSFER_BYTES = 512  # Typical SPM tile chunk per pipeline iteration


def git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def compute_noc_overhead_per_ii(data_bytes_per_ii):
    """Compute NoC transfer latency per pipeline iteration in cycles.

    In steady-state pipelining, NoC transfers overlap with computation.
    The overhead is the non-overlappable portion: hop latency + any
    serialization beyond what the II can hide.
    """
    transfer_cycles = math.ceil(data_bytes_per_ii / NOC_BANDWIDTH_BYTES_PER_CYCLE)
    hop_latency = NOC_HOP_LATENCY_CYCLES * AVG_HOP_COUNT
    # In steady state, transfer and hop latency overlap with next II's compute.
    # Only the excess beyond the II contributes as overhead.
    return transfer_cycles + hop_latency


def analyze_pipeline(domain, pipeline):
    """Analyze pipeline throughput and scheduling."""
    stages = pipeline["stages"]
    num_stages = len(stages)
    mapped_stages = [s for s in stages if s["mapped"]]
    num_mapped = len(mapped_stages)

    if num_mapped == 0:
        return {
            "domain": domain,
            "label": pipeline["label"],
            "num_stages": num_stages,
            "num_mapped": 0,
            "fully_mappable": False,
            "note": "No kernels mapped; pipeline analysis not possible",
        }

    # Per-stage cycles (II) for mapped stages
    stage_iis = [s["ii"] for s in stages if s["mapped"]]

    # Sequential total: sum of all stage IIs
    sequential_cycles = sum(stage_iis)

    # Pipeline bottleneck: max II across mapped stages
    bottleneck_ii = max(stage_iis)
    bottleneck_name = [s["name"] for s in stages if s["mapped"] and s["ii"] == bottleneck_ii][0]

    # NoC overhead model for pipelined execution:
    # Each inter-core link has a per-tile-chunk transfer. In steady state,
    # the NoC transfer from stage i to stage i+1 overlaps with stage i+1's
    # computation. The effective pipeline rate is gated by the bottleneck II.
    #
    # Per-link non-overlapped overhead = max(0, noc_latency - bottleneck_ii)
    # because the bottleneck sets the pipeline period.
    noc_lat_per_link = compute_noc_overhead_per_ii(PER_II_TRANSFER_BYTES)
    noc_links = 0
    prev_mapped_idx = -1
    for i, s in enumerate(stages):
        if s["mapped"]:
            if prev_mapped_idx >= 0:
                noc_links += 1
            prev_mapped_idx = i

    # In steady state, all NoC transfers happen within one pipeline period.
    # Excess per link = max(0, noc_latency - bottleneck_ii)
    per_link_excess = max(0, noc_lat_per_link - bottleneck_ii)
    total_noc_excess = per_link_excess * noc_links

    # Pipeline throughput cycle = bottleneck II + total non-overlapped NoC
    pipeline_cycles = bottleneck_ii + total_noc_excess

    # Speedup
    pipeline_speedup = sequential_cycles / pipeline_cycles if pipeline_cycles > 0 else 1.0

    # NoC overhead percentage
    noc_pct = total_noc_excess / pipeline_cycles * 100 if pipeline_cycles > 0 else 0

    # Core utilization: fraction of total pipeline time each core is active
    # In pipeline: each stage is active for its II out of bottleneck_ii cycles
    utilization = sum(s["ii"] / bottleneck_ii for s in mapped_stages) / num_mapped * 100

    # Throughput in steady-state: all mapped stages producing every pipeline_cycles
    ops_per_cycle = sum(s["nodes"] for s in mapped_stages) / pipeline_cycles
    throughput_gops = ops_per_cycle * CGRA_CLOCK_MHZ / 1000.0

    return {
        "domain": domain,
        "label": pipeline["label"],
        "num_stages": num_stages,
        "num_mapped": num_mapped,
        "fully_mappable": num_mapped == num_stages,
        "stage_iis": stage_iis,
        "sequential_cycles": sequential_cycles,
        "bottleneck_ii": bottleneck_ii,
        "bottleneck_name": bottleneck_name,
        "noc_overhead_cycles": round(total_noc_excess, 1),
        "pipeline_cycles": round(pipeline_cycles, 1),
        "pipeline_speedup": round(pipeline_speedup, 2),
        "noc_overhead_pct": round(noc_pct, 1),
        "core_utilization_pct": round(utilization, 1),
        "throughput_gops": round(throughput_gops, 2),
        "noc_links_active": noc_links,
    }


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E25"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E25: Multi-Kernel Application Pipelines")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")

    pipeline_rows = []
    stage_detail_rows = []

    for domain, pipeline in PIPELINES.items():
        result = analyze_pipeline(domain, pipeline)

        print(f"\n  {domain} ({pipeline['label']}):")
        print(f"    Stages: {result['num_stages']}, Mapped: {result['num_mapped']}")

        if result["num_mapped"] == 0:
            print(f"    SKIPPED: {result.get('note', 'no mapped stages')}")
            pipeline_rows.append({
                "domain": domain,
                "num_stages": result["num_stages"],
                "num_mapped": result["num_mapped"],
                "fully_mappable": result["fully_mappable"],
                "total_cycles": 0,
                "per_stage_bottleneck_ii": 0,
                "pipeline_speedup_vs_sequential": 0,
                "noc_overhead_pct": 0,
                "core_utilization_pct": 0,
                "throughput_gops": 0,
                "note": result.get("note", ""),
                "git_hash": ghash,
                "timestamp": timestamp,
            })
            continue

        print(f"    Sequential: {result['sequential_cycles']} cycles")
        print(f"    Pipeline:   {result['pipeline_cycles']:.1f} cycles "
              f"(bottleneck: {result['bottleneck_name']} II={result['bottleneck_ii']})")
        print(f"    Speedup:    {result['pipeline_speedup']:.2f}x")
        print(f"    NoC overhead: {result['noc_overhead_pct']:.1f}%")
        print(f"    Core util:  {result['core_utilization_pct']:.1f}%")
        print(f"    Throughput: {result['throughput_gops']:.2f} Gops/s")

        pipeline_rows.append({
            "domain": domain,
            "num_stages": result["num_stages"],
            "num_mapped": result["num_mapped"],
            "fully_mappable": result["fully_mappable"],
            "total_cycles": result["pipeline_cycles"],
            "per_stage_bottleneck_ii": result["bottleneck_ii"],
            "pipeline_speedup_vs_sequential": result["pipeline_speedup"],
            "noc_overhead_pct": result["noc_overhead_pct"],
            "core_utilization_pct": result["core_utilization_pct"],
            "throughput_gops": result["throughput_gops"],
            "note": f"bottleneck={result['bottleneck_name']}",
            "git_hash": ghash,
            "timestamp": timestamp,
        })

        # Per-stage detail rows
        for stage in pipeline["stages"]:
            noc_cycles = compute_noc_overhead_per_ii(PER_II_TRANSFER_BYTES) if stage["mapped"] else 0
            stage_detail_rows.append({
                "domain": domain,
                "stage": stage["name"],
                "ii": stage["ii"],
                "nodes": stage["nodes"],
                "mapped": stage["mapped"],
                "data_out_bytes": stage["data_out_bytes"],
                "noc_transfer_cycles": noc_cycles,
                "git_hash": ghash,
                "timestamp": timestamp,
            })

    # Write pipeline results CSV
    csv_path = out_dir / "pipeline_results.csv"
    pipeline_fields = [
        "domain", "num_stages", "num_mapped", "fully_mappable",
        "total_cycles", "per_stage_bottleneck_ii",
        "pipeline_speedup_vs_sequential", "noc_overhead_pct",
        "core_utilization_pct", "throughput_gops", "note",
        "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pipeline_fields)
        writer.writeheader()
        writer.writerows(pipeline_rows)
    print(f"\n  Wrote {len(pipeline_rows)} pipeline rows to {csv_path}")

    # Write per-stage detail CSV
    detail_path = out_dir / "stage_details.csv"
    detail_fields = [
        "domain", "stage", "ii", "nodes", "mapped", "data_out_bytes",
        "noc_transfer_cycles", "git_hash", "timestamp",
    ]
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(stage_detail_rows)
    print(f"  Wrote {len(stage_detail_rows)} stage detail rows to {detail_path}")

    # Summary
    analyzed = [r for r in pipeline_rows if r["throughput_gops"] > 0]
    if analyzed:
        avg_speedup = sum(r["pipeline_speedup_vs_sequential"] for r in analyzed) / len(analyzed)
        avg_noc = sum(r["noc_overhead_pct"] for r in analyzed) / len(analyzed)
        avg_util = sum(r["core_utilization_pct"] for r in analyzed) / len(analyzed)
        max_speedup = max(r["pipeline_speedup_vs_sequential"] for r in analyzed)
        best_domain = [r["domain"] for r in analyzed
                       if r["pipeline_speedup_vs_sequential"] == max_speedup][0]
    else:
        avg_speedup = avg_noc = avg_util = max_speedup = 0
        best_domain = "none"

    print(f"\n  Summary ({len(analyzed)} analyzable pipelines):")
    print(f"    Average pipeline speedup: {avg_speedup:.2f}x")
    print(f"    Average NoC overhead: {avg_noc:.1f}%")
    print(f"    Average core utilization: {avg_util:.1f}%")
    print(f"    Best speedup: {max_speedup:.2f}x ({best_domain})")

    # Domains with no mapped stages
    unmappable = [r["domain"] for r in pipeline_rows if r["throughput_gops"] == 0]
    if unmappable:
        print(f"    Domains with no pipeline analysis: {', '.join(unmappable)}")

    summary = {
        "experiment": "E25_multi_kernel_pipelines",
        "timestamp": timestamp,
        "git_hash": ghash,
        "num_domains": len(PIPELINES),
        "analyzable_domains": len(analyzed),
        "avg_pipeline_speedup": round(avg_speedup, 2),
        "avg_noc_overhead_pct": round(avg_noc, 1),
        "avg_core_utilization_pct": round(avg_util, 1),
        "best_speedup_domain": best_domain,
        "best_speedup": round(max_speedup, 2),
        "unmappable_domains": unmappable,
        "note": "Pipeline speedup is limited to mapped stages only. "
                "Domains with many unmapped kernels (robotics_vio, parts of dsp_ofdm) "
                "cannot form complete pipelines. NoC overhead from NoCScheduler model "
                "with 2x2 mesh, 256-bit links.",
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
