#!/usr/bin/env python3
"""E04: Per-Kernel Mapping Panorama.

Maps each benchmark kernel onto each core type ADG and collects
mapping quality metrics. The Loom mapper is invoked via the `loom` CLI.

Output: out/experiments/E04/mapping_matrix.csv
"""

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
LOOM_BIN = REPO_ROOT / "build" / "bin" / "loom"
ADG_DIR = REPO_ROOT / "out" / "adg_library"
BENCH_DIR = REPO_ROOT / "benchmarks" / "tapestry"
COMMON_INC = BENCH_DIR / "common"
OUTPUT_DIR = REPO_ROOT / "out" / "experiments" / "E04"
WORK_DIR = OUTPUT_DIR / "runs"

CORE_TYPES = {
    "ctrl": {"adg": "ctrl_core.fabric.mlir", "pe_count": 16},
    "gp":   {"adg": "gp_core.fabric.mlir",   "pe_count": 36},
    "dsp":  {"adg": "dsp_core.fabric.mlir",   "pe_count": 36},
    "ai":   {"adg": "ai_core.fabric.mlir",    "pe_count": 64},
}

KERNEL_DOMAINS = {
    "ai_llm": [
        "qkv_proj", "attn_score", "softmax", "attn_output",
        "ffn1", "gelu", "ffn2", "layernorm",
    ],
    "dsp_ofdm": [
        "fft_butterfly", "channel_est", "equalizer",
        "qam_demod", "viterbi", "crc_check",
    ],
    "arvr_stereo": [
        "harris_corner", "sad_matching", "stereo_disparity",
        "image_warp", "post_filter",
    ],
    "robotics_vio": [
        "imu_integration", "fast_detect", "orb_descriptor",
        "feature_match", "pose_estimate",
    ],
    "graph_analytics": [
        "bfs_traversal", "pagerank_spmv", "triangle_count", "label_prop",
    ],
    "zk_stark": [
        "ntt", "msm", "poseidon_hash", "poly_eval", "proof_compose",
    ],
}

# Extra include paths required per domain
DOMAIN_EXTRA_INCLUDES = {
    "zk_stark": [str(BENCH_DIR / "zk_stark")],
}


@dataclass
class MappingResult:
    kernel: str
    domain: str
    core_type: str
    mapped: bool
    ii: int
    node_count: int
    pe_utilization: float
    coverage: float
    compile_time_ms: float
    routed_edges: int
    total_edges: int
    unrouted_edges: int
    failure_stage: str
    failure_detail: str


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_kernel_source(domain: str, kernel: str) -> Path:
    return BENCH_DIR / domain / f"{kernel}.c"


def build_include_args(domain: str) -> list[str]:
    args = ["-I", str(COMMON_INC)]
    for inc in DOMAIN_EXTRA_INCLUDES.get(domain, []):
        args.extend(["-I", inc])
    return args


def run_mapping(kernel: str, domain: str, core_type: str,
                budget_sec: int = 60) -> MappingResult:
    """Run loom mapper for one (kernel, core_type) pair."""
    source = get_kernel_source(domain, kernel)
    adg_path = ADG_DIR / CORE_TYPES[core_type]["adg"]
    pe_count = CORE_TYPES[core_type]["pe_count"]
    run_dir = WORK_DIR / core_type / kernel
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

    # Try to find map.json
    adg_stem = CORE_TYPES[core_type]["adg"].replace(".fabric.mlir", "")
    map_json_candidates = [
        run_dir / f"{kernel}.{adg_stem}.map.json",
        run_dir / f"{kernel}.map.json",
    ]
    map_json_path = None
    for candidate in map_json_candidates:
        if candidate.exists():
            map_json_path = candidate
            break

    if map_json_path is None or return_code != 0:
        # Determine failure cause from stderr
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        combined = stderr_text + stdout_text
        failure_stage, failure_detail = classify_failure(combined, timed_out)
        return MappingResult(
            kernel=kernel, domain=domain, core_type=core_type,
            mapped=False, ii=0, node_count=0, pe_utilization=0.0,
            coverage=0.0, compile_time_ms=elapsed_ms,
            routed_edges=0, total_edges=0, unrouted_edges=0,
            failure_stage=failure_stage, failure_detail=failure_detail,
        )

    # Parse map.json
    with open(map_json_path, encoding="utf-8") as f:
        mapping = json.load(f)

    timing = mapping.get("timing", {})
    ii = timing.get("estimated_initiation_interval", 0)
    node_mappings = mapping.get("node_mappings", [])
    node_count = len(node_mappings)
    edge_routings = mapping.get("edge_routings", [])
    routed = sum(1 for e in edge_routings if e.get("kind") == "routed")
    unrouted = sum(1 for e in edge_routings if e.get("kind") == "unrouted")
    total = len(edge_routings)

    # PE utilization: unique PEs used / total PEs
    pe_names = set()
    for nm in node_mappings:
        pn = nm.get("pe_name", "")
        if pn:
            pe_names.add(pn)
    pe_util = len(pe_names) / pe_count if pe_count > 0 else 0.0

    techmap = mapping.get("techmap", {})
    coverage = techmap.get("coverage_score", 0.0)

    mapped = (unrouted == 0) and (return_code == 0)

    failure_stage = ""
    failure_detail = ""
    if not mapped:
        if unrouted > 0:
            failure_stage = "ROUTING_FAIL"
            failure_detail = f"{unrouted}/{total} edges unrouted"

    return MappingResult(
        kernel=kernel, domain=domain, core_type=core_type,
        mapped=mapped, ii=ii, node_count=node_count,
        pe_utilization=round(pe_util, 4),
        coverage=round(coverage, 4),
        compile_time_ms=round(elapsed_ms, 1),
        routed_edges=routed, total_edges=total, unrouted_edges=unrouted,
        failure_stage=failure_stage, failure_detail=failure_detail,
    )


def classify_failure(output: str, timed_out: bool) -> tuple[str, str]:
    """Classify a failed mapping into failure category."""
    if timed_out:
        return "TIMEOUT", "Mapper exceeded time budget"
    lower = output.lower()
    if "techmap" in lower and ("fail" in lower or "no candidate" in lower):
        return "TECHMAP_FAIL", "Technology mapping could not find FU candidates"
    if "placement" in lower and "fail" in lower:
        return "PLACEMENT_FAIL", "Placement failed"
    if "routing" in lower and "fail" in lower:
        return "ROUTING_FAIL", "Routing failed"
    if "frontend" in lower or "compile" in lower and "error" in lower:
        return "COMPILE_FAIL", "Frontend/DFG compilation error"
    if "error" in lower or "assert" in lower:
        return "COMPILE_FAIL", "Compilation infrastructure error"
    return "UNKNOWN", "Unclassified failure"


def main() -> int:
    if not LOOM_BIN.exists():
        print(f"ERROR: loom binary not found at {LOOM_BIN}", file=sys.stderr)
        print("Run: ninja -C build loom tapestry_adg_gen", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    git_hash = get_git_hash()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build list of all (kernel, domain, core_type) combinations
    tasks = []
    for domain, kernels in KERNEL_DOMAINS.items():
        for kernel in kernels:
            src = get_kernel_source(domain, kernel)
            if not src.exists():
                print(f"WARNING: source not found: {src}", file=sys.stderr)
                continue
            for core_type in CORE_TYPES:
                tasks.append((kernel, domain, core_type))

    total = len(tasks)
    print(f"E04: Mapping {total} (kernel, core_type) combinations")
    print(f"  Kernels: {sum(len(ks) for ks in KERNEL_DOMAINS.values())}")
    print(f"  Core types: {len(CORE_TYPES)}")
    print(f"  Git: {git_hash}")
    print()

    results: list[MappingResult] = []
    for idx, (kernel, domain, core_type) in enumerate(tasks, 1):
        label = f"[{idx}/{total}] {kernel} on {core_type}"
        print(f"  {label} ... ", end="", flush=True)
        result = run_mapping(kernel, domain, core_type, budget_sec=60)
        status = "OK" if result.mapped else f"FAIL({result.failure_stage})"
        ii_str = f"II={result.ii}" if result.mapped else ""
        print(f"{status} {ii_str} ({result.compile_time_ms:.0f}ms)")
        results.append(result)

    # Write CSV
    csv_path = OUTPUT_DIR / "mapping_matrix.csv"
    fieldnames = [
        "kernel", "domain", "core_type", "mapped", "II", "node_count",
        "pe_utilization", "coverage", "compile_time_ms",
        "routed_edges", "total_edges", "unrouted_edges",
        "failure_stage", "failure_detail",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "kernel": r.kernel,
                "domain": r.domain,
                "core_type": r.core_type,
                "mapped": r.mapped,
                "II": r.ii,
                "node_count": r.node_count,
                "pe_utilization": r.pe_utilization,
                "coverage": r.coverage,
                "compile_time_ms": r.compile_time_ms,
                "routed_edges": r.routed_edges,
                "total_edges": r.total_edges,
                "unrouted_edges": r.unrouted_edges,
                "failure_stage": r.failure_stage,
                "failure_detail": r.failure_detail,
            })

    # Write provenance
    prov_path = OUTPUT_DIR / "provenance.json"
    mapped_count = sum(1 for r in results if r.mapped)
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "E04",
            "description": "Per-Kernel Mapping Panorama",
            "git_hash": git_hash,
            "timestamp": timestamp,
            "loom_binary": str(LOOM_BIN),
            "total_runs": total,
            "mapped_count": mapped_count,
            "failed_count": total - mapped_count,
        }, f, indent=2)

    print()
    print(f"E04 complete: {mapped_count}/{total} mapped successfully")
    print(f"  CSV: {csv_path}")
    print(f"  Provenance: {prov_path}")

    # Per-core success summary
    for ct in CORE_TYPES:
        ct_results = [r for r in results if r.core_type == ct]
        ct_mapped = sum(1 for r in ct_results if r.mapped)
        print(f"  {ct}: {ct_mapped}/{len(ct_results)} mapped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
