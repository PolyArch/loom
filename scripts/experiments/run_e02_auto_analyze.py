#!/usr/bin/env python3
"""E02: Auto-Analyze Accuracy -- compare auto_analyze output vs manual TDGs.

For each domain, runs auto_analyze on the pipeline entry function and
compares the detected kernels/edges against the manual reference TDG.

If the tapestry_auto_analyze binary is not available, falls back to
structural analysis of the pipeline C sources to simulate what
auto_analyze would detect (call graph extraction + shared pointer analysis).

Usage:
    python3 scripts/experiments/run_e02_auto_analyze.py
"""

import ast
import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Domain definitions with reference TDG info and pipeline source
DOMAINS = {
    "ai_llm": {
        "pipeline_src": "benchmarks/tapestry/ai_llm/e02_pipeline/transformer_pipeline.c",
        "entry_func": "transformer_pipeline",
        "reference_tdg": "benchmarks/tapestry/ai_llm/tdg_transformer.py",
    },
    "dsp_ofdm": {
        "pipeline_src": "benchmarks/tapestry/dsp_ofdm/e02_pipeline/ofdm_pipeline.c",
        "entry_func": "ofdm_pipeline",
        "reference_tdg": "benchmarks/tapestry/dsp_ofdm/tdg_ofdm.py",
    },
    "arvr_stereo": {
        "pipeline_src": "benchmarks/tapestry/arvr_stereo/e02_pipeline/stereo_pipeline.c",
        "entry_func": "stereo_pipeline",
        "reference_tdg": "benchmarks/tapestry/arvr_stereo/tdg_stereo.py",
    },
    "robotics_vio": {
        "pipeline_src": "benchmarks/tapestry/robotics_vio/e02_pipeline/vio_pipeline.c",
        "entry_func": "vio_pipeline",
        "reference_tdg": "benchmarks/tapestry/robotics_vio/tdg_vio.py",
    },
    "graph_analytics": {
        "pipeline_src": "benchmarks/tapestry/graph_analytics/e02_pipeline/graph_pipeline.c",
        "entry_func": "graph_pipeline",
        "reference_tdg": "benchmarks/tapestry/graph_analytics/tdg_graph.py",
        # Function name -> TDG kernel name mapping
        "name_map": {"bfs_tiled": "bfs_traversal"},
    },
    "zk_stark": {
        "pipeline_src": "benchmarks/tapestry/zk_stark/e02_pipeline/stark_pipeline.c",
        "entry_func": "stark_pipeline",
        "reference_tdg": "benchmarks/tapestry/zk_stark/tdg_stark.py",
        "name_map": {"ntt_forward": "ntt"},
    },
}


def git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_reference_tdg(tdg_path):
    """Load the reference TDG from a Python description file."""
    content = (REPO_ROOT / tdg_path).read_text()

    # Extract kernels list
    kernels_match = re.search(
        r'kernels\s*=\s*\[(.+?)\]', content, re.DOTALL)
    contracts_match = re.search(
        r'contracts\s*=\s*\[(.+?)\]\s*$', content, re.DOTALL | re.MULTILINE)

    kernel_names = set()
    if kernels_match:
        for m in re.finditer(r'"name"\s*:\s*"(\w+)"', kernels_match.group(1)):
            kernel_names.add(m.group(1))

    edges = set()
    if contracts_match:
        block = contracts_match.group(1)
        producers = re.findall(r'"producer"\s*:\s*"(\w+)"', block)
        consumers = re.findall(r'"consumer"\s*:\s*"(\w+)"', block)
        for p, c in zip(producers, consumers):
            edges.add((p, c))

    return kernel_names, edges


def analyze_c_source(src_path, entry_func):
    """Structural analysis of C source to extract call graph and data deps.

    This simulates what auto_analyze does at the LLVM IR level:
    1. Find all function calls in the entry function body
    2. Identify shared pointer arguments between calls
    """
    content = (REPO_ROOT / src_path).read_text()

    # Find entry function body
    entry_pattern = rf'void\s+{entry_func}\s*\([^)]*\)\s*\{{'
    entry_match = re.search(entry_pattern, content)
    if not entry_match:
        return set(), set()

    # Extract the function body (find matching brace)
    start = entry_match.end() - 1
    depth = 1
    pos = start + 1
    while pos < len(content) and depth > 0:
        if content[pos] == '{':
            depth += 1
        elif content[pos] == '}':
            depth -= 1
        pos += 1
    entry_body = content[start:pos]

    # Find all noinline function declarations (these are the kernels)
    kernel_funcs = set()
    for m in re.finditer(
            r'__attribute__\(\(noinline\)\)\s*'
            r'(?:void|int|float|unsigned)\s+(\w+)\s*\(',
            content):
        kernel_funcs.add(m.group(1))

    # Find function calls in entry body, in order
    call_pattern = re.compile(r'(\w+)\s*\(([^;]*?)\)\s*;')
    calls = []
    for m in call_pattern.finditer(entry_body):
        fname = m.group(1)
        if fname in kernel_funcs:
            args_str = m.group(2)
            # Extract argument variable names
            args = []
            for arg in args_str.split(','):
                arg = arg.strip()
                # Get the base variable name (remove casts, offsets, etc.)
                base = re.sub(r'\([^)]*\)', '', arg).strip()
                base = re.sub(r'\s*\+.*', '', base).strip()
                base = re.sub(r'\s*\*.*', '', base).strip()
                if base:
                    args.append(base)
            calls.append((fname, args))

    detected_kernels = set(c[0] for c in calls)

    # Detect edges: shared pointer arguments between consecutive calls
    detected_edges = set()
    for i in range(len(calls)):
        for j in range(i + 1, len(calls)):
            fname_i, args_i = calls[i]
            fname_j, args_j = calls[j]
            # Check for shared arguments (potential data dependency)
            shared = set(args_i) & set(args_j)
            # Filter out scalar-looking arguments
            shared = {a for a in shared if not a.isdigit()
                       and a not in ('0', 'NULL', '')
                       and not a.startswith('IMG_')
                       and not a.startswith('NUM_')
                       and not a.startswith('MAX_')
                       and not a.startswith('NTT_')
                       and not a.startswith('MSM_')
                       and not a.startswith('DATA_')
                       and not a.startswith('CODED_')
                       and not a.startswith('FFT_')
                       and not a.startswith('QAM_')}
            if shared:
                detected_edges.add((fname_i, fname_j))

    return detected_kernels, detected_edges


def try_binary_auto_analyze(src_path, entry_func):
    """Try running the actual auto_analyze binary if available."""
    binary = REPO_ROOT / "build" / "bin" / "tapestry_auto_analyze"
    if not binary.exists():
        return None

    try:
        result = subprocess.run(
            [str(binary), str(REPO_ROOT / src_path), entry_func, "--verbose"],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO_ROOT)
        )
        if result.returncode != 0:
            return None

        # Parse output
        kernels = set()
        edges = set()
        for line in result.stdout.split('\n'):
            km = re.match(r"\s*auto-analyze:\s+kernel\s+'(\w+)'", line)
            if km:
                kernels.add(km.group(1))
            em = re.match(
                r"\s*auto-analyze:\s+edge\s+(\w+)\s*->\s*(\w+)", line)
            if em:
                edges.add((em.group(1), em.group(2)))
        return kernels, edges
    except Exception:
        return None


def compute_match_rate(reference, detected):
    """Compute match rate between two sets."""
    if not reference:
        return 1.0 if not detected else 0.0
    matched = reference & detected
    return len(matched) / len(reference) if reference else 0.0


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E02"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "auto_vs_manual.csv"

    rows = []
    used_binary = False

    print("E02: Auto-Analyze Accuracy")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print()

    for domain_name, info in DOMAINS.items():
        # Load reference (manual) TDG
        ref_kernels, ref_edges = load_reference_tdg(info["reference_tdg"])

        # Try actual binary first, fall back to structural analysis
        binary_result = try_binary_auto_analyze(
            info["pipeline_src"], info["entry_func"])

        if binary_result is not None:
            auto_kernels, auto_edges = binary_result
            method = "binary"
            used_binary = True
        else:
            auto_kernels, auto_edges = analyze_c_source(
                info["pipeline_src"], info["entry_func"])
            method = "structural"

        # Apply name mapping (C function names -> TDG kernel names)
        name_map = info.get("name_map", {})
        if name_map:
            auto_kernels = {name_map.get(k, k) for k in auto_kernels}
            auto_edges = {
                (name_map.get(p, p), name_map.get(c, c))
                for p, c in auto_edges
            }

        # Compute match rates
        kernel_match = compute_match_rate(ref_kernels, auto_kernels)
        edge_match = compute_match_rate(ref_edges, auto_edges)
        extra_edges = auto_edges - ref_edges
        missing_edges = ref_edges - auto_edges

        row = {
            "domain": domain_name,
            "manual_kernels": len(ref_kernels),
            "auto_kernels": len(auto_kernels),
            "kernel_match": round(kernel_match, 4),
            "manual_edges": len(ref_edges),
            "auto_edges": len(auto_edges),
            "edge_match": round(edge_match, 4),
            "extra_edges": len(extra_edges),
            "missing_edges": len(missing_edges),
            "method": method,
            "git_hash": ghash,
            "timestamp": timestamp,
        }
        rows.append(row)

        print(f"  {domain_name:20s} [{method}]")
        print(f"    kernels: {len(auto_kernels)}/{len(ref_kernels)} "
              f"(match={kernel_match:.1%})")
        print(f"    edges:   {len(auto_edges)}/{len(ref_edges)} "
              f"(match={edge_match:.1%}), "
              f"+{len(extra_edges)} extra, -{len(missing_edges)} missing")

        if extra_edges:
            for e in sorted(extra_edges):
                print(f"      extra: {e[0]} -> {e[1]}")
        if missing_edges:
            for e in sorted(missing_edges):
                print(f"      missing: {e[0]} -> {e[1]}")

    # Write CSV
    fieldnames = [
        "domain", "manual_kernels", "auto_kernels", "kernel_match",
        "manual_edges", "auto_edges", "edge_match",
        "extra_edges", "missing_edges", "method",
        "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")
    if not used_binary:
        print("NOTE: Used structural analysis (auto_analyze binary not built).")
        print("      Results simulate what auto_analyze would detect based on")
        print("      call graph structure and shared pointer arguments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
