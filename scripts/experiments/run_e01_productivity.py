#!/usr/bin/env python3
"""E01: Productivity Comparison -- measure lines and annotation burden.

Counts lines in three TDG formats (DSL C++, hand-written MLIR, pragma C)
across all 6 benchmark domains. Computes manual annotation fractions.

Usage:
    python3 scripts/experiments/run_e01_productivity.py
"""

import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DOMAINS = {
    "ai_llm": {
        "dir": "benchmarks/tapestry/ai_llm",
        "num_kernels": 8,
        "num_edges": 7,
    },
    "dsp_ofdm": {
        "dir": "benchmarks/tapestry/dsp_ofdm",
        "num_kernels": 6,
        "num_edges": 5,
    },
    "arvr_stereo": {
        "dir": "benchmarks/tapestry/arvr_stereo",
        "num_kernels": 5,
        "num_edges": 4,
    },
    "robotics_vio": {
        "dir": "benchmarks/tapestry/robotics_vio",
        "num_kernels": 5,
        "num_edges": 4,
    },
    "graph_analytics": {
        "dir": "benchmarks/tapestry/graph_analytics",
        "num_kernels": 4,
        "num_edges": 3,
    },
    "zk_stark": {
        "dir": "benchmarks/tapestry/zk_stark",
        "num_kernels": 5,
        "num_edges": 5,
    },
}

FORMAT_FILES = {
    "dsl_cpp": "e01_formats/tdg_dsl.cpp",
    "mlir": "e01_formats/tdg_mlir.mlir",
    "pragma_c": "e01_formats/tdg_pragma.c",
}

# Contract fields that can be specified in each format
# "total" fields per edge: ordering, data_type, rate, tile_shape,
#   visibility, double_buffering, backpressure,
#   may_fuse, may_replicate, may_pipeline, may_reorder, may_retile
TOTAL_CONTRACT_FIELDS = 12


def git_hash():
    """Get current git short hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT)
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def count_lines(filepath):
    """Count non-blank, non-comment lines in a file."""
    total = 0
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue
            if stripped.startswith("*"):
                continue
            total += 1
    return total


def count_total_lines(filepath):
    """Count total lines (including blank and comments)."""
    with open(filepath, "r") as f:
        return sum(1 for _ in f)


def count_manual_contract_fields_dsl(filepath, num_edges):
    """Count manually specified contract fields in DSL format.

    In DSL, the user specifies only the fields they provide explicitly
    via chained setters (e.g., .ordering(), .data_type(), .rate()).
    Fields not set are inferred by the compiler.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Count chained setter calls on EdgeHandle
    setters = [
        r"\.ordering\(", r"\.data_type[<(]", r"\.rate\(",
        r"\.tile_shape\(", r"\.visibility\(", r"\.double_buffering\(",
        r"\.backpressure\(", r"\.may_fuse\(", r"\.may_replicate\(",
        r"\.may_pipeline\(", r"\.may_reorder\(", r"\.may_retile\(",
    ]
    count = 0
    for setter in setters:
        count += len(re.findall(setter, content))
    return count


def count_manual_contract_fields_mlir(filepath, num_edges):
    """Count manually specified contract fields in MLIR format.

    In hand-written MLIR, every field must be explicitly specified.
    """
    return num_edges * TOTAL_CONTRACT_FIELDS


def count_manual_contract_fields_pragma(filepath, num_edges):
    """Count manually specified contract fields in pragma format.

    In pragma format, every field must be explicitly written in the pragma.
    """
    return num_edges * TOTAL_CONTRACT_FIELDS


def count_kernel_source_lines(domain_dir, num_kernels):
    """Count total lines of kernel source files in the domain directory.

    These are the .c files that contain the actual kernel implementations,
    shared across all three formats (not counted as annotation overhead).
    """
    total = 0
    domain_path = REPO_ROOT / domain_dir
    for f in sorted(domain_path.glob("*.c")):
        total += count_total_lines(str(f))
    return total


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ghash = git_hash()

    out_dir = REPO_ROOT / "out" / "experiments" / "E01"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "productivity.csv"
    rows = []

    print(f"E01 Productivity Comparison")
    print(f"  git hash: {ghash}")
    print(f"  timestamp: {timestamp}")
    print()

    for domain_name, info in DOMAINS.items():
        domain_dir = info["dir"]
        num_edges = info["num_edges"]
        kernel_src_lines = count_kernel_source_lines(domain_dir, info["num_kernels"])

        for fmt_name, fmt_file in FORMAT_FILES.items():
            filepath = REPO_ROOT / domain_dir / fmt_file

            if not filepath.exists():
                print(f"  WARNING: {filepath} not found, skipping")
                continue

            tdg_lines = count_lines(str(filepath))
            total_lines = count_total_lines(str(filepath))

            if fmt_name == "dsl_cpp":
                manual_fields = count_manual_contract_fields_dsl(
                    str(filepath), num_edges)
            elif fmt_name == "mlir":
                manual_fields = count_manual_contract_fields_mlir(
                    str(filepath), num_edges)
            else:
                manual_fields = count_manual_contract_fields_pragma(
                    str(filepath), num_edges)

            total_fields = num_edges * TOTAL_CONTRACT_FIELDS
            manual_fraction = manual_fields / total_fields if total_fields > 0 else 0.0

            row = {
                "domain": domain_name,
                "format": fmt_name,
                "tdg_lines": tdg_lines,
                "total_lines": total_lines,
                "kernel_source_lines": kernel_src_lines,
                "contract_fields_manual": manual_fields,
                "contract_fields_total": total_fields,
                "manual_fraction": round(manual_fraction, 4),
                "num_kernels": info["num_kernels"],
                "num_edges": num_edges,
                "git_hash": ghash,
                "timestamp": timestamp,
            }
            rows.append(row)

            print(f"  {domain_name:20s} {fmt_name:12s}: "
                  f"tdg={tdg_lines:4d} lines, "
                  f"manual_fields={manual_fields:3d}/{total_fields:3d} "
                  f"({manual_fraction:.1%})")

    # Write CSV
    fieldnames = [
        "domain", "format", "tdg_lines", "total_lines",
        "kernel_source_lines", "contract_fields_manual",
        "contract_fields_total", "manual_fraction",
        "num_kernels", "num_edges", "git_hash", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
