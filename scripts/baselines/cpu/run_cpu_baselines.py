#!/usr/bin/env python3
"""Orchestrate all CPU baseline benchmarks.

Compiles each C++ source with OpenMP, runs with warm-up and measurement,
collects timing, writes JSON results.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "common"
sys.path.insert(0, str(COMMON_DIR))
from result_format import make_result, write_results

# -----------------------------------------------------------------------
# Benchmark definitions
# -----------------------------------------------------------------------

BENCHMARKS = [
    {
        "name": "matmul_cpu",
        "source": "matmul_cpu.cpp",
        "binary": "matmul_cpu",
        "args_list": [
            ["256", "256", "256"],
            ["512", "512", "512"],
            ["1024", "1024", "1024"],
            ["2048", "2048", "2048", "--skip-naive"],
        ],
    },
    {
        "name": "fft_cpu",
        "source": "fft_cpu.cpp",
        "binary": "fft_cpu",
        "args_list": [
            ["20"],  # 2^20
            ["22"],  # 2^22
        ],
    },
    {
        "name": "ntt_cpu",
        "source": "ntt_cpu.cpp",
        "binary": "ntt_cpu",
        "args_list": [
            ["20"],
            ["22"],
        ],
    },
    {
        "name": "attention_cpu",
        "source": "attention_cpu.cpp",
        "binary": "attention_cpu",
        "args_list": [
            ["1", "8", "256", "64"],
            ["1", "8", "512", "64"],
        ],
    },
    {
        "name": "spmv_cpu",
        "source": "spmv_cpu.cpp",
        "binary": "spmv_cpu",
        "args_list": [
            ["10000", "10000", "0.01"],
            ["10000", "10000", "0.05"],
            ["50000", "50000", "0.001"],
        ],
    },
]

# -----------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------

CXX = os.environ.get("CXX", "g++")
CXXFLAGS = ["-O3", "-std=c++17", "-fopenmp", "-march=native"]


def compile_benchmark(source: str, binary: str) -> bool:
    """Compile a C++ source file. Returns True on success."""
    src_path = SCRIPT_DIR / source
    bin_path = SCRIPT_DIR / binary
    inc_path = SCRIPT_DIR.parent  # For common/timer.h

    if not src_path.exists():
        print(f"ERROR: Source not found: {src_path}")
        return False

    cmd = [CXX] + CXXFLAGS + [
        f"-I{inc_path}", str(src_path), "-o", str(bin_path), "-lm"
    ]
    print(f"Compiling: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed for {source}:")
        print(result.stderr)
        return False

    return True


# -----------------------------------------------------------------------
# Benchmark execution
# -----------------------------------------------------------------------

def parse_result_lines(output: str) -> List[Dict[str, Any]]:
    """Parse RESULT lines from benchmark output."""
    results = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line.startswith("RESULT"):
            continue
        tokens = line.split()[1:]
        parsed: Dict[str, Any] = {}
        for tok in tokens:
            if "=" not in tok:
                continue
            key, val = tok.split("=", 1)
            try:
                parsed[key] = float(val)
            except ValueError:
                parsed[key] = val
        results.append(parsed)
    return results


def run_single(binary: str, args: List[str]) -> List[Dict[str, Any]]:
    """Run a compiled benchmark binary and return parsed results."""
    bin_path = SCRIPT_DIR / binary
    cmd = [str(bin_path)] + args

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"ERROR running {binary}: {result.stderr}")
        return []

    return parse_result_lines(result.stdout)


# -----------------------------------------------------------------------
# CPU power measurement (AMD RAPL via perf)
# -----------------------------------------------------------------------

def measure_cpu_power_rapl(binary: str, args: List[str]) -> Optional[float]:
    """Attempt to measure CPU package power via perf stat.

    Uses: perf stat -e power/energy-pkg/ <binary> <args>
    Returns average power in watts, or None if unavailable.
    """
    bin_path = SCRIPT_DIR / binary
    cmd = ["perf", "stat", "-e", "power/energy-pkg/",
           str(bin_path)] + args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=300)
        # Parse perf output for energy.
        for line in result.stderr.split("\n"):
            if "energy-pkg" in line.lower() or "joules" in line.lower():
                parts = line.strip().split()
                for part in parts:
                    try:
                        energy_j = float(part.replace(",", ""))
                        # Very rough: we don't know the exact duration here.
                        return energy_j
                    except ValueError:
                        continue
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    output_dir = SCRIPT_DIR.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cpu_results.json"

    all_results: List[Dict[str, Any]] = []
    failed: List[str] = []

    for bench in BENCHMARKS:
        if not compile_benchmark(bench["source"], bench["binary"]):
            failed.append(bench["name"])
            continue

        for args in bench["args_list"]:
            parsed = run_single(bench["binary"], args)
            for p in parsed:
                r = make_result(
                    kernel_name=p.get("kernel", bench["name"]),
                    platform="cpu",
                    mean_time_ms=p.get("mean_ms", 0.0),
                    stddev_time_ms=p.get("stddev_ms", 0.0),
                    min_time_ms=p.get("min_ms", 0.0),
                    num_trials=10,
                    total_ops=p.get("total_ops"),
                    problem_size={k: v for k, v in p.items()
                                  if k not in {"kernel", "mean_ms",
                                               "stddev_ms", "min_ms",
                                               "total_ops"}},
                )
                all_results.append(r)

    metadata = {
        "platform": "cpu",
        "device": "AMD Ryzen 9950X3D",
        "compiler": CXX,
        "flags": " ".join(CXXFLAGS),
    }

    write_results(all_results, str(output_file), metadata=metadata)

    if failed:
        print(f"\nFailed benchmarks: {', '.join(failed)}")

    print(f"\nCPU baseline results written to: {output_file}")
    print(f"Total benchmarks run: {len(all_results)}")


if __name__ == "__main__":
    main()
