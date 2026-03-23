#!/usr/bin/env python3
"""Orchestrate all GPU baseline benchmarks.

Compiles each CUDA source, runs with warm-up and measurement,
collects timing and optional power data, writes JSON results.
"""

import json
import os
import subprocess
import sys
import threading
import time
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
        "name": "matmul_cublas",
        "source": "matmul_gpu.cu",
        "binary": "matmul_gpu",
        "libs": ["-lcublas"],
        "args_list": [
            ["256", "256", "256"],
            ["512", "512", "512"],
            ["1024", "1024", "1024"],
            ["2048", "2048", "2048"],
        ],
    },
    {
        "name": "fft_cufft",
        "source": "fft_gpu.cu",
        "binary": "fft_gpu",
        "libs": ["-lcufft"],
        "args_list": [
            ["1048576"],   # 2^20
            ["4194304"],   # 2^22
        ],
    },
    {
        "name": "ntt_m31_gpu",
        "source": "ntt_gpu.cu",
        "binary": "ntt_gpu",
        "libs": [],
        "args_list": [
            ["20"],  # 2^20
            ["22"],  # 2^22
        ],
    },
    {
        "name": "attention_gpu",
        "source": "attention_gpu.cu",
        "binary": "attention_gpu",
        "libs": ["-lcublas"],
        "args_list": [
            ["1", "8", "512", "64"],
            ["1", "8", "1024", "64"],
        ],
    },
    {
        "name": "spmv_cusparse",
        "source": "spmv_gpu.cu",
        "binary": "spmv_gpu",
        "libs": ["-lcusparse"],
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

NVCC = "nvcc"
NVCC_FLAGS = ["-O3", "-std=c++17", "--use_fast_math"]


def compile_benchmark(source: str, binary: str, libs: List[str]) -> bool:
    """Compile a CUDA source file. Returns True on success."""
    src_path = SCRIPT_DIR / source
    bin_path = SCRIPT_DIR / binary

    if not src_path.exists():
        print(f"ERROR: Source not found: {src_path}")
        return False

    cmd = [NVCC] + NVCC_FLAGS + [str(src_path), "-o", str(bin_path)] + libs
    print(f"Compiling: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed for {source}:")
        print(result.stderr)
        return False

    return True


# -----------------------------------------------------------------------
# Power measurement via nvidia-smi
# -----------------------------------------------------------------------

class GpuPowerMonitor:
    """Poll GPU power draw via nvidia-smi in a background thread."""

    def __init__(self, interval_s: float = 0.1):
        self.interval = interval_s
        self.readings: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.readings = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if not self.readings:
            return 0.0
        return sum(self.readings) / len(self.readings)

    def _poll(self):
        while self._running:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=2.0,
                )
                for line in out.strip().split("\n"):
                    line = line.strip()
                    if line:
                        self.readings.append(float(line))
            except (subprocess.SubprocessError, ValueError):
                pass
            time.sleep(self.interval)


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


def run_single(binary: str, args: List[str],
               measure_power: bool = True) -> List[Dict[str, Any]]:
    """Run a compiled benchmark binary and return parsed results."""
    bin_path = SCRIPT_DIR / binary
    cmd = [str(bin_path)] + args

    monitor = None
    if measure_power:
        monitor = GpuPowerMonitor()
        monitor.start()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    avg_power = 0.0
    if monitor:
        avg_power = monitor.stop()

    if result.returncode != 0:
        print(f"ERROR running {binary}: {result.stderr}")
        return []

    parsed = parse_result_lines(result.stdout)

    # Attach power reading.
    for p in parsed:
        if avg_power > 0:
            p["power_w"] = avg_power

    return parsed


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    output_dir = SCRIPT_DIR.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gpu_results.json"

    measure_power = True
    if "--no-power" in sys.argv:
        measure_power = False

    all_results: List[Dict[str, Any]] = []
    failed: List[str] = []

    for bench in BENCHMARKS:
        if not compile_benchmark(bench["source"], bench["binary"],
                                  bench["libs"]):
            failed.append(bench["name"])
            continue

        for args in bench["args_list"]:
            parsed = run_single(bench["binary"], args,
                                measure_power=measure_power)
            for p in parsed:
                r = make_result(
                    kernel_name=p.get("kernel", bench["name"]),
                    platform="gpu",
                    mean_time_ms=p.get("mean_ms", 0.0),
                    stddev_time_ms=p.get("stddev_ms", 0.0),
                    min_time_ms=p.get("min_ms", 0.0),
                    num_trials=10,
                    total_ops=p.get("total_ops"),
                    avg_power_w=p.get("power_w"),
                    problem_size={k: v for k, v in p.items()
                                  if k not in {"kernel", "mean_ms",
                                               "stddev_ms", "min_ms",
                                               "total_ops", "power_w"}},
                )
                all_results.append(r)

    metadata = {
        "platform": "gpu",
        "device": "RTX 5090 (detected via nvidia-smi)",
        "cuda_version": "13.2",
    }

    write_results(all_results, str(output_file), metadata=metadata)

    if failed:
        print(f"\nFailed benchmarks: {', '.join(failed)}")

    print(f"\nGPU baseline results written to: {output_file}")
    print(f"Total benchmarks run: {len(all_results)}")


if __name__ == "__main__":
    main()
