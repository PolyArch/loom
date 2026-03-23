#!/usr/bin/env python3
"""Cross-platform comparison script for CGRA / GPU / CPU baseline results.

Reads JSON results produced by run_gpu_baselines.py and run_cpu_baselines.py
(and optionally CGRA results), applies technology normalization, and produces
a summary comparison table.

Technology assumptions:
  - CGRA: SAED14nm library, 14nm equivalent
  - GPU (RTX 5090): TSMC 4nm (N4)
  - CPU (Ryzen 9950X3D): TSMC 5nm (N5, Zen5 core)
  - Normalization: scale raw perf/W by (tech_nm / 14nm)^2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR / "common"
sys.path.insert(0, str(COMMON_DIR))
from result_format import read_results

# -----------------------------------------------------------------------
# Technology normalization parameters
# -----------------------------------------------------------------------

TECH_NM = {
    "cgra": 14.0,
    "gpu": 4.0,
    "cpu": 5.0,
}

REFERENCE_NM = 14.0  # Normalize to this node.


def tech_normalization_factor(platform: str) -> float:
    """Compute (tech_nm / ref_nm)^2 scaling factor.

    Higher factor means the platform benefits from a more advanced node,
    so its raw perf/W is adjusted downward for fair comparison.
    """
    nm = TECH_NM.get(platform, REFERENCE_NM)
    return (nm / REFERENCE_NM) ** 2


# -----------------------------------------------------------------------
# Load results
# -----------------------------------------------------------------------

def load_results_by_kernel(path: str) -> Dict[str, Dict[str, Any]]:
    """Load a results JSON and index by kernel_name."""
    data = read_results(path)
    by_kernel: Dict[str, Dict[str, Any]] = {}
    for r in data.get("results", []):
        name = r.get("kernel_name", "unknown")
        # If multiple entries for same kernel, keep the one with lowest time.
        if name not in by_kernel or r.get("mean_time_ms", 1e30) < by_kernel[name].get("mean_time_ms", 1e30):
            by_kernel[name] = r
    return by_kernel


# -----------------------------------------------------------------------
# Comparison logic
# -----------------------------------------------------------------------

def compare_kernel(
    kernel_name: str,
    cgra: Optional[Dict[str, Any]],
    gpu: Optional[Dict[str, Any]],
    cpu: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build comparison record for one kernel across platforms."""
    entry: Dict[str, Any] = {"kernel": kernel_name}

    for label, data, platform in [("cgra", cgra, "cgra"),
                                   ("gpu", gpu, "gpu"),
                                   ("cpu", cpu, "cpu")]:
        if data is None:
            continue

        time_ms = data.get("mean_time_ms", 0.0)
        throughput = data.get("throughput_ops_sec", 0.0)
        power_w = data.get("avg_power_w")
        energy_j = data.get("energy_j")

        entry[f"{label}_time_ms"] = time_ms
        entry[f"{label}_throughput_ops_sec"] = throughput

        if power_w and power_w > 0:
            entry[f"{label}_power_w"] = power_w
            raw_perf_per_watt = throughput / power_w if power_w > 0 else 0
            norm_factor = tech_normalization_factor(platform)
            entry[f"{label}_raw_perf_per_watt"] = raw_perf_per_watt
            entry[f"{label}_norm_perf_per_watt"] = raw_perf_per_watt * norm_factor

        if energy_j is not None:
            entry[f"{label}_energy_j"] = energy_j

    return entry


def build_comparison(
    cgra_results: Optional[Dict[str, Dict[str, Any]]],
    gpu_results: Optional[Dict[str, Dict[str, Any]]],
    cpu_results: Optional[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Build full comparison table across all kernels."""
    all_kernels = set()
    for results in [cgra_results, gpu_results, cpu_results]:
        if results:
            all_kernels.update(results.keys())

    comparison = []
    for kernel in sorted(all_kernels):
        entry = compare_kernel(
            kernel,
            cgra_results.get(kernel) if cgra_results else None,
            gpu_results.get(kernel) if gpu_results else None,
            cpu_results.get(kernel) if cpu_results else None,
        )
        comparison.append(entry)

    return comparison


# -----------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------

def print_table(comparison: List[Dict[str, Any]]) -> None:
    """Print a human-readable comparison table."""
    header = (
        f"{'Kernel':<30} "
        f"{'GPU ms':>10} {'CPU ms':>10} {'CGRA ms':>10} "
        f"{'GPU ops/s':>14} {'CPU ops/s':>14} {'CGRA ops/s':>14}"
    )
    print("=" * len(header))
    print("TECHNOLOGY NORMALIZATION ASSUMPTIONS:")
    print(f"  CGRA: {TECH_NM['cgra']:.0f}nm (SAED14nm)")
    print(f"  GPU:  {TECH_NM['gpu']:.0f}nm  (TSMC N4, RTX 5090)")
    print(f"  CPU:  {TECH_NM['cpu']:.0f}nm  (TSMC N5, Zen5)")
    print(f"  Normalization: raw perf/W * (tech_nm / {REFERENCE_NM:.0f}nm)^2")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for entry in comparison:
        kernel = entry["kernel"]
        gpu_ms = entry.get("gpu_time_ms", 0)
        cpu_ms = entry.get("cpu_time_ms", 0)
        cgra_ms = entry.get("cgra_time_ms", 0)
        gpu_ops = entry.get("gpu_throughput_ops_sec", 0)
        cpu_ops = entry.get("cpu_throughput_ops_sec", 0)
        cgra_ops = entry.get("cgra_throughput_ops_sec", 0)

        def fmt_ms(v):
            return f"{v:10.3f}" if v else f"{'N/A':>10}"

        def fmt_ops(v):
            if not v:
                return f"{'N/A':>14}"
            if v >= 1e12:
                return f"{v / 1e12:11.2f} T/s"
            if v >= 1e9:
                return f"{v / 1e9:11.2f} G/s"
            if v >= 1e6:
                return f"{v / 1e6:11.2f} M/s"
            return f"{v:14.0f}"

        print(f"{kernel:<30} "
              f"{fmt_ms(gpu_ms)} {fmt_ms(cpu_ms)} {fmt_ms(cgra_ms)} "
              f"{fmt_ops(gpu_ops)} {fmt_ops(cpu_ops)} {fmt_ops(cgra_ops)}")

    # Power/efficiency table if available.
    has_power = any(
        "gpu_norm_perf_per_watt" in e or "cpu_norm_perf_per_watt" in e
        for e in comparison
    )
    if has_power:
        print()
        print("NORMALIZED PERF/WATT (adjusted for technology node):")
        hdr2 = f"{'Kernel':<30} {'GPU norm':>14} {'CPU norm':>14} {'CGRA norm':>14}"
        print(hdr2)
        print("-" * len(hdr2))
        for entry in comparison:
            kernel = entry["kernel"]

            def fmt_pw(key):
                v = entry.get(key, 0)
                if not v:
                    return f"{'N/A':>14}"
                if v >= 1e9:
                    return f"{v / 1e9:11.2f} G"
                if v >= 1e6:
                    return f"{v / 1e6:11.2f} M"
                return f"{v:14.2f}"

            print(f"{kernel:<30} "
                  f"{fmt_pw('gpu_norm_perf_per_watt')} "
                  f"{fmt_pw('cpu_norm_perf_per_watt')} "
                  f"{fmt_pw('cgra_norm_perf_per_watt')}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare CGRA/GPU/CPU baseline benchmark results.")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Path to gpu_results.json")
    parser.add_argument("--cpu", type=str, default=None,
                        help="Path to cpu_results.json")
    parser.add_argument("--cgra", type=str, default=None,
                        help="Path to cgra_results.json")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output comparison JSON file")
    args = parser.parse_args()

    # Auto-detect default paths.
    results_dir = SCRIPT_DIR / "results"
    if args.gpu is None and (results_dir / "gpu_results.json").exists():
        args.gpu = str(results_dir / "gpu_results.json")
    if args.cpu is None and (results_dir / "cpu_results.json").exists():
        args.cpu = str(results_dir / "cpu_results.json")
    if args.cgra is None and (results_dir / "cgra_results.json").exists():
        args.cgra = str(results_dir / "cgra_results.json")

    if not any([args.gpu, args.cpu, args.cgra]):
        print("No result files found. Run GPU/CPU baselines first, or specify paths.")
        print("Expected: results/gpu_results.json, results/cpu_results.json")
        sys.exit(1)

    gpu_results = load_results_by_kernel(args.gpu) if args.gpu else None
    cpu_results = load_results_by_kernel(args.cpu) if args.cpu else None
    cgra_results = load_results_by_kernel(args.cgra) if args.cgra else None

    comparison = build_comparison(cgra_results, gpu_results, cpu_results)

    print_table(comparison)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump({
                "technology_assumptions": {
                    k: f"{v}nm" for k, v in TECH_NM.items()
                },
                "normalization": f"(tech_nm / {REFERENCE_NM}nm)^2",
                "comparison": comparison,
            }, fh, indent=2)
        print(f"\nComparison written to: {out_path}")
    else:
        # Default output.
        out_path = results_dir / "comparison.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump({
                "technology_assumptions": {
                    k: f"{v}nm" for k, v in TECH_NM.items()
                },
                "normalization": f"(tech_nm / {REFERENCE_NM}nm)^2",
                "comparison": comparison,
            }, fh, indent=2)
        print(f"\nComparison written to: {out_path}")


if __name__ == "__main__":
    main()
