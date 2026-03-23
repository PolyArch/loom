#!/usr/bin/env python3
"""Multi-PDK comparison: run synthesis across SAED14nm, ASAP7, and SAED32.

Invokes run_synthesis.py for each PDK target and generates a comparison
table with area, timing, and power metrics.

Usage:
    python compare_pdks.py --design <name> --rtl-dir <path> \\
        [--clock-period 10.0] [--output-dir ./pdk_compare]
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_SYNTH = os.path.join(SCRIPT_DIR, "run_synthesis.py")

PDKS = ["saed14", "asap7", "saed32"]


def run_pdk_synthesis(args, pdk):
    """Run synthesis for a single PDK and return the summary dict."""
    pdk_out = os.path.join(args.output_dir, pdk)
    cmd = [
        sys.executable, RUN_SYNTH,
        "--design", args.design,
        "--rtl-dir", args.rtl_dir,
        "--pdk", pdk,
        "--clock-period", str(args.clock_period),
        "--output-dir", pdk_out,
    ]
    if args.filelist:
        cmd.extend(["--filelist", args.filelist])
    if args.hierarchical:
        cmd.append("--hierarchical")

    print(f"\n{'='*60}")
    print(f"Synthesizing with PDK: {pdk}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    # Load summary
    summary_path = os.path.join(pdk_out, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            return json.load(f)

    return {
        "design_name": args.design,
        "pdk": pdk,
        "synthesis_success": False,
        "error": f"Exit code {result.returncode}",
    }


def format_comparison_table(results):
    """Format a text comparison table from PDK synthesis results."""
    lines = []
    header = f"{'Metric':<30} " + " ".join(f"{r['pdk']:>15}" for r in results)
    lines.append(header)
    lines.append("-" * len(header))

    # Success
    row = f"{'Synthesis Success':<30} "
    row += " ".join(f"{'Yes' if r.get('synthesis_success') else 'No':>15}"
                    for r in results)
    lines.append(row)

    # Clock period
    row = f"{'Clock Period (ns)':<30} "
    row += " ".join(f"{r.get('clock_period_ns', 'N/A'):>15}" for r in results)
    lines.append(row)

    # Area metrics
    for metric, label in [
        ("total_area", "Total Area"),
        ("cell_area", "Cell Area"),
        ("net_area", "Net Area"),
    ]:
        row = f"{label:<30} "
        vals = []
        for r in results:
            area = r.get("area", {})
            v = area.get(metric, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:>15.1f}")
            else:
                vals.append(f"{v!s:>15}")
        row += " ".join(vals)
        lines.append(row)

    # Timing
    row = f"{'Worst Slack (ns)':<30} "
    vals = []
    for r in results:
        timing = r.get("timing", {})
        v = timing.get("worst_slack_ns", "N/A")
        if isinstance(v, float):
            vals.append(f"{v:>15.3f}")
        else:
            vals.append(f"{v!s:>15}")
    row += " ".join(vals)
    lines.append(row)

    row = f"{'Timing Met':<30} "
    row += " ".join(
        f"{'Yes' if r.get('timing', {}).get('met_timing') else 'No':>15}"
        for r in results)
    lines.append(row)

    # Power
    for metric, label in [
        ("total_power_mw", "Total Power (mW)"),
        ("dynamic_power_mw", "Dynamic Power (mW)"),
        ("leakage_power_mw", "Leakage Power (mW)"),
    ]:
        row = f"{label:<30} "
        vals = []
        for r in results:
            power = r.get("power", {})
            v = power.get(metric, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:>15.4f}")
            else:
                vals.append(f"{v!s:>15}")
        row += " ".join(vals)
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare synthesis results across multiple PDKs")
    parser.add_argument("--design", required=True,
                        help="Top-level design/module name")
    parser.add_argument("--rtl-dir", required=True,
                        help="Directory containing RTL source files")
    parser.add_argument("--filelist", default=None,
                        help="Explicit filelist path")
    parser.add_argument("--clock-period", type=float, default=10.0,
                        help="Clock period in ns (default: 10.0)")
    parser.add_argument("--output-dir", default="./pdk_compare",
                        help="Output directory (default: ./pdk_compare)")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Use hierarchical compile")
    parser.add_argument("--pdks", nargs="+", default=PDKS,
                        choices=PDKS,
                        help="PDKs to compare (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for pdk in args.pdks:
        summary = run_pdk_synthesis(args, pdk)
        results.append(summary)

    # Generate comparison table
    table = format_comparison_table(results)
    print(f"\n{'='*60}")
    print("PDK Comparison Results")
    print(f"{'='*60}")
    print(table)

    # Write comparison JSON
    comparison = {
        "design_name": args.design,
        "clock_period_ns": args.clock_period,
        "pdk_results": {r["pdk"]: r for r in results},
    }
    comparison_path = os.path.join(args.output_dir, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Write text table
    table_path = os.path.join(args.output_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table + "\n")

    print(f"\nComparison JSON: {comparison_path}")
    print(f"Comparison table: {table_path}")

    # Summary
    succeeded = sum(1 for r in results if r.get("synthesis_success"))
    total = len(results)
    print(f"\n{succeeded}/{total} PDK syntheses succeeded")

    if succeeded < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
