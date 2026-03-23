#!/usr/bin/env python3
"""Bar chart: Loom NoC vs Arteris FlexNoC comparison.

Compares area, latency, and throughput between the Loom-generated mesh
NoC and the Arteris FlexNoC industrial reference.

Usage:
    python plot_noc_comparison.py --input results/noc_comparison.json \
        --output figures/noc_comparison
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_noc_comparison_results, geomean
from plot_style import (
    SINGLE_COL, SINGLE_COL_TALL, DOUBLE_COL,
    create_figure, save_figure, setup_paper_style,
)

NOC_COLORS = {
    "Loom Mesh": "#4c72b0",
    "FlexNoC": "#dd8452",
}


def plot_noc_area_latency(df: pd.DataFrame, output_path: str) -> None:
    """Side-by-side bar chart comparing area and latency."""
    setup_paper_style()

    noc_types = df["noc_type"].unique()
    n_noc = len(noc_types)

    # Aggregate across benchmarks
    agg = df.groupby("noc_type").agg(
        area_um2=("area_um2", "mean"),
        latency_ns=("latency_ns", "mean"),
        throughput=("throughput_ops_sec", lambda x: geomean(x.tolist())),
    ).reset_index()

    fig, (ax1, ax2) = create_figure(size=DOUBLE_COL, ncols=2)

    # Area comparison
    colors = [NOC_COLORS.get(n, "#999999") for n in agg["noc_type"]]
    bars1 = ax1.bar(range(n_noc), agg["area_um2"].values,
                    color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(n_noc))
    ax1.set_xticklabels(agg["noc_type"].values)
    ax1.set_ylabel("Area (um$^2$)")
    ax1.set_title("NoC Area Comparison")

    # Add value labels
    for bar, val in zip(bars1, agg["area_um2"].values):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    # Latency comparison
    bars2 = ax2.bar(range(n_noc), agg["latency_ns"].values,
                    color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(n_noc))
    ax2.set_xticklabels(agg["noc_type"].values)
    ax2.set_ylabel("Latency (ns)")
    ax2.set_title("NoC Latency Comparison")

    for bar, val in zip(bars2, agg["latency_ns"].values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save_figure(fig, output_path)
    print(f"[plot_noc_comparison] Saved to {output_path}")


def plot_noc_per_benchmark(df: pd.DataFrame, output_path: str) -> None:
    """Grouped bar chart: throughput per benchmark per NoC type."""
    setup_paper_style()

    noc_types = sorted(df["noc_type"].unique())
    benchmarks = sorted(df["benchmark"].unique())

    n_bench = len(benchmarks)
    n_noc = len(noc_types)
    bar_width = 0.8 / max(n_noc, 1)
    x = np.arange(n_bench)

    fig, ax = create_figure(size=SINGLE_COL)

    for idx, noc in enumerate(noc_types):
        subset = df[df["noc_type"] == noc]
        values = []
        for bench in benchmarks:
            row = subset[subset["benchmark"] == bench]
            if not row.empty:
                values.append(row["throughput_ops_sec"].values[0])
            else:
                values.append(0.0)

        offset = (idx - (n_noc - 1) / 2) * bar_width
        ax.bar(
            x + offset, values, bar_width * 0.9,
            label=noc,
            color=NOC_COLORS.get(noc, "#999999"),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Throughput (ops/sec)")
    ax.set_title("NoC Impact on Benchmark Throughput")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_figure(fig, output_path + "_per_bench")
    print(f"[plot_noc_comparison] Saved per-bench to "
          f"{output_path}_per_bench")


def main():
    parser = argparse.ArgumentParser(
        description="Generate NoC comparison plots.")
    parser.add_argument("--input", type=str, default="eval_results",
                        help="Results directory")
    parser.add_argument("--output", type=str,
                        default="figures/noc_comparison",
                        help="Output path (without extension)")
    args = parser.parse_args()

    df = load_noc_comparison_results(args.input)
    if df.empty:
        print("[plot_noc_comparison] No data to plot.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_noc_area_latency(df, args.output)
    plot_noc_per_benchmark(df, args.output)


if __name__ == "__main__":
    main()
