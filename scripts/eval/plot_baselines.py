#!/usr/bin/env python3
"""Bar chart: CGRA vs GPU vs CPU with technology normalization.

Compares throughput and efficiency (perf/Watt, perf/mm^2) across
platforms after adjusting for technology node differences.

Usage:
    python plot_baselines.py --input results/baselines.json \
        --output figures/baselines
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_baseline_results, geomean
from plot_style import (
    PLATFORM_COLORS, PLATFORM_HATCHES,
    SINGLE_COL, DOUBLE_COL, DOUBLE_COL_TALL,
    create_figure, save_figure, setup_paper_style,
)

# Technology normalization (same as baselines/compare_all.py)
TECH_NM = {"CGRA": 14.0, "GPU": 4.0, "CPU": 5.0}
REFERENCE_NM = 14.0

# Compiler method display names
METHOD_LABELS = {
    "benders": "Benders (Loom)",
    "monolithic_ilp": "Monolithic ILP",
    "heuristic": "Heuristic",
    "GPU": "GPU (RTX 5090)",
    "CPU": "CPU (Zen 5)",
}


def tech_norm_factor(platform: str) -> float:
    """Compute technology normalization factor."""
    nm = TECH_NM.get(platform, REFERENCE_NM)
    return (nm / REFERENCE_NM) ** 2


def plot_throughput_comparison(df: pd.DataFrame, output_path: str) -> None:
    """Grouped bar chart of raw throughput per benchmark per platform."""
    setup_paper_style()

    platforms = sorted(df["platform"].unique(),
                       key=lambda p: list(METHOD_LABELS.keys()).index(p)
                       if p in METHOD_LABELS else 999)
    benchmarks = sorted(df["benchmark"].unique())

    n_bench = len(benchmarks)
    n_plat = len(platforms)
    bar_width = 0.8 / max(n_plat, 1)
    x = np.arange(n_bench)

    colors_list = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

    fig, ax = create_figure(size=DOUBLE_COL)

    for idx, plat in enumerate(platforms):
        subset = df[df["platform"] == plat]
        values = []
        for bench in benchmarks:
            row = subset[subset["benchmark"] == bench]
            if not row.empty:
                values.append(row["throughput_ops_sec"].values[0])
            else:
                values.append(0.0)

        offset = (idx - (n_plat - 1) / 2) * bar_width
        label = METHOD_LABELS.get(plat, plat)
        ax.bar(
            x + offset, values, bar_width * 0.9,
            label=label,
            color=colors_list[idx % len(colors_list)],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Throughput (ops/sec)")
    ax.set_title("Throughput: CGRA Compiler Methods vs GPU/CPU")
    ax.legend(fontsize=6, ncol=min(n_plat, 3), loc="upper right")
    ax.set_yscale("log")

    fig.tight_layout()
    save_figure(fig, output_path)
    print(f"[plot_baselines] Saved to {output_path}")


def plot_normalized_comparison(df: pd.DataFrame, output_path: str) -> None:
    """Bar chart normalized to Benders (Loom) baseline."""
    setup_paper_style()

    benders_data = df[df["platform"] == "benders"]
    if benders_data.empty:
        print("[plot_baselines] No benders data for normalization.")
        return

    baseline = benders_data.set_index("benchmark")["throughput_ops_sec"]

    platforms = sorted(df["platform"].unique())
    benchmarks = sorted(df["benchmark"].unique())

    n_bench = len(benchmarks)
    n_plat = len(platforms)
    bar_width = 0.8 / max(n_plat, 1)
    x = np.arange(n_bench)

    colors_list = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

    fig, ax = create_figure(size=DOUBLE_COL)

    for idx, plat in enumerate(platforms):
        subset = df[df["platform"] == plat]
        values = []
        for bench in benchmarks:
            row = subset[subset["benchmark"] == bench]
            base_val = baseline.get(bench, 0.0)
            if not row.empty and base_val > 0:
                values.append(row["throughput_ops_sec"].values[0] / base_val)
            else:
                values.append(0.0)

        offset = (idx - (n_plat - 1) / 2) * bar_width
        label = METHOD_LABELS.get(plat, plat)
        ax.bar(
            x + offset, values, bar_width * 0.9,
            label=label,
            color=colors_list[idx % len(colors_list)],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Normalized Throughput")
    ax.set_title("Throughput Normalized to Benders Compiler")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=6, ncol=min(n_plat, 3))

    fig.tight_layout()
    save_figure(fig, output_path + "_normalized")
    print(f"[plot_baselines] Saved normalized to {output_path}_normalized")


def plot_compile_time_vs_quality(df: pd.DataFrame, output_path: str) -> None:
    """Scatter plot: compile time vs throughput for compiler methods."""
    setup_paper_style()

    if "compile_time_sec" not in df.columns:
        return

    compiler_methods = ["benders", "monolithic_ilp", "heuristic"]
    compiler_df = df[df["platform"].isin(compiler_methods)]
    if compiler_df.empty:
        return

    fig, ax = create_figure(size=SINGLE_COL)

    colors_list = ["#4c72b0", "#dd8452", "#55a868"]
    markers = ["o", "s", "^"]

    for idx, method in enumerate(compiler_methods):
        subset = compiler_df[compiler_df["platform"] == method]
        if subset.empty:
            continue
        ax.scatter(
            subset["compile_time_sec"].values,
            subset["throughput_ops_sec"].values,
            s=25, alpha=0.7,
            color=colors_list[idx],
            marker=markers[idx],
            label=METHOD_LABELS.get(method, method),
            edgecolors="black",
            linewidth=0.3,
        )

    ax.set_xlabel("Compilation Time (sec)")
    ax.set_ylabel("Throughput (ops/sec)")
    ax.set_title("Compile Time vs Quality")
    ax.set_xscale("log")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_figure(fig, output_path + "_time_quality")
    print(f"[plot_baselines] Saved time-quality to "
          f"{output_path}_time_quality")


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline comparison plots.")
    parser.add_argument("--input", type=str, default="eval_results",
                        help="Results directory")
    parser.add_argument("--output", type=str, default="figures/baselines",
                        help="Output path (without extension)")
    args = parser.parse_args()

    df = load_baseline_results(args.input)
    if df.empty:
        print("[plot_baselines] No data to plot.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_throughput_comparison(df, args.output)
    plot_normalized_comparison(df, args.output)
    plot_compile_time_vs_quality(df, args.output)


if __name__ == "__main__":
    main()
