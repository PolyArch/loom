#!/usr/bin/env python3
"""Line plot: Benders iteration convergence analysis.

Shows upper/lower bound convergence over iterations, the optimality gap,
and number of cuts generated per iteration.

Usage:
    python plot_convergence.py --input results/convergence.json \
        --output figures/convergence
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_convergence_results
from plot_style import (
    DOMAIN_COLORS, SINGLE_COL, SINGLE_COL_TALL, DOUBLE_COL,
    create_figure, save_figure, setup_paper_style,
)


# Line styles for different benchmarks
BENCH_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
BENCH_MARKERS = ["o", "s", "^", "D", "v", ">", "<", "p"]


def plot_convergence_bounds(df: pd.DataFrame, output_path: str) -> None:
    """Plot upper/lower bound convergence for each benchmark."""
    setup_paper_style()

    benchmarks = df["benchmark"].unique()
    n_bench = len(benchmarks)

    # Use a color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_bench, 1)))

    fig, axes = create_figure(size=SINGLE_COL_TALL, nrows=2, ncols=1,
                              sharex=True)
    ax_bounds, ax_gap = axes

    for idx, bench in enumerate(benchmarks):
        subset = df[df["benchmark"] == bench].sort_values("iteration")
        iters = subset["iteration"].values
        ub = subset["upper_bound"].values
        lb = subset["lower_bound"].values
        gap = subset["gap_pct"].values

        ls = BENCH_LINESTYLES[idx % len(BENCH_LINESTYLES)]
        mk = BENCH_MARKERS[idx % len(BENCH_MARKERS)]
        color = colors[idx]

        ax_bounds.plot(iters, ub, linestyle=ls, marker=mk, color=color,
                       markersize=3, linewidth=1.0, alpha=0.8)
        ax_bounds.plot(iters, lb, linestyle=ls, marker=mk, color=color,
                       markersize=3, linewidth=1.0, alpha=0.4)

        ax_gap.plot(iters, gap, linestyle=ls, marker=mk, color=color,
                    markersize=3, linewidth=1.0, label=bench)

    ax_bounds.set_ylabel("Bound Value")
    ax_bounds.set_title("Benders Convergence")
    ax_bounds.text(0.02, 0.95, "solid=UB, faded=LB",
                   transform=ax_bounds.transAxes, fontsize=6,
                   verticalalignment="top")

    ax_gap.set_ylabel("Optimality Gap (%)")
    ax_gap.set_xlabel("Iteration")
    ax_gap.axhline(y=1.0, color="red", linestyle="--", linewidth=0.5,
                   alpha=0.5)

    # Legend below the plot
    if n_bench <= 8:
        ax_gap.legend(fontsize=6, ncol=min(n_bench, 4),
                      loc="upper right")

    fig.tight_layout()
    save_figure(fig, output_path)
    print(f"[plot_convergence] Saved to {output_path}")


def plot_convergence_boxplot(df: pd.DataFrame, output_path: str) -> None:
    """Box plot of iterations to convergence per benchmark."""
    setup_paper_style()

    # Determine convergence iteration (gap < 1%)
    convergence_iters = []
    for bench in df["benchmark"].unique():
        subset = df[df["benchmark"] == bench].sort_values("iteration")
        converged = subset[subset["gap_pct"] < 1.0]
        if not converged.empty:
            conv_iter = converged["iteration"].iloc[0]
        else:
            conv_iter = subset["iteration"].max()
        convergence_iters.append({
            "benchmark": bench,
            "convergence_iteration": conv_iter,
        })

    conv_df = pd.DataFrame(convergence_iters)
    if conv_df.empty:
        print("[plot_convergence] No convergence data for boxplot.")
        return

    fig, ax = create_figure(size=SINGLE_COL)

    ax.bar(range(len(conv_df)), conv_df["convergence_iteration"].values,
           color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(conv_df)))
    ax.set_xticklabels(conv_df["benchmark"].values, rotation=45,
                       ha="right", fontsize=7)
    ax.set_ylabel("Iterations to Convergence")
    ax.set_title("Convergence Speed per Benchmark")

    fig.tight_layout()
    save_figure(fig, output_path + "_boxplot")
    print(f"[plot_convergence] Saved boxplot to {output_path}_boxplot")


def plot_cuts_over_iterations(df: pd.DataFrame, output_path: str) -> None:
    """Plot cumulative number of cuts over iterations."""
    setup_paper_style()

    benchmarks = df["benchmark"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(benchmarks), 1)))

    fig, ax = create_figure(size=SINGLE_COL)

    for idx, bench in enumerate(benchmarks):
        subset = df[df["benchmark"] == bench].sort_values("iteration")
        iters = subset["iteration"].values
        cuts = np.cumsum(subset["num_cuts"].values)
        ax.plot(iters, cuts, marker=BENCH_MARKERS[idx % len(BENCH_MARKERS)],
                markersize=3, linewidth=1.0, color=colors[idx], label=bench)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Cuts")
    ax.set_title("Cut Generation Over Iterations")
    if len(benchmarks) <= 8:
        ax.legend(fontsize=6, ncol=min(len(benchmarks), 4))

    fig.tight_layout()
    save_figure(fig, output_path + "_cuts")
    print(f"[plot_convergence] Saved cuts plot to {output_path}_cuts")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Benders convergence plots.")
    parser.add_argument("--input", type=str, default="eval_results",
                        help="Results directory")
    parser.add_argument("--output", type=str, default="figures/convergence",
                        help="Output path (without extension)")
    args = parser.parse_args()

    df = load_convergence_results(args.input)
    if df.empty:
        print("[plot_convergence] No data to plot.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_convergence_bounds(df, args.output)
    plot_convergence_boxplot(df, args.output)
    plot_cuts_over_iterations(df, args.output)


if __name__ == "__main__":
    main()
