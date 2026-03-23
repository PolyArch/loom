#!/usr/bin/env python3
"""Multi-panel sensitivity analysis plots.

Generates sensitivity sweeps for NoC bandwidth, L2 cache size, and SPM
size, showing throughput impact.  Also produces a 2D heatmap of SPM size
vs NoC bandwidth.

Usage:
    python plot_sensitivity.py --input results/sensitivity.json \
        --output figures/sensitivity
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_sensitivity_results, geomean
from plot_style import (
    DOMAIN_COLORS, SINGLE_COL, SINGLE_COL_TALL, DOUBLE_COL, DOUBLE_COL_TALL,
    create_figure, save_figure, setup_paper_style,
)

# Display-friendly parameter names
PARAM_LABELS = {
    "noc_bandwidth": "NoC Bandwidth (GB/s)",
    "l2_size_kb": "L2 Cache Size (KB)",
    "spm_size_kb": "SPM Size (KB)",
}


def plot_sensitivity_panels(df: pd.DataFrame, output_path: str) -> None:
    """Create a multi-panel figure with one subplot per swept parameter."""
    setup_paper_style()

    parameters = df["parameter"].unique()
    n_params = len(parameters)

    if n_params == 0:
        print("[plot_sensitivity] No parameters found.")
        return

    fig, axes = create_figure(size=DOUBLE_COL_TALL, nrows=1, ncols=n_params,
                              sharey=True)
    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, parameters):
        subset = df[df["parameter"] == param]
        benchmarks = subset["benchmark"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(benchmarks), 1)))

        for idx, bench in enumerate(benchmarks):
            bench_data = subset[subset["benchmark"] == bench].sort_values("value")
            values = bench_data["value"].values
            throughput = bench_data["throughput_ops_sec"].values

            # Normalize to first value
            if throughput[0] > 0:
                throughput_norm = throughput / throughput[0]
            else:
                throughput_norm = throughput

            ax.plot(values, throughput_norm,
                    marker="o", markersize=3, linewidth=1.0,
                    color=colors[idx], label=bench)

        ax.set_xlabel(PARAM_LABELS.get(param, param))
        ax.set_xscale("log", base=2)
        if ax == axes[0]:
            ax.set_ylabel("Normalized Throughput")
        ax.set_title(param.replace("_", " ").title())

        if len(benchmarks) <= 6:
            ax.legend(fontsize=5, loc="lower right")

    fig.suptitle("Sensitivity Analysis", fontsize=10, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)
    print(f"[plot_sensitivity] Saved to {output_path}")


def plot_sensitivity_geomean(df: pd.DataFrame, output_path: str) -> None:
    """Plot geometric mean throughput across benchmarks for each parameter."""
    setup_paper_style()

    parameters = df["parameter"].unique()
    n_params = len(parameters)

    fig, axes = create_figure(size=DOUBLE_COL, nrows=1, ncols=n_params,
                              sharey=True)
    if n_params == 1:
        axes = [axes]

    param_colors = ["#4c72b0", "#dd8452", "#55a868"]

    for ax, param, color in zip(axes, parameters,
                                param_colors[:n_params]):
        subset = df[df["parameter"] == param]
        values = sorted(subset["value"].unique())

        means = []
        for val in values:
            val_data = subset[subset["value"] == val]
            gm = geomean(val_data["throughput_ops_sec"].tolist())
            means.append(gm)

        # Normalize
        if means and means[0] > 0:
            means_norm = [m / means[0] for m in means]
        else:
            means_norm = means

        ax.plot(values, means_norm, marker="s", markersize=4,
                linewidth=1.5, color=color)
        ax.fill_between(values, 1.0, means_norm, alpha=0.1, color=color)
        ax.set_xlabel(PARAM_LABELS.get(param, param))
        ax.set_xscale("log", base=2)
        if ax == axes[0]:
            ax.set_ylabel("Geomean Normalized Throughput")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)

    fig.suptitle("Sensitivity Analysis (Geometric Mean)", fontsize=10,
                 y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path + "_geomean")
    print(f"[plot_sensitivity] Saved geomean to {output_path}_geomean")


def plot_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """2D heatmap: SPM size x NoC bandwidth -> throughput.

    Only generated if both parameters are present in the data.
    """
    setup_paper_style()

    has_noc = "noc_bandwidth" in df["parameter"].values
    has_spm = "spm_size_kb" in df["parameter"].values
    if not (has_noc and has_spm):
        print("[plot_sensitivity] Skipping heatmap (need both noc_bandwidth "
              "and spm_size_kb).")
        return

    # Build a cross-product (requires separate sweep data)
    # For now, create from available data
    noc_data = df[df["parameter"] == "noc_bandwidth"]
    spm_data = df[df["parameter"] == "spm_size_kb"]

    noc_vals = sorted(noc_data["value"].unique())
    spm_vals = sorted(spm_data["value"].unique())

    if len(noc_vals) < 2 or len(spm_vals) < 2:
        return

    # Compute geomean throughput for each parameter value
    noc_means = {}
    for v in noc_vals:
        subset = noc_data[noc_data["value"] == v]
        noc_means[v] = geomean(subset["throughput_ops_sec"].tolist())

    spm_means = {}
    for v in spm_vals:
        subset = spm_data[spm_data["value"] == v]
        spm_means[v] = geomean(subset["throughput_ops_sec"].tolist())

    # Approximate 2D throughput as product of normalized factors
    base_noc = noc_means.get(noc_vals[0], 1.0) or 1.0
    base_spm = spm_means.get(spm_vals[0], 1.0) or 1.0

    heatmap_data = np.zeros((len(spm_vals), len(noc_vals)))
    for i, spm_v in enumerate(spm_vals):
        for j, noc_v in enumerate(noc_vals):
            spm_factor = (spm_means.get(spm_v, base_spm) / base_spm
                          if base_spm > 0 else 1.0)
            noc_factor = (noc_means.get(noc_v, base_noc) / base_noc
                          if base_noc > 0 else 1.0)
            heatmap_data[i, j] = spm_factor * noc_factor

    fig, ax = create_figure(size=SINGLE_COL_TALL)

    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                   origin="lower")
    ax.set_xticks(range(len(noc_vals)))
    ax.set_xticklabels([str(v) for v in noc_vals])
    ax.set_yticks(range(len(spm_vals)))
    ax.set_yticklabels([str(v) for v in spm_vals])
    ax.set_xlabel("NoC Bandwidth (GB/s)")
    ax.set_ylabel("SPM Size (KB)")
    ax.set_title("Throughput Sensitivity Heatmap")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Throughput")

    fig.tight_layout()
    save_figure(fig, output_path + "_heatmap")
    print(f"[plot_sensitivity] Saved heatmap to {output_path}_heatmap")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity analysis plots.")
    parser.add_argument("--input", type=str, default="eval_results",
                        help="Results directory")
    parser.add_argument("--output", type=str, default="figures/sensitivity",
                        help="Output path (without extension)")
    args = parser.parse_args()

    df = load_sensitivity_results(args.input)
    if df.empty:
        print("[plot_sensitivity] No data to plot.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_sensitivity_panels(df, args.output)
    plot_sensitivity_geomean(df, args.output)
    plot_heatmap(df, args.output)


if __name__ == "__main__":
    main()
