#!/usr/bin/env python3
"""Scatter plot with Pareto frontier overlay for DSE results.

Shows the design space exploration results with throughput vs area,
highlighting the Pareto-optimal designs.

Usage:
    python plot_dse_pareto.py --input results/dse_proxy.json \
        --output figures/dse_pareto
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_dse_proxy_results
from plot_style import (
    SINGLE_COL, SINGLE_COL_TALL,
    create_figure, save_figure, setup_paper_style,
)


def compute_pareto_frontier(points: np.ndarray) -> np.ndarray:
    """Compute the Pareto frontier for 2D points (maximize both axes).

    Parameters
    ----------
    points : np.ndarray of shape (N, 2)
        Each row is (x, y).

    Returns
    -------
    np.ndarray of shape (M, 2)
        Pareto-optimal points sorted by x.
    """
    sorted_idx = np.argsort(points[:, 0])
    sorted_pts = points[sorted_idx]

    pareto = [sorted_pts[0]]
    max_y = sorted_pts[0, 1]

    for pt in sorted_pts[1:]:
        if pt[1] >= max_y:
            pareto.append(pt)
            max_y = pt[1]

    return np.array(pareto)


def plot_pareto(df: pd.DataFrame, output_path: str) -> None:
    """Create a scatter plot with Pareto frontier for DSE results."""
    setup_paper_style()

    if "proxy_score" not in df.columns or "actual_throughput" not in df.columns:
        print("[plot_dse_pareto] Missing required columns.")
        return

    x = df["proxy_score"].values
    y = df["actual_throughput"].values

    # Filter out zero values
    mask = (x > 0) & (y > 0)
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) < 2:
        print("[plot_dse_pareto] Not enough valid data points.")
        return

    fig, ax = create_figure(size=SINGLE_COL_TALL)

    # Scatter all points
    tier_col = "proxy_tier"
    if tier_col in df.columns:
        tiers = df[tier_col].values[mask]
        tier_colors = {1: "#4c72b0", 2: "#dd8452", 3: "#55a868"}
        for tier in sorted(set(tiers)):
            tier_mask = tiers == tier
            ax.scatter(
                x_valid[tier_mask], y_valid[tier_mask],
                s=20, alpha=0.6,
                color=tier_colors.get(tier, "#999999"),
                label=f"Tier {tier}",
                zorder=2,
            )
    else:
        ax.scatter(x_valid, y_valid, s=20, alpha=0.6, color="#4c72b0",
                   zorder=2)

    # Compute and draw Pareto frontier
    points = np.column_stack([x_valid, y_valid])
    pareto_pts = compute_pareto_frontier(points)

    ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], "r-", linewidth=1.5,
            label="Pareto Frontier", zorder=3)
    ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], s=40, color="red",
               marker="*", zorder=4)

    ax.set_xlabel("Proxy Score (Tier 1)")
    ax.set_ylabel("Actual Throughput (ops/sec)")
    ax.set_title("DSE: Proxy Score vs Actual Throughput")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_figure(fig, output_path)
    print(f"[plot_dse_pareto] Saved to {output_path}")


def plot_pareto_area_throughput(df: pd.DataFrame, output_path: str) -> None:
    """Area vs throughput Pareto plot (if area data available)."""
    setup_paper_style()

    if "area_um2" not in df.columns:
        return

    x = df.get("area_um2", pd.Series(dtype=float)).values
    y = df.get("actual_throughput", pd.Series(dtype=float)).values

    mask = (x > 0) & (y > 0)
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) < 2:
        return

    fig, ax = create_figure(size=SINGLE_COL)

    ax.scatter(x_valid, y_valid, s=20, alpha=0.6, color="#4c72b0")

    points = np.column_stack([x_valid, y_valid])
    pareto_pts = compute_pareto_frontier(points)
    ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], "r-", linewidth=1.5)
    ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], s=40, color="red",
               marker="*")

    ax.set_xlabel("Area (um2)")
    ax.set_ylabel("Throughput (ops/sec)")
    ax.set_title("DSE: Area vs Throughput Pareto")

    fig.tight_layout()
    save_figure(fig, output_path + "_area")
    print(f"[plot_dse_pareto] Saved area plot to {output_path}_area")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DSE Pareto frontier plots.")
    parser.add_argument("--input", type=str, default="eval_results",
                        help="Results directory")
    parser.add_argument("--output", type=str, default="figures/dse_pareto",
                        help="Output path (without extension)")
    args = parser.parse_args()

    df = load_dse_proxy_results(args.input)
    if df.empty:
        print("[plot_dse_pareto] No data to plot.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_pareto(df, args.output)
    plot_pareto_area_throughput(df, args.output)


if __name__ == "__main__":
    main()
