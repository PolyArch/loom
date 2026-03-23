#!/usr/bin/env python3
"""Scatter plot: proxy score vs actual throughput with R^2 annotation.

Shows the correlation between the fast proxy metric (Tier 1) and the
full simulation throughput (Tier 3), with a regression line and R^2
coefficient annotation.

Usage:
    python plot_proxy_correlation.py --input results/dse_proxy.json \
        --output figures/proxy_correlation
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_dse_proxy_results
from plot_style import (
    SINGLE_COL, SINGLE_COL_TALL,
    create_figure, save_figure, setup_paper_style,
)


def compute_r_squared(x: np.ndarray, y: np.ndarray):
    """Compute R^2 and regression parameters.

    Returns
    -------
    r_squared : float
    slope : float
    intercept : float
    p_value : float
    """
    result = stats.linregress(x, y)
    return result.rvalue ** 2, result.slope, result.intercept, result.pvalue


def plot_proxy_vs_actual(df: pd.DataFrame, output_path: str) -> None:
    """Create scatter plot with regression line and R^2 annotation."""
    setup_paper_style()

    x = df["proxy_score"].values
    y = df["actual_throughput"].values

    # Filter positive values
    mask = (x > 0) & (y > 0)
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) < 3:
        print("[plot_proxy_correlation] Not enough data points for regression.")
        return

    r2, slope, intercept, p_val = compute_r_squared(x_valid, y_valid)

    fig, ax = create_figure(size=SINGLE_COL_TALL)

    # Scatter points
    ax.scatter(x_valid, y_valid, s=25, alpha=0.6, color="#4c72b0",
               edgecolors="black", linewidth=0.3, zorder=2)

    # Regression line
    x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r-", linewidth=1.5, alpha=0.8,
            label="Regression", zorder=3)

    # Perfect correlation reference (y = x after normalization)
    x_range = x_valid.max() - x_valid.min()
    y_range = y_valid.max() - y_valid.min()
    if x_range > 0 and y_range > 0:
        x_norm = (x_valid - x_valid.min()) / x_range
        y_norm = (y_valid - y_valid.min()) / y_range
        ax.plot(
            [x_valid.min(), x_valid.max()],
            [y_valid.min(), y_valid.max()],
            "k--", linewidth=0.5, alpha=0.3, label="y = x (ideal)",
        )

    # R^2 annotation
    ax.annotate(
        f"$R^2 = {r2:.3f}$\n$p = {p_val:.2e}$\n$n = {len(x_valid)}$",
        xy=(0.05, 0.95), xycoords="axes fraction",
        verticalalignment="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("Proxy Score (Tier 1)")
    ax.set_ylabel("Actual Throughput (Tier 3)")
    ax.set_title("DSE Proxy Fidelity")
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    save_figure(fig, output_path)
    print(f"[plot_proxy_correlation] R^2 = {r2:.4f}, saved to {output_path}")


def plot_residuals(df: pd.DataFrame, output_path: str) -> None:
    """Plot regression residuals to assess proxy quality."""
    setup_paper_style()

    x = df["proxy_score"].values
    y = df["actual_throughput"].values
    mask = (x > 0) & (y > 0)
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) < 3:
        return

    _, slope, intercept, _ = compute_r_squared(x_valid, y_valid)
    predicted = slope * x_valid + intercept
    residuals = y_valid - predicted

    fig, ax = create_figure(size=SINGLE_COL)

    ax.scatter(predicted, residuals, s=15, alpha=0.6, color="#4c72b0",
               edgecolors="black", linewidth=0.3)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.5)

    ax.set_xlabel("Predicted Throughput")
    ax.set_ylabel("Residual")
    ax.set_title("Proxy Regression Residuals")

    fig.tight_layout()
    save_figure(fig, output_path + "_residuals")
    print(f"[plot_proxy_correlation] Saved residuals to "
          f"{output_path}_residuals")


def plot_rank_correlation(df: pd.DataFrame, output_path: str) -> None:
    """Plot Spearman rank correlation to assess ordering fidelity."""
    setup_paper_style()

    x = df["proxy_score"].values
    y = df["actual_throughput"].values
    mask = (x > 0) & (y > 0)
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) < 3:
        return

    spearman_r, spearman_p = stats.spearmanr(x_valid, y_valid)

    # Rank-rank plot
    x_rank = stats.rankdata(x_valid)
    y_rank = stats.rankdata(y_valid)

    fig, ax = create_figure(size=SINGLE_COL)

    ax.scatter(x_rank, y_rank, s=15, alpha=0.6, color="#55a868",
               edgecolors="black", linewidth=0.3)
    ax.plot([0, len(x_rank) + 1], [0, len(x_rank) + 1],
            "k--", linewidth=0.5, alpha=0.3)

    ax.annotate(
        f"Spearman $\\rho = {spearman_r:.3f}$\n$p = {spearman_p:.2e}$",
        xy=(0.05, 0.95), xycoords="axes fraction",
        verticalalignment="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("Proxy Rank")
    ax.set_ylabel("Actual Rank")
    ax.set_title("Rank Correlation")

    fig.tight_layout()
    save_figure(fig, output_path + "_rank")
    print(f"[plot_proxy_correlation] Spearman rho = {spearman_r:.4f}, "
          f"saved rank plot")


def main():
    parser = argparse.ArgumentParser(
        description="Generate proxy correlation scatter plots.")
    parser.add_argument("--input", type=str, default="eval_results",
                        help="Results directory")
    parser.add_argument("--output", type=str,
                        default="figures/proxy_correlation",
                        help="Output path (without extension)")
    args = parser.parse_args()

    df = load_dse_proxy_results(args.input)
    if df.empty:
        print("[plot_proxy_correlation] No data to plot.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_proxy_vs_actual(df, args.output)
    plot_residuals(df, args.output)
    plot_rank_correlation(df, args.output)


if __name__ == "__main__":
    main()
