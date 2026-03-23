#!/usr/bin/env python3
"""Stacked bar charts for area and power breakdown by component.

Shows post-synthesis area (um^2) and power (mW) breakdown across major
hardware components: Cores, NoC, L2 Cache, SPM, Config Mem.

Usage:
    python plot_area_power.py --input results/area_power.json \
        --output figures/area_power
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_area_power_results
from plot_style import (
    COMPONENT_COLORS, SINGLE_COL, SINGLE_COL_TALL, DOUBLE_COL,
    create_figure, save_figure, setup_paper_style,
)


def plot_area_breakdown(df: pd.DataFrame, output_path: str) -> None:
    """Create a stacked bar chart of area breakdown by component."""
    setup_paper_style()

    designs = df["design"].unique() if "design" in df.columns else ["default"]
    components = list(COMPONENT_COLORS.keys())
    components_present = [c for c in components if c in df["component"].values]

    n_designs = len(designs)

    fig, ax = create_figure(size=SINGLE_COL_TALL)

    bottom = np.zeros(n_designs)
    for comp in components_present:
        values = []
        for design in designs:
            if "design" in df.columns:
                row = df[(df["design"] == design) & (df["component"] == comp)]
            else:
                row = df[df["component"] == comp]
            if not row.empty:
                values.append(row["area_um2"].values[0])
            else:
                values.append(0.0)

        ax.bar(
            range(n_designs), values, bottom=bottom,
            label=comp,
            color=COMPONENT_COLORS.get(comp, "#999999"),
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += np.array(values)

    ax.set_xticks(range(n_designs))
    ax.set_xticklabels(designs, rotation=30, ha="right")
    ax.set_ylabel("Area (um$^2$)")
    ax.set_title("Area Breakdown by Component")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    save_figure(fig, output_path + "_area")
    print(f"[plot_area_power] Saved area to {output_path}_area")


def plot_power_breakdown(df: pd.DataFrame, output_path: str) -> None:
    """Create a stacked bar chart of power breakdown by component."""
    setup_paper_style()

    designs = df["design"].unique() if "design" in df.columns else ["default"]
    components = list(COMPONENT_COLORS.keys())
    components_present = [c for c in components if c in df["component"].values]

    n_designs = len(designs)

    fig, ax = create_figure(size=SINGLE_COL_TALL)

    bottom = np.zeros(n_designs)
    for comp in components_present:
        values = []
        for design in designs:
            if "design" in df.columns:
                row = df[(df["design"] == design) & (df["component"] == comp)]
            else:
                row = df[df["component"] == comp]
            if not row.empty:
                values.append(row["power_mw"].values[0])
            else:
                values.append(0.0)

        ax.bar(
            range(n_designs), values, bottom=bottom,
            label=comp,
            color=COMPONENT_COLORS.get(comp, "#999999"),
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += np.array(values)

    ax.set_xticks(range(n_designs))
    ax.set_xticklabels(designs, rotation=30, ha="right")
    ax.set_ylabel("Power (mW)")
    ax.set_title("Power Breakdown by Component")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    save_figure(fig, output_path + "_power")
    print(f"[plot_area_power] Saved power to {output_path}_power")


def plot_area_power_pie(df: pd.DataFrame, output_path: str) -> None:
    """Create side-by-side pie charts for area and power distribution."""
    setup_paper_style()

    components = list(COMPONENT_COLORS.keys())
    components_present = [c for c in components if c in df["component"].values]

    area_vals = []
    power_vals = []
    labels = []
    colors = []

    for comp in components_present:
        comp_data = df[df["component"] == comp]
        area_sum = comp_data["area_um2"].sum()
        power_sum = comp_data["power_mw"].sum()
        if area_sum > 0 or power_sum > 0:
            area_vals.append(area_sum)
            power_vals.append(power_sum)
            labels.append(comp)
            colors.append(COMPONENT_COLORS.get(comp, "#999999"))

    if not area_vals:
        return

    fig, (ax1, ax2) = create_figure(size=DOUBLE_COL, ncols=2)

    ax1.pie(area_vals, labels=labels, colors=colors, autopct="%1.1f%%",
            textprops={"fontsize": 7})
    ax1.set_title("Area Distribution")

    ax2.pie(power_vals, labels=labels, colors=colors, autopct="%1.1f%%",
            textprops={"fontsize": 7})
    ax2.set_title("Power Distribution")

    fig.tight_layout()
    save_figure(fig, output_path + "_pie")
    print(f"[plot_area_power] Saved pie to {output_path}_pie")


def plot_timing_summary(df: pd.DataFrame, output_path: str) -> None:
    """Bar chart showing timing closure results per design."""
    setup_paper_style()

    if "timing_slack_ns" not in df.columns:
        return

    designs = df["design"].unique() if "design" in df.columns else ["default"]
    timing_data = df.drop_duplicates("design") if "design" in df.columns else df

    fig, ax = create_figure(size=SINGLE_COL)

    slack_vals = []
    bar_colors = []
    for design in designs:
        if "design" in timing_data.columns:
            row = timing_data[timing_data["design"] == design]
        else:
            row = timing_data
        if not row.empty and "timing_slack_ns" in row.columns:
            slack = row["timing_slack_ns"].values[0]
        else:
            slack = 0.0
        slack_vals.append(slack)
        bar_colors.append("#55a868" if slack >= 0 else "#c44e52")

    ax.bar(range(len(designs)), slack_vals, color=bar_colors,
           edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.5)
    ax.set_xticks(range(len(designs)))
    ax.set_xticklabels(designs, rotation=30, ha="right")
    ax.set_ylabel("Timing Slack (ns)")
    ax.set_title("Timing Closure Summary")

    fig.tight_layout()
    save_figure(fig, output_path + "_timing")
    print(f"[plot_area_power] Saved timing to {output_path}_timing")


def main():
    parser = argparse.ArgumentParser(
        description="Generate area and power breakdown plots.")
    parser.add_argument("--input", type=str, default="eval_results",
                        help="Results directory")
    parser.add_argument("--output", type=str, default="figures/area_power",
                        help="Output path (without extension)")
    args = parser.parse_args()

    df = load_area_power_results(args.input)
    if df.empty:
        print("[plot_area_power] No data to plot.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_area_breakdown(df, args.output)
    plot_power_breakdown(df, args.output)
    plot_area_power_pie(df, args.output)
    plot_timing_summary(df, args.output)


if __name__ == "__main__":
    main()
