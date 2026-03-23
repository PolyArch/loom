#!/usr/bin/env python3
"""Bar chart: throughput across domains and configurations.

Generates a grouped bar chart comparing throughput (ops/sec) for each
benchmark grouped by domain, with bars for each hardware configuration
(GENERAL, HOMO-SMALL, HOMO-LARGE, SINGLE).

Usage:
    python plot_throughput.py --input results/throughput.json \
        --output figures/throughput
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import load_throughput_results, geomean, normalize_to_baseline
from plot_style import (
    CONFIG_COLORS, CONFIG_HATCHES, DOMAIN_COLORS,
    DOUBLE_COL, SINGLE_COL,
    create_figure, save_figure, setup_paper_style,
)


def plot_throughput_by_domain(df: pd.DataFrame, output_path: str) -> None:
    """Create a grouped bar chart of throughput per benchmark and config."""
    setup_paper_style()

    # Normalize to GENERAL configuration
    df_norm = normalize_to_baseline(
        df, "throughput_ops_sec", "GENERAL",
        group_col="benchmark", label_col="config",
    )

    configs = list(CONFIG_COLORS.keys())
    configs_present = [c for c in configs if c in df_norm["config"].values]
    benchmarks = df_norm["benchmark"].unique()

    # Sort benchmarks by domain
    domain_order = list(DOMAIN_COLORS.keys())
    bench_domain = df_norm.drop_duplicates("benchmark")[["benchmark", "domain"]]
    bench_domain["domain_idx"] = bench_domain["domain"].apply(
        lambda d: domain_order.index(d) if d in domain_order else 999
    )
    bench_domain = bench_domain.sort_values("domain_idx")
    benchmarks = bench_domain["benchmark"].values

    n_bench = len(benchmarks)
    n_cfg = len(configs_present)
    bar_width = 0.8 / max(n_cfg, 1)
    x = np.arange(n_bench)

    fig, ax = create_figure(size=DOUBLE_COL)

    for idx, cfg in enumerate(configs_present):
        subset = df_norm[df_norm["config"] == cfg]
        values = []
        for bench in benchmarks:
            row = subset[subset["benchmark"] == bench]
            if not row.empty:
                values.append(row["normalized"].values[0])
            else:
                values.append(0.0)

        offset = (idx - (n_cfg - 1) / 2) * bar_width
        ax.bar(
            x + offset, values, bar_width * 0.9,
            label=cfg,
            color=CONFIG_COLORS.get(cfg, "#999999"),
            hatch=CONFIG_HATCHES.get(cfg, ""),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Normalized Throughput")
    ax.set_title("Throughput Comparison Across Configurations")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=7, ncol=len(configs_present), loc="upper right")

    # Add domain separators
    _add_domain_separators(ax, benchmarks, bench_domain)

    fig.tight_layout()
    save_figure(fig, output_path)
    print(f"[plot_throughput] Saved to {output_path}")


def plot_geomean_by_domain(df: pd.DataFrame, output_path: str) -> None:
    """Create a bar chart of geometric mean throughput per domain."""
    setup_paper_style()

    df_norm = normalize_to_baseline(
        df, "throughput_ops_sec", "GENERAL",
        group_col="benchmark", label_col="config",
    )

    configs = [c for c in CONFIG_COLORS if c in df_norm["config"].values]
    domains = [d for d in DOMAIN_COLORS if d in df_norm["domain"].values]

    n_dom = len(domains)
    n_cfg = len(configs)
    bar_width = 0.8 / max(n_cfg, 1)
    x = np.arange(n_dom)

    fig, ax = create_figure(size=SINGLE_COL)

    for idx, cfg in enumerate(configs):
        means = []
        for domain in domains:
            subset = df_norm[(df_norm["config"] == cfg)
                             & (df_norm["domain"] == domain)]
            means.append(geomean(subset["normalized"].tolist()))

        offset = (idx - (n_cfg - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, bar_width * 0.9,
            label=cfg,
            color=CONFIG_COLORS.get(cfg, "#999999"),
            hatch=CONFIG_HATCHES.get(cfg, ""),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_ylabel("Geomean Normalized Throughput")
    ax.set_title("Per-Domain Throughput Summary")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_figure(fig, output_path + "_geomean")
    print(f"[plot_throughput] Saved geomean to {output_path}_geomean")


def _add_domain_separators(ax, benchmarks, bench_domain):
    """Add vertical lines between domain groups on the x-axis."""
    prev_domain = None
    for i, bench in enumerate(benchmarks):
        row = bench_domain[bench_domain["benchmark"] == bench]
        if row.empty:
            continue
        domain = row["domain"].values[0]
        if prev_domain is not None and domain != prev_domain:
            ax.axvline(x=i - 0.5, color="gray", linestyle=":",
                       linewidth=0.5, alpha=0.5)
        prev_domain = domain


def main():
    parser = argparse.ArgumentParser(
        description="Generate throughput comparison bar charts.")
    parser.add_argument("--input", type=str, default="eval_results",
                        help="Results directory or throughput.json path")
    parser.add_argument("--output", type=str, default="figures/throughput",
                        help="Output path (without extension)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        df = load_throughput_results(str(input_path))
    else:
        df = load_throughput_results(str(input_path.parent))

    if df.empty:
        print("[plot_throughput] No data to plot.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_throughput_by_domain(df, args.output)
    plot_geomean_by_domain(df, args.output)


if __name__ == "__main__":
    main()
