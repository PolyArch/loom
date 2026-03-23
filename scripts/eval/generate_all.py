#!/usr/bin/env python3
"""Generate all plots and tables for the MICRO 2026 paper.

Reads experiment results from a specified directory and produces all
figures and LaTeX tables in one pass.

Usage:
    python generate_all.py --input eval_results --output figures
    python generate_all.py --input eval_results --output figures --tables-only
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import data_loader
import table_gen
from plot_style import setup_paper_style


def generate_plots(input_dir: str, output_dir: str) -> None:
    """Generate all figures from experiment results."""
    os.makedirs(output_dir, exist_ok=True)

    plot_modules = [
        ("plot_throughput", "throughput"),
        ("plot_convergence", "convergence"),
        ("plot_dse_pareto", "dse_pareto"),
        ("plot_proxy_correlation", "proxy_correlation"),
        ("plot_sensitivity", "sensitivity"),
        ("plot_baselines", "baselines"),
        ("plot_area_power", "area_power"),
        ("plot_noc_comparison", "noc_comparison"),
    ]

    for module_name, fig_prefix in plot_modules:
        output_path = os.path.join(output_dir, fig_prefix)
        print(f"\n--- Generating {fig_prefix} ---")

        try:
            mod = __import__(module_name)
            # Each module's main() expects --input and --output
            # Call the plotting functions directly instead
            sys.argv = [
                module_name,
                "--input", input_dir,
                "--output", output_path,
            ]
            mod.main()
        except SystemExit:
            # main() may call sys.exit(1) if no data
            print(f"  Skipped {fig_prefix} (no data available)")
        except Exception as exc:
            print(f"  ERROR generating {fig_prefix}: {exc}")


def generate_tables(input_dir: str, output_dir: str) -> None:
    """Generate all LaTeX tables from experiment results."""
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    table_generators = [
        ("throughput", data_loader.load_throughput_results,
         table_gen.generate_throughput_table),
        ("ablation", data_loader.load_ablation_results,
         table_gen.generate_ablation_table),
        ("area_power", data_loader.load_area_power_results,
         table_gen.generate_area_power_table),
        ("noc_comparison", data_loader.load_noc_comparison_results,
         table_gen.generate_noc_table),
    ]

    for name, loader, generator in table_generators:
        print(f"\n--- Generating table: {name} ---")
        try:
            df = loader(input_dir)
            if df.empty:
                print(f"  Skipped {name} (no data)")
                continue
            output_path = os.path.join(tables_dir, f"{name}.tex")
            generator(df, output_path)
        except Exception as exc:
            print(f"  ERROR generating table {name}: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all MICRO 2026 paper figures and tables.")
    parser.add_argument("--input", type=str, default="eval_results",
                        help="Directory containing experiment result JSONs")
    parser.add_argument("--output", type=str, default="figures",
                        help="Output directory for figures and tables")
    parser.add_argument("--plots-only", action="store_true",
                        help="Generate only plots (no tables)")
    parser.add_argument("--tables-only", action="store_true",
                        help="Generate only tables (no plots)")
    args = parser.parse_args()

    setup_paper_style()
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    if not args.tables_only:
        print("\n=== Generating Plots ===")
        generate_plots(args.input, args.output)

    if not args.plots_only:
        print("\n=== Generating Tables ===")
        generate_tables(args.input, args.output)

    print("\n=== Generation complete ===")


if __name__ == "__main__":
    main()
