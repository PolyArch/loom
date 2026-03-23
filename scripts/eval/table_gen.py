#!/usr/bin/env python3
"""Generate LaTeX tables from experiment data.

Produces publication-ready LaTeX table fragments that can be included
directly in the MICRO paper source via \\input{}.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# -----------------------------------------------------------------------
# Core table generation
# -----------------------------------------------------------------------


def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str,
                       column_format: Optional[str] = None,
                       float_format: str = "%.2f",
                       highlight_max: Optional[List[str]] = None) -> str:
    """Convert a DataFrame to a LaTeX table string.

    Parameters
    ----------
    df : pd.DataFrame
        Data to render.
    caption : str
        Table caption text.
    label : str
        LaTeX label for cross-referencing.
    column_format : str, optional
        LaTeX column alignment string (e.g., "lrrr").
    float_format : str
        Format string for floating-point values.
    highlight_max : list of str, optional
        Column names where the maximum value should be bolded.

    Returns
    -------
    str
        Complete LaTeX table environment.
    """
    if column_format is None:
        column_format = "l" + "r" * (len(df.columns) - 1)

    styled_df = df.copy()
    if highlight_max:
        for col in highlight_max:
            if col in styled_df.columns:
                max_val = styled_df[col].max()
                styled_df[col] = styled_df[col].apply(
                    lambda v: f"\\textbf{{{v:{float_format[1:]}}}}"
                    if v == max_val
                    else f"{v:{float_format[1:]}}"
                )

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{column_format}}}")
    lines.append("\\toprule")

    # Header row
    headers = " & ".join(str(c) for c in styled_df.columns)
    lines.append(f"{headers} \\\\")
    lines.append("\\midrule")

    # Data rows
    for _, row in styled_df.iterrows():
        cells = []
        for col in styled_df.columns:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:{float_format[1:]}}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Specific table generators
# -----------------------------------------------------------------------


def generate_throughput_table(df: pd.DataFrame, output_path: str) -> None:
    """Generate throughput comparison table (E1/E2/E3)."""
    pivot = df.pivot_table(
        index="benchmark", columns="config",
        values="throughput_ops_sec", aggfunc="mean",
    )
    pivot = pivot.reset_index()
    latex = dataframe_to_latex(
        pivot,
        caption="Throughput comparison across configurations (ops/sec).",
        label="tab:throughput",
        highlight_max=[c for c in pivot.columns if c != "benchmark"],
    )
    _write_table(latex, output_path)


def generate_ablation_table(df: pd.DataFrame, output_path: str) -> None:
    """Generate contract ablation table (E2)."""
    pivot = df.pivot_table(
        index="benchmark", columns="variant",
        values="throughput_ops_sec", aggfunc="mean",
    )
    pivot = pivot.reset_index()
    latex = dataframe_to_latex(
        pivot,
        caption="Contract ablation study: throughput impact of each contract feature.",
        label="tab:ablation",
        highlight_max=[c for c in pivot.columns if c != "benchmark"],
    )
    _write_table(latex, output_path)


def generate_area_power_table(df: pd.DataFrame, output_path: str) -> None:
    """Generate area/power breakdown table (E8)."""
    summary = df.groupby("component").agg(
        area_um2=("area_um2", "sum"),
        power_mw=("power_mw", "sum"),
    ).reset_index()
    summary.columns = ["Component", "Area (um2)", "Power (mW)"]
    latex = dataframe_to_latex(
        summary,
        caption="Post-synthesis area and power breakdown by component.",
        label="tab:area_power",
    )
    _write_table(latex, output_path)


def generate_noc_table(df: pd.DataFrame, output_path: str) -> None:
    """Generate NoC comparison table (E7)."""
    df_display = df.rename(columns={
        "noc_type": "NoC",
        "area_um2": "Area (um2)",
        "latency_ns": "Latency (ns)",
    })
    latex = dataframe_to_latex(
        df_display,
        caption="NoC topology comparison: area and latency.",
        label="tab:noc_comparison",
    )
    _write_table(latex, output_path)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _write_table(latex: str, output_path: str) -> None:
    """Write LaTeX table string to a file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(latex + "\n")
    print(f"[table_gen] Wrote {out}")
