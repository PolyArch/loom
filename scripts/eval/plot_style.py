#!/usr/bin/env python3
"""Consistent plot styling for MICRO 2026 paper figures.

Provides color palettes, hatch patterns, font configuration, and figure
size presets formatted for IEEE two-column papers (3.5" single-column,
7" double-column).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Font and rendering defaults (LaTeX-compatible)
# -----------------------------------------------------------------------

FONT_FAMILY = "serif"
FONT_SIZE = 9
AXES_LABEL_SIZE = 10
AXES_TITLE_SIZE = 10
LEGEND_FONT_SIZE = 8
TICK_LABEL_SIZE = 8
DPI = 300

# -----------------------------------------------------------------------
# Figure size presets (inches) -- IEEE two-column format
# -----------------------------------------------------------------------

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (7.0, 2.5)
SINGLE_COL_TALL = (3.5, 3.5)
DOUBLE_COL_TALL = (7.0, 4.0)

# -----------------------------------------------------------------------
# Color palettes
# -----------------------------------------------------------------------

DOMAIN_COLORS = {
    "AI/LLM": "#1f77b4",
    "AR/VR": "#ff7f0e",
    "Robotics": "#2ca02c",
    "Graph": "#d62728",
    "DSP": "#9467bd",
    "ZK/Crypto": "#8c564b",
}

CONFIG_COLORS = {
    "GENERAL": "#4c72b0",
    "HOMO-SMALL": "#dd8452",
    "HOMO-LARGE": "#55a868",
    "SINGLE": "#c44e52",
}

PLATFORM_COLORS = {
    "CGRA": "#4c72b0",
    "GPU": "#dd8452",
    "CPU": "#55a868",
}

COMPONENT_COLORS = {
    "Cores": "#4c72b0",
    "NoC": "#dd8452",
    "L2 Cache": "#55a868",
    "SPM": "#c44e52",
    "Config Mem": "#8172b3",
    "Other": "#937860",
}

ABLATION_COLORS = {
    "Full Contracts": "#4c72b0",
    "Dependency-Only": "#dd8452",
    "Coarse Tags": "#55a868",
    "No Feedback": "#c44e52",
    "No NoC-Aware": "#8172b3",
    "No Heterogeneity": "#937860",
}

# -----------------------------------------------------------------------
# Hatch patterns
# -----------------------------------------------------------------------

CONFIG_HATCHES = {
    "GENERAL": "",
    "HOMO-SMALL": "///",
    "HOMO-LARGE": "\\\\\\",
    "SINGLE": "...",
}

PLATFORM_HATCHES = {
    "CGRA": "",
    "GPU": "///",
    "CPU": "\\\\\\",
}

# -----------------------------------------------------------------------
# Style application
# -----------------------------------------------------------------------


def setup_paper_style():
    """Apply publication-quality matplotlib style settings."""
    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.size"] = FONT_SIZE
    mpl.rcParams["axes.labelsize"] = AXES_LABEL_SIZE
    mpl.rcParams["axes.titlesize"] = AXES_TITLE_SIZE
    mpl.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
    mpl.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE
    mpl.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.02
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "cm"


def create_figure(size=None, nrows=1, ncols=1, **kwargs):
    """Create a figure with paper-quality defaults.

    Parameters
    ----------
    size : tuple of float, optional
        Figure size in inches. Defaults to SINGLE_COL.
    nrows, ncols : int
        Subplot grid dimensions.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    if size is None:
        size = SINGLE_COL
    setup_paper_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=size, **kwargs)
    return fig, axes


def save_figure(fig, path, formats=None):
    """Save figure in one or more formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str
        Base path without extension.
    formats : list of str, optional
        File formats to save. Defaults to ["pdf", "png"].
    """
    if formats is None:
        formats = ["pdf", "png"]
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", format=fmt, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
