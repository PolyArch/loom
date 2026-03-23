#!/usr/bin/env python3
"""Load experiment results from JSON files into pandas DataFrames.

Provides unified loading, validation, and aggregation for all experiment
result files produced by the experiment runners.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Add baselines/common for result_format access
BASELINES_COMMON = REPO_ROOT / "scripts" / "baselines" / "common"
if str(BASELINES_COMMON) not in sys.path:
    sys.path.insert(0, str(BASELINES_COMMON))


# -----------------------------------------------------------------------
# Generic JSON loading
# -----------------------------------------------------------------------


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dict."""
    with open(path) as fh:
        return json.load(fh)


def load_json_safe(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if it does not exist."""
    if not os.path.isfile(path):
        return None
    return load_json(path)


# -----------------------------------------------------------------------
# Experiment result loading
# -----------------------------------------------------------------------


def load_experiment(results_dir: str, experiment_name: str) -> pd.DataFrame:
    """Load results for a single experiment into a DataFrame.

    Expects a file at ``<results_dir>/<experiment_name>.json`` with a
    ``results`` key containing a list of record dicts.
    """
    path = os.path.join(results_dir, f"{experiment_name}.json")
    data = load_json(path)
    records = data.get("results", [])
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_all_experiments(results_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all experiment JSON files from a results directory.

    Returns a dict mapping experiment name (without extension) to its
    DataFrame.
    """
    results = {}
    results_path = Path(results_dir)
    if not results_path.is_dir():
        return results

    for json_file in sorted(results_path.glob("*.json")):
        name = json_file.stem
        try:
            df = load_experiment(results_dir, name)
            if not df.empty:
                results[name] = df
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[data_loader] Warning: skipping {json_file}: {exc}")
    return results


# -----------------------------------------------------------------------
# Specialized loaders for each experiment type
# -----------------------------------------------------------------------


def load_throughput_results(results_dir: str) -> pd.DataFrame:
    """Load E1/E2/E3 throughput comparison data."""
    df = load_experiment(results_dir, "throughput")
    expected_cols = ["benchmark", "config", "domain", "throughput_ops_sec"]
    _validate_columns(df, expected_cols, "throughput")
    return df


def load_ablation_results(results_dir: str) -> pd.DataFrame:
    """Load E2 contract ablation study data."""
    df = load_experiment(results_dir, "ablation")
    expected_cols = ["benchmark", "variant", "throughput_ops_sec"]
    _validate_columns(df, expected_cols, "ablation")
    return df


def load_convergence_results(results_dir: str) -> pd.DataFrame:
    """Load E3 Benders convergence analysis data."""
    df = load_experiment(results_dir, "convergence")
    expected_cols = ["benchmark", "iteration", "upper_bound", "lower_bound"]
    _validate_columns(df, expected_cols, "convergence")
    return df


def load_dse_proxy_results(results_dir: str) -> pd.DataFrame:
    """Load E4 DSE proxy correlation data."""
    df = load_experiment(results_dir, "dse_proxy")
    expected_cols = ["design_id", "proxy_score", "actual_throughput"]
    _validate_columns(df, expected_cols, "dse_proxy")
    return df


def load_sensitivity_results(results_dir: str) -> pd.DataFrame:
    """Load E6 sensitivity analysis data."""
    df = load_experiment(results_dir, "sensitivity")
    expected_cols = ["parameter", "value", "throughput_ops_sec"]
    _validate_columns(df, expected_cols, "sensitivity")
    return df


def load_baseline_results(results_dir: str) -> pd.DataFrame:
    """Load E5 flat compiler baseline comparison data."""
    df = load_experiment(results_dir, "baselines")
    expected_cols = ["benchmark", "platform", "throughput_ops_sec"]
    _validate_columns(df, expected_cols, "baselines")
    return df


def load_area_power_results(results_dir: str) -> pd.DataFrame:
    """Load E8 area/power/timing synthesis results."""
    df = load_experiment(results_dir, "area_power")
    expected_cols = ["component", "area_um2", "power_mw"]
    _validate_columns(df, expected_cols, "area_power")
    return df


def load_noc_comparison_results(results_dir: str) -> pd.DataFrame:
    """Load E7 NoC comparison results."""
    df = load_experiment(results_dir, "noc_comparison")
    expected_cols = ["noc_type", "area_um2", "latency_ns"]
    _validate_columns(df, expected_cols, "noc_comparison")
    return df


# -----------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------


def geomean(values: List[float]) -> float:
    """Compute the geometric mean of a list of positive values."""
    arr = np.array([v for v in values if v > 0])
    if len(arr) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(arr))))


def normalize_to_baseline(df: pd.DataFrame, value_col: str,
                          baseline_label: str,
                          group_col: str = "benchmark",
                          label_col: str = "config") -> pd.DataFrame:
    """Normalize values relative to a baseline configuration.

    For each group (e.g., benchmark), divides the value column by the
    baseline's value.
    """
    df = df.copy()
    baselines = df[df[label_col] == baseline_label].set_index(group_col)[value_col]
    df["normalized"] = df.apply(
        lambda row: (row[value_col] / baselines.get(row[group_col], 1.0)
                     if baselines.get(row[group_col], 0) > 0 else 0.0),
        axis=1,
    )
    return df


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------


def _validate_columns(df: pd.DataFrame, expected: List[str],
                      experiment_name: str) -> None:
    """Warn if expected columns are missing from the DataFrame."""
    if df.empty:
        return
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[data_loader] Warning: {experiment_name} missing columns: "
              f"{missing}")
