#!/usr/bin/env python3
"""JSON result formatting utilities for baseline benchmarks.

All benchmark runners produce results as JSON dicts with a consistent schema
so that the comparison script can consume them uniformly.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "kernel_name",
    "platform",
    "mean_time_ms",
    "stddev_time_ms",
    "min_time_ms",
    "num_trials",
}

OPTIONAL_FIELDS = {
    "throughput_ops_sec",
    "total_ops",
    "avg_power_w",
    "energy_j",
    "problem_size",
    "extra",
}


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------

def make_result(
    kernel_name: str,
    platform: str,
    mean_time_ms: float,
    stddev_time_ms: float,
    min_time_ms: float,
    num_trials: int,
    total_ops: Optional[float] = None,
    avg_power_w: Optional[float] = None,
    problem_size: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a single benchmark result dict.

    Parameters
    ----------
    kernel_name : str
        Identifier for the kernel (e.g. "matmul_128x128").
    platform : str
        One of "gpu", "cpu", "cgra".
    mean_time_ms : float
        Mean execution time across measured trials.
    stddev_time_ms : float
        Standard deviation of execution times.
    min_time_ms : float
        Minimum execution time observed.
    num_trials : int
        Number of measured trials (excluding warm-up).
    total_ops : float, optional
        Total arithmetic operations for throughput calculation.
    avg_power_w : float, optional
        Average power draw during benchmark (watts).
    problem_size : dict, optional
        Problem dimensions (e.g. {"M": 128, "N": 128, "K": 128}).
    extra : dict, optional
        Any additional metadata.

    Returns
    -------
    dict
        Structured result dictionary.
    """
    result: Dict[str, Any] = {
        "kernel_name": kernel_name,
        "platform": platform,
        "mean_time_ms": mean_time_ms,
        "stddev_time_ms": stddev_time_ms,
        "min_time_ms": min_time_ms,
        "num_trials": num_trials,
    }

    if total_ops is not None:
        result["total_ops"] = total_ops
        if mean_time_ms > 0:
            result["throughput_ops_sec"] = total_ops / (mean_time_ms / 1000.0)

    if avg_power_w is not None:
        result["avg_power_w"] = avg_power_w
        result["energy_j"] = avg_power_w * (mean_time_ms / 1000.0)

    if problem_size is not None:
        result["problem_size"] = problem_size

    if extra is not None:
        result["extra"] = extra

    return result


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_results(
    results: List[Dict[str, Any]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a list of benchmark results to a JSON file.

    Parameters
    ----------
    results : list of dict
        Each element should come from ``make_result``.
    output_path : str
        Destination file path.
    metadata : dict, optional
        Top-level metadata (host info, date, etc.).
    """
    envelope: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    if metadata is not None:
        envelope["metadata"] = metadata

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(envelope, fh, indent=2)

    print(f"Results written to {out}")


def read_results(path: str) -> Dict[str, Any]:
    """Read a results JSON file and return the envelope dict."""
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Parsing binary output
# ---------------------------------------------------------------------------

def parse_timing_line(line: str) -> Optional[Dict[str, float]]:
    """Parse a line of the form 'RESULT key=val key=val ...' into a dict.

    Expected keys: mean_ms, stddev_ms, min_ms, total_ops, power_w
    Returns None if the line does not start with 'RESULT'.
    """
    line = line.strip()
    if not line.startswith("RESULT"):
        return None

    tokens = line.split()[1:]  # skip "RESULT"
    parsed: Dict[str, float] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        try:
            parsed[key] = float(val)
        except ValueError:
            parsed[key] = 0.0
    return parsed


# ---------------------------------------------------------------------------
# CLI entry for quick inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results.json>")
        sys.exit(1)

    data = read_results(sys.argv[1])
    print(json.dumps(data, indent=2))
