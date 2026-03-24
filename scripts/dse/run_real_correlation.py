#!/usr/bin/env python3
"""Real Tier-1 vs Tier-2 correlation experiment for the DSE proxy model.

Replaces the fabricated R^2 data. Samples design points, evaluates each with
both the analytical resource model (Tier-1) and a real tapestry-pipeline
compile (Tier-2), then computes honest correlation metrics.

Usage:
    python -m scripts.dse.run_real_correlation \
        --workload bench/matmul_workload.json \
        --num-points 50 \
        --output out/experiments/e6_dse_proxy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .design_space import DesignPoint, DesignSpace
from .dse_config import TIER2_TIMEOUT_SEC
from .proxy_model import AnalyticalResourceModel, ProxyScore, WorkloadProfile
from .run_dse import load_workload_profile

logger = logging.getLogger(__name__)


def _run_tier2_compile(
    design: DesignPoint,
    workload_path: str,
    tapestry_bin: str,
) -> Optional[Dict]:
    """Run a real Tier-2 compile for a single design point.

    Returns the parsed report.json contents, or None on failure.
    """
    arch_json = design.to_arch_json()

    with tempfile.TemporaryDirectory(prefix="corr_") as tmpdir:
        arch_path = os.path.join(tmpdir, "arch.json")
        with open(arch_path, "w") as f:
            json.dump(arch_json, f)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            tapestry_bin,
            "--system-arch", arch_path,
            "--o", output_dir,
            "--tdg", workload_path,
            "--max-benders-iter", "1",
            "--enable-sim", "false",
            "--enable-rtl", "false",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIER2_TIMEOUT_SEC,
            )
        except FileNotFoundError:
            logger.error(
                "tapestry-pipeline binary not found: %s", tapestry_bin
            )
            return None
        except subprocess.TimeoutExpired:
            logger.warning("Tier-2 compile timed out for design point")
            return None

        if result.returncode != 0:
            logger.debug(
                "tapestry-pipeline failed (rc=%d): %s",
                result.returncode,
                result.stderr[:200],
            )
            return None

        report_path = os.path.join(output_dir, "report.json")
        if not os.path.exists(report_path):
            logger.debug("No report.json produced")
            return None

        try:
            with open(report_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None


def compute_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Compute R-squared between two arrays."""
    if len(x) < 2:
        return float("nan")
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(x) < 2:
        return float("nan")
    n = len(x)
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    d = rank_x - rank_y
    rho = 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))
    return rho


def run_correlation_experiment(
    workload_path: str,
    num_points: int,
    tapestry_bin: str,
    output_dir: str,
    seed: int = 42,
) -> Dict:
    """Run the real Tier-1 vs Tier-2 correlation experiment.

    Returns a dict of correlation results.
    """
    # Load workload
    workload = load_workload_profile(workload_path)

    # Sample design points using Latin Hypercube Sampling
    space = DesignSpace(seed=seed)
    points = space.sample_latin_hypercube(num_points)

    proxy = AnalyticalResourceModel()

    tier1_throughputs: List[float] = []
    tier1_areas: List[float] = []
    tier2_throughputs: List[float] = []
    tier2_areas: List[float] = []
    successful_points = 0

    logger.info(
        "Running correlation experiment: %d design points", num_points
    )

    for i, point in enumerate(points):
        # Tier-1: analytical resource model
        t1_score = proxy.evaluate(point, workload)
        if not t1_score.feasible:
            logger.debug("Point %d: infeasible in Tier-1, skipping", i)
            continue

        # Tier-2: real compile
        t2_report = _run_tier2_compile(point, workload_path, tapestry_bin)
        if t2_report is None:
            logger.debug("Point %d: Tier-2 compile failed, skipping", i)
            continue

        t2_throughput = t2_report.get("throughput", 0.0)
        t2_area = t2_report.get("area_um2", 0.0)

        if t2_throughput <= 0 and t2_area <= 0:
            logger.debug("Point %d: Tier-2 returned zero metrics, skipping", i)
            continue

        tier1_throughputs.append(t1_score.throughput)
        tier1_areas.append(t1_score.area_um2)
        tier2_throughputs.append(t2_throughput)
        tier2_areas.append(t2_area)
        successful_points += 1

        if (i + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d evaluated, %d successful",
                i + 1, num_points, successful_points,
            )

    logger.info(
        "Completed: %d/%d points had both Tier-1 and Tier-2 results",
        successful_points, num_points,
    )

    # Compute correlation metrics
    t1_tp = np.array(tier1_throughputs)
    t1_a = np.array(tier1_areas)
    t2_tp = np.array(tier2_throughputs)
    t2_a = np.array(tier2_areas)

    r2_throughput = compute_r2(t1_tp, t2_tp)
    r2_area = compute_r2(t1_a, t2_a)
    spearman_throughput = compute_spearman(t1_tp, t2_tp)
    spearman_area = compute_spearman(t1_a, t2_a)

    results = {
        "num_sampled": num_points,
        "num_successful": successful_points,
        "method": "real_compile_comparison",
        "tier1_model": "AnalyticalResourceModel",
        "tier2_method": "tapestry-pipeline --max-benders-iter=1",
        "workload": workload_path,
        "seed": seed,
        "r2_throughput": float(r2_throughput) if not np.isnan(r2_throughput) else None,
        "r2_area": float(r2_area) if not np.isnan(r2_area) else None,
        "spearman_throughput": float(spearman_throughput) if not np.isnan(spearman_throughput) else None,
        "spearman_area": float(spearman_area) if not np.isnan(spearman_area) else None,
        "note": "Tier-1 analytical proxy vs Tier-2 single-iteration compile",
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "real_correlation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real Tier-1 vs Tier-2 correlation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--workload",
        required=True,
        help="Path to workload profile JSON.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50,
        help="Number of design points to sample.",
    )
    parser.add_argument(
        "--tapestry-bin",
        default="tapestry-pipeline",
        help="Path to the tapestry-pipeline binary.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="out/experiments/e6_dse_proxy",
        help="Output directory for correlation results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        results = run_correlation_experiment(
            workload_path=args.workload,
            num_points=args.num_points,
            tapestry_bin=args.tapestry_bin,
            output_dir=args.output,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1

    print("\n" + "=" * 60)
    print("Correlation Experiment Results")
    print("=" * 60)
    print(f"  Points sampled:       {results['num_sampled']}")
    print(f"  Points successful:    {results['num_successful']}")
    r2_tp = results.get("r2_throughput")
    r2_a = results.get("r2_area")
    sp_tp = results.get("spearman_throughput")
    sp_a = results.get("spearman_area")
    print(f"  R^2 (throughput):     {r2_tp if r2_tp is not None else 'N/A'}")
    print(f"  R^2 (area):           {r2_a if r2_a is not None else 'N/A'}")
    print(f"  Spearman (throughput): {sp_tp if sp_tp is not None else 'N/A'}")
    print(f"  Spearman (area):      {sp_a if sp_a is not None else 'N/A'}")
    print(f"\nResults saved to: {args.output}/real_correlation.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
