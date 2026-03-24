#!/usr/bin/env python3
"""CLI entry point for the LOOM Design Space Exploration framework.

Usage examples:
    # Run DSE with default settings
    python -m scripts.dse.run_dse --workload bench/matmul.tdg --output dse-out

    # Run with custom budget and tiers
    python -m scripts.dse.run_dse \\
        --workload bench/conv2d.tdg \\
        --budget 300 \\
        --tier2-threshold 0.4 \\
        --tier3-threshold 0.6 \\
        --output dse-conv2d

    # Resume a previous run
    python -m scripts.dse.run_dse \\
        --workload bench/matmul.tdg \\
        --output dse-out \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .dse_config import BOConfig, ClusteringConfig, DSEConfig, TierThresholds
from .dse_runner import DSEEngine
from .proxy_model import (
    ContractEdge,
    KernelProfile,
    WorkloadProfile,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOOM Design Space Exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--workload",
        nargs="+",
        required=True,
        help="Path(s) to workload TDG file(s) or workload profile JSON(s).",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        default="dse-output",
        help="Output directory for results, checkpoints, and plots.",
    )

    # Budget
    parser.add_argument(
        "--budget",
        type=int,
        default=200,
        help="Total BO iteration budget.",
    )
    parser.add_argument(
        "--initial-samples",
        type=int,
        default=20,
        help="Number of initial LHS samples before BO starts.",
    )

    # Tier thresholds
    parser.add_argument(
        "--tier2-threshold",
        type=float,
        default=0.3,
        help="Minimum Tier-1 composite score to promote to Tier 2.",
    )
    parser.add_argument(
        "--tier3-threshold",
        type=float,
        default=0.5,
        help="Minimum Tier-2 composite score to promote to Tier 3.",
    )

    # Clustering
    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum number of core-type clusters.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=5,
        help="Maximum number of core-type clusters.",
    )

    # Subprocess
    parser.add_argument(
        "--tapestry-bin",
        default="tapestry-pipeline",
        help="Path to the tapestry-pipeline binary.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Maximum parallel Tier-3 evaluations.",
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint in the output directory.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Iterations between checkpoint saves.",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (repeat for more).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors.",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args(argv)


def load_workload_profile(path: str) -> WorkloadProfile:
    """Load a workload profile from a JSON file.

    Expected JSON format:
    {
        "kernels": [
            {
                "name": "matmul",
                "op_histogram": {"mul": 64, "add": 64, "load": 16, "store": 8},
                "memory_footprint_bytes": 4096,
                "loads_per_iter": 16,
                "stores_per_iter": 8,
                "dfg_node_count": 32
            }, ...
        ],
        "contracts": [
            {
                "producer": "matmul",
                "consumer": "relu",
                "production_rate": 1.0,
                "consumption_rate": 1.0,
                "element_size_bytes": 4,
                "visibility": "LOCAL_SPM",
                "ordering": "FIFO"
            }, ...
        ],
        "critical_path": ["matmul", "relu"]
    }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workload file not found: {path}")

    with open(p, "r") as f:
        data = json.load(f)

    kernels = []
    for kd in data.get("kernels", []):
        kernels.append(
            KernelProfile(
                name=kd.get("name", ""),
                op_histogram=kd.get("op_histogram", {}),
                memory_footprint_bytes=kd.get("memory_footprint_bytes", 0),
                loads_per_iter=kd.get("loads_per_iter", 0),
                stores_per_iter=kd.get("stores_per_iter", 0),
                dfg_node_count=kd.get("dfg_node_count", 0),
            )
        )

    contracts = []
    for cd in data.get("contracts", []):
        contracts.append(
            ContractEdge(
                producer=cd.get("producer", ""),
                consumer=cd.get("consumer", ""),
                production_rate=cd.get("production_rate", 1.0),
                consumption_rate=cd.get("consumption_rate", 1.0),
                element_size_bytes=cd.get("element_size_bytes", 4),
                visibility=cd.get("visibility", "LOCAL_SPM"),
                ordering=cd.get("ordering", "FIFO"),
            )
        )

    return WorkloadProfile(
        kernels=kernels,
        contracts=contracts,
        critical_path=data.get("critical_path", [k.name for k in kernels]),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Configure logging
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

    # Load workload profiles
    workloads: List[WorkloadProfile] = []
    for wpath in args.workload:
        try:
            workloads.append(load_workload_profile(wpath))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logging.error("Failed to load workload %s: %s", wpath, exc)
            return 1

    if not workloads:
        logging.error("No valid workloads loaded")
        return 1

    # Build config
    config = DSEConfig(
        workload_tdgs=args.workload,
        output_dir=args.output,
        tier_thresholds=TierThresholds(
            tier2_promotion=args.tier2_threshold,
            tier3_promotion=args.tier3_threshold,
        ),
        bo=BOConfig(
            n_initial_samples=args.initial_samples,
            max_iterations=args.budget,
            seed=args.seed,
        ),
        clustering=ClusteringConfig(
            k_range=(args.k_min, args.k_max),
            seed=args.seed,
        ),
        tapestry_pipeline_bin=args.tapestry_bin,
        max_parallel_evals=args.parallel,
        enable_checkpointing=True,
        checkpoint_interval=args.checkpoint_interval,
        verbosity=0 if args.quiet else args.verbose,
    )

    # If not resuming, remove old checkpoint
    if not args.resume:
        import os
        ckpt = os.path.join(args.output, "dse_checkpoint.pkl")
        if os.path.exists(ckpt):
            os.remove(ckpt)

    # Run DSE
    engine = DSEEngine(workloads=workloads, config=config)
    result = engine.run()

    # Print summary
    print("\n" + "=" * 60)
    print("DSE Complete")
    print("=" * 60)
    print(result.summary())

    if result.pareto_front:
        print("\nPareto-optimal designs:")
        for i, entry in enumerate(result.pareto_front):
            print(f"\n--- Design {i + 1} (Tier {entry.tier}) ---")
            print(f"  Throughput: {entry.throughput:.6f}")
            print(f"  Area:       {entry.area:.0f} um2")
            print(entry.point.summary())

    print(f"\nResults saved to: {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
