"""Main DSE orchestration engine.

Coordinates multi-tier evaluation (proxy -> partial compile -> full compile),
Pareto frontier tracking, subprocess integration with tapestry-pipeline,
progress tracking, and checkpoint/resume support.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bayesian_opt import AcquisitionType, BayesianOptimizer
from .design_space import DesignPoint, DesignSpace
from .dse_config import DSEConfig, TIER2_TIMEOUT_SEC, TIER3_TIMEOUT_SEC
from .pareto import (
    ParetoEntry,
    extract_pareto_front,
    hypervolume_2d,
    merge_fronts,
    pareto_from_scores,
)
from .proxy_model import AnalyticalResourceModel, ProxyScore, WorkloadProfile
from .spectral_clustering import CoreTypeDiscovery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSE statistics
# ---------------------------------------------------------------------------

@dataclass
class DSEStats:
    """Accumulated statistics for a DSE run."""

    tier1_evaluations: int = 0
    tier2_evaluations: int = 0
    tier3_evaluations: int = 0
    tier1_time_sec: float = 0.0
    tier2_time_sec: float = 0.0
    tier3_time_sec: float = 0.0
    total_wall_time_sec: float = 0.0

    def summary(self) -> str:
        lines = [
            "DSE Statistics:",
            f"  Tier 1 evaluations: {self.tier1_evaluations} "
            f"({self.tier1_time_sec:.2f}s)",
            f"  Tier 2 evaluations: {self.tier2_evaluations} "
            f"({self.tier2_time_sec:.2f}s)",
            f"  Tier 3 evaluations: {self.tier3_evaluations} "
            f"({self.tier3_time_sec:.2f}s)",
            f"  Total wall time:    {self.total_wall_time_sec:.2f}s",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DSE result
# ---------------------------------------------------------------------------

@dataclass
class DSEResult:
    """Final result of a DSE run."""

    pareto_front: List[ParetoEntry] = field(default_factory=list)
    all_evaluated: List[ParetoEntry] = field(default_factory=list)
    stats: DSEStats = field(default_factory=DSEStats)
    core_types_discovered: int = 0

    def summary(self) -> str:
        lines = [
            f"Pareto front size: {len(self.pareto_front)}",
            f"Total evaluated: {len(self.all_evaluated)}",
            f"Core types discovered: {self.core_types_discovered}",
        ]
        if self.pareto_front:
            best = max(self.pareto_front, key=lambda e: e.throughput)
            lines.append(
                f"Best throughput: {best.throughput:.6f} "
                f"(area={best.area:.0f} um2)"
            )
        lines.append(self.stats.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------

@dataclass
class CheckpointState:
    """Serializable DSE state for resume."""

    iteration: int = 0
    observations_x: List[np.ndarray] = field(default_factory=list)
    observations_y: List[float] = field(default_factory=list)
    pareto_front: List[ParetoEntry] = field(default_factory=list)
    all_evaluated: List[ParetoEntry] = field(default_factory=list)
    stats: DSEStats = field(default_factory=DSEStats)


# ---------------------------------------------------------------------------
# DSE Engine
# ---------------------------------------------------------------------------

class DSEEngine:
    """Multi-fidelity Design Space Exploration engine.

    Orchestrates the exploration loop:
    1. Profile kernels and discover core types via spectral clustering.
    2. Define the search space around discovered core types.
    3. Run Bayesian Optimization with multi-tier evaluation:
       - Tier 1: contract proxy (~1ms)
       - Tier 2: partial compile via tapestry-pipeline (~100ms)
       - Tier 3: full compile + simulation via tapestry-pipeline (~10s)
    4. Maintain and return the Pareto frontier.
    """

    def __init__(
        self,
        workloads: List[WorkloadProfile],
        config: Optional[DSEConfig] = None,
    ):
        self.workloads = workloads
        self.config = config or DSEConfig()

        self.proxy = AnalyticalResourceModel()
        self.stats = DSEStats()
        self.pareto_front: List[ParetoEntry] = []
        self.all_evaluated: List[ParetoEntry] = []
        self._start_iteration = 0

    def run(self) -> DSEResult:
        """Execute the full DSE loop and return results."""
        wall_start = time.time()

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Load checkpoint if available
        self._maybe_load_checkpoint()

        # Use first workload for single-workload DSE
        workload = self.workloads[0] if self.workloads else WorkloadProfile()

        # Discover core types from kernel profiles
        kernels = workload.kernels
        cluster_result = None
        if kernels:
            discovery = CoreTypeDiscovery(
                k_range=self.config.clustering.k_range,
                gamma=self.config.clustering.gamma,
                seed=self.config.clustering.seed,
            )
            cluster_result = discovery.discover(kernels)
            logger.info(
                "Discovered %d core types (silhouette=%.3f, feasibility=%.3f)",
                cluster_result.k,
                cluster_result.silhouette_score,
                cluster_result.feasibility_score,
            )

            # Assign kernels to their discovered core types
            for i, kernel in enumerate(kernels):
                kernel.assigned_core_type_idx = int(cluster_result.labels[i])

        # Set up search space and optimizer
        space = DesignSpace(seed=self.config.bo.seed)
        optimizer = BayesianOptimizer(
            space=space,
            config=self.config.bo,
            acquisition=AcquisitionType.EI,
        )

        # Replay past observations if resuming
        for x_vec, y_val in zip(
            getattr(self, "_resume_x", []),
            getattr(self, "_resume_y", []),
        ):
            pt = DesignPoint.from_vector(x_vec)
            optimizer.observe(pt, y_val)

        # Main BO loop
        max_iter = self.config.bo.max_iterations
        thresholds = self.config.tier_thresholds

        for iteration in range(self._start_iteration, max_iter):
            if self.config.verbosity >= 2:
                logger.debug("Iteration %d / %d", iteration + 1, max_iter)

            # Suggest next candidate
            candidate = optimizer.suggest()

            # If we discovered core types, inject them into the candidate
            if cluster_result is not None:
                candidate = self._inject_core_types(
                    candidate, cluster_result.core_types
                )

            # Tier 1: contract proxy (fast)
            t1_start = time.time()
            tier1_score = self._evaluate_tier1(candidate, workload)
            self.stats.tier1_time_sec += time.time() - t1_start
            self.stats.tier1_evaluations += 1

            composite = tier1_score.composite_score()
            optimizer.observe(candidate, composite)

            entry = ParetoEntry(
                point=candidate,
                throughput=tier1_score.throughput,
                area=tier1_score.area_um2,
                score=tier1_score,
                tier=1,
            )

            # Tier 2: partial compile (if promising)
            if tier1_score.feasible and composite >= thresholds.tier2_promotion:
                t2_start = time.time()
                tier2_score = self._evaluate_tier2(candidate, workload)
                self.stats.tier2_time_sec += time.time() - t2_start
                self.stats.tier2_evaluations += 1

                if tier2_score is not None:
                    t2_composite = tier2_score.composite_score()
                    entry = ParetoEntry(
                        point=candidate,
                        throughput=tier2_score.throughput,
                        area=tier2_score.area_um2,
                        score=tier2_score,
                        tier=2,
                    )

                    # Tier 3: full compile + simulation (if very promising)
                    if t2_composite >= thresholds.tier3_promotion:
                        t3_start = time.time()
                        tier3_score = self._evaluate_tier3(candidate, workload)
                        self.stats.tier3_time_sec += time.time() - t3_start
                        self.stats.tier3_evaluations += 1

                        if tier3_score is not None:
                            entry = ParetoEntry(
                                point=candidate,
                                throughput=tier3_score.throughput,
                                area=tier3_score.area_um2,
                                score=tier3_score,
                                tier=3,
                            )

            if tier1_score.feasible:
                self.all_evaluated.append(entry)
                self.pareto_front = merge_fronts(self.pareto_front, [entry])

            # Periodic checkpoint
            if (
                self.config.enable_checkpointing
                and (iteration + 1) % self.config.checkpoint_interval == 0
            ):
                self._save_checkpoint(iteration + 1, optimizer)

            # Progress logging
            if self.config.verbosity >= 1 and (iteration + 1) % 10 == 0:
                hv = hypervolume_2d(self.pareto_front)
                logger.info(
                    "Iter %d/%d  pareto=%d  best_score=%.6f  HV=%.4f  "
                    "T1=%d T2=%d T3=%d",
                    iteration + 1,
                    max_iter,
                    len(self.pareto_front),
                    optimizer.best_score,
                    hv,
                    self.stats.tier1_evaluations,
                    self.stats.tier2_evaluations,
                    self.stats.tier3_evaluations,
                )

        self.stats.total_wall_time_sec = time.time() - wall_start

        # Final checkpoint
        if self.config.enable_checkpointing:
            self._save_checkpoint(max_iter, optimizer)

        # Save results
        self._save_results()

        return DSEResult(
            pareto_front=self.pareto_front,
            all_evaluated=self.all_evaluated,
            stats=self.stats,
            core_types_discovered=(
                cluster_result.k if cluster_result else 0
            ),
        )

    # -------------------------------------------------------------------
    # Tier evaluation methods
    # -------------------------------------------------------------------

    def _evaluate_tier1(
        self,
        candidate: DesignPoint,
        workload: WorkloadProfile,
    ) -> ProxyScore:
        """Tier 1: fast analytical resource model evaluation."""
        return self.proxy.evaluate(candidate, workload)

    def _evaluate_tier2(
        self,
        candidate: DesignPoint,
        workload: WorkloadProfile,
    ) -> Optional[ProxyScore]:
        """Tier 2: partial compile via tapestry-pipeline subprocess.

        Invokes tapestry-pipeline with --max-benders-iter=1 to get a
        quick feasibility check and rough quality estimate.
        """
        return self._run_tapestry(
            candidate, workload, partial=True
        )

    def _evaluate_tier3(
        self,
        candidate: DesignPoint,
        workload: WorkloadProfile,
    ) -> Optional[ProxyScore]:
        """Tier 3: full compile + simulation via tapestry-pipeline."""
        return self._run_tapestry(
            candidate, workload, partial=False
        )

    def _run_tapestry(
        self,
        candidate: DesignPoint,
        workload: WorkloadProfile,
        partial: bool,
    ) -> Optional[ProxyScore]:
        """Invoke tapestry-pipeline as a subprocess and parse results."""
        arch_json = candidate.to_arch_json()

        with tempfile.TemporaryDirectory(prefix="dse_") as tmpdir:
            arch_path = os.path.join(tmpdir, "arch.json")
            with open(arch_path, "w") as f:
                json.dump(arch_json, f)

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                self.config.tapestry_pipeline_bin,
                "--system-arch", arch_path,
                "--o", output_dir,
            ]

            # Add TDG path if available
            if self.config.workload_tdgs:
                cmd.extend(["--tdg", self.config.workload_tdgs[0]])

            if partial:
                cmd.extend([
                    "--max-benders-iter", "1",
                    "--enable-sim", "false",
                    "--enable-rtl", "false",
                ])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=TIER2_TIMEOUT_SEC if partial else TIER3_TIMEOUT_SEC,
                )
            except FileNotFoundError:
                logger.warning(
                    "tapestry-pipeline binary not found: %s",
                    self.config.tapestry_pipeline_bin,
                )
                return None
            except subprocess.TimeoutExpired:
                logger.warning("tapestry-pipeline timed out")
                return None

            if result.returncode != 0:
                logger.debug(
                    "tapestry-pipeline failed: %s", result.stderr[:200]
                )
                return None

            return self._parse_tapestry_output(output_dir, candidate)

    def _parse_tapestry_output(
        self,
        output_dir: str,
        candidate: DesignPoint,
    ) -> Optional[ProxyScore]:
        """Parse tapestry-pipeline output for score extraction."""
        report_path = os.path.join(output_dir, "report.json")
        if not os.path.exists(report_path):
            return None

        try:
            with open(report_path, "r") as f:
                report = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        throughput = report.get("throughput", 0.0)
        area = report.get("area_um2", self.proxy._estimate_area(candidate))
        comm_cost = report.get("communication_cost", 0.0)
        utilization = report.get("utilization", 0.0)

        return ProxyScore(
            throughput=throughput,
            area_um2=area,
            communication_cost=comm_cost,
            utilization=utilization,
            feasible=True,
        )

    # -------------------------------------------------------------------
    # Core-type injection
    # -------------------------------------------------------------------

    @staticmethod
    def _inject_core_types(
        candidate: DesignPoint,
        discovered_types: List[Any],
    ) -> DesignPoint:
        """Overlay discovered core-type configs onto the candidate.

        Preserves the candidate's instance_count but uses the discovered
        FU mix, grid size, and SPM size.
        """
        import copy
        result = copy.deepcopy(candidate)
        n = min(len(result.core_types), len(discovered_types))
        for i in range(n):
            disc = discovered_types[i]
            ct = result.core_types[i]
            ct.fu_alu_count = max(ct.fu_alu_count, disc.fu_alu_count)
            ct.fu_mul_count = max(ct.fu_mul_count, disc.fu_mul_count)
            ct.fu_fp_count = max(ct.fu_fp_count, disc.fu_fp_count)
            ct.fu_mem_count = max(ct.fu_mem_count, disc.fu_mem_count)
            ct.pe_grid_rows = max(ct.pe_grid_rows, disc.pe_grid_rows)
            ct.pe_grid_cols = max(ct.pe_grid_cols, disc.pe_grid_cols)
            ct.spm_size_kb = max(ct.spm_size_kb, disc.spm_size_kb)
        return result

    # -------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------

    def _checkpoint_path(self) -> str:
        return os.path.join(self.config.output_dir, "dse_checkpoint.pkl")

    def _save_checkpoint(
        self, iteration: int, optimizer: BayesianOptimizer
    ) -> None:
        """Save current DSE state to disk."""
        state = CheckpointState(
            iteration=iteration,
            observations_x=[obs.vector for obs in optimizer.observations],
            observations_y=[obs.score for obs in optimizer.observations],
            pareto_front=self.pareto_front,
            all_evaluated=self.all_evaluated,
            stats=self.stats,
        )
        path = self._checkpoint_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(state, f)
            logger.debug("Checkpoint saved at iteration %d", iteration)
        except IOError as exc:
            logger.warning("Failed to save checkpoint: %s", exc)

    def _maybe_load_checkpoint(self) -> None:
        """Load checkpoint if it exists, for resumable runs."""
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return

        try:
            with open(path, "rb") as f:
                state: CheckpointState = pickle.load(f)
        except (IOError, pickle.UnpicklingError) as exc:
            logger.warning("Failed to load checkpoint: %s", exc)
            return

        self._start_iteration = state.iteration
        self.pareto_front = state.pareto_front
        self.all_evaluated = state.all_evaluated
        self.stats = state.stats
        self._resume_x = state.observations_x
        self._resume_y = state.observations_y

        logger.info("Resumed from checkpoint at iteration %d", state.iteration)

    # -------------------------------------------------------------------
    # Results output
    # -------------------------------------------------------------------

    def _save_results(self) -> None:
        """Save final DSE results to the output directory."""
        results_path = os.path.join(self.config.output_dir, "dse_results.json")
        pareto_data = []
        for entry in self.pareto_front:
            pareto_data.append({
                "throughput": entry.throughput,
                "area_um2": entry.area,
                "tier": entry.tier,
                "design": entry.point.to_arch_json(),
            })

        output = {
            "pareto_front": pareto_data,
            "stats": {
                "tier1_evaluations": self.stats.tier1_evaluations,
                "tier2_evaluations": self.stats.tier2_evaluations,
                "tier3_evaluations": self.stats.tier3_evaluations,
                "total_wall_time_sec": self.stats.total_wall_time_sec,
            },
            "total_evaluated": len(self.all_evaluated),
        }

        try:
            with open(results_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info("Results saved to %s", results_path)
        except IOError as exc:
            logger.warning("Failed to save results: %s", exc)
