"""Bayesian Optimization with Gaussian Process surrogate.

Uses scikit-learn's GaussianProcessRegressor with a Matern kernel and
Expected Improvement (EI) or Upper Confidence Bound (UCB) acquisition.
Handles mixed integer/categorical parameters via the DesignPoint encoding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .design_space import DesignPoint, DesignSpace
from .dse_config import BOConfig

logger = logging.getLogger(__name__)


class AcquisitionType(Enum):
    UCB = "ucb"
    EI = "ei"


@dataclass
class BOObservation:
    """A single observed (point, score) pair."""

    point: DesignPoint
    vector: np.ndarray
    score: float


class BayesianOptimizer:
    """Bayesian Optimization wrapper for the LOOM design space.

    During the initial phase, uses Latin Hypercube Sampling.
    After enough observations, fits a GP model and optimizes an
    acquisition function to suggest the next candidate.
    """

    def __init__(
        self,
        space: DesignSpace,
        config: Optional[BOConfig] = None,
        acquisition: AcquisitionType = AcquisitionType.EI,
    ):
        self.space = space
        self.config = config or BOConfig()
        self.acquisition_type = acquisition

        self.observations: List[BOObservation] = []
        self._gp = None
        self._rng = np.random.RandomState(self.config.seed)
        self._initial_points: List[DesignPoint] = []
        self._initial_idx = 0

    @property
    def n_observed(self) -> int:
        return len(self.observations)

    @property
    def best_score(self) -> float:
        if not self.observations:
            return float("-inf")
        return max(obs.score for obs in self.observations)

    @property
    def best_point(self) -> Optional[DesignPoint]:
        if not self.observations:
            return None
        return max(self.observations, key=lambda o: o.score).point

    def suggest(self) -> DesignPoint:
        """Suggest the next design point to evaluate.

        Returns LHS samples during the initial phase, then uses the GP
        surrogate with acquisition-function optimization.
        """
        if self.n_observed < self.config.n_initial_samples:
            return self._suggest_initial()
        return self._suggest_bo()

    def observe(self, point: DesignPoint, score: float) -> None:
        """Record an observation."""
        vec = point.to_vector()
        self.observations.append(BOObservation(point, vec, score))

        # Invalidate cached GP
        self._gp = None

    def suggest_batch(self, batch_size: int) -> List[DesignPoint]:
        """Suggest a batch of points (for parallel evaluation).

        Uses a simple strategy: suggest one point at a time with a
        hallucinated observation at the GP mean for diversity.
        """
        suggestions: List[DesignPoint] = []
        temp_observations: List[BOObservation] = []

        for _ in range(batch_size):
            point = self.suggest()
            suggestions.append(point)

            # Hallucinate observation at GP mean for diversity
            if self._gp is not None:
                vec = point.to_vector().reshape(1, -1)
                predicted_mean = self._gp.predict(vec)[0]
            else:
                predicted_mean = 0.0

            hallucinated = BOObservation(point, point.to_vector(), predicted_mean)
            self.observations.append(hallucinated)
            temp_observations.append(hallucinated)
            self._gp = None

        # Remove hallucinated observations
        for obs in temp_observations:
            self.observations.remove(obs)
        self._gp = None

        return suggestions

    def get_gp_predictions(
        self, points: Sequence[DesignPoint]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return GP mean and std predictions for given points."""
        gp = self._fit_gp()
        X = np.array([p.to_vector() for p in points])
        mean, std = gp.predict(X, return_std=True)
        return mean, std

    # -------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------

    def _suggest_initial(self) -> DesignPoint:
        """Return the next LHS sample."""
        if not self._initial_points:
            self._initial_points = self.space.sample_latin_hypercube(
                self.config.n_initial_samples
            )
            self._initial_idx = 0

        if self._initial_idx < len(self._initial_points):
            point = self._initial_points[self._initial_idx]
            self._initial_idx += 1
            return point

        # Fallback: random sample
        return self.space.sample_random(1)[0]

    def _suggest_bo(self) -> DesignPoint:
        """Fit GP and optimize acquisition function."""
        from scipy.optimize import minimize

        gp = self._fit_gp()
        lo, hi = self.space.bounds()

        best_acq = float("-inf")
        best_vec = None
        n_restarts = 10

        for _ in range(n_restarts):
            x0 = self._rng.uniform(lo, hi)

            try:
                result = minimize(
                    lambda x: -self._acquisition(x, gp),
                    x0,
                    bounds=list(zip(lo, hi)),
                    method="L-BFGS-B",
                )
                if result.success or result.fun is not None:
                    acq_val = -result.fun
                    if acq_val > best_acq:
                        best_acq = acq_val
                        best_vec = result.x
            except Exception:
                continue

        if best_vec is None:
            # Fallback to random
            logger.warning("BO acquisition optimization failed; using random sample")
            return self.space.sample_random(1)[0]

        return DesignPoint.from_vector(best_vec)

    def _fit_gp(self):
        """Fit or return cached GP model."""
        if self._gp is not None:
            return self._gp

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        X = np.array([obs.vector for obs in self.observations])
        y = np.array([obs.score for obs in self.observations])

        # Normalize y for numerical stability
        self._y_mean = y.mean()
        self._y_std = max(y.std(), 1e-8)
        y_norm = (y - self._y_mean) / self._y_std

        lb, ub = self.config.length_scale_bounds
        kernel = Matern(nu=2.5, length_scale_bounds=(lb, ub)) + WhiteKernel(
            noise_level=0.1, noise_level_bounds=(1e-5, 1e1)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=self.config.seed,
            normalize_y=False,
        )
        gp.fit(X, y_norm)
        self._gp = gp
        return gp

    def _acquisition(self, x: np.ndarray, gp) -> float:
        """Evaluate the acquisition function at point x."""
        from scipy.stats import norm

        x_2d = x.reshape(1, -1)
        mu, sigma = gp.predict(x_2d, return_std=True)
        mu = mu[0]
        sigma = max(sigma[0], 1e-8)

        if self.acquisition_type == AcquisitionType.UCB:
            return mu + self.config.kappa * sigma

        # Expected Improvement
        y_best = max(
            (obs.score - self._y_mean) / self._y_std
            for obs in self.observations
        )
        improvement = mu - y_best
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return ei
