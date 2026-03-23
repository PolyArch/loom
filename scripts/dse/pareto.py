"""Pareto frontier utilities.

Functions for dominance checking, frontier extraction, and hypervolume
indicator computation for multi-objective (throughput vs. area) DSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .design_space import DesignPoint
from .proxy_model import ProxyScore


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

@dataclass
class ParetoEntry:
    """One evaluated design on the Pareto frontier."""

    point: DesignPoint
    throughput: float
    area: float
    score: Optional[ProxyScore] = None
    tier: int = 1  # evaluation tier that produced this entry

    @property
    def objectives(self) -> Tuple[float, float]:
        """Return (throughput, -area) so that HIGHER is BETTER in both."""
        return (self.throughput, -self.area)


def dominates(a: ParetoEntry, b: ParetoEntry) -> bool:
    """Return True if *a* Pareto-dominates *b*.

    Dominance: a is at least as good as b in all objectives, and strictly
    better in at least one.
    """
    obj_a = a.objectives
    obj_b = b.objectives
    at_least_as_good = all(oa >= ob for oa, ob in zip(obj_a, obj_b))
    strictly_better = any(oa > ob for oa, ob in zip(obj_a, obj_b))
    return at_least_as_good and strictly_better


# ---------------------------------------------------------------------------
# Frontier extraction
# ---------------------------------------------------------------------------

def extract_pareto_front(entries: Sequence[ParetoEntry]) -> List[ParetoEntry]:
    """Return the non-dominated subset of *entries*.

    Uses a simple O(n^2) sweep which is fine for the DSE population sizes
    (typically < 10,000 entries).
    """
    if not entries:
        return []

    front: List[ParetoEntry] = []
    for candidate in entries:
        is_dominated = False
        new_front: List[ParetoEntry] = []
        for existing in front:
            if dominates(existing, candidate):
                is_dominated = True
                new_front.append(existing)
            elif dominates(candidate, existing):
                # existing is dominated by candidate; drop it
                continue
            else:
                new_front.append(existing)
        if not is_dominated:
            new_front.append(candidate)
        front = new_front

    # Sort by throughput (descending) for presentation
    front.sort(key=lambda e: e.throughput, reverse=True)
    return front


# ---------------------------------------------------------------------------
# Hypervolume indicator
# ---------------------------------------------------------------------------

def hypervolume_2d(
    front: Sequence[ParetoEntry],
    ref_point: Tuple[float, float] = (0.0, 0.0),
) -> float:
    """Compute the 2-D hypervolume indicator.

    Objectives: (throughput, -area), both to be maximized.
    The reference point should be dominated by all front members.

    Args:
        front: Pareto-optimal entries.
        ref_point: (min_throughput, max_negative_area) reference.

    Returns:
        Hypervolume (area of the dominated region).
    """
    if not front:
        return 0.0

    # Sort by first objective (throughput) ascending
    points = sorted(front, key=lambda e: e.objectives[0])

    hv = 0.0
    prev_obj1 = ref_point[1]  # -area reference

    for entry in points:
        obj0, obj1 = entry.objectives
        # Width along throughput axis
        width = obj0 - ref_point[0]
        if width <= 0:
            continue
        # Height along -area axis (improvement over previous)
        height = obj1 - prev_obj1
        if height > 0:
            hv += width * height
            prev_obj1 = obj1

    return hv


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def pareto_from_scores(
    designs: Sequence[DesignPoint],
    scores: Sequence[ProxyScore],
    tier: int = 1,
) -> List[ParetoEntry]:
    """Build ParetoEntry list from parallel lists of designs and scores."""
    entries: List[ParetoEntry] = []
    for design, score in zip(designs, scores):
        if not score.feasible:
            continue
        entries.append(
            ParetoEntry(
                point=design,
                throughput=score.throughput,
                area=score.area_um2,
                score=score,
                tier=tier,
            )
        )
    return entries


def merge_fronts(
    existing: Sequence[ParetoEntry],
    new_entries: Sequence[ParetoEntry],
) -> List[ParetoEntry]:
    """Merge new entries into an existing Pareto front."""
    combined = list(existing) + list(new_entries)
    return extract_pareto_front(combined)
