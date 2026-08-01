"""Resource scheduling policy for the corpus gate."""

from __future__ import annotations

import threading

from simulation_conformance import (
    MAX_OUTER_WORKERS,
    outer_worker_limit,
)


DEFAULT_CASE_TIMEOUT_SECONDS = 120.0
DEFAULT_DFG_SIM_CASE_TIMEOUT_SECONDS = 30.0
MAX_CASE_WORKERS = MAX_OUTER_WORKERS


class CaseResourceLimiter:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._available = capacity
        self._condition = threading.Condition()

    def acquire(self, slots: int) -> None:
        if slots < 1 or slots > self._capacity:
            raise ValueError("case resource slots are outside limiter capacity")
        with self._condition:
            self._condition.wait_for(lambda: self._available >= slots)
            self._available -= slots

    def release(self, slots: int) -> None:
        with self._condition:
            self._available += slots
            if self._available > self._capacity:
                raise RuntimeError("case resource limiter released excess slots")
            self._condition.notify_all()


def case_resource_slots(case, stage: str, capacity: int) -> int:
    if capacity < 1:
        raise ValueError("case resource capacity must be positive")
    if stage != "dfg-sim":
        return 1
    source_count = max(1, len(case.sources))
    return min(source_count, capacity)


def default_jobs() -> int:
    return outer_worker_limit(memory_derived_limit=MAX_CASE_WORKERS)


def default_case_timeout(stage: str) -> float:
    if stage == "dfg-sim":
        return DEFAULT_DFG_SIM_CASE_TIMEOUT_SECONDS
    return DEFAULT_CASE_TIMEOUT_SECONDS
