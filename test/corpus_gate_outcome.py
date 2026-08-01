#!/usr/bin/env python3
"""Typed per-case outcomes for the repository corpus gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import corpus_inventory
from corpus_simulation_report import DfgSimulationMetrics


class CaseOutcome(Enum):
    PASS = "pass"
    UNSUPPORTED = "unsupported"
    FAIL = "fail"


@dataclass(frozen=True)
class CaseResult:
    case: corpus_inventory.SourceTranslationUnit | corpus_inventory.ProgramWorkload
    outcome: CaseOutcome
    category: str | None
    detail: str | None
    duration_seconds: float
    graphs: int | None = None
    actors: int | None = None
    dfg_simulation: DfgSimulationMetrics | None = None
    selected_sources: tuple[str, ...] | None = None

    @property
    def passed(self) -> bool:
        return self.outcome is CaseOutcome.PASS

    @property
    def unsupported(self) -> bool:
        return self.outcome is CaseOutcome.UNSUPPORTED

    @property
    def failed(self) -> bool:
        return self.outcome is CaseOutcome.FAIL

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "case": self.case.case,
            "category": self.category,
            "detail": self.detail,
            "duration_seconds": round(self.duration_seconds, 3),
            "identity": self.case.identity,
            "sources": len(self.case.sources),
            "status": self.outcome.value,
            "suite": self.case.suite,
        }
        if self.graphs is not None:
            payload["graphs"] = self.graphs
        if self.actors is not None:
            payload["actors"] = self.actors
        if self.dfg_simulation is not None:
            payload["dfg_simulation"] = self.dfg_simulation.as_dict()
        if self.selected_sources is not None:
            payload["selected_sources"] = list(self.selected_sources)
        return payload
