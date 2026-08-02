#!/usr/bin/env python3
"""Human and JSON reporting for typed corpus-gate outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from corpus_gate_outcome import CaseResult
from corpus_simulation_report import DfgSimulationMetrics


@dataclass(frozen=True)
class CorpusGateReportContext:
    stage: str
    jobs: int
    candidate_jobs: int
    case_timeout_seconds: float
    dfg_simulation_timeout_seconds: float
    dfg_execution_limits: Mapping[str, int]
    config: str | None
    duration_seconds: float
    human_header: str
    target: Mapping[str, str]
    tools: Mapping[str, str]


def render_human(
    results: Sequence[CaseResult], context: CorpusGateReportContext
) -> str:
    lines = [context.human_header]
    for result in results:
        case = result.case
        line = (
            f"{result.outcome.value.upper()}  {case.identity}  "
            f"({len(case.sources)} source(s), {result.duration_seconds:.2f}s wall, "
            f"{result.cpu_seconds:.2f}s CPU, "
            f"{result.peak_resident_bytes / (1024 * 1024):.1f} MiB peak RSS)"
        )
        if result.passed and result.graphs is not None:
            line += f"  graphs={result.graphs} actors={result.actors}"
        if result.dfg_simulation is not None:
            line += (
                f" calls={result.dfg_simulation.dynamic_calls}"
                f" values={result.dfg_simulation.value_lanes_compared}"
                f" bytes={result.dfg_simulation.memory_bytes_compared}"
                " wavefront-hz="
                f"{result.dfg_simulation.wavefront_steps_per_second:.0f}"
            )
        if not result.passed:
            line += f"  [{result.category}] {result.detail}"
        lines.append(line)

    passed = sum(1 for result in results if result.passed)
    unsupported = sum(1 for result in results if result.unsupported)
    failed = sum(1 for result in results if result.failed)
    lines.append(
        f"[corpus-gate] {passed} passed, {unsupported} unsupported, "
        f"{failed} failed, {len(results)} total in "
        f"{context.duration_seconds:.1f}s "
        f"(stage={context.stage}, jobs={context.jobs})"
    )

    failure_categories = _category_counts(results, failed=True)
    unsupported_categories = _category_counts(results, failed=False)
    if failure_categories:
        lines.append(
            "[corpus-gate] failures by category: "
            + _render_categories(failure_categories)
        )
    if unsupported_categories:
        lines.append(
            "[corpus-gate] unsupported by category: "
            + _render_categories(unsupported_categories)
        )
    lines.append(f"[corpus-gate] {'PASS' if failed == 0 else 'FAIL'}")
    return "\n".join(lines) + "\n"


def render_json(results: Sequence[CaseResult], context: CorpusGateReportContext) -> str:
    passed = sum(1 for result in results if result.passed)
    unsupported = sum(1 for result in results if result.unsupported)
    failed = sum(1 for result in results if result.failed)
    suite_counts: dict[str, dict[str, int]] = {}
    for result in results:
        suite = suite_counts.setdefault(
            result.case.suite, {"pass": 0, "unsupported": 0, "fail": 0}
        )
        suite[result.outcome.value] += 1

    payload: dict[str, object] = {
        "case_count": len(results),
        "case_timeout_seconds": context.case_timeout_seconds,
        "candidate_jobs": context.candidate_jobs,
        "config": context.config,
        "dfg_execution_limits": dict(context.dfg_execution_limits),
        "dfg_simulation_timeout_seconds": context.dfg_simulation_timeout_seconds,
        "cases": [result.as_dict() for result in results],
        "cpu_seconds": round(sum(result.cpu_seconds for result in results), 6),
        "duration_seconds": round(context.duration_seconds, 3),
        "failed": failed,
        "failure_categories": _category_counts(results, failed=True),
        "jobs": context.jobs,
        "passed": passed,
        "peak_resident_bytes": max(
            (result.peak_resident_bytes for result in results), default=0
        ),
        "stage": context.stage,
        "suite_counts": suite_counts,
        "target": dict(context.target),
        "tools": dict(context.tools),
        "unsupported": unsupported,
        "unsupported_categories": _category_counts(results, failed=False),
    }
    dfg_totals = DfgSimulationMetrics.zero()
    has_dfg_result = False
    for result in results:
        if result.dfg_simulation is not None:
            dfg_totals = dfg_totals.combine(result.dfg_simulation)
            has_dfg_result = True
    if has_dfg_result:
        payload["dfg_simulation"] = dfg_totals.as_dict()
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _category_counts(results: Sequence[CaseResult], *, failed: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        selected = result.failed if failed else result.unsupported
        if selected and result.category is not None:
            counts[result.category] = counts.get(result.category, 0) + 1
    return counts


def _render_categories(categories: Mapping[str, int]) -> str:
    return ", ".join(
        f"{category}={categories[category]}" for category in sorted(categories)
    )
