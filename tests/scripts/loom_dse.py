#!/usr/bin/env python3
from __future__ import annotations
"""Loom pragma design-space estimates for simple pilot kernels.

This helper is intentionally separate from cgra_schedule.py's canonical eval
blocks. It compares explicit Loom loop candidates, such as LOOM_PARALLEL(P) and
LOOM_UNROLL(U), by building the finite chunk exposed by that candidate and then
reusing the deterministic finite-resource scheduler from cgra_schedule.py.

It implements the "Optional Loom-Pragma Design-Space Estimate" section of
docs/spec-kernel-performance.md, including:

- the three-way bracket
  ``absolute_cgra_lb <= pragma_exposure_aggregate <= schedule_estimate``
  where ONLY ``absolute_cgra_lb`` is a lower bound (the other two embed the
  wave-serialization / non-overlap assumption and sit above the hardware floor);
- steady-state saturation: the per-iteration class demand, the binding class,
  and the saturation knee ``E_sat`` used to *select* a recommended exposure
  (smallest legal ``P * U >= E_sat``), rather than minimizing the wave-summed
  estimate (which always picks max ``P * U``) or forbidding scheduler backlog
  (which always picks min ``P * U`` and is self-defeating);
- ``peak_ready_backlog`` reported as a diagnostic only, never a constraint;
- optional import of measured DFG simulator execution cycles for comparison.

The result is an exploratory estimate, not a lower bound and not cycle-accurate
RTL. The current pilot covers the single dependency-parallel loop in axpy.
"""

import argparse
import csv
import dataclasses
import sys
from dataclasses import dataclass

from cgra_schedule import CLASSES, Dag, _ceil_div, evaluate, parse_config


AXPY_SOURCE_PARALLEL = 4
AXPY_SOURCE_UNROLL = 1  # No explicit LOOM_UNROLL pragma in axpy.cpp.
AXPY_SOURCE_SCHEDULE = "contiguous"
AXPY_TEST_TRIP_COUNT = 8
AXPY_SOURCE_TRIPCOUNT_TYPICAL = 256
AXPY_KERNEL = "axpy"


@dataclass(frozen=True)
class LoopCandidate:
    parallel: int
    unroll: int
    schedule: str = AXPY_SOURCE_SCHEDULE

    def validate(self) -> None:
        if self.parallel <= 0:
            raise ValueError(f"parallel must be positive, got {self.parallel}")
        if self.unroll <= 0:
            raise ValueError(f"unroll must be positive, got {self.unroll}")
        if self.schedule not in ("contiguous", "interleaved"):
            raise ValueError(
                "schedule must be 'contiguous' or 'interleaved', "
                f"got {self.schedule!r}")


@dataclass(frozen=True)
class SaturationInfo:
    """Steady-state saturation analysis for one (trip_count, config).

    Independent of the enumerated candidates: it describes the loop and the
    resource configuration, and yields the saturation knee ``E_sat``.
    """
    trip_count: int
    a_iter: int          # per-iteration P-class demand
    ld_iter: int         # per-iteration L-class demand
    st_iter: int         # per-iteration S-class demand
    binding_class: str   # 'P' | 'L' | 'S' (the class that sets the floor)
    count_binding: int   # per-iteration demand of the binding class
    cap_binding: int     # capacity of the binding class
    chunk_cp: int        # critical-path depth of one chunk (constant in E)
    E_sat: int           # smallest E with ceil(E*count_binding/cap_binding) >= CP
    absolute_cgra_lb: int  # full-trip aggregate; the only lower bound here


@dataclass(frozen=True)
class SimMetric:
    """One imported, measured DFG simulator execution-cycle datum."""
    kernel: str
    parallel: int
    unroll: int
    schedule: str
    trip_count: int
    sim_exec_cycles: int
    candidate_id: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CandidateResult:
    candidate: LoopCandidate
    trip_count: int
    exposed_iters: int
    full_waves: int
    tail_iters: int
    waves: int                    # full_waves + (1 if tail else 0); for display
    chunk_cp: int
    chunk_a: int
    chunk_ld: int
    chunk_st: int
    chunk_aggregate: int          # full chunk aggregate (one wave)
    chunk_scheduled: int          # full chunk scheduled makespan (one wave)
    absolute_cgra_lb: int         # full-trip aggregate; the ONLY lower bound
    pragma_exposure_aggregate: int  # wave-summed aggregate (exact tail); NOT a LB
    schedule_estimate: int          # wave-summed scheduled (exact tail); NOT a LB
    binding_class: str
    saturation: str               # 'latency-bound' | 'resource-bound'
    oversubscribed: bool          # exposed_iters > recommended exposure
    recommended: bool             # this exposure is the selected knee
    wave_serialization_penalty: float  # pragma_exposure_aggregate / absolute
    schedule_gap: float           # schedule_estimate / pragma_exposure_aggregate
    saturated: tuple[int, int, int]
    peak_backlog: tuple[int, int, int]
    sim_exec_cycles: int | None = None
    sim_vs_absolute: float | None = None
    sim_vs_pragma: float | None = None
    sim_vs_schedule: float | None = None


def _safe_ratio(numer: int | None, denom: int | None) -> float | None:
    if numer is None or denom is None or denom == 0:
        return None
    return numer / denom


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _factor_values(max_value: int, trip_count: int, dense: bool) -> list[int]:
    if max_value <= 0:
        raise ValueError(f"factor maximum must be positive, got {max_value}")
    if trip_count <= 0:
        raise ValueError(f"trip_count must be positive, got {trip_count}")
    upper = min(max_value, trip_count)
    if dense:
        return list(range(1, upper + 1))
    values = []
    value = 1
    while value <= upper:
        values.append(value)
        value *= 2
    return values


def enumerate_axpy_candidates(
        trip_count: int,
        max_parallel: int,
        max_unroll: int,
        schedule: str,
        dense: bool) -> list[LoopCandidate]:
    """Enumerate candidate P/U values for axpy's dependency-parallel loop."""
    values_p = _factor_values(max_parallel, trip_count, dense)
    values_u = _factor_values(max_unroll, trip_count, dense)
    candidates = [
        LoopCandidate(parallel=p, unroll=u, schedule=schedule)
        for p in values_p for u in values_u
    ]
    for candidate in candidates:
        candidate.validate()
    return candidates


def build_axpy_chunk(active_iters: int) -> Dag:
    """Build the finite chunk exposed by one axpy candidate wave.

    The chunk charges the same operation pattern as build_axpy in
    cgra_schedule.py, but only for the finite number of iterations exposed by
    LOOM_PARALLEL(P) * LOOM_UNROLL(U). For a single dependency-parallel loop,
    different P/U pairs with the same product expose the same chunk.
    """
    if active_iters <= 0:
        raise ValueError(f"active_iters must be positive, got {active_iters}")
    dag = Dag()
    r = dag.region("axpy_chunk")
    ld_alpha = r.load(kind="alpha")
    r.load(kind="N")
    for _ in range(active_iters):
        ld_x = r.load(kind="input_x")
        ld_y = r.load(kind="input_y")
        mul = r.arith(ld_x, ld_alpha, kind="mul")
        add = r.arith(mul, ld_y, kind="add")
        r.store(add, output=True, kind="output_y")
        r.induction(kind="i", compare_depends_on_read=False)
    return dag


def axpy_saturation(trip_count: int, cfg) -> SaturationInfo:
    """Per-iteration class demand, binding class, and the saturation knee E_sat.

    Per-iteration demand is the marginal cost of one iteration: chunk(2) minus
    chunk(1) cancels the once-per-wave invariant loads (alpha, N), leaving the
    intrinsic per-iteration P/L/S counts. The absolute lower bound is the
    full-trip aggregate (the same Metric-1 lower bound applied to the whole
    loop). See docs/spec-kernel-performance.md, "Steady-state saturation and
    exposure selection".
    """
    if trip_count <= 0:
        raise ValueError(f"trip_count must be positive, got {trip_count}")
    c1 = evaluate(build_axpy_chunk(1), "axpy_iter1", cfg).region_aggs[0]
    c2 = evaluate(build_axpy_chunk(2), "axpy_iter2", cfg).region_aggs[0]
    a_iter = c2.A - c1.A
    ld_iter = c2.LD - c1.LD
    st_iter = c2.ST - c1.ST
    chunk_cp = c1.CP
    demand = {"P": a_iter, "L": ld_iter, "S": st_iter}

    absolute = evaluate(build_axpy_chunk(trip_count), "axpy_full",
                        cfg).aggregate_cycles

    # Binding class: the class whose full-trip ceiling sets the resource floor.
    # max() returns the first maximal element in CLASSES order (P, L, S), so
    # ties are broken deterministically toward P then L.
    ceilings = {c: _ceil_div(trip_count * demand[c], cfg.cap(c)) for c in CLASSES}
    binding = max(CLASSES, key=lambda c: ceilings[c])
    count_binding = demand[binding]
    cap_binding = cfg.cap(binding)

    # Smallest E with ceil(E * count_binding / cap_binding) >= chunk_cp.
    # ceil(x) >= cp  <=>  x > cp-1  <=>  E*count > cap*(cp-1)
    #               <=>  E >= ceil((cap*(cp-1)+1)/count).
    if count_binding <= 0 or chunk_cp <= 0:
        e_sat = 1
    else:
        e_sat = _ceil_div(cap_binding * (chunk_cp - 1) + 1, count_binding)

    return SaturationInfo(
        trip_count=trip_count,
        a_iter=a_iter,
        ld_iter=ld_iter,
        st_iter=st_iter,
        binding_class=binding,
        count_binding=count_binding,
        cap_binding=cap_binding,
        chunk_cp=chunk_cp,
        E_sat=e_sat,
        absolute_cgra_lb=absolute,
    )


def evaluate_axpy_candidate(candidate: LoopCandidate, trip_count: int,
                            cfg, sat: SaturationInfo) -> CandidateResult:
    candidate.validate()
    if trip_count <= 0:
        raise ValueError(f"trip_count must be positive, got {trip_count}")

    requested_iters = candidate.parallel * candidate.unroll
    exposed_iters = min(trip_count, requested_iters)
    full_waves = trip_count // exposed_iters
    tail_iters = trip_count % exposed_iters

    full = evaluate(build_axpy_chunk(exposed_iters), "axpy_dse", cfg)
    agg = full.region_aggs[0]
    pressure = full.pressure
    saturated = tuple(pressure[c].saturated_cycles for c in CLASSES)
    peak_backlog = tuple(pressure[c].peak_ready_backlog for c in CLASSES)

    pragma_exposure_aggregate = full_waves * full.aggregate_cycles
    schedule_estimate = full_waves * full.scheduled_cycles
    if tail_iters > 0:
        tail = evaluate(build_axpy_chunk(tail_iters), "axpy_dse_tail", cfg)
        pragma_exposure_aggregate += tail.aggregate_cycles
        schedule_estimate += tail.scheduled_cycles

    saturation = ("latency-bound" if full.aggregate_cycles == sat.chunk_cp
                  else "resource-bound")
    wave_pen = _safe_ratio(pragma_exposure_aggregate, sat.absolute_cgra_lb)
    sched_gap = _safe_ratio(schedule_estimate, pragma_exposure_aggregate)

    return CandidateResult(
        candidate=candidate,
        trip_count=trip_count,
        exposed_iters=exposed_iters,
        full_waves=full_waves,
        tail_iters=tail_iters,
        waves=full_waves + (1 if tail_iters else 0),
        chunk_cp=agg.CP,
        chunk_a=agg.A,
        chunk_ld=agg.LD,
        chunk_st=agg.ST,
        chunk_aggregate=full.aggregate_cycles,
        chunk_scheduled=full.scheduled_cycles,
        absolute_cgra_lb=sat.absolute_cgra_lb,
        pragma_exposure_aggregate=pragma_exposure_aggregate,
        schedule_estimate=schedule_estimate,
        binding_class=sat.binding_class,
        saturation=saturation,
        oversubscribed=False,   # filled by finalize_recommendation
        recommended=False,      # filled by finalize_recommendation
        wave_serialization_penalty=wave_pen if wave_pen is not None else 0.0,
        schedule_gap=sched_gap if sched_gap is not None else 0.0,
        saturated=saturated,
        peak_backlog=peak_backlog,
    )


def select_recommended_exposure(results: list[CandidateResult],
                                sat: SaturationInfo) -> tuple[int, bool]:
    """Pick the recommended exposure: the smallest enumerated exposure that is
    resource-bound (exposed_iters >= E_sat). If none saturate the binding class
    at this trip count, recommend the largest enumerated exposure and report
    that the loop never saturates. Returns (recommended_exposure, never_sat)."""
    exposeds = sorted({r.exposed_iters for r in results})
    if not exposeds:
        return 0, True
    eligible = [e for e in exposeds if e >= sat.E_sat]
    if eligible:
        return min(eligible), False
    return max(exposeds), True


def finalize_recommendation(results: list[CandidateResult],
                            sat: SaturationInfo) -> tuple[list[CandidateResult],
                                                          int, bool]:
    rec_e, never_sat = select_recommended_exposure(results, sat)
    out = [
        dataclasses.replace(
            r,
            recommended=(r.exposed_iters == rec_e),
            oversubscribed=(r.exposed_iters > rec_e),
        )
        for r in results
    ]
    return out, rec_e, never_sat


# ---------------------------------------------------------------------------
# Imported DFG simulator metrics
# ---------------------------------------------------------------------------

_SIM_CSV_FIELDS = ("kernel", "parallel", "unroll", "schedule", "trip_count",
                   "sim_exec_cycles")


def load_sim_metrics_csv(path: str) -> list[SimMetric]:
    """Parse a measured-simulator CSV. Schema (header required):

        kernel,candidate_id,parallel,unroll,schedule,trip_count,sim_exec_cycles,notes

    candidate_id and notes are optional human-traceability columns and are not
    used for matching. Raises ValueError naming the file and offending row on a
    malformed row."""
    metrics: list[SimMetric] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in _SIM_CSV_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path}: missing required column(s) {missing}; "
                f"required header columns are {list(_SIM_CSV_FIELDS)}")
        for lineno, row in enumerate(reader, start=2):
            try:
                metric = SimMetric(
                    kernel=row["kernel"].strip(),
                    parallel=int(row["parallel"]),
                    unroll=int(row["unroll"]),
                    schedule=row["schedule"].strip(),
                    trip_count=int(row["trip_count"]),
                    sim_exec_cycles=int(row["sim_exec_cycles"]),
                    candidate_id=(row.get("candidate_id") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"{path}:{lineno}: malformed row {row!r}: {exc}") from exc
            metrics.append(metric)
    return metrics


def find_sim_metric(metrics: list[SimMetric], kernel: str,
                    candidate: LoopCandidate,
                    trip_count: int) -> SimMetric | None:
    """First metric matching kernel, parallel, unroll, schedule, trip_count."""
    for m in metrics:
        if (m.kernel == kernel and m.parallel == candidate.parallel
                and m.unroll == candidate.unroll
                and m.schedule == candidate.schedule
                and m.trip_count == trip_count):
            return m
    return None


def attach_sim_metrics(results: list[CandidateResult], kernel: str,
                       metrics: list[SimMetric]) -> list[CandidateResult]:
    out = []
    for r in results:
        m = find_sim_metric(metrics, kernel, r.candidate, r.trip_count)
        if m is None:
            out.append(r)
            continue
        out.append(dataclasses.replace(
            r,
            sim_exec_cycles=m.sim_exec_cycles,
            sim_vs_absolute=_safe_ratio(m.sim_exec_cycles, r.absolute_cgra_lb),
            sim_vs_pragma=_safe_ratio(m.sim_exec_cycles,
                                      r.pragma_exposure_aggregate),
            sim_vs_schedule=_safe_ratio(m.sim_exec_cycles, r.schedule_estimate),
        ))
    return out


# ---------------------------------------------------------------------------
# Ranking + report
# ---------------------------------------------------------------------------

def rank_results(results: list[CandidateResult]) -> list[CandidateResult]:
    """Order rows fastest-estimate first (largest exposure first) for display."""
    return sorted(
        results,
        key=lambda r: (
            r.schedule_estimate,
            -r.exposed_iters,
            r.candidate.parallel,
            r.candidate.unroll,
        ),
    )


def _current_axpy_candidate(schedule: str) -> LoopCandidate:
    return LoopCandidate(
        parallel=AXPY_SOURCE_PARALLEL,
        unroll=AXPY_SOURCE_UNROLL,
        schedule=schedule,
    )


def _group_by_exposed(results: list[CandidateResult]) -> list[list[CandidateResult]]:
    groups: dict[int, list[CandidateResult]] = {}
    for r in results:
        groups.setdefault(r.exposed_iters, []).append(r)
    ordered = []
    for exposed in sorted(groups, reverse=True):  # largest exposure first
        members = sorted(groups[exposed],
                         key=lambda r: (r.candidate.parallel, r.candidate.unroll))
        ordered.append(members)
    return ordered


def _group_sim_value(group: list[CandidateResult]):
    """Group-level imported simulator value: the single measured value, or
    'mixed' if measured candidates disagree, or None if none were measured."""
    vals = [r.sim_exec_cycles for r in group if r.sim_exec_cycles is not None]
    if not vals:
        return None
    distinct = set(vals)
    if len(distinct) == 1:
        return next(iter(distinct))
    return "mixed"


def render_axpy_report(results: list[CandidateResult], cfg, sat: SaturationInfo,
                       recommended_exposure: int, never_sat: bool,
                       schedule: str, dense: bool, has_sim: bool) -> str:
    if not results:
        raise ValueError("no candidate results to render")
    trip_count = results[0].trip_count
    current = _current_axpy_candidate(schedule)
    mode = "dense integer factors" if dense else "powers-of-two factors"

    lines: list[str] = []
    lines.append(
        f"# Loom pragma DSE: axpy  ({cfg.label}: P={cfg.P} L={cfg.L} S={cfg.S})")
    lines.append("")
    lines.append(
        f"loop=compute_loop kind=parallel schedule={schedule} "
        f"trip_count={trip_count} ({mode})")
    lines.append(
        "current source pragma candidate: "
        f"LOOM_PARALLEL({AXPY_SOURCE_PARALLEL}, {AXPY_SOURCE_SCHEDULE}), "
        f"LOOM_UNROLL({AXPY_SOURCE_UNROLL}) implied")
    lines.append("")
    lines.append(
        f"absolute_cgra_lb = {sat.absolute_cgra_lb}  "
        "(full-trip aggregate; the ONLY lower bound here)")
    lines.append(
        f"per-iteration demand: P={sat.a_iter} L={sat.ld_iter} S={sat.st_iter}; "
        f"chunk CP={sat.chunk_cp}")
    lines.append(
        f"binding class = {sat.binding_class} "
        f"({sat.count_binding}/iter vs cap {sat.cap_binding}); "
        f"E_sat = {sat.E_sat}")
    if never_sat:
        lines.append(
            f"this trip count never saturates the binding class with the "
            f"enumerated factors; recommended = largest exposure "
            f"{recommended_exposure}")
    else:
        lines.append(
            f"recommended exposure (knee) = {recommended_exposure} "
            "(smallest enumerated P*U >= E_sat)")
    lines.append(
        "bracket: absolute_cgra_lb <= pragma_exposure_aggregate <= "
        "schedule_estimate")
    lines.append(
        "note: pragma_exposure_aggregate and schedule_estimate assume waves do "
        "NOT overlap; they are NOT lower bounds (real dataflow pipelines waves "
        "and can fall below them toward absolute_cgra_lb).")
    lines.append(
        "note: peak backlog is a diagnostic, NOT a constraint -- a zero-backlog "
        "rule would select the smallest exposure and the worst throughput.")
    lines.append("")

    header = ("mark  candidates" + " " * 28 + "exposed waves pragma_agg "
              "sched_est wave_pen class           backlog(P/L/S)")
    if has_sim:
        header += "  sim_exec sim/abs sim/pragma sim/sched"
    lines.append(header)

    for group in _group_by_exposed(results):
        rep = group[0]
        marks = ""
        if rep.recommended:
            marks += "K"
        if any(r.candidate == current for r in group):
            marks += "*"
        if rep.oversubscribed:
            marks += "o"
        names = []
        for r in group:
            tag = "*" if r.candidate == current else ""
            names.append(f"P={r.candidate.parallel},U={r.candidate.unroll}{tag}")
        names_str = " ".join(names)
        backlog = "/".join(str(v) for v in rep.peak_backlog)
        row = (
            f"{marks:<4}  {names_str:<37} {rep.exposed_iters:>7} "
            f"{rep.waves:>5} {rep.pragma_exposure_aggregate:>10} "
            f"{rep.schedule_estimate:>9} "
            f"{_fmt_ratio(rep.wave_serialization_penalty):>8} "
            f"{rep.saturation:<15} {backlog:>14}")
        if has_sim:
            sim_val = _group_sim_value(group)
            if sim_val is None:
                row += "  " + f"{'n/a':>8} {'n/a':>7} {'n/a':>10} {'n/a':>9}"
            elif sim_val == "mixed":
                row += "  " + f"{'mixed':>8} {'mixed':>7} {'mixed':>10} {'mixed':>9}"
            else:
                sva = _safe_ratio(sim_val, sat.absolute_cgra_lb)
                svp = _safe_ratio(sim_val, rep.pragma_exposure_aggregate)
                svs = _safe_ratio(sim_val, rep.schedule_estimate)
                row += ("  " + f"{sim_val:>8} {_fmt_ratio(sva):>7} "
                        f"{_fmt_ratio(svp):>10} {_fmt_ratio(svs):>9}")
        lines.append(row)

    lines.append("")
    lines.append("marks: K = recommended (saturation knee), "
                 "* = current source pragma, o = oversubscribed (past the knee).")
    if any(r.tail_iters for r in results):
        lines.append(
            "Tail note: partial final waves are modeled as a separate smaller "
            "chunk (exact tail handling).")
    if has_sim:
        lines.append(
            "Simulator cycles are imported measured DFG execution cycles "
            "(keyed per P,U); only measured candidates contribute. "
            "'mixed' means measured candidates in a group disagree.")
    return "\n".join(lines)


def run_axpy(args) -> int:
    cfg = parse_config(args.config)
    sat = axpy_saturation(args.trip_count, cfg)
    candidates = enumerate_axpy_candidates(
        trip_count=args.trip_count,
        max_parallel=args.max_parallel,
        max_unroll=args.max_unroll,
        schedule=args.schedule,
        dense=args.dense,
    )
    results = [
        evaluate_axpy_candidate(candidate, args.trip_count, cfg, sat)
        for candidate in candidates
    ]
    results, recommended_exposure, never_sat = finalize_recommendation(
        results, sat)

    metrics: list[SimMetric] = []
    if args.sim_metrics_csv:
        try:
            metrics.extend(load_sim_metrics_csv(args.sim_metrics_csv))
        except (OSError, ValueError) as exc:
            print(f"error: could not load sim metrics: {exc}", file=sys.stderr)
            return 2
    if args.sim_exec_cycles is not None:
        current = _current_axpy_candidate(args.schedule)
        metrics.append(SimMetric(
            kernel=AXPY_KERNEL,
            parallel=current.parallel,
            unroll=current.unroll,
            schedule=current.schedule,
            trip_count=args.trip_count,
            sim_exec_cycles=args.sim_exec_cycles,
            candidate_id="current-source",
            notes="from --sim-exec-cycles",
        ))
    has_sim = bool(metrics)
    if has_sim:
        results = attach_sim_metrics(rank_results(results), AXPY_KERNEL, metrics)
    else:
        results = rank_results(results)

    print(render_axpy_report(results, cfg, sat, recommended_exposure, never_sat,
                             args.schedule, args.dense, has_sim))
    return 0


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _evaluate_set(trip_count: int, cfg, max_parallel: int, max_unroll: int,
                  schedule: str = AXPY_SOURCE_SCHEDULE, dense: bool = False):
    sat = axpy_saturation(trip_count, cfg)
    cands = enumerate_axpy_candidates(trip_count, max_parallel, max_unroll,
                                      schedule, dense)
    results = [evaluate_axpy_candidate(c, trip_count, cfg, sat) for c in cands]
    results, rec_e, never = finalize_recommendation(results, sat)
    return sat, results, rec_e, never


def _find(results, parallel, unroll):
    for r in results:
        if r.candidate.parallel == parallel and r.candidate.unroll == unroll:
            return r
    return None


def _run_self_tests() -> int:
    import tempfile
    import os

    errors: list[str] = []
    cfg = parse_config("6x6")

    # --- candidate enumeration (trip 8, powers of two) ---
    cands = enumerate_axpy_candidates(AXPY_TEST_TRIP_COUNT, 8, 8,
                                      AXPY_SOURCE_SCHEDULE, dense=False)
    expected = {(p, u) for p in (1, 2, 4, 8) for u in (1, 2, 4, 8)}
    got = {(c.parallel, c.unroll) for c in cands}
    if got != expected:
        errors.append(f"candidate set {sorted(got)} != {sorted(expected)}")

    # --- basic candidate shape (trip 8) ---
    sat8, res8, _, _ = _evaluate_set(AXPY_TEST_TRIP_COUNT, cfg, 8, 8)
    cur8 = _find(res8, 4, 1)
    if cur8.exposed_iters != 4 or cur8.waves != 2:
        errors.append(
            f"current axpy exposure/waves = {cur8.exposed_iters}/{cur8.waves}, "
            "want 4/2")
    if (cur8.chunk_a, cur8.chunk_ld, cur8.chunk_st) != (16, 14, 8):
        errors.append(
            f"current axpy counts = {cur8.chunk_a}/{cur8.chunk_ld}/"
            f"{cur8.chunk_st}, want 16/14/8")
    if cur8.chunk_cp != 4:
        errors.append(f"current axpy CP = {cur8.chunk_cp}, want 4")
    if cur8.schedule_estimate != cur8.chunk_scheduled * cur8.full_waves:
        errors.append("current axpy schedule_estimate is not sched*full_waves "
                      "(no tail expected at trip 8)")
    full8 = _find(res8, 8, 1)
    if full8.exposed_iters != 8 or full8.waves != 1:
        errors.append(
            f"full axpy exposure/waves = {full8.exposed_iters}/{full8.waves}, "
            "want 8/1")
    if (full8.chunk_a, full8.chunk_ld, full8.chunk_st) != (32, 26, 16):
        errors.append(
            f"full axpy counts = {full8.chunk_a}/{full8.chunk_ld}/"
            f"{full8.chunk_st}, want 32/26/16")

    # --- bracket invariants (all candidates, trip 8) ---
    for r in res8:
        if not (r.absolute_cgra_lb <= r.pragma_exposure_aggregate
                <= r.schedule_estimate):
            errors.append(
                f"bracket violated for P={r.candidate.parallel},"
                f"U={r.candidate.unroll}: "
                f"{r.absolute_cgra_lb} <= {r.pragma_exposure_aggregate} <= "
                f"{r.schedule_estimate}")
        if r.schedule_estimate <= 0:
            errors.append("schedule_estimate must be positive")

    # --- absolute_cgra_lb identity (trip 8) ---
    full_eval8 = evaluate(build_axpy_chunk(AXPY_TEST_TRIP_COUNT), "axpy",
                          cfg).aggregate_cycles
    if sat8.absolute_cgra_lb != full_eval8:
        errors.append(
            f"absolute_cgra_lb {sat8.absolute_cgra_lb} != full-trip aggregate "
            f"{full_eval8}")

    # --- saturation knee (trip 256, 6x6) ---
    sat, res, rec_e, never = _evaluate_set(256, cfg, 8, 8)
    if sat.binding_class != "L":
        errors.append(f"binding class = {sat.binding_class}, want L")
    if not (8 < sat.E_sat <= 16):
        errors.append(f"E_sat = {sat.E_sat}, want in (8, 16]")
    if never:
        errors.append("trip 256 should saturate the binding class")
    if rec_e != 16:
        errors.append(f"recommended exposure = {rec_e}, want 16")
    if sat.absolute_cgra_lb != 65:
        errors.append(f"absolute_cgra_lb = {sat.absolute_cgra_lb}, want 65")
    r8 = _find(res, 8, 1)    # exposed 8
    r16 = _find(res, 8, 2)   # exposed 16
    if r8.saturation != "latency-bound":
        errors.append(f"exposed=8 saturation = {r8.saturation}, want latency-bound")
    if r16.saturation != "resource-bound":
        errors.append(
            f"exposed=16 saturation = {r16.saturation}, want resource-bound")
    if not r16.recommended:
        errors.append("exposed=16 must be the recommended (knee) exposure")
    for parallel, unroll in ((1, 1), (1, 2), (2, 1)):
        zr = _find(res, parallel, unroll)  # zero-backlog small exposures
        if zr is not None and zr.recommended:
            errors.append(
                f"zero-backlog exposure P={parallel},U={unroll} must NOT be "
                "recommended (backlog is not a constraint)")
    # bracket again at trip 256
    for r in res:
        if not (r.absolute_cgra_lb <= r.pragma_exposure_aggregate
                <= r.schedule_estimate):
            errors.append(
                f"bracket violated at trip 256 for exposed={r.exposed_iters}")

    # --- parallel=0 rejected ---
    try:
        evaluate_axpy_candidate(LoopCandidate(0, 1), AXPY_TEST_TRIP_COUNT, cfg,
                                sat8)
        errors.append("parallel=0 was not rejected")
    except ValueError:
        pass

    # --- imported simulator metric matching ---
    csv_text = (
        "kernel,candidate_id,parallel,unroll,schedule,trip_count,"
        "sim_exec_cycles,notes\n"
        "axpy,axpy-P4-U1,4,1,contiguous,256,1234,current source pragma\n"
        "axpy,axpy-P7-U7,7,7,contiguous,256,111,not an enumerated candidate\n"
        "other,x,4,1,contiguous,256,222,wrong kernel\n"
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(csv_text)
        metrics = load_sim_metrics_csv(path)
        if len(metrics) != 3:
            errors.append(f"parsed {len(metrics)} sim metrics, want 3")
        hit = find_sim_metric(metrics, "axpy", LoopCandidate(4, 1, "contiguous"),
                              256)
        if hit is None or hit.sim_exec_cycles != 1234:
            errors.append("matching sim metric (P4,U1) not attached correctly")
        miss = find_sim_metric(metrics, "axpy",
                               LoopCandidate(2, 2, "contiguous"), 256)
        if miss is not None:
            errors.append("non-matching candidate (P2,U2) should find no metric")
        wrong_trip = find_sim_metric(metrics, "axpy",
                                     LoopCandidate(4, 1, "contiguous"), 8)
        if wrong_trip is not None:
            errors.append("trip-count mismatch should find no metric")
        attached = attach_sim_metrics(res, "axpy", metrics)
        ar = _find(attached, 4, 1)
        if ar.sim_exec_cycles != 1234 or ar.sim_vs_absolute is None:
            errors.append("attach_sim_metrics did not populate P4,U1 sim fields")
        ar22 = _find(attached, 2, 2)
        if ar22.sim_exec_cycles is not None:
            errors.append("P2,U2 should have no imported sim value")
    finally:
        os.unlink(path)

    if errors:
        for error in errors:
            print(f"  SELF-TEST FAIL: {error}")
        return 1
    print("[PASS] loom_dse self-tests")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Loom pragma design-space estimate pilots")
    parser.add_argument("--self-test", action="store_true",
                        help="run focused self-tests and exit")
    sub = parser.add_subparsers(dest="kernel")

    p_axpy = sub.add_parser("axpy", help="compare axpy compute_loop candidates")
    p_axpy.add_argument("--config", default="6x6")
    p_axpy.add_argument("--trip-count", type=int, default=AXPY_TEST_TRIP_COUNT,
                        help="loop trip count to estimate; default matches "
                             "tests/app/axpy/main.cpp")
    p_axpy.add_argument("--max-parallel", type=int, default=8)
    p_axpy.add_argument("--max-unroll", type=int, default=8)
    p_axpy.add_argument("--schedule", choices=("contiguous", "interleaved"),
                        default=AXPY_SOURCE_SCHEDULE)
    p_axpy.add_argument("--dense", action="store_true",
                        help="enumerate every integer factor instead of "
                             "powers of two")
    p_axpy.add_argument("--sim-metrics-csv", default=None,
                        help="import measured DFG simulator execution cycles "
                             "from a CSV (see load_sim_metrics_csv schema)")
    p_axpy.add_argument("--sim-exec-cycles", type=int, default=None,
                        help="measured DFG simulator execution cycles for the "
                             "current source pragma candidate (P=4,U=1)")

    args = parser.parse_args(argv)
    if args.self_test:
        return _run_self_tests()
    if args.kernel == "axpy":
        return run_axpy(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
