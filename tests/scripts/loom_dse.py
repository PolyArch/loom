#!/usr/bin/env python3
from __future__ import annotations
"""Lane-aware Loom-pragma design-space estimates (vector-coalescing model).

This helper compares explicit ``LOOM_PARALLEL(p)`` / ``LOOM_UNROLL(U)`` choices
(per loop level) for a kernel, on a CGRA resource configuration ``(P_pe, L, S)``.
It implements the "Optional Loom-Pragma Design-Space Estimate" section of
``docs/spec-kernel-performance.md`` under the **lane-aware + vector coalescing**
model that replaced the earlier banking model.

Model, in one paragraph
-----------------------
The *algorithmic* op counts of an exposed chunk depend only on the per-level
exposure (``p*u``), so ``p`` and ``U`` do NOT separate on the algorithmic
arithmetic or critical-path axes -- those stay a global pool. The two pragmas
separate on TWO physical axes. (1) **Control-overhead amortization** (mentor
Sihao): within a wave every exposed iteration is laid out spatially (unrolled),
so the only surviving loop control is one iterator advance per worker per wave.
Induction (iterator load/add/store/compare) is therefore charged ``P_tot`` times
per chunk -- once per worker -- so the total control op count scales as
``trip / U_tot``. ``LOOM_UNROLL`` amortizes control (``/U``); ``LOOM_PARALLEL``
does not (each worker keeps its own iterator). A fully-unrolled level carries no
loop control at all; a fully-consumed reduction is a spatial tree, so it too
carries no loop control. A SEQUENTIAL carried recurrence (tridiag) is the
exception: it cannot be spatially flattened, so its iterator is charged per
iteration and sits on the critical path. (2) **Vector coalescing on the
load/store axis**: unrolled iterations inside one worker touch *adjacent*
elements, so a contiguous group of ``V`` of them coalesces into one 256-bit
vector memory op (one lane-slot, free unpack/pack); parallel workers stride
across partitions and do NOT coalesce across the cut. The legacy path has no
address-level banking or per-worker port cap; named extended profiles may add
their explicit shared-scratchpad port correction alongside machine lanes
``L``/``S``.

Load accounting splits into RECURRING vs. one-time INVARIANT loads. Recurring
loop loads (per-iteration array elements over the wave index, plus induction
reads) scale with exposure and set the steady-state lane exposure and the binding
load term (``load = ceil(recurring / L)``, ``active_L = min(recurring, L)``).
Invariant loads (hoisted once per chunk -- e.g. axpy ``alpha``, gemv's whole
``x`` vector) are amortized (loaded once and held) and appear only in
``LD_eff = recurring + invariant`` (total traffic), never in the binding term.

Both P/U axes bias the model **toward LOOM_UNROLL** for
contiguous groups (mentor-confirmed): coalescing is bounded by ``V = 4`` and
vanishes once ``U >= V``, while control amortization keeps paying off as ``U``
grows. Both effects stay in the existing ``P``/``L``/``S`` pools -- there is no
separate control resource and no area term.

``absolute_cgra_lb`` is the full-trip, fully-coalesced aggregate over the full
lanes ``L``/``S`` -- the only lower bound. Every candidate estimate sits at or
above it. It is an exploratory estimate, not a lower bound and not RTL.
"""

import argparse
import itertools
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

from cgra_schedule import (Config, Dag, L, P, S, _ceil_div,
                           build_crc32, build_edge_update,
                           build_fft_butterfly,
                           build_bitonic_stage_modified,
                           build_bitonic_stage_tweak, evaluate, parse_config,
                           region_aggregate)

V = 4  # 64-bit scalar elements per 256-bit vector memory op (spec convention)
DEFAULT_SPAD_CAPACITY_BYTES = 4096
DEFAULT_SPAD_LOAD_PORTS = 2
DEFAULT_SPAD_STORE_PORTS = 2
DEFAULT_SPAD_ACCESS_CYCLES = 1
DEFAULT_SPAD_TARGET_NAME = "shared-spad-4k-r2w2-v4"

# Marker appended to the ``kind`` of a load that is loop-INVARIANT: hoisted once
# per chunk, its count independent of the wave exposure (e.g. axpy ``alpha``,
# gemv's whole ``x`` vector). Recurring loop loads (per-iteration array elements
# and induction reads) carry no marker. The DSE splits the two: recurring loads
# set the steady-state lane exposure and the binding load term, while invariant
# loads are amortized (loaded once and held) and reported only in ``LD_eff``.
INV = "__inv"


# ---------------------------------------------------------------------------
# Coalescing emit helpers
# ---------------------------------------------------------------------------
# A "vector load" is a single L-class node that fans out (free unpack) to the V
# scalar consumers it covers; a "vector store" is a single S-class node that
# depends on the V scalar producers it packs. This is exactly the spec's
# one-lane-slot / free-unpack-pack convention, and the existing list scheduler
# already handles a node with many successors/predecessors.

def _cloads(r, n, coalesce, kind, invariant=False):
    """Emit loads for ``n`` contiguous elements. If ``coalesce`` (the group is a
    contiguous run the target can vectorize), emit ``ceil(n/V)`` vector nodes,
    each returned V times (free unpack fan-out); otherwise ``n`` scalar loads.
    If ``invariant`` the group is hoisted once per chunk (count independent of
    exposure) and tagged so the load split can amortize it. Returns a list of
    ``n`` value handles."""
    tag = INV if invariant else ""
    handles = []
    if coalesce:
        i = 0
        while i < n:
            g = min(V, n - i)
            nid = r.load(kind=kind + "_vec" + tag)
            handles.extend([nid] * g)
            i += V
    else:
        for _ in range(n):
            handles.append(r.load(kind=kind + tag))
    return handles


def _cstores(r, values, coalesce, output, kind):
    """Emit stores for a contiguous group of value handles. If ``coalesce``,
    ``ceil(len/V)`` vector stores (each packs up to V producers); else one scalar
    store per value."""
    if coalesce:
        i = 0
        while i < len(values):
            grp = values[i:i + V]
            r.store(*grp, output=output, kind=kind + "_vec")
            i += V
    else:
        for v in values:
            r.store(v, output=output, kind=kind)


def _worker_control(r, n_workers, kind="iv"):
    """Control-overhead amortization (DSE only; the main ASAP CGRA model still
    charges induction per iteration per Convention 1).

    Within one exposed wave every iteration is laid out spatially (unrolled), so
    the only surviving loop control is a single iterator advance per worker per
    wave. Charge induction ``n_workers`` (= ``P_tot``) times per chunk -- NOT once
    per exposed body -- so the total control op count over all waves scales as
    ``trip / U_tot``: ``LOOM_UNROLL`` amortizes control (``/U``), ``LOOM_PARALLEL``
    does not. The iterator compare does not gate the spatial body, so it stays off
    the algorithmic critical path (``compare_depends_on_read=False``). These ops
    land in the existing P/L/S pools -- no separate control resource."""
    for _ in range(max(1, n_workers)):
        r.induction(kind=kind, compare_depends_on_read=False)


def _load_split(dag) -> tuple[int, int]:
    """Split a chunk's load lane-slots into (recurring, invariant).

    ``invariant`` = loads hoisted once per chunk (count independent of the wave
    exposure), tagged with ``INV`` by the builders. ``recurring`` = everything
    else: per-iteration array element loads (over the wave index) and induction
    reads, which scale with exposure. Recurring loads set the steady-state lane
    exposure and the binding load term; invariant loads are amortized (loaded
    once and held) and appear only in ``LD_eff = recurring + invariant``."""
    recurring = 0
    invariant = 0
    for region in dag.regions:
        region_invariant = sum(
            1 for node in region.nodes if node.cls == L and INV in node.kind)
        region_total = sum(1 for node in region.nodes if node.cls == L)
        recurring += region_total - region_invariant
        invariant += region_invariant
    return recurring, invariant


def _region_load_splits(dag) -> list[tuple[int, int]]:
    """Return recurring/invariant load counts aligned with ``dag.regions``."""
    splits = []
    for region in dag.regions:
        invariant = sum(
            1 for node in region.nodes if node.cls == L and INV in node.kind)
        total = sum(1 for node in region.nodes if node.cls == L)
        splits.append((total - invariant, invariant))
    return splits


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Level:
    """One loop level of a kernel's nest (outer to inner)."""
    name: str
    trip: int
    kind: str  # "parallel" | "reduction" | "sequential"

    def parallelizable(self) -> bool:
        # parallel and reduction (via LOOM_REDUCE) contribute to P_tot;
        # sequential cannot be parallelized.
        return self.kind in ("parallel", "reduction")

    def tiled(self) -> bool:
        # parallel levels are partitioned across waves (exposure = p*u); reduction
        # sequential levels are fully consumed within a chunk (exposure = trip).
        return self.kind == "parallel"


@dataclass(frozen=True)
class OrderSpec:
    """Explicit legal loop orders. The source order must be first."""
    legal_orders: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.legal_orders:
            raise ValueError("order specification must declare at least one order")
        if len(set(self.legal_orders)) != len(self.legal_orders):
            raise ValueError("order specification contains duplicate orders")


@dataclass(frozen=True)
class JamRule:
    """One declared legal outer-to-inner jam edge and its shared operands."""
    outer: str
    inner: str
    shared_operands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outer == self.inner:
            raise ValueError("jam rule outer and inner levels must differ")
        if len(set(self.shared_operands)) != len(self.shared_operands):
            raise ValueError(
                f"jam rule {self.outer}->{self.inner} repeats a shared operand")


@dataclass(frozen=True)
class JamPlan:
    """The explicit complete jam plan selected for one candidate."""
    name: str
    order: tuple[str, ...]
    edges: tuple[JamRule, ...]


@dataclass(frozen=True)
class JamPlanSpec:
    """One named, complete per-kernel jam choice."""
    name: str
    edges: tuple[JamRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("jam plan name must be non-empty")
        if self.name == "none" and self.edges:
            raise ValueError("the reserved jam plan 'none' must have no edges")
        if len(set((edge.outer, edge.inner) for edge in self.edges)) \
                != len(self.edges):
            raise ValueError(f"jam plan {self.name} repeats an edge")


@dataclass(frozen=True)
class AnalyticTargetSpec:
    """Named branch-local target assumptions for an extended DSE study."""
    name: str = DEFAULT_SPAD_TARGET_NAME
    capacity_bytes: int = DEFAULT_SPAD_CAPACITY_BYTES
    load_ports: int = DEFAULT_SPAD_LOAD_PORTS
    store_ports: int = DEFAULT_SPAD_STORE_PORTS
    shared_across_kernel: bool = True
    access_cycles: int = DEFAULT_SPAD_ACCESS_CYCLES
    vector_width: int = V

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("analytical target profile name must be non-empty")
        if not self.shared_across_kernel:
            raise ValueError(
                "the extended DSE currently supports only a kernel-shared "
                "scratchpad")
        if self.capacity_bytes <= 0:
            raise ValueError("analytical target capacity must be positive")
        if self.load_ports <= 0 or self.store_ports <= 0:
            raise ValueError("analytical target port counts must be positive")
        if self.access_cycles <= 0:
            raise ValueError("scratchpad access latency must be positive")
        if self.vector_width <= 0:
            raise ValueError("analytical target vector width must be positive")

    @property
    def alignment_elements(self) -> int:
        return self.vector_width


@dataclass(frozen=True)
class BufferSpec:
    """Unplaced per-kernel buffer metadata returned by a memory planner."""
    name: str
    element_bytes: int
    elements: tuple[int, ...]
    reuse_bearing: bool
    worker_invariant: bool
    replication_factor: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("buffer name must be non-empty")
        if self.element_bytes <= 0:
            raise ValueError(f"buffer {self.name} element size must be positive")
        if self.replication_factor <= 0:
            raise ValueError(
                f"buffer {self.name} replication factor must be positive")
        if (not self.reuse_bearing or self.worker_invariant) \
                and self.replication_factor != 1:
            raise ValueError(
                f"buffer {self.name} may replicate only when reusable state is "
                "worker-specific")

    @property
    def placement(self) -> str:
        if not self.reuse_bearing:
            return "direct"
        if self.worker_invariant:
            return "resident_shared"
        return "resident_replicated"


@dataclass(frozen=True)
class BufferPlan:
    """A whole-kernel buffer placement and compact source-element map."""
    name: str
    placement: str
    base_element: int | None
    replica_bases: tuple[int, ...]
    element_bytes: int
    replication_factor: int
    elements: tuple[int, ...]
    source_to_slot: tuple[tuple[int, int], ...]
    bytes_used: int

    def logical_elements(self, replica: int = 0) -> tuple[int, ...]:
        """Return compact logical indices in sorted source-element order."""
        if not self.replica_bases:
            return ()
        if replica < 0 or replica >= len(self.replica_bases):
            raise IndexError(
                f"buffer {self.name} replica {replica} is out of range")
        base = self.replica_bases[replica]
        slots = dict(self.source_to_slot)
        return tuple(base + slots[source]
                     for source in self.elements)

    def logical_element(self, source_element: int, replica: int = 0) -> int:
        if replica < 0 or replica >= len(self.replica_bases):
            raise IndexError(
                f"buffer {self.name} replica {replica} is out of range")
        slots = dict(self.source_to_slot)
        if source_element not in slots:
            raise KeyError(source_element)
        return self.replica_bases[replica] + slots[source_element]


@dataclass(frozen=True)
class MemoryPlan:
    target: AnalyticTargetSpec
    buffers: tuple[BufferPlan, ...]
    capacity_bytes_used: int
    proposed_capacity_bytes: int
    fallback: bool


@dataclass(frozen=True)
class PortMetrics:
    read_ops: int
    write_ops: int
    port_lb: int
    port_sched: int


class IllegalCandidateError(ValueError):
    """A transformed candidate is well-formed but illegal for its target."""


class NoLegalExtendedCandidateError(RuntimeError):
    """No extended candidate is legal for the selected analytical target."""


@dataclass(frozen=True)
class ScratchpadAccess:
    """One scalar resident-read request before fan-out and vector packing."""
    buffer: str
    logical_element: int
    logical_step: tuple[tuple[str, int], ...]
    replica: int = 0
    stream: str | None = None

    def __post_init__(self) -> None:
        if not self.buffer:
            raise ValueError("scratchpad access buffer must be non-empty")
        if self.logical_element < 0:
            raise ValueError("scratchpad logical element must be non-negative")
        if self.replica < 0:
            raise ValueError("scratchpad replica must be non-negative")
        if self.stream == "":
            raise ValueError("scratchpad stream must be non-empty or None")


@dataclass(frozen=True)
class PackedScratchpadAccess:
    """One scalar or legally coalesced scratchpad read operation."""
    buffer: str
    logical_elements: tuple[int, ...]
    logical_step: tuple[tuple[str, int], ...]
    replica: int
    stream: str | None


@dataclass(frozen=True)
class PhaseSummary:
    """Structured counts for one preload, full-compute, or wave phase."""
    A: int
    recurring_loads: int
    invariant_loads: int
    stores: int
    CP: int
    spad_read_accesses: tuple[PackedScratchpadAccess, ...] = ()
    spad_write_ops: int = 0
    port_metrics: PortMetrics | None = None
    base_scheduled: int | None = None
    control_A: int = 0
    control_loads: int = 0
    control_stores: int = 0

    def __post_init__(self) -> None:
        counts = (self.A, self.recurring_loads, self.invariant_loads,
                  self.stores, self.CP)
        if any(count < 0 for count in counts):
            raise ValueError("phase counts and CP must be non-negative")
        if self.base_scheduled is not None and self.base_scheduled < 0:
            raise ValueError("phase schedule must be non-negative")
        if self.spad_write_ops < 0:
            raise ValueError("phase scratchpad writes must be non-negative")
        if self.port_metrics is not None and self.port_metrics.port_lb < 0:
            raise ValueError("phase port metrics must be non-negative")
        controls = (self.control_A, self.control_loads, self.control_stores)
        if any(count < 0 for count in controls):
            raise ValueError("phase control counts must be non-negative")
        if (self.control_A > self.A
                or self.control_loads > self.recurring_loads
                or self.control_stores > self.stores):
            raise ValueError("phase control counts exceed total phase counts")


@dataclass(frozen=True)
class ExtendedExecutionSummary:
    preload: PhaseSummary
    full_compute: PhaseSummary
    compute_waves: tuple[PhaseSummary, ...]

    def __post_init__(self) -> None:
        if not self.compute_waves:
            raise ValueError("extended execution must contain a compute wave")


@dataclass(frozen=True)
class ExtendedPlanSummary:
    memory_plan: MemoryPlan
    jam_plan: JamPlan
    execution: ExtendedExecutionSummary
    schedule_structure_key: tuple
    preload_scalar_elements: int = 0
    scratchpad_reads: int = 0
    avoided_direct_loads: int = 0

    def __post_init__(self) -> None:
        if min(self.preload_scalar_elements, self.scratchpad_reads,
               self.avoided_direct_loads) < 0:
            raise ValueError("extended traffic counts must be non-negative")
        if not isinstance(self.schedule_structure_key, tuple) \
                or not self.schedule_structure_key:
            raise ValueError(
                "extended schedule-structure key must be a non-empty tuple")
        try:
            hash(self.schedule_structure_key)
        except TypeError as exc:
            raise ValueError(
                "extended schedule-structure key must be hashable") from exc


@dataclass(frozen=True)
class PhaseCost:
    aggregate: int
    scheduled: int | None
    compute: int
    load: int
    store: int
    port: PortMetrics


@dataclass(frozen=True)
class KernelSpec:
    name: str
    levels: tuple[Level, ...]
    build_chunk: object             # callable(cand: Candidate) -> Dag
    coalesce_note: str = ""
    default_config: str = "6x6"
    repeat_waves: bool = True
    selection_mode: str = "knee"  # "knee" | "latency_fallback"
    order_spec: OrderSpec | None = None
    jam_plans: tuple[JamPlanSpec, ...] = ()
    memory_planner: object | None = None
    extended_plan_builder: object | None = None

    def level(self, name: str) -> Level:
        for lv in self.levels:
            if lv.name == name:
                return lv
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Per-kernel chunk builders (split-aware; coalesce contiguous unrolled groups)
# ---------------------------------------------------------------------------
# Each builder receives the Candidate and reads (p, u) per level. It builds ONE
# wave: p workers, each with u contiguous (unrolled) iterations at each tunable
# level; reduction/sequential levels are fully consumed. Contiguous array
# accesses over an unrolled level coalesce (per worker); parallel/strided and
# induction accesses stay scalar. A/CP depend only on exposure.

def _axpy_chunk(cand):
    p, u = cand.factors("i")
    dag = Dag()
    r = dag.region("axpy")
    ld_alpha = r.load(kind="alpha" + INV)   # invariant, loaded once per chunk
    r.load(kind="N" + INV)
    _worker_control(r, p)   # one iterator advance per worker/wave (unroll amortizes)
    for _w in range(p):
        xs = _cloads(r, u, True, "input_x")   # contiguous over unrolled i
        ys = _cloads(r, u, True, "input_y")
        outs = []
        for k in range(u):
            m = r.arith(xs[k], ld_alpha, kind="mul")
            outs.append(r.arith(m, ys[k], kind="add"))
        _cstores(r, outs, True, True, "output_y")
    return dag


def _vecsum_chunk(cand):
    p, u = cand.factors("i")
    trip = 256
    dag = Dag()
    r = dag.region("vecsum")
    r.load(kind="init" + INV)
    r.load(kind="N" + INV)
    # reduction fully consumed: p contiguous worker blocks, each coalesced over
    # its whole contiguous run (u only sets the unroll grouping; the block is
    # contiguous either way, so the vector-load count is ~trip/V regardless of
    # the p/u split -> vecsum is P/U-symmetric).
    leaves = []
    per = max(1, trip // p)
    used = 0
    # The whole loop IS the reduction: fully consumed in one wave and tree-reduced
    # (a spatial tree), so it carries no per-element iterator and no per-worker
    # iterator -- the p partial-sum lanes are tree branches, not loops. Charge a
    # single fixed residual control, independent of the p/u split -> vecsum stays
    # P/U-symmetric (contrast the TILED kernels, where control amortizes by U).
    _worker_control(r, 1)
    for w in range(p):
        block = (trip - used) if w == p - 1 else per
        block = max(0, min(block, trip - used))
        used += block
        hs = _cloads(r, block, True, "A")
        for k in range(block):
            leaves.append(hs[k])
    root = r.balanced_reduction(leaves, kind="reduce")
    acc = r.arith(root, kind="acc_merge")   # merge partial into carry
    r.store(acc, output=True, kind="sum")
    return dag


def _gemv_chunk(cand):
    pi, ui = cand.factors("i")
    _pj, _uj = cand.factors("j")   # j is a fully-consumed reduction (inert)
    Ei = pi * ui
    N = 64
    dag = Dag()
    r = dag.region("gemv")
    ld_alpha = r.load(kind="alpha" + INV)
    ld_beta = r.load(kind="beta" + INV)
    r.load(kind="M" + INV)
    r.load(kind="N" + INV)
    # x[j] is invariant of i: loaded once per chunk, contiguous over j -> coalesced.
    xloads = _cloads(r, N, True, "x", invariant=True)
    _worker_control(r, pi)   # one row-iterator advance per worker/wave (unroll amortizes;
                             # the j reduction is a spatial tree -> no j-loop control)
    for _w in range(pi):
        iny = _cloads(r, ui, True, "input_y")   # y[i] contiguous over unrolled i
        outs = []
        for kk in range(ui):
            aij = _cloads(r, N, True, "A")       # A row contiguous over j
            products = []
            for jj in range(N):
                products.append(r.arith(aij[jj], xloads[jj], kind="mul"))
            rowsum = r.balanced_reduction(products, kind="reduce")
            asum = r.arith(rowsum, ld_alpha, kind="mul_alpha")
            by = r.arith(iny[kk], ld_beta, kind="mul_beta")
            outs.append(r.arith(asum, by, kind="add"))
        _cstores(r, outs, True, True, "output_y")   # y[i] contiguous over i
    return dag


def _tridiag_chunk(cand):
    # Forward elimination sweep of Thomas: a NON-associative carried recurrence
    # (division chain). Sequential -> p forced to 1; only unroll is legal. Input
    # reads coalesce (contiguous over i) but the carried CP dominates, so there
    # is no P-vs-U distinction.
    _p, u = cand.factors("i")
    trip = 64
    dag = Dag()
    r = dag.region("tridiag_fwd")
    prev_c = r.load(kind="c_prime0" + INV)
    prev_d = r.load(kind="d_prime0" + INV)
    # coalesce the input streams over the whole trip (contiguous); the recurrence
    # still serializes the arithmetic.
    la_all = _cloads(r, trip, True, "input_a")
    lb_all = _cloads(r, trip, True, "input_b")
    lc_all = _cloads(r, trip, True, "input_c")
    ld_all = _cloads(r, trip, True, "input_d")
    couts, douts = [], []
    for i in range(trip):
        ac = r.arith(la_all[i], prev_c, kind="mul")   # a*c'[i-1] (carried)
        m = r.arith(lb_all[i], ac, kind="sub")
        cprime = r.arith(lc_all[i], m, kind="div")
        ad = r.arith(la_all[i], prev_d, kind="mul")   # a*d'[i-1] (carried)
        dn = r.arith(ld_all[i], ad, kind="sub")
        dprime = r.arith(dn, m, kind="div")
        couts.append(cprime)
        douts.append(dprime)
        r.induction(kind="i", compare_depends_on_read=True)  # carried iterator
        prev_c = cprime
        prev_d = dprime
    _cstores(r, couts, True, False, "c_prime")
    _cstores(r, douts, True, True, "d_prime")
    return dag


def _conv2d_chunk(cand):
    p_out, u_out = cand.factors("out")
    _pt, _ut = cand.factors("tap")   # tap is a fully-consumed reduction (inert)
    K = _conv2d_dims()[1]
    dag = Dag()
    r = dag.region("conv2d")
    r.load(kind="params" + INV)
    _worker_control(r, p_out)   # one out-iterator advance per worker/wave (unroll amortizes;
                                # the tap reduction is a spatial tree -> no tap-loop control)
    for _w in range(p_out):
        outs = []
        for _kk in range(u_out):
            inp = _cloads(r, K, False, "input")    # strided over taps (halo)
            wt = _cloads(r, K, True, "weight")      # contiguous over taps
            products = []
            for t in range(K):
                products.append(r.arith(inp[t], wt[t], kind="mul"))
            outs.append(r.balanced_reduction(products, kind="reduce"))
        _cstores(r, outs, True, True, "output")     # output contiguous over out
    return dag


def _batchnorm_chunk(cand):
    pc, uc = cand.factors("c")
    ph, uh = cand.factors("h")
    pw, uw = cand.factors("w")
    Ec, Eh = pc * uc, ph * uh
    dag = Dag()
    r = dag.region("batchnorm")
    ld_eps = r.load(kind="eps" + INV)
    r.load(kind="C" + INV)
    r.load(kind="H" + INV)
    r.load(kind="W" + INV)
    # control amortized: one iterator advance per worker/wave over the whole
    # (c,h,w) worker set (unroll on any level amortizes its control).
    _worker_control(r, pc * ph * pw)
    for _c in range(Ec):
        lv = r.load(kind="variance")
        lm = r.load(kind="mean")
        lg = r.load(kind="gamma")
        lb = r.load(kind="beta")
        ve = r.arith(lv, ld_eps, kind="var_plus_eps")
        sq = r.arith(ve, kind="sqrt")
        inv = r.arith(sq, kind="inv_std")   # invariant across (h,w)
        for _h in range(Eh):
            for _wk in range(pw):
                ins = _cloads(r, uw, True, "input")   # contiguous over unrolled w
                outs = []
                for kk in range(uw):
                    sub = r.arith(ins[kk], lm, kind="sub")
                    nm = r.arith(sub, inv, kind="mul_inv")
                    mg = r.arith(nm, lg, kind="mul_gamma")
                    outs.append(r.arith(mg, lb, kind="add_beta"))
                _cstores(r, outs, True, True, "output")
    return dag


def _bisection_chunk(cand):
    # Single parallel loop (axpy-shaped, but 4 input streams + 2 output streams
    # and a branch). All six arrays are contiguous over i, so LOOM_UNROLL both
    # coalesces the adjacent accesses and amortizes the iterator; LOOM_PARALLEL
    # strides and carries a separate iterator per worker. Only the taken arm of
    # the if/else is counted (no predication credit): the compute chain is the
    # same either way and both arms write the same two output addresses.
    p, u = cand.factors("i")
    dag = Dag()
    r = dag.region("bisection_step")
    r.load(kind="N" + INV)   # invariant, loaded once per chunk
    _worker_control(r, p)    # one iterator advance per worker/wave (unroll amortizes)
    for _w in range(p):
        a = _cloads(r, u, True, "input_a")     # contiguous over unrolled i
        b = _cloads(r, u, True, "input_b")
        fa = _cloads(r, u, True, "input_fa")
        fc = _cloads(r, u, True, "input_fc")
        outs_a, outs_b = [], []
        for k in range(u):
            s = r.arith(a[k], b[k], kind="add")      # a + b
            c = r.arith(s, kind="mul_half")          # * 0.5 -> c
            prod = r.arith(fa[k], fc[k], kind="mul") # fa * fc
            r.arith(prod, kind="cmp_lt")             # < 0 predicate (taken-arm only)
            outs_a.append(c)   # taken arm writes (a, c) or (c, b); c is the deepest value
            outs_b.append(c)
        _cstores(r, outs_a, True, True, "output_a")  # contiguous over i
        _cstores(r, outs_b, True, True, "output_b")
    return dag


def _autocorr_chunk(cand):
    # Nested: outer PARALLEL lag loop, inner REDUCTION i loop (associative float
    # sum, tree-reduced -> no i-loop control). Structurally gemv-shaped. x[i]
    # (the un-shifted prefix) is the SAME data for every lag -> modeled invariant
    # (loaded once per chunk, gemv's x). x[i+lag] shifts with lag -> recurring,
    # contiguous over i so it coalesces. output[lag] is contiguous over lag.
    #
    # The inner reduction is modeled at its MAX length x_size (the lag=0 case);
    # the true per-lag length is x_size - lag (decays to x_size - max_lag = 96),
    # so the model conservatively over-counts inner work by ~12%. This does not
    # change the compute-bound conclusion (cross-lag reuse of x is otherwise not
    # modeled, matching the conv2d halo convention).
    p_lag, u_lag = cand.factors("lag")
    _pi, _ui = cand.factors("i")   # i is a fully-consumed reduction (inert split)
    N = 128   # x_size (inner reduction length, modeled at max)
    dag = Dag()
    r = dag.region("autocorrelation")
    r.load(kind="x_size" + INV)
    r.load(kind="max_lag" + INV)
    # x[i] prefix is invariant of lag: loaded once per chunk, contiguous -> coalesced.
    xa = _cloads(r, N, True, "x", invariant=True)
    _worker_control(r, p_lag)   # one lag-iterator advance per worker/wave (unroll
                                # amortizes; the i reduction is a spatial tree -> no i control)
    for _w in range(p_lag):
        outs = []
        for _kk in range(u_lag):
            xb = _cloads(r, N, True, "x_shift")   # x[i+lag] shifted window (recurring, coalesced)
            products = [r.arith(xa[j], xb[j], kind="mul") for j in range(N)]
            outs.append(r.balanced_reduction(products, kind="reduce"))
        _cstores(r, outs, True, True, "output")   # output[lag] contiguous over lag
    return dag


def _bit_reverse_chunk(cand):
    # Nested: outer PARALLEL i loop (independent 32-bit words), inner SEQUENTIAL
    # bit loop. The bit loop carries `result` and `value` through a
    # non-associative shift/merge recurrence -> it cannot be spatially flattened,
    # reduced, or parallelized. Following tridiag, the carried scalars are
    # threaded as dataflow edges (no per-bit round-trip), and the bit iterator is
    # charged per iteration (it stays on the critical path and cannot be
    # amortized). The outer i iterator IS amortizable (one advance per worker).
    # input_data[i] / output_reversed[i] are contiguous over i -> coalesce under
    # LOOM_UNROLL(i).
    p_i, u_i = cand.factors("i")
    _pb, _ub = cand.factors("bit")   # bit is sequential: fully laid out serially, p forced to 1
    BITS = 32
    dag = Dag()
    r = dag.region("bit_reverse")
    r.load(kind="N" + INV)
    _worker_control(r, p_i)   # outer i iterator amortized (one advance per worker/wave)
    for _w in range(p_i):
        invals = _cloads(r, u_i, True, "input_data")   # contiguous over unrolled i
        outs = []
        for _k in range(u_i):
            value = invals[_k]   # initial value (dataflow from the input load)
            result = None        # result starts at 0 (constant, anonymous)
            for _bit in range(BITS):
                shl = r.arith(result, kind="shl") if result is not None \
                    else r.arith(kind="shl")           # (result << 1); iter 0 rooted at 0
                band = r.arith(value, kind="band")     # value & 1 (fanned from one value handle)
                result = r.arith(shl, band, kind="bor")   # | -> new result (carried)
                value = r.arith(value, kind="shr")     # value >>= 1 (new value, carried)
                r.induction(kind="bit", compare_depends_on_read=True)  # inner iterator: per-iter, on CP
            outs.append(result)
        _cstores(r, outs, True, True, "output_reversed")   # contiguous over i
    return dag


def _binsearch_chunk(cand):
    # Nested: outer PARALLEL t loop (independent target searches), inner
    # SEQUENTIAL while loop with DATA-DEPENDENT termination. Modeled at the
    # worst-case probe count ceil(log2(N+1)) = 4 for N=10. The while carries
    # left/right through a non-associative, data-dependent recurrence (threaded
    # as dataflow), and its termination compare sits on the critical path per
    # probe. input_sorted[mid] is a data-dependent (non-affine) index -> a scalar
    # load that cannot coalesce. output_indices[t] is contiguous over t.
    #
    # The source pragma is LOOM_NO_PARALLEL / LOOM_NO_UNROLL; this DSE explores
    # the space anyway (as it does for every kernel) but does NOT model control
    # divergence across lanes, which is the real reason the source forbids
    # parallelizing divergent searches. See the eval discussion.
    p_t, u_t = cand.factors("t")
    _pp, _up = cand.factors("probe")   # probe is sequential: worst-case trip, p forced to 1
    PROBES = 4
    dag = Dag()
    r = dag.region("binary_search")
    r.load(kind="N" + INV)
    r.load(kind="M" + INV)
    right0 = r.load(kind="right_init" + INV)   # right = N-1, hoisted once (invariant of t)
    _worker_control(r, p_t)   # outer t iterator amortized (one advance per worker/wave)
    for _w in range(p_t):
        outs = []
        for _kk in range(u_t):
            target = r.load(kind="target")   # input_targets[t] scalar (recurring per target)
            bound = right0                   # carried left/right, threaded as dataflow
            for _pr in range(PROBES):
                r.arith(bound, kind="cmp_le")             # while (left <= right) termination (side, on CP)
                sub = r.arith(bound, kind="sub")          # right - left
                sh = r.arith(sub, kind="shift")           # >> 1
                mid = r.arith(sh, bound, kind="add_mid")  # + left -> mid
                sm = r.load(mid, kind="sorted_mid")       # input_sorted[mid]: data-dependent index,
                                                          # so the load waits on mid (on CP; no coalesce)
                r.arith(sm, target, kind="cmp_eq")        # sorted[mid] == target (break check)
                cmp_lt = r.arith(sm, target, kind="cmp_lt")   # sorted[mid] < target (branch)
                bound = r.arith(cmp_lt, bound, kind="update") # left=mid+1 or right=mid-1 (carried)
            res = r.arith(bound, kind="result")           # post-loop result read
            r.arith(res, kind="ternary_cmp")                    # (result == -1) ? ... ternary compare
            outs.append(res)
        _cstores(r, outs, True, True, "output_indices")         # output_indices[t] contiguous over t
    return dag


def _bitonic_stage_chunk(cand):
    """Representative split-aware wave for the N=8 stage=1/pass=0 fixture.

    The branch mix is periodic at four lanes: two skipped lanes, one swap lane,
    and one active non-swap lane.  Candidate exposures are powers of two, so the
    full-trip model is exact and smaller waves use the nearest conservative
    representative mix.  Conditional in-place pair accesses remain scalar:
    the no-predication gates and swap aliasing prevent treating them as a plain
    contiguous stream for vector coalescing.
    """
    p, u = cand.factors("i")
    exposed = p * u
    dag = Dag()
    r = dag.region("bitonic_stage")

    ld_stage = r.load(kind="stage" + INV)
    ld_pass = r.load(kind="pass" + INV)
    ld_N = r.load(kind="N" + INV)
    stage_p1 = r.arith(ld_stage, kind="stage_plus_1")
    distance = r.arith(ld_pass, kind="distance_shl")
    block_size = r.arith(stage_p1, kind="block_size_shl")
    r.arith(block_size, kind="half_block_shr")
    worker_indices = []
    for _ in range(p):
        worker_indices.append(
            r.induction(kind="i", compare_depends_on_read=False)["read"])

    active_count = _ceil_div(exposed, 2)
    swap_count = _ceil_div(exposed, 4)
    statuses = ([(True, True)] * swap_count
                + [(True, False)] * max(0, active_count - swap_count)
                + [(False, False)] * max(0, exposed - active_count))
    for lane, (active, swap) in enumerate(statuses):
        i_value = worker_indices[min(lane // u, p - 1)]
        block_idx = r.arith(i_value, block_size, kind="block_idx_div")
        idx_in_block = r.arith(i_value, block_size, kind="idx_in_block_mod")
        band_asc = r.arith(block_idx, kind="block_idx_and_1")
        ascending = r.arith(band_asc, kind="ascending_cmp")
        band_pred = r.arith(idx_in_block, distance, kind="idx_and_distance")
        outer_pred = r.arith(band_pred, kind="outer_pred_cmp")
        if active:
            partner = r.arith(i_value, distance, outer_pred, kind="partner_add")
            in_bounds = r.arith(partner, ld_N, kind="partner_lt_N")
            ld_i = r.load(in_bounds, kind="inplace_i")
            ld_partner = r.load(in_bounds, partner, kind="inplace_partner")
            should_swap = r.arith(ascending, ld_i, ld_partner, kind="value_cmp")
            if swap:
                r.store(ld_partner, should_swap, output=True,
                        kind="store_inplace_i")
                r.store(ld_i, should_swap, output=True,
                        kind="store_inplace_partner")
    return dag


def _bitonic_stage_modified_chunk(_cand):
    dag, _contract = build_bitonic_stage_modified()
    for node in dag.regions[0].nodes:
        if node.cls == L and node.kind in ("stage", "pass", "N"):
            node.kind += INV
    return dag


def _bitonic_stage_tweak_chunk(_cand):
    dag, _contract = build_bitonic_stage_tweak()
    for node in dag.regions[0].nodes:
        if node.cls == L and node.kind in ("stage", "pass", "N"):
            node.kind += INV
    return dag


def _clz_trip_counts_main():
    values = [0, 0x80000000, 0x40000000, 0x00000001, 0xFFFFFFFF]
    values.extend((i * 0x1234) & 0xFFFFFFFF for i in range(5, 256))
    return [None if value == 0 else 32 - value.bit_length()
            for value in values]


def _clz_chunk(cand):
    p, u = cand.factors("i")
    exposure = min(256, p * u)
    trips = _clz_trip_counts_main()
    waves = _ceil_div(len(trips), exposure)
    total_A = total_LD = total_ST = total_CP = 0
    for wave in range(waves):
        lane_trips = trips[wave * exposure:(wave + 1) * exposure]
        workers = _ceil_div(len(lane_trips), u)
        # One outer iterator per active worker plus coalesced boundary I/O.
        total_A += 2 * workers
        total_LD += workers
        total_ST += workers
        boundary = sum(_ceil_div(len(lane_trips[w * u:(w + 1) * u]), V)
                       for w in range(workers))
        total_LD += boundary
        total_ST += boundary
        lane_cps = []
        for trip in lane_trips:
            if trip is None:
                total_A += 1
                lane_cps.append(2)
            else:
                total_A += 4 * trip + 3
                total_LD += 2 * trip + 2
                total_ST += 2 * trip + 2
                lane_cps.append(5 * trip + 8)
        total_CP += max(lane_cps, default=0)
    A = _ceil_div(total_A, waves)
    LD_rec = _ceil_div(total_LD, waves)
    ST = _ceil_div(total_ST, waves)
    CP = _ceil_div(total_CP, waves)
    return _emit_counted_region(
        "clz", A, LD_rec, 1, ST, tuple(["P"] * (CP - 1) + ["S"]))


def _emit_counted_region(name, A, LD_rec, LD_inv, ST, chain_classes):
    dag = Dag()
    r = dag.region(name)
    counts = {"P": A, "L": LD_rec, "I": LD_inv, "S": ST}
    prev = None
    for cls in chain_classes:
        preds = () if prev is None else (prev,)
        if cls == "P":
            prev = r.arith(*preds, kind="critical")
        elif cls == "L":
            prev = r.load(*preds, kind="critical_load")
        else:
            prev = r.store(*preds, output=True, kind="critical_store")
        counts[cls] -= 1
    for _ in range(counts["P"]):
        r.arith(kind="counted")
    for _ in range(counts["L"]):
        r.load(kind="counted_load")
    for _ in range(counts["I"]):
        r.load(kind="parameter" + INV)
    for _ in range(counts["S"]):
        r.store(kind="counted_store")
    return dag


def _col2im_chunk(cand):
    pc, uc = cand.factors("c")
    channels = min(3, pc * uc)
    # Remove the eval's 1,365 source-level induction steps before scaling the
    # output-centric channel body. The DSE then restores one residual iterator
    # per active c worker, so U amortizes control while P retains one iterator
    # per worker. The fully exposed P1U3 floor therefore carries one iterator,
    # not all 1,365 source-loop iterations.
    workers = min(pc, channels)
    A = 10 + 4248 * channels + 2 * workers
    LD_rec = 648 * channels + workers
    LD_inv = 7
    ST = 388 * channels + workers
    return _emit_counted_region(
        "col2im", A, LD_rec, LD_inv, ST,
        ("P", "P", "P", "P", "L", "P", "P", "P", "P", "P", "P", "P", "S"))


def _crc32_chunk(_cand):
    dag, _ = build_crc32()
    return dag


def _edge_update_chunk(_cand):
    dag, _ = build_edge_update()
    return dag


def _gauss_seidel_step_chunk(_cand):
    # The outer row loop is a true in-place recurrence: row i reads the values
    # written by rows 0..i-1. Its source P/U labels therefore cannot flatten or
    # parallelize the sweep. The two j loops are associative row reductions, so
    # they are fully consumed without source-level j-loop control.
    N = 32
    dag = Dag()
    r = dag.region("gauss_seidel_step")
    r.load(kind="N" + INV)
    # input_x is read-only and reused by every row. As in gemv's x vector, load
    # the complete contiguous vector once per kernel chunk and hold it.
    input_x = _cloads(r, N, True, "input_x", invariant=True)

    def a_segment(row_base, count, kind):
        # input_A is contiguous within each lower/upper row segment. Preserve
        # one source address add per scalar access, but fuse up to V adjacent
        # loads into one vector lane-slot.
        addrs = [r.address_add(row_base, kind=kind + "_addr")
                 for _ in range(count)]
        values = []
        for start in range(0, count, V):
            group = addrs[start:start + V]
            ld = r.load(*group, kind=kind + "_vec")
            values.extend([ld] * len(group))
        return values

    output_stores = []
    for i in range(N):
        r.induction(kind="i", compare_depends_on_read=True)
        # Constants/loop indices are available at entry under the source model;
        # the row-base multiply is therefore a root on the documented CP.
        row_base = r.arith(kind="row_base_mul")

        lower_a = a_segment(row_base, i, "A_lower")
        # Coalesce the already-ready prefix, but keep output_x[i-1] as a separate
        # scalar load. Grouping the newest value with older entries would delay
        # that group behind the carried store and lengthen the six-cycle row II.
        lower_x = []
        ready_prefix = max(0, i - 1)
        for start in range(0, ready_prefix, V):
            stores = output_stores[start:min(start + V, ready_prefix)]
            ld = r.load(*stores, kind="output_x_ready_vec")
            lower_x.extend([ld] * len(stores))
        if i:
            lower_x.append(
                r.load(output_stores[i - 1], kind="output_x_latest"))
        lower_products = [r.arith(lower_a[j], lower_x[j], kind="mul_lower")
                          for j in range(i)]

        upper = N - i - 1
        upper_a = a_segment(row_base, upper, "A_upper")
        upper_x = input_x[i + 1:]
        upper_products = [r.arith(upper_a[j], upper_x[j], kind="mul_upper")
                          for j in range(upper)]

        if i == 0:
            sigma = r.balanced_reduction(upper_products, kind="sigma_reduce")
        else:
            # Everything except output_x[i-1] can be reduced while waiting for
            # the newest predecessor. The final combine keeps the accepted
            # six-cycle row recurrence: load, multiply, add, sub, div, store.
            latest = lower_products[-1]
            ready_terms = lower_products[:-1] + upper_products
            partial = r.balanced_reduction(ready_terms, kind="sigma_partial")
            sigma = r.arith(partial, latest, kind="sigma_combine")

        diag_addr = r.address_add(row_base, kind="A_diag_addr")
        diag = r.load(diag_addr, kind="A_diag")
        rhs = r.load(kind="input_b")
        numerator = r.arith(rhs, sigma, kind="sub")
        value = r.arith(numerator, diag, kind="div")
        output_stores.append(
            r.store(value, output=True, kind="output_x_store"))
    return dag


def _hist_bin_chunk(cand):
    # The source's only explicit parallel pragma belongs to the short zero-fill
    # phase. Emit those phase-local waves here, then append the dominant count
    # phase exactly once as ten concrete associative scatter buckets.
    N = 1024
    fan_ins = (110, 110, 104, 100, 100, 100, 100, 100, 100, 100)
    dag = Dag()

    # True RAW barrier: every count bucket reads the identities written by all
    # zero-fill waves. Include the partial tail (e.g. 8+2 at P1U8), with one
    # residual iterator per active worker and vector stores within each worker.
    p, u = cand.factors("zero_i")
    trip = len(fan_ins)
    wave_width = p * u
    for wave_index, start in enumerate(range(0, trip, wave_width)):
        active = min(wave_width, trip - start)
        workers = min(p, _ceil_div(active, u))
        name = "zero_fill" if wave_index == 0 else f"zero_fill.wave{wave_index}"
        zero = dag.region(name)
        _worker_control(zero, workers, kind="zero_i")
        for worker in range(workers):
            block = min(u, active - worker * u)
            for _ in range(_ceil_div(block, V)):
                zero.store(output=True, kind="output_zero_vec")

    count = dag.region("count")
    count.load(kind="N" + INV)
    num_bins = count.load(kind="num_bins" + INV)
    min_val = count.load(kind="min_val" + INV)
    max_val = count.load(kind="max_val" + INV)
    value_range = count.arith(max_val, min_val, kind="range_sub")
    bin_width = count.arith(value_range, num_bins, kind="bin_width_div")

    # input[i] is contiguous and fully consumed, so its boundary loads coalesce.
    input_values = _cloads(count, N, True, "input")
    bucket_gates = [[] for _ in fan_ins]
    bucket_output_loads = [[] for _ in fan_ins]
    for lane, value in enumerate(input_values):
        # main.cpp uses input[i] = i % 100 and bin_width=10, so this is the
        # concrete resolved bucket for the lane after the all-valid guard.
        bin_index = (lane % 100) // 10
        # Preserve short-circuit gating: the high comparison follows the low
        # comparison, and only the normal path reaches the bin calculation. The
        # clamp comparison executes, but its assignment arm is never taken.
        low = count.arith(value, min_val, kind="val_lt_min")
        high = count.arith(low, value, max_val, kind="val_ge_max")
        shifted = count.arith(high, value, min_val, kind="val_sub_min")
        bin_value = count.arith(shifted, bin_width, kind="bin_div")
        bin_store = count.store(bin_value, kind="bin_store")
        bin_load = count.load(bin_store, kind="bin_load")
        clamp = count.arith(bin_load, num_bins, kind="bin_ge_num_bins")
        bucket_gates[bin_index].append(clamp)
        bucket_output_loads[bin_index].append(
            count.load(clamp, kind="output_update_load"))

    got_fan_ins = tuple(len(bucket) for bucket in bucket_gates)
    if got_fan_ins != fan_ins:
        raise AssertionError(
            f"hist_bin fan-ins {got_fan_ins} != expected {fan_ins}")

    for gates, output_loads in zip(bucket_gates, bucket_output_loads):
        # H source updates contain H adds. Reduce H gated +1 contributions plus
        # one loaded zero identity: H+1 leaves require exactly H tree adds. The
        # remaining source output loads are counted but dead under this
        # output-centric associative interpretation.
        level = list(gates) + [output_loads[-1]]
        reduction_nodes = []
        while len(level) > 1:
            next_level = []
            index = 0
            while index + 1 < len(level):
                add = count.arith(level[index], level[index + 1],
                                  kind="bucket_add")
                reduction_nodes.append(add)
                next_level.append(add)
                index += 2
            if index < len(level):
                next_level.append(level[index])
            level = next_level
        root = level[0]
        for add in reduction_nodes:
            count.store(add, output=(add == root), kind="output_update_store")
    return dag


def _clone_region(dag, source):
    """Clone one ordered region while preserving its internal dependencies."""
    r = dag.region(source.name)
    id_map = {}
    for node in source.nodes:
        preds = [id_map[pred] for pred in node.preds]
        if node.cls == P:
            nid = r.arith(*preds, output=node.is_output, kind=node.kind)
        elif node.cls == L:
            nid = r.load(*preds, output=node.is_output, kind=node.kind)
        elif node.cls == S:
            nid = r.store(*preds, output=node.is_output, kind=node.kind)
        else:
            raise AssertionError(f"unknown FFT node class {node.cls!r}")
        id_map[node.nid] = nid


def _fft_butterfly_chunk(cand):
    p, u = cand.factors("copy_i")
    trip = 16
    wave_width = p * u
    dag = Dag()

    # Copy waves are phase-local: they drain in order before stage 1, while the
    # four FFT stages execute exactly once. Legal power-of-two factors divide
    # N=16, so every wave has p complete workers of u adjacent elements.
    for wave in range(_ceil_div(trip, wave_width)):
        r = dag.region("copy" if wave == 0 else f"copy.wave{wave}")
        if wave == 0:
            ld_N = r.load(kind="N" + INV)
            r.arith(ld_N, kind="log2f")
            r.store(kind="stage_loop_init")
        _worker_control(r, p, kind="copy_i")
        for _w in range(p):
            in_real = _cloads(r, u, True, "input_real")
            in_imag = _cloads(r, u, True, "input_imag")
            _cstores(r, in_real, True, True, "output_real")
            _cstores(r, in_imag, True, True, "output_imag")

    # Reuse the validated source-level stage DAG. Its ordered regions retain
    # the s-to-s barriers, sequential j induction, and memory-backed twiddle
    # recurrence; only the separately annotated copy loop is swept here.
    fixed_dag, _ = build_fft_butterfly(N=trip)
    for source_region in fixed_dag.regions[1:]:
        _clone_region(dag, source_region)
    return dag


def _gather_chunk(cand):
    p, u = cand.factors("i")
    dag = Dag()
    r = dag.region("gather")
    r.load(kind="N" + INV)
    ld_src_size = r.load(kind="src_size" + INV)
    _worker_control(r, p)
    for _w in range(p):
        # indices[i] and dst[i] are contiguous over an unrolled i group. The
        # loaded indices may be arbitrary, so src[indices[i]] stays scalar even
        # for the regular main.cpp fixture; read-only aliasing does not prevent
        # those scalar loads from issuing independently on different lanes.
        indices = _cloads(r, u, True, "indices")
        gathered = []
        for idx in indices:
            valid = r.arith(idx, ld_src_size, kind="idx_lt_src_size")
            gathered.append(r.load(valid, idx, kind="src_indirect"))
        _cstores(r, gathered, True, True, "dst")
    return dag


def _interpolate_trace_main():
    """Concrete per-query source trace for the N_data=32, N_query=64 fixture.

    Counts exclude outer-q control and the two parameter loads / N_data-1
    subtraction, which the chunk builder restores at wave granularity. Each LD
    count includes the contiguous input_xq[q] boundary load, and each ST count
    includes the contiguous output_yq[q] boundary store.
    """
    trace = []
    for q in range(64):
        xq = 0.5 * q
        hit = False
        for k in range(31):
            if xq >= float(k) and xq <= float(k + 1):
                probes = k + 1
                hit = True
                break
        if not hit:
            probes = 31

        if hit:
            A = 5 * probes + 7
            LD = 3 * probes + 6
            CP = 9 * probes + 7
        else:
            # The no-hit q=63 lane executes all 31 failed probes and one final
            # failing k-bound check before interpolating with the initial i=0.
            A = 5 * probes + 9
            LD = 3 * probes + 7
            CP = 9 * probes + 10
        ST = probes + 2
        trace.append((A, LD, ST, CP))
    return trace


def _interpolate_critical_chain(CP):
    """A source-shaped serial-search chain of exactly ``CP`` nodes."""
    failed_probe = ("L", "P", "L", "P", "P", "L", "P", "P", "S")
    final_check = ("L", "P")
    interpolation = ("L", "P", "L", "P", "P", "P", "P", "S")
    no_hit = failed_probe * 31 + final_check + interpolation
    if CP <= 1:
        return ("S",)
    return no_hit[:CP - 1] + ("S",)


def _interpolate_linear_chunk(cand):
    p, u = cand.factors("q")
    trace = _interpolate_trace_main()
    exposure = min(len(trace), p * u)
    waves = [trace[start:start + exposure]
             for start in range(0, len(trace), exposure)]

    total_A = total_LD_rec = total_ST = total_CP = 0
    for wave in waves:
        workers = _ceil_div(len(wave), u)
        boundary_slots = sum(
            _ceil_div(len(wave[start:start + u]), V)
            for start in range(0, len(wave), u))

        # Search and interpolation counts retain each q-private, sequential
        # concrete trace. Only input_xq[q] / output_yq[q] are contiguous over q
        # and coalesce inside one unrolled worker. input_x[k], input_x[i], and
        # input_y[i] stay recurring scalar loads: their execution or address is
        # data-dependent and conditionally gated, so this model does not assume
        # cross-query cache/broadcast reuse merely because the arrays are read-only.
        total_A += sum(row[0] for row in wave) + 1 + 2 * workers
        total_LD_rec += (sum(row[1] for row in wave) - len(wave)
                         + boundary_slots + workers)
        total_ST += (sum(row[2] for row in wave) - len(wave)
                     + boundary_slots + workers)
        total_CP += max(row[3] for row in wave)

    n_waves = len(waves)
    A = _ceil_div(total_A, n_waves)
    LD_rec = _ceil_div(total_LD_rec, n_waves)
    ST = _ceil_div(total_ST, n_waves)
    CP = _ceil_div(total_CP, n_waves)
    return _emit_counted_region(
        "interpolate_linear", A, LD_rec, 2, ST,
        _interpolate_critical_chain(CP))


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

def _conv2d_dims(C_in=3, C_out=4, H=8, W=8, KH=3, KW=3, stride=1):
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1
    n_out = C_out * OH * OW
    K = C_in * KH * KW
    return n_out, K


def _active_worker_groups(extent: int, parallel: int,
                          unroll: int) -> tuple[tuple[int, int], ...]:
    """Return local start/size groups for one exact partial-exposure wave."""
    groups = []
    consumed = 0
    for _worker in range(parallel):
        if consumed >= extent:
            break
        size = min(unroll, extent - consumed)
        groups.append((consumed, size))
        consumed += size
    if consumed != extent:
        raise ValueError(
            f"wave extent {extent} exceeds P{parallel}U{unroll} exposure")
    return tuple(groups)


def _vectorized_worker_groups(extent: int, parallel: int, unroll: int,
                              vector_width: int) \
        -> tuple[tuple[int, int], ...]:
    groups = []
    for worker_start, worker_size in _active_worker_groups(
            extent, parallel, unroll):
        for offset in range(0, worker_size, vector_width):
            groups.append((worker_start + offset,
                           min(vector_width, worker_size - offset)))
    return tuple(groups)


def _parallel_wave_boxes(
        spec: KernelSpec, cand: Candidate, level_names: tuple[str, ...]) \
        -> tuple[tuple[tuple[tuple[str, int], ...],
                       tuple[tuple[str, int], ...]], ...]:
    """Materialize exact wave origins and tail shapes over the whole kernel."""
    traversal = tuple(
        name for name in candidate_order(spec, cand) if name in level_names)
    boxes = []

    def rec(index: int, origins: dict[str, int], shapes: dict[str, int]) -> None:
        if index == len(traversal):
            boxes.append((
                tuple((name, origins[name]) for name in level_names),
                tuple((name, shapes[name]) for name in level_names)))
            return
        name = traversal[index]
        level = spec.level(name)
        base = 0
        total = level.trip
        parallel, unroll = cand.factors(name)
        exposure = parallel * unroll
        for offset in range(0, total, exposure):
            origins[name] = base + offset
            shapes[name] = min(exposure, total - offset)
            rec(index + 1, origins, shapes)

    rec(0, {}, {})
    return tuple(boxes)


def _coordinate_groups(
        order: tuple[str, ...], origins: dict[str, int],
        shapes: dict[str, int], cand: Candidate,
        grouped_level: str | None, vector_width: int) \
        -> tuple[tuple[tuple[tuple[str, int], ...], ...], ...]:
    """Enumerate scalar points, optionally vector-grouping one innermost level."""
    if grouped_level is None:
        return tuple((tuple(zip(order, values)),)
                     for values in itertools.product(*(
                         range(origins[name], origins[name] + shapes[name])
                         for name in order)))
    if order[-1] != grouped_level:
        raise ValueError("only the selected innermost level may vector-pack")
    outer = order[:-1]
    parallel, unroll = cand.factors(grouped_level)
    local_groups = _vectorized_worker_groups(
        shapes[grouped_level], parallel, unroll, vector_width)
    groups = []
    outer_values = itertools.product(*(
        range(origins[name], origins[name] + shapes[name]) for name in outer))
    for values in outer_values:
        prefix = dict(zip(outer, values))
        for local_start, size in local_groups:
            group = []
            for offset in range(size):
                point = dict(prefix)
                point[grouped_level] = (
                    origins[grouped_level] + local_start + offset)
                group.append(tuple((name, point[name]) for name in order))
            groups.append(tuple(group))
    return tuple(groups)


def _phase_with_optional_schedule(
        phase: PhaseSummary, dag: Dag | None, cfg: Config,
        label: str, schedule: bool) -> PhaseSummary:
    if not schedule:
        return phase
    if dag is None or len(dag.regions) != 1:
        raise ValueError(f"{label}: scheduled phase requires one DAG region")
    region = dag.regions[0]
    aggregate = region_aggregate(region, cfg)
    recurring, invariant = _load_split(dag)
    observed = (aggregate.A, recurring, invariant, aggregate.ST, aggregate.CP)
    expected = (phase.A, phase.recurring_loads, phase.invariant_loads,
                phase.stores, phase.CP)
    if observed != expected:
        raise AssertionError(
            f"{label}: phase counts {expected} disagree with DAG {observed}")
    scheduled = evaluate(dag, label, cfg).scheduled_cycles
    return PhaseSummary(
        A=phase.A, recurring_loads=phase.recurring_loads,
        invariant_loads=phase.invariant_loads, stores=phase.stores,
        CP=phase.CP, spad_read_accesses=phase.spad_read_accesses,
        spad_write_ops=phase.spad_write_ops,
        port_metrics=phase.port_metrics,
        base_scheduled=scheduled, control_A=phase.control_A,
        control_loads=phase.control_loads,
        control_stores=phase.control_stores)


def _zero_phase(schedule: bool) -> PhaseSummary:
    return PhaseSummary(0, 0, 0, 0, 0,
                        base_scheduled=(0 if schedule else None))


def _combine_wave_phases(waves: tuple[PhaseSummary, ...]) -> PhaseSummary:
    return PhaseSummary(
        A=sum(wave.A for wave in waves),
        recurring_loads=sum(wave.recurring_loads for wave in waves),
        invariant_loads=sum(wave.invariant_loads for wave in waves),
        stores=sum(wave.stores for wave in waves),
        CP=max((wave.CP for wave in waves), default=0),
        spad_read_accesses=tuple(
            access for wave in waves for access in wave.spad_read_accesses),
        spad_write_ops=sum(wave.spad_write_ops for wave in waves),
        control_A=sum(wave.control_A for wave in waves),
        control_loads=sum(wave.control_loads for wave in waves),
        control_stores=sum(wave.control_stores for wave in waves))


def _resident_preload_phase(
        memory: MemoryPlan, cfg: Config | None,
        schedule: bool) -> tuple[PhaseSummary, int]:
    access_specs = []
    scalar_elements = 0
    for buffer in memory.buffers:
        if buffer.placement in ("direct", "direct-fallback"):
            continue
        elements = buffer.elements
        scalar_elements += len(elements) * buffer.replication_factor
        for replica in range(buffer.replication_factor):
            for group_index, group in enumerate(preload_logical_accesses(
                    buffer, memory.target, replica=replica)):
                access_specs.append((
                    buffer.name, group, replica,
                    (("preload", group_index),)))
    operation_count = len(access_specs)
    phase = PhaseSummary(
        A=0, recurring_loads=operation_count, invariant_loads=0,
        stores=operation_count, CP=(2 if operation_count else 0),
        spad_write_ops=operation_count)
    if not schedule:
        return phase, scalar_elements
    if cfg is None:
        raise ValueError("scheduled preload phase requires a resource config")
    dag = Dag()
    region = dag.region("preload")
    for buffer_name, _group, _replica, _step in access_specs:
        load = region.load(kind=f"preload_{buffer_name}")
        region.store(load, output=True, kind=f"spad_{buffer_name}")
    return (_phase_with_optional_schedule(
        phase, dag, cfg, "resident_preload", True), scalar_elements)


def _batchnorm_memory_planner(
        spec: KernelSpec, _cand: Candidate,
        _target: AnalyticTargetSpec) -> tuple[BufferSpec, ...]:
    element_count = math.prod(spec.level(name).trip for name in ("c", "h", "w"))
    elements = tuple(range(element_count))
    return (
        BufferSpec("input", 4, elements, False, True),
        BufferSpec("output", 4, elements, False, True),
    )


def _batchnorm_wave_phase(
        spec: KernelSpec, cand: Candidate,
        origins_tuple: tuple[tuple[str, int], ...],
        shapes_tuple: tuple[tuple[str, int], ...], cfg: Config,
        target: AnalyticTargetSpec, schedule: bool) -> PhaseSummary:
    order = tuple(name for name in candidate_order(spec, cand)
                  if name in ("c", "h", "w"))
    origins = dict(origins_tuple)
    shapes = dict(shapes_tuple)
    grouped_level = "w" if order[-1] == "w" else None
    groups = _coordinate_groups(
        order, origins, shapes, cand, grouped_level, target.vector_width)
    worker_count = math.prod(len(_active_worker_groups(
        shapes[name], *cand.factors(name))) for name in ("c", "h", "w"))
    element_count = math.prod(shapes.values())
    channel_count = shapes["c"]
    phase = PhaseSummary(
        A=4 * element_count + 3 * channel_count + 2 * worker_count,
        recurring_loads=len(groups) + worker_count,
        invariant_loads=4 + 4 * channel_count,
        stores=len(groups) + worker_count,
        CP=8, control_A=2 * worker_count,
        control_loads=worker_count, control_stores=worker_count)
    if not schedule:
        return phase

    dag = Dag()
    region = dag.region("batchnorm")
    epsilon = region.load(kind="eps" + INV)
    region.load(kind="C" + INV)
    region.load(kind="H" + INV)
    region.load(kind="W" + INV)
    _worker_control(region, worker_count)
    channel_values = {}
    for channel in range(origins["c"], origins["c"] + shapes["c"]):
        variance = region.load(kind="variance" + INV)
        mean = region.load(kind="mean" + INV)
        gamma = region.load(kind="gamma" + INV)
        beta = region.load(kind="beta" + INV)
        adjusted = region.arith(variance, epsilon, kind="var_plus_eps")
        root = region.arith(adjusted, kind="sqrt")
        inverse = region.arith(root, kind="inv_std")
        channel_values[channel] = (mean, gamma, beta, inverse)
    for group in groups:
        input_value = region.load(
            kind=("input_w_vec" if grouped_level == "w" else "input_scalar"))
        outputs = []
        for point_tuple in group:
            point = dict(point_tuple)
            mean, gamma, beta, inverse = channel_values[point["c"]]
            centered = region.arith(input_value, mean, kind="sub")
            normalized = region.arith(centered, inverse, kind="mul_inv")
            scaled = region.arith(normalized, gamma, kind="mul_gamma")
            outputs.append(region.arith(scaled, beta, kind="add_beta"))
        region.store(*outputs, output=True, kind="output_vec")
    return _phase_with_optional_schedule(
        phase, dag, cfg, "batchnorm", True)


def _batchnorm_extended_plan_builder(
        spec: KernelSpec, cand: Candidate, cfg: Config,
        target: AnalyticTargetSpec, schedule: bool) -> ExtendedPlanSummary:
    memory = derive_memory_plan(spec, cand, target)
    jam = derive_jam_plan(spec, cand)
    boxes = _parallel_wave_boxes(spec, cand, ("c", "h", "w"))
    waves = tuple(_batchnorm_wave_phase(
        spec, cand, origins, shapes, cfg, target, schedule)
        for origins, shapes in boxes)
    structure = tuple(
        (shapes, tuple(
            (name, tuple(size for _start, size in _active_worker_groups(
                dict(shapes)[name], *cand.factors(name))))
            for name in ("c", "h", "w")))
        for _origins, shapes in boxes)
    return ExtendedPlanSummary(
        memory_plan=memory, jam_plan=jam,
        execution=ExtendedExecutionSummary(
            preload=_zero_phase(schedule),
            full_compute=_combine_wave_phases(waves), compute_waves=waves),
        schedule_structure_key=(
            "batchnorm", candidate_order(spec, cand), jam.name, structure))


def _gemv_memory_planner(
        spec: KernelSpec, _cand: Candidate,
        _target: AnalyticTargetSpec) -> tuple[BufferSpec, ...]:
    row_count = spec.level("i").trip
    column_count = spec.level("j").trip
    x_set = tuple(range(column_count))
    row_set = tuple(range(row_count))
    matrix_set = tuple(range(row_count * column_count))
    return (
        BufferSpec("x", 4, x_set, True, True),
        BufferSpec("A", 4, matrix_set, False, True),
        BufferSpec("input_y", 4, row_set, False, True),
        BufferSpec("output_y", 4, row_set, False, True),
    )


def _gemv_wave_phase(
        spec: KernelSpec, cand: Candidate,
        origins_tuple: tuple[tuple[str, int], ...],
        shapes_tuple: tuple[tuple[str, int], ...], cfg: Config,
        target: AnalyticTargetSpec, schedule: bool,
        memory: MemoryPlan | None = None,
        share_x: bool = True) \
        -> tuple[PhaseSummary, int]:
    origins = dict(origins_tuple)
    shapes = dict(shapes_tuple)
    parallel, unroll = cand.factors("i")
    row_groups = _active_worker_groups(shapes["i"], parallel, unroll)
    row_vector_groups = _vectorized_worker_groups(
        shapes["i"], parallel, unroll, target.vector_width)
    worker_count = len(row_groups)
    row_count = shapes["i"]
    column_count = spec.level("j").trip
    x_groups_per_worker = _ceil_div(column_count, target.vector_width)
    reduction_depth = math.ceil(math.log2(column_count)) \
        if column_count > 1 else 0
    resident = memory is not None
    x_reader_count = worker_count if share_x else row_count
    x_load_ops = x_reader_count * x_groups_per_worker
    spad_accesses = []
    if resident:
        x_buffer = next(buffer for buffer in memory.buffers
                        if buffer.name == "x")
        logical_x = x_buffer.logical_elements()
        for reader_index in range(x_reader_count):
            for group_index in range(x_groups_per_worker):
                start = group_index * target.vector_width
                group = logical_x[start:start + target.vector_width]
                spad_accesses.append(PackedScratchpadAccess(
                    "x", group,
                    (("reader", reader_index), ("group", group_index)),
                    0, "x"))
    phase = PhaseSummary(
        A=(2 * column_count + 2) * row_count + 2 * worker_count,
        recurring_loads=(
            x_groups_per_worker * row_count
            + len(row_vector_groups) + worker_count
            + (x_load_ops if resident else 0)),
        invariant_loads=4 + (0 if resident else x_load_ops),
        stores=len(row_vector_groups) + worker_count,
        CP=5 + reduction_depth, spad_read_accesses=tuple(spad_accesses),
        control_A=2 * worker_count, control_loads=worker_count,
        control_stores=worker_count)
    if not schedule:
        return phase, x_reader_count * column_count

    dag = Dag()
    region = dag.region("gemv")
    alpha = region.load(kind="alpha" + INV)
    beta = region.load(kind="beta" + INV)
    region.load(kind="M" + INV)
    region.load(kind="N" + INV)
    _worker_control(region, worker_count)
    x_buffer = (next(buffer for buffer in memory.buffers
                     if buffer.name == "x") if resident else None)
    logical_x = x_buffer.logical_elements() if x_buffer else ()

    def emit_x_handles() -> list[int]:
        handles = []
        for _group_index in range(x_groups_per_worker):
            kind = "spad_x" if resident else "x" + INV
            handle = region.load(kind=kind)
            group_size = min(
                target.vector_width, column_count - len(handles))
            handles.extend(handle for _ in range(group_size))
        return handles

    for worker_index, (worker_start, worker_size) in enumerate(row_groups):
        shared_x_handles = emit_x_handles() if share_x else None
        for local_offset in range(0, worker_size, target.vector_width):
            group_size = min(target.vector_width, worker_size - local_offset)
            y_value = region.load(kind="input_y_vec")
            outputs = []
            for lane in range(group_size):
                _row = origins["i"] + worker_start + local_offset + lane
                x_handles = (shared_x_handles if shared_x_handles is not None
                             else emit_x_handles())
                a_handles = []
                for group_index in range(x_groups_per_worker):
                    handle = region.load(kind="A_vec")
                    size = min(
                        target.vector_width, column_count - len(a_handles))
                    a_handles.extend(handle for _ in range(size))
                products = [
                    region.arith(a_handles[j], x_handles[j], kind="mul")
                    for j in range(column_count)]
                row_sum = region.balanced_reduction(products, kind="reduce")
                scaled = region.arith(row_sum, alpha, kind="mul_alpha")
                prior = region.arith(y_value, beta, kind="mul_beta")
                outputs.append(region.arith(scaled, prior, kind="add"))
            region.store(*outputs, output=True, kind="output_y_vec")
    return (_phase_with_optional_schedule(
        phase, dag, cfg, "gemv_resident" if resident else "gemv_direct", True),
        x_reader_count * column_count)


def _gemv_extended_plan_builder(
        spec: KernelSpec, cand: Candidate, cfg: Config,
        target: AnalyticTargetSpec, schedule: bool) -> ExtendedPlanSummary:
    memory = derive_memory_plan(spec, cand, target)
    jam = derive_jam_plan(spec, cand)
    resident = not memory.fallback
    share_x = any(
        edge.outer == "i" and edge.inner == "j"
        and "x" in edge.shared_operands for edge in jam.edges)
    preload, preload_scalar_elements = (
        _resident_preload_phase(memory, cfg, schedule)
        if resident else (_zero_phase(schedule), 0))
    boxes = _parallel_wave_boxes(spec, cand, ("i",))
    waves = []
    scratchpad_reads = 0
    direct_x_scalar_loads = 0
    structure = []
    for origins, shapes in boxes:
        phase, scalar_reads = _gemv_wave_phase(
            spec, cand, origins, shapes, cfg, target, schedule,
            memory=(memory if resident else None), share_x=share_x)
        waves.append(phase)
        if resident:
            scratchpad_reads += scalar_reads
        direct_x_scalar_loads += scalar_reads
        structure.append((
            shapes, tuple(size for _start, size in _active_worker_groups(
                dict(shapes)["i"], *cand.factors("i")))))
    waves_tuple = tuple(waves)
    return ExtendedPlanSummary(
        memory_plan=memory, jam_plan=jam,
        execution=ExtendedExecutionSummary(
            preload=preload, full_compute=_combine_wave_phases(waves_tuple),
            compute_waves=waves_tuple),
        schedule_structure_key=(
            "gemv", jam.name, memory.fallback, tuple(structure)),
        preload_scalar_elements=preload_scalar_elements,
        scratchpad_reads=scratchpad_reads,
        avoided_direct_loads=(
            max(0, direct_x_scalar_loads - preload_scalar_elements)
            if resident else 0))


def _conv2d_output_sources(
        co: int, oh: int, ow: int) \
        -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return tap-ordered input/weight indices and the output index."""
    input_sources = []
    weight_sources = []
    tap = 0
    for ci in range(3):
        for kh in range(3):
            for kw in range(3):
                input_sources.append(
                    ci * 8 * 8 + (oh + kh) * 8 + (ow + kw))
                weight_sources.append(co * 27 + tap)
                tap += 1
    output_source = co * 6 * 6 + oh * 6 + ow
    return tuple(input_sources), tuple(weight_sources), output_source


@lru_cache(maxsize=None)
def _conv2d_whole_address_sets() -> tuple[tuple[int, ...],
                                          tuple[int, ...], tuple[int, ...]]:
    input_sources = set()
    weight_sources = set()
    output_sources = set()
    for co in range(4):
        for oh in range(6):
            for ow in range(6):
                inputs, weights, output = _conv2d_output_sources(co, oh, ow)
                input_sources.update(inputs)
                weight_sources.update(weights)
                output_sources.add(output)
    return (tuple(sorted(input_sources)), tuple(sorted(weight_sources)),
            tuple(sorted(output_sources)))


def _conv2d_memory_planner(
        _spec: KernelSpec, _cand: Candidate,
        _target: AnalyticTargetSpec) -> tuple[BufferSpec, ...]:
    input_set, weight_set, output_set = _conv2d_whole_address_sets()
    return (
        BufferSpec("input", 4, input_set, True, True),
        BufferSpec("weight", 4, weight_set, True, True),
        BufferSpec("output", 4, output_set, False, True),
    )


@lru_cache(maxsize=None)
def _conv2d_spad_accesses(
        cand: Candidate,
        origins_tuple: tuple[tuple[str, int], ...],
        shapes_tuple: tuple[tuple[str, int], ...],
        target: AnalyticTargetSpec,
        share_input_across_co: bool,
        share_weight_across_oh: bool,
        share_weight_across_ow: bool,
        input_layout: tuple[int, tuple[tuple[int, int], ...]],
        weight_layout: tuple[int, tuple[tuple[int, int], ...]]) \
        -> tuple[PackedScratchpadAccess, ...]:
    origins = dict(origins_tuple)
    shapes = dict(shapes_tuple)

    def make_reader_map(name: str) -> dict[int, int]:
        mapping = {}
        groups = _active_worker_groups(shapes[name], *cand.factors(name))
        for reader_index, (local_start, size) in enumerate(groups):
            for offset in range(size):
                mapping[origins[name] + local_start + offset] = reader_index
        return mapping

    co_readers = make_reader_map("co")
    oh_readers = make_reader_map("oh")
    ow_readers = make_reader_map("ow")
    input_base, input_slot_pairs = input_layout
    weight_base, weight_slot_pairs = weight_layout
    input_slots = dict(input_slot_pairs)
    weight_slots = dict(weight_slot_pairs)
    order = tuple(name for name in cand.order or ("co", "oh", "ow", "tap")
                  if name in ("co", "oh", "ow"))
    reservations = []
    seen_input = set()
    seen_weight = set()
    for values in itertools.product(*(
            range(origins[name], origins[name] + shapes[name])
            for name in order)):
        point = dict(zip(order, values))
        inputs, weights, _output = _conv2d_output_sources(
            point["co"], point["oh"], point["ow"])
        input_key = ((co_readers[point["co"]]
                      if share_input_across_co else point["co"]),
                     point["oh"], point["ow"])
        if input_key not in seen_input:
            seen_input.add(input_key)
            for group in coalesced_element_groups(inputs, target.vector_width):
                logical = tuple(input_base + input_slots[source]
                                for source in group)
                reservations.append(PackedScratchpadAccess(
                    "input", logical,
                    (("operation", len(reservations)),), 0, "input"))
        weight_key = (
            point["co"],
            (oh_readers[point["oh"]]
             if share_weight_across_oh else point["oh"]),
            (ow_readers[point["ow"]]
             if share_weight_across_ow else point["ow"]),
        )
        if weight_key not in seen_weight:
            seen_weight.add(weight_key)
            for group in coalesced_element_groups(weights, target.vector_width):
                logical = tuple(weight_base + weight_slots[source]
                                for source in group)
                reservations.append(PackedScratchpadAccess(
                    "weight", logical,
                    (("operation", len(reservations)),), 0, "weight"))
    return tuple(reservations)


@lru_cache(maxsize=None)
def _conv2d_wave_phase(
        spec: KernelSpec, cand: Candidate,
        origins_tuple: tuple[tuple[str, int], ...],
        shapes_tuple: tuple[tuple[str, int], ...], cfg: Config | None,
        target: AnalyticTargetSpec, schedule: bool,
        share_input_across_co: bool,
        share_weight_across_oh: bool,
        share_weight_across_ow: bool,
        spad_accesses: tuple[PackedScratchpadAccess, ...] | None = None) \
        -> tuple[PhaseSummary, int]:
    origins = dict(origins_tuple)
    shapes = dict(shapes_tuple)
    independent_order = tuple(
        name for name in candidate_order(spec, cand)
        if name in ("co", "oh", "ow"))
    points = tuple(tuple(zip(independent_order, values))
                   for values in itertools.product(*(
                       range(origins[name], origins[name] + shapes[name])
                       for name in independent_order)))
    co_groups = _active_worker_groups(
        shapes["co"], *cand.factors("co"))
    oh_groups = _active_worker_groups(
        shapes["oh"], *cand.factors("oh"))
    ow_groups = _active_worker_groups(
        shapes["ow"], *cand.factors("ow"))
    worker_count = math.prod(len(_active_worker_groups(
        shapes[name], *cand.factors(name)))
        for name in ("co", "oh", "ow"))
    output_count = math.prod(shapes.values())
    input_reader_count = len(co_groups) if share_input_across_co \
        else shapes["co"]
    input_patch_count = input_reader_count * shapes["oh"] * shapes["ow"]
    resident = spad_accesses is not None
    input_load_ops = input_patch_count * (9 if resident else 27)
    weight_oh_readers = len(oh_groups) if share_weight_across_oh \
        else shapes["oh"]
    weight_ow_readers = len(ow_groups) if share_weight_across_ow \
        else shapes["ow"]
    weight_reader_count = (
        shapes["co"] * weight_oh_readers * weight_ow_readers)
    weight_load_ops = weight_reader_count * 7
    grouped_output_level = (
        "ow" if independent_order[-1] == "ow" else None)
    output_groups = _coordinate_groups(
        independent_order, origins, shapes, cand,
        grouped_output_level, target.vector_width)

    def reader_map(name: str, groups: tuple[tuple[int, int], ...]) -> dict[int, int]:
        mapping = {}
        for reader_index, (local_start, size) in enumerate(groups):
            for offset in range(size):
                mapping[origins[name] + local_start + offset] = reader_index
        return mapping

    co_reader_by_value = reader_map("co", co_groups)
    oh_reader_by_value = reader_map("oh", oh_groups)
    ow_reader_by_value = reader_map("ow", ow_groups)
    phase = PhaseSummary(
        A=53 * output_count + 2 * worker_count,
        recurring_loads=input_load_ops + weight_load_ops + worker_count,
        invariant_loads=1,
        stores=len(output_groups) + worker_count,
        CP=8, spad_read_accesses=(spad_accesses or ()),
        control_A=2 * worker_count, control_loads=worker_count,
        control_stores=worker_count)
    scalar_resident_reads = (
        input_patch_count * 27 + weight_reader_count * 27)
    if not schedule:
        return phase, scalar_resident_reads
    if cfg is None:
        raise ValueError("scheduled Conv2d phase requires a resource config")

    dag = Dag()
    region = dag.region("conv2d")
    region.load(kind="params" + INV)
    _worker_control(region, worker_count)
    input_handle_cache = {}
    weight_handle_cache = {}
    output_handles = {}
    for point_tuple in points:
        point = dict(point_tuple)
        inputs, weights, output_source = _conv2d_output_sources(
            point["co"], point["oh"], point["ow"])
        input_key = ((co_reader_by_value[point["co"]]
                      if share_input_across_co else point["co"]),
                     point["oh"], point["ow"])
        input_handles = input_handle_cache.get(input_key)
        if input_handles is None:
            by_source = {}
            if resident:
                for group in coalesced_element_groups(
                        inputs, target.vector_width):
                    handle = region.load(kind="spad_input")
                    for source in group:
                        by_source[source] = handle
            else:
                for source in inputs:
                    by_source[source] = region.load(kind="input_scalar")
            input_handles = tuple(by_source[source] for source in inputs)
            input_handle_cache[input_key] = input_handles
        weight_key = (
            point["co"],
            (oh_reader_by_value[point["oh"]]
             if share_weight_across_oh else point["oh"]),
            (ow_reader_by_value[point["ow"]]
             if share_weight_across_ow else point["ow"]),
        )
        weight_by_source = weight_handle_cache.get(weight_key)
        if weight_by_source is None:
            weight_by_source = {}
            for group in coalesced_element_groups(
                    weights, target.vector_width):
                handle = region.load(
                    kind="spad_weight" if resident else "weight_vec")
                for source in group:
                    weight_by_source[source] = handle
            weight_handle_cache[weight_key] = weight_by_source
        products = [region.arith(
            input_handles[index], weight_by_source[weights[index]], kind="mul")
            for index in range(27)]
        output_handles[output_source] = region.balanced_reduction(
            products, kind="reduce")
    for group in output_groups:
        handles = []
        for point_tuple in group:
            point = dict(point_tuple)
            _inputs, _weights, output_source = _conv2d_output_sources(
                point["co"], point["oh"], point["ow"])
            handles.append(output_handles[output_source])
        region.store(*handles, output=True, kind="output_vec")
    return (_phase_with_optional_schedule(
        phase, dag, cfg, "conv2d_resident" if resident else "conv2d_direct",
        True), scalar_resident_reads)


def _conv2d_extended_plan_builder(
        spec: KernelSpec, cand: Candidate, cfg: Config,
        target: AnalyticTargetSpec, schedule: bool) -> ExtendedPlanSummary:
    memory = derive_memory_plan(spec, cand, target)
    jam = derive_jam_plan(spec, cand)
    share_input = any(
        edge.outer == "co" and "input" in edge.shared_operands
        for edge in jam.edges)
    share_weight_oh = any(
        edge.outer == "oh" and "weight" in edge.shared_operands
        for edge in jam.edges)
    share_weight_ow = any(
        edge.outer == "ow" and "weight" in edge.shared_operands
        for edge in jam.edges)
    resident = not memory.fallback
    phase_cfg = cfg if schedule else None
    preload, preload_scalar_elements = (
        _resident_preload_phase(memory, phase_cfg, schedule)
        if resident else (_zero_phase(schedule), 0))
    input_buffer = next(buffer for buffer in memory.buffers
                        if buffer.name == "input")
    weight_buffer = next(buffer for buffer in memory.buffers
                         if buffer.name == "weight")
    input_layout = (input_buffer.base_element, input_buffer.source_to_slot)
    weight_layout = (weight_buffer.base_element, weight_buffer.source_to_slot)
    boxes = _parallel_wave_boxes(spec, cand, ("co", "oh", "ow"))
    waves = []
    structure = []
    scratchpad_reads = 0
    direct_resident_scalar_loads = 0
    for origins, shapes in boxes:
        accesses = (_conv2d_spad_accesses(
            cand, origins, shapes, target,
            share_input, share_weight_oh, share_weight_ow,
            input_layout, weight_layout) if resident else None)
        phase, scalar_reads = _conv2d_wave_phase(
            spec, cand, origins, shapes, phase_cfg, target, schedule,
            share_input, share_weight_oh, share_weight_ow,
            spad_accesses=accesses)
        waves.append(phase)
        if resident:
            scratchpad_reads += scalar_reads
        direct_resident_scalar_loads += scalar_reads
        structure.append((
            shapes, tuple(
                (name, tuple(size for _start, size in _active_worker_groups(
                    dict(shapes)[name], *cand.factors(name))))
                for name in ("co", "oh", "ow"))))
    waves_tuple = tuple(waves)
    return ExtendedPlanSummary(
        memory_plan=memory, jam_plan=jam,
        execution=ExtendedExecutionSummary(
            preload=preload, full_compute=_combine_wave_phases(waves_tuple),
            compute_waves=waves_tuple),
        schedule_structure_key=(
            "conv2d", candidate_order(spec, cand), jam.name, memory.fallback,
            share_input,
            share_weight_oh, share_weight_ow,
            tuple(structure)),
        preload_scalar_elements=preload_scalar_elements,
        scratchpad_reads=scratchpad_reads,
        avoided_direct_loads=(
            max(0, direct_resident_scalar_loads - preload_scalar_elements)
            if resident else 0))


def _batchnorm_extended_chunk(cand: Candidate) -> Dag:
    spec = KERNELS["batchnorm"]
    shapes = tuple((name, min(spec.level(name).trip,
                              math.prod(cand.factors(name))))
                   for name in ("c", "h", "w"))
    phase = _batchnorm_wave_phase(
        spec, cand, (("c", 0), ("h", 0), ("w", 0)), shapes,
        parse_config("6x6"), AnalyticTargetSpec(), False)
    return _emit_counted_region(
        "batchnorm", phase.A, phase.recurring_loads,
        phase.invariant_loads, phase.stores,
        ("L", "P", "P", "P", "P", "P", "P", "S"))


def _gemv_extended_chunk(cand: Candidate) -> Dag:
    spec = KERNELS["gemv"]
    extent = min(spec.level("i").trip, math.prod(cand.factors("i")))
    column_count = spec.level("j").trip
    reduction_depth = math.ceil(math.log2(column_count)) \
        if column_count > 1 else 0
    jam = derive_jam_plan(spec, cand)
    share_x = any("x" in edge.shared_operands for edge in jam.edges)
    phase, _scalar_reads = _gemv_wave_phase(
        spec, cand, (("i", 0),), (("i", extent),), parse_config("6x6"),
        AnalyticTargetSpec(), False, share_x=share_x)
    return _emit_counted_region(
        "gemv", phase.A, phase.recurring_loads,
        phase.invariant_loads, phase.stores,
        ("L",) + ("P",) * (reduction_depth + 3) + ("S",))


def _conv2d_extended_chunk(cand: Candidate) -> Dag:
    spec = KERNELS["conv2d"]
    shapes = tuple((name, min(spec.level(name).trip,
                              math.prod(cand.factors(name))))
                   for name in ("co", "oh", "ow"))
    jam = derive_jam_plan(spec, cand)
    phase, _scalar_reads = _conv2d_wave_phase(
        spec, cand, (("co", 0), ("oh", 0), ("ow", 0)), shapes,
        parse_config("6x6"), AnalyticTargetSpec(), False,
        any(edge.outer == "co" and "input" in edge.shared_operands
            for edge in jam.edges),
        any(edge.outer == "oh" and "weight" in edge.shared_operands
            for edge in jam.edges),
        any(edge.outer == "ow" and "weight" in edge.shared_operands
            for edge in jam.edges))
    return _emit_counted_region(
        "conv2d", phase.A, phase.recurring_loads,
        phase.invariant_loads, phase.stores,
        ("L",) + ("P",) * 6 + ("S",))


KERNELS: dict[str, KernelSpec] = {}


def _register():
    KERNELS["axpy"] = KernelSpec(
        name="axpy",
        levels=(Level("i", 256, "parallel"),),
        build_chunk=_axpy_chunk,
        coalesce_note=(
            "input_x/input_y/output_y are contiguous over i. Two axes both favor "
            "LOOM_UNROLL over LOOM_PARALLEL at a fixed product: (1) coalescing -- a "
            "worker's U adjacent accesses fuse into ceil(U/V) vector ops while "
            "parallel strides across workers (bounded by V=4, gone once U>=V); (2) "
            "control amortization -- the iterator is charged once per worker, so "
            "fewer workers (more unroll) means fewer i-loads/adds/stores (keeps "
            "paying past U=V). So unroll strictly beats parallel at fixed product."),
    )
    KERNELS["vecsum"] = KernelSpec(
        name="vecsum",
        levels=(Level("i", 256, "reduction"),),
        build_chunk=_vecsum_chunk,
        coalesce_note=(
            "A is contiguous over i and the reduction is fully consumed in one "
            "wave AND tree-reduced (a spatial tree), so it carries no per-element "
            "and no per-worker iterator -- control is a fixed residual for any "
            "split. A also coalesces to ~trip/V vector loads regardless of the p/u "
            "split. Both the control and coalescing axes are inert -> vecsum is "
            "P/U-symmetric, and CP-bound on the log-depth merge tree."),
    )
    M, N = 32, 48
    KERNELS["gemv"] = KernelSpec(
        name="gemv",
        levels=(Level("i", M, "parallel"), Level("j", N, "reduction")),
        build_chunk=_gemv_extended_chunk,
        coalesce_note=(
            "A[i][j] and x[j] are contiguous over j (a fully-consumed reduction, "
            "tree-reduced), so they coalesce identically and the j-loop carries no "
            "control -> the dot-product path is P/U-symmetric. On the row level i, "
            "LOOM_UNROLL(i) beats LOOM_PARALLEL(i) two ways: it coalesces the "
            "contiguous y[i]/output_y[i] accesses (parallel strides) and it "
            "amortizes the row iterator (charged once per worker). The A-load term "
            "is split-symmetric and large, so the i-level edge is modest but real."),
        order_spec=OrderSpec((("i", "j"),)),
        jam_plans=(JamPlanSpec(
            "i-j-share-x", (JamRule("i", "j", ("x",)),)),),
        memory_planner=_gemv_memory_planner,
        extended_plan_builder=_gemv_extended_plan_builder,
    )
    _n_out, K = _conv2d_dims()
    KERNELS["conv2d"] = KernelSpec(
        name="conv2d",
        levels=(Level("co", 4, "parallel"), Level("oh", 6, "parallel"),
                Level("ow", 6, "parallel"), Level("tap", K, "reduction")),
        build_chunk=_conv2d_extended_chunk,
        coalesce_note=(
            "co/oh/ow are explicit independent levels and tap is the pinned, "
            "fully-consumed reduction. Declared legal orders recompute whether ow "
            "is the innermost independent output dimension. Exact input halos and "
            "co-specific weights are resident_shared; output remains direct."),
        order_spec=OrderSpec((
            ("co", "oh", "ow", "tap"),
            ("co", "ow", "oh", "tap"),
            ("oh", "co", "ow", "tap"),
            ("oh", "ow", "co", "tap"),
            ("ow", "co", "oh", "tap"),
            ("ow", "oh", "co", "tap"),
        )),
        jam_plans=(
            JamPlanSpec(
                "share-input",
                (JamRule("co", "tap", ("input",)),)),
            JamPlanSpec(
                "share-weight-oh",
                (JamRule("oh", "tap", ("weight",)),)),
            JamPlanSpec(
                "share-weight-ow",
                (JamRule("ow", "tap", ("weight",)),)),
            JamPlanSpec(
                "share-input-weight-oh",
                (JamRule("co", "tap", ("input",)),
                 JamRule("oh", "tap", ("weight",)))),
            JamPlanSpec(
                "share-input-weight-ow",
                (JamRule("co", "tap", ("input",)),
                 JamRule("ow", "tap", ("weight",)))),
            JamPlanSpec(
                "share-weight-oh-ow",
                (JamRule("oh", "tap", ("weight",)),
                 JamRule("ow", "tap", ("weight",)))),
            JamPlanSpec(
                "share-all",
                (JamRule("co", "tap", ("input",)),
                 JamRule("oh", "tap", ("weight",)),
                 JamRule("ow", "tap", ("weight",)))),
        ),
        memory_planner=_conv2d_memory_planner,
        extended_plan_builder=_conv2d_extended_plan_builder,
    )
    KERNELS["tridiag_solve"] = KernelSpec(
        name="tridiag_solve",
        levels=(Level("i", 64, "sequential"),),
        build_chunk=_tridiag_chunk,
        coalesce_note=(
            "The forward sweep carries a NON-associative recurrence (division "
            "chain): LOOM_PARALLEL is illegal (p forced to 1) and the serial CP "
            "dominates. Input streams coalesce but it does not matter -> the "
            "kernel stays critical-path bound with no P-vs-U distinction."),
    )
    KERNELS["batchnorm"] = KernelSpec(
        name="batchnorm",
        levels=(Level("c", 4, "parallel"), Level("h", 8, "parallel"),
                Level("w", 8, "parallel")),
        build_chunk=_batchnorm_extended_chunk,
        coalesce_note=(
            "input/output are contiguous over the innermost w. LOOM_UNROLL(w) "
            "coalesces a worker's adjacent w-accesses (ceil(U_w/V) vector ops) "
            "while LOOM_PARALLEL(w) strides -> unroll-on-w beats parallel-on-w on "
            "the load/store term (while U_w < V). c/h are strided for input and do "
            "not coalesce, but LOOM_UNROLL on ANY level still amortizes the "
            "iterator (charged once per worker over the c*h*w worker set), so "
            "unroll cuts control ops even where coalescing cannot. Compute-bound, "
            "so those load/store savings show as lane headroom, not a lower floor. "
            "mean/variance/gamma/beta are per-channel invariants (once per "
            "exposed channel)."),
        order_spec=OrderSpec((
            ("c", "h", "w"),
            ("c", "w", "h"),
        )),
        memory_planner=_batchnorm_memory_planner,
        extended_plan_builder=_batchnorm_extended_plan_builder,
    )
    KERNELS["bisection_step"] = KernelSpec(
        name="bisection_step",
        levels=(Level("i", 64, "parallel"),),
        build_chunk=_bisection_chunk,
        coalesce_note=(
            "All six arrays (input_a/b/fa/fc, output_a/b) are contiguous over i. "
            "This is axpy-shaped: LOOM_UNROLL(i) beats LOOM_PARALLEL(i) two ways "
            "-- it coalesces the 4 input loads and 2 output stores into vector "
            "ops (bounded by V=4) and it amortizes the iterator (charged once per "
            "worker, keeps paying past U=V). The if/else is counted taken-arm-only "
            "(no predication credit) and the compute is a global pool, so the "
            "branch does not separate P from U. Load-heavy shape (4 input streams "
            "to 2 output streams), but compute-bound after coalescing + control "
            "amortization."),
    )
    KERNELS["autocorrelation"] = KernelSpec(
        name="autocorrelation",
        levels=(Level("lag", 32, "parallel"), Level("i", 128, "reduction")),
        build_chunk=_autocorr_chunk,
        coalesce_note=(
            "gemv-shaped: outer PARALLEL lag, inner REDUCTION i (associative float "
            "sum, tree-reduced -> no i-loop control -> the dot-product path is "
            "P/U-symmetric). x[i] (the un-shifted prefix) is the same data for "
            "every lag -> modeled invariant (loaded once per chunk). x[i+lag] "
            "shifts with lag -> recurring, but contiguous over i so it coalesces. "
            "On the lag level, LOOM_UNROLL(lag) beats LOOM_PARALLEL(lag): it "
            "coalesces the contiguous output[lag] stores and amortizes the lag "
            "iterator. Inner length modeled at max x_size=128 (true length "
            "x_size-lag runs down to 97 at lag=31 -> conservative ~14% over-count, "
            "4096 vs true 3600); cross-lag reuse of x otherwise not modeled (conv2d "
            "halo convention)."),
    )
    KERNELS["bit_reverse"] = KernelSpec(
        name="bit_reverse",
        levels=(Level("i", 256, "parallel"), Level("bit", 32, "sequential")),
        build_chunk=_bit_reverse_chunk,
        coalesce_note=(
            "outer PARALLEL i (independent 32-bit words), inner SEQUENTIAL bit "
            "loop carrying result/value through a non-associative shift/merge "
            "recurrence. The carried scalars are threaded as dataflow (no per-bit "
            "round-trip, unlike the conservative ASAP eval) and the bit iterator "
            "is charged per iteration (it stays on CP and cannot be amortized). "
            "The 4 bitops/bit form a large global arithmetic pool -> COMPUTE-bound "
            "(contrast the store-bound ASAP result, which charges per-bit result/"
            "value stores). LOOM_UNROLL(i) coalesces input_data/output_reversed "
            "and amortizes the OUTER i iterator, but the inner sequential loop "
            "dominates, so the P-vs-U edge is small."),
    )
    KERNELS["binary_search"] = KernelSpec(
        name="binary_search",
        levels=(Level("t", 5, "parallel"), Level("probe", 4, "sequential")),
        build_chunk=_binsearch_chunk,
        coalesce_note=(
            "outer PARALLEL t (independent target searches), inner SEQUENTIAL "
            "while with DATA-DEPENDENT termination (worst-case ceil(log2(N+1))=4 "
            "probes). The left/right recurrence is threaded as dataflow and its "
            "termination compare sits on CP per probe; input_sorted[mid] is a "
            "non-affine (data-dependent) scalar load that cannot coalesce. This is "
            "a COUNTEREXAMPLE like tridiag: the per-target serial recurrence and a "
            "tiny problem (M=5 targets) leave it CP/latency-bound, so no P-vs-U "
            "split helps. The source LOOM_NO_PARALLEL/LOOM_NO_UNROLL reflects "
            "control divergence, which this DSE does not model."),
    )
    KERNELS["bitonic_stage"] = KernelSpec(
        name="bitonic_stage",
        levels=(Level("i", 8, "parallel"),),
        build_chunk=_bitonic_stage_chunk,
        coalesce_note=(
            "i is parallel for the documented N=8, stage=1, pass=0 fixture: "
            "active lanes touch disjoint compare pairs. The branch mix is one "
            "active lane per pair and one committing swap lane per four i lanes. "
            "Conditional in-place pair accesses remain scalar because strict "
            "compare-to-body gates and swap aliasing do not form a plain "
            "contiguous vector stream. LOOM_UNROLL therefore helps only through "
            "outer-iterator control amortization; the 11-cycle gated CP dominates."),
    )
    KERNELS["bitonic_stage-modified"] = KernelSpec(
        name="bitonic_stage-modified",
        levels=(Level("i", 8, "sequential"),),
        build_chunk=_bitonic_stage_modified_chunk,
        coalesce_note=(
            "The outer i loop is sequential: its loop-counter carry and the "
            "in-place N/2..N-1 read-modify-write chain cross iterations. Parallel "
            "factors are illegal and unroll cannot flatten the recurrence, so "
            "equivalent unroll labels use the canonical P1U1 representative for "
            "the fully consumed serial DAG."),
    )
    KERNELS["bitonic_stage-tweak"] = KernelSpec(
        name="bitonic_stage-tweak",
        levels=(Level("i", 8, "sequential"),),
        build_chunk=_bitonic_stage_tweak_chunk,
        coalesce_note=(
            "The unconditional inplace[i]-=1 and active-lane inplace[i]++ create "
            "same-slot and partner RAW chains across the in-place stage. Parallel "
            "factors are therefore illegal and unroll cannot flatten the memory "
            "recurrence; equivalent unroll labels use the canonical P1U1 "
            "representative for the same 17-cycle serial DAG."),
    )
    KERNELS["clz"] = KernelSpec(
        name="clz",
        levels=(Level("i", 256, "parallel"),),
        build_chunk=_clz_chunk,
        coalesce_note=(
            "outer i is parallel; each lane has a private data-dependent while "
            "recurrence whose trip count is the concrete main.cpp leading-zero "
            "count. Contiguous i-unroll coalesces boundary input/output traffic "
            "and amortizes outer control, while the longest K=31 lane keeps CP "
            "at 163 once exposed."),
    )
    KERNELS["col2im"] = KernelSpec(
        name="col2im",
        levels=(Level("c", 3, "parallel"), Level("kh", 3, "reduction")),
        build_chunk=_col2im_chunk,
        coalesce_note=(
            "channels are independent. For each exposed channel, overlapping "
            "kh/kw contributions are consumed as output-centric associative "
            "reduction buckets, so kh is fully consumed and its P/U labels are "
            "equivalent. Channel slices are separated by H*W and do not coalesce "
            "across c. The eval's per-iteration induction work is removed, then "
            "one residual c iterator is charged per active worker, so c-unroll "
            "amortizes control while c-parallel retains one iterator per worker."),
    )
    KERNELS["crc32"] = KernelSpec(
        name="crc32",
        levels=(Level("i", 256, "sequential"),),
        build_chunk=_crc32_chunk,
        coalesce_note=(
            "the optimized lookup-table DAG carries crc across all byte updates. "
            "Each next table address depends on the preceding crc result, so "
            "parallel factors are illegal and unroll cannot flatten the trace; "
            "equivalent labels use the canonical P1U1 representative."),
    )
    KERNELS["edge_update"] = KernelSpec(
        name="edge_update",
        levels=(Level("kernel", 1, "sequential"),),
        build_chunk=_edge_update_chunk,
        coalesce_note=(
            "the source has no Loom parallel/unroll pragma. The copy, bounds "
            "check, data-dependent CSR search, and matched overwrite are modeled "
            "as the concrete serial kernel trace; there is no legal P/U level to "
            "sweep."),
    )
    KERNELS["fft_butterfly"] = KernelSpec(
        name="fft_butterfly",
        levels=(Level("copy_i", 16, "parallel"),),
        build_chunk=_fft_butterfly_chunk,
        coalesce_note=(
            "the annotated copy loop is parallel and its two input/output "
            "streams are contiguous, so copy_i-unroll coalesces them and "
            "amortizes copy control. Copy waves are ordered before four fixed "
            "once-only FFT stage regions. Those stages are barrier-ordered by "
            "in-place array RAW hazards; within each stage k blocks are "
            "independent, but j remains sequential because both its iterator "
            "and the generated twiddle w<-w*wm are carried recurrences. Thus a "
            "copy candidate's waves do not repeat the stage work."),
        repeat_waves=False,
    )
    KERNELS["gauss_seidel_step"] = KernelSpec(
        name="gauss_seidel_step",
        levels=(Level("i", 32, "sequential"),),
        build_chunk=_gauss_seidel_step_chunk,
        coalesce_note=(
            "the outer i loop is a true in-place Gauss-Seidel recurrence: row i "
            "reads output_x values written by earlier rows, so parallel factors "
            "are illegal and unroll labels cannot flatten the sweep. The lower "
            "and upper j sums are fully consumed associative reductions with no "
            "j-loop control. The read-only input_x vector is loaded once and held; "
            "independent contiguous input_A row segments and already-ready "
            "lower-triangle output_x prefixes coalesce. The newest output_x[i-1] "
            "read and sequential output_x stores stay scalar to preserve the row "
            "recurrence. Equivalent i-unroll labels use the canonical P1U1 "
            "representative."),
        selection_mode="latency_fallback",
    )
    KERNELS["gather"] = KernelSpec(
        name="gather",
        levels=(Level("i", 1024, "parallel"),),
        build_chunk=_gather_chunk,
        coalesce_note=(
            "i is parallel and every concrete fixture lane takes the valid arm. "
            "indices[i] and dst[i] are contiguous, so i-unroll coalesces those "
            "streams and amortizes the iterator. src[indices[i]] remains an "
            "indirect scalar load: the loaded indices are not an affine address "
            "sequence the vector interface may coalesce, although the read-only "
            "loads are independent and may occupy separate load lanes. The DSE "
            "uses the spec-wide V=4 convention despite the uint32_t element type."),
    )
    KERNELS["hist_bin"] = KernelSpec(
        name="hist_bin",
        levels=(Level("zero_i", 10, "parallel"),),
        build_chunk=_hist_bin_chunk,
        coalesce_note=(
            "the annotated zero_i loop is parallel; its contiguous output stores "
            "coalesce within each unrolled worker, and its phase-local waves "
            "include the partial tail. All zero-fill waves are ordered before one "
            "fixed 1024-input count region because the later updates read those "
            "zero identities. All inputs take the valid guard path; the bin clamp "
            "compare executes but its assignment arm is untaken. "
            "Concrete per-bin fan-ins are 110,110,104,100,100,100,100,100,100,100 "
            "and form associative output-centric trees. Contiguous input loads "
            "coalesce, while data-dependent output scatter loads/stores and "
            "memory-backed bin scalars remain scalar. The dominant scatter phase "
            "has no independent tiled P/U level and executes exactly once for "
            "every zero_i candidate."),
        repeat_waves=False,
    )
    KERNELS["interpolate_linear"] = KernelSpec(
        name="interpolate_linear",
        levels=(Level("q", 64, "parallel"),),
        build_chunk=_interpolate_linear_chunk,
        coalesce_note=(
            "q lanes are independent, but each lane keeps its private sequential "
            "data-dependent k search and selected i state. The helper uses the "
            "concrete main.cpp trace (1024 probes: 63 hits and one final-check "
            "no-hit lane) and the same deterministic wave-average convention as "
            "clz. q-unroll coalesces only contiguous input_xq[q] / output_yq[q] "
            "boundary traffic and amortizes q control. Search input_x[k] loads "
            "and tail input_x/input_y[i] loads remain recurring scalar accesses: "
            "conditional termination and data-dependent indices prevent vector "
            "coalescing, and no cross-query cache/broadcast reuse is assumed."),
        selection_mode="latency_fallback",
    )


_register()


# ---------------------------------------------------------------------------
# Candidate = a per-level (parallel, unroll) assignment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    # tuple aligned with spec.levels: (level_name, parallel, unroll)
    split: tuple[tuple[str, int, int], ...]
    order: tuple[str, ...] = ()
    jam_plan: str = "none"

    def factors(self, name: str) -> tuple[int, int]:
        for n, p, u in self.split:
            if n == name:
                return p, u
        raise KeyError(name)

    def signature(self) -> str:
        signature = " ".join(f"{n}:P{p}U{u}" for n, p, u in self.split)
        if self.order:
            signature += " order=" + ">".join(self.order)
        if self.jam_plan != "none":
            signature += f" jam={self.jam_plan}"
        return signature


def _powers_of_two_through(limit: int) -> list[int]:
    values = []
    factor = 1
    while factor <= limit:
        values.append(factor)
        factor *= 2
    return values


def _source_order(spec: KernelSpec) -> tuple[str, ...]:
    return tuple(level.name for level in spec.levels)


def _validated_orders(spec: KernelSpec) -> tuple[tuple[str, ...], ...]:
    """Validate declared orders and return the source-order default."""
    source = _source_order(spec)
    if spec.order_spec is None:
        return (source,)
    orders = spec.order_spec.legal_orders
    if orders[0] != source:
        raise ValueError(
            f"{spec.name}: first legal order {orders[0]} must be source order "
            f"{source}")
    expected = set(source)
    for order in orders:
        if len(order) != len(source) or set(order) != expected:
            raise ValueError(
                f"{spec.name}: legal order {order} must be a permutation of "
                f"{source}")
    return orders


def candidate_order(spec: KernelSpec, cand: Candidate) -> tuple[str, ...]:
    """Resolve and validate a candidate order; empty means source order."""
    orders = _validated_orders(spec)
    order = cand.order or orders[0]
    if order not in orders:
        raise ValueError(
            f"{spec.name}: undeclared loop order {order}; legal orders are "
            f"{orders}")
    return order


def _validated_jam_plans(spec: KernelSpec) -> tuple[JamPlanSpec, ...]:
    names = {level.name for level in spec.levels}
    plans = (JamPlanSpec("none"),) + spec.jam_plans
    plan_names = [plan.name for plan in plans]
    if len(set(plan_names)) != len(plan_names):
        raise ValueError(f"{spec.name}: duplicate jam plan names {plan_names}")
    for plan in plans:
        for rule in plan.edges:
            if rule.outer not in names or rule.inner not in names:
                raise ValueError(
                    f"{spec.name}: jam rule {rule.outer}->{rule.inner} names an "
                    "unknown level")
            if not spec.level(rule.outer).tiled():
                raise ValueError(
                    f"{spec.name}: jam outer level {rule.outer} must be parallel")
        if plan.name == "none" and plan.edges:
            raise ValueError(
                f"{spec.name}: reserved jam plan none must be empty")
    return plans


def derive_jam_plan(spec: KernelSpec, cand: Candidate) -> JamPlan:
    """Resolve and validate one explicit complete jam plan."""
    order = candidate_order(spec, cand)
    plans = {plan.name: plan for plan in _validated_jam_plans(spec)}
    if cand.jam_plan not in plans:
        raise ValueError(
            f"{spec.name}: undeclared jam plan {cand.jam_plan!r}; legal plans "
            f"are {tuple(plans)}")
    plan = plans[cand.jam_plan]
    positions = {name: index for index, name in enumerate(order)}
    for edge in plan.edges:
        _parallel, unroll = cand.factors(edge.outer)
        if unroll <= 1:
            raise ValueError(
                f"{spec.name}: jam plan {plan.name} requires "
                f"{edge.outer} unroll > 1")
        if positions[edge.outer] >= positions[edge.inner]:
            raise ValueError(
                f"{spec.name}: jam plan {plan.name} requires {edge.inner} "
                f"beneath {edge.outer} in order {order}")
    return JamPlan(plan.name, order, plan.edges)


def _uses_extended_candidate_space(spec: KernelSpec) -> bool:
    return (spec.order_spec is not None or bool(spec.jam_plans)
            or spec.memory_planner is not None
            or spec.extended_plan_builder is not None)


def _legacy_spec_view(spec: KernelSpec) -> KernelSpec:
    """Drop extended metadata while retaining the exact legacy search inputs."""
    return KernelSpec(
        name=spec.name, levels=spec.levels, build_chunk=spec.build_chunk,
        coalesce_note=spec.coalesce_note, default_config=spec.default_config,
        repeat_waves=spec.repeat_waves, selection_mode=spec.selection_mode)


def _enumerate_extended_candidates(
        spec: KernelSpec, max_parallel: int | None,
        max_unroll: int | None, exposure_cap: int | None) -> list[Candidate]:
    orders = _validated_orders(spec)
    jam_plans = _validated_jam_plans(spec)
    base_candidates = enumerate_candidates(
        _legacy_spec_view(spec), max_parallel, max_unroll, exposure_cap)
    if orders == (_source_order(spec),) and jam_plans == (JamPlanSpec("none"),):
        return base_candidates

    out: list[Candidate] = []
    source = _source_order(spec)
    for order in orders:
        stored_order = () if order == source else order
        for base in base_candidates:
            for plan in jam_plans:
                cand = Candidate(base.split, stored_order, plan.name)
                try:
                    derive_jam_plan(spec, cand)
                except ValueError:
                    continue
                out.append(cand)
    return out


def enumerate_candidates(spec: KernelSpec, max_parallel: int | None = None,
                         max_unroll: int | None = None,
                         exposure_cap: int | None = None) -> list[Candidate]:
    """Cartesian product of per-level (p,u) choices, honoring legality:
    sequential -> p=1; p*u <= trip per tiled level. By default every power-of-two
    factor through the concrete trip count is considered. Optional caps are
    explicit diagnostic restrictions, not part of the spec-defined search.

    Reduction and sequential levels are fully consumed inside every chunk, so
    their factor labels cannot change exposure or the built DAG. Canonicalize
    those equivalent aliases to P1U1 instead of repeatedly evaluating them.
    """
    if _uses_extended_candidate_space(spec):
        return _enumerate_extended_candidates(
            spec, max_parallel, max_unroll, exposure_cap)

    per_level_choices = []
    for lv in spec.levels:
        if not lv.tiled():
            per_level_choices.append([(lv.name, 1, 1)])
            continue

        pmax = lv.trip if max_parallel is None else min(max_parallel, lv.trip)
        umax = lv.trip if max_unroll is None else min(max_unroll, lv.trip)
        choices = []
        for p in _powers_of_two_through(pmax):
            for u in _powers_of_two_through(umax):
                if p * u <= lv.trip:
                    choices.append((lv.name, p, u))
        per_level_choices.append(choices)

    out: list[Candidate] = []

    def rec(idx: int, acc: list):
        if idx == len(per_level_choices):
            cand = Candidate(tuple(acc))
            tiled_exp = 1
            for lv in spec.levels:
                p, u = cand.factors(lv.name)
                if lv.tiled():
                    tiled_exp *= p * u
            if exposure_cap is None or tiled_exp <= exposure_cap:
                out.append(cand)
            return
        for choice in per_level_choices[idx]:
            acc.append(choice)
            rec(idx + 1, acc)
            acc.pop()

    rec(0, [])
    return out


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return _ceil_div(value, alignment) * alignment


def _normalized_whole_buffer_specs(
        buffer_specs: tuple[BufferSpec, ...]) -> tuple[BufferSpec, ...]:
    names = [buffer.name for buffer in buffer_specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate buffer names in memory plan: {names}")
    normalized = []
    for buffer in buffer_specs:
        elements = tuple(sorted(set(buffer.elements)))
        if any(element < 0 for element in elements):
            raise ValueError(
                f"buffer {buffer.name} contains a negative source element index")
        normalized.append(BufferSpec(
            name=buffer.name, element_bytes=buffer.element_bytes,
            elements=elements, reuse_bearing=buffer.reuse_bearing,
            worker_invariant=buffer.worker_invariant,
            replication_factor=buffer.replication_factor))
    return tuple(normalized)


def _layout_whole_memory_plan(
        target: AnalyticTargetSpec,
        raw_buffers: tuple[BufferSpec, ...]) -> MemoryPlan:
    """Lay out one whole-kernel resident set or select direct fallback."""
    buffers = _normalized_whole_buffer_specs(raw_buffers)
    cursor_bytes = 0
    proposed = []
    for buffer in buffers:
        placement = buffer.placement
        bases = []
        allocation_start = cursor_bytes
        if placement != "direct" and buffer.elements:
            alignment_bytes = target.alignment_elements * buffer.element_bytes
            for _replica in range(buffer.replication_factor):
                cursor_bytes = _align_up(cursor_bytes, alignment_bytes)
                bases.append(cursor_bytes // buffer.element_bytes)
                cursor_bytes += len(buffer.elements) * buffer.element_bytes
        proposed.append((buffer, placement, tuple(bases),
                         cursor_bytes - allocation_start))

    fallback = cursor_bytes > target.capacity_bytes
    plans = []
    for buffer, placement, bases, bytes_used in proposed:
        resolved_placement = (
            "direct-fallback" if fallback and placement != "direct"
            else placement)
        resolved_bases = () if resolved_placement in (
            "direct", "direct-fallback") else bases
        plans.append(BufferPlan(
            name=buffer.name, placement=resolved_placement,
            base_element=(resolved_bases[0] if resolved_bases else None),
            replica_bases=resolved_bases,
            element_bytes=buffer.element_bytes,
            replication_factor=buffer.replication_factor,
            elements=buffer.elements,
            source_to_slot=tuple(
                (source, offset) for offset, source in enumerate(buffer.elements)),
            bytes_used=(0 if not resolved_bases else bytes_used)))
    return MemoryPlan(
        target=target, buffers=tuple(plans),
        capacity_bytes_used=(0 if fallback else cursor_bytes),
        proposed_capacity_bytes=cursor_bytes, fallback=fallback)


def derive_memory_plan(spec: KernelSpec, cand: Candidate,
                       target: AnalyticTargetSpec) -> MemoryPlan:
    """Invoke a whole-kernel planner and derive resident or direct fallback."""
    if spec.memory_planner is None:
        raw_buffers: tuple[BufferSpec, ...] = ()
    else:
        if not callable(spec.memory_planner):
            raise TypeError(f"{spec.name}: memory_planner must be callable")
        raw_buffers = tuple(spec.memory_planner(spec, cand, target))
    return _layout_whole_memory_plan(target, raw_buffers)


def contiguous_element_runs(elements) -> tuple[tuple[int, ...], ...]:
    """Return sorted unique source elements partitioned into contiguous runs."""
    ordered = sorted(set(elements))
    if not ordered:
        return ()
    runs = []
    current = [ordered[0]]
    for element in ordered[1:]:
        if element == current[-1] + 1:
            current.append(element)
        else:
            runs.append(tuple(current))
            current = [element]
    runs.append(tuple(current))
    return tuple(runs)


def coalesced_element_groups(elements, vector_width: int) \
        -> tuple[tuple[int, ...], ...]:
    """Split each contiguous source run into fixed-width vector operations."""
    if vector_width <= 0:
        raise ValueError("vector width must be positive")
    groups = []
    for run in contiguous_element_runs(elements):
        for start in range(0, len(run), vector_width):
            groups.append(run[start:start + vector_width])
    return tuple(groups)


def preload_logical_accesses(
        buffer: BufferPlan, target: AnalyticTargetSpec,
        replica: int = 0) \
        -> tuple[tuple[int, ...], ...]:
    """Return coalesced destination elements for one resident-buffer fill."""
    if buffer.placement in ("direct", "direct-fallback"):
        return ()
    if replica < 0 or replica >= len(buffer.replica_bases):
        raise IndexError(
            f"buffer {buffer.name} replica is out of range")
    base = buffer.replica_bases[replica]
    source_to_offset = dict(buffer.source_to_slot)
    accesses = []
    for group in coalesced_element_groups(buffer.elements, target.vector_width):
        accesses.append(tuple(base + source_to_offset[source]
                              for source in group))
    return tuple(accesses)


def order_contiguous_stream(
        spec: KernelSpec, cand: Candidate, stream: str,
        contiguous_level: str | None) -> str | None:
    """Declare a vector stream only when its contiguous level is innermost."""
    if not stream:
        raise ValueError("contiguous stream name must be non-empty")
    if contiguous_level is None:
        return None
    if contiguous_level not in {level.name for level in spec.levels}:
        raise ValueError(
            f"{spec.name}: unknown contiguous level {contiguous_level!r}")
    return (stream if candidate_order(spec, cand)[-1] == contiguous_level
            else None)

def pack_scratchpad_accesses(
        accesses, target: AnalyticTargetSpec) -> tuple[PackedScratchpadAccess, ...]:
    """Apply same-step fan-out, then pack only declared legal streams.

    Fan-out deduplicates exactly matching buffer/address/logical-step/replica
    requests. A non-empty, matching ``stream`` declaration is additionally
    required before distinct contiguous addresses may coalesce.
    """
    unique: list[ScratchpadAccess] = []
    seen: dict[tuple, int] = {}
    for access in accesses:
        key = (access.buffer, access.logical_element,
               access.logical_step, access.replica)
        prior_index = seen.get(key)
        if prior_index is not None:
            prior = unique[prior_index]
            if prior.stream != access.stream:
                unique[prior_index] = ScratchpadAccess(
                    prior.buffer, prior.logical_element, prior.logical_step,
                    prior.replica, None)
            continue
        seen[key] = len(unique)
        unique.append(access)

    packed_with_order: list[tuple[int, PackedScratchpadAccess]] = []
    stream_buckets: dict[tuple, list[tuple[int, int]]] = {}
    for order_index, access in enumerate(unique):
        if access.stream is None:
            packed_with_order.append((
                order_index,
                PackedScratchpadAccess(
                    access.buffer, (access.logical_element,),
                    access.logical_step, access.replica, None)))
            continue
        key = (access.buffer, access.logical_step,
               access.replica, access.stream)
        stream_buckets.setdefault(key, []).append(
            (order_index, access.logical_element))

    for (buffer, logical_step, replica, stream), requests in \
            stream_buckets.items():
        first_order = {element: order for order, element in requests}
        for group in coalesced_element_groups(
                (element for _order, element in requests), target.vector_width):
            packed_with_order.append((
                min(first_order[element] for element in group),
                PackedScratchpadAccess(
                    buffer, group, logical_step, replica, stream)))

    packed_with_order.sort(key=lambda item: item[0])
    return tuple(access for _order, access in packed_with_order)


def scratchpad_port_metrics(
        read_ops: int, write_ops: int,
        target: AnalyticTargetSpec) -> PortMetrics:
    """Return deterministic non-pipelined scratchpad port pressure."""
    if read_ops < 0 or write_ops < 0:
        raise ValueError("scratchpad operation counts must be non-negative")
    read_cycles = _ceil_div(read_ops, target.load_ports) * target.access_cycles
    write_cycles = _ceil_div(write_ops, target.store_ports) * target.access_cycles
    port_cycles = max(read_cycles, write_cycles)
    return PortMetrics(read_ops, write_ops, port_cycles, port_cycles)


def packed_access_port_metrics(
        accesses, write_ops: int,
        target: AnalyticTargetSpec) -> PortMetrics:
    return scratchpad_port_metrics(len(tuple(accesses)), write_ops, target)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _p_tot(spec: KernelSpec, cand: Candidate) -> int:
    prod = 1
    for lv in spec.levels:
        if lv.parallelizable():
            p, _ = cand.factors(lv.name)
            prod *= p
    return prod


def _waves(spec: KernelSpec, cand: Candidate) -> int:
    w = 1
    for lv in spec.levels:
        if lv.tiled():
            p, u = cand.factors(lv.name)
            w *= _ceil_div(lv.trip, p * u)
    return max(1, w)


def _exposed_iters(spec: KernelSpec, cand: Candidate) -> int:
    e = 1
    for lv in spec.levels:
        p, u = cand.factors(lv.name)
        e *= (p * u) if lv.tiled() else lv.trip
    return e


@dataclass
class CandResult:
    cand: Candidate
    p_tot: int
    active_L: int
    active_S: int
    exposed_iters: int
    waves: int
    CP: int
    A: int
    LD: int          # recurring load lane-slots (drive binding / lane exposure)
    ST: int
    ld_eff: int      # LD_eff = recurring + one-time invariant loads (total traffic)
    chunk_aggregate: int
    chunk_scheduled: int | None
    pragma_exposure_aggregate: int
    schedule_estimate: int | None
    binding_class: str
    saturation: str             # latency-bound | resource-bound
    util: tuple                 # (P, L, S)
    flags: set = field(default_factory=set)
    target_profile: AnalyticTargetSpec | None = None
    jam_plan: JamPlan | None = None
    memory_plan: MemoryPlan | None = None
    plan_cgra_lb: int | None = None
    absolute_cgra_lb: int | None = None
    preload_scalar_elements: int = 0
    preload_load_ops: int = 0
    preload_spad_write_ops: int = 0
    scratchpad_reads: int = 0
    avoided_direct_loads: int = 0
    capacity_bytes_used: int = 0
    spad_port_lb: int = 0
    spad_port_sched: int = 0
    schedule_structure_key: tuple = ()
    recurring_demand: tuple[int, int, int, int, int] = ()  # P/L/S/R/W
    nominal_terms: tuple[int, int, int, int, int] = ()     # P/L/S/R/W
    nominal_cp: int = 0


@dataclass
class SearchOutcome:
    results: list[CandResult]
    recommendation: CandResult
    absolute_lb: int
    demand: dict
    search_scope: str
    target_profile: AnalyticTargetSpec | None = None
    legal_candidate_count: int = 0
    deduped_group_count: int = 0


def _cost_extended_phase(phase: PhaseSummary, cfg: Config,
                         target: AnalyticTargetSpec,
                         schedule: bool) -> PhaseCost:
    compute = _ceil_div(phase.A, cfg.P)
    load = _ceil_div(phase.recurring_loads, cfg.L)
    store = _ceil_div(phase.stores, cfg.S)
    base_aggregate = max(phase.CP, compute, load, store)
    port = (phase.port_metrics if phase.port_metrics is not None
            else packed_access_port_metrics(
                phase.spad_read_accesses, phase.spad_write_ops, target))
    aggregate = max(base_aggregate, port.port_lb)
    scheduled = None
    if schedule:
        if phase.base_scheduled is None:
            raise ValueError("scheduled extended phase omitted base_scheduled")
        if phase.base_scheduled < base_aggregate:
            raise AssertionError(
                f"phase schedule {phase.base_scheduled} < aggregate "
                f"{base_aggregate}")
        scheduled = max(phase.base_scheduled, port.port_sched)
        if scheduled < aggregate:
            raise AssertionError(
                f"port-corrected phase schedule {scheduled} < aggregate "
                f"{aggregate}")
    return PhaseCost(aggregate, scheduled, compute, load, store, port)


def _evaluate_extended_candidate(
        spec: KernelSpec, cand: Candidate, cfg: Config, schedule: bool,
        target: AnalyticTargetSpec | None) -> CandResult:
    if not callable(spec.extended_plan_builder):
        raise TypeError(f"{spec.name}: extended_plan_builder must be callable")
    resolved_target = target or AnalyticTargetSpec()
    plan = spec.extended_plan_builder(
        spec, cand, cfg, resolved_target, schedule)
    if not isinstance(plan, ExtendedPlanSummary):
        raise TypeError(
            f"{spec.name}: extended_plan_builder must return "
            "ExtendedPlanSummary")
    if plan.memory_plan.target != resolved_target:
        raise ValueError(
            f"{spec.name}: extended plan returned the wrong target profile")

    execution = plan.execution
    preload_cost = _cost_extended_phase(
        execution.preload, cfg, resolved_target, schedule)
    full_compute_cost = _cost_extended_phase(
        execution.full_compute, cfg, resolved_target, False)
    all_wave_phases = execution.compute_waves
    all_wave_costs = tuple(_cost_extended_phase(
        wave, cfg, resolved_target, schedule) for wave in all_wave_phases)

    plan_cgra_lb = preload_cost.aggregate + full_compute_cost.aggregate
    pragma_aggregate = preload_cost.aggregate + sum(
        cost.aggregate for cost in all_wave_costs)
    schedule_estimate = None
    chunk_scheduled = None
    if schedule:
        schedule_estimate = preload_cost.scheduled + sum(
            cost.scheduled for cost in all_wave_costs)
        chunk_scheduled = sum(cost.scheduled for cost in all_wave_costs)

    if plan_cgra_lb > pragma_aggregate:
        raise AssertionError(
            f"{spec.name} {cand.signature()}: plan_cgra_lb {plan_cgra_lb} > "
            f"p_agg {pragma_aggregate}")
    if schedule and pragma_aggregate > schedule_estimate:
        raise AssertionError(
            f"{spec.name} {cand.signature()}: p_agg {pragma_aggregate} > "
            f"sched {schedule_estimate}")

    compute_total = sum(cost.compute for cost in all_wave_costs)
    load_total = sum(cost.load for cost in all_wave_costs)
    store_total = sum(cost.store for cost in all_wave_costs)
    read_port_wave_total = sum(
        _ceil_div(cost.port.read_ops, resolved_target.load_ports)
        * resolved_target.access_cycles for cost in all_wave_costs)
    write_port_wave_total = sum(
        _ceil_div(cost.port.write_ops, resolved_target.store_ports)
        * resolved_target.access_cycles for cost in all_wave_costs)
    recurring_read_port_cycles = (
        _ceil_div(
            sum(cost.port.read_ops for cost in all_wave_costs),
            resolved_target.load_ports)
        * resolved_target.access_cycles)
    recurring_write_port_cycles = (
        _ceil_div(
            sum(cost.port.write_ops for cost in all_wave_costs),
            resolved_target.store_ports)
        * resolved_target.access_cycles)
    compute_aggregate = sum(cost.aggregate for cost in all_wave_costs)
    all_latency_bound = all(
        phase.CP > max(cost.compute, cost.load, cost.store, cost.port.port_lb)
        for phase, cost in zip(all_wave_phases, all_wave_costs))
    saturation = "latency-bound" if all_latency_bound else "resource-bound"
    binding_terms = {
        "P": compute_total, "L": load_total, "S": store_total,
        "R": read_port_wave_total, "W": write_port_wave_total,
    }
    binding_class = max(
        binding_terms, key=lambda key: (binding_terms[key], key == "L"))
    denom = compute_aggregate if compute_aggregate > 0 else 1
    util = (compute_total / denom, load_total / denom, store_total / denom)

    recurring_loads = sum(phase.recurring_loads for phase in all_wave_phases)
    invariant_loads = sum(phase.invariant_loads for phase in all_wave_phases)
    preload_load_ops = (
        execution.preload.recurring_loads + execution.preload.invariant_loads)
    preload_store_ops = execution.preload.spad_write_ops
    algorithmic_A = sum(
        phase.A - phase.control_A for phase in all_wave_phases)
    recurring_data_loads = sum(
        phase.recurring_loads - phase.control_loads
        for phase in all_wave_phases)
    data_stores = sum(
        phase.stores - phase.control_stores for phase in all_wave_phases)
    nominal_index = max(
        range(len(all_wave_phases)),
        key=lambda index: (
            all_wave_phases[index].A
            - all_wave_phases[index].control_A,
            all_wave_phases[index].recurring_loads
            - all_wave_phases[index].control_loads,
            all_wave_phases[index].stores
            - all_wave_phases[index].control_stores))
    nominal_phase = all_wave_phases[nominal_index]
    nominal_cost = all_wave_costs[nominal_index]
    return CandResult(
        cand=cand, p_tot=_p_tot(spec, cand),
        active_L=max(1, min(max(
            (phase.recurring_loads for phase in all_wave_phases), default=0),
            cfg.L)),
        active_S=max(1, min(max(
            (phase.stores for phase in all_wave_phases), default=0), cfg.S)),
        exposed_iters=_exposed_iters(spec, cand),
        waves=len(all_wave_phases),
        CP=sum(phase.CP for phase in all_wave_phases),
        A=sum(phase.A for phase in all_wave_phases),
        LD=recurring_loads,
        ST=sum(phase.stores for phase in all_wave_phases),
        ld_eff=recurring_loads + invariant_loads,
        chunk_aggregate=compute_aggregate,
        chunk_scheduled=chunk_scheduled,
        pragma_exposure_aggregate=pragma_aggregate,
        schedule_estimate=schedule_estimate,
        binding_class=binding_class, saturation=saturation, util=util,
        target_profile=resolved_target, jam_plan=plan.jam_plan,
        memory_plan=plan.memory_plan, plan_cgra_lb=plan_cgra_lb,
        preload_scalar_elements=plan.preload_scalar_elements,
        preload_load_ops=preload_load_ops,
        preload_spad_write_ops=preload_store_ops,
        scratchpad_reads=plan.scratchpad_reads,
        avoided_direct_loads=plan.avoided_direct_loads,
        capacity_bytes_used=plan.memory_plan.capacity_bytes_used,
        spad_port_lb=(preload_cost.port.port_lb
                      + sum(cost.port.port_lb for cost in all_wave_costs)),
        spad_port_sched=(
            preload_cost.port.port_sched
            + sum(cost.port.port_sched for cost in all_wave_costs)
            if schedule else 0),
        schedule_structure_key=plan.schedule_structure_key,
        recurring_demand=(algorithmic_A, recurring_data_loads,
                          data_stores, recurring_read_port_cycles,
                          recurring_write_port_cycles),
        nominal_terms=(nominal_cost.compute, nominal_cost.load,
                       nominal_cost.store,
                       _ceil_div(nominal_cost.port.read_ops,
                                 resolved_target.load_ports)
                       * resolved_target.access_cycles,
                       _ceil_div(nominal_cost.port.write_ops,
                                 resolved_target.store_ports)
                       * resolved_target.access_cycles),
        nominal_cp=nominal_phase.CP)


def evaluate_candidate(spec: KernelSpec, cand: Candidate, cfg: Config,
                       schedule: bool = True,
                       target: AnalyticTargetSpec | None = None) -> CandResult:
    if spec.extended_plan_builder is not None:
        return _evaluate_extended_candidate(
            spec, cand, cfg, schedule, target)

    # Build the vectorized chunk DAG and schedule it on the FULL machine lanes.
    # With no banking, the only per-cycle cap is L/S. Two axes separate P from U:
    # vector coalescing (which lowered LD/ST in the DAG) and control amortization.
    #
    # Load accounting splits into recurring vs. one-time invariant loads. Only the
    # RECURRING loads set the steady-state lane exposure and the binding load
    # term; invariant loads are amortized (loaded once and held) and reported only
    # in LD_eff. active_L = min(recurring, L) is the recurring lane-exposure
    # diagnostic.
    dag = spec.build_chunk(cand)
    if schedule:
        res = evaluate(dag, spec.name, cfg)
        region_aggs = res.region_aggs
        chunk_scheduled = res.scheduled_cycles
    else:
        region_aggs = [region_aggregate(region, cfg) for region in dag.regions]
        chunk_scheduled = None
    load_splits = _region_load_splits(dag)
    recurring_LD = sum(recurring for recurring, _ in load_splits)
    invariant_LD = sum(invariant for _, invariant in load_splits)
    ld_eff = recurring_LD + invariant_LD   # total load traffic across regions
    waves = _waves(spec, cand)
    p_tot = _p_tot(spec, cand)

    # Ordered regions retain separate resource ceilings. This matters for
    # barrier-ordered kernels such as FFT: max-of-kernel totals would allow work
    # from different phases to overlap and understate the spec-defined bound.
    region_terms = []
    for agg, (region_recurring, _region_invariant) in zip(
            region_aggs, load_splits):
        region_load = _ceil_div(region_recurring, cfg.L)
        region_terms.append((agg.CP, agg.compute, region_load, agg.store))
    CP = sum(terms[0] for terms in region_terms)
    compute = sum(terms[1] for terms in region_terms)
    load = sum(terms[2] for terms in region_terms)
    store = sum(terms[3] for terms in region_terms)
    chunk_aggregate = sum(max(terms) for terms in region_terms)
    active_L = max(1, min(max(
        (recurring for recurring, _ in load_splits), default=0), cfg.L))
    active_S = max(1, min(max(
        (agg.ST for agg in region_aggs), default=0), cfg.S))

    # Binding is summarized over the ordered phase totals. For a multi-region
    # kernel this is descriptive only; the actual aggregate remains the sum of
    # each region's maximum above.
    terms = {"P": compute, "L": load, "S": store}
    binding_class = max(terms, key=lambda k: (terms[k], k == "L"))
    all_latency_bound = all(cp > max(comp, ld, st)
                            for cp, comp, ld, st in region_terms)
    saturation = "latency-bound" if all_latency_bound else "resource-bound"
    denom = chunk_aggregate if chunk_aggregate > 0 else 1
    util = (compute / denom, load / denom, store / denom)
    estimate_multiplier = waves if spec.repeat_waves else 1

    return CandResult(
        cand=cand, p_tot=p_tot, active_L=active_L, active_S=active_S,
        exposed_iters=_exposed_iters(spec, cand), waves=waves,
        CP=CP, A=sum(agg.A for agg in region_aggs), LD=recurring_LD,
        ST=sum(agg.ST for agg in region_aggs), ld_eff=ld_eff,
        chunk_aggregate=chunk_aggregate, chunk_scheduled=chunk_scheduled,
        pragma_exposure_aggregate=estimate_multiplier * chunk_aggregate,
        schedule_estimate=(estimate_multiplier * chunk_scheduled
                           if chunk_scheduled is not None else None),
        binding_class=binding_class, saturation=saturation, util=util)


def _ensure_scheduled(spec: KernelSpec, result: CandResult, cfg: Config,
                      target: AnalyticTargetSpec | None = None) -> None:
    if result.schedule_estimate is not None:
        return
    if spec.extended_plan_builder is not None:
        resolved_target = target or result.target_profile
        scheduled = evaluate_candidate(
            spec, result.cand, cfg, schedule=True, target=resolved_target)
        if (scheduled.plan_cgra_lb != result.plan_cgra_lb
                or scheduled.pragma_exposure_aggregate
                != result.pragma_exposure_aggregate
                or scheduled.schedule_structure_key
                != result.schedule_structure_key
                or scheduled.recurring_demand != result.recurring_demand
                or scheduled.nominal_terms != result.nominal_terms
                or scheduled.nominal_cp != result.nominal_cp):
            raise AssertionError(
                f"{spec.name} {result.cand.signature()}: scheduled rerun changed "
                "the aggregate or schedule structure")
        result.chunk_scheduled = scheduled.chunk_scheduled
        result.schedule_estimate = scheduled.schedule_estimate
        result.spad_port_sched = scheduled.spad_port_sched
        if result.absolute_cgra_lb is not None and not (
                result.absolute_cgra_lb <= result.plan_cgra_lb
                <= result.pragma_exposure_aggregate
                <= result.schedule_estimate):
            raise AssertionError(
                f"{spec.name} {result.cand.signature()}: extended four-term "
                "bracket violated after scheduling")
        return
    scheduled = evaluate_candidate(spec, result.cand, cfg, schedule=True)
    result.chunk_scheduled = scheduled.chunk_scheduled
    result.schedule_estimate = scheduled.schedule_estimate


def _evaluate_aggregate_task(args) -> CandResult:
    spec, cand, cfg = args
    return evaluate_candidate(spec, cand, cfg, schedule=False)


def _evaluate_extended_aggregate_task(args) -> CandResult | None:
    spec, cand, cfg, target = args
    try:
        return evaluate_candidate(
            spec, cand, cfg, schedule=False, target=target)
    except IllegalCandidateError:
        return None


def _full_candidate(spec: KernelSpec) -> Candidate:
    """Full unroll of the whole loop, maximally coalesced: p=1, u=trip on every
    parallelizable level (reduction/sequential are fully consumed anyway)."""
    split = []
    for lv in spec.levels:
        if lv.tiled():
            split.append((lv.name, 1, lv.trip))
        else:
            split.append((lv.name, 1, lv.trip))
    return Candidate(tuple(split))


def absolute_cgra_lb(spec: KernelSpec, cfg: Config) -> tuple[int, dict]:
    """Full-trip, fully-coalesced aggregate over FULL lanes: the only lower
    bound. Aggregate-only (no scheduling) so it scales to the full unrolled DAG.
    Also returns the full-trip demand for the binding-class analysis."""
    dag = spec.build_chunk(_full_candidate(spec))
    region_aggs = [region_aggregate(region, cfg) for region in dag.regions]
    load_splits = _region_load_splits(dag)
    region_terms = []
    for agg, (region_recurring, _region_invariant) in zip(
            region_aggs, load_splits):
        region_terms.append((agg.CP, agg.compute,
                             _ceil_div(region_recurring, cfg.L), agg.store))
    recurring_LD = sum(recurring for recurring, _ in load_splits)
    invariant_LD = sum(invariant for _, invariant in load_splits)
    aggregate = sum(max(terms) for terms in region_terms)
    demand = {
        "A": sum(agg.A for agg in region_aggs),
        "LD": recurring_LD,
        "LD_eff": recurring_LD + invariant_LD,
        "ST": sum(agg.ST for agg in region_aggs),
        "CP": sum(terms[0] for terms in region_terms),
        "compute": sum(terms[1] for terms in region_terms),
        "load": sum(terms[2] for terms in region_terms),
        "store": sum(terms[3] for terms in region_terms),
        "aggregate": aggregate,
        "region_count": len(region_terms),
    }
    return aggregate, demand


# ---------------------------------------------------------------------------
# Recommendation and flags
# ---------------------------------------------------------------------------

def _extended_family_identity(
        spec: KernelSpec, result: CandResult) -> tuple:
    if result.memory_plan is None:
        raise ValueError("extended candidate omitted its memory plan")
    placement = tuple(
        (buffer.name, buffer.placement, buffer.replication_factor)
        for buffer in result.memory_plan.buffers)
    return (
        candidate_order(spec, result.cand),
        result.cand.jam_plan,
        placement,
    )


def _extended_family_knees(
        spec: KernelSpec,
        results: list[CandResult]) -> dict[tuple, CandResult]:
    by_family_exposure: dict[tuple, dict[int, list[CandResult]]] = {}
    for result in results:
        by_family_exposure.setdefault(
            _extended_family_identity(spec, result), {}).setdefault(
                result.exposed_iters, []).append(result)

    knees = {}
    for identity, by_exposure in by_family_exposure.items():
        exposures = sorted(by_exposure)
        frontier = {}
        for exposure in exposures:
            best_compute = min(
                result.chunk_aggregate for result in by_exposure[exposure])
            frontier[exposure] = [
                result for result in by_exposure[exposure]
                if result.chunk_aggregate == best_compute]

        future_min = {}
        running = [math.inf] * 5
        for exposure in reversed(exposures):
            for result in frontier[exposure]:
                if len(result.recurring_demand) != 5:
                    raise ValueError(
                        "extended recommendation omitted P/L/S/R/W demand")
                for index, demand in enumerate(result.recurring_demand):
                    running[index] = min(running[index], demand)
            future_min[exposure] = tuple(int(value) for value in running)

        for exposure in exposures:
            eligible = []
            for result in frontier[exposure]:
                if len(result.nominal_terms) != 5:
                    raise ValueError(
                        "extended recommendation omitted nominal P/L/S/R/W "
                        "terms")
                max_term = max(result.nominal_terms)
                if max_term < result.nominal_cp:
                    continue
                dominant = [
                    index for index, term in enumerate(result.nominal_terms)
                    if term == max_term]
                if any(
                        result.recurring_demand[index]
                        == future_min[exposure][index]
                        for index in dominant):
                    eligible.append(result)
            if eligible:
                knees[identity] = min(
                    eligible,
                    key=lambda result: (
                        result.pragma_exposure_aggregate,
                        result.recurring_demand[1]
                        + result.recurring_demand[2],
                        result.recurring_demand[3]
                        + result.recurring_demand[4],
                        result.p_tot,
                        result.cand.signature()))
                break
    return knees


def _extended_global_rank(result: CandResult) -> tuple:
    return (
        result.pragma_exposure_aggregate,
        result.recurring_demand[1] + result.recurring_demand[2],
        result.recurring_demand[3] + result.recurring_demand[4],
        result.exposed_iters,
        result.p_tot,
        result.cand.signature(),
    )


def recommend(spec: KernelSpec, results: list[CandResult]) -> CandResult | None:
    """Apply the spec's exposure-selection policy.

    Legacy candidates use the first best-coalesced resource-bound exposure.
    Extended candidates derive a recurring-demand-mature knee independently for
    each order/jam/placement family, then choose the lowest-cost family knee.
    """
    if not results:
        return None
    if any(result.target_profile is not None for result in results):
        family_knees = _extended_family_knees(spec, results)
        if family_knees:
            return min(family_knees.values(), key=_extended_global_rank)
        return min(results, key=_extended_global_rank)
    if not spec.repeat_waves:
        # The builder already emits every phase-local wave and every fixed
        # once-only region. A fixed stage may be resource-bound even when the
        # tunable phase is underexposed, so the single-loop saturation test is
        # not meaningful. Select the smallest exposure that reaches the best
        # whole-kernel estimate, then prefer the most coalesced/fewest-worker
        # representative at that exposure.
        best_estimate = min(r.pragma_exposure_aggregate for r in results)
        best = [r for r in results
                if r.pragma_exposure_aggregate == best_estimate]
        min_exposure = min(r.exposed_iters for r in best)
        best = [r for r in best if r.exposed_iters == min_exposure]
        return min(best, key=lambda r: (r.LD + r.ST, r.p_tot,
                                        r.cand.signature()))
    by_exp: dict[int, list[CandResult]] = {}
    for r in results:
        by_exp.setdefault(r.exposed_iters, []).append(r)
    for E in sorted(by_exp):
        best = min(by_exp[E], key=lambda r: (r.LD + r.ST, r.p_tot,
                                             r.pragma_exposure_aggregate))
        if best.saturation == "resource-bound":
            return best
    # nothing saturates (pure latency-bound sweep): take the best estimate.
    return min(results, key=lambda r: (r.pragma_exposure_aggregate,
                                       r.exposed_iters))


def annotate_flags(spec: KernelSpec, results: list[CandResult],
                   rec: CandResult) -> None:
    latency_fallback = spec.selection_mode == "latency_fallback"
    extended = _uses_extended_candidate_space(spec)
    family_knees = _extended_family_knees(spec, results) if extended else {}
    for r in results:
        same_candidate = (r.cand == rec.cand if extended
                          else r.cand.split == rec.cand.split)
        if same_candidate:
            r.flags.add("recommended")
            continue  # the pick carries no starved/oversubscribed marker
        if not spec.repeat_waves:
            continue
        if extended:
            family_knee = family_knees.get(
                _extended_family_identity(spec, r))
            if family_knee is None:
                continue
            if r.exposed_iters < family_knee.exposed_iters:
                r.flags.add("bandwidth-starved")
            elif (r.saturation == "resource-bound"
                  and r.exposed_iters > family_knee.exposed_iters):
                r.flags.add("oversubscribed")
            continue
        if latency_fallback:
            if r.pragma_exposure_aggregate > rec.pragma_exposure_aggregate:
                r.flags.add("bandwidth-starved")
            continue
        # bandwidth-starved: below the knee (latency-bound: resources idle while
        # the critical path drains).
        if r.saturation == "latency-bound":
            r.flags.add("bandwidth-starved")
        # oversubscribed: resource-bound past the knee -- more exposure than the
        # recommendation, buying only wave-serialization rounding at the cost of
        # transient backlog and area.
        if (r.saturation == "resource-bound"
                and r.exposed_iters > rec.exposed_iters):
            r.flags.add("oversubscribed")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _dedup(results: list[CandResult]) -> list[list[CandResult]]:
    """Group candidates with identical performance (same chunk counts, active
    widths, waves, estimate) -- e.g. inert reduction/parallel pragmas."""
    groups: dict[tuple, list[CandResult]] = {}
    for r in results:
        legacy_key = (r.active_L, r.active_S, r.LD, r.ST, r.A, r.CP, r.waves,
                      r.pragma_exposure_aggregate)
        if r.target_profile is None:
            key = legacy_key
        else:
            key = legacy_key + (
                r.cand.order, r.cand.jam_plan, r.target_profile,
                r.jam_plan, r.memory_plan, r.plan_cgra_lb,
                r.capacity_bytes_used, r.spad_port_lb,
                r.preload_scalar_elements, r.preload_load_ops,
                r.preload_spad_write_ops, r.scratchpad_reads,
                r.avoided_direct_loads,
                r.schedule_structure_key,
                r.recurring_demand, r.nominal_terms, r.nominal_cp)
        groups.setdefault(key, []).append(r)
    grouped = []
    for group in groups.values():
        group.sort(key=lambda r: ("recommended" not in r.flags,
                                  r.p_tot, r.cand.signature()))
        grouped.append(group)
    ordered = sorted(grouped,
                     key=lambda g: (g[0].pragma_exposure_aggregate,
                                    g[0].exposed_iters, -g[0].p_tot))
    return ordered


def _fmt_util(u):
    return "/".join(str(round(x * 100)) for x in u)


def _extended_candidate_signature(spec: KernelSpec, cand: Candidate) -> str:
    split = " ".join(f"{name}:P{parallel}U{unroll}"
                     for name, parallel, unroll in cand.split)
    order = ">".join(candidate_order(spec, cand))
    return f"{split} order={order} jam={cand.jam_plan}"


def _format_compact_ints(values) -> str:
    unique = tuple(sorted(set(values)))
    if not unique:
        return "none"
    if len(unique) <= 6:
        return ",".join(str(value) for value in unique)
    return f"{unique[0]}..{unique[-1]} ({len(unique)} values)"


def _format_extended_jam(jam_plan: JamPlan | None) -> str:
    if jam_plan is None or not jam_plan.edges:
        return "none"
    rendered = []
    for edge in jam_plan.edges:
        operands = (f"[{','.join(edge.shared_operands)}]"
                    if edge.shared_operands else "")
        rendered.append(f"{edge.outer}->{edge.inner}{operands}")
    return f"{jam_plan.name}: " + ", ".join(rendered)


def _format_extended_memory(memory: MemoryPlan) -> str:
    rendered = []
    for buffer in memory.buffers:
        if buffer.placement in ("direct", "direct-fallback"):
            rendered.append(f"{buffer.name}={buffer.placement}")
            continue
        bases = _format_compact_ints(buffer.replica_bases)
        rendered.append(
            f"{buffer.name}={buffer.placement}(base_elem={bases},"
            f"replicas={buffer.replication_factor},bytes={buffer.bytes_used})")
    return "; ".join(rendered)


def _render_extended_report(
        spec: KernelSpec, cfg: Config, results: list[CandResult], lb: int,
        rec: CandResult, search_scope: str, top: int,
        brief_recommendations: tuple[tuple[str, str], ...]) -> str:
    target = rec.target_profile
    memory = rec.memory_plan
    bounded = search_scope.startswith("bounded diagnostic")
    best_estimate_fallback = not _extended_family_knees(spec, results)
    _ensure_scheduled(spec, rec, cfg, target)
    out = [
        f"# Loom pragma DSE (analytic_prefilter): {spec.name}  ({cfg.label})",
        "",
        (f"Evidence: `analytic_prefilter`; target `{target.name}`; one "
         f"{target.capacity_bytes}-byte scratchpad shared across this kernel; "
         f"R={target.load_ports}, W={target.store_ports}; "
         f"{target.access_cycles}-cycle non-pipelined access; fixed "
         f"V={target.vector_width}."),
        f"Search: {search_scope}.",
    ]
    groups = _dedup(results)
    if bounded:
        out.append(
            f"Candidates: {len(results)} legal, {len(groups)} deduplicated "
            f"groups; `bounded_search_floor={lb}` is the minimum plan floor "
            "inside this diagnostic subset, not the profile-global floor.")
    else:
        out.append(
            f"Candidates: {len(results)} legal, {len(groups)} deduplicated "
            f"groups; `absolute_cgra_lb={lb}` is the profile-global floor.")
    out.append("")
    header = (f"{'flags':<8} {'candidate':<112} {'plan_lb':>7} {'p_agg':>7} "
              f"{'sched':>7} {'cap_B':>6} {'spad lb/s':>11} "
              f"{'class':<14} {'util P/L/S':>11}")
    out.append(header)
    out.append("-" * len(header))
    shown = groups
    omitted = 0
    if top and len(groups) > top:
        shown = groups[:top]
        if not any("recommended" in group[0].flags for group in shown):
            rec_group = next((group for group in groups
                              if "recommended" in group[0].flags), None)
            if rec_group is not None:
                shown = shown + [rec_group]
        omitted = len(groups) - len(shown)
    for group in shown:
        result = group[0]
        _ensure_scheduled(spec, result, cfg, target)
        flags = ""
        if "recommended" in result.flags:
            flags += "B" if bounded else "K"
        if "bandwidth-starved" in result.flags:
            flags += "b"
        if "oversubscribed" in result.flags:
            flags += "o"
        signature = _extended_candidate_signature(spec, result.cand)
        if len(group) > 1:
            signature += f"  (+{len(group)-1} eq)"
        port = f"{result.spad_port_lb}/{result.spad_port_sched}"
        out.append(
            f"{flags:<8} {signature:<112} {result.plan_cgra_lb:>7} "
            f"{result.pragma_exposure_aggregate:>7} "
            f"{result.schedule_estimate:>7} "
            f"{result.capacity_bytes_used:>6} {port:>11} "
            f"{result.saturation:<14} {_fmt_util(result.util):>11}")
    if omitted:
        out.append(f"... ({omitted} more groups omitted; use --top 0 for the "
                   "full sweep)")
    out.extend((
        "",
        (f"{'BEST BOUNDED' if bounded else 'RECOMMENDED'}: "
         f"{_extended_candidate_signature(spec, rec.cand)}  -> "
         f"plan_lb={rec.plan_cgra_lb}, "
         f"p_agg={rec.pragma_exposure_aggregate}, "
         f"sched={rec.schedule_estimate}, "
         f"{'best-estimate fallback' if best_estimate_fallback else rec.saturation}"),
        (
            ("flags: B=best bounded estimate; no eligible family knee and no "
             "global recommendation are claimed.")
            if best_estimate_fallback else
            ("flags: B=best row in this bounded diagnostic; no global "
             "recommendation is claimed, b/o are relative to each "
             "transformation family's bounded knee.")
        ) if bounded else (
            ("flags: K=recommended best-estimate fallback; no transformation "
             "family has an eligible resource-bound knee.")
            if best_estimate_fallback else
            ("flags: K=recommended family knee, b=below that row's family knee "
             "(latency-bound or recurring-traffic immature), o=oversubscribed "
             "relative to that row's family knee.")
        ),
        f"Order: `{'>'.join(candidate_order(spec, rec.cand))}`.",
        f"Jam: {_format_extended_jam(rec.jam_plan)}.",
        f"Memory: {_format_extended_memory(memory)}.",
        (f"Capacity: {rec.capacity_bytes_used}/{target.capacity_bytes} B; "
         f"proposed={memory.proposed_capacity_bytes} B; "
         f"fallback={'yes' if memory.fallback else 'no'}."),
        (f"Scratchpad ports: lb={rec.spad_port_lb} cycles; "
         f"sched={rec.spad_port_sched} cycles; "
         f"gap={rec.spad_port_sched - rec.spad_port_lb} cycles."),
        (f"Traffic: preload={rec.preload_scalar_elements} scalar elements, "
         f"{rec.preload_load_ops} external-L ops, "
         f"{rec.preload_spad_write_ops} scratchpad-W ops; "
         f"spad_reads={rec.scratchpad_reads} scalar requests after jam fan-out; "
         f"avoided_direct={rec.avoided_direct_loads} scalar external loads."),
    ))
    if brief_recommendations:
        out.append("")
        for label, signature in brief_recommendations:
            suffix = "bounded best" if bounded else "recommendation"
            out.append(f"{label} {suffix}: {signature}.")
    return "\n".join(out)


def render_report(spec: KernelSpec, cfg: Config, results: list[CandResult],
                  lb: int, demand: dict, rec: CandResult, search_scope: str,
                  top: int = 0,
                  brief_recommendations: tuple[tuple[str, str], ...] = ()) -> str:
    if rec.target_profile is not None:
        return _render_extended_report(
            spec, cfg, results, lb, rec, search_scope, top,
            brief_recommendations)

    bounded = search_scope.startswith("bounded diagnostic")
    out = []
    out.append(f"# Loom pragma DSE (lane-aware + vector coalescing): "
               f"{spec.name}  ({cfg.label})")
    out.append("")
    nest = ", ".join(f"{lv.name}[{lv.trip},{lv.kind}]" for lv in spec.levels)
    ft_terms = {"P": demand["compute"], "L": demand["load"], "S": demand["store"]}
    ft_binding = max(ft_terms, key=lambda k: (ft_terms[k], k == "L"))
    binding = "critical-path" if demand["CP"] > max(ft_terms.values()) \
        else {"P": "compute", "L": "load", "S": "store"}[ft_binding]
    out.append(f"Search: {search_scope}.")
    if demand.get("region_count", 1) == 1:
        floor_text = (
            f"`absolute_cgra_lb={lb}=max(CP {demand['CP']}, "
            f"compute {demand['compute']}, load {demand['load']}, "
            f"store {demand['store']})`, with {binding} pressure binding")
    else:
        floor_text = (
            f"`absolute_cgra_lb={lb}` from the sum of "
            f"{demand['region_count']} ordered-region aggregates "
            f"(region-summed CP {demand['CP']}, compute ceilings "
            f"{demand['compute']}, load ceilings {demand['load']}, and store "
            f"ceilings {demand['store']})")
    out.append(
        f"Loop nest: `{nest}`; {spec.coalesce_note} Full-trip counts are "
        f"`A={demand['A']}`, `LD_rec={demand['LD']}`, "
        f"`LD_eff={demand['LD_eff']}`, `ST={demand['ST']}`, and "
        f"`CP={demand['CP']}`, giving the only lower bound, {floor_text}; "
        f"`p_agg` and `sched` are wave-serialized estimates.")
    out.append("")

    header = (f"{'flags':<8} {'split':<26} {'Ptot':>4} {'aL':>3} {'aS':>3} "
              f"{'LD_eff':>6} {'exp':>5} {'wav':>5} {'cagg':>5} {'p_agg':>7} "
              f"{'sched':>7} {'class':<14} {'util P/L/S':>11}")
    out.append(header)
    out.append("-" * len(header))
    all_groups = _dedup(results)
    shown = all_groups
    omitted = 0
    if top and len(all_groups) > top:
        shown = all_groups[:top]
        if not any("recommended" in g[0].flags for g in shown):
            rec_group = next((g for g in all_groups
                              if "recommended" in g[0].flags), None)
            if rec_group is not None:
                shown = shown + [rec_group]
        omitted = len(all_groups) - len(shown)
    for group in shown:
        r = group[0]
        _ensure_scheduled(spec, r, cfg)
        fl = ""
        if "recommended" in r.flags:
            fl += "B" if bounded else "K"
        if "bandwidth-starved" in r.flags:
            fl += "b"
        if "oversubscribed" in r.flags:
            fl += "o"
        split = r.cand.signature()
        if len(group) > 1:
            split += f"  (+{len(group)-1} eq)"
        out.append(
            f"{fl:<8} {split:<26} {r.p_tot:>4} {r.active_L:>3} {r.active_S:>3} "
            f"{r.ld_eff:>6} {r.exposed_iters:>5} {r.waves:>5} "
            f"{r.chunk_aggregate:>5} {r.pragma_exposure_aggregate:>7} "
            f"{r.schedule_estimate:>7} {r.saturation:<14} {_fmt_util(r.util):>11}")
    if omitted:
        out.append(f"... ({omitted} more groups omitted; use --top 0 for the "
                   "full sweep)")
    out.append("")
    ratio = rec.pragma_exposure_aggregate / lb if lb else float("nan")
    latency_fallback = spec.selection_mode == "latency_fallback"
    if not spec.repeat_waves:
        selection = "phase-composed"
    elif latency_fallback:
        selection = "latency-bound best-estimate fallback"
    else:
        selection = rec.saturation
    out.append(f"{'BEST BOUNDED' if bounded else 'RECOMMENDED'}: "
               f"{rec.cand.signature()}  -> "
               f"exposure={rec.exposed_iters}, "
               f"pragma_agg={rec.pragma_exposure_aggregate} "
               f"({ratio:.2f}x the floor), {selection}")
    if bounded:
        out.append("flags: B=best row in this bounded diagnostic; no global "
                   "recommendation is claimed.")
    elif not spec.repeat_waves:
        out.append("flags: K=recommended (smallest tunable-phase exposure "
                   "that reaches the best phase-composed estimate).")
    elif latency_fallback:
        out.append("flags: K=recommended (smallest split reaching the best "
                   "estimate; no resource-bound knee), b=higher "
                   "wave-serialized estimate.")
    else:
        out.append("flags: K=recommended (saturation knee E_sat), "
                   "b=bandwidth-starved (latency-bound: resources idle), "
                   "o=oversubscribed (past the knee, no estimate gain).")
    out.append("")
    out.append(_pu_contrast(spec, cfg, results, rec))
    if brief_recommendations:
        out.append("")
        for label, signature in brief_recommendations:
            suffix = "bounded best" if bounded else "recommendation"
            out.append(f"{label} {suffix}: {signature}.")
    return "\n".join(out)


def _pu_contrast(spec: KernelSpec, cfg: Config,
                 results: list[CandResult], rec: CandResult) -> str:
    """Fixed-product P-vs-U contrast on the primary parallelizable level: hold
    p*u constant, vary the split, show the load/store-term difference (or its
    absence)."""
    prim = next((lv for lv in spec.levels if lv.kind == "parallel"), None)
    if prim is None:
        prim = next((lv for lv in spec.levels if lv.parallelizable()), None)
    if prim is None:
        return "P-vs-U contrast: no parallelizable level."

    def others_trivial(r):
        return all(p == 1 and u == 1
                   for n, p, u in r.cand.split if n != prim.name)

    by_prod: dict[int, list[CandResult]] = {}
    for r in results:
        if not others_trivial(r):
            continue
        p, u = r.cand.factors(prim.name)
        by_prod.setdefault(p * u, []).append(r)
    candidates = {prod: rs for prod, rs in by_prod.items() if len(rs) >= 2}
    if not candidates:
        return "P-vs-U contrast: primary level has no fixed-product split set."

    def spread(prod):
        vals = [r.pragma_exposure_aggregate for r in candidates[prod]]
        return (max(vals) / min(vals), prod)

    rec_p, rec_u = rec.cand.factors(prim.name)
    rec_prod = rec_p * rec_u
    prod = rec_prod if rec_prod in candidates else max(candidates, key=spread)
    rows = sorted(candidates[prod], key=lambda r: -r.cand.factors(prim.name)[0])
    lines = [f"P-vs-U at fixed product {prod} on level '{prim.name}' "
             f"(other levels at P1U1):"]
    lines.append(f"  {'split':<12} {'LD_rec':>6} {'LD_eff':>6} {'ST':>5} "
                 f"{'p_agg':>7} {'note'}")
    best = min(r.pragma_exposure_aggregate for r in rows)
    for r in rows:
        p, u = r.cand.factors(prim.name)
        if r.pragma_exposure_aggregate == best:
            note = "best" if len({x.pragma_exposure_aggregate for x in rows}) > 1 \
                else "tie (control/coalescing sit below the binding term)"
        else:
            note = (f"{r.pragma_exposure_aggregate/best:.2f}x slower "
                    "(parallel: extra iterators + strided, no coalesce)")
        lines.append(f"  P{p}U{u:<10} {r.LD:>6} {r.ld_eff:>6} {r.ST:>5} "
                     f"{r.pragma_exposure_aggregate:>7} {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _config_key(cfg: Config) -> tuple[int, int, int]:
    return cfg.P, cfg.L, cfg.S


def _normalize_brief_configs(primary: Config,
                             configs: list[Config]) -> list[Config]:
    """Deduplicate equivalent capacities and suppress the detailed config."""
    seen = {_config_key(primary)}
    normalized = []
    for cfg in configs:
        key = _config_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cfg)
    return normalized


def _target_name(capacity_bytes: int, load_ports: int, store_ports: int,
                 access_cycles: int, vector_width: int = V) -> str:
    capacity = "4k" if capacity_bytes == 4096 else f"{capacity_bytes}b"
    latency = "" if access_cycles == 1 else f"-a{access_cycles}"
    return (f"shared-spad-{capacity}-r{load_ports}w{store_ports}"
            f"{latency}-v{vector_width}")


def make_target(
        capacity_bytes: int = DEFAULT_SPAD_CAPACITY_BYTES,
        load_ports: int = DEFAULT_SPAD_LOAD_PORTS,
        store_ports: int = DEFAULT_SPAD_STORE_PORTS,
        access_cycles: int = DEFAULT_SPAD_ACCESS_CYCLES) -> AnalyticTargetSpec:
    return AnalyticTargetSpec(
        name=_target_name(
            capacity_bytes, load_ports, store_ports, access_cycles),
        capacity_bytes=capacity_bytes, load_ports=load_ports,
        store_ports=store_ports, access_cycles=access_cycles)


def _search_extended(
        spec: KernelSpec, cfg: Config, max_parallel: int | None,
        max_unroll: int | None, exposure_cap: int | None, jobs: int,
        target: AnalyticTargetSpec | None) -> SearchOutcome:
    resolved_target = target or AnalyticTargetSpec()
    cands = enumerate_candidates(spec, max_parallel, max_unroll, exposure_cap)
    if jobs == 1 or len(cands) < 2:
        results = []
        for cand in cands:
            try:
                result = evaluate_candidate(
                    spec, cand, cfg, schedule=False, target=resolved_target)
            except IllegalCandidateError:
                continue
            results.append(result)
    else:
        chunksize = max(1, len(cands) // (jobs * 4))
        tasks = ((spec, cand, cfg, resolved_target) for cand in cands)
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            evaluated = executor.map(
                _evaluate_extended_aggregate_task, tasks, chunksize=chunksize)
            results = [result for result in evaluated if result is not None]
    if not results:
        raise NoLegalExtendedCandidateError(
            f"{spec.name}: no legal candidate for target "
            f"{resolved_target.name}")

    absolute_lb = min(result.plan_cgra_lb for result in results)
    for result in results:
        result.absolute_cgra_lb = absolute_lb
        if not (absolute_lb <= result.plan_cgra_lb
                <= result.pragma_exposure_aggregate):
            raise AssertionError(
                f"{spec.name} {result.cand.signature()}: extended aggregate "
                "bracket violated")
        if (result.schedule_estimate is not None
                and result.pragma_exposure_aggregate
                > result.schedule_estimate):
            raise AssertionError(
                f"{spec.name} {result.cand.signature()}: extended scheduled "
                "bracket violated")

    rec = recommend(spec, results)
    if rec is None:
        raise RuntimeError(f"{spec.name}: candidate search produced no result")
    caps = []
    if max_parallel is not None:
        caps.append(f"P<={max_parallel}")
    if max_unroll is not None:
        caps.append(f"U<={max_unroll}")
    if exposure_cap is not None:
        caps.append(f"exposure<={exposure_cap}")
    if caps:
        search_scope = "bounded diagnostic (" + ", ".join(caps) + ")"
    else:
        search_scope = "complete legal power-of-two factors through each trip count"
    return SearchOutcome(
        results=results, recommendation=rec, absolute_lb=absolute_lb,
        demand={"aggregate": absolute_lb}, search_scope=search_scope,
        target_profile=resolved_target, legal_candidate_count=len(results),
        deduped_group_count=len(_dedup(results)))


def search(spec: KernelSpec, cfg: Config, max_parallel: int | None = None,
           max_unroll: int | None = None, exposure_cap: int | None = None,
           jobs: int = 1,
           target: AnalyticTargetSpec | None = None) -> SearchOutcome:
    if spec.extended_plan_builder is not None:
        return _search_extended(
            spec, cfg, max_parallel, max_unroll, exposure_cap, jobs, target)

    lb, demand = absolute_cgra_lb(spec, cfg)
    cands = enumerate_candidates(spec, max_parallel, max_unroll, exposure_cap)
    if jobs == 1 or len(cands) < 2:
        results = [evaluate_candidate(spec, c, cfg, schedule=False)
                   for c in cands]
    else:
        chunksize = max(1, len(cands) // (jobs * 4))
        tasks = ((spec, cand, cfg) for cand in cands)
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(_evaluate_aggregate_task, tasks,
                                        chunksize=chunksize))
    rec = recommend(spec, results)
    if rec is None:
        raise RuntimeError(f"{spec.name}: candidate search produced no result")
    caps = []
    if max_parallel is not None:
        caps.append(f"P<={max_parallel}")
    if max_unroll is not None:
        caps.append(f"U<={max_unroll}")
    if exposure_cap is not None:
        caps.append(f"exposure<={exposure_cap}")
    if caps:
        search_scope = "bounded diagnostic (" + ", ".join(caps) + ")"
    else:
        search_scope = "complete legal power-of-two factors through each trip count"
    return SearchOutcome(results, rec, lb, demand, search_scope)


def run(spec: KernelSpec, cfg: Config, max_parallel: int | None = None,
        max_unroll: int | None = None, exposure_cap: int | None = None,
        top: int = 0, jobs: int = 1,
        brief_configs: list[Config] | None = None,
        target: AnalyticTargetSpec | None = None) -> tuple[str, CandResult, int]:
    outcome = search(
        spec, cfg, max_parallel, max_unroll, exposure_cap, jobs, target)
    annotate_flags(spec, outcome.results, outcome.recommendation)
    brief_recommendations = []
    for brief_cfg in brief_configs or []:
        brief = search(spec, brief_cfg, max_parallel, max_unroll, exposure_cap,
                       jobs, target)
        brief_signature = (
            _extended_candidate_signature(spec, brief.recommendation.cand)
            if spec.extended_plan_builder is not None
            else brief.recommendation.cand.signature())
        brief_recommendations.append(
            (brief_cfg.label, brief_signature))
    report = render_report(
        spec, cfg, outcome.results, outcome.absolute_lb, outcome.demand,
        outcome.recommendation, outcome.search_scope, top=top,
        brief_recommendations=tuple(brief_recommendations))
    rec = outcome.recommendation
    lb = outcome.absolute_lb
    return report, rec, lb


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Lane-aware Loom-pragma design-space estimate")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("kernel", nargs="?",
                        help="kernel name: " + ", ".join(sorted(KERNELS)))
    parser.add_argument("--config", default="6x6")
    parser.add_argument(
        "--brief-config", action="append", default=[],
        help=("append one terse recommendation for this configuration; may be "
              "repeated"))
    parser.add_argument(
        "--max-parallel", type=int,
        help=("optional diagnostic cap; default searches powers of two through "
              "each trip count"))
    parser.add_argument(
        "--max-unroll", type=int,
        help=("optional diagnostic cap; default searches powers of two through "
              "each trip count"))
    parser.add_argument(
        "--exposure-cap", type=int,
        help="optional diagnostic cap on total parallel exposure; default is uncapped")
    parser.add_argument(
        "--spad-capacity-bytes", type=int,
        default=DEFAULT_SPAD_CAPACITY_BYTES,
        help=("shared scratchpad capacity for extended pilots "
              f"(default: {DEFAULT_SPAD_CAPACITY_BYTES})"))
    parser.add_argument(
        "--spad-load-ports", type=int, default=DEFAULT_SPAD_LOAD_PORTS,
        help=("logical scratchpad load ports for extended pilots "
              f"(default: {DEFAULT_SPAD_LOAD_PORTS})"))
    parser.add_argument(
        "--spad-store-ports", type=int, default=DEFAULT_SPAD_STORE_PORTS,
        help=("logical scratchpad store ports for extended pilots "
              f"(default: {DEFAULT_SPAD_STORE_PORTS})"))
    parser.add_argument(
        "--spad-access-cycles", type=int, default=DEFAULT_SPAD_ACCESS_CYCLES,
        help=("non-pipelined scratchpad access latency "
              f"(default: {DEFAULT_SPAD_ACCESS_CYCLES})"))
    parser.add_argument("--top", type=int, default=24,
                        help="show the best N candidate groups (0 schedules all)")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="parallel workers for exhaustive candidate evaluation")
    args = parser.parse_args(argv)

    for name in ("max_parallel", "max_unroll", "exposure_cap",
                 "spad_capacity_bytes", "spad_load_ports",
                 "spad_store_ports", "spad_access_cycles"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    if args.self_test:
        return _run_self_tests()
    if not args.kernel or args.kernel not in KERNELS:
        parser.print_help()
        print("\nkernels: " + ", ".join(sorted(KERNELS)))
        return 1
    spec = KERNELS[args.kernel]
    try:
        cfg = parse_config(args.config)
        brief_configs = [parse_config(text) for text in args.brief_config]
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    brief_configs = _normalize_brief_configs(cfg, brief_configs)
    target = make_target(
        args.spad_capacity_bytes, args.spad_load_ports,
        args.spad_store_ports, args.spad_access_cycles)
    report, _, _ = run(spec, cfg, args.max_parallel, args.max_unroll,
                       args.exposure_cap, top=args.top, jobs=args.jobs,
                       brief_configs=brief_configs, target=target)
    print(report)
    return 0


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _synthetic_memory_planner(
        _spec: KernelSpec, _cand: Candidate,
        _target: AnalyticTargetSpec) -> tuple[BufferSpec, ...]:
    """Small picklable whole-working-set planner for infrastructure tests."""
    return (
        BufferSpec(
            "resident", 4, tuple(range(6)),
            reuse_bearing=True, worker_invariant=True),
        BufferSpec(
            "stream", 4, tuple(range(6)),
            reuse_bearing=False, worker_invariant=True),
    )


def _synthetic_compute_phase(
        source_elements: tuple[int, ...], wave_index: int,
        memory: MemoryPlan, cfg: Config, target: AnalyticTargetSpec,
        schedule: bool) -> PhaseSummary:
    resident = not memory.fallback
    packed = ()
    if resident:
        buffer = next(plan for plan in memory.buffers
                      if plan.name == "resident")
        accesses = tuple(
            ScratchpadAccess(
                buffer.name, buffer.logical_element(source),
                (("wave", wave_index),), stream=buffer.name)
            for source in source_elements)
        packed = pack_scratchpad_accesses(accesses, target)
    extent = len(source_elements)
    phase = PhaseSummary(
        A=extent * cfg.P,
        recurring_loads=(len(packed) if resident else extent),
        invariant_loads=0,
        stores=0,
        CP=max(1, extent),
        spad_read_accesses=packed)
    if not schedule:
        return phase
    base_aggregate = max(
        phase.CP, _ceil_div(phase.A, cfg.P),
        _ceil_div(phase.recurring_loads, cfg.L))
    return PhaseSummary(
        A=phase.A, recurring_loads=phase.recurring_loads,
        invariant_loads=0, stores=0, CP=phase.CP,
        spad_read_accesses=phase.spad_read_accesses,
        base_scheduled=base_aggregate + 1)


def _synthetic_extended_plan_builder(
        spec: KernelSpec, cand: Candidate, cfg: Config,
        target: AnalyticTargetSpec, schedule: bool) -> ExtendedPlanSummary:
    """Build one whole-kernel preload and exact compute waves."""
    memory = derive_memory_plan(spec, cand, target)
    jam = derive_jam_plan(spec, cand)
    preload, scalar_elements = (
        _resident_preload_phase(memory, cfg, schedule)
        if not memory.fallback else (_zero_phase(schedule), 0))
    resident = next(plan for plan in memory.buffers
                    if plan.name == "resident")
    waves = []
    wave_extents = []
    for wave_index, (origins, shapes) in enumerate(
            _parallel_wave_boxes(spec, cand, ("i",))):
        origin = dict(origins)["i"]
        extent = dict(shapes)["i"]
        source_elements = tuple(range(origin, origin + extent))
        wave_extents.append(extent)
        waves.append(_synthetic_compute_phase(
            source_elements, wave_index, memory, cfg, target, schedule))
    wave_tuple = tuple(waves)
    scratchpad_reads = (
        sum(wave_extents) if resident.placement == "resident_shared" else 0)
    return ExtendedPlanSummary(
        memory_plan=memory,
        jam_plan=jam,
        execution=ExtendedExecutionSummary(
            preload=preload,
            full_compute=_combine_wave_phases(wave_tuple),
            compute_waves=wave_tuple),
        schedule_structure_key=(
            "synthetic", candidate_order(spec, cand), jam.name,
            memory.fallback, tuple(wave_extents)),
        preload_scalar_elements=scalar_elements,
        scratchpad_reads=scratchpad_reads,
        avoided_direct_loads=(
            max(0, scratchpad_reads - scalar_elements)
            if not memory.fallback else 0))


def _synthetic_extended_spec() -> KernelSpec:
    return KernelSpec(
        name="extended_search_test",
        levels=(Level("i", 6, "parallel"),
                Level("j", 8, "reduction")),
        build_chunk=_gemv_chunk,
        order_spec=OrderSpec((("i", "j"), ("j", "i"))),
        jam_plans=(
            JamPlanSpec(
                "i-j-share-data",
                (JamRule("i", "j", ("resident",)),)),),
        memory_planner=_synthetic_memory_planner,
        extended_plan_builder=_synthetic_extended_plan_builder)


def _run_extended_infrastructure_tests(errors: list[str]) -> None:
    def expect_value_error(label, callback) -> None:
        try:
            callback()
        except ValueError:
            return
        errors.append(f"{label}: expected ValueError")

    default_target = AnalyticTargetSpec()
    factory_default = make_target()
    expected_default = ("shared-spad-4k-r2w2-v4", 4096, 2, 2, 1, V)
    for label, target in (
            ("AnalyticTargetSpec", default_target),
            ("make_target", factory_default)):
        observed = (
            target.name, target.capacity_bytes, target.load_ports,
            target.store_ports, target.access_cycles, target.vector_width)
        if observed != expected_default:
            errors.append(
                f"{label} default target expected {expected_default}, got "
                f"{observed}")

    # A source-only order declaration must preserve the legacy candidate set.
    legacy_spec = KernelSpec(
        name="legacy_default_test", levels=(Level("i", 5, "parallel"),),
        build_chunk=_axpy_chunk)
    source_order_spec = KernelSpec(
        name="source_order_test", levels=legacy_spec.levels,
        build_chunk=_axpy_chunk,
        order_spec=OrderSpec((("i",),)))
    legacy_candidates = enumerate_candidates(legacy_spec)
    source_candidates = enumerate_candidates(source_order_spec)
    if source_candidates != legacy_candidates:
        errors.append(
            "source-only OrderSpec must preserve the legacy candidate set")
    if any(cand.order or cand.jam_plan != "none"
           for cand in source_candidates):
        errors.append(
            "source-only OrderSpec must keep default candidate metadata")

    spec = _synthetic_extended_spec()
    split_u2 = (("i", 1, 2), ("j", 1, 1))
    split_u1 = (("i", 1, 1), ("j", 1, 1))
    unjammed = derive_jam_plan(spec, Candidate(split_u2))
    if unjammed.name != "none" or unjammed.edges:
        errors.append("outer-loop unroll must not imply jam")

    jammed = derive_jam_plan(
        spec, Candidate(split_u2, jam_plan="i-j-share-data"))
    if jammed.edges != (JamRule("i", "j", ("resident",)),):
        errors.append(f"explicit jam plan mismatch: {jammed.edges}")
    expect_value_error(
        "jam requires outer U greater than one",
        lambda: derive_jam_plan(
            spec, Candidate(split_u1, jam_plan="i-j-share-data")))
    expect_value_error(
        "jam requires inner beneath outer",
        lambda: derive_jam_plan(
            spec, Candidate(
                split_u2, ("j", "i"), "i-j-share-data")))
    expect_value_error(
        "undeclared jam plan rejection",
        lambda: derive_jam_plan(
            spec, Candidate(split_u2, jam_plan="missing")))

    candidates = enumerate_candidates(
        spec, max_parallel=2, max_unroll=2, exposure_cap=4)
    none_candidates = [
        candidate for candidate in candidates
        if candidate.jam_plan == "none"]
    jam_candidates = [
        candidate for candidate in candidates
        if candidate.jam_plan == "i-j-share-data"]
    if (len(candidates), len(none_candidates), len(jam_candidates)) != \
            (10, 8, 2):
        errors.append(
            "explicit-jam enumeration expected total/none/jam 10/8/2, got "
            f"{len(candidates)}/{len(none_candidates)}/{len(jam_candidates)}")
    if set(Candidate.__dataclass_fields__) != {"split", "order", "jam_plan"}:
        errors.append("candidate representation still exposes a tile axis")
    if any(candidate.jam_plan != "none"
           and candidate.factors("i")[1] <= 1 for candidate in candidates):
        errors.append("enumeration admitted jam without outer unroll")
    if any(candidate.jam_plan != "none"
           and candidate_order(spec, candidate) != ("i", "j")
           for candidate in candidates):
        errors.append("enumeration admitted jam with the inner loop outside")

    exact_target = make_target(capacity_bytes=24)
    under_target = make_target(capacity_bytes=23)
    exact = _layout_whole_memory_plan(
        exact_target, _synthetic_memory_planner(
            spec, Candidate(split_u1), exact_target))
    under = _layout_whole_memory_plan(
        under_target, _synthetic_memory_planner(
            spec, Candidate(split_u1), under_target))
    exact_resident, exact_stream = exact.buffers
    under_resident, under_stream = under.buffers
    if (exact.fallback, exact.capacity_bytes_used,
            exact.proposed_capacity_bytes,
            exact_resident.placement, exact_stream.placement) != \
            (False, 24, 24, "resident_shared", "direct"):
        errors.append(f"exact-fit whole working set mismatch: {exact}")
    if (under.fallback, under.capacity_bytes_used,
            under.proposed_capacity_bytes,
            under_resident.placement, under_stream.placement) != \
            (True, 0, 24, "direct-fallback", "direct"):
        errors.append(f"one-byte-under fallback mismatch: {under}")

    fallback_result = evaluate_candidate(
        spec, Candidate(split_u1), parse_config("6x6"),
        schedule=False, target=under_target)
    if (not fallback_result.memory_plan.fallback
            or fallback_result.preload_scalar_elements != 0
            or fallback_result.preload_load_ops != 0
            or fallback_result.preload_spad_write_ops != 0
            or fallback_result.scratchpad_reads != 0
            or fallback_result.spad_port_lb != 0):
        errors.append(
            "direct fallback must be legal with zero preload/scratchpad traffic")

    one_port = scratchpad_port_metrics(
        5, 3, make_target(load_ports=1, store_ports=1))
    wider = scratchpad_port_metrics(
        5, 3, make_target(load_ports=2, store_ports=3))
    slower = scratchpad_port_metrics(
        5, 3, make_target(load_ports=2, store_ports=3, access_cycles=2))
    store_narrow = scratchpad_port_metrics(
        1, 5, make_target(load_ports=4, store_ports=1))
    store_wide = scratchpad_port_metrics(
        1, 5, make_target(load_ports=4, store_ports=5))
    if (one_port.port_lb, wider.port_lb, slower.port_lb) != (5, 3, 6):
        errors.append(
            "load-port count or access latency did not affect port pressure")
    if (store_narrow.port_lb, store_wide.port_lb) != (5, 1):
        errors.append("store-port count did not affect port pressure")
    gemv_probe = Candidate((("i", 1, 1), ("j", 1, 1)))
    gemv_one_port = evaluate_candidate(
        KERNELS["gemv"], gemv_probe, parse_config("6x6"),
        schedule=False, target=make_target(load_ports=1))
    gemv_two_ports = evaluate_candidate(
        KERNELS["gemv"], gemv_probe, parse_config("6x6"),
        schedule=False, target=make_target(load_ports=2))
    gemv_slow_port = evaluate_candidate(
        KERNELS["gemv"], gemv_probe, parse_config("6x6"),
        schedule=False, target=make_target(load_ports=1, access_cycles=2))
    if not (gemv_two_ports.pragma_exposure_aggregate
            < gemv_one_port.pragma_exposure_aggregate
            < gemv_slow_port.pragma_exposure_aggregate):
        errors.append(
            "scratchpad load-port count/latency did not affect the candidate "
            "estimate")
    if len({
            gemv_one_port.scratchpad_reads,
            gemv_two_ports.scratchpad_reads,
            gemv_slow_port.scratchpad_reads}) != 1:
        errors.append("target ports changed logical GEMV scratchpad traffic")

    target = AnalyticTargetSpec()
    same_step = (("j", 0),)
    packed_fanout = pack_scratchpad_accesses((
        ScratchpadAccess("x", 4, same_step, 0, "x_j"),
        ScratchpadAccess("x", 4, same_step, 0, "x_j"),
    ), target)
    if len(packed_fanout) != 1 or packed_fanout[0].logical_elements != (4,):
        errors.append("same-address/same-step scratchpad reads must fan out")
    packed_steps = pack_scratchpad_accesses((
        ScratchpadAccess("x", 4, (("j", 0),), 0, "x_j"),
        ScratchpadAccess("x", 4, (("j", 1),), 0, "x_j"),
    ), target)
    if len(packed_steps) != 2:
        errors.append("same address at different logical steps must stay separate")
    packed_stream = pack_scratchpad_accesses(tuple(
        ScratchpadAccess("x", element, same_step, 0, "x_j")
        for element in range(4)), target)
    if len(packed_stream) != 1 or \
            packed_stream[0].logical_elements != (0, 1, 2, 3):
        errors.append("one declared contiguous stream must pack at fixed V=4")
    unpacked_stream = pack_scratchpad_accesses(tuple(
        ScratchpadAccess("x", element, same_step)
        for element in range(4)), target)
    if len(unpacked_stream) != 4:
        errors.append("contiguous accesses without a stream must stay scalar")
    split_streams = pack_scratchpad_accesses((
        ScratchpadAccess("x", 0, same_step, 0, "left"),
        ScratchpadAccess("x", 1, same_step, 0, "right"),
    ), target)
    if len(split_streams) != 2:
        errors.append("different streams must not coalesce")

    report_target = make_target(
        capacity_bytes=24, load_ports=2, store_ports=3, access_cycles=2)
    if report_target.name != "shared-spad-24b-r2w3-a2-v4":
        errors.append(f"target identity mismatch: {report_target.name}")
    outcome = search(
        spec, parse_config("6x6"), 2, 2, 4, jobs=1,
        target=report_target)
    if outcome.target_profile != report_target:
        errors.append("extended search lost target identity")
    for result in outcome.results:
        if not (outcome.absolute_lb <= result.plan_cgra_lb
                <= result.pragma_exposure_aggregate):
            errors.append("extended aggregate bracket failed")
            break
    rec = outcome.recommendation
    structure_before = rec.schedule_structure_key
    _ensure_scheduled(spec, rec, parse_config("6x6"), report_target)
    if not (outcome.absolute_lb <= rec.plan_cgra_lb
            <= rec.pragma_exposure_aggregate <= rec.schedule_estimate):
        errors.append("extended scheduled four-term bracket failed")
    if rec.schedule_structure_key != structure_before:
        errors.append("lazy scheduling changed extended structure identity")

    report, report_rec, report_lb = run(
        spec, parse_config("6x6"), 2, 2, 4, top=2, jobs=1,
        target=report_target)
    required_report_text = (
        report_target.name,
        "R=2, W=3",
        "2-cycle non-pipelined access",
        "plan_lb",
        "spad lb/s",
        "Jam:",
        "Capacity:",
        "Scratchpad ports:",
        "fallback=no",
        "bounded_search_floor=",
        "BEST BOUNDED:",
        "flags: B=best row in this bounded diagnostic",
    )
    if (any(text not in report for text in required_report_text)
            or "absolute_cgra_lb=" in report
            or "RECOMMENDED:" in report
            or report_lb != outcome.absolute_lb
            or report_rec.target_profile != report_target
            or report_rec.schedule_estimate is None):
        errors.append("extended report omitted target, jam, capacity, or port data")
    forbidden = ("ideal-dma", "bank_", "tile")
    if any(text in report.lower() for text in forbidden):
        errors.append("extended report retained a removed tile/bank/DMA concept")
    fallback_report, fallback_rec, _ = run(
        KERNELS["batchnorm"],
        parse_config("P=100000,L=100000,S=100000"),
        max_parallel=1, max_unroll=1, exposure_cap=1,
        top=1, jobs=1, target=target)
    if (fallback_rec.saturation != "latency-bound"
            or "BEST BOUNDED:" not in fallback_report
            or "best-estimate fallback" not in fallback_report
            or "no eligible family knee" not in fallback_report):
        errors.append(
            "all-latency extended report must identify its best-estimate "
            "fallback")


def _read_smoke_uint_constants(
        kernel: str, names: tuple[str, ...]) -> dict[str, int]:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    main_path = os.path.join(
        repo_root, "tests", "app", kernel, "main.cpp")
    with open(main_path, encoding="utf-8") as stream:
        source = stream.read()
    declarations = dict(re.findall(
        r"^\s*const\s+uint32_t\s+([A-Za-z_][A-Za-z0-9_]*)"
        r"\s*=\s*([0-9]+)\s*;",
        source, flags=re.MULTILINE))
    missing = [name for name in names if name not in declarations]
    if missing:
        raise ValueError(
            f"{main_path} lacks literal uint32_t fixture constants: "
            f"{', '.join(missing)}")
    return {name: int(declarations[name]) for name in names}


def _run_smoke_fixture_tests(errors: list[str]) -> None:
    try:
        batch = _read_smoke_uint_constants(
            "batchnorm", ("C", "H", "W"))
        gemv_dims = _read_smoke_uint_constants("gemv", ("M", "N"))
        conv = _read_smoke_uint_constants(
            "conv2d", ("C_in", "C_out", "H", "W", "KH", "KW",
                       "stride_h", "stride_w"))
    except (OSError, ValueError) as error:
        errors.append(f"smoke-fixture source audit failed: {error}")
        return

    batchnorm = KERNELS["batchnorm"]
    batch_trips = tuple(
        batchnorm.level(name).trip for name in ("c", "h", "w"))
    batch_expected = (batch["C"], batch["H"], batch["W"])
    if batch_trips != batch_expected:
        errors.append(
            "Batchnorm DSE trips do not match main.cpp C/H/W: "
            f"{batch_trips} != {batch_expected}")
    batch_candidate = Candidate((
        ("c", 1, 1), ("h", 1, 1), ("w", 1, 1)))
    batch_buffers = {
        buffer.name: len(buffer.elements)
        for buffer in derive_memory_plan(
            batchnorm, batch_candidate, AnalyticTargetSpec()).buffers}
    batch_elements = math.prod(batch_expected)
    if batch_buffers != {"input": batch_elements, "output": batch_elements}:
        errors.append(
            "Batchnorm DSE buffers do not match the main.cpp element count: "
            f"{batch_buffers}")

    gemv = KERNELS["gemv"]
    gemv_trips = (gemv.level("i").trip, gemv.level("j").trip)
    gemv_expected = (gemv_dims["M"], gemv_dims["N"])
    if gemv_trips != gemv_expected:
        errors.append(
            "GEMV DSE trips do not match main.cpp M/N: "
            f"{gemv_trips} != {gemv_expected}")
    gemv_candidate = Candidate((("i", 1, 1), ("j", 1, 1)))
    gemv_buffers = {
        buffer.name: len(buffer.elements)
        for buffer in derive_memory_plan(
            gemv, gemv_candidate, AnalyticTargetSpec()).buffers}
    m, n = gemv_expected
    expected_gemv_buffers = {
        "x": n, "A": m * n, "input_y": m, "output_y": m}
    if gemv_buffers != expected_gemv_buffers:
        errors.append(
            "GEMV DSE buffers do not match main.cpp M/N: "
            f"{gemv_buffers} != {expected_gemv_buffers}")

    oh = (conv["H"] - conv["KH"]) // conv["stride_h"] + 1
    ow = (conv["W"] - conv["KW"]) // conv["stride_w"] + 1
    tap = conv["C_in"] * conv["KH"] * conv["KW"]
    conv2d = KERNELS["conv2d"]
    conv_trips = tuple(
        conv2d.level(name).trip for name in ("co", "oh", "ow", "tap"))
    conv_expected = (conv["C_out"], oh, ow, tap)
    if conv_trips != conv_expected:
        errors.append(
            "Conv2d DSE trips do not match main.cpp dimensions: "
            f"{conv_trips} != {conv_expected}")
    input_set, weight_set, output_set = _conv2d_whole_address_sets()
    conv_buffer_sizes = (
        len(input_set), len(weight_set), len(output_set))
    expected_conv_buffer_sizes = (
        conv["C_in"] * conv["H"] * conv["W"],
        conv["C_out"] * tap,
        conv["C_out"] * oh * ow)
    if conv_buffer_sizes != expected_conv_buffer_sizes:
        errors.append(
            "Conv2d DSE address sets do not match main.cpp dimensions: "
            f"{conv_buffer_sizes} != {expected_conv_buffer_sizes}")


def _run_extended_pilot_tests(errors: list[str]) -> None:
    cfg = parse_config("6x6")
    target = AnalyticTargetSpec()

    def check_scheduled_bracket(
            label: str, spec: KernelSpec, candidate: Candidate,
            local_target: AnalyticTargetSpec) -> None:
        result = evaluate_candidate(
            spec, candidate, cfg, schedule=False, target=local_target)
        result.absolute_cgra_lb = result.plan_cgra_lb
        _ensure_scheduled(spec, result, cfg, local_target)
        if not (result.plan_cgra_lb <= result.pragma_exposure_aggregate
                <= result.schedule_estimate):
            errors.append(f"{label}: focused scheduled bracket failed")

    batchnorm = KERNELS["batchnorm"]
    if _validated_orders(batchnorm) != (
            ("c", "h", "w"), ("c", "w", "h")):
        errors.append("Batchnorm must expose exactly its two declared orders")
    batch_split = (("c", 1, 1), ("h", 1, 1), ("w", 1, 4))
    batch_source = evaluate_candidate(
        batchnorm, Candidate(batch_split), cfg, schedule=False, target=target)
    batch_interchanged = evaluate_candidate(
        batchnorm, Candidate(batch_split, ("c", "w", "h")), cfg,
        schedule=False, target=target)
    if not (batch_source.LD < batch_interchanged.LD
            and batch_source.ST < batch_interchanged.ST):
        errors.append("Batchnorm may coalesce w only when w is innermost")
    if (batch_source.jam_plan.name != "none"
            or any(buffer.placement != "direct"
                   for buffer in batch_source.memory_plan.buffers)
            or batch_source.preload_scalar_elements != 0
            or batch_source.scratchpad_reads != 0):
        errors.append("Batchnorm must remain unjammed and direct-memory")
    batch_unrolled = derive_jam_plan(
        batchnorm, Candidate(
            (("c", 1, 2), ("h", 1, 1), ("w", 1, 1))))
    if batch_unrolled.edges:
        errors.append("Batchnorm outer unroll must not imply jam")
    check_scheduled_bracket(
        "Batchnorm", batchnorm, Candidate(batch_split), target)

    gemv = KERNELS["gemv"]
    gemv_split = (("i", 1, 2), ("j", 1, 1))
    gemv_none = Candidate(gemv_split)
    gemv_jam = Candidate(gemv_split, jam_plan="i-j-share-x")
    gemv_none_result = evaluate_candidate(
        gemv, gemv_none, cfg, schedule=False, target=target)
    gemv_jam_result = evaluate_candidate(
        gemv, gemv_jam, cfg, schedule=False, target=target)
    gemv_rows = gemv.level("i").trip
    gemv_columns = gemv.level("j").trip
    gemv_jam_waves = _ceil_div(gemv_rows, 2)
    expected_unjammed_reads = gemv_rows * gemv_columns
    expected_jammed_reads = gemv_jam_waves * gemv_columns
    x_buffer = next(buffer for buffer in gemv_jam_result.memory_plan.buffers
                    if buffer.name == "x")
    if (x_buffer.placement, gemv_jam_result.preload_scalar_elements,
            gemv_none_result.scratchpad_reads,
            gemv_jam_result.scratchpad_reads) != \
            ("resident_shared", gemv_columns,
             expected_unjammed_reads, expected_jammed_reads):
        errors.append(
            "GEMV explicit jam resident traffic mismatch: "
            f"{x_buffer.placement}, {gemv_jam_result.preload_scalar_elements}, "
            f"{gemv_none_result.scratchpad_reads}, "
            f"{gemv_jam_result.scratchpad_reads}")
    if (gemv_none_result.A, gemv_none_result.ST, gemv_none_result.CP) != \
            (gemv_jam_result.A, gemv_jam_result.ST, gemv_jam_result.CP):
        errors.append("GEMV jam changed arithmetic, stores, or critical path")
    if gemv_none_result.scratchpad_reads - \
            gemv_jam_result.scratchpad_reads != \
            expected_unjammed_reads - expected_jammed_reads:
        errors.append("GEMV i->j jam did not remove the expected x readers")
    gemv_outcome = search(gemv, cfg, jobs=1, target=target)
    if (gemv_outcome.recommendation.cand.jam_plan != "i-j-share-x"
            or gemv_outcome.recommendation.cand.factors("i") != (1, 4)
            or gemv_outcome.recommendation.pragma_exposure_aggregate != 94):
        errors.append(
            "GEMV family-knee selection must prefer i:P1U4 explicit jam at "
            "p_agg=94 for the M=32, N=48 smoke fixture")

    gemv_capacity = gemv_columns * 4
    gemv_exact = derive_memory_plan(
        gemv, gemv_none, make_target(capacity_bytes=gemv_capacity))
    gemv_under_target = make_target(capacity_bytes=gemv_capacity - 1)
    gemv_under = evaluate_candidate(
        gemv, gemv_jam, cfg, schedule=False, target=gemv_under_target)
    if (gemv_exact.fallback, gemv_exact.capacity_bytes_used,
            gemv_exact.proposed_capacity_bytes) != \
            (False, gemv_capacity, gemv_capacity):
        errors.append(f"GEMV exact-capacity placement mismatch: {gemv_exact}")
    if (not gemv_under.memory_plan.fallback
            or gemv_under.memory_plan.proposed_capacity_bytes != gemv_capacity
            or gemv_under.preload_scalar_elements != 0
            or gemv_under.scratchpad_reads != 0
            or gemv_under.spad_port_lb != 0):
        errors.append("GEMV capacity overflow must use direct fallback")
    check_scheduled_bracket("GEMV", gemv, gemv_jam, target)

    conv2d = KERNELS["conv2d"]
    orders = _validated_orders(conv2d)
    if len(orders) != 6 or any(order[-1] != "tap" for order in orders):
        errors.append("Conv2d must expose six legal orders with tap last")
    conv_split = (("co", 1, 1), ("oh", 1, 1),
                  ("ow", 1, 1), ("tap", 1, 1))
    conv_none = Candidate(conv_split)
    input_set, weight_set, output_set = _conv2d_whole_address_sets()
    if (len(input_set), len(weight_set), len(output_set)) != (192, 108, 144):
        errors.append("Conv2d whole-kernel address sets must be 192/108/144")

    conv_exact_target = make_target(capacity_bytes=1200)
    conv_exact = derive_memory_plan(conv2d, conv_none, conv_exact_target)
    input_buffer, weight_buffer, output_buffer = conv_exact.buffers
    if (conv_exact.fallback, conv_exact.capacity_bytes_used,
            conv_exact.proposed_capacity_bytes,
            input_buffer.placement, weight_buffer.placement,
            output_buffer.placement) != \
            (False, 1200, 1200, "resident_shared",
             "resident_shared", "direct"):
        errors.append(f"Conv2d exact-capacity placement mismatch: {conv_exact}")

    conv_result = evaluate_candidate(
        conv2d, conv_none, cfg, schedule=False, target=target)
    if (conv_result.preload_scalar_elements,
            conv_result.scratchpad_reads,
            conv_result.avoided_direct_loads) != (300, 7776, 7476):
        errors.append(
            "Conv2d resident traffic expected 300/7776/7476, got "
            f"{conv_result.preload_scalar_elements}/"
            f"{conv_result.scratchpad_reads}/"
            f"{conv_result.avoided_direct_loads}")
    port3_target = make_target(load_ports=3)
    port3_plan = conv2d.extended_plan_builder(
        conv2d, conv_none, cfg, port3_target, False)
    port3_read_ops = sum(
        len(wave.spad_read_accesses)
        for wave in port3_plan.execution.compute_waves)
    port3_wave_ceilings = sum(
        _ceil_div(len(wave.spad_read_accesses), port3_target.load_ports)
        for wave in port3_plan.execution.compute_waves)
    port3_result = evaluate_candidate(
        conv2d, conv_none, cfg, schedule=False, target=port3_target)
    expected_port3_demand = _ceil_div(
        port3_read_ops, port3_target.load_ports)
    if (port3_result.recurring_demand[3] != expected_port3_demand
            or not port3_wave_ceilings > expected_port3_demand):
        errors.append(
            "recurring scratchpad demand must exclude per-wave ceiling rounding")

    conv_under_target = make_target(capacity_bytes=1199)
    conv_under = evaluate_candidate(
        conv2d, conv_none, cfg, schedule=False, target=conv_under_target)
    if (not conv_under.memory_plan.fallback
            or conv_under.memory_plan.proposed_capacity_bytes != 1200
            or conv_under.capacity_bytes_used != 0
            or conv_under.preload_scalar_elements != 0
            or conv_under.scratchpad_reads != 0
            or conv_under.spad_port_lb != 0):
        errors.append("Conv2d capacity overflow must use direct fallback")

    conv_jam_split = (("co", 1, 2), ("oh", 1, 1),
                      ("ow", 1, 1), ("tap", 1, 1))
    conv_unjammed = evaluate_candidate(
        conv2d, Candidate(conv_jam_split), cfg,
        schedule=False, target=target)
    conv_jammed = evaluate_candidate(
        conv2d, Candidate(
            conv_jam_split, jam_plan="share-input"),
        cfg, schedule=False, target=target)
    if (conv_unjammed.A, conv_unjammed.ST, conv_unjammed.CP) != \
            (conv_jammed.A, conv_jammed.ST, conv_jammed.CP):
        errors.append("Conv2d jam changed arithmetic, stores, or critical path")
    if not conv_jammed.scratchpad_reads < conv_unjammed.scratchpad_reads:
        errors.append("Conv2d share-input jam did not reduce input readers")

    source_ow = Candidate((
        ("co", 1, 1), ("oh", 1, 1),
        ("ow", 1, 4), ("tap", 1, 1)))
    non_ow_inner = Candidate(
        source_ow.split, ("co", "ow", "oh", "tap"))
    source_result = evaluate_candidate(
        conv2d, source_ow, cfg, schedule=False, target=target)
    non_ow_result = evaluate_candidate(
        conv2d, non_ow_inner, cfg, schedule=False, target=target)
    if not source_result.ST < non_ow_result.ST:
        errors.append(
            "Conv2d output stores may coalesce only with innermost ow")

    conv_candidates = enumerate_candidates(conv2d)
    conv_none_count = sum(
        candidate.jam_plan == "none" for candidate in conv_candidates)
    if (conv_none_count, len(conv_candidates)) != (1296, 4374):
        errors.append(
            "Conv2d no-tile candidate count expected 1296 before jam variants "
            f"and 4374 total, got {conv_none_count}/{len(conv_candidates)}")
    conv_default_candidate = Candidate(
        (("co", 1, 4), ("oh", 1, 4),
         ("ow", 1, 4), ("tap", 1, 1)),
        order=("co", "oh", "ow", "tap"), jam_plan="share-all")
    conv_default_result = evaluate_candidate(
        conv2d, conv_default_candidate, cfg, schedule=False, target=target)
    if (conv_default_result.plan_cgra_lb,
            conv_default_result.pragma_exposure_aggregate) != (256, 265):
        errors.append(
            "Conv2d default 2R/2W share-all candidate expected "
            "plan_lb/p_agg=256/265")
    scheduled_conv = Candidate((
        ("co", 1, 4), ("oh", 1, 4),
        ("ow", 1, 4), ("tap", 1, 1)))
    check_scheduled_bracket(
        "Conv2d", conv2d, scheduled_conv, target)


def _run_self_tests() -> int:
    errors: list[str] = []
    cfg = parse_config("6x6")

    _run_smoke_fixture_tests(errors)
    _run_extended_infrastructure_tests(errors)
    _run_extended_pilot_tests(errors)

    cfg4 = parse_config("4x4")
    cfg8 = parse_config("8x8")
    if _config_key(cfg4) != (16, 8, 8):
        errors.append(f"4x4 config mapped to {_config_key(cfg4)}")
    if _config_key(cfg8) != (64, 16, 16):
        errors.append(f"8x8 config mapped to {_config_key(cfg8)}")
    normalized = _normalize_brief_configs(
        cfg, [cfg4, parse_config("P=16,L=8,S=8"), cfg, cfg8, cfg4])
    if [brief.label for brief in normalized] != ["4x4", "8x8"]:
        errors.append(
            "brief config normalization must preserve first-seen unique labels")

    alternate_expected = {
        "axpy": {"4x4": "i:P1U32", "8x8": "i:P1U128"},
        "vecsum": {"4x4": "i:P1U1", "8x8": "i:P1U1"},
    }
    for kernel, expected_by_config in alternate_expected.items():
        for alt_cfg in (cfg4, cfg8):
            outcome = search(KERNELS[kernel], alt_cfg, jobs=1)
            got = outcome.recommendation.cand.signature()
            expected = expected_by_config[alt_cfg.label]
            if got != expected:
                errors.append(
                    f"{kernel} {alt_cfg.label}: expected {expected}, got {got}")

    brief_report, _, _ = run(
        KERNELS["axpy"], cfg, top=1, jobs=1,
        brief_configs=_normalize_brief_configs(cfg, [cfg4, cfg8]))
    plain_report, _, _ = run(KERNELS["axpy"], cfg, top=1, jobs=1)
    for expected_line in (
            "4x4 recommendation: i:P1U32.",
            "8x8 recommendation: i:P1U128."):
        if expected_line not in brief_report:
            errors.append(f"brief report omitted {expected_line!r}")
        if expected_line in plain_report:
            errors.append(f"plain report unexpectedly included {expected_line!r}")
    if brief_report.count("flags    split") != 1:
        errors.append("multi-config report must contain exactly one detailed table")

    # Search completeness: uncapped enumeration covers every power-of-two factor
    # through the concrete trip count. Caps remain available only when explicitly
    # requested.
    search_spec = KernelSpec(
        name="search_test", levels=(Level("i", 5, "parallel"),),
        build_chunk=_axpy_chunk)
    uncapped = enumerate_candidates(search_spec)
    uncapped_splits = {cand.split for cand in uncapped}
    for expected in ((('i', 4, 1),), (('i', 1, 4),), (('i', 2, 2),)):
        if expected not in uncapped_splits:
            errors.append(f"uncapped search omitted legal factor {expected}")
    for excluded in ((('i', 3, 1),), (('i', 1, 5),)):
        if excluded in uncapped_splits:
            errors.append(f"power-of-two search included {excluded}")
    if any(p * u > 5 for cand in uncapped for _, p, u in cand.split):
        errors.append("uncapped search exceeded the concrete trip count")
    capped = enumerate_candidates(search_spec, max_parallel=2, max_unroll=2,
                                  exposure_cap=3)
    if any(p > 2 or u > 2 or p * u > 3
           for cand in capped for _, p, u in cand.split):
        errors.append("explicit diagnostic caps were not enforced")

    # axpy: the vector-coalescing distinction must make UNROLL beat PARALLEL at a
    # fixed product in the U<V regime. Build P4U1 (strided, no coalesce) vs P1U4
    # (one worker, 4 adjacent -> 1 vector op).
    spec = KERNELS["axpy"]
    p4 = evaluate_candidate(spec, Candidate((("i", 4, 1),)), cfg)
    u4 = evaluate_candidate(spec, Candidate((("i", 1, 4),)), cfg)
    if not (u4.LD < p4.LD):
        errors.append(f"axpy P1U4 LD {u4.LD} !< P4U1 LD {p4.LD} "
                      "(unroll must coalesce contiguous loads)")
    if not (u4.pragma_exposure_aggregate <= p4.pragma_exposure_aggregate):
        errors.append(
            f"axpy P1U4 p_agg {u4.pragma_exposure_aggregate} > P4U1 "
            f"{p4.pragma_exposure_aggregate} (unroll must not lose)")
    # Control amortization (mentor Sihao): at a fixed product, UNROLL amortizes the
    # iterator (one advance per worker/wave) while PARALLEL keeps one iterator per
    # worker, so P1U4 must charge strictly fewer arithmetic ops than P4U1.
    if not (u4.A < p4.A):
        errors.append(f"axpy P1U4 A {u4.A} !< P4U1 A {p4.A} "
                      "(unroll must amortize control ops)")
    # The ALGORITHMIC critical path is still a global pool (control compares sit off
    # it), so CP must match across the split.
    if p4.CP != u4.CP:
        errors.append("axpy P4U1 vs P1U4 CP must match (algorithmic CP is a pool)")

    # Once U >= V=4, further unroll coalesces fully -> P8U1 and P1U8 both fully
    # coalesce at product 8, so they tie on LD (bounded distinction).
    p8 = evaluate_candidate(spec, Candidate((("i", 8, 1),)), cfg)
    u8 = evaluate_candidate(spec, Candidate((("i", 1, 8),)), cfg)
    # P1U8 coalesces 8->2 vec; P8U1 strides 8 scalar -> P1U8 still <= P8U1.
    if not (u8.LD <= p8.LD):
        errors.append(f"axpy P1U8 LD {u8.LD} !<= P8U1 LD {p8.LD}")

    # bracket: every candidate sits at or above the (vector-aware) floor.
    lb, _ = absolute_cgra_lb(spec, cfg)
    for name in ("axpy",):
        ks = KERNELS[name]
        klb, _ = absolute_cgra_lb(ks, cfg)
        for c in enumerate_candidates(ks, 8, 8, 256):
            r = evaluate_candidate(ks, c, cfg)
            if not (klb <= r.pragma_exposure_aggregate <= r.schedule_estimate):
                errors.append(
                    f"{name} {c.signature()}: bracket violated lb={klb} "
                    f"pragma={r.pragma_exposure_aggregate} "
                    f"sched={r.schedule_estimate}")
                break

    # batchnorm: unroll-on-w must beat parallel-on-w (input/output contiguous
    # over w); c/h are strided and give no coalescing edge.
    bn = KERNELS["batchnorm"]
    wpar = evaluate_candidate(bn, Candidate(
        (("c", 1, 1), ("h", 1, 1), ("w", 4, 1))), cfg)
    wunr = evaluate_candidate(bn, Candidate(
        (("c", 1, 1), ("h", 1, 1), ("w", 1, 4))), cfg)
    if not (wunr.LD < wpar.LD):
        errors.append(f"batchnorm w-unroll LD {wunr.LD} !< w-parallel {wpar.LD}")

    # vecsum: reduction fully consumed + contiguous -> P and U symmetric.
    vs = KERNELS["vecsum"]
    v_p = evaluate_candidate(vs, Candidate((("i", 8, 1),)), cfg)
    v_u = evaluate_candidate(vs, Candidate((("i", 1, 8),)), cfg)
    if (v_p.LD, v_p.ST, v_p.A, v_p.CP, v_p.pragma_exposure_aggregate) != \
       (v_u.LD, v_u.ST, v_u.A, v_u.CP, v_u.pragma_exposure_aggregate):
        errors.append("vecsum P8U1 vs P1U8 must be identical (reduction is "
                      "fully consumed and contiguous -> P/U-symmetric)")

    # tridiag_solve: sequential -> p forced to 1, CP-bound, no distinction.
    t = KERNELS["tridiag_solve"]
    tcands = enumerate_candidates(t, 8, 8, 256)
    if any(p > 1 for c in tcands for _, p, _ in c.split):
        errors.append("tridiag_solve must not enumerate any parallel factor > 1")
    tu1 = evaluate_candidate(t, Candidate((("i", 1, 1),)), cfg)
    tu8 = evaluate_candidate(t, Candidate((("i", 1, 8),)), cfg)
    if tu1.pragma_exposure_aggregate != tu8.pragma_exposure_aggregate:
        errors.append("tridiag_solve U1 vs U8 must be identical (no distinction)")
    tlb, tdem = absolute_cgra_lb(t, cfg)
    if tdem["CP"] != tlb:
        errors.append("tridiag_solve floor must be CP-bound (serial recurrence)")

    # bisection_step: axpy-shaped single parallel loop. Unroll must coalesce and
    # amortize control -> at a fixed product, P1U4 beats P4U1 on loads and A.
    bs = KERNELS["bisection_step"]
    bs_p4 = evaluate_candidate(bs, Candidate((("i", 4, 1),)), cfg)
    bs_u4 = evaluate_candidate(bs, Candidate((("i", 1, 4),)), cfg)
    if not (bs_u4.LD < bs_p4.LD):
        errors.append(f"bisection_step P1U4 LD {bs_u4.LD} !< P4U1 LD {bs_p4.LD}")
    if not (bs_u4.A < bs_p4.A):
        errors.append(f"bisection_step P1U4 A {bs_u4.A} !< P4U1 A {bs_p4.A}")

    # autocorrelation: gemv-shaped. Inner i reduction is P/U-symmetric; the outer
    # lag level gives a (modest) unroll edge on loads at a fixed product. Floor
    # must be compute-bound (large tree of products).
    ac = KERNELS["autocorrelation"]
    ac_lb, ac_dem = absolute_cgra_lb(ac, cfg)
    if ac_dem["compute"] != ac_lb:
        errors.append("autocorrelation floor must be compute-bound")
    ac_p4 = evaluate_candidate(ac, Candidate((("lag", 4, 1), ("i", 1, 1))), cfg)
    ac_u4 = evaluate_candidate(ac, Candidate((("lag", 1, 4), ("i", 1, 1))), cfg)
    if not (ac_u4.LD <= ac_p4.LD):
        errors.append(f"autocorrelation lag-unroll LD {ac_u4.LD} !<= "
                      f"lag-parallel {ac_p4.LD}")

    # bit_reverse: inner bit loop is sequential (p forced to 1). Threading the
    # carried result/value as dataflow leaves the 4 bitops/bit as a global pool ->
    # compute-bound.
    br = KERNELS["bit_reverse"]
    br_cands = enumerate_candidates(br, 8, 8, 256)
    if any(n == "bit" and p > 1 for c in br_cands for n, p, _ in c.split):
        errors.append("bit_reverse: sequential bit level must not parallelize")
    br_lb, br_dem = absolute_cgra_lb(br, cfg)
    if br_dem["compute"] != br_lb:
        errors.append("bit_reverse floor must be compute-bound (bitop pool)")

    # bitonic family: preserve the corrected loop-invariant half_block count,
    # exact full-trip DSE totals, and the sequential legality of both variants.
    bitonic_expected = {
        "bitonic_stage": (11, 66, 9, 12, 5),
        "bitonic_stage-modified": (31, 133, 52, 55, 48),
        "bitonic_stage-tweak": (17, 92, 28, 31, 24),
    }
    for name, expected in bitonic_expected.items():
        _lb, demand = absolute_cgra_lb(KERNELS[name], cfg)
        got = (demand["CP"], demand["A"], demand["LD"],
               demand["LD_eff"], demand["ST"])
        if got != expected:
            errors.append(f"{name}: full-trip totals {got} != {expected}")
    for name in ("bitonic_stage-modified", "bitonic_stage-tweak"):
        cands = enumerate_candidates(KERNELS[name], 8, 8, 256)
        if any(p > 1 for c in cands for _, p, _ in c.split):
            errors.append(f"{name}: sequential i level must not parallelize")

    new_expected = {
        "clz": (163, 13612, 6997, 6998, 6997),
        "col2im": (13, 12756, 1945, 1952, 1165),
        "crc32": (7175, 8706, 3585, 3585, 2563),
        "edge_update": (6, 40, 38, 38, 37),
        "fft_butterfly": (71, 701, 252, 253, 302),
        "gauss_seidel_step": (198, 3136, 527, 536, 64),
        "gather": (4, 1026, 1281, 1283, 257),
        "hist_bin": (17, 6148, 2305, 2309, 2052),
    }
    for name, expected in new_expected.items():
        _lb, demand = absolute_cgra_lb(KERNELS[name], cfg)
        got = (demand["CP"], demand["A"], demand["LD"],
               demand["LD_eff"], demand["ST"])
        if got != expected:
            errors.append(f"{name}: full-trip totals {got} != {expected}")
    for name in ("crc32", "edge_update", "gauss_seidel_step"):
        cands = enumerate_candidates(KERNELS[name], 8, 8, 256)
        if any(p > 1 for c in cands for _, p, _ in c.split):
            errors.append(f"{name}: sequential level must not parallelize")

    # fft_butterfly: copy waves are local to the tunable copy phase. The four
    # stage regions execute once, in order, and retain the validated sequential
    # j/twiddle recurrence depths instead of being repeated per copy wave.
    fft = KERNELS["fft_butterfly"]
    fft_u16_cand = Candidate((("copy_i", 1, 16),))
    fft_u16_dag = fft.build_chunk(fft_u16_cand)
    fft_names = [region.name for region in fft_u16_dag.regions]
    if fft_names != ["copy", "s=1", "s=2", "s=3", "s=4"]:
        errors.append(f"fft_butterfly: ordered regions {fft_names}")
    fft_cps = [region_aggregate(region, cfg).CP
               for region in fft_u16_dag.regions]
    if fft_cps != [2, 8, 11, 17, 33]:
        errors.append(f"fft_butterfly: ordered-region CPs {fft_cps}")
    fft_u8 = evaluate_candidate(
        fft, Candidate((("copy_i", 1, 8),)), cfg)
    fft_u16 = evaluate_candidate(fft, fft_u16_cand, cfg)
    if (fft_u8.waves, fft_u8.pragma_exposure_aggregate,
            fft_u8.schedule_estimate) != (2, 73, 78):
        errors.append(
            "fft_butterfly: P1U8 expected two copy waves, p_agg=73, sched=78")
    if (fft_u16.waves, fft_u16.pragma_exposure_aggregate,
            fft_u16.schedule_estimate) != (1, 71, 75):
        errors.append(
            "fft_butterfly: P1U16 expected one copy wave, p_agg=71, sched=75")

    # gauss_seidel_step: the lower-triangle RAW chain keeps the outer loop
    # serial, while the row-local reductions preserve the accepted six-cycle
    # recurrence and 198-cycle full-sweep CP.
    gauss = KERNELS["gauss_seidel_step"]
    gauss_result = evaluate_candidate(
        gauss, Candidate((("i", 1, 1),)), cfg)
    if (gauss_result.pragma_exposure_aggregate,
            gauss_result.schedule_estimate) != (198, 198):
        errors.append(
            "gauss_seidel_step: expected p_agg=sched=198 for the serial sweep")
    gauss_dag = gauss.build_chunk(Candidate((("i", 1, 1),)))
    latest_reads = sum(
        node.kind == "output_x_latest" for node in gauss_dag.regions[0].nodes)
    ready_vectors = sum(
        node.kind == "output_x_ready_vec" for node in gauss_dag.regions[0].nodes)
    if (latest_reads, ready_vectors) != (31, 128):
        errors.append(
            "gauss_seidel_step: expected 31 scalar latest reads and 128 "
            f"ready-prefix vectors, got {(latest_reads, ready_vectors)}")

    # hist_bin: zero-fill waves are local to the annotated zero_i phase and the
    # count region executes once after them. P1U8 must include the two-element
    # tail, then the concrete count trace takes all 1024 normal paths, no clamp
    # assignments, and exactly 1024 bucket adds over the accepted fan-ins.
    hist = KERNELS["hist_bin"]
    hist_cand = Candidate((("zero_i", 1, 8),))
    hist_dag = hist.build_chunk(hist_cand)
    hist_names = [region.name for region in hist_dag.regions]
    if hist_names != ["zero_fill", "zero_fill.wave1", "count"]:
        errors.append(f"hist_bin: ordered regions {hist_names}")
    hist_kinds = [node.kind for region in hist_dag.regions for node in region.nodes]
    if hist_kinds.count("bucket_add") != 1024:
        errors.append(
            f"hist_bin: bucket adds {hist_kinds.count('bucket_add')} != 1024")
    if "clamp_store" in hist_kinds:
        errors.append("hist_bin: concrete fixture must not take the clamp arm")
    hist_result = evaluate_candidate(hist, hist_cand, cfg)
    if (hist_result.waves, hist_result.pragma_exposure_aggregate,
            hist_result.schedule_estimate) != (2, 194, 263):
        errors.append(
            "hist_bin: P1U8 expected two zero waves, p_agg=194, sched=263")
    _report, hist_rec, _lb = run(hist, cfg, top=16)
    if hist_rec.cand.signature() != "zero_i:P1U8":
        errors.append(
            f"hist_bin: expected zero_i:P1U8 recommendation, got "
            f"{hist_rec.cand.signature()}")

    # interpolate_linear: full exposure must preserve the concrete 64-query
    # trace while replacing outer-q source induction with one residual iterator.
    interpolate = KERNELS["interpolate_linear"]
    _lb, demand = absolute_cgra_lb(interpolate, cfg)
    expected = (289, 5573, 3410, 3412, 1105)
    got = (demand["CP"], demand["A"], demand["LD"],
           demand["LD_eff"], demand["ST"])
    if got != expected:
        errors.append(
            f"interpolate_linear: full-trip totals {got} != {expected}")
    interpolate_p4 = evaluate_candidate(
        interpolate, Candidate((("q", 4, 1),)), cfg)
    interpolate_u4 = evaluate_candidate(
        interpolate, Candidate((("q", 1, 4),)), cfg)
    if not (interpolate_u4.LD < interpolate_p4.LD
            and interpolate_u4.ST < interpolate_p4.ST
            and interpolate_u4.A < interpolate_p4.A):
        errors.append(
            "interpolate_linear: q-unroll must coalesce boundary traffic and "
            "amortize q control relative to q-parallel at fixed exposure")

    # gather: only the contiguous indices/dst streams coalesce. The indirect
    # src[indices[i]] loads stay scalar, while unroll still saves vector slots
    # and iterator work relative to parallel workers at fixed exposure.
    gather = KERNELS["gather"]
    gather_p4 = evaluate_candidate(
        gather, Candidate((("i", 4, 1),)), cfg)
    gather_u4 = evaluate_candidate(
        gather, Candidate((("i", 1, 4),)), cfg)
    if not (gather_u4.LD < gather_p4.LD):
        errors.append(
            f"gather P1U4 LD {gather_u4.LD} !< P4U1 LD {gather_p4.LD}")
    if gather_u4.LD != 6:
        errors.append(
            f"gather P1U4 LD {gather_u4.LD} != 6; indirect src loads must stay "
            "scalar")

    # binary_search: COUNTEREXAMPLE. Inner probe loop is sequential (p forced to
    # 1); data-dependent recurrence + tiny M=5 -> CP-bound, no P-vs-U distinction.
    bsr = KERNELS["binary_search"]
    bsr_cands = enumerate_candidates(bsr, 8, 8, 256)
    if any(n == "probe" and p > 1 for c in bsr_cands for n, p, _ in c.split):
        errors.append("binary_search: sequential probe level must not parallelize")
    bsr_lb, bsr_dem = absolute_cgra_lb(bsr, cfg)
    if bsr_dem["CP"] != bsr_lb:
        errors.append("binary_search floor must be CP-bound (serial search)")

    # CLZ regression: an explicit factor-8 diagnostic cap stops at P8U8, while
    # the complete power-of-two search reaches the better P1U64 row.
    clz = KERNELS["clz"]
    _report, clz_bounded, _ = run(clz, cfg, 8, 8, 256, top=1)
    if clz_bounded.cand.signature() != "i:P8U8":
        errors.append(f"clz bounded search expected i:P8U8, got "
                      f"{clz_bounded.cand.signature()}")
    _report, clz_complete, _ = run(
        clz, cfg, top=1, jobs=min(8, os.cpu_count() or 1))
    if clz_complete.cand.signature() != "i:P1U64":
        errors.append(f"clz complete search expected i:P1U64, got "
                      f"{clz_complete.cand.signature()}")
    clz_u64 = evaluate_candidate(clz, Candidate((("i", 1, 64),)), cfg,
                                 schedule=False)
    if clz_u64.pragma_exposure_aggregate != 584:
        errors.append(f"clz P1U64 expected p_agg 584, got "
                      f"{clz_u64.pragma_exposure_aggregate}")

    # every kernel: end-to-end run, a recommendation exists, bracket holds on it.
    for name, ks in KERNELS.items():
        if name in ("batchnorm", "gemv", "conv2d"):
            # Extended pilots have focused order/jam/memory/direct-reference and
            # scheduled-bracket coverage above; repeating their multidimensional
            # search here would dominate helper self-test time.
            continue
        _report, rec, klb = run(ks, cfg, 8, 8, 256, top=1)
        if rec is None:
            errors.append(f"{name}: no recommendation produced")
            continue
        if not (klb <= rec.pragma_exposure_aggregate <= rec.schedule_estimate):
            errors.append(
                f"{name}: recommended bracket violated lb={klb} "
                f"pragma={rec.pragma_exposure_aggregate} "
                f"sched={rec.schedule_estimate}")

    if errors:
        for e in errors:
            print(f"  SELF-TEST FAIL: {e}")
        return 1
    print("[PASS] loom_dse lane-aware + vector-coalescing self-tests")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
