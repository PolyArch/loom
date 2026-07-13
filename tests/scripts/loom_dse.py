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
across partitions and do NOT coalesce across the cut. There is no banking and no
per-worker port cap -- the only caps are the machine lanes ``L``/``S``.

Load accounting splits into RECURRING vs. one-time INVARIANT loads. Recurring
loop loads (per-iteration array elements over the tiled index, plus induction
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
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from cgra_schedule import (Config, Dag, L, _ceil_div,
                           build_crc32, build_edge_update,
                           build_bitonic_stage_modified,
                           build_bitonic_stage_tweak, evaluate, parse_config,
                           region_aggregate)

V = 4  # 64-bit scalar elements per 256-bit vector memory op (spec convention)

# Marker appended to the ``kind`` of a load that is loop-INVARIANT: hoisted once
# per chunk, its count independent of the tiled exposure (e.g. axpy ``alpha``,
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

    ``invariant`` = loads hoisted once per chunk (count independent of the tiled
    exposure), tagged with ``INV`` by the builders. ``recurring`` = everything
    else: per-iteration array element loads (over the tiled index) and induction
    reads, which scale with exposure. Recurring loads set the steady-state lane
    exposure and the binding load term; invariant loads are amortized (loaded
    once and held) and appear only in ``LD_eff = recurring + invariant``."""
    nodes = dag.regions[0].nodes
    invariant = sum(1 for n in nodes if n.cls == L and INV in n.kind)
    total = sum(1 for n in nodes if n.cls == L)
    return total - invariant, invariant


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
        # parallel levels are tiled across waves (exposure = p*u); reduction and
        # sequential levels are fully consumed within a chunk (exposure = trip).
        return self.kind == "parallel"


@dataclass(frozen=True)
class KernelSpec:
    name: str
    levels: tuple[Level, ...]
    build_chunk: object             # callable(cand: Candidate) -> Dag
    coalesce_note: str = ""
    default_config: str = "6x6"

    def level(self, name: str) -> Level:
        for lv in self.levels:
            if lv.name == name:
                return lv
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Per-kernel chunk builders (split-aware; coalesce contiguous unrolled groups)
# ---------------------------------------------------------------------------
# Each builder receives the Candidate and reads (p, u) per level. It builds ONE
# wave: p workers, each with u contiguous (unrolled) iterations at each tiled
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


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

def _conv2d_dims(C_in=3, C_out=4, H=8, W=8, KH=3, KW=3, stride=1):
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1
    n_out = C_out * OH * OW
    K = C_in * KH * KW
    return n_out, K


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
    M, N = 64, 64
    KERNELS["gemv"] = KernelSpec(
        name="gemv",
        levels=(Level("i", M, "parallel"), Level("j", N, "reduction")),
        build_chunk=_gemv_chunk,
        coalesce_note=(
            "A[i][j] and x[j] are contiguous over j (a fully-consumed reduction, "
            "tree-reduced), so they coalesce identically and the j-loop carries no "
            "control -> the dot-product path is P/U-symmetric. On the row level i, "
            "LOOM_UNROLL(i) beats LOOM_PARALLEL(i) two ways: it coalesces the "
            "contiguous y[i]/output_y[i] accesses (parallel strides) and it "
            "amortizes the row iterator (charged once per worker). The A-load term "
            "is split-symmetric and large, so the i-level edge is modest but real."),
    )
    n_out, K = _conv2d_dims()
    KERNELS["conv2d"] = KernelSpec(
        name="conv2d",
        levels=(Level("out", n_out, "parallel"), Level("tap", K, "reduction")),
        build_chunk=_conv2d_chunk,
        coalesce_note=(
            "output pixels (out = C_out*OH*OW) are parallel; the K = C_in*KH*KW "
            "taps are a fully-consumed reduction (tree-reduced -> no tap iterator). "
            "input is strided over taps (halo) so it does NOT coalesce and "
            "dominates loads; weight is contiguous but reduction-inert; output is "
            "contiguous over out. LOOM_UNROLL(out) beats LOOM_PARALLEL(out) two "
            "ways: it coalesces the output stores and amortizes the out iterator "
            "(charged once per worker). Load-bound on the strided input, so the "
            "edge is modest. Halo reuse / weight sharing not modeled."),
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
        build_chunk=_batchnorm_chunk,
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
            "crc is a non-associative carried state across all bytes and bits. "
            "Parallel factors are illegal and unroll cannot flatten the trace, "
            "so equivalent unroll labels use the canonical P1U1 representative "
            "for the fully consumed serial DAG."),
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


_register()


# ---------------------------------------------------------------------------
# Candidate = a per-level (parallel, unroll) assignment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    # tuple aligned with spec.levels: (level_name, parallel, unroll)
    split: tuple[tuple[str, int, int], ...]

    def factors(self, name: str) -> tuple[int, int]:
        for n, p, u in self.split:
            if n == name:
                return p, u
        raise KeyError(name)

    def signature(self) -> str:
        return " ".join(f"{n}:P{p}U{u}" for n, p, u in self.split)


def _powers_of_two_through(limit: int) -> list[int]:
    values = []
    factor = 1
    while factor <= limit:
        values.append(factor)
        factor *= 2
    return values


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


def evaluate_candidate(spec: KernelSpec, cand: Candidate, cfg: Config,
                       schedule: bool = True) -> CandResult:
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
        agg = res.region_aggs[0]
        chunk_scheduled = res.scheduled_cycles
    else:
        agg = region_aggregate(dag.regions[0], cfg)
        chunk_scheduled = None
    recurring_LD, invariant_LD = _load_split(dag)
    ld_eff = recurring_LD + invariant_LD   # == agg.LD (total traffic)
    waves = _waves(spec, cand)
    p_tot = _p_tot(spec, cand)

    # steady-state binding: load term uses RECURRING loads only (invariants
    # amortized); compute and store are unchanged.
    load = _ceil_div(recurring_LD, cfg.L)
    compute = agg.compute
    store = agg.store
    chunk_aggregate = max(agg.CP, compute, load, store)
    active_L = max(1, min(recurring_LD, cfg.L))
    active_S = max(1, min(agg.ST, cfg.S))

    # binding class of this chunk: the largest resource term (arith stays a
    # global pool; ties favor loads).
    terms = {"P": compute, "L": load, "S": store}
    binding_class = max(terms, key=lambda k: (terms[k], k == "L"))
    max_resource = max(compute, load, store)
    saturation = "latency-bound" if agg.CP > max_resource else "resource-bound"
    denom = chunk_aggregate if chunk_aggregate > 0 else 1
    util = (compute / denom, load / denom, store / denom)

    return CandResult(
        cand=cand, p_tot=p_tot, active_L=active_L, active_S=active_S,
        exposed_iters=_exposed_iters(spec, cand), waves=waves,
        CP=agg.CP, A=agg.A, LD=recurring_LD, ST=agg.ST, ld_eff=ld_eff,
        chunk_aggregate=chunk_aggregate, chunk_scheduled=chunk_scheduled,
        pragma_exposure_aggregate=waves * chunk_aggregate,
        schedule_estimate=(waves * chunk_scheduled
                           if chunk_scheduled is not None else None),
        binding_class=binding_class, saturation=saturation, util=util)


def _ensure_scheduled(spec: KernelSpec, result: CandResult, cfg: Config) -> None:
    if result.schedule_estimate is not None:
        return
    scheduled = evaluate_candidate(spec, result.cand, cfg, schedule=True)
    result.chunk_scheduled = scheduled.chunk_scheduled
    result.schedule_estimate = scheduled.schedule_estimate


def _evaluate_aggregate_task(args) -> CandResult:
    spec, cand, cfg = args
    return evaluate_candidate(spec, cand, cfg, schedule=False)


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
    agg = region_aggregate(dag.regions[0], cfg)
    recurring_LD, invariant_LD = _load_split(dag)
    # steady-state floor: recurring loads only (invariants amortized once over the
    # whole trip). LD_eff = recurring + invariant is the total traffic.
    load = _ceil_div(recurring_LD, cfg.L)
    aggregate = max(agg.CP, agg.compute, load, agg.store)
    demand = {"A": agg.A, "LD": recurring_LD,
              "LD_eff": recurring_LD + invariant_LD, "ST": agg.ST, "CP": agg.CP,
              "compute": agg.compute, "load": load, "store": agg.store,
              "aggregate": aggregate}
    return aggregate, demand


# ---------------------------------------------------------------------------
# Recommendation and flags
# ---------------------------------------------------------------------------

def recommend(spec: KernelSpec, results: list[CandResult]) -> CandResult | None:
    """Spec "Exposure selection": recommend the saturation knee E_sat -- the
    smallest exposure at which the BEST-COALESCED candidate becomes
    resource-bound. Beyond E_sat the wave-summed estimate only creeps toward the
    floor through wave-serialization rounding (not real steady-state gain), and
    larger exposure is oversubscribed.

    A candidate that saturates at *lower* exposure than E_sat does so only because
    it wastes lane-slots (uncoalesced strided loads); we must not reward that.
    So we walk exposures upward and, at each, take the most-coalesced candidate
    (min LD+ST); the first exposure whose best candidate is resource-bound is the
    knee."""
    if not results:
        return None
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
    for r in results:
        if r.cand.split == rec.cand.split:
            r.flags.add("recommended")
            continue  # the pick carries no starved/oversubscribed marker
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
        key = (r.active_L, r.active_S, r.LD, r.ST, r.A, r.CP, r.waves,
               r.pragma_exposure_aggregate)
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


def render_report(spec: KernelSpec, cfg: Config, results: list[CandResult],
                  lb: int, demand: dict, rec: CandResult, search_scope: str,
                  top: int = 0) -> str:
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
    out.append(
        f"Loop nest: `{nest}`; {spec.coalesce_note} Full-trip counts are "
        f"`A={demand['A']}`, `LD_rec={demand['LD']}`, "
        f"`LD_eff={demand['LD_eff']}`, `ST={demand['ST']}`, and "
        f"`CP={demand['CP']}`, giving the only lower bound, "
        f"`absolute_cgra_lb={lb}=max(CP {demand['CP']}, "
        f"compute {demand['compute']}, load {demand['load']}, "
        f"store {demand['store']})`, with {binding} pressure binding; "
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
            fl += "K"
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
    out.append(f"RECOMMENDED: {rec.cand.signature()}  -> "
               f"exposure={rec.exposed_iters}, "
               f"pragma_agg={rec.pragma_exposure_aggregate} "
               f"({ratio:.2f}x the floor), {rec.saturation}")
    out.append("flags: K=recommended (saturation knee E_sat), "
               "b=bandwidth-starved (latency-bound: resources idle), "
               "o=oversubscribed (past the knee, no estimate gain).")
    out.append("")
    out.append(_pu_contrast(spec, cfg, results, rec))
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

def run(spec: KernelSpec, cfg: Config, max_parallel: int | None = None,
        max_unroll: int | None = None, exposure_cap: int | None = None,
        top: int = 0, jobs: int = 1) -> tuple[str, CandResult, int]:
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
    annotate_flags(spec, results, rec)
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
    report = render_report(spec, cfg, results, lb, demand, rec, search_scope,
                           top=top)
    return report, rec, lb


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Lane-aware Loom-pragma design-space estimate")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("kernel", nargs="?",
                        help="kernel name: " + ", ".join(sorted(KERNELS)))
    parser.add_argument("--config", default="6x6")
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
        help="optional diagnostic cap on total tiled exposure; default is uncapped")
    parser.add_argument("--top", type=int, default=24,
                        help="show the best N candidate groups (0 schedules all)")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="parallel workers for exhaustive candidate evaluation")
    args = parser.parse_args(argv)

    for name in ("max_parallel", "max_unroll", "exposure_cap"):
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
    cfg = parse_config(args.config)
    report, _, _ = run(spec, cfg, args.max_parallel, args.max_unroll,
                       args.exposure_cap, top=args.top, jobs=args.jobs)
    print(report)
    return 0


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> int:
    errors: list[str] = []
    cfg = parse_config("6x6")

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
    for name in ("axpy", "batchnorm"):
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
        "crc32": (50152, 51682, 18945, 18945, 19971),
        "edge_update": (6, 40, 38, 38, 37),
    }
    for name, expected in new_expected.items():
        _lb, demand = absolute_cgra_lb(KERNELS[name], cfg)
        got = (demand["CP"], demand["A"], demand["LD"],
               demand["LD_eff"], demand["ST"])
        if got != expected:
            errors.append(f"{name}: full-trip totals {got} != {expected}")
    for name in ("crc32", "edge_update"):
        cands = enumerate_candidates(KERNELS[name], 8, 8, 256)
        if any(p > 1 for c in cands for _, p, _ in c.split):
            errors.append(f"{name}: sequential level must not parallelize")

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
