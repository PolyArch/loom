#!/usr/bin/env python3
from __future__ import annotations
"""CGRA kernel performance: aggregate lower bound + finite-resource estimate.

Reference implementation of docs/spec-kernel-performance.md. Standard library
only (no third-party scheduler dependency). Two metrics over the same dynamic
operation DAG, resource classes, and one-cycle latency:

1. Aggregate CGRA lower bound (preserved): per region
   ``max(CP, ceil(A/P), ceil(LD/L), ceil(ST/S))``, summed over barrier-ordered
   regions.
2. Finite-resource schedule estimate (NOT a lower bound, NOT cycle-accurate
   RTL): a deterministic criticality-priority list schedule under finite
   ``P``/``L``/``S`` capacities; reports scheduled cycles, the gap vs. the
   aggregate bound, and a local resource-pressure summary.

Entry points (mirrors tests/scripts/check_bridge_tags.py):
  cgra_schedule.py --self-test            run synthetic + golden + drift checks
  cgra_schedule.py report  <kernel> [--config 6x6]   print the canonical block
  cgra_schedule.py write  [<kernel> ...] [--config 6x6]   write eval blocks
  cgra_schedule.py --check [<kernel> ...] [--config 6x6]   read-only drift check
"""

import argparse
import heapq
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resource classes
# ---------------------------------------------------------------------------

P = "P"  # arithmetic PE work: add, sub, address_add, mul, div, cmp, bitop,
         # shift, transcendental
L = "L"  # load-issue lane
S = "S"  # store-issue lane
CLASSES = (P, L, S)


def _ceil_div(numer: int, denom: int) -> int:
    """ceil(numer/denom) with ceil(0/d) == 0; denom must be positive."""
    if numer <= 0:
        return 0
    return -(-numer // denom)


# ---------------------------------------------------------------------------
# Resource configuration
# ---------------------------------------------------------------------------

class Config:
    """A CGRA resource configuration: separate P/L/S issue capacities."""

    __slots__ = ("P", "L", "S", "label")

    def __init__(self, p: int, l: int, s: int, label: str = ""):
        if p <= 0 or l <= 0 or s <= 0:
            raise ValueError(
                "invalid resource configuration: P/L/S capacities must all be "
                f"positive (got P={p}, L={l}, S={s}); a zero-capacity class "
                "cannot drain and would never complete")
        self.P = p
        self.L = l
        self.S = s
        self.label = label or f"P={p},L={l},S={s}"

    def cap(self, cls: str) -> int:
        return self.P if cls == P else self.L if cls == L else self.S


def parse_config(text: str) -> Config:
    """Parse a configuration string.

    Accepts the grid form ``AxB`` -> ``P=A*B, L=A+B, S=A+B`` (an A-by-B PE grid
    with one load and one store lane per row+column), so ``6x6`` is the
    canonical ``P=36, L=12, S=12``. Also accepts the explicit form
    ``P=..,L=..,S=..``.
    """
    text = text.strip()
    if "=" in text:
        vals = {}
        for part in text.split(","):
            key, _, value = part.partition("=")
            vals[key.strip().upper()] = int(value)
        return Config(vals["P"], vals["L"], vals["S"], label=text)
    if "x" in text:
        a_str, _, b_str = text.partition("x")
        a, b = int(a_str), int(b_str)
        return Config(a * b, a + b, a + b, label=text)
    raise ValueError(f"unrecognized config string: {text!r}")


CONFIG_6x6 = parse_config("6x6")  # P=36, L=12, S=12


# ---------------------------------------------------------------------------
# Operation DAG
# ---------------------------------------------------------------------------

class Node:
    """One counted dynamic operation: a resource class, predecessors, unit
    latency, and a kernel-output flag (used only for CP)."""

    __slots__ = ("nid", "cls", "preds", "is_output", "kind")

    def __init__(self, nid: int, cls: str, preds, is_output: bool, kind: str):
        self.nid = nid
        self.cls = cls
        self.preds = preds  # list[int] of predecessor node ids (same region)
        self.is_output = is_output
        self.kind = kind  # descriptive label only (e.g. "add", "address_add")


class Region:
    """A set of nodes scheduled together (they may overlap freely). Regions are
    the unit of barrier composition: an ordered list of regions is summed."""

    def __init__(self, name: str, dag: "Dag"):
        self.name = name
        self._dag = dag
        self.nodes: list[Node] = []
        self._by_id: dict[int, Node] = {}

    def _add(self, cls: str, preds, is_output: bool, kind: str) -> int:
        for p in preds:
            if p not in self._by_id:
                raise ValueError(
                    f"node predecessor {p} is not in region {self.name!r}; "
                    "edges must stay within a region (barriers carry ordering)")
        nid = self._dag._next_id
        self._dag._next_id += 1
        node = Node(nid, cls, list(preds), is_output, kind)
        self.nodes.append(node)
        self._by_id[nid] = node
        return nid

    # --- DAG primitives (each returns the new node id) ---

    def load(self, *preds, output: bool = False, kind: str = "load") -> int:
        return self._add(L, preds, output, kind)

    def store(self, *preds, output: bool = False, kind: str = "store") -> int:
        return self._add(S, preds, output, kind)

    def arith(self, *preds, output: bool = False, kind: str = "arith") -> int:
        return self._add(P, preds, output, kind)

    def address_add(self, *preds, kind: str = "address_add") -> int:
        """An add/sub inside a subscript expression: arithmetic (P) work."""
        return self._add(P, preds, False, kind)

    def balanced_reduction(self, leaves, kind: str = "reduce") -> int:
        """Build an explicit balanced binary tree of P-class adds over the leaf
        node ids. n leaves -> (n-1) adds, depth ceil(log2(n)). For a single
        leaf, returns it unchanged (no add)."""
        level = list(leaves)
        if not level:
            raise ValueError("balanced_reduction needs at least one leaf")
        while len(level) > 1:
            nxt = []
            i = 0
            while i + 1 < len(level):
                nxt.append(self.arith(level[i], level[i + 1], kind=kind))
                i += 2
            if i < len(level):  # odd element carries up unchanged
                nxt.append(level[i])
            level = nxt
        return level[0]

    def induction(self, kind: str = "iv") -> dict:
        """A loop iterator step: read, increment, write-back, bound compare.
        Returns the four node ids. The read/compare feed address/bound work;
        the write-back is off the output path."""
        rd = self.load(kind=kind + "_load")
        inc = self.arith(rd, kind=kind + "_add")
        wr = self.store(inc, kind=kind + "_store")
        cmp = self.arith(rd, kind=kind + "_cmp")
        return {"read": rd, "add": inc, "store": wr, "cmp": cmp}


class Dag:
    """An ordered list of regions sharing one monotonic node-id counter (node
    id == construction/append order, used as the scheduler tie-break)."""

    def __init__(self):
        self._next_id = 0
        self.regions: list[Region] = []

    def region(self, name: str) -> Region:
        r = Region(name, self)
        self.regions.append(r)
        return r


# ---------------------------------------------------------------------------
# Depth, height, reachability
# ---------------------------------------------------------------------------

def _successors(region: Region) -> dict[int, list[int]]:
    succ: dict[int, list[int]] = {n.nid: [] for n in region.nodes}
    for n in region.nodes:
        for p in n.preds:
            succ[p].append(n.nid)
    return succ


def _depths(region: Region) -> dict[int, int]:
    """Longest dependency chain (in nodes) ending at each node. Relies on nodes
    being appended after their predecessors (topological order)."""
    depth: dict[int, int] = {}
    for n in region.nodes:
        d = 1
        for p in n.preds:
            if depth[p] + 1 > d:
                d = depth[p] + 1
        depth[n.nid] = d
    return depth


def _heights(region: Region, succ: dict[int, list[int]]) -> dict[int, int]:
    """Longest dependency chain (in nodes) starting at each node and running to
    a region sink. Every node feeds a single virtual region sink, so height is
    defined for dead/disconnected nodes too."""
    height: dict[int, int] = {}
    for n in reversed(region.nodes):  # reverse topological order
        h = 1
        for s in succ[n.nid]:
            if height[s] + 1 > h:
                h = height[s] + 1
        height[n.nid] = h
    return height


def _output_reachable(region: Region) -> set[int]:
    """Nodes from which a kernel-output node is reachable (backward BFS from the
    output-flagged nodes over predecessor edges)."""
    reach: set[int] = set()
    stack = [n.nid for n in region.nodes if n.is_output]
    reach.update(stack)
    by_id = region._by_id
    while stack:
        cur = stack.pop()
        for p in by_id[cur].preds:
            if p not in reach:
                reach.add(p)
                stack.append(p)
    return reach


# ---------------------------------------------------------------------------
# Metric 1: aggregate CGRA lower bound (per region)
# ---------------------------------------------------------------------------

class RegionAggregate:
    __slots__ = ("name", "A", "LD", "ST", "CP", "compute", "load", "store",
                 "aggregate")

    def __init__(self, name, A, LD, ST, CP, compute, load, store, aggregate):
        self.name = name
        self.A = A
        self.LD = LD
        self.ST = ST
        self.CP = CP
        self.compute = compute
        self.load = load
        self.store = store
        self.aggregate = aggregate


def region_aggregate(region: Region, cfg: Config) -> RegionAggregate:
    A = sum(1 for n in region.nodes if n.cls == P)
    LD = sum(1 for n in region.nodes if n.cls == L)
    ST = sum(1 for n in region.nodes if n.cls == S)
    depth = _depths(region)
    reach = _output_reachable(region)
    cp = max((depth[nid] for nid in reach), default=0)
    compute = _ceil_div(A, cfg.P)
    load = _ceil_div(LD, cfg.L)
    store = _ceil_div(ST, cfg.S)
    aggregate = max(cp, compute, load, store)
    return RegionAggregate(region.name, A, LD, ST, cp, compute, load, store,
                           aggregate)


# ---------------------------------------------------------------------------
# Metric 2: deterministic criticality-priority list schedule (per region)
# ---------------------------------------------------------------------------

class ClassPressure:
    __slots__ = ("saturated_cycles", "longest_run", "peak_ready_backlog")

    def __init__(self):
        self.saturated_cycles = 0
        self.longest_run = 0
        self.peak_ready_backlog = 0


class RegionSchedule:
    __slots__ = ("name", "makespan", "pressure")

    def __init__(self, name, makespan, pressure):
        self.name = name
        self.makespan = makespan
        self.pressure = pressure  # dict[class -> ClassPressure]


def schedule_region(region: Region, cfg: Config) -> RegionSchedule:
    """Issue ready ops each cycle, highest reverse-height first, ties by
    ascending node id, up to the per-class capacity, until every node (including
    dead/disconnected) has issued. Deterministic by construction."""
    succ = _successors(region)
    height = _heights(region, succ)
    by_id = region._by_id

    pending = {n.nid: len(n.preds) for n in region.nodes}
    ready = {P: [], L: [], S: []}
    for n in region.nodes:
        if pending[n.nid] == 0:
            heapq.heappush(ready[n.cls], (-height[n.nid], n.nid))

    pressure = {c: ClassPressure() for c in CLASSES}
    cur_run = {c: 0 for c in CLASSES}
    total = len(region.nodes)
    issued = 0
    cycle = 0

    while issued < total:
        cycle += 1
        # Local pressure: ready ops that cannot issue this cycle.
        for c in CLASSES:
            backlog = len(ready[c]) - cfg.cap(c)
            if backlog > pressure[c].peak_ready_backlog:
                pressure[c].peak_ready_backlog = backlog
        newly_ready: list[int] = []
        for c in CLASSES:
            cap = cfg.cap(c)
            count = 0
            while ready[c] and count < cap:
                _, nid = heapq.heappop(ready[c])
                count += 1
                issued += 1
                for s in succ[nid]:
                    pending[s] -= 1
                    if pending[s] == 0:
                        newly_ready.append(s)
            if count == cap:  # class fully utilized this cycle
                pressure[c].saturated_cycles += 1
                cur_run[c] += 1
                if cur_run[c] > pressure[c].longest_run:
                    pressure[c].longest_run = cur_run[c]
            else:
                cur_run[c] = 0
        for nid in newly_ready:
            node = by_id[nid]
            heapq.heappush(ready[node.cls], (-height[nid], nid))

    return RegionSchedule(region.name, cycle, pressure)


# ---------------------------------------------------------------------------
# Kernel-level metrics + conditional invariant
# ---------------------------------------------------------------------------

class KernelResult:
    def __init__(self, kernel, cfg, region_aggs, region_scheds):
        self.kernel = kernel
        self.cfg = cfg
        self.region_aggs = region_aggs
        self.region_scheds = region_scheds
        self.aggregate_cycles = sum(r.aggregate for r in region_aggs)
        self.scheduled_cycles = sum(r.makespan for r in region_scheds)
        self.gap_cycles = self.scheduled_cycles - self.aggregate_cycles
        if self.aggregate_cycles == 0:
            self.gap_ratio = 1.0
        else:
            self.gap_ratio = self.scheduled_cycles / self.aggregate_cycles
        # Composed local pressure across regions (boundaries break runs).
        self.pressure = {c: ClassPressure() for c in CLASSES}
        for rs in region_scheds:
            for c in CLASSES:
                self.pressure[c].saturated_cycles += rs.pressure[c].saturated_cycles
                self.pressure[c].longest_run = max(
                    self.pressure[c].longest_run, rs.pressure[c].longest_run)
                self.pressure[c].peak_ready_backlog = max(
                    self.pressure[c].peak_ready_backlog,
                    rs.pressure[c].peak_ready_backlog)


def evaluate(dag: Dag, kernel: str, cfg: Config) -> KernelResult:
    region_aggs = [region_aggregate(r, cfg) for r in dag.regions]
    region_scheds = [schedule_region(r, cfg) for r in dag.regions]
    result = KernelResult(kernel, cfg, region_aggs, region_scheds)
    # Conditional invariant: under a matched partition/op-set/capacities with
    # all ops issuing, the schedule never beats the aggregate lower bound.
    for ra, rs in zip(region_aggs, region_scheds):
        if rs.makespan < ra.aggregate:
            raise AssertionError(
                f"invariant violated in region {ra.name!r}: scheduled "
                f"{rs.makespan} < aggregate {ra.aggregate}")
    if result.scheduled_cycles < result.aggregate_cycles:
        raise AssertionError(
            f"invariant violated for {kernel!r}: scheduled "
            f"{result.scheduled_cycles} < aggregate {result.aggregate_cycles}")
    return result


# ---------------------------------------------------------------------------
# Canonical eval-block formatter (marker-bounded)
# ---------------------------------------------------------------------------

def marker_begin(kernel: str) -> str:
    return f"<!-- BEGIN CGRA-SCHED:{kernel} -->"


def marker_end(kernel: str) -> str:
    return f"<!-- END CGRA-SCHED:{kernel} -->"


def render_block(result: KernelResult) -> str:
    """Render the marker-bounded finite-resource schedule estimate block."""
    cfg = result.cfg
    lines = []
    lines.append(marker_begin(result.kernel))
    lines.append("### Finite-Resource Schedule Estimate (time-local)")
    lines.append("")
    lines.append(
        "*Reproducible estimate for the deterministic criticality-priority "
        "list-schedule policy defined in "
        "[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md). "
        "It is **not** a lower bound (the aggregate model above is the lower "
        "bound) and **not** cycle-accurate RTL; it exposes the short windows of "
        "local `P`/`L`/`S` pressure that the aggregate model smooths over.*")
    lines.append("")
    lines.append(
        f"**Resource configuration:** `P = {cfg.P}`, `L = {cfg.L}`, "
        f"`S = {cfg.S}` (`{cfg.label}`).")
    lines.append("")
    multi = len(result.region_aggs) > 1
    lines.append(
        "| region | CP | A | LD | ST | aggregate | scheduled (makespan) |")
    lines.append(
        "|--------|---:|--:|---:|---:|----------:|---------------------:|")
    for ra, rs in zip(result.region_aggs, result.region_scheds):
        lines.append(
            f"| {ra.name} | {ra.CP} | {ra.A} | {ra.LD} | {ra.ST} | "
            f"{ra.aggregate} | {rs.makespan} |")
    if multi:
        lines.append(
            f"| **total** |  |  |  |  | **{result.aggregate_cycles}** | "
            f"**{result.scheduled_cycles}** |")
    lines.append("")
    lines.append(
        f"- **scheduled_cycles** = {result.scheduled_cycles}  "
        "(sum of ordered-region makespans)")
    lines.append(
        f"- **aggregate_cycles** = {result.aggregate_cycles}  "
        "(the lower bound above, unchanged)")
    lines.append(
        f"- **gap_cycles** = {result.gap_cycles}  (scheduled − aggregate)")
    lines.append(
        f"- **gap_ratio** = {_fmt_ratio(result.gap_ratio)}  "
        "(scheduled / aggregate)")
    lines.append("")
    lines.append(
        "**Local `P`/`L`/`S` pressure** "
        "(saturated cycles / longest saturated run / peak ready backlog):")
    for c in CLASSES:
        pr = result.pressure[c]
        lines.append(
            f"- `{c}`: {pr.saturated_cycles} / {pr.longest_run} / "
            f"{pr.peak_ready_backlog}")
    lines.append("")
    lines.append(marker_end(result.kernel))
    return "\n".join(lines)


def _fmt_ratio(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


# ---------------------------------------------------------------------------
# Marker-bounded eval writer / reader
# ---------------------------------------------------------------------------

def apply_block(text: str, kernel: str, block: str) -> str:
    """Return ``text`` with the kernel's marker block set to ``block``. If the
    markers exist, replace only the bytes between (and including) them;
    otherwise append after the existing content. Touches nothing else."""
    begin, end = marker_begin(kernel), marker_end(kernel)
    if begin in text and end in text:
        head = text[:text.index(begin)]
        tail = text[text.index(end) + len(end):]
        return head + block + tail
    sep = "" if text.endswith("\n") else "\n"
    return text + sep + "\n" + block + "\n"


def extract_block(text: str, kernel: str):
    """Return the current marker block text (inclusive) or None if absent."""
    begin, end = marker_begin(kernel), marker_end(kernel)
    if begin in text and end in text:
        return text[text.index(begin):text.index(end) + len(end)]
    return None


# ---------------------------------------------------------------------------
# Pilot DAG builders + builder contracts
# ---------------------------------------------------------------------------

class RegionContract:
    """A builder's declared expectation for one region, checked against the
    constructed DAG and against the eval's golden numbers."""

    __slots__ = ("name", "A", "LD", "ST", "CP", "aggregate")

    def __init__(self, name, A, LD, ST, CP, aggregate):
        self.name = name
        self.A = A
        self.LD = LD
        self.ST = ST
        self.CP = CP
        self.aggregate = aggregate


def build_axpy(N: int = 8):
    """output_y[i] = alpha * input_x[i] + input_y[i] (parallel, single region).

    Per lane: load input_x ‖ load input_y ‖ load i; mul; add; store output_y;
    i++ ; store i; compare i<N. alpha and N are hoisted loads (charged once).
    """
    dag = Dag()
    r = dag.region("axpy")
    ld_alpha = r.load(kind="alpha")
    r.load(kind="N")  # N param hoist (charged once; feeds the bound compare)
    for _ in range(N):
        ld_x = r.load(kind="input_x")
        ld_y = r.load(kind="input_y")
        mul = r.arith(ld_x, ld_alpha, kind="mul")
        add = r.arith(mul, ld_y, kind="add")
        r.store(add, output=True, kind="output_y")
        r.induction(kind="i")  # i read + i++ + i write + i<N compare
    contract = [RegionContract("axpy", A=32, LD=26, ST=16, CP=4, aggregate=4)]
    return dag, contract


def build_autocorrelation(x_size: int = 128, max_lag: int = 32):
    """output[lag] = sum_i x[i]*x[i+lag] (outer lag parallel, inner reduction).

    Per inner iter: address_add for &x[i+lag]; load x[i] ‖ load x[i+lag]; mul.
    The N_lag products feed a balanced reduction; the root stores output[lag].
    Inner/outer inductions and the x_size-lag bound sub are counted overhead.
    """
    dag = Dag()
    r = dag.region("autocorrelation")
    r.load(kind="x_size")
    r.load(kind="max_lag")
    for lag in range(max_lag):
        n_lag = x_size - lag
        r.arith(kind="bound_sub")  # x_size - lag, once per outer iter
        products = []
        for _ in range(n_lag):
            aa = r.address_add(kind="addr_i_plus_lag")  # &x[i+lag]
            ld_a = r.load(kind="x_i")                    # bare subscript
            ld_b = r.load(aa, kind="x_i_plus_lag")
            products.append(r.arith(ld_a, ld_b, kind="mul"))
            r.induction(kind="i")  # inner induction (off output path)
        root = r.balanced_reduction(products, kind="reduce")
        r.store(root, output=True, kind="output_lag")
        r.induction(kind="lag")  # outer induction (off output path)
    contract = [RegionContract("autocorrelation", A=18064, LD=10834, ST=3664,
                               CP=11, aggregate=903)]
    return dag, contract


def _fft_butterfly(r, ld_wr, ld_wi, cos, sin, ld_j):
    """Emit one butterfly. ld_wr/ld_wi are the (loaded) carried twiddle for this
    iteration; cos/sin are the broadcast wm_r/wm_i; ld_j is the loaded iterator.
    Returns (st_wr, st_wi) -- the twiddle carry stores for the next iteration."""
    # Data operand address: (k+j) then (+m/2); k and m/2 are folded constants.
    addr1 = r.address_add(ld_j, kind="k_plus_j")
    addr2 = r.address_add(addr1, kind="plus_m_half")
    ld_xr = r.load(addr2, kind="out_real_upper")
    ld_xi = r.load(addr2, kind="out_imag_upper")
    ld_ur = r.load(addr1, kind="out_real_lower")
    ld_ui = r.load(addr1, kind="out_imag_lower")
    # t = w * X  (4 muls -> t_r sub, t_i add)
    mt1 = r.arith(ld_wr, ld_xr, kind="mul")
    mt2 = r.arith(ld_wi, ld_xi, kind="mul")
    mt3 = r.arith(ld_wr, ld_xi, kind="mul")
    mt4 = r.arith(ld_wi, ld_xr, kind="mul")
    t_r = r.arith(mt1, mt2, kind="sub")
    t_i = r.arith(mt3, mt4, kind="add")
    # u +/- t -> 4 array stores
    out_kr = r.arith(ld_ur, t_r, kind="add")
    out_ki = r.arith(ld_ui, t_i, kind="add")
    out_pr = r.arith(ld_ur, t_r, kind="sub")
    out_pi = r.arith(ld_ui, t_i, kind="sub")
    r.store(out_kr, output=True, kind="st_real_lower")
    r.store(out_ki, output=True, kind="st_imag_lower")
    r.store(out_pr, output=True, kind="st_real_upper")
    r.store(out_pi, output=True, kind="st_imag_upper")
    # w = w * wm  (II=4 carry: load w -> mul -> add/sub -> store w)
    mw1 = r.arith(ld_wr, cos, kind="mul")
    mw2 = r.arith(ld_wi, sin, kind="mul")
    mw3 = r.arith(ld_wr, sin, kind="mul")
    mw4 = r.arith(ld_wi, cos, kind="mul")
    new_wr = r.arith(mw1, mw2, kind="sub")
    new_wi = r.arith(mw3, mw4, kind="add")
    st_wr = r.store(new_wr, kind="st_w_real")
    st_wi = r.store(new_wi, kind="st_w_imag")
    return st_wr, st_wi


def build_fft_butterfly(N: int = 16):
    """Radix-2 DIT FFT, in-place. copy -> s=1 -> s=2 -> s=3 -> s=4 are five
    barrier-ordered (summed) regions because stage s+1 reads output_* elements
    stage s overwrote in place. The j loop carries the twiddle w<-w*wm (II=4).
    The three kernel-once residual ops (N load, log2f, s-loop init store) ride
    in the copy region (they overlap it and add no cycles)."""
    log2n = int(round(math.log2(N)))
    dag = Dag()

    # --- copy region (parallel) ---
    c = dag.region("copy")
    for _ in range(N):
        ld_ir = c.load(kind="input_real")
        ld_ii = c.load(kind="input_imag")
        c.store(ld_ir, output=True, kind="output_real")
        c.store(ld_ii, output=True, kind="output_imag")
    for _ in range(N):
        c.induction(kind="copy_i")
    c.store(kind="copy_i_init")  # i=0 init store
    # Three kernel-once residual ops overlap the copy phase.
    ld_N = c.load(kind="N")
    c.arith(ld_N, kind="log2f")           # stage-loop bound transcendental
    c.store(kind="stage_loop_init")       # s=1 init store

    # --- stages s = 1..log2(N) ---
    stage_contracts = []
    for s in range(1, log2n + 1):
        m = 1 << s
        blocks = N // m
        per_block = m // 2  # j-trip
        rg = dag.region(f"s={s}")
        # Per-stage prologue: 2 shifts, 1 divide, cos, sin (broadcast wm).
        rg.arith(kind="shift_m")
        rg.arith(kind="shift_m_half")
        rg.arith(kind="div_twopi_m")
        cos = rg.arith(kind="cos_wm_r")
        sin = rg.arith(kind="sin_wm_i")
        rg.store(kind="k_init")  # k=0 init store, once per stage
        for _ in range(blocks):
            # w init for the block (w_r=1, w_i=0): two stores.
            rg.store(kind="w_r_init")
            rg.store(kind="w_i_init")
            rg.store(kind="j_init")  # j=0 init store, once per block
            prev_wr = prev_wi = prev_stj = None
            for j in range(per_block):
                ld_wr = rg.load(prev_wr, kind="w_r") if prev_wr is not None \
                    else rg.load(kind="w_r")
                ld_wi = rg.load(prev_wi, kind="w_i") if prev_wi is not None \
                    else rg.load(kind="w_i")
                # j induction is a sequential carry (II=3): load j -> j++ ->
                # store j; the loaded j also drives the data-operand address.
                ld_j = rg.load(prev_stj, kind="j_load") if prev_stj is not None \
                    else rg.load(kind="j_load")
                add_j = rg.arith(ld_j, kind="j_add")
                prev_stj = rg.store(add_j, kind="j_store")
                rg.arith(ld_j, kind="j_cmp")
                prev_wr, prev_wi = _fft_butterfly(
                    rg, ld_wr, ld_wi, cos, sin, ld_j)
            rg.induction(kind="k")  # one k step per block (parallel; off path)
        rg.induction(kind="s")  # one s step per stage (init store lives in copy)

        b = blocks
        stage_contracts.append(RegionContract(
            name=f"s={s}",
            A=167 + 2 * b,
            LD=57 + b,
            ST=58 + 4 * b,
            CP={1: 8, 2: 11, 3: 17, 4: 33}[s],
            aggregate={1: 8, 2: 11, 3: 17, 4: 33}[s],
        ))

    # copy contract: documented copy phase (32/48/49) + 3 kernel-once residuals.
    contract = [RegionContract("copy", A=33, LD=49, ST=50, CP=2, aggregate=5)]
    contract.extend(stage_contracts)
    return dag, contract


def build_conv2d(C_in=3, C_out=4, H=8, W=8, KH=3, KW=3, stride=1):
    """2D convolution with a zero-fill prologue. Single region: the zero-fill is
    a dead write-after-write (the convolution overwrites output[] and never
    reads the zeros), so it overlaps the convolution rather than acting as a
    barrier. Each of the n_out output lanes reduces K = C_in*KH*KW tap products.

    Built faithfully from kernel structure (no filler) to reproduce the eval
    golden numbers (C_in=3,C_out=4,H=W=8,KH=KW=3): CP=17, A=74515, LD=13716,
    ST=6220, aggregate 6x6 = 2070. The binding chain runs through the derived
    OH/OW bound, h/w, the input-index expression, load, multiply, the K-tap
    reduction, and the output store.
    """
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1
    n_out = C_out * OH * OW   # output positions (144)
    K = C_in * KH * KW        # taps per output (27)
    dag = Dag()
    r = dag.region("conv2d")

    # Scalar-parameter loads (8); one feeds the derived-bound chain.
    params = [r.load(kind="param") for _ in range(8)]
    p0 = params[0]

    # Derived bounds OH/OW = (H-KH)/stride + 1 (sub -> div -> add), a structural
    # unroll prerequisite that prefixes the binding chain (ready at depth 4).
    oh = r.arith(r.arith(r.arith(p0, kind="oh_sub"), kind="oh_div"),
                 kind="oh_add")
    ow = r.arith(r.arith(r.arith(p0, kind="ow_sub"), kind="ow_div"),
                 kind="ow_add")

    # Hoisted, loop-invariant products charged once and broadcast.
    r.arith(kind="zero_fill_bound_mul")
    r.arith(kind="in_index_hoist")
    r.arith(kind="ker_index_hoist1")
    r.arith(kind="ker_index_hoist2")
    r.arith(kind="out_index_hoist")

    # Zero-fill prologue: n_out dead stores (WAW; overlaps the convolution).
    for _ in range(n_out):
        r.store(kind="zero_fill")

    # Convolution: n_out independent output lanes, each a K-tap reduction.
    for _ in range(n_out):
        products = []
        for _ in range(K):
            # h = oh*stride + kh ; w = ow*stride + kw (depend on OH/OW prefix).
            h = r.arith(r.arith(oh, kind="h_mul"), kind="h_add")
            w = r.arith(r.arith(ow, kind="w_mul"), kind="w_add")
            # input index ci*(H*W) + h*W + w : 2 muls + 2 address_adds.
            in_ci = r.arith(kind="in_ci_HW")
            in_hw = r.arith(h, kind="in_h_W")
            in_part = r.address_add(in_hw, w, kind="in_hW_plus_w")
            in_addr = r.address_add(in_ci, in_part, kind="in_index")
            ld_in = r.load(in_addr, kind="input")
            # kernel index co*(C_in*KH*KW)+ci*(KH*KW)+kh*KW+kw (off path):
            # 3 muls + 3 address_adds.
            k0 = r.arith(kind="ker_co")
            k1 = r.arith(kind="ker_ci")
            k2 = r.arith(kind="ker_kh")
            ka0 = r.address_add(k2, kind="ker_khKW_plus_kw")
            ka1 = r.address_add(k1, ka0, kind="ker_partial")
            k_addr = r.address_add(k0, ka1, kind="ker_index")
            ld_ker = r.load(k_addr, kind="kernel")
            products.append(r.arith(ld_in, ld_ker, kind="tap_mul"))
        root = r.balanced_reduction(products, kind="reduce")  # K-1 adds
        # output index co*(OH*OW) + oh*OW + ow : 2 muls + 2 address_adds.
        o_mul0 = r.arith(oh, kind="out_co")
        o_mul1 = r.arith(oh, kind="out_ohOW")
        oaddr0 = r.address_add(o_mul1, kind="out_addr0")
        oaddr1 = r.address_add(o_mul0, oaddr0, kind="out_addr1")
        r.store(root, oaddr1, output=True, kind="output")

    # Induction work: I dynamic iterator steps across the seven source loops
    # (zero-fill i, co, oh, ow, ci, kh, kw) -- each a load+add+store+compare.
    I = (n_out + C_out + C_out * OH + C_out * OH * OW
         + n_out * C_in + n_out * C_in * KH + n_out * C_in * KH * KW)
    for _ in range(I):
        r.induction(kind="iv")

    contract = [RegionContract("conv2d", A=74515, LD=13716, ST=6220, CP=17,
                               aggregate=2070)]
    return dag, contract


BUILDERS = {
    "axpy": build_axpy,
    "autocorrelation": build_autocorrelation,
    "fft_butterfly": build_fft_butterfly,
    "conv2d": build_conv2d,
}

PILOTS = ("axpy", "autocorrelation", "fft_butterfly")

EVAL_PATHS = {
    name: Path(__file__).resolve().parents[1] / "app" / name / f"{name}_eval.md"
    for name in BUILDERS
}


def build_kernel(kernel: str):
    if kernel not in BUILDERS:
        raise KeyError(f"unknown kernel {kernel!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[kernel]()


def check_contract(dag: Dag, contract, cfg: Config):
    """Verify the constructed DAG matches the builder's declared contract.
    Returns the list of RegionAggregate for further checks."""
    if len(dag.regions) != len(contract):
        raise AssertionError(
            f"region count {len(dag.regions)} != contract {len(contract)}")
    aggs = []
    for region, decl in zip(dag.regions, contract):
        ra = region_aggregate(region, cfg)
        aggs.append(ra)
        for field in ("A", "LD", "ST", "CP", "aggregate"):
            got = getattr(ra, field)
            want = getattr(decl, field)
            if got != want:
                raise AssertionError(
                    f"{region.name}: {field} = {got}, contract says {want}")
    return aggs


# ---------------------------------------------------------------------------
# Report / write / check entry points
# ---------------------------------------------------------------------------

def report(kernel: str, cfg: Config) -> str:
    dag, contract = build_kernel(kernel)
    check_contract(dag, contract, cfg)
    result = evaluate(dag, kernel, cfg)
    out = [f"# {kernel}  ({cfg.label}: P={cfg.P} L={cfg.L} S={cfg.S})", ""]
    out.append("per-region validation:")
    for ra, rs in zip(result.region_aggs, result.region_scheds):
        out.append(
            f"  {ra.name:<14} CP={ra.CP:<3} A={ra.A:<6} LD={ra.LD:<6} "
            f"ST={ra.ST:<6} aggregate={ra.aggregate:<4} "
            f"scheduled={rs.makespan}")
    out.append(
        f"  {'TOTAL':<14} aggregate={result.aggregate_cycles} "
        f"scheduled={result.scheduled_cycles} "
        f"gap={result.gap_cycles} ratio={_fmt_ratio(result.gap_ratio)}")
    out.append("")
    out.append(render_block(result))
    return "\n".join(out)


def write_eval(kernel: str, cfg: Config) -> bool:
    """Write/refresh the kernel's marker block in its eval file. Returns True if
    the file content changed."""
    dag, contract = build_kernel(kernel)
    check_contract(dag, contract, cfg)
    result = evaluate(dag, kernel, cfg)
    block = render_block(result)
    path = EVAL_PATHS[kernel]
    text = path.read_text()
    new_text = apply_block(text, kernel, block)
    if new_text != text:
        path.write_text(new_text)
        return True
    return False


def check_eval(kernel: str, cfg: Config) -> list[str]:
    """Read-only: re-derive the block and confirm it matches what is written in
    the eval. Returns a list of drift messages (empty == clean)."""
    dag, contract = build_kernel(kernel)
    problems = []
    try:
        check_contract(dag, contract, cfg)
    except AssertionError as exc:
        problems.append(f"{kernel}: contract drift: {exc}")
        return problems
    result = evaluate(dag, kernel, cfg)
    block = render_block(result)
    path = EVAL_PATHS[kernel]
    if not path.exists():
        problems.append(f"{kernel}: eval file missing: {path}")
        return problems
    current = extract_block(path.read_text(), kernel)
    if current is None:
        problems.append(f"{kernel}: no CGRA-SCHED block found in {path.name}")
    elif current != block:
        problems.append(
            f"{kernel}: written block in {path.name} differs from freshly "
            "computed block (drift); re-run `write` to refresh")
    return problems


# ---------------------------------------------------------------------------
# Self-test: synthetic edge cases + golden anchors + drift guard
# ---------------------------------------------------------------------------

def _chain(region, cls, length):
    prev = None
    nid = None
    for _ in range(length):
        nid = region._add(cls, [prev] if prev is not None else [], False, "chain")
        prev = nid
    return nid


def _run_synthetic_tests(errors):
    cfg = parse_config("P=2,L=2,S=2")

    # 1. Pure chain: scheduled == CP == aggregate.
    d = Dag(); r = d.region("chain")
    last = _chain(r, P, 5)
    r._by_id[last].is_output = True
    res = evaluate(d, "chain", cfg)
    ra = res.region_aggs[0]
    if not (ra.CP == 5 and res.scheduled_cycles == 5 and ra.aggregate == 5):
        errors.append(f"chain: CP/agg/sched != 5 ({ra.CP}/{ra.aggregate}/"
                      f"{res.scheduled_cycles})")

    # 2. Wide burst: scheduled == ceil(N/cap).
    d = Dag(); r = d.region("burst")
    for _ in range(7):
        r.arith(output=True)
    res = evaluate(d, "burst", parse_config("P=3,L=1,S=1"))
    if res.scheduled_cycles != 3:  # ceil(7/3)
        errors.append(f"burst: scheduled {res.scheduled_cycles} != 3")

    # 3. Mixed P/L/S pressure in one cycle.
    d = Dag(); r = d.region("mixed")
    for _ in range(4):
        r.load()
    for _ in range(4):
        r.arith()
    st = r.store(output=True)
    res = evaluate(d, "mixed", parse_config("P=2,L=2,S=2"))
    # loads: ceil(4/2)=2, arith ceil(4/2)=2, store 1 -> aggregate dominated by
    # resources; scheduled must be >= aggregate.
    if res.scheduled_cycles < res.aggregate_cycles:
        errors.append("mixed: invariant violated")

    # 4. Ordered-phase sum vs 5. unordered overlap.
    d = Dag()
    a = d.region("p1"); _chain(a, P, 3); a.nodes[-1].is_output = True
    b = d.region("p2"); _chain(b, P, 3); b.nodes[-1].is_output = True
    ordered = evaluate(d, "ordered", parse_config("P=4,L=1,S=1"))
    if ordered.scheduled_cycles != 6:  # 3 + 3 summed
        errors.append(f"ordered: {ordered.scheduled_cycles} != 6")
    d = Dag(); r = d.region("overlap")
    _chain(r, P, 3); r.nodes[-1].is_output = True
    base = len(r.nodes)
    _chain(r, P, 3); r.nodes[-1].is_output = True
    overlap = evaluate(d, "overlap", parse_config("P=4,L=1,S=1"))
    if overlap.scheduled_cycles != 3:  # two independent chains overlap
        errors.append(f"overlap: {overlap.scheduled_cycles} != 3")

    # 6. Dead/disconnected work pushes makespan above a CP-bound aggregate.
    d = Dag(); r = d.region("dead")
    last = _chain(r, P, 3); r._by_id[last].is_output = True  # CP = 3
    for _ in range(20):
        r.load(kind="dead_load")  # disconnected, not output
    res = evaluate(d, "dead", parse_config("P=8,L=4,S=1"))
    if res.region_aggs[0].CP != 3:
        errors.append(f"dead: CP {res.region_aggs[0].CP} != 3")
    # load = ceil(20/4) = 5 > CP 3, so aggregate 5 and scheduled >= 5.
    if res.aggregate_cycles != 5 or res.scheduled_cycles < 5:
        errors.append(f"dead: agg/sched {res.aggregate_cycles}/"
                      f"{res.scheduled_cycles} (want agg 5, sched>=5)")

    # 7. Equal-priority stable tie-break (ascending node id) + determinism.
    d = Dag(); r = d.region("ties")
    ids = [r.arith(output=True) for _ in range(5)]
    cfg_one = parse_config("P=1,L=1,S=1")
    s1 = schedule_region(r, cfg_one)
    s2 = schedule_region(r, cfg_one)
    if s1.makespan != 5 or s2.makespan != 5:
        errors.append("ties: makespan != 5")
    # Re-schedule must be identical (determinism).
    if render_block(evaluate(d, "ties", cfg_one)) != \
            render_block(evaluate(d, "ties", cfg_one)):
        errors.append("ties: non-deterministic output")

    # 8. CP-bound aggregate, gap 0.
    d = Dag(); r = d.region("cpbound")
    last = _chain(r, P, 6); r._by_id[last].is_output = True
    res = evaluate(d, "cpbound", parse_config("P=8,L=8,S=8"))
    if res.gap_cycles != 0 or res.scheduled_cycles != 6:
        errors.append(f"cpbound: gap {res.gap_cycles}, sched "
                      f"{res.scheduled_cycles}")

    # 9. Resource-bound aggregate with a local-pressure gap (scheduled > agg).
    d = Dag(); r = d.region("rbound")
    root = r.arith()
    leaves = [r.arith(root, output=True) for _ in range(10)]
    res = evaluate(d, "rbound", parse_config("P=3,L=1,S=1"))
    # 1 root at cycle1, then 10 leaves at ceil(10/3)=4 cycles -> scheduled 5;
    # aggregate = max(CP=2, ceil(11/3)=4) = 4 -> gap 1.
    if res.scheduled_cycles <= res.aggregate_cycles:
        errors.append(f"rbound: expected scheduled > aggregate, got "
                      f"{res.scheduled_cycles} vs {res.aggregate_cycles}")

    # 10. A class with no ops (only P here).
    d = Dag(); r = d.region("noLS")
    last = _chain(r, P, 4); r._by_id[last].is_output = True
    res = evaluate(d, "noLS", parse_config("P=2,L=2,S=2"))
    if res.region_aggs[0].LD != 0 or res.region_aggs[0].ST != 0:
        errors.append("noLS: nonzero LD/ST")

    # 11. Empty DAG/region.
    d = Dag(); d.region("empty")
    res = evaluate(d, "empty", cfg)
    if not (res.aggregate_cycles == 0 and res.scheduled_cycles == 0
            and res.gap_cycles == 0 and res.gap_ratio == 1.0):
        errors.append("empty: metrics not all-zero / ratio!=1.0")

    # 12. Zero-capacity config rejected.
    for bad in ("P=0,L=1,S=1", "P=1,L=0,S=1", "P=1,L=1,S=0"):
        try:
            parse_config(bad)
            errors.append(f"zero-cap {bad}: not rejected")
        except ValueError:
            pass


def _run_golden_tests(errors):
    cfg = CONFIG_6x6
    expect = {
        "axpy": {"aggregate": 4,
                 "regions": [("axpy", 4, 32, 26, 16, 4)]},
        "autocorrelation": {"aggregate": 903,
                            "regions": [("autocorrelation", 11, 18064, 10834,
                                         3664, 903)]},
        "fft_butterfly": {"aggregate": 74,
                          "regions": [("copy", 2, 33, 49, 50, 5),
                                      ("s=1", 8, 183, 65, 90, 8),
                                      ("s=2", 11, 175, 61, 74, 11),
                                      ("s=3", 17, 171, 59, 66, 17),
                                      ("s=4", 33, 169, 58, 62, 33)]},
        "conv2d": {"aggregate": 2070,
                   "regions": [("conv2d", 17, 74515, 13716, 6220, 2070)]},
    }
    for kernel in PILOTS + ("conv2d",):
        dag, contract = build_kernel(kernel)
        try:
            aggs = check_contract(dag, contract, cfg)
        except AssertionError as exc:
            errors.append(f"{kernel}: {exc}")
            continue
        result = evaluate(dag, kernel, cfg)
        spec = expect[kernel]
        if result.aggregate_cycles != spec["aggregate"]:
            errors.append(f"{kernel}: aggregate {result.aggregate_cycles} != "
                          f"{spec['aggregate']}")
        rows = [(ra.name, ra.CP, ra.A, ra.LD, ra.ST, ra.aggregate)
                for ra in aggs]
        if rows != spec["regions"]:
            errors.append(f"{kernel}: region rows {rows} != {spec['regions']}")
        for ra, rs in zip(result.region_aggs, result.region_scheds):
            if rs.makespan < ra.aggregate:
                errors.append(f"{kernel}/{ra.name}: scheduled {rs.makespan} < "
                              f"aggregate {ra.aggregate}")
        # Determinism: identical rendered block across two runs.
        if render_block(evaluate(*build_and_eval(kernel, cfg))) != \
                render_block(result):
            errors.append(f"{kernel}: non-deterministic block")


def build_and_eval(kernel, cfg):
    dag, contract = build_kernel(kernel)
    check_contract(dag, contract, cfg)
    return dag, kernel, cfg


def _run_drift_test(errors):
    """Exercise apply_block / extract_block / check_eval without touching the
    real eval files (operate on in-memory text)."""
    cfg = CONFIG_6x6
    dag, contract = build_kernel("axpy")
    check_contract(dag, contract, cfg)
    block = render_block(evaluate(dag, "axpy", cfg))

    base = "# AXPY\n\nbody prose\n## ASAP Model Notes\nprotected\n"
    once = apply_block(base, "axpy", block)
    if "ASAP Model Notes\nprotected" not in once:
        errors.append("drift: protected ASAP notes not preserved")
    if extract_block(once, "axpy") != block:
        errors.append("drift: extract_block mismatch after write")
    # Idempotent: re-applying the same block changes nothing.
    twice = apply_block(once, "axpy", block)
    if twice != once:
        errors.append("drift: re-apply not idempotent")
    if once.count(marker_begin("axpy")) != 1:
        errors.append("drift: marker duplicated on re-apply")
    # Hand-edited numbers must be detected as drift.
    tampered = once.replace("**scheduled_cycles**", "**scheduled_cycles** XX")
    if extract_block(tampered, "axpy") == block:
        errors.append("drift: tamper not detected")


def run_self_tests() -> int:
    errors: list[str] = []
    _run_synthetic_tests(errors)
    _run_golden_tests(errors)
    _run_drift_test(errors)
    if errors:
        for e in errors:
            print(f"  SELF-TEST FAIL: {e}")
        return 1
    print("[PASS] cgra_schedule self-tests (synthetic + golden pilots + drift)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv) -> int:
    parser = argparse.ArgumentParser(
        description="CGRA aggregate lower bound + finite-resource estimate")
    sub = parser.add_subparsers(dest="cmd")

    p_report = sub.add_parser("report", help="print the canonical eval block")
    p_report.add_argument("kernel", choices=sorted(BUILDERS))
    p_report.add_argument("--config", default="6x6")

    p_write = sub.add_parser("write", help="write eval blocks (default: pilots)")
    p_write.add_argument("kernels", nargs="*")
    p_write.add_argument("--config", default="6x6")

    p_check = sub.add_parser("check", help="read-only drift check")
    p_check.add_argument("kernels", nargs="*")
    p_check.add_argument("--config", default="6x6")

    args = parser.parse_args(argv)

    if args.cmd == "report":
        print(report(args.kernel, parse_config(args.config)))
        return 0
    if args.cmd == "write":
        cfg = parse_config(args.config)
        kernels = args.kernels or list(PILOTS)
        for kernel in kernels:
            changed = write_eval(kernel, cfg)
            print(f"{kernel}: {'updated' if changed else 'unchanged'} "
                  f"{EVAL_PATHS[kernel].name}")
        return 0
    if args.cmd == "check":
        cfg = parse_config(args.config)
        kernels = args.kernels or list(PILOTS)
        problems = []
        for kernel in kernels:
            problems.extend(check_eval(kernel, cfg))
        if problems:
            for p in problems:
                print(f"  CHECK FAIL: {p}")
            return 1
        print(f"[PASS] check: {', '.join(kernels)} match freshly computed blocks")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        sys.exit(run_self_tests())
    if len(sys.argv) >= 2 and sys.argv[1] == "--check":
        sys.exit(main(["check"] + sys.argv[2:]))
    sys.exit(main(sys.argv[1:]))
