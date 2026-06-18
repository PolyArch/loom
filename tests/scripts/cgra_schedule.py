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
import tempfile
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
        # cls is always one of the resource-class names P/L/S, which are also
        # this object's capacity attributes.
        return getattr(self, cls)


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

    def induction(self, kind: str = "iv",
                  compare_depends_on_read: bool = True) -> dict:
        """A loop iterator step: read, increment, write-back, bound compare.
        Returns the four node ids. Sequential carries compare against the loaded
        iterator; fully-unrolled iterator compares are rooted counted overhead."""
        rd = self.load(kind=kind + "_load")
        inc = self.arith(rd, kind=kind + "_add")
        wr = self.store(inc, kind=kind + "_store")
        cmp_preds = [rd] if compare_depends_on_read else []
        cmp = self.arith(*cmp_preds, kind=kind + "_cmp")
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


class MultiCaseResult:
    def __init__(self, kernel, cfg, cases):
        self.kernel = kernel
        self.cfg = cfg
        self.cases = cases  # list[(case_name, KernelResult)]


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


def evaluate_multicase(kernel: str, cfg: Config) -> MultiCaseResult:
    cases = []
    for case_name, dag, contract in build_multicase_kernel(kernel):
        check_contract(dag, contract, cfg)
        cases.append((case_name, evaluate(dag, kernel, cfg)))
    return MultiCaseResult(kernel, cfg, cases)


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
    note = KERNEL_NOTES.get(result.kernel)
    if note:
        lines.append("")
        lines.append(note)
    lines.append("")
    lines.append(marker_end(result.kernel))
    return "\n".join(lines)


def render_multicase_block(result: MultiCaseResult) -> str:
    """Render a marker-bounded block for kernels whose eval reports multiple
    independent input cases. Cases are separate kernel invocations, so their
    cycles are not summed as ordered regions."""
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
    lines.append(
        "`wildcard_match` is reported per input case; these rows are separate "
        "kernel invocations and are **not** summed as ordered regions.")
    lines.append("")
    lines.append(
        "| case | CP | A | LD | ST | aggregate | scheduled | gap | ratio |")
    lines.append(
        "|------|---:|--:|---:|---:|----------:|----------:|----:|------:|")
    for case_name, case in result.cases:
        ra = case.region_aggs[0]
        lines.append(
            f"| {case_name} | {ra.CP} | {ra.A} | {ra.LD} | {ra.ST} | "
            f"{case.aggregate_cycles} | {case.scheduled_cycles} | "
            f"{case.gap_cycles} | {_fmt_ratio(case.gap_ratio)} |")
    lines.append("")
    lines.append("**Local `P`/`L`/`S` pressure by case** "
                 "(saturated cycles / longest saturated run / peak ready backlog):")
    for case_name, case in result.cases:
        lines.append(f"- `{case_name}`:")
        for c in CLASSES:
            pr = case.pressure[c]
            lines.append(
                f"  - `{c}`: {pr.saturated_cycles} / {pr.longest_run} / "
                f"{pr.peak_ready_backlog}")
    lines.append("")
    lines.append(marker_end(result.kernel))
    return "\n".join(lines)


# Per-kernel reconciliation notes appended inside the block (deterministic, so
# the read-only --check still matches byte-for-byte).
KERNEL_NOTES = {
    "fft_butterfly": (
        "> The `copy` row carries the three kernel-once residual ops — the `N` "
        "load, the `log2f(N)` transcendental, and the `s`-loop init store — "
        "that overlap the copy phase, so its `A`/`LD`/`ST` is the documented "
        "copy phase (`32`/`48`/`49`) plus those three. This leaves the copy "
        "aggregate at `5` and the stage-sum aggregate at `74`, matching the "
        "section above."),
}


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
        # Parallel iterator overhead: compare is counted, but each lane's i is
        # a rooted constant in the fully-unrolled DAG.
        r.induction(kind="i", compare_depends_on_read=False)
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
            # `lag` (parallel) and `i` (reduction) are both fully unrolled, so
            # each is a per-lane compile-time constant available at cycle 1. The
            # `i+lag` address-add therefore has no predecessor edge from the
            # induction reads -- it is a cycle-1 root (see the "fully-unrolled
            # iterators are per-lane constants" rule in
            # docs/spec-kernel-performance.md). This is what gives CP=11; wiring
            # the induction reads in would make it CP=12 and break the eval.
            aa = r.address_add(kind="addr_i_plus_lag")  # &x[i+lag]
            ld_a = r.load(kind="x_i")                    # bare subscript
            ld_b = r.load(aa, kind="x_i_plus_lag")
            products.append(r.arith(ld_a, ld_b, kind="mul"))
            # Inner induction: counted overhead, off the output-reachable path
            # (reduction dim -> i is a per-lane constant, not a carried value).
            r.induction(kind="i", compare_depends_on_read=False)
        root = r.balanced_reduction(products, kind="reduce")
        r.store(root, output=True, kind="output_lag")
        # Outer induction: counted overhead, off path (lag is a parallel-dim
        # per-lane constant).
        r.induction(kind="lag", compare_depends_on_read=False)
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
        c.induction(kind="copy_i", compare_depends_on_read=False)
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
            # Per-block initializers store compile-time constants: w_r=1.0f,
            # w_i=0.0f, j=0. They are counted ops (memory-backed scalars), but
            # the first body reads below consume those *constants*, which the
            # ASAP baseline makes available at cycle 1 (the constant / loop-
            # invariant rule) -- exactly as the committed eval derives
            # "w^(0) ready at cycle 1". So the j=0 reads are modeled as roots
            # (no edge from the init store), not RAW-dependent on the init
            # writes. Wiring an init->first-read edge would push every stage's
            # carried chain by one cycle (s=1 CP 8->9, ... s=4 33->34, phase sum
            # 74->78), contradicting the documented per-stage CP and golden
            # aggregate 74. See docs/spec-kernel-performance.md (constant-init
            # reads). Only iterations j>=1 carry a real RAW edge from the prior
            # store.
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
                # j=0 reads the constant 0 (a root, per the rule above).
                ld_j = rg.load(prev_stj, kind="j_load") if prev_stj is not None \
                    else rg.load(kind="j_load")
                add_j = rg.arith(ld_j, kind="j_add")
                prev_stj = rg.store(add_j, kind="j_store")
                rg.arith(ld_j, kind="j_cmp")
                prev_wr, prev_wi = _fft_butterfly(
                    rg, ld_wr, ld_wi, cos, sin, ld_j)
            rg.induction(kind="k", compare_depends_on_read=False)
        # The stage loop is materialized into explicit stage regions; ordinary
        # source loop-control work is counted overhead, not the stage RAW carry.
        rg.induction(kind="s", compare_depends_on_read=False)

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
        r.induction(kind="iv", compare_depends_on_read=False)

    contract = [RegionContract("conv2d", A=74515, LD=13716, ST=6220, CP=17,
                               aggregate=2070)]
    return dag, contract


def build_batchnorm(C=4, H=8, W=8):
    """Batch normalization, single region, all three dims (c,h,w) parallel.

    output[c,h,w] = gamma[c]*(input[idx]-mean[c])*inv_std[c] + beta[c], with
    inv_std[c] = 1.0/sqrt(variance[c]+eps) and idx = c*(H*W) + h*W + w. All
    C*H*W pixel lanes fully unroll and overlap (no carried dependence). The
    per-channel inv_std/mean/gamma/beta and H*W are loop-invariant across (h,w):
    each channel loads variance/mean/gamma/beta once, and H*W is a hoisted
    cycle-1 root (the precomputed product).

    The binding chain (CP=10) is the per-pixel index/normalize path:
    H*W hoist -> c*HW -> +h*W -> +w (idx) -> load input -> sub mean ->
    *inv_std -> *gamma -> +beta -> store. The index multiplies are *regular*
    arithmetic producing the named scalar idx (address_adds = 0, since the
    access is the bare subscript input[idx]). c/h/w are fully-unrolled parallel
    iterators -> per-lane compile-time constants whose induction reads lie off
    the output-reachable path (dead w.r.t. CP, per the spec's "fully-unrolled
    iterators are per-lane constants" rule); the index arithmetic therefore does
    not depend on those reads. The chain's cycle-1 root is the loop-invariant
    H*W product (a precomputed mul), so c*HW sits at depth 2 -- the same depth
    the iterator read previously gave, leaving CP=10 unchanged while moving the
    iterator reads off the path. Reproduces the eval golden numbers (C=4,H=W=8):
    CP=10, A=2645, LD=568, ST=548, aggregate 6x6 = 74.
    """
    dag = Dag()
    r = dag.region("batchnorm")
    # Global hoisted scalar loads (eps,C,H,W) + the loop-invariant H*W product,
    # rooted as a precomputed constant available at cycle 1.
    ld_eps = r.load(kind="eps")
    r.load(kind="C")
    r.load(kind="H")
    ld_W = r.load(kind="W")
    hw = r.arith(kind="HW_hoist")  # H*W hoist -> cycle-1 root
    for _ in range(C):
        c_iv = r.induction(kind="c", compare_depends_on_read=False)
        ld_var = r.load(kind="variance")
        ld_mean = r.load(kind="mean")
        ld_gamma = r.load(kind="gamma")
        ld_beta = r.load(kind="beta")
        # Per-channel inv_std = 1.0 / sqrt(variance + eps); finishes by depth 4
        # and overlaps the longer per-pixel address chain.
        add_ve = r.arith(ld_var, ld_eps, kind="var_plus_eps")
        sq = r.arith(add_ve, kind="sqrt")
        inv_std = r.arith(sq, kind="inv_std_div")  # 1.0 / sqrt (const numerator)
        for _ in range(H):
            h_iv = r.induction(kind="h", compare_depends_on_read=False)
            for _ in range(W):
                w_iv = r.induction(kind="w", compare_depends_on_read=False)
                # idx = c*HW + h*W + w. c/h/w are fully-unrolled *parallel*
                # iterators, hence per-lane compile-time constants; their
                # induction reads are counted overhead that must lie off the
                # output-reachable path (dead w.r.t. CP). So the index arithmetic
                # does NOT take edges from c_iv/h_iv/w_iv["read"]; it depends only
                # on the loop-invariant H*W product (hoisted, depth-1 root) and W
                # load. The binding chain therefore runs through the H*W hoist ->
                # c*HW (depth 2) -> ... keeping CP=10, with the parallel iterator
                # reads off the path per docs/spec-kernel-performance.md.
                m_chw = r.arith(hw, kind="c_mul_HW")        # c (const) * HW
                m_hw = r.arith(ld_W, kind="h_mul_W")        # h (const) * W
                a1 = r.arith(m_chw, m_hw, kind="cHW_plus_hW")
                idx = r.arith(a1, kind="idx_plus_w")        # + w (const)
                ld_in = r.load(idx, kind="input")          # input[idx]
                sub = r.arith(ld_in, ld_mean, kind="input_minus_mean")
                norm = r.arith(sub, inv_std, kind="mul_inv_std")
                mg = r.arith(norm, ld_gamma, kind="mul_gamma")
                ab = r.arith(mg, ld_beta, kind="add_beta")
                r.store(ab, output=True, kind="output")  # output[idx]
    contract = [RegionContract("batchnorm", A=2645, LD=568, ST=548, CP=10,
                               aggregate=74)]
    return dag, contract


def build_bit_reverse(N=256, BITS=32):
    """Per-element 32-bit reversal. Outer i-loop is parallel (each lane writes a
    distinct output_reversed[i], no cross-lane carry) and fully unrolled; the
    inner bit-loop is sequential, carrying two scalar recurrences:
      result = (result << 1) | (value & 1);  value >>= 1;
    The `result` recurrence (load result -> <<1 -> | -> store result) is the
    II=4 chain that sets the per-iter critical path; `value` (load -> >>1 ->
    store) and the `bit` induction are II=3 and slack. `value` is read twice per
    iter with no intervening write, so it is loaded once and fanned to `&1` and
    `>>1`. `result` and `bit` are initialized to the constant 0, so their first
    reads are rooted (constant-init carry); `value` is initialized to the loaded
    input_data[i], so its first read depends on the prologue store.

    Single region (no in-place RAW across lanes). Reproduces the eval golden
    numbers (N=256, BITS=32): CP=132, A=49664, LD=25345, ST=25600,
    aggregate 6x6 = 2134.
    """
    dag = Dag()
    r = dag.region("bit_reverse")
    r.load(kind="N")  # hoisted N-param load, charged once
    for _ in range(N):
        # Outer induction (i): parallel dim. The bound compare is rooted and the
        # whole i induction (read, increment, store, compare) is counted overhead
        # that lies off the output-reachable path -- a fully-unrolled parallel
        # iterator is a per-lane compile-time constant (see the "fully-unrolled
        # iterators are per-lane constants" rule in docs/spec-kernel-performance.md),
        # so it must be dead with respect to CP.
        ld_i = r.load(kind="i_load")
        i_add = r.arith(ld_i, kind="i_add")
        r.store(i_add, kind="i_store")
        r.arith(kind="i_cmp")  # i < N, rooted
        # Prologue: value = input_data[i]; result = 0 (constant store).
        ld_in = r.load(kind="input_data")  # bare [i] on a per-lane const i: rooted,
                                           # no edge from the induction read
        st_value = r.store(ld_in, kind="value_init")
        r.store(kind="result_init")  # result = 0 (literal; rooted store)
        # Sequential bit-loop: result (II=4) and value (II=3) carries.
        prev_r = None       # result init = const 0 -> first read rooted
        prev_v = st_value   # value init = input_data[i] -> first read RAW
        prev_bit = None     # bit init = const 0 -> first read rooted
        for _ in range(BITS):
            ld_r = r.load(prev_r, kind="result") if prev_r is not None \
                else r.load(kind="result")
            ld_v = r.load(prev_v, kind="value")  # one load fanned to &1 and >>1
            ld_bit = r.load(prev_bit, kind="bit") if prev_bit is not None \
                else r.load(kind="bit")
            shl = r.arith(ld_r, kind="shl")          # result << 1
            band = r.arith(ld_v, kind="band")        # value & 1
            bor = r.arith(shl, band, kind="bor")     # |
            shr = r.arith(ld_v, kind="shr")          # value >> 1
            prev_r = r.store(bor, kind="result_store")
            prev_v = r.store(shr, kind="value_store")
            add_bit = r.arith(ld_bit, kind="bit_add")
            prev_bit = r.store(add_bit, kind="bit_store")
            r.arith(ld_bit, kind="bit_cmp")          # bit < 32 (seq -> read dep)
        # Epilogue: output_reversed[i] = result.
        ld_res = r.load(prev_r, kind="result_final")
        r.store(ld_res, output=True, kind="output")
    contract = [RegionContract("bit_reverse", A=49664, LD=25345, ST=25600,
                               CP=132, aggregate=2134)]
    return dag, contract


def build_bisection_step(N=64):
    """Bisection midpoint selection, parallel over i (fully unrolled, single
    region). Per lane:
        c = (input_a[i] + input_b[i]) * 0.5;
        if (input_fa[i]*input_fc[i] < 0) { output_a = input_a[i]; output_b = c; }
        else                             { output_a = c; output_b = input_b[i]; }
    Under strict no-predication, the gating compare must retire before any op
    inside the if/else body, so both conditional output stores take an edge from
    the compare (giving CP=4). Only the taken arm is counted; both arms write the
    same two addresses, so the store count is identical either way -- the model
    wires the T arm (output_a = a, output_b = c). All subscripts are bare [i]:
    no address-add, and (like axpy) the input loads are rooted -- the
    fully-unrolled iterator is a per-lane constant, so the loads have no parent
    and the induction work stays off the output path. Reproduces the eval golden
    numbers (N=64): CP=4, A=384, LD=321, ST=192, aggregate 6x6 = 27.
    """
    dag = Dag()
    r = dag.region("bisection_step")
    r.load(kind="N")  # hoisted param load, charged once
    for _ in range(N):
        ld_a = r.load(kind="input_a")
        ld_b = r.load(kind="input_b")
        ld_fa = r.load(kind="input_fa")
        ld_fc = r.load(kind="input_fc")
        add = r.arith(ld_a, ld_b, kind="a_plus_b")
        c = r.arith(add, kind="mul_half")          # (a + b) * 0.5
        mul_p = r.arith(ld_fa, ld_fc, kind="fa_mul_fc")
        cmp = r.arith(mul_p, kind="cmp_lt0")        # fa*fc < 0
        # No-predication gate: the taken arm's stores wait for the compare.
        r.store(ld_a, cmp, output=True, kind="output_a")  # T arm: output_a = a
        r.store(c, cmp, output=True, kind="output_b")     # T arm: output_b = c
        r.induction(kind="i", compare_depends_on_read=False)
    contract = [RegionContract("bisection_step", A=384, LD=321, ST=192, CP=4,
                               aggregate=27)]
    return dag, contract


def build_bitonic_stage(N=8):
    """One bitonic compare-exchange stage, parallel over i (fully unrolled,
    single region). For the documented test vector (N=8, stage=1, pass=0 ->
    distance=1, block_size=4, input [3,1,4,2,8,6,7,5]) the four nested
    no-predication gates split the lanes into three types:
      - skipped (i in {1,3,5,7}): outer_pred = F, body never entered -- chain
        ends at the outer compare (depth 6);
      - active non-swap (i in {4,6}): outer_pred = T, partner < N = T,
        should_swap = F -- partner add, bound check, two inplace loads, one
        value compare (cmp_lt, the taken arm of `if (ascending)`), no stores
        (depth 10);
      - swap (i in {0,2}): as active, plus the two inplace swap stores (depth 11).
    Each gating compare takes an edge into every op of the body it guards, and
    only the taken arm is counted. The dead `half_block = block_size >> 1` is
    loop-invariant, so it is counted once and stays off the output path.
    Loop-invariant distance/block_size are prologue dataflow broadcast to all
    lanes; inplace subscripts are bare scalars (no address_add);
    inplace[i]/inplace[partner] are each loaded once and fanned to the value
    compare and the swap store. Reproduces the eval golden numbers
    (N=8, distance=1): CP=11, A=80, LD=19,
    ST=12, aggregate 6x6 = 11.
    """
    dag = Dag()
    r = dag.region("bitonic_stage")
    # Prologue (loop-invariant, broadcast): distance = 1<<pass, block_size =
    # 1<<(stage+1); stage/pass/N are hoisted param loads.
    ld_stage = r.load(kind="stage")
    ld_pass = r.load(kind="pass")
    ld_N = r.load(kind="N")
    stage_p1 = r.arith(ld_stage, kind="stage_plus_1")      # add
    distance = r.arith(ld_pass, kind="distance_shl")        # 1 << pass (bitop)
    block_size = r.arith(stage_p1, kind="block_size_shl")   # 1 << (stage+1) (bitop)
    r.arith(block_size, kind="half_block_shr")              # bs>>1 (dead bitop)

    # Lane composition for the documented inputs: 4 skipped, 2 active non-swap,
    # 2 swap. (active, swap) flags select the taken-arm-only op set per lane.
    lanes = ([(False, False)] * 4    # i in {1,3,5,7}: outer_pred = F
             + [(True, False)] * 2   # i in {4,6}: active, should_swap = F
             + [(True, True)] * 2)   # i in {0,2}: active swap
    for active, swap in lanes:
        iv = r.induction(kind="i", compare_depends_on_read=False)
        i_rd = iv["read"]
        # Unconditional per-lane compute (before the gates, every lane).
        block_idx = r.arith(i_rd, block_size, kind="block_idx_div")        # i / bs
        idx_in_block = r.arith(i_rd, block_size, kind="idx_in_block_mod")  # i % bs
        band_asc = r.arith(block_idx, kind="block_idx_and_1")  # block_idx & 1
        ascending = r.arith(band_asc, kind="ascending_cmp")    # == 0 (compare)
        band_pred = r.arith(idx_in_block, distance, kind="idx_and_distance")
        outer_pred = r.arith(band_pred, kind="outer_pred_cmp")  # == 0 (compare)
        if active:
            # Inside outer body: partner = i + distance (gated by outer_pred).
            partner = r.arith(i_rd, distance, outer_pred, kind="partner_add")
            in_bounds = r.arith(partner, ld_N, kind="partner_lt_N")  # compare
            # Inside partner<N body: two inplace loads (bare-scalar subscripts).
            ld_ip_i = r.load(in_bounds, kind="inplace_i")
            ld_ip_p = r.load(in_bounds, partner, kind="inplace_partner")
            # Inside if(ascending): only the taken-arm value compare fires.
            should_swap = r.arith(ascending, ld_ip_i, ld_ip_p, kind="value_cmp")
            if swap:
                # Inside if(should_swap): swap stores reuse the C9 loads.
                r.store(ld_ip_p, should_swap, output=True, kind="store_inplace_i")
                r.store(ld_ip_i, should_swap, output=True,
                        kind="store_inplace_partner")
    contract = [RegionContract("bitonic_stage", A=80, LD=19, ST=12, CP=11,
                               aggregate=11)]
    return dag, contract


def build_bitonic_stage_modified(N=8):
    """Modified bitonic stage: the active (if) branch appends an in-place
    `for j in [N/2, N): inplace[j] *= 2` loop and the else branch does
    `inplace[i] -= 1`. The outer `i` loop is therefore SEQUENTIAL -- a
    read-modify-write recurrence aliases through inplace[N/2..N-1] across
    successive if-iters (and the i in {5,7} else writes). Single region; the
    in-place carry is modeled as RAW memory edges (each writer's load depends on
    the previous committing writer's store to that slot).

    Because the dim is sequential, its iterator is NOT a per-lane constant: per
    docs/spec-kernel-performance.md ("fully-unrolled iterators are per-lane
    constants" and its deliberate sequential contrast), a sequential-dim
    iterator read is part of the carried chain, so each iter's `load i` chains
    from the prior iter's `store i` (the same treatment the FFT butterfly `j`
    gets in build_fft_butterfly). The induction link load i -> i+1 -> store i is
    II=3, and 8 iters chain as 7 links -- this is the loop's carried recurrence.

    For the documented inputs (N=8, stage=1, pass=0 -> distance=1,
    block_size=4, input [3,1,4,2,8,6,7,5]):
      - i in {0,2,4,6} take the if branch; i in {1,3,5,7} the else.
      - should_swap = 1 only on i in {0,2} (commit swap of inplace[0,1]/[2,3],
        disjoint from the carried slice); i in {4,6} load+compare but do not
        store.
    The binding chain is the sequential-iterator induction chain: block_size at
    depth 3 -> iter0 load i at depth 4 -> ... -> iter7 load i at depth 25 (seven
    II=3 links), then iter7's predicate (i%bs -> & distance -> ==0 outer_pred at
    depth 28) feeds its else write inplace[7] (load 29 -> sub 30 -> store 31).
    This 8-deep iterator chain dwarfs the 5-link inplace memory recurrence, so it
    sets CP=31. The first iter's `load i` takes block_size as its depth floor;
    later iters chain from the prior committing `store i`.

    Reproduces the eval golden numbers (N=8, distance=1): CP=31, A=133, LD=55,
    ST=48, aggregate 6x6 = 31.
    """
    dag = Dag()
    r = dag.region("bitonic_stage-modified")
    # Prologue (loop-invariant, broadcast): distance=1<<pass, stage+1,
    # N/2=N>>1 (j-loop init), block_size=1<<(stage+1).
    ld_pass = r.load(kind="pass")
    ld_stage = r.load(kind="stage")
    ld_N = r.load(kind="N")
    distance = r.arith(ld_pass, kind="distance_shl")       # 1 << pass (bitop)
    stage_p1 = r.arith(ld_stage, kind="stage_plus_1")       # add
    r.arith(ld_N, kind="n_half_shr")                        # N >> 1 (bitop)
    block_size = r.arith(stage_p1, kind="block_size_shl")   # 1<<(stage+1) (bitop)
    r.arith(block_size, kind="half_block_shr")              # bs>>1 (dead bitop)

    half = N // 2
    last_writer = {}  # inplace slot -> last committing store node (None = init)
    prev_i_store = None  # sequential iterator carry: prior iter's `store i`

    for i in range(N):
        active = (i % 2 == 0)
        # Sequential-dim iterator carry: iter k's `load i` chains from iter
        # k-1's `store i` (read-after-write on the loop counter). The induction
        # link load i -> i+1 -> store i is II=3 and is the recurrence that sets
        # the critical path; the first iter's read takes block_size as its depth
        # floor (loop-invariant prologue). The bound compare is rooted overhead.
        if prev_i_store is not None:
            ld_i = r.load(prev_i_store, kind="i_load")
        else:
            ld_i = r.load(block_size, kind="i_load")
        i_add = r.arith(ld_i, kind="i_add")
        prev_i_store = r.store(i_add, kind="i_store")
        r.arith(kind="i_cmp")  # i < N, rooted
        # Unconditional predicate (every lane).
        block_idx = r.arith(ld_i, block_size, kind="block_idx_div")       # i / bs
        idx_in_block = r.arith(ld_i, block_size, kind="idx_in_block_mod")  # i % bs
        band_asc = r.arith(block_idx, kind="block_idx_and_1")
        ascending = r.arith(band_asc, kind="ascending_cmp")  # == 0 (compare)
        band_pred = r.arith(idx_in_block, distance, kind="idx_and_distance")
        outer_pred = r.arith(band_pred, kind="outer_pred_cmp")  # == 0 (compare)
        if active:
            # --- outer-if body, gated by outer_pred ---
            partner = r.arith(ld_i, distance, outer_pred, kind="partner_add")
            in_bounds = r.arith(partner, ld_N, kind="partner_lt_N")  # compare
            # Compare-swap loads (bare-scalar subscripts), gated by partner<N;
            # they read the carried slots i and partner=i+1.
            pre_i = last_writer.get(i)
            pre_p = last_writer.get(i + 1)
            ld_cs_i = r.load(in_bounds, *( [pre_i] if pre_i is not None else [] ),
                             kind="cs_inplace_i")
            ld_cs_p = r.load(in_bounds, partner,
                             *( [pre_p] if pre_p is not None else [] ),
                             kind="cs_inplace_partner")
            should_swap = r.arith(ascending, ld_cs_i, ld_cs_p, kind="value_cmp")
            if i in (0, 2):  # should_swap = 1 -> commit (off the carried slice)
                st_i = r.store(ld_cs_p, should_swap, output=True, kind="swap_i")
                st_p = r.store(ld_cs_i, should_swap, output=True,
                               kind="swap_partner")
                last_writer[i] = st_i
                last_writer[i + 1] = st_p
            # --- j-loop: for j in [half, N): inplace[j] *= 2 (parallel within
            # the if-iter; serialized across if-iters via the memory carry). ---
            for j in range(half, N):
                r.induction(kind="j", compare_depends_on_read=False)
                prev = last_writer.get(j)
                ld_j = r.load(outer_pred, *( [prev] if prev is not None else [] ),
                              kind="jloop_inplace")
                mul = r.arith(ld_j, kind="jloop_mul")       # *= 2
                last_writer[j] = r.store(mul, output=True, kind="jloop_store")
        else:
            # --- else body, gated by ~outer_pred (same compare retires at 7);
            # inplace[i] -= 1 carries through the slice for i in {5,7}. ---
            prev = last_writer.get(i)
            ld_e = r.load(outer_pred, *( [prev] if prev is not None else [] ),
                          kind="else_inplace")
            sub = r.arith(ld_e, kind="else_sub")            # -= 1
            last_writer[i] = r.store(sub, output=True, kind="else_store")
    contract = [RegionContract("bitonic_stage-modified", A=133, LD=55, ST=48,
                               CP=31, aggregate=31)]
    return dag, contract


def build_binary_search(N=10, M=5):
    """Binary search of M targets over a sorted array of N (data-dependent
    termination). Outer t is parallel (each target privatizes left/right/result
    and writes a distinct output_indices[t]); the inner while is sequential with
    an input-dependent trip count, carrying left/right (and result on break) via
    scalar. Single region.

    For the documented inputs (sorted=[1,3,..,19], targets=[7,2,15,20,1]) the
    per-target probe paths are fixed: trips {4,3,2,4,3} with exits
    {break, fail, break, fail, break} and the update sequences traced below.
    Each non-break inner iter is a 10-cycle no-predication recurrence
    (load left/right -> bound compare -> right-left -> >>1 -> +left = mid ->
    load sorted[mid] -> cmp_eq -> cmp_lt -> update -> store), gated by three
    nested compares; the carry advances 10 per iter regardless of which of
    left/right is updated. A break iter stops at the cmp_eq result store (8
    cycles); a non-break exit pays a final failing bound compare. The post-loop
    `(result == -1) ? ... ` read of result waits on the loop-termination compare
    (data-dependent-termination: the exit compare is on the critical path), so
    the binding target is the deepest exit, t=3 (target=20, trip 4, non-break):
    depth 10*4 + 8 = 48. Reproduces the eval golden numbers (N=10, M=5):
    CP=48, A=124, LD=69, ST=41, aggregate 6x6 = 48.
    """
    dag = Dag()
    r = dag.region("binary_search")
    # Hoisted loop-invariants: N, M loads; right-init = N - 1 (charged once).
    ld_N = r.load(kind="N")
    r.load(kind="M")
    n_minus_1 = r.arith(ld_N, kind="N_minus_1")  # sub, hoisted

    # (update sequence for the full iters, is_break). 'add' = left = mid+1
    # (cmp_lt true), 'sub' = right = mid-1 (cmp_lt false). Break targets append
    # one terminal cmp_eq iter; non-break targets append a failing bound check.
    targets = [
        (["sub", "add", "add"], True),          # t0 target 7,  trip 4, break
        (["sub", "sub", "add"], False),         # t1 target 2,  trip 3, fail
        (["add"], True),                        # t2 target 15, trip 2, break
        (["add", "add", "add", "add"], False),  # t3 target 20, trip 4, fail
        (["sub", "sub"], True),                 # t4 target 1,  trip 3, break
    ]

    def inner_iter(target, last_left, last_right):
        """Emit the shared head of one inner iter; returns (cmp_eq, mid, ld_l,
        ld_r). The bound compare gates the body; cmp_eq gates the else."""
        ld_l = r.load(last_left, kind="left")
        ld_r = r.load(last_right, kind="right")
        cmp_le = r.arith(ld_l, ld_r, kind="bound_le")            # left <= right
        sub_rl = r.arith(ld_r, ld_l, cmp_le, kind="right_minus_left")
        shift = r.arith(sub_rl, kind="half_shift")               # >> 1
        mid = r.arith(shift, ld_l, kind="mid_add")               # left + (..)
        ld_s = r.load(mid, kind="sorted_mid")                    # sorted[mid]
        cmp_eq = r.arith(ld_s, target, kind="eq_target")         # == target
        return cmp_eq, mid, ld_s

    for updates, is_break in targets:
        r.induction(kind="t", compare_depends_on_read=False)  # parallel, off path
        target = r.load(kind="input_targets")    # bare [t], anonymous; rooted
        last_left = r.store(kind="left_init")     # left = 0 (const)
        last_right = r.store(n_minus_1, kind="right_init")  # right = N - 1
        r.store(kind="result_init")               # result = -1 (const)
        for upd in updates:  # full iters (cmp_eq false -> cmp_lt -> update)
            cmp_eq, mid, ld_s = inner_iter(target, last_left, last_right)
            cmp_lt = r.arith(cmp_eq, ld_s, target, kind="lt_target")  # < target
            update = r.arith(cmp_lt, mid,
                             kind="upd_add" if upd == "add" else "upd_sub")
            st = r.store(update,
                         kind="left_store" if upd == "add" else "right_store")
            if upd == "add":
                last_left = st
            else:
                last_right = st
        if is_break:
            # Terminal iter: cmp_eq true -> result = mid, break.
            cmp_eq, mid, _ = inner_iter(target, last_left, last_right)
            exit_node = r.store(mid, cmp_eq, kind="result_store")
        else:
            # Failing bound check: left <= right is false (termination compare).
            ld_l = r.load(last_left, kind="left")
            ld_r = r.load(last_right, kind="right")
            exit_node = r.arith(ld_l, ld_r, kind="bound_le_fail")
        # Post-loop: output_indices[t] = (result == -1) ? 0xFFFFFFFF : result.
        # The result read waits for the loop to terminate (exit_node on the CP).
        ld_res = r.load(exit_node, kind="result_final")
        cmp_res = r.arith(ld_res, kind="result_eq_neg1")
        r.store(cmp_res, output=True, kind="output_indices")
    contract = [RegionContract("binary_search", A=124, LD=69, ST=41, CP=48,
                               aggregate=48)]
    return dag, contract


def build_gather(N=1024, src_size=256):
    """Indirect read gather for the concrete main.cpp inputs.

    All generated indices are valid (`idx = (i*3) % src_size`), so every lane
    takes the in-bounds arm: load indices[i] -> compare idx<src_size -> load
    src[idx] -> store dst[i]. The outer iterator is fully unrolled; its
    induction work is counted but stays off the output-reachable path, and all
    subscripts are bare variables/scalars (no address_adds). Reproduces the eval
    numbers (N=1024,V=1024): CP=4, A=3072, LD=3074, ST=2048, aggregate 257.
    """
    dag = Dag()
    r = dag.region("gather")
    r.load(kind="N")
    r.load(kind="src_size")
    for _ in range(N):
        ld_idx = r.load(kind="indices")
        cmp = r.arith(ld_idx, kind="idx_lt_src_size")
        ld_src = r.load(cmp, ld_idx, kind="src_idx")
        r.store(ld_src, cmp, output=True, kind="dst")
        r.induction(kind="i", compare_depends_on_read=False)
    contract = [RegionContract("gather", A=3072, LD=3074, ST=2048, CP=4,
                               aggregate=257)]
    return dag, contract


def build_edge_update(E=16, K=2):
    """Single CSR edge update for the concrete main.cpp trace.

    Copy and search stay in one region: the later matched update overwrites one
    copied output_weights slot, but the copy value is never read before being
    overwritten, so this is a WAW inside one schedulable region rather than a
    barrier. The binding chain is the bounds check -> row_ptr[src+1] load ->
    matched col_indices load -> match compare -> update store, giving CP=6.
    Reproduces the eval numbers (E=16,K=2): A=40, LD=38, ST=37.
    """
    dag = Dag()
    r = dag.region("edge_update")

    # Copy loop, fully unrolled. Each copied slot is a terminal output unless
    # later overwritten; marking them all as outputs does not affect CP because
    # the matched update chain is deeper.
    r.store(kind="copy_i_init")
    for _ in range(E):
        ld_w = r.load(kind="input_weights")
        r.store(ld_w, output=True, kind="copy_output_weights")
        r.induction(kind="copy_i", compare_depends_on_read=False)

    # Bounds + row-pointer chain.
    bounds = r.arith(kind="src_ge_num_nodes")
    r.load(bounds, kind="row_start")
    aa = r.address_add(bounds, kind="row_ptr_src_plus_1")
    row_end = r.load(aa, kind="row_end")

    # Search loop overhead for the K executed iterations plus its init store.
    r.store(kind="search_i_init")
    for _ in range(K):
        r.induction(kind="search_i", compare_depends_on_read=False)

    # Two concrete scan iterations: the first misses, the second matches.
    ld_col0 = r.load(row_end, kind="col_indices_miss")
    r.arith(ld_col0, kind="match_cmp_miss")
    ld_col1 = r.load(row_end, kind="col_indices_match")
    match = r.arith(ld_col1, kind="match_cmp_hit")
    r.store(match, output=True, kind="updated_output_weight")

    contract = [RegionContract("edge_update", A=40, LD=38, ST=37, CP=6,
                               aggregate=6)]
    return dag, contract


def build_interpolate_linear(N_data=32, N_query=64):
    """Linear interpolation for the concrete main.cpp trace.

    The outer query loop is parallel; each lane runs a private linear search and
    writes one output. The documented inputs have 1024 total interval probes,
    63 hit queries, and one no-hit query (xq=31.5) that pays the final failing
    k-bound check before falling through with i=0. Reproduces the eval numbers:
    CP=289, A=5699, LD=3523, ST=1216, aggregate 6x6 = 294.
    """
    dag = Dag()
    r = dag.region("interpolate_linear")

    # Hoisted parameter work. N_data - 1 is counted as a loop-invariant sub;
    # the bound value is broadcast to the search checks.
    r.load(kind="N_query")
    ld_n_data = r.load(kind="N_data")
    r.arith(ld_n_data, kind="N_data_minus_1")

    for q in range(N_query):
        r.induction(kind="q", compare_depends_on_read=False)
        ld_xq = r.load(kind="input_xq")
        i_init = r.store(kind="i_init")

        xq_value = 0.5 * q
        hit = False
        last_k_store = None
        exit_node = None

        for k in range(N_data - 1):
            k_preds = [last_k_store] if last_k_store is not None else []
            ld_k = r.load(*k_preds, kind="k")
            k_bound = r.arith(ld_k, kind="k_lt_bound")
            ld_xk = r.load(k_bound, kind="input_x_k")
            cmp_lo = r.arith(ld_xk, ld_xq, kind="xq_ge_xk")
            k_plus_1 = r.address_add(ld_k, cmp_lo, kind="k_plus_1")
            ld_xkp1 = r.load(k_plus_1, kind="input_x_k_plus_1")
            cmp_hi = r.arith(ld_xkp1, ld_xq, kind="xq_le_xkp1")

            if xq_value >= float(k) and xq_value <= float(k + 1):
                exit_node = r.store(cmp_hi, ld_k, kind="i_hit_store")
                hit = True
                break

            k_inc = r.arith(cmp_hi, ld_k, kind="k_inc")
            last_k_store = r.store(k_inc, kind="k_store")

        if not hit:
            # One final failing for-loop check. It gates the interpolation tail
            # but does not execute a body or k++.
            ld_k = r.load(last_k_store, kind="k_final")
            exit_node = r.arith(ld_k, kind="k_bound_fail")

        ld_i_preds = [exit_node]
        if not hit:
            ld_i_preds.append(i_init)
        ld_i = r.load(*ld_i_preds, kind="i_final")

        ld_x0 = r.load(ld_i, kind="input_x_i")
        ld_y0 = r.load(ld_i, kind="input_y_i")
        x_i_p1 = r.address_add(ld_i, kind="x_i_plus_1")
        y_i_p1 = r.address_add(ld_i, kind="y_i_plus_1")
        ld_x1 = r.load(x_i_p1, kind="input_x_i_plus_1")
        ld_y1 = r.load(y_i_p1, kind="input_y_i_plus_1")

        num = r.arith(ld_xq, ld_x0, kind="xq_minus_x0")
        den = r.arith(ld_x1, ld_x0, kind="x1_minus_x0")
        t = r.arith(num, den, kind="div_t")
        dy = r.arith(ld_y1, ld_y0, kind="y1_minus_y0")
        prod = r.arith(t, dy, kind="mul_t_dy")
        out = r.arith(ld_y0, prod, kind="add_y0")
        r.store(out, output=True, kind="output_yq")

    contract = [RegionContract("interpolate_linear", A=5699, LD=3523,
                               ST=1216, CP=289, aggregate=294)]
    return dag, contract


def build_bitonic_stage_tweak(N=8):
    """Bitonic stage tweak: baseline compare-swap plus active-lane `++` and
    unconditional `-=1`.

    The concrete inputs have active lanes 0,2,4,6 and swap commits only on 0,2.
    This builder tracks the latest writer to each inplace slot so the same-slot
    RAWs (swap -> ++ -> -=1) and partner RAWs (swap i -> odd partner decrement)
    are represented inside the single region. Final decrement stores are marked
    as the terminal outputs. Reproduces the eval numbers: CP=17, A=92, LD=31,
    ST=24, aggregate 17.
    """
    dag = Dag()
    r = dag.region("bitonic_stage-tweak")

    ld_stage = r.load(kind="stage")
    ld_pass = r.load(kind="pass")
    ld_N = r.load(kind="N")
    stage_p1 = r.arith(ld_stage, kind="stage_plus_1")
    distance = r.arith(ld_pass, kind="distance_shl")
    block_size = r.arith(stage_p1, kind="block_size_shl")
    r.arith(block_size, kind="half_block_shr")

    last_writer: dict[int, int] = {}
    for i in range(N):
        active = (i % 2 == 0)
        swap = i in (0, 2)
        iv = r.induction(kind="i", compare_depends_on_read=False)
        i_rd = iv["read"]
        block_idx = r.arith(i_rd, block_size, kind="block_idx_div")
        idx_in_block = r.arith(i_rd, block_size, kind="idx_in_block_mod")
        band_asc = r.arith(block_idx, kind="block_idx_and_1")
        ascending = r.arith(band_asc, kind="ascending_cmp")
        band_pred = r.arith(idx_in_block, distance, kind="idx_and_distance")
        outer_pred = r.arith(band_pred, kind="outer_pred_cmp")
        if active:
            partner = r.arith(i_rd, distance, outer_pred, kind="partner_add")
            in_bounds = r.arith(partner, ld_N, kind="partner_lt_N")
            deps_i = [in_bounds]
            deps_p = [in_bounds, partner]
            if i in last_writer:
                deps_i.append(last_writer[i])
            if i + 1 in last_writer:
                deps_p.append(last_writer[i + 1])
            ld_ip_i = r.load(*deps_i, kind="inplace_i")
            ld_ip_p = r.load(*deps_p, kind="inplace_partner")
            should_swap = r.arith(ascending, ld_ip_i, ld_ip_p, kind="value_cmp")
            if swap:
                st_i = r.store(ld_ip_p, should_swap, kind="swap_inplace_i")
                st_p = r.store(ld_ip_i, should_swap, kind="swap_inplace_partner")
                last_writer[i] = st_i
                last_writer[i + 1] = st_p

            # `inplace[i]++` is inside the outer body. On swap lanes the RAW from
            # the swap store orders it; on non-swap active lanes the outer gate
            # is the only dependency modeled by the eval's critical path notes.
            pp_deps = [outer_pred]
            if i in last_writer:
                pp_deps.append(last_writer[i])
            ld_pp = r.load(*pp_deps, kind="inplace_post_inc")
            add_pp = r.arith(ld_pp, kind="post_inc")
            last_writer[i] = r.store(add_pp, kind="post_inc_store")

        # Unconditional `inplace[i] -= 1`, ordered after the latest same-slot
        # writer (including a partner swap from the previous even lane).
        sub_deps = []
        if i in last_writer:
            sub_deps.append(last_writer[i])
        ld_sub = r.load(*sub_deps, kind="inplace_dec")
        sub = r.arith(ld_sub, kind="dec_sub")
        last_writer[i] = r.store(sub, output=True, kind="dec_store")

    contract = [RegionContract("bitonic_stage-tweak", A=92, LD=31, ST=24,
                               CP=17, aggregate=17)]
    return dag, contract


def _clz_trip_counts_main():
    values = [0, 0x80000000, 0x40000000, 0x00000001, 0xFFFFFFFF]
    values.extend(i * 0x1234 for i in range(5, 256))
    trips = []
    for v in values:
        trips.append(None if v == 0 else 32 - v.bit_length())
    return trips


def build_clz():
    """Count leading zeros for the concrete N=256 main.cpp input.

    The outer loop is fully unrolled. Each nonzero lane has a sequential while
    chain carrying `mask` and `count`; the maximum K is 31, so CP=163. The
    checked-in eval was originally written for six explicit lanes; this builder
    models the actual main.cpp `N=256` input and includes the hoisted `N` load.
    """
    trips = _clz_trip_counts_main()
    dag = Dag()
    r = dag.region("clz")
    r.load(kind="N")
    r.store(kind="i_init")
    for k in trips:
        r.induction(kind="i", compare_depends_on_read=False)
        ld_in = r.load(kind="input_data")
        is_zero = r.arith(ld_in, kind="value_eq_zero")
        if k is None:
            r.store(is_zero, output=True, kind="output_zero")
            continue
        st_count = r.store(is_zero, kind="count_init")
        st_mask = r.store(is_zero, kind="mask_init")
        for _ in range(k):
            ld_mask = r.load(st_mask, kind="mask")
            band = r.arith(ld_mask, ld_in, kind="value_and_mask")
            cmp = r.arith(band, kind="while_cmp")
            shift = r.arith(cmp, ld_mask, kind="mask_shift")
            ld_count = r.load(cmp, st_count, kind="count")
            add_count = r.arith(ld_count, kind="count_inc")
            st_count = r.store(add_count, kind="count_store")
            st_mask = r.store(shift, kind="mask_store")
        ld_mask = r.load(st_mask, kind="mask_exit")
        band = r.arith(ld_mask, ld_in, kind="exit_and")
        exit_cmp = r.arith(band, kind="exit_cmp")
        ld_count = r.load(exit_cmp, st_count, kind="count_final")
        r.store(ld_count, output=True, kind="output_count")
    contract = [RegionContract("clz", A=14122, LD=7445, ST=7445, CP=163,
                               aggregate=621)]
    return dag, contract


def _crc32_true_arm_trace(N=256):
    polynomial = 0xEDB88320
    crc = 0xFFFFFFFF
    trace = []
    for i in range(N):
        data = (i * 0x12345678) & 0xFFFFFFFF
        for byte_idx in range(4):
            byte = (data >> (byte_idx * 8)) & 0xFF
            crc ^= byte
            byte_trace = []
            for _ in range(8):
                take = bool(crc & 1)
                byte_trace.append(take)
                if take:
                    crc = ((crc >> 1) ^ polynomial) & 0xFFFFFFFF
                else:
                    crc = (crc >> 1) & 0xFFFFFFFF
            trace.append(byte_trace)
    return trace


def build_crc32(N=256):
    """CRC32 over the concrete main.cpp data stream.

    The CRC scalar is a non-associative carried state, so the byte and bit
    loops form one long sequential chain. The builder simulates the source
    recurrence to choose the taken arm of every bit iteration (K=4065 true XOR
    arms) and constructs the counted dynamic DAG for that trace. Off-path
    induction work is counted separately. Reproduces CP=50152, A=51682,
    LD=18945, ST=19971, aggregate 50152.
    """
    trace = _crc32_true_arm_trace(N)
    K = sum(1 for byte in trace for take in byte if take)
    if K != 4065:
        raise AssertionError(f"crc32 trace K={K}, expected 4065")

    dag = Dag()
    r = dag.region("crc32")

    prev_crc = r.store(kind="crc_init")
    r.store(kind="i_init")

    byte_iter = iter(trace)
    first_outer = True
    first_byte = True
    for _ in range(N):
        i_iv = r.induction(kind="i", compare_depends_on_read=True)
        # The committed CRC eval charges a one-time cold-start gap before the
        # steady crc recurrence. Use existing counted induction work to gate the
        # first input load; later input loads overlap the carried crc chain.
        if first_outer:
            ld_input = r.load(i_iv["store"], kind="input_data")
            first_outer = False
        else:
            ld_input = r.load(kind="input_data")
        r.store(kind="byte_idx_init")
        for _ in range(4):
            byte_iv = r.induction(kind="byte_idx", compare_depends_on_read=True)
            r.store(kind="bit_init")
            gate = prev_crc
            byte_deps = [byte_iv["read"], ld_input]
            if first_byte:
                byte_deps.extend([i_iv["cmp"], byte_iv["cmp"]])
                first_byte = False
            # Pre-bit crc ^= byte path: five cycles from the current gate to the
            # new crc store, matching the eval's steady-state byte contribution.
            mul = r.arith(gate, *byte_deps, kind="byte_mul")
            data_shift = r.arith(mul, kind="data_shift")
            byte_mask = r.arith(data_shift, kind="byte_and")
            ld_crc = r.load(gate, prev_crc, kind="crc_prebit")
            xor_byte = r.arith(byte_mask, ld_crc, kind="crc_xor_byte")
            prev_crc = r.store(xor_byte, kind="crc_store_prebit")
            for take in next(byte_iter):
                r.induction(kind="bit", compare_depends_on_read=True)
                ld_crc = r.load(prev_crc, kind="crc_bit")
                bit = r.arith(ld_crc, kind="crc_and_1")
                cmp = r.arith(bit, kind="crc_lsb_cmp")
                shift = r.arith(cmp, ld_crc, kind="crc_shift")
                if take:
                    x = r.arith(shift, kind="crc_xor_poly")
                    prev_crc = r.store(x, kind="crc_store_true")
                else:
                    prev_crc = r.store(shift, kind="crc_store_false")

    ld_final = r.load(prev_crc, kind="crc_final")
    inv = r.arith(ld_final, kind="crc_not")
    r.store(inv, output=True, kind="output_checksum")

    contract = [RegionContract("crc32", A=51682, LD=18945, ST=19971, CP=50152,
                               aggregate=50152)]
    return dag, contract


def build_kmp_table(M=16):
    """KMP failure table for the concrete main.cpp pattern.

    The outer loop carries `j` and may follow failure links through previously
    written table entries. This builder encodes the checked-in trace table:
    fallback counts and exit suffixes per i. Source-order output_table stores
    gate the next outer iteration so the constructed CP matches the eval's
    157-cycle serial chain. Reproduces A=96, LD=88, ST=50.
    """
    trace = [
        (0, "j0_false"),
        (0, "j0_true"),
        (0, "jpos_match"),
        (1, "j0_false"),
        (0, "j0_true"),
        (0, "jpos_match"),
        (0, "jpos_match"),
        (0, "jpos_match"),
        (1, "jpos_match"),
        (2, "j0_true"),
        (0, "jpos_match"),
        (0, "jpos_match"),
        (0, "jpos_match"),
        (0, "jpos_match"),
        (1, "j0_false"),
    ]
    dag = Dag()
    r = dag.region("kmp_table")
    r.load(kind="M")
    r.store(kind="output_table_0")
    prev_j = r.store(kind="j_init")
    r.store(kind="i_init")

    prev_outer = None
    for idx, (fallbacks, exit_kind) in enumerate(trace):
        iv = r.induction(kind="i", compare_depends_on_read=True)
        gate = iv["cmp"] if idx == 0 else prev_outer
        pat_i = r.load(kind="pattern_i")
        for _ in range(fallbacks):
            ld_j = r.load(*([prev_j, gate] if gate is not None else [prev_j]),
                          kind="j_while")
            cmp_pos = r.arith(ld_j, kind="j_gt_0")
            pat_j = r.load(cmp_pos, kind="pattern_j")
            cmp_mis = r.arith(pat_i, pat_j, kind="pattern_mismatch")
            aa = r.address_add(cmp_mis, ld_j, kind="j_minus_1")
            lps = r.load(aa, kind="output_table_fallback")
            prev_j = r.store(lps, kind="j_fallback_store")
            gate = prev_j

        ld_j = r.load(*([prev_j, gate] if gate is not None else [prev_j]),
                      kind="j_exit")
        cmp_pos = r.arith(ld_j, kind="j_gt_0_exit")
        if exit_kind.startswith("j0"):
            pat0 = r.load(cmp_pos, kind="pattern_0")
            cmp_eq = r.arith(pat_i, pat0, kind="final_eq")
        else:
            pat_j = r.load(cmp_pos, kind="pattern_j_exit")
            cmp_mis = r.arith(pat_i, pat_j, kind="pattern_mismatch_exit")
            cmp_eq = r.arith(cmp_mis, pat_i, pat_j, kind="final_eq")

        if exit_kind.endswith("true") or exit_kind == "jpos_match":
            inc_j = r.arith(cmp_eq, ld_j, kind="j_inc")
            prev_j = r.store(inc_j, kind="j_inc_store")
            reload_j = r.load(prev_j, kind="j_reload")
            prev_outer = r.store(reload_j, output=True, kind="output_table_i")
        else:
            prev_outer = r.store(cmp_eq, output=True, kind="output_table_i")

    contract = [RegionContract("kmp_table", A=96, LD=88, ST=50, CP=157,
                               aggregate=157)]
    return dag, contract


def _build_wildcard_case(case_name: str, position_specs, has_final_fail: bool,
                         contract: RegionContract):
    """Build one concrete wildcard_match test case.

    ``position_specs`` is a list of (chars, is_match_position), where chars are
    "match", "wildcard", or "mismatch". A mismatch char breaks the inner loop;
    a match position scans all chars, then stores output=1 and returns. If no
    position matches, ``has_final_fail`` adds the final failing outer-bound test
    and output=0 store. The first `i`/`j` loads are counted but rooted so they do
    not extend the cold-start CP, matching the eval's constant-init treatment.
    """
    dag = Dag()
    r = dag.region(case_name)

    ld_N = r.load(kind="N")
    ld_M = r.load(kind="M")
    bound_sub = r.arith(ld_N, ld_M, kind="N_minus_M")
    prologue_cmp = r.arith(ld_M, ld_N, kind="M_gt_N")
    r.store(kind="i_init")

    prev_i_store = None
    for pos_index, (chars, is_match_position) in enumerate(position_specs):
        ld_i = r.load(*( [prev_i_store] if prev_i_store is not None else [] ),
                      kind="i_load")
        if pos_index == 0:
            outer_cmp = r.arith(prologue_cmp, bound_sub, kind="outer_bound")
        else:
            outer_cmp = r.arith(ld_i, bound_sub, kind="outer_bound")
        match_store = r.store(kind="match_init")
        r.store(kind="j_init")

        prev_j_store = None
        inner_gate = outer_cmp
        match_zero_store = None
        for char_index, char_kind in enumerate(chars):
            ld_j = r.load(*( [prev_j_store] if prev_j_store is not None else [] ),
                          kind="j_load")
            if char_index == 0:
                j_bound = r.arith(inner_gate, kind="j_bound")
            else:
                j_bound = r.arith(ld_j, inner_gate, kind="j_bound")
            ld_pat = r.load(j_bound, kind="pattern_j")
            cmp_wc = r.arith(ld_pat, kind="pattern_ne_wildcard")
            if char_kind == "wildcard":
                inc_j = r.arith(cmp_wc, ld_j, kind="j_inc")
                prev_j_store = r.store(inc_j, kind="j_store")
                inner_gate = prev_j_store
                continue

            addr = r.address_add(cmp_wc, ld_i, ld_j, kind="i_plus_j")
            ld_text = r.load(addr, kind="text_i_plus_j")
            cmp_ch = r.arith(ld_text, ld_pat, kind="text_ne_pattern")
            if char_kind == "match":
                inc_j = r.arith(cmp_ch, ld_j, kind="j_inc")
                prev_j_store = r.store(inc_j, kind="j_store")
                inner_gate = prev_j_store
            else:
                match_zero_store = r.store(cmp_ch, kind="match_zero")
                inner_gate = match_zero_store
                break

        if is_match_position:
            ld_j_exit = r.load(prev_j_store, kind="j_exit")
            inner_exit = r.arith(ld_j_exit, kind="j_bound_exit")
            ld_match = r.load(inner_exit, match_store, kind="match_read")
            cmp_match = r.arith(ld_match, kind="match_cmp")
            r.store(cmp_match, output=True, kind="output_match_true")
            continue

        # Failing position: read the just-cleared match and continue with i++.
        ld_match = r.load(inner_gate, match_zero_store, kind="match_read")
        cmp_match = r.arith(ld_match, kind="match_cmp")
        inc_i = r.arith(cmp_match, ld_i, kind="i_inc")
        prev_i_store = r.store(inc_i, kind="i_store")

    if has_final_fail:
        ld_i = r.load(prev_i_store, kind="i_final")
        fail_cmp = r.arith(ld_i, bound_sub, kind="outer_bound_fail")
        r.store(fail_cmp, output=True, kind="output_match_false")

    return dag, [contract]


def build_wildcard_match_cases():
    tc1_positions = [(["mismatch"], False) for _ in range(10)]
    tc1_positions.append(
        (["match", "match", "wildcard", "match", "match", "wildcard",
          "match", "match"], True))
    tc2_positions = [(["mismatch"], False) for _ in range(57)]
    tc3_positions = [(["wildcard"] * 8, True)]
    return [
        ("TC1", *_build_wildcard_case(
            "TC1", tc1_positions, False,
            RegionContract("TC1", A=111, LD=77, ST=52, CP=203,
                           aggregate=203))),
        ("TC2", *_build_wildcard_case(
            "TC2", tc2_positions, True,
            RegionContract("TC2", A=402, LD=288, ST=230, CP=745,
                           aggregate=745))),
        ("TC3", *_build_wildcard_case(
            "TC3", tc3_positions, False,
            RegionContract("TC3", A=29, LD=21, ST=12, CP=55,
                           aggregate=55))),
    ]


def build_sort_insertion(N=512):
    """Insertion sort for the reverse-order main.cpp input.

    The trace is the reverse-order worst case: outer i=1..N-1 executes i shift
    bodies and exits via j < 0 before writing key to output[0]. The in-place
    output[] dependencies are modeled at element granularity. Key i+1 body r
    reads the value written by key i body r, so the DAG is a pipelined wavefront
    rather than a whole-key serial chain.

    The copy and insertion-sort work intentionally stay in one region: copy
    stores feed later sort loads through explicit per-element RAW edges, allowing
    the scheduler to preserve the wavefront overlap instead of imposing a coarse
    copy->sort barrier. This differs from build_sort_quick's coarser two-region
    stack-machine model by design.
    """
    dag = Dag()
    r = dag.region("sort_insertion")

    r.load(kind="N")
    r.store(kind="copy_i_init")
    output_last = []
    for _ in range(N):
        ld_in = r.load(kind="input")
        output_last.append(r.store(ld_in, kind="output_copy"))
        r.induction(kind="copy_i", compare_depends_on_read=False)

    r.store(kind="outer_i_init")
    for i in range(1, N):
        r.induction(kind="outer_i", compare_depends_on_read=False)
        key = r.load(output_last[i], kind="key_output_i")
        j_init_val = r.arith(kind="j_init_sub")
        prev_j_store = r.store(j_init_val, kind="j_init")
        for j in range(i - 1, -1, -1):
            ld_j = r.load(prev_j_store, kind="j")
            cmp_nonneg = r.arith(ld_j, kind="j_ge_0")
            ld_out = r.load(cmp_nonneg, output_last[j], kind="output_j")
            cmp_gt = r.arith(ld_out, key, kind="output_gt_key")
            addr = r.address_add(cmp_gt, ld_j, kind="j_plus_1")
            is_final_output = i == N - 1
            output_last[j + 1] = r.store(
                addr, ld_out, output=is_final_output, kind="shift_store")
            dec_j = r.arith(cmp_gt, ld_j, kind="j_dec")
            prev_j_store = r.store(dec_j, kind="j_store")
        ld_j = r.load(prev_j_store, kind="j_exit")
        cmp_exit = r.arith(ld_j, kind="j_ge_0_exit")
        addr = r.address_add(cmp_exit, ld_j, kind="j_plus_1_final")
        output_last[0] = r.store(
            addr, key, output=(i == N - 1), kind="key_store")

    contract = [
        RegionContract("sort_insertion", A=526843, LD=264190, ST=264191,
                       CP=5112, aggregate=22016),
    ]
    return dag, contract


def build_sort_quick(N=1024):
    """Iterative quicksort for the concrete pseudo-random main.cpp input.

    The copy region is ordered before the in-place stack-machine sort because
    the sort reads output[] values written by copy. This is an intentional
    coarse phase model: a single-region DAG could connect each first quicksort
    read of output[k] to that element's copy store, but this builder preserves
    the copy->sort RAW as a phase-granularity barrier. This differs from
    build_sort_insertion's finer element-wavefront model by design. The sort
    builder replays the exact source-level stack trace and emits counted
    operations according to the eval formulas. It intentionally does not try to
    expose fork-join recursive parallelism because the source is a sequential
    explicit-stack machine.
    """
    values = [float((i * 7 + 13) % N) for i in range(N)]
    dag = Dag()

    copy = dag.region("copy")
    copy.load(kind="N")
    copy.store(kind="copy_i_init")
    for _ in range(N):
        ld_in = copy.load(kind="input")
        copy.store(ld_in, output=True, kind="output_copy")
        copy.induction(kind="copy_i", compare_depends_on_read=False)

    r = dag.region("sort")
    ld_N = r.load(kind="N")
    r.arith(ld_N, kind="N_le_1")
    r.arith(ld_N, kind="N_minus_1")
    top_last = r.store(kind="top_init")
    top_val = -1
    stack_vals = [None] * (2 * N + 4)
    stack_last = [None] * (2 * N + 4)
    output_last = [None] * N
    output_vals = list(values)

    stats = {"W": 0, "R": 0, "Z": 0, "C": 0, "S": 0, "L_p": 0, "R_p": 0}

    def push_value(value, dep):
        nonlocal top_last, top_val
        ld_top = r.load(top_last, kind="top_push")
        inc = r.arith(ld_top, dep, kind="top_inc")
        top_last = r.store(inc, kind="top_store")
        top_val += 1
        stack_vals[top_val] = value
        stack_last[top_val] = r.store(inc, dep, kind="stack_push")

    def pop_value(dep, which, top_node=None):
        nonlocal top_last, top_val
        ld_top = top_node if top_node is not None else \
            r.load(top_last, kind=f"top_pop_{which}")
        idx = top_val
        val = stack_vals[idx]
        st_dep = stack_last[idx]
        loaded = r.load(ld_top, dep, *( [st_dep] if st_dep is not None else [] ),
                        kind=f"stack_pop_{which}")
        dec = r.arith(ld_top, dep, kind=f"top_dec_{which}")
        top_last = r.store(dec, kind=f"top_store_{which}")
        top_val -= 1
        return val, loaded

    # Initial range push: stack[++top] = 0; stack[++top] = N-1.
    push_value(0, top_last)
    push_value(N - 1, ld_N)

    while top_val >= 0:
        stats["W"] += 1
        ld_top = r.load(top_last, kind="top_while")
        cmp_while = r.arith(ld_top, kind="top_ge_0")
        high, high_node = pop_value(cmp_while, "high", top_node=ld_top)
        low, low_node = pop_value(cmp_while, "low")
        range_cmp = r.arith(high_node, low_node, kind="low_ge_high")
        if low >= high:
            stats["Z"] += 1
            continue
        stats["R"] += 1

        pivot = output_vals[high]
        pivot_last = output_last[high]
        pivot_node = r.load(range_cmp, *( [pivot_last] if pivot_last is not None else [] ),
                            kind="pivot_load")
        i_val = low
        i_state = r.store(range_cmp, kind="part_i_init")
        j_state = r.store(range_cmp, kind="scan_j_init")

        for j in range(low, high):
            stats["C"] += 1
            ld_j = r.load(j_state, kind="scan_j")
            cmp_j = r.arith(ld_j, kind="j_lt_high")
            inc_j = r.arith(ld_j, kind="j_inc")
            j_state = r.store(inc_j, kind="j_store")
            outj_last = output_last[j]
            ld_outj = r.load(cmp_j, *( [outj_last] if outj_last is not None else [] ),
                             kind="output_j")
            cmp_pivot = r.arith(ld_outj, pivot_node, kind="output_le_pivot")
            if output_vals[j] <= pivot:
                stats["S"] += 1
                ld_i = r.load(cmp_pivot, i_state, kind="part_i")
                outi_last = output_last[i_val]
                ld_outi = r.load(ld_i, *( [outi_last] if outi_last is not None else [] ),
                                 kind="output_i")
                inc_i = r.arith(ld_i, kind="part_i_inc")
                i_state = r.store(inc_i, kind="part_i_store")
                st_i = r.store(cmp_pivot, ld_outj, ld_outi,
                               kind="swap_store_i")
                st_j = r.store(cmp_pivot, ld_outi, kind="swap_store_j")
                output_last[i_val] = st_i
                output_last[j] = st_j
                output_vals[i_val], output_vals[j] = output_vals[j], output_vals[i_val]
                i_val += 1

        ld_i_final = r.load(i_state, j_state, kind="part_i_final")
        outi_last = output_last[i_val]
        ld_outi_final = r.load(ld_i_final,
                               *( [outi_last] if outi_last is not None else [] ),
                               kind="output_i_final")
        st_i = r.store(ld_i_final, pivot_node, ld_outi_final,
                       output=True, kind="pivot_store_i")
        st_h = r.store(ld_i_final, ld_outi_final, output=True,
                       kind="pivot_store_high")
        output_last[i_val] = st_i
        output_last[high] = st_h
        output_vals[i_val], output_vals[high] = output_vals[high], output_vals[i_val]
        pivot_idx = i_val

        left_cmp = r.arith(st_i, st_h, kind="left_push_cmp")
        if pivot_idx > low:
            stats["L_p"] += 1
            left_hi = r.arith(left_cmp, kind="pivot_minus_1")
            push_value(low, left_cmp)
            push_value(pivot_idx - 1, left_hi)
        right_cmp = r.arith(left_cmp, st_i, st_h, top_last,
                            kind="right_push_cmp")
        if pivot_idx < high:
            stats["R_p"] += 1
            right_lo = r.arith(right_cmp, kind="pivot_plus_1")
            push_value(pivot_idx + 1, right_lo)
            push_value(high, right_cmp)
        else:
            top_last = right_cmp

    # Final failing while check.
    ld_top = r.load(top_last, kind="top_final")
    r.arith(ld_top, output=True, kind="top_ge_0_final")

    expected = {"W": 1024, "R": 678, "Z": 346, "C": 25773, "S": 21104,
                "L_p": 516, "R_p": 507}
    if stats != expected:
        raise AssertionError(f"sort_quick trace {stats} != {expected}")

    contract = [
        RegionContract("copy", A=2048, LD=2049, ST=2049, CP=2, aggregate=171),
        RegionContract("sort", A=106949, LD=101934, ST=97942, CP=95886,
                       aggregate=95886),
    ]
    return dag, contract


BUILDERS = {
    "axpy": build_axpy,
    "autocorrelation": build_autocorrelation,
    "fft_butterfly": build_fft_butterfly,
    "conv2d": build_conv2d,
    "batchnorm": build_batchnorm,
    "bit_reverse": build_bit_reverse,
    "bisection_step": build_bisection_step,
    "bitonic_stage": build_bitonic_stage,
    "bitonic_stage-modified": build_bitonic_stage_modified,
    "binary_search": build_binary_search,
    "gather": build_gather,
    "edge_update": build_edge_update,
    "interpolate_linear": build_interpolate_linear,
    "bitonic_stage-tweak": build_bitonic_stage_tweak,
    "clz": build_clz,
    "crc32": build_crc32,
    "kmp_table": build_kmp_table,
    "sort_insertion": build_sort_insertion,
    "sort_quick": build_sort_quick,
}

MULTICASE_BUILDERS = {
    "wildcard_match": build_wildcard_match_cases,
}

PILOTS = ("axpy", "autocorrelation", "fft_butterfly")

# Kernels with a committed CGRA-SCHED eval block: the default set for the
# `write`/`check` commands and the self-test eval-check, so the read-only drift
# guard covers every checked-in block (including conv2d).
WRITTEN_KERNELS = PILOTS + ("conv2d", "batchnorm", "bit_reverse",
                            "bisection_step", "bitonic_stage",
                            "bitonic_stage-modified", "binary_search",
                            "gather", "edge_update", "interpolate_linear",
                            "bitonic_stage-tweak", "clz", "crc32",
                            "kmp_table", "wildcard_match", "sort_insertion",
                            "sort_quick")

EVAL_PATHS = {
    name: Path(__file__).resolve().parents[1] / "app" / name / f"{name}_eval.md"
    for name in tuple(BUILDERS) + tuple(MULTICASE_BUILDERS)
}


def build_kernel(kernel: str):
    if kernel not in BUILDERS:
        raise KeyError(f"unknown kernel {kernel!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[kernel]()


def build_multicase_kernel(kernel: str):
    if kernel not in MULTICASE_BUILDERS:
        raise KeyError(
            f"unknown multi-case kernel {kernel!r}; known: "
            f"{sorted(MULTICASE_BUILDERS)}")
    return MULTICASE_BUILDERS[kernel]()


def check_contract(dag: Dag, contract, cfg: Config):
    """Verify the constructed DAG matches the builder's declared contract.

    The op counts (`A`/`LD`/`ST`) and `CP` are configuration-independent, so they
    are checked for any config. The declared `aggregate` is the golden 6x6 value,
    so it is only checked when `cfg` is the 6x6 configuration; this keeps `report`/
    `write`/`check` usable with other configs (whose aggregate legitimately
    differs) instead of aborting against the 6x6 number.
    Returns the list of RegionAggregate for further checks."""
    if len(dag.regions) != len(contract):
        raise AssertionError(
            f"region count {len(dag.regions)} != contract {len(contract)}")
    is_6x6 = (cfg.P, cfg.L, cfg.S) == (CONFIG_6x6.P, CONFIG_6x6.L, CONFIG_6x6.S)
    fields = ("A", "LD", "ST", "CP", "aggregate") if is_6x6 \
        else ("A", "LD", "ST", "CP")
    aggs = []
    for region, decl in zip(dag.regions, contract):
        # Region names encode the phase/barrier contract (e.g. the FFT stage
        # order), so a renamed or reordered region must not pass even if the
        # numeric fields match.
        if region.name != decl.name:
            raise AssertionError(
                f"region name {region.name!r} != contract {decl.name!r} "
                "(wrong name or order)")
        ra = region_aggregate(region, cfg)
        aggs.append(ra)
        for field in fields:
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
    if kernel in MULTICASE_BUILDERS:
        result = evaluate_multicase(kernel, cfg)
        out = [f"# {kernel}  ({cfg.label}: P={cfg.P} L={cfg.L} S={cfg.S})", ""]
        out.append("per-case validation:")
        for case_name, case in result.cases:
            ra = case.region_aggs[0]
            rs = case.region_scheds[0]
            out.append(
                f"  {case_name:<14} CP={ra.CP:<3} A={ra.A:<6} "
                f"LD={ra.LD:<6} ST={ra.ST:<6} aggregate={ra.aggregate:<4} "
                f"scheduled={rs.makespan}")
        out.append("")
        out.append(render_multicase_block(result))
        return "\n".join(out)
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
    if kernel in MULTICASE_BUILDERS:
        block = render_multicase_block(evaluate_multicase(kernel, cfg))
    else:
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
    problems = []
    if kernel in MULTICASE_BUILDERS:
        try:
            block = render_multicase_block(evaluate_multicase(kernel, cfg))
        except AssertionError as exc:
            problems.append(f"{kernel}: contract drift: {exc}")
            return problems
    else:
        dag, contract = build_kernel(kernel)
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

    # 11. Fully-unrolled induction compares are rooted; sequential compares use
    # the loaded carried iterator.
    d = Dag(); r = d.region("induction_cmp")
    seq = r.induction(kind="seq")
    unrolled = r.induction(kind="unrolled", compare_depends_on_read=False)
    if r._by_id[seq["cmp"]].preds != [seq["read"]]:
        errors.append("induction_cmp: sequential compare is not read-dependent")
    if r._by_id[unrolled["cmp"]].preds:
        errors.append("induction_cmp: unrolled compare is not rooted")

    # 12. Empty DAG/region.
    d = Dag(); d.region("empty")
    res = evaluate(d, "empty", cfg)
    if not (res.aggregate_cycles == 0 and res.scheduled_cycles == 0
            and res.gap_cycles == 0 and res.gap_ratio == 1.0):
        errors.append("empty: metrics not all-zero / ratio!=1.0")

    # 13. Zero-capacity config rejected.
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
        "batchnorm": {"aggregate": 74,
                      "regions": [("batchnorm", 10, 2645, 568, 548, 74)]},
        "bit_reverse": {"aggregate": 2134,
                        "regions": [("bit_reverse", 132, 49664, 25345, 25600,
                                     2134)]},
        "bisection_step": {"aggregate": 27,
                           "regions": [("bisection_step", 4, 384, 321, 192,
                                        27)]},
        "bitonic_stage": {"aggregate": 11,
                          "regions": [("bitonic_stage", 11, 80, 19, 12, 11)]},
        "bitonic_stage-modified": {
            "aggregate": 31,
            "regions": [("bitonic_stage-modified", 31, 133, 55, 48, 31)]},
        "binary_search": {"aggregate": 48,
                          "regions": [("binary_search", 48, 124, 69, 41, 48)]},
        "gather": {"aggregate": 257,
                   "regions": [("gather", 4, 3072, 3074, 2048, 257)]},
        "edge_update": {"aggregate": 6,
                        "regions": [("edge_update", 6, 40, 38, 37, 6)]},
        "interpolate_linear": {
            "aggregate": 294,
            "regions": [("interpolate_linear", 289, 5699, 3523, 1216, 294)]},
        "bitonic_stage-tweak": {
            "aggregate": 17,
            "regions": [("bitonic_stage-tweak", 17, 92, 31, 24, 17)]},
        "clz": {"aggregate": 621,
                "regions": [("clz", 163, 14122, 7445, 7445, 621)]},
        "crc32": {"aggregate": 50152,
                  "regions": [("crc32", 50152, 51682, 18945, 19971,
                               50152)]},
        "kmp_table": {"aggregate": 157,
                      "regions": [("kmp_table", 157, 96, 88, 50, 157)]},
        "sort_insertion": {
            "aggregate": 22016,
            "regions": [("sort_insertion", 5112, 526843, 264190, 264191,
                         22016)]},
        "sort_quick": {
            "aggregate": 96057,
            "regions": [("copy", 2, 2048, 2049, 2049, 171),
                        ("sort", 95886, 106949, 101934, 97942, 95886)]},
    }
    multi_expect = {
        "wildcard_match": [
            ("TC1", 203, 111, 77, 52, 203),
            ("TC2", 745, 402, 288, 230, 745),
            ("TC3", 55, 29, 21, 12, 55),
        ],
    }
    for kernel in WRITTEN_KERNELS:
        if kernel in MULTICASE_BUILDERS:
            try:
                multi = evaluate_multicase(kernel, cfg)
            except AssertionError as exc:
                errors.append(f"{kernel}: {exc}")
                continue
            rows = []
            for case_name, result in multi.cases:
                ra = result.region_aggs[0]
                rows.append((case_name, ra.CP, ra.A, ra.LD, ra.ST,
                             ra.aggregate))
                if result.scheduled_cycles < result.aggregate_cycles:
                    errors.append(f"{kernel}/{case_name}: scheduled "
                                  f"{result.scheduled_cycles} < aggregate "
                                  f"{result.aggregate_cycles}")
            if rows != multi_expect[kernel]:
                errors.append(f"{kernel}: case rows {rows} != "
                              f"{multi_expect[kernel]}")
            if render_multicase_block(evaluate_multicase(kernel, cfg)) != \
                    render_multicase_block(multi):
                errors.append(f"{kernel}: non-deterministic block")
            continue
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
    """Exercise apply_block / extract_block without touching the real eval files
    (operate on in-memory text)."""
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


def _run_eval_check_tests(errors):
    """Exercise the read-only eval drift checker (check_eval) against the
    checked-in eval files (positive) and against tampered / blockless temp
    copies (negative), without modifying any repository file."""
    cfg = CONFIG_6x6
    written = list(WRITTEN_KERNELS)
    # Positive: every checked-in eval block matches its freshly computed block.
    for kernel in written:
        problems = check_eval(kernel, cfg)
        if problems:
            errors.append(f"eval-check: {kernel} unexpectedly drifted: "
                          f"{problems}")
    # Negative: drift and a missing block are both detected. Redirect the eval
    # path to a temp copy (restored in finally) so no repo file is touched.
    kernel = "axpy"
    orig_path = EVAL_PATHS[kernel]
    text = orig_path.read_text()
    block = extract_block(text, kernel)
    if block is None:
        errors.append("eval-check: axpy eval has no block to tamper")
        return
    with tempfile.TemporaryDirectory() as td:
        tampered_path = Path(td) / "tampered_eval.md"
        tampered_path.write_text(
            text.replace("**aggregate_cycles** = 4",
                         "**aggregate_cycles** = 999"))
        blockless_path = Path(td) / "blockless_eval.md"
        blockless_path.write_text(text.replace(block, "stub body"))
        try:
            EVAL_PATHS[kernel] = tampered_path
            problems = check_eval(kernel, cfg)
            if not problems:
                errors.append("eval-check: tampered block not flagged as drift")
            EVAL_PATHS[kernel] = blockless_path
            problems = check_eval(kernel, cfg)
            if not any("no CGRA-SCHED block" in p for p in problems):
                errors.append("eval-check: missing block not flagged")
        finally:
            EVAL_PATHS[kernel] = orig_path


def _run_contract_validation_test(errors):
    """check_contract must reject a region whose declared name disagrees with the
    constructed region (wrong name or order), even when numeric fields match."""
    dag, contract = build_kernel("axpy")
    decl = contract[0]
    renamed = [RegionContract("WRONG", decl.A, decl.LD, decl.ST, decl.CP,
                              decl.aggregate)]
    try:
        check_contract(dag, renamed, CONFIG_6x6)
        errors.append("contract-validation: wrong region name not rejected")
    except AssertionError:
        pass


def run_self_tests() -> int:
    errors: list[str] = []
    _run_synthetic_tests(errors)
    _run_golden_tests(errors)
    _run_contract_validation_test(errors)
    _run_drift_test(errors)
    _run_eval_check_tests(errors)
    if errors:
        for e in errors:
            print(f"  SELF-TEST FAIL: {e}")
        return 1
    print("[PASS] cgra_schedule self-tests "
          "(synthetic + golden pilots + contract + drift + eval-check)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv) -> int:
    parser = argparse.ArgumentParser(
        description="CGRA aggregate lower bound + finite-resource estimate")
    sub = parser.add_subparsers(dest="cmd")
    choices = sorted(tuple(BUILDERS) + tuple(MULTICASE_BUILDERS))

    p_report = sub.add_parser("report", help="print the canonical eval block")
    p_report.add_argument("kernel", choices=choices)
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
        kernels = args.kernels or list(WRITTEN_KERNELS)
        for kernel in kernels:
            changed = write_eval(kernel, cfg)
            print(f"{kernel}: {'updated' if changed else 'unchanged'} "
                  f"{EVAL_PATHS[kernel].name}")
        return 0
    if args.cmd == "check":
        cfg = parse_config(args.config)
        kernels = args.kernels or list(WRITTEN_KERNELS)
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
