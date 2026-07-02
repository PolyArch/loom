#!/usr/bin/env python3
from __future__ import annotations
"""Banking-aware Loom-pragma design-space estimates.

This helper compares explicit ``LOOM_PARALLEL(P)`` / ``LOOM_UNROLL(U)`` choices
(per loop level) for a kernel, on a CGRA resource configuration ``(P, L, S)``. It
implements the "Optional Loom-Pragma Design-Space Estimate" section of
``docs/spec-kernel-performance.md``, including the *banking-aware P-vs-U
distinction on the load/store axis*.

Model, in one paragraph
-----------------------
The op counts of an exposed chunk depend only on the per-level *exposure*
(``p*u`` per level), so ``P`` and ``U`` do not separate on the arithmetic /
critical-path axes (those stay a global pool). They separate on the **load/store
axis** through *banking*: ``LOOM_PARALLEL`` partitions the strided arrays into
independent banks (one port per worker), while ``LOOM_UNROLL`` piles a worker's
``U*ld_iter`` accesses onto that worker's single port. The concurrent load/store
issue widths a chunk can use are therefore

    active_L = min(P_tot, B_L, L)      active_S = min(P_tot, B_S, S)

where ``P_tot`` is the product of the parallel factors over parallelizable
levels, and ``B_L``/``B_S`` are the effective bank counts of the binding load /
store arrays (an explicit ``LOOM_MEMORY_BANK(B)`` caps a bank count; an array
partitioned over levels ``ℓ`` has ``B = Π p_ℓ`` absent a cap; a broadcast /
single-element array has ``B = 1``). The chunk is then scheduled with the
effective configuration ``(P, active_L, active_S)`` and summed over waves. Only
``absolute_cgra_lb`` (the full-trip aggregate over full lanes) is a lower bound.

This is a directed, load/store-focused model: it deliberately does NOT model the
opposing control-overhead amortization (which would favor ``U``), so the
recommendation is intentionally biased toward parallelism. It is an exploratory
estimate, not a lower bound and not cycle-accurate RTL.
"""

import argparse
import sys
from dataclasses import dataclass, field

from cgra_schedule import Config, Dag, _ceil_div, evaluate, parse_config

BIG = 1 << 30


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
        # reduction levels may use parallel workers only via LOOM_REDUCE; both
        # parallel and reduction contribute to P_tot. sequential cannot.
        return self.kind in ("parallel", "reduction")

    def tiled(self) -> bool:
        # parallel levels are tiled across waves (exposure = p*u); reduction and
        # sequential levels are fully consumed within a chunk (exposure = trip).
        return self.kind == "parallel"


@dataclass(frozen=True)
class BindingArray:
    """The array that binds a memory class, with its banking description."""
    name: str
    bank_cap: int | None            # explicit LOOM_MEMORY_BANK cap; None = uncapped
    partition_levels: tuple[str, ...]  # levels whose parallelism banks this array


@dataclass(frozen=True)
class KernelSpec:
    name: str
    levels: tuple[Level, ...]
    load_binding: BindingArray
    store_binding: BindingArray
    build_chunk: object             # callable(exposure: dict[str,int]) -> Dag
    default_config: str = "6x6"
    banking_note: str = ""

    def level(self, name: str) -> Level:
        for lv in self.levels:
            if lv.name == name:
                return lv
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Per-kernel chunk builders
# ---------------------------------------------------------------------------
# Each builder receives ``exposure``: a dict level-name -> iterations exposed at
# that level (p*u for tiled/parallel levels, full trip for reduction levels). The
# op counts it produces depend ONLY on exposure, never on the P/U split -- the
# split enters solely through active_L/active_S (see evaluate_candidate).

def _axpy_chunk(exp):
    E = exp["i"]
    dag = Dag()
    r = dag.region("axpy")
    ld_alpha = r.load(kind="alpha")
    r.load(kind="N")
    for _ in range(E):
        lx = r.load(kind="input_x")
        ly = r.load(kind="input_y")
        m = r.arith(lx, ld_alpha, kind="mul")
        a = r.arith(m, ly, kind="add")
        r.store(a, output=True, kind="output_y")
        r.induction(kind="i", compare_depends_on_read=False)
    return dag


def _vecsum_chunk(exp):
    E = exp["i"]
    dag = Dag()
    r = dag.region("vecsum")
    r.load(kind="init")
    r.load(kind="N")
    leaves = []
    for _ in range(E):
        leaves.append(r.load(kind="A"))
        r.induction(kind="i", compare_depends_on_read=False)
    root = r.balanced_reduction(leaves, kind="reduce")
    acc = r.arith(root, kind="acc_merge")   # merge partial into carry (associative)
    r.store(acc, output=True, kind="sum")
    return dag


def _gemv_chunk(exp):
    E_i = exp["i"]
    N = exp["j"]  # reduction dim: fully exposed within each row
    dag = Dag()
    r = dag.region("gemv")
    ld_alpha = r.load(kind="alpha")
    ld_beta = r.load(kind="beta")
    r.load(kind="M")
    r.load(kind="N")
    # x[j] is invariant of i: loaded once per chunk, reused across the E_i rows.
    xloads = [r.load(kind="x") for _ in range(N)]
    for _ in range(E_i):
        r.induction(kind="i", compare_depends_on_read=False)
        products = []
        for jj in range(N):
            aij = r.load(kind="A")
            products.append(r.arith(aij, xloads[jj], kind="mul"))
            r.induction(kind="j", compare_depends_on_read=False)
        rowsum = r.balanced_reduction(products, kind="reduce")
        asum = r.arith(rowsum, ld_alpha, kind="mul_alpha")
        ly = r.load(kind="input_y")
        by = r.arith(ly, ld_beta, kind="mul_beta")
        r.store(r.arith(asum, by, kind="add"), output=True, kind="output_y")
    return dag


def _tridiag_chunk(exp):
    # Forward elimination sweep of the Thomas algorithm: a NON-associative
    # carried recurrence (m depends on c_prime[i-1]; d_prime[i] depends on
    # d_prime[i-1] through a division). It cannot be reduced (unlike vecsum) and
    # cannot be parallelized -- the sole legal dim is a sequential chain.
    E = exp["i"]
    dag = Dag()
    r = dag.region("tridiag_fwd")
    prev_c = r.load(kind="c_prime0")   # c_prime[0] root
    prev_d = r.load(kind="d_prime0")   # d_prime[0] root
    for _ in range(E):
        la = r.load(kind="input_a")
        lb = r.load(kind="input_b")
        lc = r.load(kind="input_c")
        ld = r.load(kind="input_d")
        ac = r.arith(la, prev_c, kind="mul")       # a*c_prime[i-1]  (carried)
        m = r.arith(lb, ac, kind="sub")            # m = b - a*c'
        cprime = r.arith(lc, m, kind="div")        # c_prime[i] = c/m
        ad = r.arith(la, prev_d, kind="mul")       # a*d_prime[i-1]  (carried)
        dn = r.arith(ld, ad, kind="sub")
        dprime = r.arith(dn, m, kind="div")        # d_prime[i] = (d-a*d')/m
        r.store(cprime, kind="c_prime")
        r.store(dprime, output=True, kind="d_prime")
        r.induction(kind="i", compare_depends_on_read=True)  # carried iterator
        prev_c = cprime
        prev_d = dprime
    return dag


def _conv2d_chunk(exp):
    E_out = exp["out"]
    K = exp["tap"]  # reduction dim (C_in*KH*KW): fully exposed per output pixel
    dag = Dag()
    r = dag.region("conv2d")
    r.load(kind="params")
    for _ in range(E_out):
        r.induction(kind="out", compare_depends_on_read=False)
        products = []
        for _ in range(K):
            li = r.load(kind="input")
            lw = r.load(kind="weight")
            products.append(r.arith(li, lw, kind="mul"))
            r.induction(kind="tap", compare_depends_on_read=False)
        r.store(r.balanced_reduction(products, kind="reduce"),
                output=True, kind="output")
    return dag


def _batchnorm_chunk(exp):
    Ec, Eh, Ew = exp["c"], exp["h"], exp["w"]
    dag = Dag()
    r = dag.region("batchnorm")
    ld_eps = r.load(kind="eps")
    r.load(kind="C")
    r.load(kind="H")
    r.load(kind="W")
    for _ in range(Ec):
        r.induction(kind="c", compare_depends_on_read=False)
        lv = r.load(kind="variance")
        lm = r.load(kind="mean")
        lg = r.load(kind="gamma")
        lb = r.load(kind="beta")
        ve = r.arith(lv, ld_eps, kind="var_plus_eps")
        sq = r.arith(ve, kind="sqrt")
        inv = r.arith(sq, kind="inv_std")   # invariant across (h,w)
        for _ in range(Eh):
            r.induction(kind="h", compare_depends_on_read=False)
            for _ in range(Ew):
                r.induction(kind="w", compare_depends_on_read=False)
                li = r.load(kind="input")
                sub = r.arith(li, lm, kind="sub")
                nm = r.arith(sub, inv, kind="mul_inv")
                mg = r.arith(nm, lg, kind="mul_gamma")
                r.store(r.arith(mg, lb, kind="add_beta"),
                        output=True, kind="output")
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
        load_binding=BindingArray("input_x", None, ("i",)),
        store_binding=BindingArray("output_y", None, ("i",)),
        build_chunk=_axpy_chunk,
        banking_note=(
            "input_x/input_y/output_y are partitioned across the parallel "
            "workers (contiguous distribution), so B = P_tot; no explicit "
            "LOOM_MEMORY_BANK cap."),
    )
    KERNELS["vecsum"] = KernelSpec(
        name="vecsum",
        levels=(Level("i", 256, "reduction"),),
        load_binding=BindingArray("A", None, ("i",)),
        # The single scalar sum is 1 store; the store class is dominated by the
        # per-worker iterator write-backs, which are partitioned across the
        # LOOM_REDUCE workers (B_S = P_tot).
        store_binding=BindingArray("iter+sum", None, ("i",)),
        build_chunk=_vecsum_chunk,
        banking_note=(
            "A is partitioned across the LOOM_REDUCE workers (B_L = P_tot). "
            "Stores are dominated by per-worker iterator write-backs "
            "(B_S = P_tot); the final sum is a single scalar store."),
    )
    M, N = 64, 64
    KERNELS["gemv"] = KernelSpec(
        name="gemv",
        levels=(Level("i", M, "parallel"), Level("j", N, "reduction")),
        # A carries LOOM_MEMORY_BANK(4, block): block-partitioned over rows (i),
        # so B_L = min(4, p_i) and column (j) parallelism adds no A ports.
        load_binding=BindingArray("A", 4, ("i",)),
        store_binding=BindingArray("output_y", None, ("i",)),
        build_chunk=_gemv_chunk,
        banking_note=(
            "A has LOOM_MEMORY_BANK(4, block): block-partitioned over rows (i), "
            "so B_L = min(4, p_i) and parallelizing the inner column reduction "
            "(j) adds NO A ports. x is broadcast (loaded once per chunk, reused "
            "across rows). output_y is partitioned over rows (B_S = p_i)."),
    )
    n_out, K = _conv2d_dims()
    KERNELS["conv2d"] = KernelSpec(
        name="conv2d",
        levels=(Level("out", n_out, "parallel"), Level("tap", K, "reduction")),
        load_binding=BindingArray("input", None, ("out",)),
        store_binding=BindingArray("output", None, ("out",)),
        build_chunk=_conv2d_chunk,
        banking_note=(
            "Output pixels (out = C_out*OH*OW) parallel; the K = C_in*KH*KW taps "
            "are a reduction fully consumed per pixel. input/weight modeled as "
            "partitioned over the output-pixel workers (B = P_tot); input halo "
            "reuse and weight sharing are not modeled (conservative loads)."),
    )
    KERNELS["tridiag_solve"] = KernelSpec(
        name="tridiag_solve",
        levels=(Level("i", 64, "sequential"),),
        load_binding=BindingArray("input_a", None, ()),   # nothing partitions it
        store_binding=BindingArray("d_prime", None, ()),
        build_chunk=_tridiag_chunk,
        banking_note=(
            "The forward sweep carries a NON-associative recurrence "
            "(division chain), so LOOM_PARALLEL is illegal and LOOM_REDUCE does "
            "not apply: P_tot is forced to 1, B_L = B_S = 1, active_L = "
            "active_S = 1. Only LOOM_UNROLL is legal, and it adds no bank/port "
            "-- so there is no P-vs-U distinction and the kernel stays "
            "critical-path (serial) bound."),
    )
    KERNELS["batchnorm"] = KernelSpec(
        name="batchnorm",
        levels=(Level("c", 4, "parallel"), Level("h", 8, "parallel"),
                Level("w", 8, "parallel")),
        load_binding=BindingArray("input", None, ("c", "h", "w")),
        store_binding=BindingArray("output", None, ("c", "h", "w")),
        build_chunk=_batchnorm_chunk,
        banking_note=(
            "All three dims (c,h,w) are parallel; input/output partitioned over "
            "all of them (B = P_tot). mean/variance/gamma/beta are per-channel "
            "invariants (loaded once per exposed channel)."),
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


def _pow2_upto(limit: int) -> list[int]:
    vals, v = [], 1
    while v <= limit:
        vals.append(v)
        v *= 2
    return vals or [1]


def enumerate_candidates(spec: KernelSpec, max_parallel: int, max_unroll: int,
                         exposure_cap: int) -> list[Candidate]:
    """Cartesian product of per-level (p,u) choices, honoring legality:
    sequential -> p=1; p*u <= trip per level; total tiled exposure <=
    exposure_cap (keeps chunks small)."""
    per_level_choices = []
    for lv in spec.levels:
        pmax = min(max_parallel, lv.trip) if lv.parallelizable() else 1
        umax = min(max_unroll, lv.trip)
        choices = []
        for p in _pow2_upto(pmax):
            for u in _pow2_upto(umax):
                if p * u <= lv.trip:
                    choices.append((lv.name, p, u))
        per_level_choices.append(choices)

    out: list[Candidate] = []

    def rec(idx: int, acc: list):
        if idx == len(per_level_choices):
            cand = Candidate(tuple(acc))
            # cap total exposure of tiled (parallel) levels to keep chunks small
            tiled_exp = 1
            for lv in spec.levels:
                p, u = cand.factors(lv.name)
                if lv.tiled():
                    tiled_exp *= p * u
            if tiled_exp <= exposure_cap:
                out.append(cand)
            return
        for choice in per_level_choices[idx]:
            acc.append(choice)
            rec(idx + 1, acc)
            acc.pop()

    rec(0, [])
    return out


# ---------------------------------------------------------------------------
# Banking + evaluation
# ---------------------------------------------------------------------------

def _exposure(spec: KernelSpec, cand: Candidate) -> dict[str, int]:
    exp = {}
    for lv in spec.levels:
        p, u = cand.factors(lv.name)
        exp[lv.name] = (p * u) if lv.tiled() else lv.trip
    return exp


def _p_tot(spec: KernelSpec, cand: Candidate) -> int:
    prod = 1
    for lv in spec.levels:
        if lv.parallelizable():
            p, _ = cand.factors(lv.name)
            prod *= p
    return prod


def _effective_banks(spec: KernelSpec, cand: Candidate,
                     arr: BindingArray) -> int:
    partition_workers = 1
    for name in arr.partition_levels:
        p, _ = cand.factors(name)
        partition_workers *= p
    cap = arr.bank_cap if arr.bank_cap is not None else BIG
    return max(1, min(cap, partition_workers))


def _waves(spec: KernelSpec, cand: Candidate) -> int:
    w = 1
    for lv in spec.levels:
        if lv.tiled():
            p, u = cand.factors(lv.name)
            w *= _ceil_div(lv.trip, p * u)
    return max(1, w)


@dataclass
class CandResult:
    cand: Candidate
    p_tot: int
    active_L: int
    active_S: int
    exposure: dict
    exposed_iters: int          # total inner iterations in one chunk
    waves: int
    CP: int
    A: int
    LD: int
    ST: int
    chunk_aggregate: int
    chunk_scheduled: int
    pragma_exposure_aggregate: int
    schedule_estimate: int
    binding_class: str
    saturation: str             # latency-bound | resource-bound
    util: tuple                 # (P, L, S)
    flags: set = field(default_factory=set)


def evaluate_candidate(spec: KernelSpec, cand: Candidate,
                       cfg: Config) -> CandResult:
    exp = _exposure(spec, cand)
    p_tot = _p_tot(spec, cand)
    B_L = _effective_banks(spec, cand, spec.load_binding)
    B_S = _effective_banks(spec, cand, spec.store_binding)
    active_L = max(1, min(p_tot, B_L, cfg.L))
    active_S = max(1, min(p_tot, B_S, cfg.S))
    eff = Config(cfg.P, active_L, active_S,
                 label=f"P={cfg.P},L={active_L},S={active_S}")

    res = evaluate(spec.build_chunk(exp), spec.name, eff)
    agg = res.region_aggs[0]
    waves = _waves(spec, cand)
    chunk_agg = res.aggregate_cycles
    chunk_sched = res.scheduled_cycles

    # binding memory class of this chunk (L vs S; arithmetic stays global pool)
    binding_class = "L" if agg.load >= agg.store else "S"
    # resource-bound once any resource term reaches CP (the binding class fills
    # its issue width every cycle); latency-bound only while CP strictly
    # dominates every resource term (ports idle draining the critical path).
    max_resource = max(agg.compute, agg.load, agg.store)
    saturation = "latency-bound" if agg.CP > max_resource else "resource-bound"
    denom = chunk_agg if chunk_agg > 0 else 1
    util = (agg.compute / denom, agg.load / denom, agg.store / denom)

    exposed = 1
    for lv in spec.levels:
        exposed *= exp[lv.name]

    return CandResult(
        cand=cand, p_tot=p_tot, active_L=active_L, active_S=active_S,
        exposure=exp, exposed_iters=exposed, waves=waves,
        CP=agg.CP, A=agg.A, LD=agg.LD, ST=agg.ST,
        chunk_aggregate=chunk_agg, chunk_scheduled=chunk_sched,
        pragma_exposure_aggregate=waves * chunk_agg,
        schedule_estimate=waves * chunk_sched,
        binding_class=binding_class, saturation=saturation, util=util)


def absolute_cgra_lb(spec: KernelSpec, cfg: Config) -> tuple[int, dict]:
    """Full-trip aggregate over FULL lanes: the only lower bound. Also returns
    the per-iteration demand for the binding-class analysis."""
    full_exp = {lv.name: lv.trip for lv in spec.levels}
    res = evaluate(spec.build_chunk(full_exp), spec.name + "_full", cfg)
    agg = res.region_aggs[0]
    demand = {"A": agg.A, "LD": agg.LD, "ST": agg.ST, "CP": agg.CP,
              "compute": agg.compute, "load": agg.load, "store": agg.store,
              "aggregate": res.aggregate_cycles}
    return res.aggregate_cycles, demand


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def recommend(spec: KernelSpec, results: list[CandResult],
              cfg: Config) -> CandResult | None:
    """Banking-aware selection (docs/spec-kernel-performance.md, "Exposure
    selection under banking"): saturate the binding memory class -- pick the
    candidate reaching the max achievable active width on the binding class,
    then the SMALLEST such exposure that is resource-bound (fewest workers /
    least unroll). Falls back to max-active latency-bound if none saturate."""
    if not results:
        return None
    binding = results[0].binding_class

    def active(r: CandResult) -> int:
        return r.active_L if binding == "L" else r.active_S

    best_active = max(active(r) for r in results)
    pool = [r for r in results if active(r) == best_active]
    resource_bound = [r for r in pool if r.saturation == "resource-bound"]
    if resource_bound:
        # smallest exposure that already saturates -> fewest workers, least U
        return min(resource_bound,
                   key=lambda r: (r.exposed_iters, r.p_tot,
                                  r.pragma_exposure_aggregate))
    # nobody saturates: take the max-active candidate with the best estimate
    return min(pool, key=lambda r: (r.pragma_exposure_aggregate, r.exposed_iters))


def annotate_flags(spec: KernelSpec, results: list[CandResult],
                   rec: CandResult) -> None:
    binding = results[0].binding_class

    def active(r: CandResult) -> int:
        return r.active_L if binding == "L" else r.active_S

    best_active = max(active(r) for r in results)
    for r in results:
        if r is rec or (r.cand.split == rec.cand.split):
            r.flags.add("recommended")
        if active(r) < best_active:
            r.flags.add("bandwidth-starved")
        # port-serialized: same active width as the recommendation but more
        # exposure (extra unroll/parallel past the knee) with no throughput gain
        if (active(r) == active(rec) and r.exposed_iters > rec.exposed_iters):
            r.flags.add("oversubscribed")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _dedup(results: list[CandResult]) -> list[list[CandResult]]:
    """Group candidates that produce identical performance (same active widths
    and chunk counts and waves) -- e.g. inert inner-reduction pragmas."""
    groups: dict[tuple, list[CandResult]] = {}
    for r in results:
        key = (r.active_L, r.active_S, r.LD, r.ST, r.A, r.CP, r.waves,
               r.pragma_exposure_aggregate)
        groups.setdefault(key, []).append(r)
    ordered = sorted(groups.values(),
                     key=lambda g: (g[0].pragma_exposure_aggregate,
                                    -g[0].p_tot, g[0].exposed_iters))
    return ordered


def _fmt_util(u):
    return "/".join(str(round(x * 100)) for x in u)


def render_report(spec: KernelSpec, cfg: Config, results: list[CandResult],
                  lb: int, demand: dict, rec: CandResult, top: int = 0) -> str:
    L = []
    L.append(f"# Loom pragma DSE (banking-aware): {spec.name}  "
             f"({cfg.label})")
    L.append("")
    nest = ", ".join(f"{lv.name}[{lv.trip},{lv.kind}]" for lv in spec.levels)
    L.append(f"loop nest (outer->inner): {nest}")
    L.append(f"banking: {spec.banking_note}")
    L.append("")
    L.append(f"absolute_cgra_lb = {lb}  (full-trip aggregate over full lanes "
             f"L={cfg.L},S={cfg.S}; the ONLY lower bound)")
    L.append(f"full-trip counts: A={demand['A']} LD={demand['LD']} "
             f"ST={demand['ST']} CP={demand['CP']} | "
             f"compute={demand['compute']} load={demand['load']} "
             f"store={demand['store']}")
    binding = results[0].binding_class
    L.append(f"binding memory class = {binding}   "
             f"(P_tot scales active_{binding} up to banks and cap {cfg.cap(binding)})")
    L.append("")
    L.append("Only absolute_cgra_lb is a lower bound. pragma_agg / sched_est "
             "assume waves do NOT overlap and sit above it.")
    L.append("active_L/active_S = min(P_tot, banks, lane cap): parallel raises "
             "them, unroll does not. This is the P-vs-U distinction.")
    L.append("")

    header = (f"{'flags':<10} {'split':<26} {'Ptot':>4} {'aL':>3} {'aS':>3} "
              f"{'exp':>5} {'wav':>5} {'cagg':>5} {'p_agg':>7} {'sched':>7} "
              f"{'class':<14} {'util P/L/S':>11}")
    L.append(header)
    L.append("-" * len(header))
    all_groups = _dedup(results)
    shown = all_groups
    omitted = 0
    if top and len(all_groups) > top:
        shown = all_groups[:top]
        rec_shown = any("recommended" in g[0].flags for g in shown)
        if not rec_shown:
            rec_group = next((g for g in all_groups
                              if "recommended" in g[0].flags), None)
            if rec_group is not None:
                shown = shown + [rec_group]
        omitted = len(all_groups) - len(shown)
    for group in shown:
        r = group[0]
        fl = ""
        if "recommended" in r.flags:
            fl += "K"
        if "bandwidth-starved" in r.flags:
            fl += "b"
        if "oversubscribed" in r.flags:
            fl += "o"
        if len(group) > 1:
            split = r.cand.signature() + f"  (+{len(group)-1} equiv)"
        else:
            split = r.cand.signature()
        L.append(
            f"{fl:<10} {split:<26} {r.p_tot:>4} {r.active_L:>3} {r.active_S:>3} "
            f"{r.exposed_iters:>5} {r.waves:>5} {r.chunk_aggregate:>5} "
            f"{r.pragma_exposure_aggregate:>7} {r.schedule_estimate:>7} "
            f"{r.saturation:<14} {_fmt_util(r.util):>11}")
    if omitted:
        L.append(f"... ({omitted} more groups omitted; all bandwidth-starved, "
                 "sorted by p_agg -- use --top 0 for the full sweep)")
    L.append("")
    L.append(f"RECOMMENDED: {rec.cand.signature()}  -> P_tot={rec.p_tot}, "
             f"active_{binding}={rec.active_L if binding=='L' else rec.active_S}, "
             f"pragma_agg={rec.pragma_exposure_aggregate} "
             f"({rec.pragma_exposure_aggregate/lb:.2f}x the floor)")
    L.append("flags: K=recommended (saturation knee at max bandwidth), "
             "b=bandwidth-starved (memory ports idle -> raise P), "
             "o=oversubscribed (extra unroll/exposure past the knee, no gain).")
    L.append("")
    L.append(_pu_contrast(spec, cfg, results))
    return "\n".join(L)


def _pu_contrast(spec: KernelSpec, cfg: Config,
                 results: list[CandResult]) -> str:
    """Fixed-product P-vs-U contrast on the primary parallelizable level: hold
    p*u constant, vary the split, show the load/aggregate difference."""
    prim = None
    for lv in spec.levels:
        if lv.kind == "parallel":
            prim = lv
            break
    if prim is None:
        for lv in spec.levels:
            if lv.parallelizable():
                prim = lv
                break
    if prim is None:
        return "P-vs-U contrast: no parallelizable level."

    # choose the largest product on the primary level that has >=2 splits and
    # holds the other levels at (1,1)
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

    # pick the product whose splits spread the MOST (largest max/min p_agg) --
    # for a bank-capped level the max product may be past the cap where all
    # splits tie, so the widest-spread product is the illustrative one.
    def spread(prod):
        vals = [r.pragma_exposure_aggregate for r in candidates[prod]]
        return (max(vals) / min(vals), prod)

    prod = max(candidates, key=spread)
    rows = sorted(candidates[prod],
                  key=lambda r: -r.cand.factors(prim.name)[0])
    out = [f"P-vs-U at fixed product {prod} on level '{prim.name}' "
           f"(other levels at P1U1):"]
    out.append(f"  {'split':<12} {'Ptot':>4} {'active':>6} {'p_agg':>7} "
               f"{'note':<28}")
    best = min(r.pragma_exposure_aggregate for r in rows)
    for r in rows:
        p, u = r.cand.factors(prim.name)
        a = r.active_L if r.binding_class == "L" else r.active_S
        note = "best (all bandwidth)" if r.pragma_exposure_aggregate == best \
            else f"{r.pragma_exposure_aggregate/best:.1f}x slower (unroll serializes)"
        out.append(f"  P{p}U{u:<10} {r.p_tot:>4} {a:>6} "
                   f"{r.pragma_exposure_aggregate:>7} {note:<28}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(spec: KernelSpec, cfg: Config, max_parallel: int, max_unroll: int,
        exposure_cap: int, top: int = 0) -> tuple[str, CandResult, int]:
    lb, demand = absolute_cgra_lb(spec, cfg)
    cands = enumerate_candidates(spec, max_parallel, max_unroll, exposure_cap)
    results = [evaluate_candidate(spec, c, cfg) for c in cands]
    rec = recommend(spec, results, cfg)
    annotate_flags(spec, results, rec)
    report = render_report(spec, cfg, results, lb, demand, rec, top=top)
    return report, rec, lb


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Banking-aware Loom-pragma design-space estimate")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("kernel", nargs="?",
                        help="kernel name: " + ", ".join(sorted(KERNELS)))
    parser.add_argument("--config", default="6x6")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--max-unroll", type=int, default=8)
    parser.add_argument("--exposure-cap", type=int, default=256,
                        help="skip candidates whose tiled exposure exceeds this")
    parser.add_argument("--top", type=int, default=0,
                        help="show only the best N candidate groups (0 = all)")
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_tests()
    if not args.kernel or args.kernel not in KERNELS:
        parser.print_help()
        print("\nkernels: " + ", ".join(sorted(KERNELS)))
        return 1
    spec = KERNELS[args.kernel]
    cfg = parse_config(args.config)
    report, _, _ = run(spec, cfg, args.max_parallel, args.max_unroll,
                       args.exposure_cap, top=args.top)
    print(report)
    return 0


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> int:
    errors: list[str] = []
    cfg = parse_config("6x6")

    # axpy: the banking distinction must make parallel beat unroll at fixed
    # product. Build P8U1 vs P1U8 (product 8) and compare.
    spec = KERNELS["axpy"]
    p8 = evaluate_candidate(spec, Candidate((("i", 8, 1),)), cfg)
    p1 = evaluate_candidate(spec, Candidate((("i", 1, 8),)), cfg)
    if not (p8.active_L > p1.active_L):
        errors.append(f"axpy P8U1 active_L {p8.active_L} !> P1U8 {p1.active_L}")
    if not (p8.pragma_exposure_aggregate < p1.pragma_exposure_aggregate):
        errors.append(
            f"axpy P8U1 p_agg {p8.pragma_exposure_aggregate} !< P1U8 "
            f"{p1.pragma_exposure_aggregate} (parallel must beat unroll)")
    # same product -> identical chunk op counts (product-only op counts)
    if (p8.A, p8.LD, p8.ST, p8.CP) != (p1.A, p1.LD, p1.ST, p1.CP):
        errors.append("axpy P8U1 vs P1U8 chunk op counts must be identical "
                      "(op counts are product-only; only ports differ)")

    # axpy absolute_cgra_lb over full lanes should match the legacy value 65.
    lb, _ = absolute_cgra_lb(spec, cfg)
    if lb != 65:
        errors.append(f"axpy absolute_cgra_lb {lb} != 65")

    # gemv: inner column (j) parallelism must be inert (A banked over rows),
    # while row (i) parallelism raises active_L.
    g = KERNELS["gemv"]
    gi = evaluate_candidate(g, Candidate((("i", 4, 1), ("j", 1, 1))), cfg)
    gj = evaluate_candidate(g, Candidate((("i", 1, 1), ("j", 4, 1))), cfg)
    if gi.active_L <= gj.active_L:
        errors.append(f"gemv row-parallel active_L {gi.active_L} must exceed "
                      f"col-parallel {gj.active_L} (A banked over rows)")
    # A bank cap 4: row parallelism beyond 4 must not raise active_L past 4.
    g8 = evaluate_candidate(g, Candidate((("i", 8, 1), ("j", 1, 1))), cfg)
    if g8.active_L != 4:
        errors.append(f"gemv P_i=8 active_L {g8.active_L} != 4 (bank cap)")

    # vecsum: reduction is legal to parallelize; loads scale with P_tot.
    v = KERNELS["vecsum"]
    v1 = evaluate_candidate(v, Candidate((("i", 1, 8),)), cfg)
    v8 = evaluate_candidate(v, Candidate((("i", 8, 1),)), cfg)
    if v8.active_L <= v1.active_L:
        errors.append("vecsum parallel reduction must raise active_L over unroll")
    if v8.pragma_exposure_aggregate >= v1.pragma_exposure_aggregate:
        errors.append("vecsum parallel reduction must beat unroll on p_agg")
    if v8.saturation != "resource-bound" or v8.CP >= v8.chunk_aggregate:
        errors.append("vecsum P8 must be load-bound (not CP/latency-bound)")

    # tridiag_solve: sequential -> P forced to 1, no P-vs-U distinction, and
    # CP (serial chain) must dominate the aggregate.
    t = KERNELS["tridiag_solve"]
    tu1 = evaluate_candidate(t, Candidate((("i", 1, 1),)), cfg)
    tu8 = evaluate_candidate(t, Candidate((("i", 1, 8),)), cfg)
    if tu1.p_tot != 1 or tu8.p_tot != 1:
        errors.append("tridiag_solve must force P_tot=1 (sequential carry)")
    if tu1.active_L != 1 or tu1.active_S != 1:
        errors.append("tridiag_solve must have active_L=active_S=1 (no banking)")
    if tu1.pragma_exposure_aggregate != tu8.pragma_exposure_aggregate:
        errors.append("tridiag_solve U1 vs U8 must be identical (no distinction)")
    tlb, tdem = absolute_cgra_lb(t, cfg)
    if tdem["CP"] != tlb:
        errors.append("tridiag_solve floor must be CP-bound (serial recurrence)")
    # parallel candidate must be illegal (not enumerated).
    tcands = enumerate_candidates(t, 8, 8, 256)
    if any(p > 1 for c in tcands for _, p, _ in c.split):
        errors.append("tridiag_solve must not enumerate any parallel factor > 1")

    # every kernel: run end-to-end, recommendation exists, bracket holds.
    for name, ks in KERNELS.items():
        report, rec, lb = run(ks, cfg, 8, 8, 256)
        if rec is None:
            errors.append(f"{name}: no recommendation produced")
            continue
        for r in [rec]:
            if not (lb <= r.pragma_exposure_aggregate <= r.schedule_estimate):
                errors.append(
                    f"{name}: bracket violated lb={lb} "
                    f"pragma={r.pragma_exposure_aggregate} "
                    f"sched={r.schedule_estimate}")

    if errors:
        for e in errors:
            print(f"  SELF-TEST FAIL: {e}")
        return 1
    print("[PASS] loom_dse banking-aware self-tests")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
