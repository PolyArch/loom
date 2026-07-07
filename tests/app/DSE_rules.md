# Loom-Pragma DSE Rules

This file summarizes the shared rules used by
`tests/app/<kernel>/<kernel>_loom_dse.md`. The authoritative definition remains
the "Optional Loom-Pragma Design-Space Estimate" section of
[`docs/spec-kernel-performance.md`](../../docs/spec-kernel-performance.md); if
this summary and the spec diverge, the spec wins.

## Scope and objective

The Loom-pragma DSE is an optional estimate for comparing explicit
`LOOM_PARALLEL(P)` / `LOOM_UNROLL(U)` choices before committing to compiler or
hardware mapping work. It is not the fully-unrolled ASAP metric, not the
aggregate CGRA lower bound from the main eval files, and not cycle-accurate RTL.

The objective is to minimize estimated cycles subject to the machine load/store
lane caps. For the standard `6x6` resource configuration, `P_pe = 36` arithmetic
lanes and `L = S = 12` load/store lanes. The model has no separate area term, no
control/body area tradeoff, and no separate control resource; all effects reduce
or increase work inside the existing `P`/`L`/`S` pools.

## Reported quantities

Per-candidate estimates use wave-serialized chunks:

```text
exposed_iters = min(trip_count, P_tot * U_tot)
waves         = ceil(trip_count / exposed_iters)
```

The table reports two candidate estimates:

- `pragma_exposure_aggregate` (`p_agg`): wave-summed aggregate estimate.
- `schedule_estimate` (`sched`): wave-summed finite-resource schedule estimate.

Both are estimates, not lower bounds. The only lower bound in these DSE files is
`absolute_cgra_lb`, the full-trip, fully-coalesced, fully-amortized aggregate over
the full machine lanes:

```text
absolute_cgra_lb <= pragma_exposure_aggregate <= schedule_estimate
```

The right two quantities assume waves do not overlap, so real pipelined dataflow
can fall below `p_agg` / `sched` toward `absolute_cgra_lb`. Real hardware and DFG
lowering overhead can also push measured cycles above `sched`.

Because the DSE credits vector memory operations and control amortization, its
`absolute_cgra_lb` can be below the scalar Metric-1 aggregate in a kernel's main
`*_eval.md` file. That gap is expected when vector coalescing or induction
amortization applies.

## P and U semantics

`P_tot` is the product of `LOOM_PARALLEL` factors over parallelizable levels.
`U_tot` is the product of `LOOM_UNROLL` factors. A shorthand such as `P4U16`
means four parallel workers and an unroll factor of sixteen on that level.

Loop legality still comes from the source dependence:

- Dependency-parallel loops may use parallel workers and unroll.
- Sequential loops must preserve carried recurrences; their iterator and carried
  state remain per-iteration and can sit on the critical path.
- Reductions may use parallel workers only when the carried operation is modeled
  as a legal reduction tree.

Algorithmic arithmetic and `CP` use a global pool. A kernel's intended math does
not become cheaper merely because exposure came from `P` or from `U`. The split
matters through the two DSE-specific effects below.

## Lane-aware memory and vector coalescing

The current provisional model ignores explicit `LOOM_MEMORY_BANK` parameters and
address-level bank conflicts. The only caps on eligible memory exposure are the
machine load/store lane counts `L` and `S`.

This is the mentor-confirmed reversal from the old banking model, which favored
`LOOM_PARALLEL`. Under the lane-aware + vector-coalescing model, contiguous /
tiled loops generally favor `LOOM_UNROLL`: coalescing saturates at `U = V = 4`,
but control amortization keeps paying as `U` grows whenever the saved iterator
work affects a binding term.

For scalar independent accesses, unrolled memory operations may issue to
different lanes when they are independent and target-distinct. `P4U1` and `P1U4`
can therefore expose the same four scalar load lanes when their memory exposure
is equally eligible.

For contiguous same-array 64-bit accesses, the model may coalesce up to
`V = 4` elements into one 256-bit vector memory operation:

```text
vector_loads  = ceil(load_group_elems / V)
vector_stores = ceil(store_group_elems / V)
```

One vector load occupies one load lane for that cycle and produces up to four
scalar values after zero-cost unpack. Vector stores and zero-cost pack follow the
same rule. `LOOM_UNROLL` exposes adjacent accesses inside one worker, so it can
earn this coalescing credit for contiguous groups. `LOOM_PARALLEL` workers stride
across partitions and do not coalesce with each other. The coalescing credit is
bounded by `V` and is gone once the group is already fully coalesced.

## Recurring vs. invariant loads

A chunk's load lane-slots split into two kinds, and only one drives the binding:

- **Recurring loads** (`LD_rec`) — per-iteration array element loads over the
  tiled index, plus induction reads. Their count **scales with exposure**. These
  set the steady-state load lane exposure and the binding load term:
  `load = ceil(LD_rec / L)`, `aL = min(LD_rec, L)`.
- **Invariant loads** (`LD_inv`) — values hoisted once per chunk, count
  **independent of exposure**: a kernel's size/param scalars, `alpha`/`beta`, a
  reduction's seed, and any array loaded once and reused across the tiled loop
  (e.g. gemv's whole `x` vector, which is invariant of the row index). They are
  **amortized** — loaded once and held, broadcast by free fan-out — so they do
  **not** establish sustained per-cycle lane pressure and are excluded from the
  binding load term.

The reported total traffic is `LD_eff = LD_rec + LD_inv`. Only `LD_rec` binds;
`LD_inv` is reported (so the full traffic stays visible) but amortized out of the
steady-state rate and the `absolute_cgra_lb` floor. The finite-resource
`schedule_estimate` still issues each wave's `LD_inv` re-load, so for kernels with
a large invariant (gemv's `x`) `sched` can sit noticeably above the
invariant-amortized `p_agg`.

## Control amortization

Within an exposed wave, spatially unrolled iterations share one iterator per
worker. The induction load/add/store/compare is charged once per worker per wave,
not once per element:

```text
control work over the loop ~= P_tot * waves ~= trip_count / U_tot
```

This is why unroll-heavy splits can keep improving after vector coalescing has
saturated. A fully consumed reduction is lowered as a spatial tree and carries no
per-element loop control for the reduced dimension. Sequential carried
recurrences are the exception: they cannot be spatially flattened, so their
iterator and carried state stay per-iteration.

This DSE-local rule is more optimistic than the ASAP baseline in the main evals,
which charges induction per iteration as a conservative source-level accounting
rule.

## Exposure selection

Do not choose the maximum `P_tot * U_tot` merely because the wave-summed estimate
keeps decreasing with exposure. The recommendation is the saturation knee:
the smallest legal exposure whose binding resource becomes resource-bound.

Rows are flagged as:

- `K`: recommended knee.
- `b`: bandwidth-starved / latency-bound; resources are still idle, so more
  exposure improves throughput.
- `o`: oversubscribed; additional exposure is past the knee and mainly adds
  transient backlog, area pressure, or mapping pressure for little or no modeled
  throughput gain.

`peak_ready_backlog` is a transient list-schedule diagnostic, not a feasibility
constraint. Prefer the per-class utilization columns for pressure: a binding
class reaches `100%` exactly when the wave is resource-bound.

## Table columns and flags

- `flags`: `K`, `b`, and `o` markers described above.
- `split`: the tested pragma factoring, such as `i:P1U64`.
- `Ptot`: total parallel workers requested by the split.
- `aL` / `aS`: active load/store lane slots the candidate presents per cycle,
  after vector coalescing and clamped to the machine caps. `aL` counts
  **recurring** loop loads only (`min(LD_rec, L)`); one-time invariant loads are
  amortized out (see *Recurring vs. invariant loads*).
- `LD_eff`: total load traffic for the chunk, `LD_rec + LD_inv` (recurring plus
  one-time invariant loads). Reported for visibility; the binding load term uses
  `LD_rec` only.
- `exp`: exposed iterations per wave.
- `wav`: number of waves needed to cover the trip count.
- `cagg`: aggregate estimate for one exposed wave.
- `p_agg`: wave-summed `pragma_exposure_aggregate`; lower is better for comparing
  candidates, but it is not a lower bound.
- `sched`: finite-resource `schedule_estimate`; also not a lower bound.
- `class`: whether the wave is `latency-bound` or `resource-bound`.
- `util P/L/S`: per-class utilization, computed as each class term divided by
  the wave aggregate.

## Comparing measured DFG cycles

The bracket

```text
absolute_cgra_lb <= pragma_exposure_aggregate <= schedule_estimate
```

relates model quantities only. It is not a bound on measured DFG simulator
cycles. When simulator cycles are available, useful ratios are:

- `sim / absolute_cgra_lb`: distance from the resource floor.
- `sim / p_agg`: distance from the wave-serialized aggregate estimate.
- `sim / schedule_estimate`: overhead beyond the finite-resource schedule
  estimate, including lowering, mapping, handshake backpressure, and memory
  latency effects.

## Helper

The reference helper is `tests/scripts/loom_dse.py`. It builds each candidate's
vectorized chunk DAG, schedules it with `tests/scripts/cgra_schedule.py`, sums
over waves, and emits the candidate table, `absolute_cgra_lb`, and recommended
saturation-knee exposure.
