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
lanes and `L = S = 12` load/store lanes. Named extended target profiles may also
add tile-capacity legality and a DSE-local scratchpad-bank correction. The model
has no separate area term, no control/body area tradeoff, and no new scheduler
resource class.

Checked-in kernel notes keep the detailed table and explanation on `6x6`. They
may append concise `4x4` and `8x8` split recommendations computed by independent
runs of the same legal power-of-two search; these lines do not choose between
CGRA sizes.

## Extended pilot model

Only Batchnorm, GEMV, and Conv2d use the extended model in this revision. Their
reports carry evidence kind `analytic_prefilter`: the estimator may study loop
transformations and target features that this branch's compiler, mapper, and
hardware do not materialize. Every such assumption is named in the report. All
other kernels keep the legacy source-order, direct-memory columns and behavior.

The canonical study profile is `shared-spad-4k-v4` with one 4096-byte scratchpad
shared by the workers of one mapped kernel, four cyclic single-ported banks,
one-cycle modeled access, fixed vector width `V = 4` source elements, and serial
preload. Profile identity includes preload mode. `ideal_dma` is a separate
double-buffered sensitivity mode, not confirmed hardware and not part of the
canonical serial ranking.

### Loop order, fixed width, and automatic jam

Extended kernels search only explicitly declared legal loop orders, including
the source order. After selecting an order, the helper recomputes which access is
actually contiguous, so coalescing follows the selected order and address
function. Vector width is not searched: contiguous groups use
`ceil(group_elems / 4)`, including one vector node for a partial final group. A
future width search requires width-dependent cost or legality such as masked-lane
or pack/unpack overhead, alignment restrictions, or width-dependent bank
conflicts.

Unrolling a non-innermost tiled level automatically derives its maximal legal
unroll-and-jam: the unrolled outer-loop copies advance together at each inner
step. Jam is not searched, does not multiply exposure again, and adds no modeled
arithmetic, control, or area cost. It may remove redundant loads invariant in the
jammed outer dimension, reducing load and total node counts. Reports identify the
derived jam edges; for GEMV, `i->j` shares `x[j]` across simultaneous rows while
each row retains a private reduction accumulator.

Jam legality and shared operands come from explicit per-kernel `JamRule`-style
metadata. The derived plan uses the candidate's selected order and split plus
that declaration; it never infers an edge or shared operand from address equality
or access patterns. Undeclared jam edges are illegal.

### Scratchpad placement and banking

Scratchpad capacity, banking, latency, and sharing scope come from the named
analytical profile, not from `fabric.memory`, `is_private`, `numRegion`, or
`LOOM_MEMORY_BANK`. Placement is derived:

- `resident_shared`: read-only, worker-invariant reuse is stored once.
- `resident_replicated`: genuinely worker-specific reusable data has the required
  number of private copies.
- `direct`: reuse-free streaming data stays in external memory by classification.

Direct memory is not a capacity-overflow fallback. A resident working set that
does not fit makes the tile illegal. Resident buffers receive deterministic
logical base offsets. Held-once buffers occupy a stable region in metadata
declaration order. Per-tile-refilled buffers are packed together in declaration
order into one frame sized by the largest complete concrete-tile layout; serial
allocates one frame and `ideal_dma` allocates two. This uses the maximum combined
tile footprint, not the sum of each buffer's separate maximum. Bases and replica
segments are aligned to four elements. Each tile's sorted unique source indices
are compacted into its held segment or refill frame, so sparse source-address
holes consume no capacity while alignment, held data, replicas, and refill frames
do. With that base-adjusted compact `element_index`, the canonical mapping is
`bank(element_index) = element_index % 4`.

Every dynamic resident-data access consumes scratchpad bandwidth, including
preload writes. Free read fan-out is allowed only for the same address requested
at the same logical inner step. Each scalar or vector read or write reserves every
bank it touches. For per-bank reservation counts `demand[b]`:

```text
total_bank_slots = sum(demand[b] for b in banks)
bank_lb          = max(max(demand[b] for b in banks),
                       ceil(total_bank_slots / bank_count))
```

`bank_sched` is a deterministic earliest-fit packing in stable node order: an
operation is placed only when all banks it touches are free. A scratchpad region
uses `max(existing_aggregate, bank_lb)` for its aggregate and
`max(existing_schedule, bank_sched)` for its schedule estimate.

### Virtual tiles and preload modes

The tile search is analytical and does not change source or compiler behavior.
For a tileable trip `T`, legal sizes are powers of two `<= T` plus the exact full
trip `T`. Non-dividing sizes create exact smaller tail tiles. Reuse-bearing
buffers are resident and sized from each tile's unique address set; reuse-free
buffers are streaming/direct. Every concrete tile's footprint includes source
element size and the derived replication factor and must fit the profile.
For each tileable level, the nominal candidate must satisfy
`parallel * unroll <= tile_size`; a smaller tail tile simply activates fewer
lanes and is costed with its actual shape.

For concrete tiles `0 .. N - 1`, `preload[t]` is the exact external-load,
scratchpad-write, and preload-bank cost described below; `compute[t]` is the
tile's recurring compute, direct-memory, scratchpad-read, and bank cost. Canonical
serial composition is:

```text
serial_total = sum(preload[t] + compute[t] for t in 0 .. N - 1)
```

Serial capacity counts every resident allocation `1x`. The separate sensitivity
uses exact, possibly nonuniform tile and tail costs:

```text
ideal_dma_total = preload[0]
                + sum(max(compute[t], preload[t + 1]) for t in 0 .. N - 2)
                + compute[N - 1]
```

`ideal_dma` counts per-tile-refilled allocations `2x` and held-once allocations
`1x`; it assumes the inactive ping-pong fill does not contend with current-tile
scratchpad reads. Preload occurs once per concrete tile, never once per wave. The
saturation knee is chosen from recurring compute, not the preload prologue. The
report keeps the detailed serial result and appends a concise, separately labeled
`ideal_dma` recommendation.

Each resident fill groups unique scalar elements by contiguous address and uses
the same fixed `V = 4` coalescing convention as compute. Every coalesced group
emits one external `L` operation and one corresponding scratchpad `S` operation:

```text
preload_scalar_elems = sum(group_elems for group in contiguous_groups)
preload_L_ops        = sum(ceil(group_elems / V) for group in contiguous_groups)
preload_spad_S_ops   = preload_L_ops
```

Partial final groups still occupy one lane-slot. Scratchpad preload writes reserve
their exact destination-bank sets and participate in `bank_lb` and `bank_sched`.
`preload[t]` therefore includes the fill's `L`/`S` lane-slot and bank cost. Reports
separate scalar preload elements from coalesced lane-slot operations when both are
shown.

## Reported quantities

Per-candidate estimates use wave-serialized chunks:

```text
exposed_iters = min(trip_count, P_tot * U_tot)
waves         = ceil(trip_count / exposed_iters)
```

The table reports two wave-serialized candidate estimates:

- `pragma_exposure_aggregate` (`p_agg`): wave-summed aggregate estimate.
- `schedule_estimate` (`sched`): wave-summed finite-resource schedule estimate.

Both are estimates, not lower bounds. Legacy candidates use the three-term
bracket:

```text
absolute_cgra_lb <= pragma_exposure_aggregate <= schedule_estimate
```

Extended candidates also report `plan_cgra_lb`, the full-exposure aggregate over
that candidate's transformed operation set, concrete tiles, fixed-width
coalescing, placement, bank floor, and preload mode, without partial-exposure wave
serialization:

```text
absolute_cgra_lb <= plan_cgra_lb
                 <= pragma_exposure_aggregate
                 <= schedule_estimate
```

For each kernel/configuration/profile identity, including preload mode, there is
one global `absolute_cgra_lb`. In an extended profile it is the minimum legal
`plan_cgra_lb`; a candidate-specific plan floor never replaces the global value.
Unmodified kernels retain the existing full-trip, fully-coalesced,
fully-amortized `absolute_cgra_lb` over the full machine lanes.

The right two quantities assume waves do not overlap, so real pipelined dataflow
can fall below `p_agg` / `sched` toward the corresponding plan or global floor.
Real hardware and DFG lowering overhead can also push measured cycles above
`sched`. Only `absolute_cgra_lb` is the report-global lower bound;
`plan_cgra_lb` is a candidate-specific transformed-plan floor. Never call
`p_agg`, `sched`, or the `ideal_dma` sensitivity recommendation a lower bound.

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
matters through vector coalescing and control amortization on the legacy path.
Extended pilots additionally apply their declared loop order, derived jam, tile,
and memory plan.

## Lane-aware memory and vector coalescing

The legacy direct-memory path ignores explicit `LOOM_MEMORY_BANK` parameters and
address-level external-memory bank conflicts. Its only caps on eligible memory
exposure are the machine load/store lane counts `L` and `S`. The extended profile
still does not model external-memory banking; it applies only the explicit
internal scratchpad-bank rule above.

This is the mentor-confirmed reversal from the old banking model, which favored
`LOOM_PARALLEL`. Under the lane-aware + vector-coalescing model, contiguous /
tiled loops generally favor `LOOM_UNROLL`: coalescing saturates at `U = V = 4`,
but control amortization keeps paying as `U` grows whenever the saved iterator
work affects a binding term.

For scalar independent accesses, unrolled memory operations may issue to
different lanes when they are independent and target-distinct. `P4U1` and `P1U4`
can therefore expose the same four scalar load lanes when their memory exposure
is equally eligible.

For contiguous same-array accesses, the model may coalesce up to `V = 4`
same-type source elements into one vector memory operation. In the existing
64-bit examples this is one 256-bit operation:

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

For extended pilots, resource-bound is necessary but not sufficient. Jam,
interchange, and resident fan-out can remove recurring traffic at a larger legal
exposure, so an early bank- or load-bound row may still be
**recurring-traffic immature**. At each exposure the helper keeps the minimum
recurring-compute frontier (preload excluded), then compares full-kernel
control-free `P/L/S` demand and combined compute-bank demand against their best
values at that or any larger exposure. A candidate becomes knee-eligible only
when a resource dominates its nominal full compute wave and at least one tied
dominant demand has reached that future minimum. The first eligible exposure is
the knee; the selected serial or `ideal_dma` composition ranks exact frontier
ties. This prevents repeated loads from manufacturing an artificially early
saturation point while leaving the legacy rule unchanged.

The helper search covers every power-of-two `P` and `U` allowed by the concrete
trip counts and dependency legality. Source pragma values are hints, not search
maxima. There is no implicit factor-of-eight or global exposure cap;
non-power-of-two factors are outside the current DSE scope. Extended pilots take
the cross-product with their explicitly declared legal orders and legal tile
sizes. Vector width remains fixed; maximal jam, placement, logical offsets, and
bank mapping are deterministic derived decisions, not additional search axes.
Explicit
`--max-parallel`, `--max-unroll`, and
`--exposure-cap` options are bounded diagnostic overrides; reports produced with
them are not global recommendations. Fully consumed reduction or sequential
levels may use one canonical `P1U1` label when all factor labels build the same
DAG.

Rows are flagged as:

- `K`: recommended knee.
- `b`: below the knee. A row may be latency-bound, or it may already be
  resource-bound while its dominant recurring traffic is still immature; more
  legal exposure improves the modeled throughput in either case.
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

Extended pilot reports additionally document:

- `evidence` / `target`: `analytic_prefilter`, the exact target-profile identity
  including preload mode, and its fixed `V = 4` convention.
- `order`: the selected declared-legal loop order.
- `jam`: derived maximal legal jam edges and any operand shared by that jam.
- `memory`: each buffer's `direct`, `resident_shared`, or
  `resident_replicated` placement, deterministic logical base offset, and
  replication decision.
- `tile`, `tail`, and `num_tiles`: selected tile sizes, exact tail shape, and
  number of concrete tiles.
- `capacity`: mode-specific resident bytes used versus 4096 bytes, including
  replication and the serial `1x` or `ideal_dma` `1x`/`2x` rule.
- `bank_lb` / `bank_sched`: cyclic-bank floor and deterministic bank-packing
  makespan; the report also identifies conflicts or serialization that cause a
  gap.
- `preload`, `spad_reads`, and `avoided_direct`: external fill traffic (scalar
  elements plus coalesced external-`L` / scratchpad-`S` lane-slot operations),
  scratchpad-read traffic, and external traffic eliminated by residency.
- `plan_cgra_lb`: the candidate-specific transformed-plan floor used in the
  four-term bracket. It is not the report-global `absolute_cgra_lb`.
- `candidates` / `deduped`: total legal candidates and equivalent groups retained
  after deterministic deduplication.

The detailed table and recommendation use canonical serial composition. A concise
`ideal_dma` line is a separately ranked sensitivity result with its overlap and
no-contention assumptions; it is not a lower bound.

## Comparing measured DFG cycles

The legacy bracket

```text
absolute_cgra_lb <= pragma_exposure_aggregate <= schedule_estimate
```

relates model quantities only. It is not a bound on measured DFG simulator
cycles. Extended candidates insert `plan_cgra_lb` between the two leftmost terms.
When simulator cycles are available, useful ratios are:

- `sim / absolute_cgra_lb`: distance from the resource floor.
- `sim / p_agg`: distance from the wave-serialized aggregate estimate.
- `sim / schedule_estimate`: overhead beyond the finite-resource schedule
  estimate, including lowering, mapping, handshake backpressure, and memory
  latency effects.

## Helper

The reference helper is `tests/scripts/loom_dse.py`. It builds each candidate's
vectorized chunk DAG, schedules it with `tests/scripts/cgra_schedule.py`, sums
over waves, and emits the candidate table, report-global `absolute_cgra_lb`, and
recommended saturation-knee exposure. Extended pilots also emit
`plan_cgra_lb`, order/jam, tile, placement, traffic, capacity, and bank evidence,
plus the separate `ideal_dma` sensitivity recommendation.
