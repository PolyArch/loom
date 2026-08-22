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
add whole-working-set capacity and DSE-local scratchpad-port constraints. The model
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

Maintained extended-profile recommendations model the concrete parameters in
each kernel's `main.cpp` smoke test. Reports name those parameters and their
source; the extended helper does not replace them with synthetic square or
power-of-two sizes. A separate sensitivity run may use different dimensions
only when it is labeled as a non-smoke-test study and kept out of the maintained
recommendation and summary row. Fixture changes require fresh capacity,
traffic, cycle, and recommendation results rather than scaled values. Legacy
direct-memory reports keep their documented fixture policy until they migrate
to an extended profile.

When no concrete hardware target is supplied, the fallback study profile is
`shared-spad-4k-r2w2-v4` with one 4096-byte scratchpad shared by the workers of
one mapped kernel, two logical load ports, two logical store ports, one-cycle
modeled access, and fixed vector width `V = 4` source elements. This 2R/2W
profile is an analytical baseline, not a guarantee that residency beats direct
memory. Capacity, load/store-port counts, and access latency are target
parameters and part of profile identity. A report modeling a compatible
kernel-shared `fabric.memory` with positive `ldCount` and `stCount` must pass its
capacity and port counts as overrides. Other Fabric memory configurations are
outside this optional model and must not be represented silently by the
fallback. The fallback profile does not assert a bank topology or DMA engine.

### Loop order, fixed width, and explicit jam

Extended kernels search only explicitly declared legal loop orders, including
the source order. After selecting an order, the helper recomputes which access is
actually contiguous, so coalescing follows the selected order and address
function. Vector width is not searched: contiguous groups use
`ceil(group_elems / 4)`, including one vector node for a partial final group. A
future width search requires width-dependent cost or legality such as masked-lane
or pack/unpack overhead or alignment restrictions.

Outer-loop unrolling does not imply jam. Each candidate selects one complete
per-kernel jam plan, and `none` is always legal. A nonempty plan is legal only
when each named outer loop is a dependency-parallel DSE level with `U > 1` and
the selected order places each inner loop beneath its outer loop. Jam does not
multiply exposure. Until a separate DFG-size or routing model is specified, it
adds no arithmetic, control, routing, or area cost; it may only remove declared
invariant operand loads. GEMV therefore searches both `jam=none` and
`jam=i-j-share-x` rather than granting `i->j[x]` sharing to every row-unrolled
candidate.

Complete jam plans and shared operands are explicit per-kernel metadata. The
helper never infers an edge from address equality and does not enumerate an
arbitrary power set of edges.

### Scratchpad placement, ports, and fallback

Scratchpad capacity, logical load/store-port counts, latency, and sharing scope
come from the named analytical profile. A run receives these values explicitly;
it does not infer them from a source `LOOM_MEMORY_BANK` annotation. Buffer intent
is declared:

- `resident_shared`: read-only, worker-invariant reuse is stored once.
- `resident_replicated`: genuinely worker-specific reusable data has the required
  number of private copies.
- `direct`: reuse-free streaming data stays in external memory by classification.

The helper lays out the complete proposed resident set once, in declaration
order, with vector-width alignment and deterministic compact logical offsets. If
that whole set fits, the proposed buffers are resident. If it does not fit, all
proposed resident buffers become `direct-fallback`; the candidate remains legal.
The DSE does not search smaller scratchpad tiles or resident-buffer subsets.

Every dynamic resident-data access consumes scratchpad bandwidth. Free read
fan-out is allowed only for the same address requested at the same logical inner
step. A scalar or coalesced vector operation occupies one port for the target's
access latency. For one ordered region:

```text
spad_read_lb  = ceil(spad_R / load_ports) * access_cycles
spad_write_lb = ceil(spad_W / store_ports) * access_cycles
spad_port_lb  = max(spad_read_lb, spad_write_lb)
```

`spad_port_sched` is deterministic stable-order earliest-fit packing onto the
separate load and store ports. A scratchpad region uses
`max(existing_aggregate, spad_port_lb)` for its aggregate and
`max(existing_schedule, spad_port_sched)` for its schedule estimate.

### Whole-working-set preload

When residency succeeds, preload occurs once per modeled kernel execution, never
once per wave. Each resident fill groups unique scalar elements by contiguous
address and uses the fixed `V = 4` convention. Every coalesced group emits one
external `L` operation and one scratchpad-write operation:

```text
preload_scalar_elems = sum(group_elems for group in contiguous_groups)
preload_L_ops        = sum(ceil(group_elems / V) for group in contiguous_groups)
preload_spad_W_ops   = preload_L_ops
```

Partial final groups still occupy one lane slot. Fallback plans have zero preload
and zero scratchpad traffic; their proposed resident accesses become direct
external-memory traffic. The saturation knee is chosen from recurring compute,
not the preload prologue.

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
that candidate's transformed operation set, fixed-width coalescing, deterministic
placement, scratchpad-port correction, and once-per-kernel preload, without
partial-exposure wave serialization:

```text
absolute_cgra_lb <= plan_cgra_lb
                 <= pragma_exposure_aggregate
                 <= schedule_estimate
```

For each kernel/configuration/profile identity, including capacity and port
counts, there is one global `absolute_cgra_lb`. In an extended profile it is the
minimum legal `plan_cgra_lb`; a candidate-specific plan floor never replaces the
global value.
Unmodified kernels retain the existing full-trip, fully-coalesced,
fully-amortized `absolute_cgra_lb` over the full machine lanes.

The right two quantities assume waves do not overlap, so real pipelined dataflow
can fall below `p_agg` / `sched` toward the corresponding plan or global floor.
Real hardware and DFG lowering overhead can also push measured cycles above
`sched`. Only `absolute_cgra_lb` is the report-global lower bound;
`plan_cgra_lb` is a candidate-specific transformed-plan floor. Never call
`p_agg` or `sched` a lower bound.

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
Extended pilots additionally apply their declared loop order, explicit jam plan,
and deterministic resident-or-fallback memory plan.

## Lane-aware memory and vector coalescing

The legacy direct-memory path ignores explicit `LOOM_MEMORY_BANK` parameters and
address-level external-memory bank conflicts. Its only caps on eligible memory
exposure are the machine load/store lane counts `L` and `S`. The extended profile
also does not model banking; it applies only the explicit scratchpad load/store
port rule above.

This is the mentor-confirmed reversal from the old banking model, which favored
`LOOM_PARALLEL`. Under the lane-aware + vector-coalescing model, contiguous
parallel loops generally favor `LOOM_UNROLL`: coalescing saturates at `U = V = 4`,
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
  exposed index, plus induction reads. Their count **scales with exposure**. These
  set the steady-state load lane exposure and the binding load term:
  `load = ceil(LD_rec / L)`, `aL = min(LD_rec, L)`.
- **Invariant loads** (`LD_inv`) — values hoisted once per chunk, count
  **independent of exposure**: a kernel's size/param scalars, `alpha`/`beta`, a
  reduction's seed, and any array loaded once and reused across the exposed loop
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

For extended pilots, resource-bound is necessary but not sufficient. Within one
explicit loop-order, jam-plan, and actual-placement family, resident fan-out can
remove recurring traffic at a larger legal exposure, so an early port- or
load-bound row may still be **recurring-traffic immature**. At each family
exposure the helper keeps the minimum recurring-compute frontier (preload
excluded), then compares full-kernel control-free `P/L/S/R/W` demand against the
best values in the same family at that or any larger exposure. A candidate
becomes knee-eligible only when a resource dominates its nominal full compute
wave and at least one tied dominant demand has reached that future minimum. The
first eligible exposure is that family's knee; total `p_agg` and recurring
traffic rank exact frontier ties. The global recommendation is the family knee
with minimum `p_agg`, recurring data traffic, scratchpad-port demand, exposure,
worker count, and deterministic signature, in that order. Jammed and unjammed
candidates are distinct families. If no family has an eligible knee, the helper
uses the global best-estimate fallback. This prevents repeated loads from
manufacturing an artificially early saturation point while leaving the legacy
rule unchanged.

The helper search covers every power-of-two `P` and `U` allowed by the concrete
trip counts and dependency legality. Source pragma values are hints, not search
maxima. There is no implicit factor-of-eight or global exposure cap;
non-power-of-two factors are outside the current DSE scope. Extended pilots take
the cross-product with their explicitly declared legal orders and complete jam
plans. Vector width remains fixed; placement and logical offsets are deterministic
derived decisions, not additional search axes.
Explicit
`--max-parallel`, `--max-unroll`, and
`--exposure-cap` options are bounded diagnostic overrides; reports produced with
them use `BEST BOUNDED`, a `B` marker, and `bounded_search_floor` rather than
claiming a global recommendation or profile-global floor. Fully consumed
reduction or sequential levels may use one canonical `P1U1` label when all
factor labels build the same DAG.

Rows are flagged as:

- `K`: globally recommended family knee, or the global best-estimate fallback
  when no family has an eligible knee.
- `B`: best row in an explicitly bounded diagnostic; not a global
  recommendation.
- `b`: below that row's family knee. A row may be latency-bound, or it may already be
  resource-bound while its dominant recurring traffic is still immature; more
  legal exposure improves the modeled throughput in either case.
- `o`: oversubscribed relative to that row's family knee; additional exposure
  mainly adds
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

- `evidence` / `target`: `analytic_prefilter`, the exact capacity, load/store-port
  counts, access latency, sharing scope, and fixed `V = 4` convention.
- `order`: the selected declared-legal loop order.
- `jam`: the explicit complete jam plan and any operand shared by that plan.
- `memory`: each buffer's `direct`, `direct-fallback`, `resident_shared`, or
  `resident_replicated` placement, deterministic logical base offset, and
  replication decision.
- `capacity`: resident bytes used versus target capacity, including replication
  and alignment, plus whether the proposed resident set fell back to direct.
- `spad_port_lb` / `spad_port_sched`: scratchpad-port floor and deterministic
  port-packing makespan in modeled cycles.
- `preload`, `spad_reads`, and `avoided_direct`: external fill traffic (scalar
  elements plus coalesced external-`L` / scratchpad-write operations),
  scratchpad-read traffic, and external traffic eliminated by residency.
- `plan_cgra_lb`: the candidate-specific transformed-plan floor used in the
  four-term bracket. It is not the report-global `absolute_cgra_lb`.
- `candidates` / `deduped`: total legal candidates and equivalent groups retained
  after deterministic deduplication.

The extended table renders these exact aliases:

- `candidate`: the complete pragma split plus explicit `order=` and `jam=`.
  Source order and `jam=none` are printed rather than hidden by internal defaults.
- `plan_lb`, `p_agg`, and `sched`: `plan_cgra_lb`,
  `pragma_exposure_aggregate`, and `schedule_estimate`, in modeled cycles.
- `cap_B`: resident capacity in bytes; zero when the proposed set uses direct
  fallback.
- `spad lb/s`: total compute/preload `spad_port_lb` and deterministic
  `spad_port_sched`, in modeled cycles.

Selected-plan traffic units are explicit: preload reports scalar elements plus
coalesced external-`L` and scratchpad-write operations; `spad_reads`
reports scalar requests after declared jam fan-out; `avoided_direct` reports
scalar external loads eliminated after paying the preload traffic. Logical buffer
bases are element offsets, while capacity is bytes. Direct fallback is the actual
candidate placement and participates normally in recommendation and the profile
floor.

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
`plan_cgra_lb`, order/jam, placement, traffic, capacity, fallback, and scratchpad
port evidence.
