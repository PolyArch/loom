# Kernel Performance Modeling Specification

## Overview

This document is the authoritative specification for the kernel performance
metrics reported in `tests/app/<kernel>/<kernel>_eval.md`. It defines **two**
metrics and the rules that relate them:

1. The **aggregate CGRA lower bound** — a closed-form resource/latency bound that
   already exists in the eval files. It stays in place as the lower bound.
2. The **finite-resource schedule estimate** — a new, deterministic
   list-schedule estimate that exposes short windows in which many operations
   become ready at once and temporarily saturate a resource class. It is
   **explicitly not a lower bound and not cycle-accurate RTL**; it is a
   reproducible estimate for the scheduling policy defined here.

Both metrics are computed over the same dynamic operation set, the same resource
classes, and the same one-cycle latency, so that they are directly comparable and
reported side by side.

The normative words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are used in
the usual sense. Where this spec and an implementation disagree, **the spec
wins** (see `.claude/CLAUDE.md`, "Spec-First Development"); align the code to the
spec, or stop and notify the maintainer if the spec is internally contradictory.

The reference implementation of both metrics is the standard-library Python
helper `tests/scripts/cgra_schedule.py` (see [Helper Tool Contract](#helper-tool-contract)).
The committed policy text in `AGENTS.md` and `.claude/CLAUDE.md` (the
"CGRA-Constrained Model" section) points here for the definitions.

## Adopted Baseline: the ASAP Dynamic-Op-Count Model

Both metrics are layered on top of the **ASAP (as-soon-as-possible) dynamic
operation model** documented in
[`tests/app/ASAP_rules.md`](../tests/app/ASAP_rules.md). That model is summarized
here so this spec can be understood standalone; `ASAP_rules.md` remains the
authoritative source for the op-counting conventions themselves, and this spec
does not restate every convention verbatim.

Salient properties of the adopted baseline, all of which this spec depends on:

- **Unlimited hardware, full unrolling.** Every loop dimension with no carried
  dependence is fully unrolled; the model answers "what is the shortest possible
  schedule for this DAG?", not "what would a real machine deliver?".
- **One cycle per operation.** Every counted dynamic operation has unit latency.
- **`total_cycles` (here written `CP`)** is the critical-path depth: the longest
  dependence chain from a kernel input to a kernel output, one cycle per op.
- **Op counts are total dynamic work** and are independent of scheduling. They
  include algorithmic arithmetic, loop-carried state updates, all memory I/O,
  induction variables (per-iter load/add/store/compare), address generation, and
  dead computations. Loop-invariant values are hoisted and charged once.
- **`address_adds`** (adds/subs that appear inside a `[]` subscript expression)
  are tracked separately from regular `adds`, but are still ordinary arithmetic
  work. A bare subscript such as `a[i]` charges **no** `address_add`.
- **Loop classification.** A dimension is *parallel* (contributes its per-iter
  critical path once), *sequential* (contributes `trip × II`), or *reduction*
  (associative carry, tree-reduced: `ceil(log2(trip))` depth, `trip − 1` ops).
- **Control flow (no predication).** Branches are not predicated. Only the ops on
  the taken branch of an `if`/`if-else`/ternary/conditional store are counted, and
  no op inside a branch body may fire before its gating compare retires; nested
  `if`s serialize cumulatively. No `mux`/`select`/AND-enable op is credited, even
  for patterns the compiler lowers to `handshake.mux`/`arith.select`.
- **Scalar load/store.** A source scalar is **memory-backed** — charging a
  1-cycle `L` per named read and a 1-cycle `S` per named write — if it has more
  than one assignment site, carries state across iterations, or aliases an
  array/output. Within one iteration, a memory-backed scalar read several times
  with no intervening write is loaded **once** and fanned out. A scalar assigned
  exactly once and not loop-carried is **anonymous dataflow**: free fan-out from
  its defining op, with no `L`/`S` charged.

This spec adds a **resource model** (how many of each op can issue per cycle) and
a **schedule model** (a deterministic policy for which ready op issues when) on
top of those op counts and dependencies. It changes none of the op-counting
conventions.

## Resource Model

A CGRA in this model has three **separate** resource classes, with no shared or
bidirectional memory port:

- `P` — arithmetic processing elements; one arithmetic op per PE per cycle.
- `L` — load-issue lanes; one load per lane per cycle.
- `S` — store-issue lanes; one store per lane per cycle.

A resource *configuration* is the triple `(P, L, S)` of positive integers. The
canonical example used in the eval files is **`6x6`**: `P = 36`, `L = 12`,
`S = 12`. A configuration in which any of `P`, `L`, `S` is `0` is invalid and
**MUST** be rejected with a clear error (it cannot drain its class and would
otherwise loop forever).

For analytical helper input, the grid shorthand `AxB` **MUST** map to
`P = A * B` and `L = S = A + B`. Thus the standard DSE configurations are
`4x4 = (16, 8, 8)`, `6x6 = (36, 12, 12)`, and `8x8 = (64, 16, 16)`. This is a
resource-model convention for these estimates, not a claim that every physical
ADG with the same PE dimensions exposes that number of load or store ports.

### Op-class mapping (MUST match the aggregate model exactly)

Every counted dynamic operation is assigned **exactly one** resource class:

| Counted op | Class |
|------------|-------|
| load (array, kernel-input boundary, scalar, induction-variable read) | `L` |
| store (array, kernel-output boundary, scalar, induction-variable write) | `S` |
| add, sub | `P` |
| `address_add` (add/sub inside a subscript expression) | `P` |
| multiply, divide | `P` |
| compare (loop-bound check, conditional test) | `P` |
| bitop, shift | `P` |
| transcendental (`sqrt`, `exp`, `cos`, `sin`, `log`, …) | `P` |

Consequences that an implementation **MUST** honor:

- `address_adds` are **`P`** work, not load or store traffic.
- Induction-variable and memory-backed-scalar loads/stores are **`L`/`S`** work,
  never free.
- Every op has latency **1** in this version of the model.

Define, over a given op set:

- `A` = number of `P`-class ops,
- `LD` = number of `L`-class ops,
- `ST` = number of `S`-class ops.

These `A`/`LD`/`ST` totals **MUST** equal the eval's arithmetic/load/store op
counts for the same kernel and sizes. (`A` is the sum of `adds`, `address_adds`,
multiplies, divides, compares, bitops, shifts, and transcendentals.)

## Operation DAG

The unit of analysis is a directed acyclic graph of operation nodes.

- A **node** is one counted dynamic operation. It has:
  - a stable integer **id** equal to its construction (append) order in the
    builder — the lower the id, the earlier it was constructed;
  - a resource **class** in `{P, L, S}`;
  - unit **latency** (1 cycle);
  - a list of **predecessor** node ids (data dependencies); the node may issue
    only after all predecessors have issued in an earlier cycle;
  - an **output flag** marking it as a kernel-output-reachable terminal (used
    only for `CP`; see [Dead Work](#dead-work-and-cp-semantics)).
- An **edge** `u → v` means `v` consumes a value produced by `u`. The graph is
  acyclic: loop-carried recurrences are unrolled into a finite chain of distinct
  nodes (e.g. a reduction is an explicit balanced tree; a sequential carry of
  trip `T` is a chain of `T` updates).

Builders **MUST** construct the actual node-and-edge DAG for the fully unrolled
dynamic op set. A builder that returns only op-count totals, with no constructed
DAG and no declared region/barrier contract, does **not** satisfy this spec.

**Constant-initialized carries.** When a sequential carry (a loop-carried
scalar, accumulator, or iterator) is initialized to a **compile-time constant**,
the carry's **first** read consumes that constant, not a computed value. Per the
adopted ASAP constant / loop-invariant rule, a constant is available at cycle 1,
so the first read is modeled as a **root** (no predecessor edge from the
initializing store); the initializing store is still counted as its own op. Only
the **second and later** reads carry a real read-after-write edge from the prior
iteration's write. This matches the baseline's documented critical paths — e.g.
the FFT per-block twiddle is initialized to `w = (1.0f, 0.0f)` and the iterator
to `j = 0`, so `w^(0)` and `j^(0)` are ready at cycle 1, giving the per-stage
`CP` of `8/11/17/33` and the phase sum `74`. Adding an init→first-read edge would
inflate every carried chain by one cycle and is therefore non-conforming.

**Fully-unrolled iterators are per-lane constants.** When a **parallel** or
**reduction** dimension is fully unrolled, its iterator takes a distinct
**compile-time constant** value in each unrolled lane/leaf. Per the constant /
loop-invariant rule, that value is available at cycle 1, so:

- Address or compare arithmetic whose only varying inputs are such iterators
  (e.g. `&x[i + lag]` where both `i` and `lag` are fully-unrolled iterators) is
  modeled as a **root** (depth 1) — it does **not** take a predecessor edge from
  the iterator's modeled induction read.
- The per-iteration induction work (the iterator load, increment, write-back, and
  bound compare counted under the op-count conventions) is **counted overhead**
  that, for a parallel/reduction dim, lies **off** the output-reachable critical
  path (it is dead with respect to `CP`, while still scheduled and counted).

This is what makes `autocorrelation`'s `CP = 11` (`1` address-add + `1` load +
`1` mul + `ceil(log2 N)` reduce + `1` store): the `i + lag` address-add is a
cycle-1 root. Feeding the induction reads into the address would push it to
depth 2 and the kernel to `CP = 12`, contradicting the committed eval and the
golden anchor, and is therefore non-conforming.

The contrast with a **sequential** carry is deliberate: a sequential-dim iterator
(e.g. the FFT butterfly `j`, whose loop carries the twiddle recurrence) is **not**
a per-lane constant — its read is part of the carried chain, so the data-operand
address **does** depend on the iterator read (giving the FFT index path its
`3p + 4` depth). Builders therefore wire the induction read into the address for
sequential dims and leave it rooted for fully-unrolled parallel/reduction dims.

### Depth and height

- **Depth** of a node is the length, in nodes, of the longest dependency chain
  ending at it: `depth(n) = 1 + max(depth(p) for p in preds(n))`, or `1` if it
  has no predecessors.
- **Height** of a node is the length, in nodes, of the longest dependency chain
  starting at it and running forward to a region sink (a node with no successors
  inside its region): `height(n) = 1 + max(height(s) for s in succs(n))`, or `1`
  if it has no successors. Conceptually every region has a single virtual sink
  that all terminal nodes feed, so height is well defined for every node,
  including dead or disconnected nodes. Height is used **only** as the scheduler
  priority; it does not enter any reported cycle count.

## Regions and Phase Composition

A kernel's DAG is partitioned into an **ordered list of regions**. A region is a
set of nodes that are scheduled **together** (they may overlap freely). Regions
are the unit of barrier composition:

- **Ordered (summed) regions.** When a later region has a **true read-after-write
  barrier** on an earlier region — a value the later region *reads* from in-place
  memory that the earlier region *wrote* — the two regions are barrier-ordered.
  Barrier-ordered regions run in sequence; their cycle counts are **summed**.
  The canonical example is `fft_butterfly`: stage `s+1` reads `output_*[]`
  elements that stage `s` overwrote in place, so `copy → s=1 → s=2 → s=3 → s=4`
  are five summed regions.
- **Overlapping (single-region) work.** Independent work, and **dead
  write-after-write** work (a value written and then overwritten with **no
  intervening read**, e.g. the `conv2d` zero-fill), is **not** a barrier. Such
  work **MUST** stay in **one** region so the scheduler can overlap it. A
  dead-WAW ordering is cheap and is captured by the single region's own resource
  pressure; it **MUST NOT** be modeled as a separate summed region.

The decision of where region boundaries fall is a **modeling choice made by the
builder**, declared in its builder contract, and **MUST** be justified by a true
RAW barrier through in-place memory. Implementations **MUST NOT**:

- sum phases that are only dead-WAW ordered (this would over-count, e.g. it would
  inflate `conv2d`); or
- collapse true barrier-ordered phases into a single kernel-wide `max` (this
  would under-count, e.g. it would collapse `fft_butterfly` to the global ASAP
  value `71` instead of the phase sum `74`).

### Dead work and CP semantics

- A node is **output-reachable** if a directed path leads from it to an
  output-flagged node. All other nodes are **dead** (or disconnected).
- `CP` for a region is the maximum **depth** over its output-reachable nodes
  (equivalently, the depth of its deepest output node). `CP` is computed over
  output-reachable ops **only**.
- Dead work is still **counted** (it contributes to `A`/`LD`/`ST`) and is still
  **scheduled** (it consumes `P`/`L`/`S`), but it is **excluded from `CP`**, in
  keeping with the ASAP rule that a dead op can never lie on the critical path to
  an output.
- If a **non-empty** region has **no** output-reachable nodes (it contains only
  dead/disconnected work), then `CP = 0`: its aggregate is resource-only
  (`max(0, compute, load, store)`), and the scheduler still issues all of its
  nodes (so its makespan is at least `1`).

## Metric 1: Aggregate CGRA Lower Bound

For a single region with counts `A`/`LD`/`ST`, critical path `CP`, and
configuration `(P, L, S)`:

```
compute   = ceil(A  / P)
load      = ceil(LD / L)
store     = ceil(ST / S)
aggregate = max(CP, compute, load, store)
```

For a kernel partitioned into the ordered region list `R1, R2, …, Rk`:

```
aggregate_cycles = Σ_i aggregate(R_i)
```

(For a single-region kernel the sum is just that one region's aggregate.)

This is a genuine **lower bound** for the resource model, resting on two
independent facts that hold for *any* schedule on `(P, L, S)`:

- **Outputs cannot be ready before `CP`.** Each output node sits at the end of a
  dependency chain of `CP` unit-latency ops, so no output can retire earlier than
  cycle `CP`. (Dead work is excluded from `CP`; this clause bounds output
  readiness only.)
- **No class can drain faster than its ceiling.** A region has `A`/`LD`/`ST`
  counted ops — including dead work, which must still issue — and at most `P`/`L`/`S`
  of each class issue per cycle, so the region needs at least
  `compute`/`load`/`store` cycles to issue every op of each class.

Taking the `max` of these gives the per-region bound; barrier-ordered regions
cannot overlap, so their bounds sum. The aggregate bound is preserved unchanged
by this spec; the finite-resource estimate below sits on top of it.

An empty region (no nodes) has `A = LD = ST = 0`, `CP = 0`, and
`aggregate = 0`.

## Metric 2: Finite-Resource Schedule Estimate

### Deterministic criticality-priority list scheduler (per region)

A region is scheduled independently, cycle by cycle, starting at cycle 1:

1. A node is **ready** at cycle `t` if every predecessor was **issued at a cycle
   ≤ t − 1**. (With unit latency, a node issued at cycle `c` is available to
   successors at cycle `c + 1`.) A node with no predecessors is ready at cycle 1.
2. Partition the currently-ready, not-yet-issued nodes by class.
3. For each class in the fixed order `(P, L, S)`, issue ready nodes of that class
   **highest priority first**, up to the class capacity (`P`, `L`, or `S`) for
   this cycle. Priority is the node **height** (larger height first); ties are
   broken by **ascending stable node id**. Concretely, nodes are ordered by the
   key `(-height, +id)`.
4. Advance to the next cycle and repeat until **all** counted nodes in the
   region have been issued — **including dead and disconnected nodes**.

The region **makespan** is the last cycle at which any node is issued. Scheduling
**MUST** continue until every counted op has issued; a schedule that stops before
dead/disconnected ops are issued is non-conforming.

The scheduler **MUST** be deterministic: two runs on the same DAG and
configuration **MUST** produce byte-identical output. Determinism requires the
stable-node-id tie-break and **MUST NOT** depend on `dict`/`set` iteration order,
hashing, wall-clock time, or randomness.

### Scheduled cycles and gap metrics

For the ordered region list `R1, …, Rk`:

```
scheduled_cycles = Σ_i makespan(R_i)
gap_cycles       = scheduled_cycles - aggregate_cycles
gap_ratio        = scheduled_cycles / aggregate_cycles
```

- `gap_ratio` is **always** the ratio `scheduled / aggregate` (not
  `(scheduled − aggregate) / aggregate`), and **MUST** be labeled
  `scheduled / aggregate` wherever it is reported.
- When `aggregate_cycles == 0` (an empty region or empty kernel), all of
  `aggregate_cycles`, `scheduled_cycles`, and `gap_cycles` are `0`, and
  `gap_ratio` is reported as `1.0`. Implementations **MUST NOT** divide by zero.

### Conditional invariant

The invariant

```
scheduled_cycles >= aggregate_cycles
```

holds — and **MUST** be asserted in tests — **whenever** both metrics use the
**same** region partition, the **same** counted-op set, the **same** `(P, L, S)`
capacities, and **all** counted ops must issue. It holds per region and after
summing the ordered regions. It is **conditional** on that matched setup; it is
not claimed to hold across mismatched partitions or op sets. A reported
`scheduled_cycles < aggregate_cycles` under a matched setup indicates a scheduler
or invariant bug.

### Local pressure summary

To expose *where* the local bursts occur, the schedule reports, per class
`c ∈ {P, L, S}` (aggregated across the kernel's regions):

- `saturated_cycles` — the number of cycles in which the number of class-`c` ops
  issued equals the class capacity;
- `longest_run` — the longest run of consecutive saturated cycles for class `c`;
- `peak_ready_backlog` — the maximum over cycles of
  `max(0, (ready class-c ops) − capacity)`, i.e. the largest number of ready
  class-`c` ops that could not issue in a single cycle because the class was
  saturated.

When a kernel has multiple regions, these are composed across regions as follows.
Region boundaries **break** saturation runs (regions are scheduled independently,
so a run does not carry across a barrier). `saturated_cycles` is the **sum** over
regions of each region's saturated-cycle count; `longest_run` is the **max** over
regions of each region's longest within-region run; `peak_ready_backlog` is the
**max** over regions of each region's peak backlog.

These are descriptive diagnostics, not bounds.

### What the estimate is and is not

The finite-resource schedule estimate is a **reproducible estimate** for the
scheduling policy defined above. It is meaningfully tighter than the aggregate
bound when local resource pressure exceeds the smoothed average. It is:

- **NOT a lower bound.** It depends on a specific (greedy, height-priority)
  scheduling policy; a different policy could do better or worse. Only the
  aggregate metric is the lower bound. The phrase "lower bound" **MUST NOT** be
  applied to the schedule estimate anywhere.
- **NOT cycle-accurate RTL.** It makes no claim about a specific hardware
  implementation.
- **Free of place-and-route claims.** It models only per-cycle issue capacity per
  class. It **MUST NOT** be described as modeling mapper, placement, routing,
  or memory-bank-conflict effects.

## Optional Loom-Pragma Design-Space Estimate

The aggregate CGRA lower bound and the finite-resource schedule estimate above
operate on the fully-unrolled ASAP dynamic operation set. A separate exploratory
model **MAY** be used to compare explicit Loom loop-pragmas before committing to
compiler or hardware mapping work. This optional model is called the
**Loom-pragma design-space estimate**.

### Analytical scope and target profiles

The reference helper is branch-local analytical evidence with evidence kind
`analytic_prefilter`. It **MAY** explore loop transformations and target features
that this branch's compiler, mapper, or hardware does not materialize. Every such
assumption **MUST** belong to a named analytical target profile and **MUST** be
printed in the report; an analytical profile is not evidence that the named
hardware exists.

The canonical extended-study profile is `shared-spad-4k-r1w1-v4`. It fixes one
4096-byte scratchpad shared by all workers executing one mapped kernel, one
logical scratchpad load port, one logical scratchpad store port, one-cycle
modeled scratchpad access, and vector width `V = 4` source elements. Capacity,
load-port count, store-port count, and access latency are hardware parameters of
the analytical target and **MAY** be overridden explicitly. Profile identity
includes all of those values. The default does not assert a bank count, bank
mapping, DMA engine, arbitration behavior, or particular hardware topology.

**Objective.** Choose the pragma that minimizes modeled cycles subject to the hard
`≤ L` load-lane / `≤ S` store-lane per-cycle limit (`L = S = 12` for `6x6`).
Within one transformation family, the candidate is the lane-saturation knee —
the smallest mature exposure whose traffic saturates the binding resource.
Extended search compares those family knees by modeled cycles. Extended
target-profile candidates additionally obey the profile's whole-working-set
capacity and scratchpad-port constraints.
Cycle count credits vector coalescing, control-overhead amortization, and any
explicitly declared transformation or resident-memory effect. There is **no area
term** and no control/body area tradeoff. The legacy direct-memory path has only
the machine lane caps; a named extended profile may add analytical capacity and
scratchpad-port constraints without claiming implemented hardware.

**Notation (this section).** To keep the pragma factors distinct from the machine
resource configuration, this section writes the machine arithmetic-PE, load-lane,
and store-lane counts as `P_pe`, `L`, and `S` (the Resource Model triple; for
`6x6`, `P_pe = 36`, `L = S = 12`). The pragma factors are written `p` (one
`LOOM_PARALLEL` factor), `P_tot` (the product of `p` over parallelizable levels),
and `U` (a `LOOM_UNROLL` factor), with `U_tot` the product of `U` over all levels;
candidate shorthand `PaUb` means `p = a`, `U = b`. `V` is the modeled vector
width in source elements. The current helper fixes `V = 4`; in the existing
64-bit examples this is one 256-bit vector memory operation. Buffer-capacity
accounting uses each buffer's actual source element size.

A DSE report **MAY** render one configuration in full and append terse split
recommendations for additional configurations. Each terse recommendation
**MUST** use the same loop legality, complete candidate enumeration, and
recommendation policy as a full report for that configuration; it **MUST NOT**
be scaled or inferred from the detailed configuration's split. Alternate lines
compare pragma choices within each named configuration only. They do not rank,
select, or recommend a CGRA size.

The Loom-pragma design-space estimate consumes source-loop metadata such as loop
kind, trip count, `LOOM_PARALLEL(P)`, `LOOM_UNROLL(U)`, and schedule strategy.
Loop kind controls legality: dependency-parallel loops may use parallel workers,
sequential loops must preserve carried recurrences, and reductions may use
parallel workers only when the carried operation is treated as a reduction. The
`LOOM_PARALLEL(P)` and `LOOM_UNROLL(U)` values control finite exposure: a single
candidate chunk contains only the iterations exposed by the chosen workers and
unroll factor, rather than the whole fully-unrolled loop nest.

### Searched loop order and fixed vectorization scope

An extended kernel **MAY** search loop interchange in addition to the source
order. Interchange means selecting a different nesting order for existing loop
levels without changing the kernel's mathematical result. Legality **MUST** be
declared per kernel as an explicit, auditable set of permitted level orders; the
estimator **MUST NOT** infer interchange legality only from observed address
patterns. The source order is one permitted order. After selecting an order, the
estimator **MUST** recompute each access's innermost varying dimension and actual
contiguity. Fixed-width coalescing therefore follows the selected order and
address function, not the source-text order.

Vector width remains fixed at `V = 4` for all kernels in this model. A contiguous
group uses the existing `ceil(group_elems / V)` rule, and a partial final group of
one to three elements still uses one vector memory node under the current
zero-overhead pack/unpack convention. Vector width is not candidate state and is
not a searched axis. A future extension **MAY** search width only after it defines
candidate-dependent width costs or legality, such as masked-lane or pack/unpack
overhead, alignment restrictions, or bank conflicts whose demand changes with
width. A separately named target profile may fix a different width for a
sensitivity study, but one DSE run **MUST NOT** enumerate widths as candidates
under the present cost model.

### Explicit unroll-and-jam choices

Unroll-and-jam means unrolling an outer loop by `U`, conceptually replicating its
enclosed inner loops `U` times, and fusing ("jamming") those copies so the `U`
outer iterations advance in lockstep at each inner step. Outer-loop unrolling
does **not** imply jam. An extended candidate selects one explicitly declared
complete jam plan, and `none` is always a legal plan. A nonempty jam plan is legal
only when every named outer level is a dependency-parallel DSE level with
`U > 1` and every named inner level occurs beneath its outer level in the
selected loop order. Jam is not an additional exposure factor; total exposure
remains the product of `p * U` over parallel levels.

Jam legality and load sharing **MUST** be declared as explicit per-kernel
metadata. Each declaration names a complete plan and its permitted
outer-to-inner edges; an edge names any operands that may be shared. Candidate
enumeration uses only those complete plans. It **MUST NOT** infer a legal edge or
shared operand from address equality or access patterns, and it **MUST NOT**
enumerate an arbitrary power set of independently declared edges.

Until a separate DFG-size or routing cost is specified, jam adds no modeled
arithmetic, control, routing, or area penalty. Its only credited effect is
eliminating redundant loads of operands invariant in the jammed outer dimension;
load-node and total-node counts may therefore decrease. In GEMV, a candidate may
select `i-j-share-x`: simultaneous unrolled rows share one `x[j]` load at each
`j` step, while each row retains its own private reduction accumulator. The same
unroll split with `jam=none` does not receive that sharing credit. Reports
**MUST** identify the selected jam plan and its shared operands.

For a single dependency-parallel loop, an implementation of this optional model
SHOULD build one candidate chunk with:

```
exposed_iters = min(trip_count, P_tot * U)
waves         = ceil(trip_count / exposed_iters)
```

The implementation then schedules that chunk with the deterministic
finite-resource scheduler defined above and reports an estimated total such as:

```
estimated_cycles = chunk_scheduled_cycles * waves
```

More detailed implementations MAY model a smaller tail chunk separately, nested
loop placement, reduction merge trees, and schedule strategy effects such as
contiguous versus interleaved distribution. Any such report **MUST** state the
assumptions it uses.

### Wave serialization and what the bracket means

The `estimated_cycles = chunk_scheduled_cycles * waves` form above models
**wave-serialized** execution: each wave's chunk is scheduled in isolation and the
makespans are summed, as if no wave begins before the previous wave fully drains.
Real Loom dataflow maps the DAG spatially and **pipelines** successive waves, so
this form is a conservative over-estimate that falls **monotonically** as exposure
grows. Used directly as the objective, it therefore always selects the maximum
`P_tot · U`, which is **not** a meaningful design recommendation (see exposure
selection below).

Legacy direct-memory candidates **MAY** report three quantities. With exposure
`E = min(trip_count, P_tot · U)`, `full_waves = trip_count // E`, and
`tail = trip_count % E`, define a wave-summed aggregate and a wave-summed schedule
estimate:

```
pragma_exposure_aggregate = full_waves * chunk_aggregate(E) + (chunk_aggregate(tail) if tail else 0)
schedule_estimate         = full_waves * chunk_scheduled(E) + (chunk_scheduled(tail) if tail else 0)
```

where `chunk_aggregate(E)` is the aggregate bound (Metric-1 form, with the
lane-aware effective terms defined in the lane-aware subsection below) of the
`E`-iteration chunk and `chunk_scheduled(E)` is its finite-resource schedule
makespan (Metric 2). These relate to the genuine resource floor by the legacy
bracket

```
absolute_cgra_lb  <=  pragma_exposure_aggregate  <=  schedule_estimate
```

An extended transformed candidate **MUST** additionally report its
`plan_cgra_lb`, the candidate-specific full-exposure aggregate defined in the
scratchpad-composition subsection below. For every legal extended candidate, and
for every candidate whose finite-resource schedule is materialized, the
four-term bracket is

```
absolute_cgra_lb <= plan_cgra_lb <= pragma_exposure_aggregate <= schedule_estimate
```

There is one report-global `absolute_cgra_lb` for each exact
kernel/configuration/target-profile identity, including capacity and port counts.
For an extended profile it is `min(plan_cgra_lb)` over all legal transformed
candidates in that profile. A candidate-specific `plan_cgra_lb` **MUST NOT**
replace that global value under the name `absolute_cgra_lb`. Unmodified kernels
retain the legacy computation below exactly.

On the legacy path, `absolute_cgra_lb` is the aggregate CGRA lower bound (Metric
1) evaluated
over the kernel's **full unrolled** op set, independent of `p` and `U`. When the
DSE credits vector coalescing (see the lane-aware subsection below), the
memory terms of `absolute_cgra_lb` are computed over the **fully-coalesced**
lane-slot counts of the full-unroll op set — full unrolling exposes the maximal
contiguous run, so it coalesces maximally — divided by the full machine lanes
`L`/`S`:

```
absolute_cgra_lb = max(CP, ceil(A / P_pe), ceil(LD_rec_full / L), ceil(ST_vec_full / S))
```

`LD_rec_full` is the full-unroll **recurring** vector load lane-slot count
(per-iteration array loads + the residual iterator); `ST_vec_full` is the
full-unroll vector store lane-slot count. Both reduce to the scalar `LD`/`ST` when
no coalescing is legal. One-time **invariant** loads are amortized (loaded once and
held) and excluded from this floor; the reported total traffic is
`LD_eff_full = LD_rec_full + LD_inv`. This is the genuine resource floor for the
vector-capable target the candidates run on. Here `A` is the full-unroll op count,
whose control is amortized to a single residual iterator. The
left inequality holds because (i) any partial-exposure wave coalesces contiguous
groups no larger than the full run, so its per-wave vector lane-slots sum to at
least `LD_vec_full`/`ST_vec_full`; (ii) `active_L ≤ L`, `active_S ≤ S`; and (iii)
any finite candidate carries the same algorithmic arithmetic **plus** its own
un-amortized per-worker control (`P_tot · waves ≥ 1` iterator sets ≥ the floor's
single residual), so its `A` and induction loads/stores are no smaller than the
floor's. Combined with `ceil` subadditivity across the wave partition and an
unchanged `CP`, every candidate therefore sits at or above `absolute_cgra_lb`. The right inequality is
the per-region conditional invariant of Metric 2 summed over waves.

Because it credits both vector coalescing **and** control-overhead amortization
(the full-unroll floor carries a single residual iterator, not `T` of them; see
below), this DSE `absolute_cgra_lb` can be well **below** the scalar Metric-1
aggregate reported in the kernel's main `## CGRA-Constrained Model` section (which
models neither vector memory ops nor control amortization, charging induction per
iteration per the ASAP baseline). That gap is expected and often large: for
memory-light kernels the induction stream dominates the scalar aggregate, and full
unrolling amortizes it away. The two coincide only when no coalescing applies and
the loop is already fully unrolled.

**Only the `*_cgra_lb` quantities are resource floors.**
`absolute_cgra_lb` is the sole report-global lower bound. For an extended
candidate, `plan_cgra_lb` is a candidate-specific transformed-plan floor and is
not the global floor. `pragma_exposure_aggregate` and `schedule_estimate` embed
the wave-serialization assumption and therefore sit above the corresponding
floor; real pipelined execution can fall below them. The phrase "lower bound"
**MUST NOT** be applied to `pragma_exposure_aggregate` or `schedule_estimate`.
In particular, the ratio
`pragma_exposure_aggregate / absolute_cgra_lb` measures the
**wave-serialization penalty of the chosen exposure** and **MUST NOT** be
described as a pure "finite-exposure penalty," since it is dominated by the
model's non-overlap assumption rather than a hardware cost.

### Steady-state saturation and exposure selection

To *select* an exposure, an implementation **MUST NOT** minimize the wave-summed
estimate directly (it is monotone in exposure) and **MUST NOT** constrain on zero
scheduler backlog (see below). It instead reasons about **steady-state resource
saturation**.

For a single dependency-parallel loop of trip count `T`, define the
**per-iteration algorithmic class demand** — the `P`/`L`/`S` op counts charged by
the kernel's intended math in one iteration, with loop-invariant values hoisted
and charged once for the whole loop, and **excluding loop control** (the iterator
load/add/store/compare), which is amortized separately below. The load/store
demands are **effective lane-slot** demands: counted as vector lane-slots after
any legal coalescing (a contiguous group of `V` same-array elements amortizes to
`1/V` lane-slots per iteration), and as scalar lane-slots otherwise:

```
a_iter, ld_iter, st_iter          # algorithmic only; control amortized separately
```

**Control amortization.** Within an exposed wave every iteration is spatial
(unrolled), so the surviving loop control is one iterator advance per worker per
wave: induction is charged `P_tot` times per chunk, hence `P_tot · waves ≈ T / U_tot`
times over the whole loop (`U_tot` = product of unroll factors). At **full unroll**
(`U_tot = T`) only a single residual iterator remains, so the resource floor uses
the algorithmic demand alone:

```
absolute_cgra_lb = max(CP, ceil(T * a_iter / P_pe), ceil(T * ld_iter / L), ceil(T * st_iter / S))
```

which is the same `absolute_cgra_lb` as in the bracket above (`T * ld_iter`
= `LD_rec_full` since invariants are hoisted out of `ld_iter`, `T * st_iter`
= `ST_vec_full`, control ≈ 0). This is precisely why the DSE floor can drop far
below the per-iteration-induction scalar aggregate.

The **binding class** is the class achieving that `max` (excluding `CP`); let
`count_binding` be its effective per-iteration demand and `cap_binding` its
**machine** capacity (`P_pe`, `L`, or `S`). A single wave's chunk has effective
class totals `A_eff ≈ E · a_iter + P_tot · a_ctrl`,
`LD_rec ≈ E · ld_iter + P_tot · ld_ctrl` recurring loads, `ST_eff ≈ E · st_iter +
P_tot · st_ctrl` (the control term is the un-amortized `P_tot` per-worker iterators
of *this* wave). The `O(1)` per-wave invariant re-loads form `LD_inv`; they are
amortized out of the binding term and reported only in `LD_eff = LD_rec + LD_inv`.
Scheduled under the chunk's effective lane widths, the aggregate is the same
`chunk_aggregate` defined in the lane-aware subsection. Note the control term
scales with `P_tot`, not `E` — the lever that makes a parallel-heavy split pay more
control than an unroll-heavy one at equal exposure:

```
chunk_aggregate(E) = max(CP, ceil(A_eff / P_pe), ceil(LD_rec / L), ceil(ST_eff / active_S))
```

For the saturation analysis below, take the exposure large enough that the
binding class's effective width has reached its machine cap
(`active_binding = cap_binding`); this is the regime in which added exposure stops
widening issue and starts stacking work. A wave is then **latency-bound** while
`chunk_aggregate(E) == CP` — the binding class idles for
`CP − ceil(E * count_binding / cap_binding)` cycles of every wave — and
**resource-bound** once the binding class's per-wave ceiling reaches `CP`. The
**saturation exposure** `E_sat` is the smallest exposure at which the binding
class becomes resource-bound:

```
E_sat = smallest E such that ceil(E * count_binding / cap_binding) >= CP
```

Physically, `E_sat` is the exposure at which the binding resource's
initiation-interval pressure first fills its issue width every cycle. Below
`E_sat`, adding parallelism strictly improves throughput (the binding resource is
idle part of each wave). At and above `E_sat`, each wave is resource-bound; under
pipelined execution the steady-state rate equals the resource floor, and further
exposure yields **no modeled steady-state throughput gain** — the wave-summed
aggregate creeps toward `absolute_cgra_lb` only through per-wave ceiling rounding
and invariant-reload amortization, while the instantaneous (transient) ready
backlog rises linearly. `E_sat` is therefore the diminishing-returns **knee**.

Extended candidates need one additional guard because interchange, an explicitly
selected jam plan, and resident-memory fan-out can change the recurring operation
set as exposure grows. A small wave may already be resource-bound while the same
transformation still eliminates repeated loads at a larger exposure. Such a row
is **recurring-demand immature**, not the saturation knee.

For each transformation family and exposure, the extended search **MUST** first
retain the candidates with the minimum recurring-compute aggregate at that
exposure, excluding every preload prologue. A family is the resolved loop order,
explicit jam plan, and actual placement identity. For each retained candidate
the search records full-kernel recurring demand `D[P/L/S/R/W]`: arithmetic,
effective data-load lane slots, effective data-store lane slots,
scratchpad-read port cycles, and scratchpad-write port cycles. These demands
exclude preload, invariant loads, iterator/control work, and per-wave ceiling
rounding. Looking from the current exposure through every larger legal exposure
of the same family, define the future minimum of each demand class over those
recurring-compute-frontier candidates.

An extended candidate is knee-eligible only when its nominal full, non-tail
compute wave is resource-bound and at least one class tied for that wave's
dominant resource term has already reached its future-minimum full-kernel demand.
Within each family, the first exposure with an eligible candidate is that
family's knee; exact frontier ties at that exposure are ranked by total
`pragma_exposure_aggregate`, then by recurring data traffic, scratchpad-port
demand, worker count, and deterministic signature. The global recommendation is
the family knee with the smallest `pragma_exposure_aggregate`, then the smallest
recurring data traffic, scratchpad-port demand, exposure, worker count, and
deterministic signature. Jammed and unjammed candidates are distinct
transformations rather than future states of one another. If no family has an
eligible knee, select the global best-estimate fallback. The legacy direct-memory
selection rule is unchanged.

Within a transformation family, an implementation that selects an exposure
**SHOULD** recommend the smallest legal exposure `E >= E_sat` — the smallest
`P_tot · U` consistent with pragma legality and loop nesting. Rows below or
above the knee are classified relative to their own family knee; the globally
selected family knee alone receives the recommendation marker. Larger exposures
are **oversubscribed**: diminishing aggregate gains traded against linearly
growing transient backlog and hardware area.

### Backlog is a diagnostic, not a constraint

The `peak_ready_backlog` reported by the finite-resource scheduler is the largest
number of ready ops of a class that could not issue in one cycle. In the chunk
model it is a **transient artifact** of fully unrolling `E` iterations and
releasing all their independent roots at cycle 1; it is **not** a steady-state
hardware property, because pipelined dataflow throttles iteration entry by the
initiation interval rather than releasing a whole chunk at once.

An implementation **MUST NOT** use `peak_ready_backlog == 0` as a feasibility
constraint when selecting an exposure. Because the wave-summed estimate falls
monotonically with exposure while backlog rises monotonically with exposure, a
zero-backlog constraint selects the **smallest** exposures and therefore the
**worst** throughput — the opposite of the design goal. Backlog **SHOULD** be
reported as a pressure diagnostic only and read together with `E_sat`: backlog
that appears only beyond `E_sat` signals oversubscription, not infeasibility.

### Steady-state resource utilization (preferred pressure diagnostic)

Because `peak_ready_backlog` is a transient cycle-1 release artifact (above), the
**preferred** steady-state pressure diagnostic is **per-class utilization**: the
fraction of a wave's makespan during which a class would be busy if its work were
spread evenly across the wave. For an exposed chunk with the effective class terms
`compute = ceil(A_eff/P_pe)`, `load = ceil(LD_eff/active_L)`,
`store = ceil(ST_eff/active_S)` and aggregate `agg = max(CP, compute, load, store)`
(the same terms as `chunk_aggregate`):

```
util_P = compute / agg
util_L = load    / agg
util_S = store   / agg
```

Each `util_c` lies in `(0, 1]`. Properties an implementation **MUST** preserve:

- The binding class reads `util = 1.0` **exactly when** the wave is
  **resource-bound** (`agg` is set by a resource term, i.e. `exposed >= E_sat`).
- When the wave is **latency-bound** (`agg = CP`, `exposed < E_sat`), **every**
  `util_c < 1.0` — correctly showing the resource classes idle while the critical
  path drains.

This is the honest backpressure proxy: it reports *which* class saturates and
*how much headroom* the others have, in a way that does not depend on the cycle-1
release. A pipelined dataflow execution that sustains the binding-class rate would
exhibit ~100% utilization on that class with no growing queue — which is what
`util` reports and what `peak_ready_backlog` misrepresents as a spike.

Reports **SHOULD** present `util` as the primary pressure signal.
`peak_ready_backlog` **MAY** still be reported, but **MUST** be labeled a
transient list-schedule artifact and **MUST NOT** be presented as a steady-state
quantity or a hardware queue depth.

### Lane-aware P/U and vector coalescing (load/store axis)

`LOOM_PARALLEL` and `LOOM_UNROLL` are physically distinct (see
[`docs/spec-pragma.md`](./spec-pragma.md)): a `LOOM_PARALLEL` factor `p` maps to
separate worker groups over data partitions, `U` enlarges one worker's dataflow
graph. The load/store-focused DSE **MUST NOT** assume that unrolled memory
operations serialize through one lane merely because they live in one worker. If
the unrolled loop bodies are independent, their memory operations are independent
DAG nodes and may issue to different load/store lanes in the same cycle, subject
to memory independence, vector-interface legality, and the machine-wide `L`/`S`
lane counts. The legacy direct-memory path ignores explicit `LOOM_MEMORY_BANK`
interactions and address-level external-memory bank conflicts. A named analytical
target profile may instead declare scratchpad capacity and logical port counts;
those parameters are not inferred from a source pragma.

This DSE models the two axes on which `LOOM_PARALLEL` and `LOOM_UNROLL` physically
diverge; it still **MUST NOT** claim to model place-and-route or cycle-accurate
RTL. The *algorithmic* arithmetic pool and the critical path stay **global**:
`compute` and `CP` for the kernel's intended math do not distinguish whether
exposure came from `p` or `U`. The two pragmas separate on exactly these axes:

1. **Vector coalescing** (load/store axis, detailed below): a worker's `U`
   adjacent same-array accesses fuse into `ceil(U_mem/V)` vector lane-slots, while
   `LOOM_PARALLEL` workers stride across data partitions and do not coalesce.
   Bounded by `V`; the credit is gone once `U ≥ V`.
2. **Control-overhead amortization** (all pools): within an exposed wave every
   iteration is laid out spatially, so the only surviving loop control is one
   iterator advance per worker per wave. Induction (iterator load / add / store /
   compare) is therefore charged `P_tot` times per chunk — **once per worker** —
   so the total control op count over the whole loop scales as `trip / U_tot`.
   `LOOM_UNROLL` amortizes control (`/U`); `LOOM_PARALLEL` keeps one iterator per
   worker and does not. A fully-consumed reduction is a spatial (tree-reduced)
   dataflow graph and carries no loop control for any split. The **sequential**
   carried recurrence is the exception: it cannot be spatially flattened, so its
   iterator is charged per iteration and lies on the critical path
   (`tridiag_solve`, `trsv_lower/upper`, `gauss_seidel_step`, `kmp_table`).

Both effects only *reduce op counts* inside the existing `P`/`L`/`S` pools —
there is **no separate control resource, no capacity knob, and no area term**.
Both bias the estimate **toward `LOOM_UNROLL`** for contiguous parallel loops,
which is the mentor-confirmed direction. A named scratchpad profile may add the
DSE-local port correction defined below, but it does not add a resource class or
an area term to the finite-resource scheduler.

This choice is DSE-local and deliberately more optimistic than the ASAP baseline
(`## Adopted Baseline`), which charges induction per iteration even under full
unrolling as a conservative floor. The DSE credits the spatial-unroll amortization
that real Loom dataflow performs; the two models therefore diverge on induction by
design.

**Scalar lane exposure.** Let `P_tot` be the total number of parallel workers
(the product of the per-level `LOOM_PARALLEL` factors over parallelizable
levels). For each load/store access group, define an eligible unroll width
`U_mem`:

- `U_mem = U` when the unrolled memory operations are independent, refer to
  distinct elements, and the target can present them concurrently to memory
  lanes. A dependency-parallel loop with `a[i] = b[i]` unrolled by 4 is the
  canonical case: `b[i]`, `b[i+1]`, `b[i+2]`, and `b[i+3]` can be issued in the
  same cycle on four load lanes if four lanes are available.
- `U_mem < U` when only some unrolled operations are independent and eligible for
  concurrent memory issue.
- `U_mem = 1` when a carried dependence, possible alias, unknown access pattern,
  or target limitation forces the accesses to serialize.

Writing `w_L`/`w_S` for the concurrent load/store **lane-slots one worker
presents** per access group, the chunk's concurrent load- and store-issue widths
are

```
active_L = min(P_tot * w_L, L)
active_S = min(P_tot * w_S, S)
```

For **scalar** (uncoalesced) independent accesses each element is its own
lane-slot, so `w_L = U_mem_L` and `w_S = U_mem_S`; the **Vector load coalescing**
paragraph below lowers `w` to the vector-load count when a group coalesces. Thus,
for scalar independent accesses, `P4U1` and `P1U4` can expose the same four load
lanes. The DSE **MUST NOT** recommend `LOOM_PARALLEL` over `LOOM_UNROLL` on
load/store-lane grounds alone when the two candidates have the same eligible
memory exposure.

**Legacy direct-memory no-banking assumption.** A
`LOOM_PARALLEL(p, contiguous)` (or `block`) level partitions work across `p`
workers, while `LOOM_UNROLL(U)` exposes adjacent iterations inside a worker.
Either form can create concurrent memory demand for dependency-independent
accesses. For the legacy path, explicit `LOOM_MEMORY_BANK(B, ...)` parameters and
address-level external-memory bank conflicts are ignored: the only caps on
eligible scalar lane exposure are the target `L` and `S` lane counts. Legacy
reports **MUST** state this assumption. A named analytical scratchpad profile
**MUST** state its logical load/store-port counts and access latency separately.

**Vector load coalescing.** A target may support a vector memory operation plus a
vector `unpack` operation. Under the current target convention, one vector
memory operation covers `V = 4` same-type source elements; in the existing
64-bit examples this is 256 bits. Vector stores and `pack` are modeled as the
inverse operation. When `U` exposes a
contiguous group of same-array, same-type element loads or stores, and the
target's alignment and vector-interface rules allow it, the DSE **MAY** coalesce
the scalar accesses into vector memory operations. For `V = 4` elements per
vector operation:

```
scalar_loads_without_vector = load_group_elems
vector_loads                = ceil(load_group_elems / V)
load_lane_slots             = vector_loads
scalar_stores_without_vector = store_group_elems
vector_stores                = ceil(store_group_elems / V)
store_lane_slots            = vector_stores
```

For example, four contiguous 64-bit loads `b[i]` through `b[i+3]` may become one
256-bit vector load. That vector load occupies one load lane for that cycle and
produces up to four scalar values after unpack. Partial vector groups also
occupy one load lane, so a group of one to three 64-bit elements still
contributes one load-lane slot if it is issued as a vector load. Store-side
vectorization follows the same rule with vector stores and `pack`: one 256-bit
vector store occupies one store lane and writes up to four scalar elements.

The finite-resource scheduler (Metric 2) **supports vector coalescing**: each
coalesced access is a single memory node of its class occupying **one** lane-slot,
and its `unpack`/`pack` is modeled as zero-cost fan-out to (from) the scalar
consumers (producers). `chunk_scheduled(E)` therefore schedules the **vectorized**
chunk DAG under `(P_pe, active_L, active_S)`, and the per-region invariant
`chunk_scheduled >= chunk_aggregate` holds over that DAG. Because `unpack`/`pack`
carry no modeled `P`-class work, this remains a **load/store-focused** estimate:
it charges no resource class for (un)packing, and — like every metric here —
**MUST NOT** be read as a place-and-route or cycle-accurate RTL result.

Because one vector memory operation occupies one lane slot while carrying up to
four 64-bit elements, vector coalescing reduces both the memory instruction /
stream-operation count and the load/store lane-slot demand for contiguous
unrolled groups. Accordingly, a coalesced group's per-worker concurrent lane
usage is its **vector-load count**, so `w_L = ceil(U_mem_L / V)` (and
`w_S = ceil(U_mem_S / V)`) for that group — this is the `w_L`/`w_S` fed into
`active_L`/`active_S` above, and the reason a coalesced group needs fewer active
lanes than its scalar element count `U_mem`.

**Recurring vs. invariant loads.** The load lane-slots of a chunk split into two
kinds, and only one of them sets the steady-state lane pressure:

- **Recurring loads** (`LD_rec`) — per-iteration array element loads over the
  exposed index plus induction reads. Their count **scales with exposure**. These
  set the steady-state load lane exposure and the binding load term.
- **Invariant loads** (`LD_inv`) — values hoisted once per chunk, their count
  **independent of exposure** (e.g. axpy's `alpha`, gemv's whole `x` vector, a
  kernel's size/param scalars, a recurrence's seed). They are **amortized**:
  loaded once and held (broadcast by free fan-out to every consumer), so they do
  **not** establish sustained per-cycle lane pressure.

The reported total traffic is `LD_eff = LD_rec + LD_inv`. The **binding load term
uses `LD_rec` only**; `LD_inv` is amortized out of it. The reported active load
lanes are the recurring lane exposure `active_L = min(LD_rec, L)` (and
`active_S = min(ST_eff, S)`, since stores in these kernels are all recurring).

**Wave formulas.** The exposed chunk (`E = Π_level p·u` iterations, reductions as
balanced merge trees, inner-invariant loads amortized per the nested-loop note
below) is lowered to effective chunk totals. `LD_rec(candidate)` /
`LD_eff(candidate)` and `ST_eff(candidate)` are load/store **lane-slot** counts
after scalar lane eligibility and optional vector coalescing have been applied;
`A_eff(candidate)` excludes unpack/pack under the provisional free-unpack
convention but still includes any modeled address-generation work. The chunk
aggregate is then

```
chunk_aggregate = max(CP,
                      ceil(A_eff(candidate) / P_pe),
                      ceil(LD_rec(candidate) / L),        # recurring loads only; invariants amortized
                      ceil(ST_eff(candidate) / active_S))
```

`P_pe` in the compute term is the machine compute-lane count (Notation above), not
a pragma factor. `ceil(LD_rec / L)` equals `ceil(LD_rec / active_L)` because
`active_L = min(LD_rec, L)`. `pragma_exposure_aggregate` / `schedule_estimate` sum
this over the waves as before; the finite-resource `schedule_estimate` still
issues each wave's `LD_inv` re-load, so it sits at or above the
invariant-amortized aggregate. Two modeled effects make `P4U1` and `P1U4` differ at equal
product, both biasing **toward** `LOOM_UNROLL` (the intended, mentor-confirmed
direction):

- **Vector coalescing.** With the provisional one-lane vector convention, an
  unroll-heavy candidate has a smaller load/store term because contiguous unrolled
  accesses reduce `LD_eff` / `ST_eff` (unrolled iterations are adjacent and
  coalesce; parallel workers stride across partitions and do not). Bounded by `V`.
- **Control amortization.** The `P_tot` per-worker iterators are charged once each
  per chunk, so a parallel-heavy split (`P4U1`, `P_tot = 4`) carries four iterator
  load/add/store/compare sets while the unroll-heavy split (`P1U4`, `P_tot = 1`)
  carries one. This shrinks `A_eff` / `LD_eff` / `ST_eff` for unroll and, unlike
  coalescing, keeps paying past `U = V`. It even separates the two where coalescing
  cannot (strided accesses), though it changes the *aggregate* only when the
  affected class is binding.

**Exposure selection under lane-aware memory.** The implementation should select
the smallest legal `(p, U)` candidate whose effective load/store terms saturate
the binding memory class (i.e. reach `E_sat`) after scalar lane eligibility and
vector coalescing are applied. It **SHOULD** flag candidates below that point as
**bandwidth-starved**. It **SHOULD** flag larger candidates as
**oversubscribed** only when extra exposure no longer improves the effective
binding memory term and only increases transient backlog, area, mapping pressure,
or non-modeled control/work. On the legacy path, `absolute_cgra_lb` is the
full-trip, fully-coalesced aggregate over full lanes `L`/`S`. In an extended
profile it is the minimum legal `plan_cgra_lb` for that profile. It remains the
sole report-global lower bound; every Loom-pragma candidate estimate sits at or
above its plan and global floors.

**Search completeness.** The current design-space search uses power-of-two
pragma factors and **MUST** consider every power-of-two `p` and `U` allowed by
the modeled trip counts and dependency legality; it **MUST NOT** impose an
implicit factor-of-eight or global exposure limit. An explicit source
`LOOM_PARALLEL(P)` or `LOOM_UNROLL(U)` value is a candidate hint, not a search
maximum. Factors that
cannot expose additional iterations (`p * U > trip_count`) may be omitted as
duplicate exposure. This power-of-two policy is an intentional current DSE
scope choice; non-power-of-two factors may be added in a future extension.
Reduction and sequential levels that are fully consumed in every candidate may
likewise use one canonical factor label when all labels build the same DAG. An
implementation **MAY** accept user-requested factor or exposure caps for
diagnostic runs, but a capped report **MUST** label itself as a bounded search and
**MUST NOT** present its result as the global recommendation. When an extended
profile-global floor is derived by minimizing over candidates, a capped report
must instead label that minimum as a bounded-search floor.

For an extended pilot, candidate enumeration takes the cross-product of the legal
`p` factors, legal `U` factors, explicitly declared legal loop orders, and
explicitly declared complete jam plans. Fixed vector width and scratchpad
placement are not search axes. Logical base offsets and resident-versus-fallback
placement are deterministic derived decisions.

**When a kernel shows no `LOOM_PARALLEL`-vs-`LOOM_UNROLL` distinction.** Because
control amortization now separates `p` from `U` on the op counts of any parallel
level, genuine symmetry in the **cycle aggregate** arises only when both modeled
effects (coalescing and control) fall off the binding path. Report symmetry, with
the reason, in these cases:

- **Fully-consumed reduction dimension.** A reduction is lowered to a spatial
  (tree-reduced) dataflow graph: it carries no per-element and no per-worker loop
  control, and its contiguous inputs coalesce identically for any split. Both axes
  are inert, so the dimension is exactly symmetric (`vecsum`'s whole loop;
  `gemv`'s `j`; `conv2d`'s `tap`).
- **Latency-bound.** If the aggregate is set by `CP`, exposure does not change it
  and control sits off the critical path, so the candidates are symmetric
  (`tridiag_solve`; `vecsum` at its `CP` floor).
- **Compute-bound (approximately).** If the algorithmic `compute` term binds and
  the small `P_tot·a_ctrl` control-arith delta does not tip its ceiling, the
  parallel- and unroll-heavy splits tie in the aggregate even though their load /
  store / control op counts differ (`batchnorm`'s `c`/`h` at fixed product). Report
  the tie in the aggregate while noting the underlying control/coalescing gap.

Otherwise a parallel level is **asymmetric and favors `LOOM_UNROLL`**: control
amortization shrinks its per-worker iterator count, and (for contiguous accesses)
coalescing shrinks its lane-slot term. A **sequential** level cannot be
parallelized at all (`P_tot = 1` there), so only legal unroll exposure applies,
its iterator stays per-iteration on the critical path, and carried recurrences
still limit throughput.

**Stream-unit diagnostic.** `LOOM_PARALLEL(P)` creates `P_tot` stream units,
while `LOOM_UNROLL(U)` increases the exposed work inside each stream unit. The
current DSE may report stream-unit count as a diagnostic, but stream units do not
add resource cost in the provisional selection objective.

A **nested** loop adds a second, orthogonal effect: the per-level distribution of
exposure (how much on the outer vs the inner loop) changes the chunk's op counts,
because a value loop-invariant with respect to the inner loop is loaded once and
reused across inner iterations — so exposing the inner loop **amortizes** that
outer-invariant traffic while exposing the outer loop **replicates** it. (The
same level-asymmetry arises when an inner level is a reduction.) This level effect
composes with the lane-aware memory model above, which still applies *within* any
single level.

### Shared scratchpad, ports, and direct fallback

A named analytical target profile supplies scratchpad capacity, logical load-port
count, logical store-port count, access latency, and sharing scope. These values
are hardware parameters, matching the role of `ldCount` and `stCount` on
`fabric.memory`, but a DSE run receives them explicitly and does not infer them
from a particular ADG, `is_private`, `numRegion`, or source
`LOOM_MEMORY_BANK` annotation. Under `shared-spad-4k-r1w1-v4`, all workers of one
mapped kernel share one logical scratchpad address space.

Buffer intent is declared per kernel:

- `resident_shared`: read-only reuse that is invariant across the workers that
  consume it is proposed once for shared residency.
- `resident_replicated`: genuinely worker-specific reusable state is proposed
  once per required private copy.
- `direct`: reuse-free streaming traffic remains in direct external memory.

For each proposed resident buffer, collect the exact unique source elements used
by the whole modeled kernel. Sort those elements, compact them into offsets
`0 .. count - 1`, align each declaration-order buffer or replica segment to the
fixed vector width in elements, and assign deterministic non-overlapping logical
base offsets. Source-address holes do not consume capacity; compacted data,
alignment padding, and replicas do.

Placement is deterministic and whole-plan. If the complete proposed resident set
fits the target capacity, every proposed buffer receives its declared resident
placement. If it does not fit, every proposed resident buffer becomes
`direct-fallback`; the candidate remains legal and uses direct external-memory
traffic. The helper **MUST NOT** search subsets of resident buffers or smaller
scratchpad tiles. Reuse-free buffers remain ordinary `direct` traffic in either
case.

When residency succeeds, external preload occurs once per modeled kernel
execution, never once per partial-exposure wave. Unique scalar elements are
grouped by contiguous source address and coalesced at the fixed `V = 4`. Each
coalesced group emits one external `L` operation and one scratchpad-write
operation, with a partial final group still occupying one lane slot:

```
preload_scalar_elems = sum(group_elems for group in contiguous_groups)
preload_L_ops        = sum(ceil(group_elems / V) for group in contiguous_groups)
preload_spad_W_ops   = preload_L_ops
```

Fallback plans have zero preload and zero scratchpad traffic. Their formerly
resident accesses are accounted as direct external loads.

Every dynamic resident-data use consumes scratchpad bandwidth. Free read fan-out
is allowed only when multiple consumers request the same logical address at the
same logical inner step. Equal addresses used at different output positions or
steps remain distinct scratchpad reads. One scalar or coalesced vector operation
occupies one logical scratchpad port for `access_cycles` modeled cycles.

For one scratchpad-bearing ordered region, let `spad_R` and `spad_W` be its
logical scratchpad read and write operation counts. The port floor is

```
spad_read_lb  = ceil(spad_R / load_ports) * access_cycles
spad_write_lb = ceil(spad_W / store_ports) * access_cycles
spad_port_lb  = max(spad_read_lb, spad_write_lb)
```

The deterministic port schedule, `spad_port_sched`, processes operations in
stable node order and assigns each to the earliest cycle with a free port of the
correct kind for the full access latency. Read and write ports are independent.
The ordered-region corrections are

```
scratchpad_aggregate = max(existing_aggregate, spad_port_lb)
scratchpad_schedule  = max(existing_schedule, spad_port_sched)
```

This correction is local to the Loom-pragma DSE. It does not add a resource class
to the generic scheduler and does not model banks, replacement, coherence,
arbitration, DMA overlap, or place-and-route.

### Whole-working-set composition

The extended DSE may explore analytical loop order and jam choices while leaving
the checked-in source and compiler behavior unchanged. It does not search virtual
tile sizes. The modeled kernel has one whole-working-set memory plan and one
possible preload prologue.

For a resident candidate, total composition is the once-per-kernel preload plus
the wave-serialized recurring computation. For a fallback candidate the preload
term is zero and the recurring computation uses direct memory. The saturation
knee is selected from recurring compute pressure rather than the preload
prologue.

For an extended candidate, `plan_cgra_lb` is the full-exposure aggregate over the
same transformed operation set, fixed-width coalescing, deterministic placement,
scratchpad-port correction, and once-per-kernel preload, but without
partial-exposure wave serialization. For exact profile identity `profile`, the
single global floor is

```
absolute_cgra_lb(profile) = min(plan_cgra_lb(candidate, profile)
                                for candidate in legal_candidates(profile))
```

Thus every legal extended candidate preserves
`absolute_cgra_lb <= plan_cgra_lb <= pragma_exposure_aggregate`, and every
materialized finite-resource schedule additionally preserves
`pragma_exposure_aggregate <= schedule_estimate`. Reports **MUST** expose the
selected order and explicit jam plan; placement and logical offsets; target
capacity and load/store-port counts; resident bytes and fallback status;
scratchpad port floor and schedule; and external preload, scratchpad-read, and
avoided-direct traffic. Direct fallback is the candidate's actual placement and
participates normally in ranking and the profile-global floor.

This exploratory estimate is **not** the aggregate CGRA lower bound, **not** the
fully-unrolled ASAP metric, and **not** cycle-accurate RTL. It models candidate
memory-lane exposure and optional vector coalescing. Neither the legacy path nor
the default scratchpad profile models address-level bank conflicts. Neither path
is a place-and-route model. The
DSE **MUST NOT** replace or rename the aggregate lower bound in a kernel's main
`*_eval.md` file. The phrase "lower bound" **MUST NOT** be applied to
`pragma_exposure_aggregate` or `schedule_estimate`.

**Reference implementation.** The Loom-pragma design-space estimate is
implemented by `tests/scripts/loom_dse.py`, which reuses the DAG primitives,
list scheduler, and aggregate computer of `tests/scripts/cgra_schedule.py`
(Metrics 1–2). It builds each candidate's vectorized chunk DAG (coalescing
contiguous unrolled groups per the rule above), schedules it, sums over waves,
and emits the per-candidate table, the report-global `absolute_cgra_lb`, and the
recommended saturation-knee exposure. Extended reports additionally emit
`plan_cgra_lb` and the named transformation, explicit jam, placement, traffic,
capacity, and scratchpad-port diagnostics defined above. It exposes a
`--self-test` entry point
covering the unroll-favoring, symmetric, sequential, and extended-profile cases
and their bracket invariants.
Per-kernel design-space
evals live at `tests/app/<kernel>/<kernel>_loom_dse.md`; they are distinct from the
`_eval.md` files and carry only this optional estimate.

## Eval Reporting Format

Each pilot eval (`tests/app/<kernel>/<kernel>_eval.md`) already contains a
`## CGRA-Constrained Model` section with the aggregate prose and the
`CP`/`A`/`LD`/`ST`/aggregate numbers. The finite-resource estimate is added as a
**marker-bounded block** that the helper writes and re-writes automatically.

- The block is delimited by HTML-comment markers of the form
  `<!-- BEGIN CGRA-SCHED:<kernel> -->` … `<!-- END CGRA-SCHED:<kernel> -->`.
- The block is placed **after** the existing section content (appended at the end
  of the file on first write).
- On re-run, the helper replaces **only** the bytes between its own markers and
  changes nothing else. Re-running **MUST NOT** duplicate the block or edit bytes
  outside the markers.
- The helper **MUST NOT** modify any text under a `## ASAP Model Notes` heading,
  and **MUST NOT** rewrite the pre-existing aggregate prose.

The block **MUST** report:

- the resource configuration (e.g. `P = 36`, `L = 12`, `S = 12`);
- the retained `CP`, `A`, `LD`, `ST`, and `aggregate_cycles`;
- `scheduled_cycles`, `gap_cycles`, and `gap_ratio` (labeled `scheduled /
  aggregate`);
- the local `P/L/S` pressure summary;
- for a multi-region kernel, a per-region table (per-region `CP`/`A`/`LD`/`ST`,
  aggregate, makespan) before the kernel total; single-region kernels show the
  one region's row as well.

Every written block **MUST** describe its finite-resource result as an
"estimate" for the defined scheduling policy, distinct from the aggregate lower
bound, and **MUST NOT** use the words "lower bound" for it.

## Helper Tool Contract

The reference implementation is a single standard-library Python module at
`tests/scripts/cgra_schedule.py`. It:

- uses the **Python standard library only** (no third-party scheduler library
  such as `networkx`);
- exposes reusable DAG primitives — at least `load`, `store`, `arith`,
  `address_add`, a balanced reduction builder, induction/control helpers, and a
  `region` constructor — plus height computation, the list scheduler, the
  aggregate computer, the metrics, the pressure summary, and a canonical
  eval-block formatter;
- **MUST NOT** parse the prose Markdown evals to derive a DAG; builders construct
  the DAG directly from kernel structure;
- provides per-kernel **builders** that construct the full unrolled DAG and
  declare a **builder contract**: region names, per-region `A`/`LD`/`ST`,
  per-region `CP`, barriers, and the expected aggregate. The builder's
  constructed DAG **MUST** match its declared contract.
- provides a **`--self-test`** entry point that runs and exits zero (mirroring
  `tests/scripts/check_bridge_tags.py --self-test`), covering the synthetic edge
  cases, the golden pilot anchors, the zero-capacity rejection, and a read-only
  drift check;
- provides a **report mode** (e.g. `report <kernel> --config 6x6`) that emits the
  canonical eval block and prints the per-region validation rows;
- provides a read-only **`--check`** mode that re-derives the numbers and
  confirms they match what is written in each pilot eval, failing if an eval's
  written numbers differ from the freshly computed values.

### Golden anchors

The pilot builders **MUST** reproduce the hand-derived eval numbers exactly:

| kernel | sizes | `CP` | `A` | `LD` | `ST` | aggregate `6x6` |
|--------|-------|-----:|----:|-----:|-----:|---------------:|
| `axpy` | `N = 8` | 4 | 32 | 26 | 16 | `max(4, ⌈32/36⌉, ⌈26/12⌉, ⌈16/12⌉) = 4` |
| `autocorrelation` | `x_size = 128`, `max_lag = 32` | 11 | 18064 | 10834 | 3664 | `max(11, 502, 903, 306) = 903` |
| `fft_butterfly` | `N = 16` | (per stage) | (per phase) | (per phase) | (per phase) | phase sum `5 + 8 + 11 + 17 + 33 = 74` |

`fft_butterfly` is barrier-ordered into five regions; the per-phase aggregate
table is:

| phase | `CP` | `A` | `LD` | `ST` | aggregate |
|-------|-----:|----:|-----:|-----:|----------:|
| copy  | 2  | 32  | 48 | 49 | 5  |
| s = 1 | 8  | 183 | 65 | 90 | 8  |
| s = 2 | 11 | 175 | 61 | 74 | 11 |
| s = 3 | 17 | 171 | 59 | 66 | 17 |
| s = 4 | 33 | 169 | 58 | 62 | 33 |

The three kernel-once residual ops (the `N` load, the `log2f(N)` transcendental,
and the `s`-loop init store) overlap the copy phase and add no cycles; the
builder places them **in the copy region** (so the constructed copy region
measures `A = 33`, `LD = 49`, `ST = 50` = the documented copy phase plus the
three residuals), keeping the copy aggregate at `max(2, 1, 5, 5) = 5` and the
phase sum at `74`. For each pilot, `scheduled_cycles >= aggregate_cycles`
**MUST** hold.

`conv2d` (`CP = 17`, `A = 74515`, `LD = 13716`, `ST = 6220`, aggregate
`6x6 = 2070`, a single region because its zero-fill is a dead WAW) is a follow-up
builder, not part of the first deliverable.

## Validation

Validation for this feature is **Python-only and standalone**:
`python3 tests/scripts/cgra_schedule.py --self-test` runs without an EDA
toolchain and without a C++/`ninja` build, because the feature changes only
Python and Markdown files. The first deliverable covers `axpy`,
`autocorrelation`, and `fft_butterfly`; `conv2d` is delivered as a follow-up with
a runtime/scale check and **MUST NOT** block the first deliverable.

## Related Documents

- `tests/app/ASAP_rules.md` — the authoritative ASAP dynamic-op-count and
  critical-path conventions this spec builds on.
- `AGENTS.md` — the committed "CGRA-Constrained Model" agent policy that
  references this spec.
- `tests/app/<kernel>/<kernel>_eval.md` — per-kernel evals carrying both metrics.
- `tests/scripts/cgra_schedule.py` — the reference helper implementing Metrics 1–2.
- `tests/scripts/loom_dse.py` — the reference helper implementing the optional
  Loom-pragma design-space estimate; per-kernel results in
  `tests/app/<kernel>/<kernel>_loom_dse.md`.
- `tests/scripts/check_bridge_tags.py` — the `--self-test` convention mirrored
  here.
