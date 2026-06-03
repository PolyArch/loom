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
operation model** documented in `AGENTS.md` ("Performance Modeling"). That model
is summarized here so this spec can be understood standalone; `AGENTS.md` remains
the authoritative source for the op-counting conventions themselves, and this
spec does not restate every convention verbatim.

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

- `AGENTS.md` — "Performance Modeling": the authoritative ASAP dynamic-op-count
  and critical-path conventions this spec builds on, and the committed
  "CGRA-Constrained Model" policy text that references this spec.
- `tests/app/<kernel>/<kernel>_eval.md` — per-kernel evals carrying both metrics.
- `tests/scripts/cgra_schedule.py` — the reference helper implementing this spec.
- `tests/scripts/check_bridge_tags.py` — the `--self-test` convention mirrored
  here.
