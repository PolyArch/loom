# ASAP Model Notes
- update edge weight in a Compressed Sparse Row graph

## Cycle Count/Critical Path
    Initial state: inputs available
    c1: bounds compare  src_node >= num_nodes
    c2: load row_ptr[src_node] (=row_start)  ‖  addr_add src_node+1     [gated by c1]
    c3: load row_ptr[src_node+1] (=row_end)  ‖  compute i = row_start+m
    c4: load input_col_indices[i]            ‖  bound compare i<row_end  (overlap, off path)
    c5: match compare  col == dst_node
    c6: store output_weights[i]              (needs match @c5)

    Total ≈ 6, with the copy loop running in parallel

# Edge Update Performance
Parameters (from `main.cpp`): `num_nodes = 8`, `num_edges = 16`, `src = 2`,
`dst = 4`, `new_weight = 100`.
- `row_ptr = {0, 2, 4, 7, 10, 12, 14, 15, 16}`
- `col_indices = {1, 2, 0, 3, 0, 4, 5, 1, 2, 6, 3, 7, 4, 6, 7, 5}`
- `input_weights[i] = i + 1`
- For `src = 2`: `row_start = row_ptr[2] = 4`, `row_end = row_ptr[3] = 7` →
  degree `D = 3`. The scan reads `col_indices[4] = 0` (miss), then
  `col_indices[5] = 4 == dst` (hit) → stores `output_weights[5] = 100` and
  returns. So `K = 2` search iterations execute; matched index = 5.
- Expected `output_weights = {1,2,3,4,5,100,7,8,9,10,11,12,13,14,15,16}`.

Size parameters used in the formulas:
- `E` = `num_edges` (copy-loop trip count) = 16
- `D` = degree of `src_node` = `row_ptr[src_node+1] − row_ptr[src_node]` = 3
  (max search-loop trip count)
- `K` = search iterations actually executed (`K ≤ D`, data-dependent: the loop
  returns on the first match, so `K` = match position + 1, or `D` on no
  match) = 2

## Loop classification

| dim | trip_count | kind | II | notes |
|-----|------------|------|----|-------|
| copy `i` | `E` = 16 | parallel | n/a | each iter copies `input_weights[i] → output_weights[i]` at a distinct index; no value crosses iterations. Fully unrolled. Connects to the rest only through a write-after-write on `output_weights[matched]`, which resolves long before the matched store. |
| search `i` | `K ≤ D`, `K` = 2 | parallel (data-dependent termination) | n/a | each iter touches only its own `input_col_indices[i]` and a distinct `output_weights[i]`; no carried register/accumulator/in-place dep, so it is parallel by data-dependence. The early `return` makes trip `K` input-dependent, and the match compare `col_indices[i] == dst_node` is the termination predicate on the critical path. |

Neither loop carries a register/accumulator/in-place dependence, so neither
contributes a `trip × II` term to `total_cycles`: under full unrolling each lane
treats its iterator as a per-lane constant, so the induction work stays **off the
critical path**. Op counts, however, are independent of scheduling — the
source-level loop-control work is still counted. Both loops therefore charge
per-iteration iv ops (load, add, store, compare) in the op totals exactly as a
sequential loop would; the only difference is that here those ops do not extend
`total_cycles`. (Contrast `crc32_eval.md`, whose **sequential** dims charge the
same iv ops *and* put them on the critical path.)

`row_start`, `row_end`, and the loaded `input_col_indices[i]` value are each
assigned exactly once and not loop-carried, so they are anonymous dataflow —
free fan-out from the defining op, no scalar L/S. Only the underlying array
loads (`row_ptr[…]`, `input_col_indices[i]`) are charged. `src_node`,
`num_nodes`, `dst_node`, and `new_weight` are kernel-input scalars and fan out
freely. The conditional store of `new_weight` is the sole algorithmic store of
the search phase and fires only on the matched iteration.

## Critical path (`total_cycles`)

The binding chain is the upstream bounds → row-pointer chain feeding **one**
search-loop iteration; it does not run through the copy loop and does not scale
with any trip count:

```
1 (bounds cmp  src_node >= num_nodes)        [gates all downstream under no-pred]
+ 1 (load row_ptr[src_node] = row_start  ‖  address_add src_node+1)
+ 1 (load row_ptr[src_node+1] = row_end  ‖  compute search i)
+ 1 (load input_col_indices[i]           ‖  bound cmp i<row_end overlaps, off path)
+ 1 (match cmp  input_col_indices[i] == dst_node)
+ 1 (store output_weights[i] = new_weight)   [needs match]
= 6
```

`total_cycles ≈ 6`, constant in `E`, `D`, and `K`. Why the constant depth holds:
- **Copy loop is hidden.** Its per-iteration chain is just
  `load input_weights[i] → store output_weights[i]` (≈2 cycles; bare `[i]`
  subscripts → no address arithmetic), starting at c0. It links to the rest
  only through a write-after-write on `output_weights`, and the copy store
  (~c2) precedes the matched search store (~c6), so it never extends the path.
- **The loop-bound compare does not serialize.** `i < row_end` overlaps the
  body's data path (c4) and adds no cycle once `row_end` is a known scalar —
  treated symmetrically with the copy loop's `i < num_edges`, which likewise
  adds no cycle (both are still counted as iv compares in the op totals). The
  search store lands at c6 only because its chain cannot
  *start* until `row_start`/`row_end` are produced by the upstream
  bounds-check → `row_ptr` chain (c1–c3); the copy chain has no such upstream
  dependence and starts at c0.
- **The match compare stays on the path** as the genuine data-dependent
  termination predicate of the early-return loop. Iterations are mutually
  independent (each touches only its own `i`), so full unrolling counts the
  per-iteration chain once — the early return selects *which* iteration stores
  but does not lengthen the longest single-iteration chain.

## Op counts

### Per-phase formulas
- **Copy loop** (parallel, trip `E`): `E` loads (`input_weights[i]`) + `E`
  stores (`output_weights[i]`), plus induction: `E` iv loads + `E + 1` iv stores
  (`E` writebacks + `i = 0` init) + `E` iv adds (`i++`) + `E` iv compares
  (`i < num_edges`). No address_adds (bare `[i]`).
- **Bounds + row pointers**: `1` compare (`src_node >= num_nodes`) + `2` loads
  (`row_ptr[src_node]`, `row_ptr[src_node+1]`) + `1` address_add (`src_node+1`,
  inline subscript arithmetic).
- **Search loop** (parallel, data-dependent termination, trip `K`): `K` loads
  (`input_col_indices[i]`) + `K` compares (`col_indices[i] == dst_node`, the
  termination predicate) + `1` store (matched `output_weights[i] = new_weight`),
  plus induction over the `K` executed iters: `K` iv loads + `K + 1` iv stores
  (`K` writebacks + `i = row_start` init) + `K` iv adds (`i++`) + `K` iv compares
  (bound `i < row_end`). No address_adds (bare `[i]`).

### Algorithmic
| op | count | source |
|----|-------|-------|
| loads    | `E + K + 2` = **20** | `input_weights[i]` (E=16) + `input_col_indices[i]` (K=2) + `row_ptr[·]` (2) |
| stores   | `E + 1` = **17**     | `output_weights[i]` copy (E=16) + matched `output_weights[5]=new_weight` (1) |
| compares | `K + 1` = **3**      | `src_node >= num_nodes` (1) + `col_indices[i] == dst_node` per executed iter (K=2) |

### Overhead (address-gen, induction)
| op | count | source |
|----|-------|-------|
| address_adds | **1** | `src_node + 1` inline in `row_ptr[src_node+1]`. All other subscripts (`[i]`, `[src_node]`) are bare → no address_add. |
| iv loads | **18** | copy `i` (`E` = 16) + search `i` (`K` = 2), one read per executed iter |
| iv stores | **20** | copy `i` (16 writebacks + 1 `i = 0` init = 17) + search `i` (2 writebacks + 1 `i = row_start` init = 3) |
| iv adds | **18** | `i++`: copy (`E` = 16) + search (`K` = 2) |
| iv compares | **18** | bound checks: copy `i < num_edges` (16) + search `i < row_end` (2) |

### Totals
| op | total |
|----|------:|
| loads        | **38** |
| stores       | **37** |
| adds         | **18** |
| compares     | **21** |
| address_adds | **1**  |
| muls / divs / shifts / bitops / transcendentals | 0 |

The work is dominated by the bulk copy: its algorithmic `E` loads + `E` stores
(32) plus per-iter induction traffic (`E` iv loads + `E + 1` iv stores = 33)
make up 65 of the 75 memory ops; the actual edge update is a 2-iteration scan
(`K` loads + iv, `K` match compares, one store). Both copy and scan run off the
critical path of the upstream bounds → row-pointer chain, so `total_cycles`
stays at ≈6 regardless of `E`, `D`, or `K`.

## Data Dependency Graph
The binding chain is the upstream bounds → row-pointer chain feeding one search
iteration (red). The copy loop (top, parallel) and the non-matching search
iterations run concurrently and off the critical path; they connect only via a
write-after-write on `output_weights` that resolves well before the matched
store. Bare `[i]`/`[src_node]` subscripts charge no address-gen cycle; only the
inline `src_node + 1` does.

```mermaid
graph TD
    %% Kernel inputs (anonymous dataflow — free fan-out)
    src(("src_node"))
    nn(("num_nodes"))
    dst(("dst_node"))
    nw(("new_weight"))

    %% Copy loop — parallel, hidden under the upstream chain
    iw(("input_weights[i]"))
    cp_st(("store output_weights[i]"))
    iw -->|load| cp_st

    %% Bounds check → row-pointer chain
    bcmp((" src >= num_nodes "))
    rs(("load row_ptr[src] = row_start"))
    aadd((" src + 1 "))
    re(("load row_ptr[src+1] = row_end"))

    src --> bcmp
    nn --> bcmp
    bcmp -. F: valid src .-> rs
    bcmp -. F: valid src .-> aadd
    src --> aadd
    aadd --> re

    %% One search iteration  (i = row_start + m depends only on row_start)
    ci(("compute i = row_start + m"))
    lc(("load col_indices[i]"))
    mcmp((" col == dst "))
    st(("store output_weights[i] = new_weight"))
    bnd((" i < row_end "))

    rs --> ci
    ci --> lc
    lc --> mcmp
    dst --> mcmp
    mcmp -. T: match → store, return .-> st
    nw --> st

    %% Bound compare overlaps the body and is off the critical path
    ci --> bnd
    re --> bnd

    %% WAW: matched store overwrites the copied weight (copy store precedes it)
    cp_st -. WAW (resolves early) .-> st

    %% Critical path (6 cycles): bounds cmp → load row_start → compute i → load col_indices → match cmp → store
    linkStyle 3,7,8,9,11 stroke:#ff0000,stroke-width:3px;
```

The constant 6-cycle depth is set by the upstream bounds → row-pointer chain
(`bounds cmp → load row_start → load row_end`) plus one search iteration's
`load col_indices[i] → match cmp → store`. The copy loop and the additional
non-matching scan iterations add op-count work but never extend the path.

## CGRA-Constrained Model

The ASAP bound above assumes unlimited functional units and memory bandwidth.
This section adds the aggregate lower bound for a CGRA with separate arithmetic
and memory-issue resources, following `docs/spec-kernel-performance.md`.

The copy loop and matched update are one schedulable region. The matched update
overwrites one copied `output_weights[]` slot, but the copied value is not read
before being overwritten, so this write-after-write relation is not a RAW
barrier and is not split into ordered phases.

With `6x6` resources (`P = 36`, `L = 12`, `S = 12`):

- `CP = 6`
- `A = adds (18) + address_adds (1) + compares (21) = 40`
- `LD = 38`
- `ST = 37`

```
compute = ceil(40 / 36) = 2
load    = ceil(38 / 12) = 4
store   = ceil(37 / 12) = 4
cycles  = max(6, 2, 4, 4) = 6
```

**Bottleneck: dependency-bound.** The row-pointer and matched-update chain is
longer than every aggregate resource term for this small CSR update.

<!-- BEGIN CGRA-SCHED:edge_update -->
### Finite-Resource Schedule Estimate (time-local)

*Reproducible estimate for the deterministic criticality-priority list-schedule policy defined in [`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md). It is **not** a lower bound (the aggregate model above is the lower bound) and **not** cycle-accurate RTL; it exposes the short windows of local `P`/`L`/`S` pressure that the aggregate model smooths over.*

**Resource configuration:** `P = 36`, `L = 12`, `S = 12` (`6x6`).

| region | CP | A | LD | ST | aggregate | scheduled (makespan) |
|--------|---:|--:|---:|---:|----------:|---------------------:|
| edge_update | 6 | 40 | 38 | 37 | 6 | 6 |

- **scheduled_cycles** = 6  (sum of ordered-region makespans)
- **aggregate_cycles** = 6  (the lower bound above, unchanged)
- **gap_cycles** = 0  (scheduled − aggregate)
- **gap_ratio** = 1  (scheduled / aggregate)

**Local `P`/`L`/`S` pressure** (saturated cycles / longest saturated run / peak ready backlog):
- `P`: 0 / 0 / 0
- `L`: 3 / 3 / 22
- `S`: 2 / 2 / 10

<!-- END CGRA-SCHED:edge_update -->
