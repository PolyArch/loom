# ASAP Model Notes
s loop sequential, k loop parallel, j loop sequential
- s = 4 takes the longest to compute since j iterates m/2 or 8 times

Example for N = 16, adjacent stages s = 2 and s = 3:

  At s = 2, m = 4. One butterfly is:

  k = 4, j = 1
  output_real[k + j + m/2] = output_real[4 + 1 + 2] = output_real[7]

  So stage 2 writes output_real[7].

  At s = 3, m = 8. One later butterfly is:

  k = 0, j = 3
  float t_r = ... output_real[k + j + m/2]
            = ... output_real[0 + 3 + 4]
            = ... output_real[7]

  So stage 3 reads output_real[7].

  That is a RAW edge.

Even when the s loop is unrolled, stage s+1 consumes the in-place array values produced by stage s. Since each stage writes every output_real/imag[i], and the next stage reads every output_real/imag[i], the stages must be barrier-ordered:

copy -> s=1 -> s=2 -> s=3 -> s=4

So unrolling s creates parallel hardware/graphs, but the data dependencies still force sequential execution between stages.

k iterations within one stage can run in parallel because their index ranges are disjoint. Since all k blocks in the same stage have the same j trip count, the stage latency is just one k block’s j-loop depth, not the sum across all k blocks.

The critical path is the sum of each stage’s j-loop critical-path depth. Each j loop is sequential because of the twiddle recurrence w = w * wm, and its per-stage depth also includes the index/address/load/butterfly/store chain.

So the structure for the critical path is:

total_cycles =
  copy depth
  + stage(s=1) j-chain depth
  + stage(s=2) j-chain depth
  + stage(s=3) j-chain depth
  + stage(s=4) j-chain depth

**s is unrolled structurally, but still sequential in latency because of RAW hazards through the in-place output arrays.**

## Cycle Count/Critical Path
Radix-2 DIT, in-place. For the ideal infinite-throughput schedule, the outer
stage loop is materialized as four unrolled stage instances (`s = 1..4`), but
the stage instances are separated by whole-stage barriers. A later stage may
read an element written by the previous stage, so the barriers are required RAW
ordering edges, not optional serialization. For example, stage 2 (`m = 4`) at
block base `k = 4`, `j = 1` touches index `4 + 1 + 2 = 7`; stage 3 (`m = 8`) at
`k = 0`, `j = 3` touches `0 + 3 + 4 = 7`. Without a barrier, stage 3 could read
the old value of element 7.

Copy loop (parallel, fully unrolled):
- c1: load `input_real[i]` ‖ load `input_imag[i]` (bare `[i]`, no addr_add)
- c2: store `output_real[i]` ‖ store `output_imag[i]`

Per-stage prologue (array-independent, overlaps earlier stages):
- `1 << s = m` → `-2π/m` → `cosf` ‖ `sinf = wm_r, wm_i`

Inside each stage, the `k` blocks are independent and fully unrolled. The stage
depth is therefore one block's `j`-loop depth. The `j` loop is still sequential
because it carries the running twiddle `(w_r, w_i)`.

Inside the `j` loop there are two independent chains that meet only at `t = w·X`:

TWIDDLE coefficient (scalar recurrence, II = 4):
- load `w_r` ‖ load `w_i`
- `w·wm` products (4 muls, parallel)
- `new_w_r` sub ‖ `new_w_i` add
- store `w_r` ‖ store `w_i` (carry to the next `j` iteration)
- `w^(p)` is ready at `1 + 4p`; the final iteration's `w` update is dead for
  latency because no later `j` iteration consumes it.

DATA operand (addressed load):
- `j` induction carry (`load j → j+1 → store j`, II = 3)
- inline address arithmetic: `k+j` then `+m/2`
- load `output[k+j+m/2] = X`
- `X^(p)` is ready at `3p + 4`

Convergence at the butterfly:
- `t = w·X` waits for the slower of the twiddle and data chains
- `t_r` sub ‖ `t_i` add
- `u ± t` (the `u = output[k+j]` load is earlier and off the binding chain)
- store `output[k+j]` ‖ store `output[k+j+m/2]`

Per-stage critical path:
```
stage_CP(m) = max(2m + 1,      // twiddle-bound path
                  1.5m + 5)    // j/index/address-bound path
```

| s | `m` | twiddle-bound `2m+1` | index-bound `1.5m+5` | `stage_CP` | binding |
|---|-----|----------------------|----------------------|-----------:|---------|
| 1 | 2  | 5  | 8  | 8  | index/address |
| 2 | 4  | 9  | 11 | 11 | index/address |
| 3 | 8  | 17 | 17 | 17 | tie |
| 4 | 16 | 33 | 29 | 33 | twiddle recurrence |

The barrier-ordered stage graph is:
```
copy -> barrier -> stage(s=1) -> barrier -> stage(s=2)
     -> barrier -> stage(s=3) -> barrier -> stage(s=4)
```

So the total critical path is:
```
total_cycles = 2 (copy)
             + 8 + 11 + 17 + 33
             = 71
```

# FFT Butterfly Performance
Parameters (from `main.cpp`): `N = 16`, `log2(N) = 4`. 

Radix-2 decimation-in-time Cooley-Tukey: after copying the (bit-reversed) input
into the working `output_*` buffers, `log2(N)` stages combine sub-transforms of
size `m = 2^s`. Per stage, twiddle `wm = e^(−2πi/m)` is formed once, then each
`m`-block runs `m/2` butterflies on the pairs `(k+j, k+j+m/2)`, rotating a running
twiddle `w ← w·wm` between butterflies.

Per-stage geometry (`N = 16`):

| s | `m = 2^s` | `m/2` (j-trip) | `N/m` (k-blocks) | j-iters in stage = `(N/m)·(m/2)` |
|---|-----------|----------------|------------------|----------------------------------|
| 1 | 2  | 1 | 8 | 8 |
| 2 | 4  | 2 | 4 | 8 |
| 3 | 8  | 4 | 2 | 8 |
| 4 | 16 | 8 | 1 | 8 |

Total butterflies = `Σ (N/m)·(m/2) = log2(N)·N/2 = 4·8 = 32`. Total k-block
instances = `Σ N/m = 8+4+2+1 = 15`.

## Loop classification

| dim | trip_count | kind | II | notes |
|-----|------------|------|----|-------|
| copy `i` | `N` = 16 | parallel | — | `output[i] = input[i]` at a distinct index; no carry. `LOOM_PARALLEL` + `LOOM_UNROLL(8)`, fully unrolled. Bare `[i]` subscripts → no address arithmetic. |
| stage `s` | 4 materialized instances | sequential | per-stage critical path (whole-stage barrier) | The outer loop is modeled as unrolled stage graphs for latency, but adjacent stages are barrier-ordered because they read-modify-write overlapping `output_*` elements. This removes ordinary `s` stream/induction latency from the critical path, while preserving the required RAW ordering between stages. |
| block `k` | `N/m` (8,4,2,1) | parallel | — | block `k` owns the disjoint index range `[k, k+m)`; no value crosses blocks. Fully unrolled → `k` is a per-lane constant on the body critical path and folds into the address constants; the source `k` induction is still counted in op totals. |
| butterfly `j` | `m/2` (1,2,4,8) | sequential | 4 | carries the running twiddle `(w_r, w_i)` via `w ← w·wm`. Non-associative complex rotation → cannot tree-reduce. The array footprints of successive `j` are disjoint, so the **only** carry is the twiddle. |

`w_r`/`w_i` are loop-carried and have two assignment sites (`= 1.0f`/`= 0.0f` init,
then `= new_w_r`/`= new_w_i`), so they are **memory-backed** — each named read is a
1-cycle load, each write a 1-cycle store. Within one `j`-iter `w_r` is read four
times (`t_r`, `t_i`, `new_w_r`, `new_w_i`) with no intervening write, so it is
loaded **once** and fanned to all four uses; `w_i` likewise. `wm_r`/`wm_i` have a
single assignment site per stage and are not carried, so they are loop-invariant
dataflow — computed once per stage and broadcast, no scalar L/S. `t_r`, `t_i`,
`u_r`, `u_i`, `new_w_r`, `new_w_i` are single-assignment, non-carried → anonymous
dataflow (the array loads / op results flow directly into their consumers with no
named L/S).

## Critical path (`total_cycles`)

The outer `s` loop is unrolled into four explicit stage instances. Those stage
graphs are still ordered by whole-stage barriers because stage `s+1` may read
the same element that stage `s` just wrote. The barrier itself is not charged as
an op; it constrains the schedule so every read in the next stage waits until all
stores in the previous stage complete.

Within a stage the `k`-blocks are parallel, so the stage's depth is one block's
`j`-loop depth. Two **data-independent** chains advance in parallel inside that
`j`-loop and meet only at the butterfly product `t = w·X`:

> **Twiddle coefficient vs. data operand — they don't share an address.** The
> twiddle `w = (w_r, w_i)` is the rotation *coefficient*: a loop-carried **scalar**
> held at a fixed address, advanced by `w ← w·wm` from the previous `w` and the
> loop-invariant `wm` alone (lines 132–133). It is **never** fetched from the data
> array, so it needs no `k+j+m/2` address. That address exists solely to load the
> *data operand* `X = output[k+j+m/2]` (lines 117–118). The two converge only at
> `t = w·X`; the *next* twiddle `w·wm` branches off **before** that convergence, so
> it never waits on the address or the load — hence the stage depth is the **max**
> of the two arrival times, not their sum. (The dependency you'd expect if the
> twiddle were a precomputed-table load `w = twiddle[k+j+m/2]` does not exist here:
> the kernel *generates* the twiddle on the fly.)

**Twiddle recurrence (II = 4)** — the binding carry on `(w_r, w_i)`:
```
load w_r ‖ load w_i
  → (w_r·wm_r, w_i·wm_i, w_r·wm_i, w_i·wm_r)        [4 muls, parallel]
  → new_w_r = (w_r·wm_r) − (w_i·wm_i)  ‖  new_w_i = (w_r·wm_i) + (w_i·wm_r)
  → store w_r ‖ store w_i                           [closes the carry]
```
`load → mul → add/sub → store` = **II = 4**. Producing the twiddle `w^(p)` used by
butterfly `j = p` takes `p` such updates, so `w^(m/2−1)` (the last butterfly's
twiddle) is ready at `1 + 4·(m/2 − 1)`. (The twiddle update *inside* the final
`j`-iter is **dead** — no later iter consumes it — but per the dead-computation
rule it is still counted as ops; it just does not lie on the path.)

**Index / induction path** — independent of `w`, this feeds the same final
butterfly:
```
load j → (k + j) → (k+j + m/2) → load output[k+j+m/2] → mul (needs w too) → t → u±t → store
```
The `j` induction is its own sequential carry (`load j → j+1 → store j`, II = 3),
and `output[k+j+m/2]` carries inline subscript arithmetic — `k+j` then `+m/2`, two
address-adds on the chain (`k` is a parallel-block constant, `m/2` is a hoisted
per-stage constant). So the load of the upper operand for butterfly `j = p` lands
at `3p + 4`.

The last butterfly's store therefore completes at
`max( w-recurrence, index-path ) + butterfly tail`:
```
stage_CP(m) = max( 2m + 1,        // w-bound:   [1 + 4·(m/2−1)] load w^(m/2−1) + (mul→t→u±t→store)
                   1.5m + 5 )      // index-bound: [3·(m/2−1)+4] load output + (mul→t→u±t→store)
```

| s | `m` | w-bound `2m+1` | index-bound `1.5m+5` | `stage_CP` | binding |
|---|-----|----------------|----------------------|-----------|---------|
| 1 | 2  | 5  | 8  | **8**  | index/address |
| 2 | 4  | 9  | 11 | **11** | index/address |
| 3 | 8  | 17 | 17 | **17** | tie |
| 4 | 16 | 33 | 29 | **33** | twiddle recurrence |

For small `m` (early stages) the **address computation dominates** — exactly the
regime the model flags for FFT — while for large `m` (late stages) the serial
twiddle rotation dominates; they cross at `m = 8`.

The copy loop (`load input[i] → store output[i]`, 2 cycles, fully unrolled) writes
the whole array that stage 1 reads, so it precedes stage 1. The per-stage
twiddle prologues (`1<<s → −2π/m → cos ‖ sin`) are array-independent and all four
can be computed ahead of their stage barriers, so they stay off the path.

```
total_cycles = 2 (copy)
             + Σ_{s=1}^{4} stage_CP(m_s)
             = 2 + 8 + 11 + 17 + 33
             = 71
```

`critical_path = 2 (copy) + max(5,8) + max(9,11) + max(17,17) + max(33,29) = 71`.
The stage sequence is binding because of the inter-stage RAW barriers; within
the last stage, the 8-deep serial twiddle rotation dominates that stage's depth.

## Op counts

### Per-phase formulas
- **Copy loop** (parallel, trip `N`): `2N` loads (`input_real[i]`, `input_imag[i]`)
  + `2N` stores (`output_real[i]`, `output_imag[i]`). Bare `[i]` → 0 address_adds,
  and parallel → the induction ops are counted as work but do not lie on the
  critical path.
- **Per butterfly** (`32` total): `6` loads (`output_real/imag[k+j+m/2]`,
  `output_real/imag[k+j]` = 4 array; `w_r`, `w_i` = 2 scalar) + `6` stores (4 array,
  2 scalar `w`) + `8` muls (4 for `t = w·X`, 4 for `w·wm`) + `4` adds (`t_i`,
  `u_r+t_r`, `u_i+t_i`, `new_w_i`) + `4` subs (`t_r`, `u_r−t_r`, `u_i−t_i`,
  `new_w_r`) + `2` address_adds (`k+j`, then `+m/2`; one offset pair fans to the
  real/imag and load/store of both accesses).
- **Per k-block** (`15` total): `2` stores (`w_r=1`, `w_i=0` init).
- **Per stage** (`4` total): `1` shift (`m = 1<<s`) + `1` shift (`m/2 ≡ m>>1`,
  hoisted) + `1` divide (`−2π/m`, float) + `1` cos + `1` sin.
- **Once per kernel**: `1` load (`N`) + `1` transcendental (`log2f(N)`, hoisted
  stage-loop bound). The outer stage loop is unrolled for latency, but the source
  loop-control work is still counted as dynamic work.
- **Copy `i` induction** (parallel, 16 iters): `16` loads + `17` stores
  (16 writebacks + `i=0` init) + `16` adds (`i++`) + `16` compares.
- **`k` induction** (parallel, 15 iters across 4 stages): `15` loads + `19`
  stores (15 writebacks + 4 `k=0` inits) + `15` adds (`k += m`) + `15` compares.
- **`j` induction** (sequential, 32 iters across 15 blocks): `32` loads + `47`
  stores (32 writebacks + 15 `j=0` inits) + `32` adds (`j++`) + `32` compares.
- **`s` induction** (source loop, 4 stages): `4` loads + `5` stores
  (4 writebacks + `s=1` init) + `4` adds (`s++`) + `4` compares.

### Algorithmic
| op | count | source |
|----|-------|--------|
| loads  | 160 | copy `input_*[i]` (32) + butterfly array reads `output_*[k+j]`, `output_*[k+j+m/2]` (4·32 = 128) |
| stores | 160 | copy `output_*[i]` (32) + butterfly array writes (4·32 = 128) |
| muls   | 256 | 8/iter (4 twiddle-rotate `w·X`, 4 `w·wm`) · 32 |
| adds   | 128 | 4/iter (`t_i`, `u_r+t_r`, `u_i+t_i`, `new_w_i`) · 32 |
| subs   | 128 | 4/iter (`t_r`, `u_r−t_r`, `u_i−t_i`, `new_w_r`) · 32 |
| divides | 4  | `−2π/m` per stage (genuine float divide) |
| shifts | 8   | `m = 1<<s` (4) + `m/2 ≡ m>>1` (4) |
| transcendentals | 9 | `cos` (4) + `sin` (4) + `log2f(N)` bound (1) |

### Overhead (loop-carried twiddle L/S, induction, address-gen, inits)
| op | count | source |
|----|-------|--------|
| loads        | 132 | `w_r`/`w_i` body loads (2·32 = 64) + `i` iv (16) + `k` iv (15) + `j` iv (32) + `s` iv (4) + `N` hoist (1) |
| stores       | 182 | `w_r`/`w_i` body stores (2·32 = 64) + `w` init per k-block (2·15 = 30) + `i` iv (17) + `k` iv (19) + `j` iv (47) + `s` iv (5) |
| adds         | 67  | `i++` (16) + `k += m` (15) + `j++` (32) + `s++` (4) |
| address_adds | 64  | `k+j` (1) + `+m/2` (1) per butterfly · 32 — inline subscript arithmetic in `output_*[k+j+m/2]` / `output_*[k+j]` |
| compares     | 67  | `i < N` (16) + `k < N` (15) + `j < m/2` (32) + `s ≤ log2(N)` (4) |

The copy `i`, block `k`, and stage `s` dimensions are materialized/unrolled for
latency, so their iterators do not extend `total_cycles`. They are still counted
as dynamic work, because op counts measure the source-level loop-control work and
are independent of scheduling.

### Totals
| op | total |
|----|------:|
| loads        | **292** |
| stores       | **342** |
| muls         | **256** |
| adds         | **195** |
| subs         | **128** |
| address_adds | **64**  |
| divides      | **4**   |
| shifts       | **8**   |
| transcendentals | **9** |
| compares     | **67**  |

Memory traffic dominates: 634 of the load+store ops, of which the loop-carried
`w_r`/`w_i` round-trips (64 loads + 94 stores incl. inits) and the induction
writebacks are pure model overhead — under a register-resident twiddle they would
collapse and the `j`-loop II would fall from 4 to 2.

## Data Dependency Graph
One butterfly (`j`-iter). The red chain is the carried twiddle recurrence
(`load w → mul → add/sub → store w`, II = 4) that closes via the back-edges into
the next iter's `w` loads. The butterfly itself (`t = w·X`, `u ± t`) is
feed-forward; the index path (`k+j`, `+m/2`) feeds the array loads in parallel and
binds the early stages. The final-iter twiddle update is computed but dead.

```mermaid
graph TD
    %% Carried twiddle (memory-backed: each named read is a 1-cycle load)
    w_r(("ld w_r<br>(carry)"))
    w_i(("ld w_i<br>(carry)"))

    %% Loop-invariant stage twiddle (computed once/stage, broadcast)
    wm_r(("wm_r"))
    wm_i(("wm_i"))

    %% Index path (k = block const, j from induction, m/2 hoisted)
    aj((" k + j "))
    ap((" + m/2 "))

    %% Array loads
    or_p(("ld output_real[k+j+m/2]"))
    oi_p(("ld output_imag[k+j+m/2]"))
    or_k(("ld output_real[k+j] = u_r"))
    oi_k(("ld output_imag[k+j] = u_i"))

    %% Twiddle products  t = w·X
    mt1((" * "))
    mt2((" * "))
    mt3((" * "))
    mt4((" * "))
    t_r((" − → t_r "))
    t_i((" + → t_i "))

    %% Output combiners  u ± t  → stores
    add_kr((" + "))
    add_ki((" + "))
    sub_pr((" − "))
    sub_pi((" − "))
    st_kr(("st output_real[k+j]"))
    st_ki(("st output_imag[k+j]"))
    st_pr(("st output_real[k+j+m/2]"))
    st_pi(("st output_imag[k+j+m/2]"))

    %% Twiddle update (the II=4 recurrence)
    mw1((" * "))
    mw2((" * "))
    mw3((" * "))
    mw4((" * "))
    new_w_r((" − → new_w_r "))
    new_w_i((" + → new_w_i "))
    st_wr(("st w_r"))
    st_wi(("st w_i"))

    %% Index path → loads
    aj --> ap
    aj --> or_k
    aj --> oi_k
    ap --> or_p
    ap --> oi_p

    %% t = w·X
    w_r --> mt1
    or_p --> mt1
    w_i --> mt2
    oi_p --> mt2
    mt1 --> t_r
    mt2 --> t_r
    w_r --> mt3
    oi_p --> mt3
    w_i --> mt4
    or_p --> mt4
    mt3 --> t_i
    mt4 --> t_i

    %% u ± t → stores
    or_k --> add_kr
    t_r --> add_kr
    oi_k --> add_ki
    t_i --> add_ki
    or_k --> sub_pr
    t_r --> sub_pr
    oi_k --> sub_pi
    t_i --> sub_pi
    add_kr --> st_kr
    add_ki --> st_ki
    sub_pr --> st_pr
    sub_pi --> st_pi

    %% w-update recurrence
    w_r --> mw1
    wm_r --> mw1
    w_i --> mw2
    wm_i --> mw2
    w_r --> mw3
    wm_i --> mw3
    w_i --> mw4
    wm_r --> mw4
    mw1 --> new_w_r
    mw2 --> new_w_r
    mw3 --> new_w_i
    mw4 --> new_w_i
    new_w_r --> st_wr
    new_w_i --> st_wi

    %% Loop-carry back-edges (RAW into next iter's w loads; dead on the last iter)
    st_wr -.->|RAW carry| w_r
    st_wi -.->|RAW carry| w_i

    %% Critical path: load w_r → mul (mw1) → new_w_r → store w_r → carry  (II = 4)
    linkStyle 29 stroke:#ff0000,stroke-width:3px;
    linkStyle 37 stroke:#ff0000,stroke-width:3px;
    linkStyle 41 stroke:#ff0000,stroke-width:3px;
    linkStyle 43 stroke:#ff0000,stroke-width:3px;
```

The two dashed back-edges are the loop-carried RAW on `(w_r, w_i)`; they set
`II_j = 4` (load → mul → sub/add → store). They are live on iters `0 … m/2 − 2`
and **dead on `j = m/2 − 1`** (the produced twiddle is never consumed). The
butterfly and the index path are feed-forward; for early stages the index path
(`k+j → +m/2 → load`) reaches the final butterfly later than the short twiddle
chain and sets the stage depth, while for stage 4 the 8-deep twiddle rotation
dominates. Above this per-butterfly graph, stage-level barriers serialize
`copy`, `s=1`, `s=2`, `s=3`, and `s=4` to protect the in-place array RAW hazards.

## CGRA-Constrained Model

The ASAP bound above assumes unlimited functional units and memory bandwidth. This section adds a second lower bound for a CGRA with **separate** arithmetic and memory-issue resources (no shared or bidirectional memory port):

- `P` — arithmetic PEs, homogeneous, one op/cycle each (divides, compares, shifts, **cos/sin** and other transcendentals included).
- `L` — load-issue lanes, one load/cycle each.
- `S` — store-issue lanes, one store/cycle each.

Every counted load consumes an `L` slot and every counted store an `S` slot — **including** the loop-carried `w_r`/`w_i` round-trips, the per-block twiddle inits, and the induction-variable accesses. Every counted non-load/store op (muls, adds, subs, `address_adds`, divides, shifts, transcendentals, compares) consumes a `P` slot. With `CP` the ASAP dependency bound, `A` the counted non-load/store ops, `LD` the loads, and `ST` the stores:

```
compute = ceil(A / P)
load    = ceil(LD / L)
store   = ceil(ST / S)
cycles  = max(CP, compute, load, store)
```

**Multi-phase composition.** This kernel is barrier-ordered: `copy → s=1 → s=2 → s=3 → s=4` must run in sequence (each stage RAW-depends on the previous stage's in-place writes). With ordered phases the lower bound is the **sum** of each phase's `max(CP_phase, compute_phase, load_phase, store_phase)`, not the kernel-wide `max`. The per-phase op counts below partition the eval's totals exactly (`ΣA = 730` + 1 hoisted `log2f` = 731; `ΣLD = 291` + 1 `N` load = 292; `ΣST = 341` + 1 stage-loop init store = 342); the residual once-per-kernel ops overlap the copy phase and add no cycles.

| phase | CP | A | LD | ST | compute=⌈A/36⌉ | load=⌈LD/12⌉ | store=⌈ST/12⌉ | phase cycles | binding |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---------|
| copy  | 2  | 32  | 48 | 49 | 1 | 4 | 5 | **5**  | store |
| s = 1 | 8  | 183 | 65 | 90 | 6 | 6 | 8 | **8**  | dependency = store |
| s = 2 | 11 | 175 | 61 | 74 | 5 | 6 | 7 | **11** | dependency |
| s = 3 | 17 | 171 | 59 | 66 | 5 | 5 | 6 | **17** | dependency |
| s = 4 | 33 | 169 | 58 | 62 | 5 | 5 | 6 | **33** | dependency (twiddle recurrence) |

**6×6 example (`P = 36`, `L = 12`, `S = 12`).**
```
cycles = 5 (copy) + 8 (s=1) + 11 (s=2) + 17 (s=3) + 33 (s=4) = 74
```
(For reference, the dependency-only phase sum is `2 + 8 + 11 + 17 + 33 = 71` — the existing ASAP `total_cycles`. A naive kernel-wide aggregate `max(CP, ⌈731/36⌉, ⌈292/12⌉, ⌈342/12⌉) = max(71, 21, 25, 29) = 71` would *under*-count, because the stage barriers forbid overlapping the phases.)

**Bottleneck: dependency-bound, with a store-bound copy phase.** The stage chain (twiddle recurrence + inter-stage RAW barriers) supplies 69 of the 74 cycles, and the 8-deep serial twiddle rotation makes stage 4 alone cost 33. Finite resources add just +3 over ASAP: the copy phase is store-bound (`store = 5 > CP = 2`, because the 16-point copy plus its induction writes need ≥ 5 store-issue cycles on 12 lanes), and stage 1's store bound (8) merely ties its dependency depth. Widening `P`/`L`/`S` cannot shrink the stage recurrences, so the kernel stays latency-limited.
