# AXPY Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps the axpy-specific setup,
modeling rationale, helper output, and recommendation.

Kernel: `tests/app/axpy/axpy.cpp` — loop `compute_loop`

Current source pragma:

```cpp
compute_loop:
LOOM_PARALLEL(4, contiguous)
LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
for (uint32_t i = 0; i < N; i++) {
    output_y[i] = alpha * input_x[i] + input_y[i];
}
```

This file selects `LOOM_PARALLEL(P)` / `LOOM_UNROLL(U)` under the shared
lane-aware + vector-coalescing model, which is defined by
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma
Design-Space Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate (sweep parallel up to 16 and unroll up to 64 so the split can reach
the saturation knee by pure unroll):

```bash
python3 tests/scripts/loom_dse.py axpy --config 6x6 --max-parallel 16 --max-unroll 64
```

## Why P and U differ

`compute_loop` is dependency-parallel: each iteration writes a distinct
`output_y[i]`, and `alpha` is read-only. Axpy's algorithmic chain is the same for
every split (`input_x/input_y` loads -> multiply -> add -> `output_y` store), so
the intended math and `CP` do not distinguish `LOOM_PARALLEL` from `LOOM_UNROLL`.

The split matters for axpy because all three array accesses are **contiguous** over
`i`. A worker's unrolled `input_x`, `input_y`, and `output_y` accesses can
coalesce into 256-bit vector ops, while parallel workers stride across partitions
and do not coalesce with each other. At the same fixed product `P*U`, the
parallel-heavy split also carries more iterators: `P16U4` has sixteen worker
iterators, while `P1U64` has one.

For axpy, the fixed-product table below shows that split directly: `P16U4` is
already coalesced, but it still loses to unroll-heavy splits because it carries
more worker iterators.

## Setup

- Resource config: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`); vector width
  `V = 4` (one 256-bit vector op per four 64-bit elements)
- Trip count: `256`; distribution `contiguous`
- Per-iteration **algorithmic** demand: `L = 2` (`input_x`, `input_y`), `S = 1`
  (`output_y`), `P = 2` (mul, add); `CP = 4` (`load → mul → add → store`). These
  loads are **recurring** — they scale with exposure.
- Per-**worker** control: one iterator load / add / store / compare per worker
  per wave (not per iteration). The iterator load is recurring (it scales with the
  worker count).
- **Invariant** loads: `alpha` and `N` are hoisted once per chunk (count
  independent of exposure), so they are amortized — loaded once and held — and do
  not count toward the recurring lane exposure `aL` or the binding load term.
- Coalescing: the two contiguous array loads and the contiguous output store
  coalesce `4:1` under unroll; the induction load/store does not coalesce, but it
  is now charged per worker (not per element), so it nearly vanishes at high `U`.
- Full-trip counts (full unroll, fully coalesced): `A = 514` (`512` algorithmic +
  `2` residual control), `LD_rec = 129` recurring (`128` coalesced array vecs + a
  residual iterator), `LD_inv = 2` (`alpha`/`N`, loaded once), `LD_eff = 131`,
  `ST = 65` (`64` coalesced + residual), `CP = 4`, giving
  `compute = ceil(514/36) = 15`, `load = ceil(LD_rec/12) = ceil(129/12) = 11`,
  `store = ceil(65/12) = 6`.
- `absolute_cgra_lb = 15` — the full-trip, **fully-coalesced, invariant-amortized**
  aggregate (`max(4, 15, 11, 6) = 15`), and the **only** lower bound. The binding
  class is **compute (`P`)**. Two amortizations shape it: control amortization
  collapsed the per-iteration induction stream (`256` `i`-loads + `256` `i`-stores)
  to a single residual iterator, and the two `alpha`/`N` invariant loads are held
  once rather than counted per cycle — so `LD_rec = 129` and the load term is `11`,
  below the arithmetic term `15`. For axpy the invariant split barely moves the
  number (only `2` loads), but it is the same rule that reshapes gemv, whose whole
  `x` vector is invariant of the row index.

## Results (`--max-parallel 16 --max-unroll 64`)

```text
# Loom pragma DSE (lane-aware + vector coalescing): axpy  (6x6)

loop nest (outer->inner): i[256,parallel]
coalescing: input_x/input_y/output_y are contiguous over i. Two axes both favor LOOM_UNROLL over LOOM_PARALLEL at a fixed product: (1) coalescing -- a worker's U adjacent accesses fuse into ceil(U/V) vector ops while parallel strides across workers (bounded by V=4, gone once U>=V); (2) control amortization -- the iterator is charged once per worker, so fewer workers (more unroll) means fewer i-loads/adds/stores (keeps paying past U=V). So unroll strictly beats parallel at fixed product.

absolute_cgra_lb = 15  (full-trip, fully-coalesced, invariant-amortized aggregate over full lanes L=12,S=12; the ONLY lower bound)
full-trip counts: A=514 LD_rec=129 LD_eff=131 ST=65 CP=4 | compute=15 load=11 store=6   (load term = ceil(LD_rec/L); invariants amortized)
binding class (full trip) = P   (P_pe=36, L=12, S=12; V=4 64-bit elems/vec)

Only absolute_cgra_lb is a lower bound. pragma_agg / sched_est assume waves do NOT overlap and sit at or above it.
aL = active load lanes = min(recurring loads, L): the recurring loop loads set the lane exposure and the binding load term. LD_eff = recurring + one-time invariant loads (total traffic); invariant loads (loaded once and held) are amortized out of the binding term.
Algorithmic arith/CP is a global pool (P and U tie there). P and U separate on TWO axes, both favoring LOOM_UNROLL: (1) control amortization -- unroll shares one iterator across U bodies, so control ops scale as trip/U (parallel keeps an iterator per worker); (2) vector coalescing of contiguous accesses (bounded by V, gone once U>=V). Sequential carries keep per-iter control on CP.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
o        i:P8U32                       8  12  12    138   256     1    15      15      17 resource-bound   100/80/40
o        i:P4U64                       4  12  12    134   256     1    15      15      17 resource-bound   100/73/40
         i:P8U8                        8  12  12     42    64     4     4      16      24 resource-bound  100/100/50
         i:P4U16                       4  12  12     38    64     4     4      16      24 resource-bound   100/75/50
         i:P2U32                       2  12  12     36    64     4     4      16      24 resource-bound   100/75/50
K        i:P1U64                       1  12  12     35    64     4     4      16      24 resource-bound   100/75/50
o        i:P16U8                      16  12  12     82   128     2     8      16      20 resource-bound   100/88/50
o        i:P8U16                       8  12  12     74   128     2     8      16      20 resource-bound   100/75/50
o        i:P4U32                       4  12  12     70   128     2     8      16      20 resource-bound   100/75/38
o        i:P2U64                       2  12  12     68   128     2     8      16      20 resource-bound   100/75/38
o        i:P16U16                     16  12  12    146   256     1    16      16      17 resource-bound   100/75/44
         i:P16U4                      16  12  12     50    64     4     5      20      28 resource-bound   100/80/60
         i:P16U2                      16  12  12     50    32     8     4      32      56 resource-bound   75/100/75
b        i:P8U4                        8  12  12     26    32     8     4      32      40 latency-bound     75/50/50
b        i:P4U8                        4  12  12     22    32     8     4      32      32 latency-bound     50/50/25
b        i:P2U16                       2  12  10     20    32     8     4      32      32 latency-bound     50/50/25
b        i:P1U32                       1  12   9     19    32     8     4      32      32 latency-bound     50/50/25
         i:P16U1                      16  12  12     50    16    16     4      64     112 resource-bound   50/100/75
b        i:P8U2                        8  12  12     26    16    16     4      64      80 latency-bound     50/50/50
b        i:P4U4                        4  12   8     14    16    16     4      64      64 latency-bound     50/25/25
b        i:P2U8                        2  10   6     12    16    16     4      64      64 latency-bound     25/25/25
b        i:P1U16                       1   9   5     11    16    16     4      64      64 latency-bound     25/25/25
b        i:P8U1                        8  12  12     26     8    32     4     128     160 latency-bound     25/50/50
b        i:P4U2                        4  12   8     14     8    32     4     128     128 latency-bound     25/25/25
b        i:P2U4                        2   6   4      8     8    32     4     128     128 latency-bound     25/25/25
b        i:P1U8                        1   5   3      7     8    32     4     128     128 latency-bound     25/25/25
b        i:P4U1                        4  12   8     14     4    64     4     256     256 latency-bound     25/25/25
b        i:P2U2                        2   6   4      8     4    64     4     256     256 latency-bound     25/25/25
b        i:P1U4                        1   3   2      5     4    64     4     256     256 latency-bound     25/25/25
b        i:P2U1                        2   6   4      8     2   128     4     512     512 latency-bound     25/25/25
b        i:P1U2                        1   3   2      5     2   128     4     512     512 latency-bound     25/25/25
b        i:P1U1                        1   3   2      5     1   256     4    1024    1024 latency-bound     25/25/25

RECOMMENDED: i:P1U64  -> exposure=64, pragma_agg=16 (1.07x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 64 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P16U4              48     50    32      20 1.25x slower (parallel: extra iterators + strided, no coalesce)
  P8U8              40     42    24      16 best
  P4U16             36     38    20      16 best
  P2U32             34     36    18      16 best
  P1U64             33     35    17      16 best
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## The P-vs-U distinction, made concrete

At a **fixed product** `P·U = 64`, the split alone changes the estimate, and the
change now has **two** components:

| split | LD_rec | LD_eff | ST | p_agg | reading |
|-------|-------:|-------:|---:|------:|---------|
| `P16U4` | 48 | 50 | 32 | 20 | 1.25× slower — 16 iterators + strided (`U=4` still coalesces) |
| `P8U8`  | 40 | 42 | 24 | 16 | best — 8 iterators, fully coalesced |
| `P4U16` | 36 | 38 | 20 | 16 | best — 4 iterators |
| `P2U32` | 34 | 36 | 18 | 16 | best — 2 iterators |
| `P1U64` | 33 | 35 | 17 | 16 | best — **1 iterator**, minimal control |

`LD_rec` is the recurring (binding) load count; `LD_eff = LD_rec + 2` adds the two
one-time `alpha`/`N` invariants (constant across the split, amortized out of the
binding). Read the `LD_rec` column top to bottom: `48 → 40 → 36 → 34 → 33`. All
five rows have `U ≥ V = 4`, so **coalescing is already saturated** — the array-load
term is the same for all of them. The remaining `LD_rec` slide is **pure control
amortization**: each halving of the worker count `P_tot` removes that many
per-worker iterator loads. `P16U4` carries 16 iterators, `P1U64` carries one. That
is why `P16U4` loses even though it is fully coalesced: unroll's advantage does
**not** vanish at `U = V` once control amortization is in the model. The `p_agg`
gap (`20` vs `16`) appears because `P16U4`'s extra iterators push its recurring
load term over the compute term at this exposure.

## Recommendation

**`i:P1U64` is the recommended knee (`K`)**: exposure `64`, `p_agg = 16`
(`1.07×` the `15`-cycle floor), resource-bound. It is the **smallest exposure at
which the binding class first becomes resource-bound** (`E_sat`), and the
representative with the fewest workers (least control) among the exposure-64
factorings. Walking the table:

- The current source pragma **`P=4, U=1`** is *bandwidth-starved* (`b`): with no
  unroll there is nothing to coalesce and four separate iterators to pay, its wave
  is latency-bound, and `p_agg` sits at `256` (`~17×` the floor).
- Candidates below the knee (`b`) are **latency-bound**: the critical path
  (`CP = 4`) drains before the (now-small) load/compute streams fill their lanes.
  Adding exposure here strictly improves throughput.
- **`i:P1U64`** is the first split where the fully-amortized, fully-coalesced work
  saturates the binding class every wave (`util_P = 100`, resource-bound) — the
  diminishing-returns knee. It ties in `p_agg` with `P8U8`, `P4U16`, and `P2U32`
  (all `16`); the tool reports the pure-unroll representative because it carries
  the fewest iterators, so it is the cleanest and lowest-control choice.
- Rows above the knee (`o`, e.g. `P8U32`, `P16U16`) shave `p_agg` only through
  per-wave ceiling rounding down toward the floor of `15`; the steady-state rate
  is already at the floor, so they trade area and transient backlog for no real
  throughput gain and are flagged **oversubscribed**.
- Note that at the same exposure `64`, `P16U4` reaches `p_agg = 20` versus
  `P1U64`'s `16`: the difference is entirely the 15 extra iterators `P16U4`
  carries. Pushing exposure onto **unroll** rather than parallel is what earns
  the better knee.

## Comparing against measured DFG simulator cycles

Use the shared comparison rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For axpy, the DSE `absolute_cgra_lb` is `15` because it credits both vector
coalescing and control-overhead amortization. This sits well below the scalar
Metric-1 aggregate reported in the kernel's main `## CGRA-Constrained Model`
section, which charges induction per iteration and models neither vector memory
ops nor control amortization. That gap is expected for this memory-light kernel:
the induction stream dominates the scalar aggregate, and full unrolling amortizes
it away.
