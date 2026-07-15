# Bisection Step Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps only the bisection_step-specific
setup, helper output, and recommendation.

Kernel: `tests/app/bisection_step/bisection_step.cpp` — one bisection root-finding
step: `c = (a+b)*0.5`, then pick `[a,c]` if `f(a)*f(c) < 0` else `[c,b]`.

Current source pragma:

```cpp
LOOM_PARALLEL()
LOOM_UNROLL()
for (uint32_t i = 0; i < N; i++) {
    float c = (input_a[i] + input_b[i]) * 0.5f;

    if (input_fa[i] * input_fc[i] < 0.0f) {
        output_a[i] = input_a[i];
        output_b[i] = c;
    } else {
        output_a[i] = c;
        output_b[i] = input_b[i];
    }
}
```

Model: the lane-aware + vector-coalescing Loom-pragma DSE in
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py bisection_step --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 24
```

## Why P and U differ

The `i` loop is dependency-parallel: each iteration reads four distinct inputs
(`input_a/b/fa/fc[i]`) and writes two distinct outputs (`output_a[i]`,
`output_b[i]`), with no carried scalar. The algorithmic chain is the same for
every split (`load → add → mul → store`), so `P` vs `U` only changes memory shape
and loop-control overhead. The `if (fa*fc < 0)` is counted taken-arm-only (no
predication credit), the arithmetic uses a global pool, and both arms write the
**same two** output addresses — so the branch does **not** separate `P` from `U`.

All six arrays are contiguous over `i`. Unroll exposes adjacent accesses inside
one worker, so they coalesce into 256-bit vector ops. Parallel workers stride
across partitions and do not coalesce with each other; they also carry separate
iterators. At fixed product `P*U = 32`, this is why `P16U2` keeps 16 iterators
while `P1U32` keeps one.

## Setup

- Resource config: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`); vector width
  `V = 4` (one 256-bit vector op per four 64-bit elements)
- Trip count: `64`; distribution `contiguous`
- Per-iteration **algorithmic** demand: `L = 4` (`input_a/b/fa/fc`), `S = 2`
  (`output_a/b`), `P = 4` (add `a+b`, mul `*0.5 → c`, mul `fa*fc`, cmp `< 0`);
  `CP = 4` (`load → add → mul → store`). These four loads are **recurring** — they
  scale with exposure.
- Per-**worker** control: one iterator load / add / store / compare per worker
  per wave (not per iteration). The iterator load is recurring.
- **Invariant** loads: `N` is hoisted once per chunk (count independent of
  exposure), so it is amortized — loaded once and held — and does not count toward
  the recurring lane exposure `aL` or the binding load term.
- Coalescing: the four contiguous input loads and the two contiguous output stores
  coalesce `4:1` under unroll; the induction load/store does not coalesce.
- Full-trip counts (full unroll, fully coalesced): `A = 258` (`256` algorithmic +
  `2` residual control), `LD_rec = 65` recurring (`64` coalesced array vecs + a
  residual iterator), `LD_inv = 1` (`N`, loaded once), `LD_eff = 66`, `ST = 33`
  (`32` coalesced + residual), `CP = 4`, giving
  `compute = ceil(258/36) = 8`, `load = ceil(LD_rec/12) = ceil(65/12) = 6`,
  `store = ceil(33/12) = 3`.
- `absolute_cgra_lb = 8` — the full-trip, **fully-coalesced, invariant-amortized**
  aggregate (`max(4, 8, 6, 3) = 8`), and the **only** lower bound. The binding
  class is **compute (`P`)**. The four algorithmic ops per element dominate once
  the 4-input/2-output streams coalesce and the iterator amortizes: `LD_rec = 65`
  so the load term is `6`, below the arithmetic term `8`.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): bisection_step  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[64,parallel]`; All six arrays (input_a/b/fa/fc, output_a/b) are contiguous over i. This is axpy-shaped: LOOM_UNROLL(i) beats LOOM_PARALLEL(i) two ways -- it coalesces the 4 input loads and 2 output stores into vector ops (bounded by V=4) and it amortizes the iterator (charged once per worker, keeps paying past U=V). The if/else is counted taken-arm-only (no predication credit) and the compute is a global pool, so the branch does not separate P from U. Load-heavy shape (4 input streams to 2 output streams), but compute-bound after coalescing + control amortization. Full-trip counts are `A=258`, `LD_rec=65`, `LD_eff=66`, `ST=33`, and `CP=4`, giving the only lower bound, `absolute_cgra_lb=8=max(CP 4, compute 8, load 6, store 3)`, with compute pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
         i:P8U4                        8  12  12     41    32     2     4       8      12 resource-bound  100/100/50
         i:P4U8                        4  12  12     37    32     2     4       8      12 resource-bound   100/75/50
         i:P2U16                       2  12  12     35    32     2     4       8      12 resource-bound   100/75/50
K        i:P1U32                       1  12  12     34    32     2     4       8      12 resource-bound   100/75/50
o        i:P16U4                      16  12  12     81    64     1     8       8       9 resource-bound   100/88/50
o        i:P8U8                        8  12  12     73    64     1     8       8       9 resource-bound   100/75/50
o        i:P4U16                       4  12  12     69    64     1     8       8       9 resource-bound   100/75/38
o        i:P2U32                       2  12  12     67    64     1     8       8       9 resource-bound   100/75/38
o        i:P1U64                       1  12  12     66    64     1     8       8      10 resource-bound   100/75/38
         i:P16U2                      16  12  12     81    32     2     7      14      18 resource-bound   71/100/57
o        i:P32U2                      32  12  12    161    64     1    14      14      16 resource-bound   64/100/57
         i:P8U2                        8  12  12     41    16     4     4      16      24 resource-bound   75/100/50
b        i:P4U4                        4  12  12     21    16     4     4      16      16 latency-bound     50/50/25
b        i:P2U8                        2  12  10     19    16     4     4      16      16 latency-bound     50/50/25
b        i:P1U16                       1  12   9     18    16     4     4      16      16 latency-bound     50/50/25
o        i:P64U1                      64  12  12    321    64     1    27      27      29 resource-bound   41/100/59
         i:P16U1                      16  12  12     81    16     4     7      28      36 resource-bound   43/100/57
         i:P32U1                      32  12  12    161    32     2    14      28      32 resource-bound   43/100/57
         i:P8U1                        8  12  12     41     8     8     4      32      48 resource-bound   50/100/50
b        i:P4U2                        4  12  12     21     8     8     4      32      32 latency-bound     50/50/25
b        i:P2U4                        2  10   6     11     8     8     4      32      32 latency-bound     25/25/25
b        i:P1U8                        1   9   5     10     8     8     4      32      32 latency-bound     25/25/25
b        i:P4U1                        4  12  12     21     4    16     4      64      64 latency-bound     25/50/25
b        i:P2U2                        2  10   6     11     4    16     4      64      64 latency-bound     25/25/25
... (4 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: i:P1U32  -> exposure=32, pragma_agg=8 (1.00x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 32 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P32U1             160    161    96      28 3.50x slower (parallel: extra iterators + strided, no coalesce)
  P16U2              80     81    48      14 1.75x slower (parallel: extra iterators + strided, no coalesce)
  P8U4              40     41    24       8 best
  P4U8              36     37    20       8 best
  P2U16             34     35    18       8 best
  P1U32             33     34    17       8 best

4x4 recommendation: i:P1U16.
8x8 recommendation: i:P1U64.
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Reading the fixed-product block

The fixed-product block isolates the split choice: all rows expose 32 iterations.
`P8U4`, `P4U8`, `P2U16`, and `P1U32` all have `U >= V = 4`, so coalescing is
already saturated; the residual `LD_rec` drop (`40 -> 36 -> 34 -> 33`) is pure
control amortization (fewer workers → fewer iterator loads), and all four tie at
`p_agg = 8` (the floor). `P16U2` is the outlier at `p_agg = 14` (`1.75×` slower):
with `U = 2 < V`, each 256-bit vector is only half-filled (partial `2:1`
coalescing → `64` input vecs instead of `32`) **and** it carries 16 iterators, so
`LD_rec = 80` and the wave becomes load-bound (`util L = 100%`). This is a
stronger penalty than axpy's fixed-product block, where every candidate had
`U >= V` and the slow parallel row lost on iterator count alone.

## Recommendation

**`i:P1U32` is the recommended knee (`K`)**: exposure `32`, `p_agg = 8`
(`1.00×` the `8`-cycle floor), resource-bound on compute. The helper enumerates
only **power-of-two** factors, so `32` is the smallest *enumerated* exposure at or
above the compute-saturation crossover — the analytical point where the arithmetic
term first reaches `CP = 4` is near exposure `27`, between the enumerated `16` and
`32`. `i:P1U32` is the fewest-workers representative among the exposure-`32` ties.

The `b`-flagged rows below it are latency-bound: their waves finish on `CP = 4`
before they fill the machine. Not *every* row below the knee is latency-bound,
though — a few low-`U` rows (e.g. `i:P8U2` and `i:P16U1`, both at exposure 16, and
`i:P8U1` at exposure 8) are already resource-bound, but only because they waste load
lanes on half-filled/strided vectors and pin `L` at `100%` without coalescing. That
is lane-starvation, not the knee, which is why the recommender skips them for the
fully-coalesced `P1U32`. Rows above the knee (`o`, exposure 64) only add exposure
and mapping pressure — `p_agg` stays pinned at the `8`-cycle floor, so there is no
steady-state gain. The current source pragma leaves both `LOOM_PARALLEL()` and
`LOOM_UNROLL()` unspecified; setting `U = 32` (i.e. `P1U32`) reaches the knee while
coalescing all six streams and paying a single iterator.

## Comparing against measured DFG simulator cycles

Use the shared comparison rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For bisection_step, the DSE floor (`8`) sits **well below** the CGRA-constrained
Metric-1 aggregate (`27`, load-bound) in the main `bisection_step_eval.md`. That
scalar model charges per-iteration induction loads/stores and does not coalesce,
so `LD = 321` and `load = ceil(321/12) = 27` dominate. This DSE coalesces the four
input and two output streams `4:1` (`LD_rec = 65`) and amortizes the iterator to
one per worker, dropping the load term to `6` and making the kernel
**compute-bound at 8**. The gap is exactly the vector-coalescing +
control-amortization credit this model takes and the scalar ASAP aggregate does
not.
