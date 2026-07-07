# AXPY Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps only the axpy-specific
setup, helper output, and recommendation.

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

Model: the lane-aware + vector-coalescing Loom-pragma DSE in
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py axpy --config 6x6 --max-parallel 16 --max-unroll 64
```

## Why P and U differ

`compute_loop` is dependency-parallel: each iteration writes a distinct
`output_y[i]`, and `alpha` is read-only. Axpy's algorithmic chain is the same for
every split (`input_x/input_y` loads -> multiply -> add -> `output_y` store), so
`P` vs `U` only changes memory shape and loop-control overhead.

All three array accesses are contiguous over `i`. Unroll exposes adjacent
accesses inside one worker, so they can coalesce into 256-bit vector ops.
Parallel workers stride across partitions and do not coalesce with each other;
they also carry separate iterators. At fixed product `P*U = 64`, this is why
`P16U4` has 16 iterators while `P1U64` has one.

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
  coalesce `4:1` under unroll; the induction load/store does not coalesce.
- Full-trip counts (full unroll, fully coalesced): `A = 514` (`512` algorithmic +
  `2` residual control), `LD_rec = 129` recurring (`128` coalesced array vecs + a
  residual iterator), `LD_inv = 2` (`alpha`/`N`, loaded once), `LD_eff = 131`,
  `ST = 65` (`64` coalesced + residual), `CP = 4`, giving
  `compute = ceil(514/36) = 15`, `load = ceil(LD_rec/12) = ceil(129/12) = 11`,
  `store = ceil(65/12) = 6`.
- `absolute_cgra_lb = 15` — the full-trip, **fully-coalesced, invariant-amortized**
  aggregate (`max(4, 15, 11, 6) = 15`), and the **only** lower bound. The binding
  class is **compute (`P`)**. This is below the old scalar induction-heavy floor
  because full unroll leaves one residual iterator and holds the two invariant
  `alpha`/`N` loads once; `LD_rec = 129`, so the load term is `11`, below the
  arithmetic term `15`.

## Results (`--max-parallel 16 --max-unroll 64`)

```text
# Loom pragma DSE (lane-aware + vector coalescing): axpy  (6x6)

loop nest: i[256, parallel]; input_x/input_y/output_y are contiguous over i.
unroll coalesces adjacent accesses; parallel workers stride and carry separate iterators.
absolute_cgra_lb = 15 = max(CP 4, compute 15, load 11, store 6); it is the only lower bound.
full-trip counts: A=514 LD_rec=129 LD_eff=131 ST=65 CP=4; binding class = P.
p_agg and sched are wave-serialized estimates; shared rules are in ../DSE_rules.md.

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

## Reading the fixed-product block

The fixed-product block in the helper output isolates the split choice. All rows
there expose 64 iterations, and all have `U >= V = 4`, so coalescing is already
saturated. The remaining `LD_rec` drop (`48 -> 40 -> 36 -> 34 -> 33`) is control
amortization: fewer workers means fewer iterator loads. `P16U4` therefore loses
to `P1U64` even though both are fully coalesced.

## Recommendation

**`i:P1U64` is the recommended knee (`K`)**: exposure `64`, `p_agg = 16`
(`1.07×` the `15`-cycle floor), resource-bound. It is the **smallest exposure at
which the binding class first becomes resource-bound** (`E_sat`), and the
representative with the fewest workers among the exposure-64 ties.

Rows below it are latency-bound (`b`): their waves finish on `CP = 4` before they
fill the machine. Rows above it (`o`) only move `p_agg` toward the floor through
wave-ceiling effects; they add exposure and mapping pressure without improving
the steady-state rate. The current source pragma `P4U1` is far below the knee:
without unroll it cannot coalesce, pays four iterators, and has `p_agg = 256`.

## Comparing against measured DFG simulator cycles

Use the shared comparison rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For axpy, the DSE floor is lower than the scalar Metric-1 aggregate in the main
eval because this model credits vector coalescing and control amortization.
