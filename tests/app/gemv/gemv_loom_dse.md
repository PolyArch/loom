# gemv Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps the gemv-specific setup,
modeling rationale, helper output, and recommendation.

Kernel: `tests/app/gemv/gemv.cpp` — `output_y[i] = alpha·Σ_j A[i,j]·x[j] + beta·input_y[i]`

Current source pragma:

```cpp
LOOM_ACCEL()
void gemv_dsa(const uint32_t alpha,
              LOOM_MEMORY_BANK(4, block) LOOM_STREAM const uint32_t* A,
              LOOM_STREAM const uint32_t* x, const uint32_t beta, ...) {
    for (uint32_t i = 0; i < M; i++) {        // rows: parallel
        uint32_t sum = 0;
        for (uint32_t j = 0; j < N; j++)      // cols: reduction
            sum += A[i * N + j] * x[j];
        output_y[i] = alpha * sum + beta * input_y[i];
    }
}
```

This file selects `LOOM_PARALLEL(P)` / `LOOM_UNROLL(U)` under the shared
lane-aware + vector-coalescing model, which is defined by
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma
Design-Space Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).
The `LOOM_MEMORY_BANK(4, block)` on `A` is left in the source but **ignored** by
this provisional DSE (no banks, no bank caps).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py gemv --config 6x6 --max-parallel 8 --top 14
```

## Why P and U differ

`gemv` is a nested kernel: an outer **parallel** row loop `i` and an inner
**reduction** column loop `j`. The two pragmas separate only where an axis is
non-inert, and gemv makes most axes inert:

- **The dot-product path is P/U-symmetric.** `A[i][j]` and `x[j]` are contiguous
  over `j`, and `j` is a **fully-consumed reduction** (tree-reduced), so they
  coalesce identically for any split and the `j`-loop carries no control.
  Parallelizing or unrolling `j` only reshapes the merge tree — it changes no lane
  demand (the inner level is inert).
- **On the row level `i`, unroll has a real but modest edge.** `LOOM_UNROLL(i)`
  coalesces the contiguous `input_y[i]`/`output_y[i]` accesses (parallel strides)
  and amortizes the row iterator (charged once per worker). But the dominant
  `A`-load term is split-symmetric, so the edge only surfaces on the (non-binding)
  store and iterator terms.

## Setup

- Resource config: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`); `V = 4`. `M = N = 64`.
- Per-**row** algorithmic demand: `N` `A`-loads and the shared `x`-loads feed `N`
  multiplies and an `N`-leaf reduction (`N−1` adds), then `alpha·sum`,
  `beta·input_y[i]`, and a final add. The `A`/`x` loads coalesce over `j`.
- **Invariant loads (the gemv-defining feature):** the whole `x` vector is
  **invariant of the row index `i`** — loaded once per chunk and reused by every
  row — plus the `alpha`/`beta`/`M`/`N` scalars. At full trip these total
  `LD_inv = 20` (`16` `x`-vecs + `4` scalars). They are amortized out of the
  binding load term and appear only in `LD_eff`, so gemv is the clearest case of a
  large `LD_eff − LD_rec` gap.
- Full-trip counts: `A = 8322`, `LD_rec = 1041` recurring (`1024` `A`-load vecs +
  `16` `input_y` vecs + `1` residual iterator), `LD_inv = 20`, `LD_eff = 1061`,
  `ST = 17`, `CP = 11`, giving `compute = ceil(8322/36) = 232`,
  `load = ceil(LD_rec/12) = ceil(1041/12) = 87`, `store = 2`.
- `absolute_cgra_lb = 232` — the full-trip, **fully-coalesced, invariant-amortized**
  aggregate (`max(11, 232, 87, 2) = 232`), and the **only** lower bound. The
  binding class is **compute (`P`)**: gemv does `~2` arithmetic ops per `A`-load
  (multiply + reduction add), so once the per-iteration induction is amortized and
  `x` is held once, the arithmetic term dominates the (now much smaller) load term.

## Results (`--top 14`)

```text
# Loom pragma DSE (lane-aware + vector coalescing): gemv  (6x6)

loop nest: i[64, parallel] (rows), j[64, reduction] (cols); A[i][j]/x[j] contiguous over j.
j is fully consumed (tree-reduced) -> the dot-product path is P/U-symmetric; the row-i edge is coalescing y[i] + amortizing the row iterator.
absolute_cgra_lb = 232 = max(CP 11, compute 232, load 87, store 2); it is the only lower bound.
full-trip counts: A=8322 LD_rec=1041 LD_eff=1061 ST=17 CP=11; binding class = P (compute).
p_agg and sched are wave-serialized estimates; shared rules are in ../DSE_rules.md.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
o        i:P2U4 j:P1U1  (+15 eq)       2  12   4    152   512     8    29     232     280 resource-bound    100/38/3
o        i:P1U8 j:P1U1  (+15 eq)       1  12   3    151   512     8    29     232     280 resource-bound    100/38/3
o        i:P4U4 j:P1U1  (+15 eq)       4  12   8    284  1024     4    58     232     252 resource-bound    100/38/2
o        i:P2U8 j:P1U1  (+15 eq)       2  12   6    282  1024     4    58     232     252 resource-bound    100/38/2
o        i:P8U4 j:P1U1  (+15 eq)       8  12  12    548  2048     2   116     232     238 resource-bound    100/38/2
o        i:P4U8 j:P1U1  (+15 eq)       4  12  12    544  2048     2   116     232     238 resource-bound    100/38/1
o        i:P8U8 j:P1U1  (+15 eq)       8  12  12   1068  4096     1   232     232     235 resource-bound    100/38/1
o        i:P8U2 j:P1U1  (+15 eq)       8  12  12    292  1024     4    59     236     252 resource-bound    100/39/3
         i:P4U1 j:P1U1  (+15 eq)       4  12   8     92   256    16    15     240     352 resource-bound    100/40/7
         i:P2U2 j:P1U1  (+15 eq)       2  12   4     88   256    16    15     240     352 resource-bound    100/40/7
K        i:P1U4 j:P1U1  (+15 eq)       1  12   2     86   256    16    15     240     352 resource-bound    100/40/7
o        i:P8U1 j:P1U1  (+15 eq)       8  12  12    164   512     8    30     240     280 resource-bound    100/40/7
o        i:P4U2 j:P1U1  (+15 eq)       4  12   8    156   512     8    30     240     280 resource-bound    100/40/3
b        i:P2U1 j:P1U1  (+15 eq)       2  12   4     56   128    32    11     352     512 latency-bound      73/27/9
... (2 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: i:P1U4 j:P1U1  -> exposure=256, pragma_agg=240 (1.03x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 8 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P8U1             144    164    16     240 1.03x slower (parallel: extra iterators + strided, no coalesce)
  P4U2             136    156     8     240 1.03x slower (parallel: extra iterators + strided, no coalesce)
  P2U4             132    152     4     232 best
  P1U8             131    151     3     232 best
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## The P-vs-U distinction, made concrete

At a **fixed product** `P·U = 8` on the row level (inner `j` at `P1U1`), the split
moves only the non-binding terms:

| split | LD_rec | LD_eff | ST | p_agg | reading |
|-------|-------:|-------:|---:|------:|---------|
| `P8U1` | 144 | 164 | 16 | 240 | 1.03× slower — 8 row iterators + strided stores |
| `P4U2` | 136 | 156 | 8 | 240 | 1.03× slower |
| `P2U4` | 132 | 152 | 4 | 232 | best — coalesced stores, 2 iterators |
| `P1U8` | 131 | 151 | 3 | 232 | best — **1 iterator**, coalesced stores |

`LD_rec` slides `144 → 131` as unroll amortizes the row iterator and coalesces the
output stores; `LD_eff = LD_rec + 20` carries the constant `x`+scalar invariant
throughout (its `20` slots never enter the binding term). But because **compute
binds** (`232`), that load/store slide barely moves `p_agg` (`240 → 232`, only
`1.03×`). gemv is therefore **largely P/U-symmetric**: the unroll edge is real but
second-order behind the arithmetic term.

## Recommendation

**`i:P1U4 j:P1U1` is the recommended knee (`K`)**: exposure `256` (4 rows × their
full 64-tap reduction), `p_agg = 240` (`1.03×` the `232` floor), resource-bound on
compute. It is `E_sat` — the smallest exposure at which the arithmetic term
overtakes `CP = 11`. `i:P2U2` and `i:P4U1` tie it exactly at `240`; the tool
reports the pure-unroll representative because it carries the fewest row iterators.

- Below the knee, `i:P2U1` (exposure `128`) is **latency-bound** (`b`): the merge
  tree drains before compute fills its lanes, so resources idle.
- Above the knee (`o`, e.g. `i:P2U4`, `i:P8U8`) `p_agg` only creeps toward the
  `232` floor through per-wave rounding while transient backlog and area grow.

**Note the `sched` gap.** At the knee `sched = 352` sits well above `p_agg = 240`
because the finite-resource schedule re-issues the `x` vector every wave (16 waves
× 16 `x`-vecs), whereas the invariant-amortized aggregate loads it once. This is
the concrete signature of a large invariant (see *Recurring vs. invariant loads*
in `DSE_rules.md`), and the quantity most likely to appear in a real DFG run if
`x` is not held in local storage.

## Comparing against measured DFG simulator cycles

Use the shared comparison rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For gemv, `absolute_cgra_lb = 232` is compute-bound and credits vector coalescing,
control amortization, and `x`-invariant amortization; it sits below the scalar
Metric-1 aggregate in the kernel's main `## CGRA-Constrained Model` section, which
charges induction per iteration and models no vector memory ops.
