# gemv Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the gemv-specific
setup, helper output, and recommendation.

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

This uses the shared lane-aware + vector-coalescing DSE from
[`DSE_rules.md`](../DSE_rules.md) and
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).
The source `LOOM_MEMORY_BANK(4, block)` on `A` is intentionally **ignored** by
this provisional no-banking DSE.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py gemv --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 14
```

## Gemv-specific setup

`gemv` is a nested kernel: an outer **parallel** row loop `i` and an inner
**reduction** column loop `j`.

- DSE fixture: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`), `V = 4`,
  `M = N = 64`. This is the helper's modeling size, not the smoke-test fixture in
  `main.cpp`.
- The dot-product `j` level is **P/U-symmetric**: `A[i][j]` and `x[j]` are
  contiguous over the fully-consumed reduction, so they coalesce identically and
  the `j` loop carries no control.
- The row `i` level has a small unroll edge: `LOOM_UNROLL(i)` coalesces
  contiguous `input_y[i]` loads and `output_y[i]` stores, and it amortizes the row
  iterator. The large `A` row term is split-symmetric, so the edge is real but
  modest.
- `x[j]` is invariant of `i`: the full `x` vector plus `alpha`/`beta`/`M`/`N`
  contributes `LD_inv = 20` (`16` `x` vectors + `4` scalars). These loads appear in
  `LD_eff` but are amortized out of the binding load term.
- Full-trip counts: `A = 8322`, `LD_rec = 1041`, `LD_eff = 1061`, `ST = 17`,
  `CP = 11`. Thus `absolute_cgra_lb = max(11, ceil(8322/36), ceil(1041/12), 2) =
  232`, the only lower bound. The binding class is compute (`P`).

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): gemv  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[64,parallel], j[64,reduction]`; A[i][j] and x[j] are contiguous over j (a fully-consumed reduction, tree-reduced), so they coalesce identically and the j-loop carries no control -> the dot-product path is P/U-symmetric. On the row level i, LOOM_UNROLL(i) beats LOOM_PARALLEL(i) two ways: it coalesces the contiguous y[i]/output_y[i] accesses (parallel strides) and it amortizes the row iterator (charged once per worker). The A-load term is split-symmetric and large, so the i-level edge is modest but real. Full-trip counts are `A=8322`, `LD_rec=1041`, `LD_eff=1061`, `ST=17`, and `CP=11`, giving the only lower bound, `absolute_cgra_lb=232=max(CP 11, compute 232, load 87, store 2)`, with compute pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
o        i:P2U4 j:P1U1                 2  12   4    152   512     8    29     232     280 resource-bound    100/38/3
o        i:P1U8 j:P1U1                 1  12   3    151   512     8    29     232     280 resource-bound    100/38/3
o        i:P4U4 j:P1U1                 4  12   8    284  1024     4    58     232     252 resource-bound    100/38/2
o        i:P2U8 j:P1U1                 2  12   6    282  1024     4    58     232     252 resource-bound    100/38/2
o        i:P1U16 j:P1U1                1  12   5    281  1024     4    58     232     252 resource-bound    100/38/2
o        i:P8U4 j:P1U1                 8  12  12    548  2048     2   116     232     238 resource-bound    100/38/2
o        i:P4U8 j:P1U1                 4  12  12    544  2048     2   116     232     238 resource-bound    100/38/1
o        i:P2U16 j:P1U1                2  12  10    542  2048     2   116     232     238 resource-bound    100/38/1
o        i:P1U32 j:P1U1                1  12   9    541  2048     2   116     232     238 resource-bound    100/38/1
o        i:P16U4 j:P1U1               16  12  12   1076  4096     1   232     232     235 resource-bound    100/38/1
o        i:P8U8 j:P1U1                 8  12  12   1068  4096     1   232     232     235 resource-bound    100/38/1
o        i:P4U16 j:P1U1                4  12  12   1064  4096     1   232     232     235 resource-bound    100/38/1
o        i:P2U32 j:P1U1                2  12  12   1062  4096     1   232     232     235 resource-bound    100/38/1
o        i:P1U64 j:P1U1                1  12  12   1061  4096     1   232     232     235 resource-bound    100/38/1
K        i:P1U4 j:P1U1                 1  12   2     86   256    16    15     240     352 resource-bound    100/40/7
... (13 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: i:P1U4 j:P1U1  -> exposure=256, pragma_agg=240 (1.03x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 4 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P4U1              72     92     8     240 tie (control/coalescing sit below the binding term)
  P2U2              68     88     4     240 tie (control/coalescing sit below the binding term)
  P1U4              66     86     2     240 tie (control/coalescing sit below the binding term)

4x4 recommendation: i:P1U2 j:P1U1.
8x8 recommendation: i:P1U8 j:P1U1.
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Recommendation and reading

**`i:P1U4 j:P1U1` is the recommended knee (`K`)**: exposure `256` (4 rows × their
full 64-tap reduction), `p_agg = 240` (`1.03×` the `232` floor), resource-bound on
compute. It is `E_sat` — the smallest exposure at which the arithmetic term
overtakes `CP = 11`. `i:P2U2` and `i:P4U1` tie it exactly at `240`; the tool
reports the pure-unroll representative because it carries the fewest row iterators.

At fixed row-level product `P·U = 8`, unroll improves both memory and the small
compute-control term: `LD_rec` falls `144 → 131` from row-iterator amortization
and `input_y` coalescing, while `ST` falls `16 → 3` from `output_y` coalescing.
The helper also reduces `A` from `1056 → 1042`; because compute binds, that small
control delta is what changes `cagg` from `30` to `29` and `p_agg` from `240` to
`232`. The improvement is only `1.03×`, so gemv remains largely P/U-symmetric.

Below the knee, `i:P2U1` is latency-bound (`b`): the merge tree drains before the
compute lanes fill. Above it (`o`, e.g. `i:P2U4`, `i:P8U8`), `p_agg` only creeps
toward the `232` floor through per-wave rounding while transient backlog and area
pressure grow.

**Note the `sched` gap.** At the knee `sched = 352` sits well above `p_agg = 240`
because the finite-resource schedule re-issues the `x` vector every wave (16 waves
× 16 `x`-vecs), whereas the invariant-amortized aggregate loads it once. This is
the concrete signature of a large invariant (see *Recurring vs. invariant loads*
in `DSE_rules.md`), and the quantity most likely to appear in a real DFG run if
`x` is not held in local storage.

Measured DFG simulator comparisons should use the shared rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For gemv, the DSE floor credits vector coalescing, row-control amortization, and
`x`-invariant amortization, so treat it as this optional DSE floor rather than a
scalar CGRA aggregate.
