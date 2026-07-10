# Autocorrelation Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
autocorrelation-specific setup, helper output, and recommendation.

Kernel: `tests/app/autocorrelation/autocorrelation.cpp` —
`output[lag] = Σ_i x[i]·x[i+lag]`

Current source pragma:

```cpp
LOOM_ACCEL()
void autocorrelation_dsa(const float *__restrict__ x, float *__restrict__ output,
                         uint32_t x_size, uint32_t max_lag) {
    LOOM_PARALLEL(4, contiguous)
    LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
    for (uint32_t lag = 0; lag < max_lag; lag++) {   // lags: parallel
        float sum = 0.0f;
        for (uint32_t i = 0; i < x_size - lag; i++)  // taps: reduction
            sum += x[i] * x[i + lag];
        output[lag] = sum;
    }
}
```

This uses the shared lane-aware + vector-coalescing DSE from
[`DSE_rules.md`](../DSE_rules.md) and
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py autocorrelation --config 6x6 --max-parallel 8 --top 14
```

## Autocorrelation-specific setup

`autocorrelation` is a nested kernel: an outer **parallel** lag loop `lag` and an
inner **reduction** tap loop `i`. It is gemv-shaped — each `lag` privatizes its own
`sum` and writes a distinct `output[lag]`, with no carry through register or memory.

- DSE fixture: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`), `V = 4`, `x_size = 128`,
  `max_lag = 32`. Unlike gemv, these are exactly the smoke-test fixture in
  `main.cpp`, so the modeling size matches what the simulator runs.
- The dot-product `i` level is **P/U-symmetric**: the carried dep is an associative
  float sum, so it is lowered as a spatial reduction tree and carries no per-element
  `i` loop control. The reduction path is identical whether exposure comes from `P`
  or `U`.
- The lag `lag` level has a small unroll edge: `LOOM_UNROLL(lag)` coalesces the
  contiguous `output[lag]` stores and amortizes the lag iterator, so it beats
  `LOOM_PARALLEL(lag)` (which keeps an iterator per worker and strides its stores).
- `x[i]` (the un-shifted prefix) is the **same data every lag reads**, so it is
  modeled **invariant** (loaded once per chunk, contiguous → coalesced), directly
  analogous to gemv's whole `x` vector being invariant of the row. `x[i+lag]` is a
  per-lag-shifted window → **recurring**, but contiguous over `i`, so it coalesces.
  The invariant term is `LD_inv = LD_eff − LD_rec = 34` (`32` coalesced `x`-prefix
  vectors + `2` param scalars `x_size`/`max_lag`); it is amortized out of the
  binding load term.
- **Modeling caveat.** The inner reduction is modeled at its **maximum** length
  `x_size = 128` (the `lag = 0` case). The true per-lag length is `x_size − lag`,
  so with `lag < max_lag = 32` the executed lengths run from `128` down to `97`
  (at `lag = 31`). Modeling all 32 lags at `128` therefore conservatively
  **over-counts** the inner work: `32·128 = 4096` products against the true
  `Σ_{lag=0}^{31} (128 − lag) = 3600`, ≈ `14%`. Cross-lag reuse of the overlapping
  `x` windows is otherwise not modeled (the conservative conv2d-halo convention).
  This does not change the compute-bound conclusion.
- Full-trip counts: `A = 8162`, `LD_rec = 1025`, `LD_eff = 1059`, `ST = 9`,
  `CP = 10`. Thus `absolute_cgra_lb = max(10, ceil(8162/36), ceil(1025/12), 1) =
  227`, the only lower bound. The binding class is compute (`P`).

## Results (`--top 14`)

```text
# Loom pragma DSE (lane-aware + vector coalescing): autocorrelation  (6x6)

Loop nest: `lag[32,parallel]`, `i[128,reduction]`. This gemv-shaped DSE treats `x[i]` as an invariant coalesced prefix and `x[i+lag]` as recurring but coalesced over `i`; `LOOM_UNROLL(lag)` has the small edge because it coalesces `output[lag]` stores and amortizes the lag iterator. The model uses `P_pe=36`, `L=S=12`, `V=4`, models every lag at the max `x_size=128` length (4096 products vs true 3600, about 14% conservative over-count), and reports `absolute_cgra_lb=227` as the only lower bound from full-trip counts `A=8162 LD_rec=1025 LD_eff=1059 ST=9 CP=10` with compute binding (`compute=227 load=86 store=1`); `p_agg` and `sched` are wave-serialized estimates, `aL` is recurring-load exposure, and invariant loads are amortized out of the binding load term.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
o        lag:P4U8 i:P1U1  (+15 eq)     4  12  12   1062  4096     1   227     227     232 resource-bound    100/38/0
o        lag:P4U2 i:P1U1  (+15 eq)     4  12   8    294  1024     4    57     228     252 resource-bound    100/39/2
o        lag:P2U4 i:P1U1  (+15 eq)     2  12   4    292  1024     4    57     228     252 resource-bound    100/39/2
o        lag:P1U8 i:P1U1  (+15 eq)     1  12   3    291  1024     4    57     228     252 resource-bound    100/39/2
o        lag:P8U2 i:P1U1  (+15 eq)     8  12  12    554  2048     2   114     228     238 resource-bound    100/39/2
o        lag:P4U4 i:P1U1  (+15 eq)     4  12   8    550  2048     2   114     228     238 resource-bound    100/38/1
o        lag:P2U8 i:P1U1  (+15 eq)     2  12   6    548  2048     2   114     228     238 resource-bound    100/38/1
o        lag:P8U4 i:P1U1  (+15 eq)     8  12  12   1066  4096     1   228     228     232 resource-bound    100/38/1
o        lag:P4U1 i:P1U1  (+15 eq)     4  12   8    166   512     8    29     232     288 resource-bound    100/38/3
o        lag:P2U2 i:P1U1  (+15 eq)     2  12   4    164   512     8    29     232     288 resource-bound    100/38/3
o        lag:P1U4 i:P1U1  (+15 eq)     1  12   2    163   512     8    29     232     288 resource-bound    100/38/3
o        lag:P8U1 i:P1U1  (+15 eq)     8  12  12    298  1024     4    58     232     252 resource-bound    100/38/3
         lag:P2U1 i:P1U1  (+15 eq)     2  12   4    100   256    16    15     240     352 resource-bound    100/40/7
K        lag:P1U2 i:P1U1  (+15 eq)     1  12   2     99   256    16    15     240     352 resource-bound    100/40/7
... (1 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: lag:P1U2 i:P1U1  -> exposure=256, pragma_agg=240 (1.06x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 8 on level 'lag' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P8U1             264    298    16     232 1.02x slower (parallel: extra iterators + strided, no coalesce)
  P4U2             260    294     8     228 best
  P2U4             258    292     4     228 best
  P1U8             257    291     3     228 best
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Recommendation and reading

**`lag:P1U2 i:P1U1` is the recommended knee (`K`)**: exposure `256` (2 lags × their
full 128-tap reduction), `p_agg = 240` (`1.06×` the `227` floor), resource-bound on
compute. It is `E_sat` — the smallest exposure at which the arithmetic term overtakes
`CP = 10`. `lag:P2U1` ties it exactly at `240`; the tool reports the pure-unroll
representative because it carries the fewest lag iterators.

At fixed lag-level product `P·U = 8`, unroll improves both memory and the small
compute-control term: `LD_rec` falls `264 → 257` from lag-iterator amortization, and
`ST` falls `16 → 3` from `output[lag]` coalescing. Because compute binds, this only
moves `p_agg` from `232` (`P8U1`) to `228` (the unrolled ties) — a `1.02×` edge, so
autocorrelation remains largely P/U-symmetric.

Below the knee, `lag:P1U1` (exposure `128`) is latency-bound (`b`): the merge tree
drains before the compute lanes fill (util `80/30/10`). Above the knee (`o`, e.g.
`lag:P2U4`, `lag:P4U8`), `p_agg` only creeps toward the `227` floor through per-wave
rounding while transient backlog and area pressure grow.

**Note the `sched` gap.** At the knee `sched = 352` sits well above `p_agg = 240`
because the finite-resource schedule estimate re-issues the invariant `x` prefix
every wave (16 waves × 32 `x`-vecs), whereas the invariant-amortized aggregate loads
it once. This is the concrete signature of a large invariant (see *Recurring vs.
invariant loads* in `DSE_rules.md`): `sched` is itself an **estimate** under a
policy that reloads invariants per wave — not a lower bound, and not a prediction of
measured cycles. Whether a real DFG run pays this depends on whether the `x` prefix
is actually held in local storage across the row sweep.

## Comparing against measured DFG simulator cycles

Use the shared rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).

The ASAP CGRA model in `autocorrelation_eval.md` is **load-bound** at `903`
(`LD = 10,834`, `load = ceil(10,834/12)`): it charges **both** `x[i]` and `x[i+lag]`
scalar loads per inner iter plus per-iteration induction reads, with no vector
coalescing and no invariant amortization. This DSE instead amortizes the invariant
`x` prefix (loaded once per chunk) and coalesces the contiguous `x[i+lag]` and
`output[lag]` streams, dropping the binding load term to `load = 86`
(`ceil(1025/12)`) and making the kernel **compute-bound** at `227`. So the DSE floor
credits vector coalescing, lag-control amortization, and `x`-invariant amortization —
treat it as this optional DSE floor rather than the scalar CGRA aggregate.
