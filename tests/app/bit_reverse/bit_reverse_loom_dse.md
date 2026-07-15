# Bit Reverse Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
bit_reverse-specific setup, helper output, and recommendation.

Kernel: `tests/app/bit_reverse/bit_reverse.cpp` — reverse the 32 bits of each of
`N` words (`output_reversed[i]` is the bit-reversal of `input_data[i]`).

Current source pragma:

```cpp
LOOM_PARALLEL()
LOOM_UNROLL(8)
for (uint32_t i = 0; i < N; i++) {          // words: parallel
    uint32_t value = input_data[i];
    uint32_t result = 0;
    for (uint32_t bit = 0; bit < 32; bit++) {   // bits: sequential
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    output_reversed[i] = result;
}
```

This uses the shared lane-aware + vector-coalescing DSE from
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py bit_reverse --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 16
```

## Bit-reverse-specific setup

`bit_reverse` is a nested kernel: an outer **parallel** word loop `i` and an
inner **sequential** bit loop.

- DSE fixture: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`), `V = 4`, `N = 256`,
  `BITS = 32`. These match the smoke-test fixture in `main.cpp` (`N = 256`), so
  the modeling size is exactly what the simulator runs.
- The outer `i` loop is **parallel**: each iteration reverses one independent
  32-bit word and writes a distinct `output_reversed[i]`. It may take parallel
  workers and unroll.
- The inner `bit` loop is **sequential**: it carries `result` and `value`
  through a **non-associative** shift/merge recurrence
  (`result = (result << 1) | (value & 1); value >>= 1`). Like the
  [`tridiag_solve`](../tridiag_solve/tridiag_solve_loom_dse.md) forward sweep,
  this recurrence cannot be parallelized, reduced, or spatially flattened, so
  the model forces `P_tot = 1` on `bit` and its iterator stays per-iteration on
  the critical path.
- Following the tridiag convention, the carried `result`/`value` are threaded as
  **dataflow edges** (no per-bit memory round-trip), and the `bit` iterator is
  charged **per iteration** — a sequential carry keeps per-iter control and
  **cannot** be amortized. The **outer** `i` iterator *is* amortizable (one
  advance per worker/wave).
- `input_data[i]` / `output_reversed[i]` are contiguous over `i`, so they
  coalesce under `LOOM_UNROLL(i)` (`V = 4`). The 4 bitops per bit (`<< 1`,
  `& 1`, `|`, `>> 1`) form a large global arithmetic pool.
- Full-trip counts: `A = 49154` (≈ `4·BITS·N = 32768` bitops + the per-bit
  iterator add+compare ≈ `2·BITS·N = 16384`), `LD_rec = 8257`, `LD_eff = 8258`,
  `ST = 8257` (per-bit iterator reads/writes dominate; the coalesced boundary
  I/O and the amortized outer-`i` iterator are the small remainder), `CP = 66`
  (the result recurrence, ≈ `2·BITS` deep plus a boundary load + store). Thus
  `absolute_cgra_lb = max(66, ceil(49154/36), ceil(8257/12), ceil(8257/12)) =
  1366`, binding class **compute** (`P`).

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): bit_reverse  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[256,parallel], bit[32,sequential]`; outer PARALLEL i (independent 32-bit words), inner SEQUENTIAL bit loop carrying result/value through a non-associative shift/merge recurrence. The carried scalars are threaded as dataflow (no per-bit round-trip, unlike the conservative ASAP eval) and the bit iterator is charged per iteration (it stays on CP and cannot be amortized). The 4 bitops/bit form a large global arithmetic pool -> COMPUTE-bound (contrast the store-bound ASAP result, which charges per-bit result/value stores). LOOM_UNROLL(i) coalesces input_data/output_reversed and amortizes the OUTER i iterator, but the inner sequential loop dominates, so the P-vs-U edge is small. Full-trip counts are `A=49154`, `LD_rec=8257`, `LD_eff=8258`, `ST=8257`, and `CP=66`, giving the only lower bound, `absolute_cgra_lb=1366=max(CP 66, compute 1366, load 689, store 689)`, with compute pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
o        i:P4U32 bit:P1U1              4  12  12   4133  4096     2   683    1366    1588 resource-bound   100/51/51
o        i:P2U64 bit:P1U1              2  12  12   4131  4096     2   683    1366    1588 resource-bound   100/51/51
o        i:P1U128 bit:P1U1             1  12  12   4130  4096     2   683    1366    1588 resource-bound   100/51/51
o        i:P8U32 bit:P1U1              8  12  12   8265  8192     1  1366    1366    1586 resource-bound   100/50/50
o        i:P4U64 bit:P1U1              4  12  12   8261  8192     1  1366    1366    1586 resource-bound   100/50/50
o        i:P2U128 bit:P1U1             2  12  12   8259  8192     1  1366    1366    1586 resource-bound   100/50/50
o        i:P1U256 bit:P1U1             1  12  12   8258  8192     1  1366    1366    1586 resource-bound   100/50/50
o        i:P16U16 bit:P1U1            16  12  12   8273  8192     1  1367    1367    1587 resource-bound   100/50/50
o        i:P4U8 bit:P1U1               4  12  12   1037  1024     8   171    1368    1600 resource-bound   100/51/51
o        i:P2U16 bit:P1U1              2  12  12   1035  1024     8   171    1368    1600 resource-bound   100/51/51
o        i:P1U32 bit:P1U1              1  12  12   1034  1024     8   171    1368    1600 resource-bound   100/51/51
o        i:P8U8 bit:P1U1               8  12  12   2073  2048     4   342    1368    1592 resource-bound   100/51/51
o        i:P4U16 bit:P1U1              4  12  12   2069  2048     4   342    1368    1592 resource-bound   100/51/51
o        i:P2U32 bit:P1U1              2  12  12   2067  2048     4   342    1368    1592 resource-bound   100/51/51
o        i:P1U64 bit:P1U1              1  12  12   2066  2048     4   342    1368    1592 resource-bound   100/51/51
o        i:P16U8 bit:P1U1             16  12  12   4145  4096     2   684    1368    1590 resource-bound   100/51/51
K        i:P1U16 bit:P1U1              1  12  12    518   512    16    86    1376    1520 resource-bound   100/51/51
... (28 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: i:P1U16 bit:P1U1  -> exposure=512, pragma_agg=1376 (1.01x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 16 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P16U1             544    545   544    1392 1.01x slower (parallel: extra iterators + strided, no coalesce)
  P8U2             528    529   528    1376 best
  P4U4             520    521   520    1376 best
  P2U8             518    519   518    1376 best
  P1U16            517    518   517    1376 best

4x4 recommendation: i:P1U8 bit:P1U1.
8x8 recommendation: i:P1U32 bit:P1U1.
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Recommendation and reading

**`i:P1U16 bit:P1U1` is the recommended knee (`K`)**: exposure `512` (words per
wave × their full 32-bit sequential reversal), `p_agg = 1376` (`1.01×` the
`1366` compute floor), resource-bound on the arithmetic pool. It is the smallest
power-of-two exposure that saturates `P`; `i:P2U8`, `i:P4U4`, and `i:P8U2` tie
it at `1376`, and the tool reports the unroll-heaviest representative because it
carries the fewest word iterators.

The **P-vs-U edge on the outer `i` is small**. At fixed word-level product
`P·U = 32`, `P4U8` (`p_agg = 1368`) is only `1.01×` faster than `P8U4`
(`p_agg = 1376`): unroll coalesces the contiguous `input_data`/`output_reversed`
groups and amortizes the word iterator, but those savings sit far below the
binding compute term. The inner **sequential** bit loop supplies the bulk of the
`A = 49154` bitops and the whole `CP`, and it is identical under every legal `i`
factoring, so the split barely moves the aggregate.

Below the knee the rows are latency-bound (`b`): `i:P8U1` and smaller expose too
few words per wave, so `P` idles (`util` `67%` and down) and `p_agg` climbs with
the wave count. Above it (`o`, e.g. `i:P4U8`, `i:P8U8`), `p_agg` only drifts toward
the `1366` floor through per-wave rounding while transient backlog grows. Read the
legend's "no estimate gain" as *no steady-state throughput gain*: an oversubscribed
row can even show a marginally **lower** `p_agg` than the knee — `i:P4U8` reports
`1368` versus the knee's `1376` — but that `8`-cycle difference is pure
wave-ceiling rounding (`P4U8` fits the trip in 8 waves with no remainder), not a
real per-cycle speedup, and it costs `4×` the exposure and backlog.

## Comparing against measured DFG simulator cycles

The single largest model divergence for this kernel is the carried-scalar
accounting, and it flips the binding class:

- The ASAP `bit_reverse_eval.md` is **store-bound** (`store = 2134`,
  `ST = 25,600`). ASAP conservatively charges a per-bit memory round-trip for
  `result`, `value`, **and** the `bit` iterator — 3 loads + 3 stores every bit —
  so the scalar store traffic dominates.
- This DSE instead threads the carried `result`/`value` as **dataflow edges**
  (only the `bit` iterator plus the coalesced boundary `input_data`/
  `output_reversed` touch memory), so `ST` drops from `25,600` to `8,257` and
  the kernel becomes **compute-bound** on the 4 bitops/bit (`A = 49154`,
  `compute = 1366`) rather than store-bound.

Because the DSE floor credits dataflow threading of the carried scalars,
boundary coalescing, and outer-`i` control amortization, `absolute_cgra_lb =
1366` sits well below the scalar ASAP aggregate (`2134`) for the same `6x6`
machine. This is expected: the DSE is an optional floor, not the scalar CGRA
aggregate.

Measured DFG simulator comparisons should use the shared rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
Treat `absolute_cgra_lb = 1366` as this DSE's optional floor; a real DFG run that
does not thread `result`/`value` through registers — spilling them to memory each
bit — would regress toward the store-bound ASAP figure instead.
