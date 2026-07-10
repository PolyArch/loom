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
python3 tests/scripts/loom_dse.py bit_reverse --config 6x6 --max-parallel 8 --max-unroll 8 --top 16
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

## Results (`--top 16`)

```text
# Loom pragma DSE (lane-aware + vector coalescing): bit_reverse  (6x6)

loop nest (outer->inner): i[256,parallel], bit[32,sequential]. The outer word loop is parallel, while the inner bit loop is sequential because result/value form a non-associative shift/merge recurrence; carried scalars are threaded as dataflow, but the bit iterator stays per-iteration and on CP. absolute_cgra_lb = 1366 = max(CP 66, compute 1366, load 689, store 689) is the only lower bound, with full-trip counts A=49154, LD_rec=8257, LD_eff=8258, ST=8257; p_agg and sched are wave-serialized estimates, and unroll helps only through outer-word control amortization plus vector coalescing of input_data/output_reversed.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
o        i:P4U8 bit:P1U1  (+3 eq)      4  12  12   1037  1024     8   171    1368    1600 resource-bound   100/51/51
o        i:P8U8 bit:P1U1  (+3 eq)      8  12  12   2073  2048     4   342    1368    1592 resource-bound   100/51/51
         i:P8U2 bit:P1U1  (+3 eq)      8  12  12    529   512    16    86    1376    1536 resource-bound   100/51/51
         i:P4U4 bit:P1U1  (+3 eq)      4  12  12    521   512    16    86    1376    1520 resource-bound   100/51/51
K        i:P2U8 bit:P1U1  (+3 eq)      2  12  12    519   512    16    86    1376    1520 resource-bound   100/51/51
o        i:P8U4 bit:P1U1  (+3 eq)      8  12  12   1041  1024     8   172    1376    1600 resource-bound   100/51/51
b        i:P8U1 bit:P1U1  (+3 eq)      8  12  12    273   256    32    66    2112    2112 latency-bound     67/35/35
b        i:P4U2 bit:P1U1  (+3 eq)      4  12  12    265   256    32    66    2112    2112 latency-bound     65/33/33
b        i:P2U4 bit:P1U1  (+3 eq)      2  12  12    261   256    32    66    2112    2112 latency-bound     65/33/33
b        i:P1U8 bit:P1U1  (+3 eq)      1  12  12    260   256    32    66    2112    2112 latency-bound     65/33/33
b        i:P4U1 bit:P1U1  (+3 eq)      4  12  12    137   128    64    66    4224    4224 latency-bound     33/18/18
b        i:P2U2 bit:P1U1  (+3 eq)      2  12  12    133   128    64    66    4224    4224 latency-bound     33/17/17
b        i:P1U4 bit:P1U1  (+3 eq)      1  12  12    131   128    64    66    4224    4224 latency-bound     33/17/17
b        i:P2U1 bit:P1U1  (+3 eq)      2  12  12     69    64   128    66    8448    8448 latency-bound       17/9/9
b        i:P1U2 bit:P1U1  (+3 eq)      1  12  12     67    64   128    66    8448    8448 latency-bound       17/9/9
b        i:P1U1 bit:P1U1  (+3 eq)      1  12  12     35    32   256    66   16896   16896 latency-bound        9/5/5

RECOMMENDED: i:P2U8 bit:P1U1  -> exposure=512, pragma_agg=1376 (1.01x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 32 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P8U4            1040   1041  1040    1376 1.01x slower (parallel: extra iterators + strided, no coalesce)
  P4U8            1036   1037  1036    1368 best
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Recommendation and reading

**`i:P2U8 bit:P1U1` is the recommended knee (`K`)**: exposure `512` (words per
wave × their full 32-bit sequential reversal), `p_agg = 1376` (`1.01×` the
`1366` compute floor), resource-bound on the arithmetic pool. It is the smallest
exposure that saturates `P`; `i:P4U4` and `i:P8U2` tie it exactly at `1376`, and
the tool reports the unroll-heaviest representative because it carries the fewest
word iterators.

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

## Broader unroll note

If the sweep allows `--max-unroll 16`, `i:P1U16 bit:P1U1` performs at least as
well as the current-table knee `i:P2U8 bit:P1U1`: both report `p_agg = 1376` and
`sched = 1520`, while `P1U16` carries one fewer outer word iterator per wave
(`A = 3074`, `LD_rec = 517`, `ST = 517`, versus `A = 3076`, `LD_rec = 518`,
`ST = 518` for `P2U8`). The checked-in table uses `--max-unroll 8`, matching the
current source pragma, so `P1U16` is outside that displayed sweep.
