# Bitonic Stage Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps only the
bitonic-stage-specific setup, helper output, and recommendation.

Kernel: `tests/app/bitonic_stage/bitonic_stage.cpp` — loop over `i`

Current source pragma:

```cpp
LOOM_PARALLEL(4, interleaved)
LOOM_TRIPCOUNT_RANGE(10, 1000)
for (uint32_t i = 0; i < N; i++) {
    // One conditional compare-exchange pair rooted at i.
}
```

Model: the lane-aware + vector-coalescing Loom-pragma DSE in
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py bitonic_stage --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 16
```

## Setup

- Resource config: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`); vector width
  `V = 4` (one 256-bit vector op per four 64-bit lane slots).
- Fixture from `main.cpp`: `N = 8`, `stage = 1`, `pass = 0`,
  `inplace = {3, 1, 4, 2, 8, 6, 7, 5}`. Therefore `distance = 1` and
  `block_size = 4`.
- The `i` loop is parallel for this fixture. Only `i = {0, 2, 4, 6}` pass the
  outer predicate, their partners are `{1, 3, 5, 7}`, and the compare-exchange
  pairs are disjoint. Lanes `i = {0, 2}` swap; lanes `i = {4, 6}` compare but
  do not store.
- Strict no-predication is preserved: the outer pair predicate, `partner < N`,
  ascending/descending selection, and `should_swap` gate serialize before the
  operations in their taken bodies. The fixture's longest source-level chain is
  `CP = 11`.
- `distance`, `block_size`, and the dead `half_block` computation are
  loop-invariant and charged once per chunk. In particular, `half_block` is
  dead hoisted work, not per-exposed-iteration work.
- The recurring traffic is branch-dependent in-place I/O plus worker iterator
  traffic. `N`, `stage`, and `pass` are invariant loads and are amortized out of
  the binding recurring-load term, while remaining visible in `LD_eff`.

## Split interpretation

`P` creates interleaved workers, matching the source pragma; each worker carries
its own iterator. `U` exposes consecutive `i` values inside a worker and
amortizes that worker's loop control. Unlike a dense map kernel, however, only
alternate fixture lanes enter the compare-exchange body, and each active lane
touches both `inplace[i]` and `inplace[i + distance]`. The helper must therefore
apply the actual branch mask and paired in-place access pattern. These gated,
aliasing pair accesses remain scalar rather than receiving dense-stream vector
coalescing credit; a nominal `P*U` product alone is not enough to rank the split.

The DSE comparison is consequently about three effects: reaching the
11-cycle dependency depth with enough exposed pairs, reducing iterator work via
unroll, and avoiding extra workers or exposure after the binding resource has
saturated. `absolute_cgra_lb` is the only lower bound. `p_agg` and `sched` are
wave-serialized estimates used to compare legal pragma splits.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): bitonic_stage  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[8,parallel]`; i is parallel for the documented N=8, stage=1, pass=0 fixture: active lanes touch disjoint compare pairs. The branch mix is one active lane per pair and one committing swap lane per four i lanes. Conditional in-place pair accesses remain scalar because strict compare-to-body gates and swap aliasing do not form a plain contiguous vector stream. LOOM_UNROLL therefore helps only through outer-iterator control amortization; the 11-cycle gated CP dominates. Full-trip counts are `A=66`, `LD_rec=9`, `LD_eff=12`, `ST=5`, and `CP=11`, giving the only lower bound, `absolute_cgra_lb=11=max(CP 11, compute 2, load 1, store 1)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
b        i:P8U1                        8  12  12     19     8     1    11      11      11 latency-bound      27/18/9
b        i:P4U2                        4  12   8     15     8     1    11      11      11 latency-bound       18/9/9
b        i:P2U4                        2  10   6     13     8     1    11      11      11 latency-bound       18/9/9
K        i:P1U8                        1   9   5     12     8     1    11      11      11 latency-bound       18/9/9
b        i:P4U1                        4   8   6     11     4     2    11      22      22 latency-bound       18/9/9
b        i:P2U2                        2   6   4      9     4     2    11      22      22 latency-bound       18/9/9
b        i:P1U4                        1   5   3      8     4     2    11      22      22 latency-bound        9/9/9
b        i:P2U1                        2   4   4      7     2     4    11      44      44 latency-bound        9/9/9
b        i:P1U2                        1   3   3      6     2     4    11      44      44 latency-bound        9/9/9
b        i:P1U1                        1   3   3      6     1     8    11      88      88 latency-bound        9/9/9

RECOMMENDED: i:P1U8  -> exposure=8, pragma_agg=11 (1.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 8 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P8U1              16     19    12      11 tie (control/coalescing sit below the binding term)
  P4U2              12     15     8      11 tie (control/coalescing sit below the binding term)
  P2U4              10     13     6      11 tie (control/coalescing sit below the binding term)
  P1U8               9     12     5      11 tie (control/coalescing sit below the binding term)

4x4 recommendation: i:P1U8.
8x8 recommendation: i:P1U8.
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Recommendation

**`i:P1U8` is the recommended row (`K`)**: it exposes the full eight-iteration
fixture in one wave and reaches `p_agg = sched = 11`, exactly the
`absolute_cgra_lb = 11`. The kernel never becomes resource-bound in this sweep;
the serialized branch gates keep every full-exposure split latency-bound, so
the recommendation is the fewest-worker representative among the four
full-exposure ties (`P8U1`, `P4U2`, `P2U4`, and `P1U8`).

Unroll wins the tie by amortizing iterator traffic: across those equal
`11`-cycle rows, `LD_rec` falls from `16` to `9` and stores from `12` to `5` as
the split moves from `P8U1` to `P1U8`. The current source split `i:P4U1` exposes
only four iterations per wave, so it pays two copies of the 11-cycle gated path
and reports `p_agg = sched = 22`. This is a dependency-depth result, not a
bandwidth saturation knee; `p_agg` and `sched` remain estimates even though both
happen to equal the lower bound for the recommended fixture split.

## Comparing against measured DFG simulator cycles

Use the shared comparison rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For this fixture, the main eval's scalar aggregate and finite-resource schedule
are both `11` cycles because dependency depth binds. The Loom DSE may report a
different aggregate floor after vector coalescing, invariant amortization, and
per-worker control amortization; that difference is intentional. Compare any
measured simulator result separately against `absolute_cgra_lb`, `p_agg`, and
`sched`, without treating either estimate as a lower bound.
