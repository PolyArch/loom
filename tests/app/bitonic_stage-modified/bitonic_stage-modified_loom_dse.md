# bitonic_stage-modified Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps only the kernel-specific
setup, helper output, and recommendation rationale.

Kernel: `tests/app/bitonic_stage-modified/bitonic_stage-modified.cpp`
(`N = 8`, `stage = 1`, `pass = 0`, hence `distance = 1` and
`block_size = 4`). Relative to baseline `bitonic_stage`, every even outer
iteration runs a four-element multiply-in-place loop over `inplace[4..7]`, and
every odd iteration decrements `inplace[i]`.

```cpp
for (uint32_t i = 0; i < N; i++) {
    // bitonic compare/swap setup and conditional swap
    if ((idx_in_block & distance) == 0) {
        // optional compare/swap
        for (uint32_t j = N / 2; j < N; j++)
            inplace[j] *= 2;
    } else {
        inplace[i] -= 1;
    }
}
```

Model: the lane-aware + vector-coalescing Loom-pragma DSE in
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py bitonic_stage-modified --config 6x6 --top 16
```

## Why the outer loop is sequential

The source's `LOOM_PARALLEL(4, interleaved)` cannot make outer `i` iterations
independent. Its in-place writes overlap across iterations: each taken if-arm
reads and rewrites the same
`inplace[N/2..N-1]` slice, while odd iterations `i = 5, 7` write elements in
that slice. A later iteration can therefore consume a value written by an
earlier one. This is a non-reduction memory recurrence, so neither
`LOOM_PARALLEL` nor `LOOM_REDUCE` can legalize concurrent outer iterations.

Once that memory dependence makes `i` sequential, the iterator also becomes a
carried recurrence with `II = 3`; its chain and final else-arm tail set
`CP = 31`. The baseline has disjoint compare pairs, no repeated upper-half
rewrite, and remains `CP = 11`, whereas this variant's sequential outer chain
raises the depth to 31 cycles. The current
full-trip scalar counts are `A = 133`, `LD = 55`, and `ST = 48`; on a `6x6`
configuration their aggregate resource terms are only `4`, `5`, and `4`, so
the scalar CGRA aggregate is dependency-bound at 31.

## What pragma search can change

- **Outer `i`: no legal speedup.** Its parallel factor is forced to one.
  Unrolling cannot flatten or overlap the carried recurrence, cannot reduce its
  31-cycle critical path, and cannot turn the interleaved source pragma into
  independent workers.
- **Inner `j`: already consumed inside the serial body.** Its four lanes touch
  distinct elements within one taken outer iteration, but the current DSE helper
  models that local work as part of the fully consumed `i` recurrence rather
  than exposing a separate pragma level.
- **Expected effect: none in the current legal search.** Equivalent `U` labels
  for the fully consumed sequential level use the canonical `P1U1`
  representative because they all build the same full DAG.

Thus the search cannot remove or repackage the kernel's serial bottleneck. The
smallest representative is preferred when all candidates tie at the
dependency-bound estimate.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): bitonic_stage-modified  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[8,sequential]`; The outer i loop is sequential: its loop-counter carry and the in-place N/2..N-1 read-modify-write chain cross iterations. Parallel factors are illegal and unroll cannot flatten the recurrence, so equivalent unroll labels use the canonical P1U1 representative for the fully consumed serial DAG. Full-trip counts are `A=133`, `LD_rec=52`, `LD_eff=55`, `ST=48`, and `CP=31`, giving the only lower bound, `absolute_cgra_lb=31=max(CP 31, compute 4, load 5, store 4)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        i:P1U1                        1  12  12     55     8     1    31      31      31 latency-bound     13/16/13

RECOMMENDED: i:P1U1  -> exposure=8, pragma_agg=31 (1.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U contrast: no parallelizable level.
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Recommendation

**`i:P1U1` is the recommended representative.** Every sequential unroll label
consumes the same full recurrence and reports `p_agg = sched = 31`, equal to
`absolute_cgra_lb = 31`. The helper therefore uses the canonical `P1U1` label
instead of implying nonexistent spatial exposure.

Reserve **lower bound** for `absolute_cgra_lb`. Both `p_agg` and `sched` assume
wave serialization and are estimates; real execution may overlap waves and
fall below them toward `absolute_cgra_lb`.

## Comparing against the baseline

Baseline `bitonic_stage` has `CP = 11`, `A = 80`, `LD = 19`, and `ST = 12`.
This modified kernel adds repeated writes to the upper half and an odd-lane
decrement path, producing `CP = 31`, `A = 133`, `LD = 55`, and `ST = 48`. The
important DSE difference is not merely the extra work: the repeated in-place
updates create cross-iteration ordering, so outer-loop pragma exposure cannot
recover the baseline's parallel schedule.

Measured DFG comparisons should follow
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
