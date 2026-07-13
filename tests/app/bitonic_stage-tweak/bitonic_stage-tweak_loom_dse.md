# Bitonic Stage Tweak Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
bitonic_stage-tweak-specific setup, helper output, and recommendation.

Kernel: `tests/app/bitonic_stage-tweak/bitonic_stage-tweak.cpp` — execute one
in-place bitonic compare-swap stage, then increment each active left endpoint
and decrement every element.

Current source pragma:

```cpp
LOOM_PARALLEL(4, interleaved)
LOOM_TRIPCOUNT_RANGE(10, 1000)
for (uint32_t i = 0; i < N; i++) {
    // Compare/swap inplace[i] with inplace[i + distance] on active lanes.
    if ((idx_in_block & distance) == 0) {
        // ... conditional compare-swap ...
        inplace[i]++;
    }
    inplace[i] -= 1;
}
```

This uses the shared lane-aware + vector-coalescing DSE from
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py bitonic_stage-tweak --config 6x6 --top 16
```

## Bitonic-stage-tweak-specific setup

`bitonic_stage-tweak` has one outer `i` loop, but it is not a legal spatial
dimension. The DSE fixture is the smoke-test case from `main.cpp`: `6x6`
(`P_pe = 36`, `L = 12`, `S = 12`), `V = 4`, `N = 8`, `stage = 1`, and
`pass = 0`, giving `distance = 1` and `block_size = 4`.

- The baseline `bitonic_stage` compare-swap writes both `inplace[i]` and
  `inplace[partner]`, where `partner = i + distance`. The tweak then performs
  `inplace[i]++` on every active lane and unconditional `inplace[i] -= 1` on
  every iteration. These are ordered read-modify-write chains on the same
  in-place array.
- Within an active iteration, a committed swap must precede `++`, and `++`
  must precede `-= 1` on `inplace[i]`. Across iterations, the swap's write to
  `inplace[i + distance]` can feed that later iteration's unconditional
  `-= 1`; for the fixture, iter 0 writes `inplace[1]` before iter 1 decrements
  it, and iter 2 similarly feeds iter 3. This is a true memory-carried
  dependence, not removable loop-control overhead.
- Therefore `i` is **sequential** and `P` is forced to one. Every `U` label
  builds the same fully consumed recurrence, so the helper uses the canonical
  `i:P1U1` representative. The checked-in `LOOM_PARALLEL(4, interleaved)` is
  outside the legal DSE space for this tweak.
- Because no loop dimension is legally exposed, vector coalescing and
  parallel-iterator amortization do not apply. The sequential iterator and
  the array read-modify-write traffic remain ordered. From the current eval
  fixture, the full dynamic totals are `CP = 17`, `A = 92`, `LD = 31`, and
  `ST = 24`, so `absolute_cgra_lb = max(17, ceil(92/36), ceil(31/12),
  ceil(24/12)) = 17`, binding on dependency depth. This
  `absolute_cgra_lb` is the only lower bound; `p_agg` and `sched` are
  wave-serialized estimates.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): bitonic_stage-tweak  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[8,sequential]`; The unconditional inplace[i]-=1 and active-lane inplace[i]++ create same-slot and partner RAW chains across the in-place stage. Parallel factors are therefore illegal and unroll cannot flatten the memory recurrence; equivalent unroll labels use the canonical P1U1 representative for the same 17-cycle serial DAG. Full-trip counts are `A=92`, `LD_rec=28`, `LD_eff=31`, `ST=24`, and `CP=17`, giving the only lower bound, `absolute_cgra_lb=17=max(CP 17, compute 3, load 3, store 2)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        i:P1U1                        1  12  12     31     8     1    17      17      17 latency-bound     18/18/12

RECOMMENDED: i:P1U1  -> exposure=8, pragma_agg=17 (1.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U contrast: no parallelizable level.
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Recommendation

**Keep `i:P1U1`.** There is no legal spatial saturation-knee search for this source:
the compare-swap, `++`, and unconditional `-= 1` create in-place RAW/WAW
ordering both within an iteration and from an active iteration to a later
partner iteration. Any `P > 1` candidate would violate that ordering; the
other `U` labels are semantically equivalent, not real overlap. If spatial
exposure is required, the algorithm must first be
rewritten to remove the cross-iteration alias, for example by separating the
stage result from the subsequent element updates; that would be a different
kernel and requires a new dependence analysis.

## Comparing against measured DFG simulator cycles

Use the shared comparison rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For this fixture, compare measured cycles against `absolute_cgra_lb = 17`, then
against the regenerated `p_agg` and `sched` values separately. Only
`absolute_cgra_lb` is a lower bound; `p_agg` and `sched` remain deterministic
estimates for the helper's exposure and scheduling policies.
