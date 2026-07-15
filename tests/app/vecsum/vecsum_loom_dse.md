# vecsum Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps the vecsum-specific setup,
helper output, and recommendation.

Kernel: `tests/app/vecsum/vecsum.cpp` — `sum = init_value + Σ_i A[i]`

Current source pragma:

```cpp
LOOM_REDUCE(+)
uint32_t sum = init_value;
LOOM_PARALLEL(4)
for (uint32_t i = 0; i < N; i++) {
    sum += A[i];
}
```

This file selects `LOOM_PARALLEL(P)` / `LOOM_UNROLL(U)` under the shared
lane-aware + vector-coalescing model, which is defined by
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma
Design-Space Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py vecsum --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 24
```

## Why P and U are symmetric

`vecsum` carries `sum`, but the carry is associative, so `LOOM_REDUCE(+)`
legalizes a balanced merge tree. That makes both P/U differentiators inert:

- the whole loop is a fully consumed spatial reduction, so no per-element or
  per-worker iterator changes the split;
- `A` is one contiguous run, so its `256` element loads coalesce to `~trip/V = 64`
  vector loads for any split.

Every candidate therefore collapses into one equivalence group: vecsum shows no
`LOOM_PARALLEL`-vs-`LOOM_UNROLL` distinction.

## Setup

- Resource config: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`); `V = 4`; `N = 256`.
  The reduction is fully consumed in one wave, so exposure is always `256`.
- Full-trip counts: `A = 258`, `LD_rec = 65` (`64` coalesced `A`-vecs + `1`
  residual iterator), `LD_inv = 2` (`init`, `N`), `LD_eff = 67`, `ST = 2`,
  `CP = 11`.
- `absolute_cgra_lb = 11 = max(CP 11, compute 8, load 6, store 1)`. It is the
  only lower bound and is CP-bound on the log-depth merge tree.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): vecsum  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[256,reduction]`; A is contiguous over i and the reduction is fully consumed in one wave AND tree-reduced (a spatial tree), so it carries no per-element and no per-worker iterator -- control is a fixed residual for any split. A also coalesces to ~trip/V vector loads regardless of the p/u split. Both the control and coalescing axes are inert -> vecsum is P/U-symmetric, and CP-bound on the log-depth merge tree. Full-trip counts are `A=258`, `LD_rec=65`, `LD_eff=67`, `ST=2`, and `CP=11`, giving the only lower bound, `absolute_cgra_lb=11=max(CP 11, compute 8, load 6, store 1)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        i:P1U1                        1  12   2     67   256     1    11      11      16 latency-bound      73/55/9

RECOMMENDED: i:P1U1  -> exposure=256, pragma_agg=11 (1.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U contrast: primary level has no fixed-product split set.

4x4 recommendation: i:P1U1.
8x8 recommendation: i:P1U1.
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Recommendation

All candidates land in one equivalence group (`+19 eq`), so the split choice is
indifferent. The helper prints the smallest representative, `i:P1U1`, with
`exposure = 256`, `p_agg = 11` (`1.00x` the floor), and latency bound on the merge
tree (`util P/L/S = 73/55/9`). The `P16U4`/`P8U8` contrast rows confirm identical
`LD_rec = 65`, `LD_eff = 67`, and `p_agg = 11`. In this CP-bound case, read `K` as
the recommended representative of the tie rather than as a resource-saturation
knee; the source `LOOM_PARALLEL(4)` is simply another point in the equivalence
group.

## Comparing against measured DFG simulator cycles

Use the shared comparison rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For vecsum, `absolute_cgra_lb = 11` is CP-bound (the log-depth reduction tree) and
credits control amortization — the per-element induction that a serial model would
charge is gone. It sits below the scalar Metric-1 aggregate in the kernel's main
`## CGRA-Constrained Model` section, which charges induction per iteration.
