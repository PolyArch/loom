# Binary Search Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps only the binary_search-specific
setup, helper output, and recommendation.

Kernel: `tests/app/binary_search/binary_search.cpp` — for each of `M` targets,
binary-search a sorted array of `N` and record the found index (or `0xFFFFFFFF`).

Current source pragma:

```cpp
LOOM_NO_PARALLEL
LOOM_NO_UNROLL
for (uint32_t t = 0; t < M; t++) {
    float target = input_targets[t];
    int32_t left = 0;
    int32_t right = static_cast<int32_t>(N) - 1;
    int32_t result = -1;

    while (left <= right) {                       // data-dependent termination
        int32_t mid = left + (right - left) / 2;
        if (input_sorted[mid] == target) { result = mid; break; }
        else if (input_sorted[mid] < target) left = mid + 1;   // carries left
        else right = mid - 1;                                   // carries right
    }
    output_indices[t] = (result == -1) ? 0xFFFFFFFF : (uint32_t)result;
}
```

This uses the shared lane-aware + vector-coalescing DSE from
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py binary_search --config 6x6 --top 0
```

## Why this kernel does not demonstrate a P-vs-U distinction

Like [`tridiag_solve`](../tridiag_solve/tridiag_solve_loom_dse.md), binary_search
is a **counterexample**: it is included to show *why* the lane-aware model's
P-vs-U levers cannot help here. It is nested — an outer `t` loop and an inner
`probe` level — but neither the outer split nor the inner recurrence exposes a
knob that moves the binding aggregate:

- **The outer `t` loop is genuinely parallel, so the DSE does enumerate a split.**
  Each of the `M = 5` target searches is logically independent (private `target`,
  `left`, `right`, `result`; distinct `output_indices[t]`), so the model treats `t`
  as parallel and sweeps `LOOM_PARALLEL`/`LOOM_UNROLL` over it. This is the axis
  that fails to matter, not an axis that is illegal.
- **The inner `probe` level is a data-dependent-termination serial recurrence.**
  The `while (left <= right)` loop carries `left`/`right` through a
  **non-associative, data-dependent** update; there is no reduction to exploit and
  no independent probes. `probe` is modeled at its worst-case trip
  `ceil(log2(N+1)) = 4`, with `P` forced to 1. The termination compare
  (`left <= right`) and the data-dependent load `input_sorted[mid]` sit on the
  critical path every probe. Because `mid` is a **non-affine (data-dependent)**
  index, `input_sorted[mid]` is a **scalar load that cannot coalesce**.
- **The aggregate is set by `CP`, and `CP` is the serial probe chain.** The
  per-probe carry chain (`sub → shift → add_mid → load sorted[mid] → cmp_lt →
  update`) runs 4 times, giving `CP = 27`. The whole-kernel arithmetic
  (`compute = 5`), loads (`load = 3`), and stores (`store = 1`) are tiny — `M = 5`
  targets is a very small problem — so every resource term sits far below `CP`.
  When `CP` binds, the outer `t` split is irrelevant by construction.

The source writes `LOOM_NO_PARALLEL` / `LOOM_NO_UNROLL`. This DSE explores the
P/U space anyway (as it does for every kernel), but it does **not** model control
divergence across parallel lanes — the real reason the source forbids
parallelizing divergent, data-dependent searches (each lane would take a
different number of probes and a different branch). Even setting divergence
aside, the exploration confirms the kernel is CP-bound, so a wider fabric or a
different P/U split gives no modeled throughput benefit here.

## Setup

- DSE fixture: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`), `V = 4`. Sizes are the
  smoke-test fixture: `N = 10` (sorted array), `M = 5` (targets).
- Loop nest: outer `t[5, parallel]`, inner `probe[4, sequential]` (data-dependent
  termination, worst-case `ceil(log2(N+1)) = 4` probes; `P` forced to 1).
- `left`/`right` are threaded as dataflow (no per-probe scalar round-trip); the
  termination compare and `input_sorted[mid]` (non-affine index, no coalescing)
  stay on `CP` each probe.
- `output_indices[t]` is contiguous over `t`; `N`/`M`/`right_init` (`= N-1`) are
  loop-invariant and amortized. `input_targets[t]` is a recurring per-target scalar.
- Full-trip counts: `A = 152`, `LD_rec = 26`, `LD_eff = 29`, `ST = 3`, `CP = 27`.
  Thus `absolute_cgra_lb = max(CP 27, compute 5, load 3, store 1) = 27`, CP-bound;
  it is the only lower bound. Binding class is `P` (compute), but even that term
  (`5`) sits far below `CP`.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): binary_search  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `t[5,parallel], probe[4,sequential]`; outer PARALLEL t (independent target searches), inner SEQUENTIAL while with DATA-DEPENDENT termination (worst-case ceil(log2(N+1))=4 probes). The left/right recurrence is threaded as dataflow and its termination compare sits on CP per probe; input_sorted[mid] is a non-affine (data-dependent) scalar load that cannot coalesce. This is a COUNTEREXAMPLE like tridiag: the per-target serial recurrence and a tiny problem (M=5 targets) leave it CP/latency-bound, so no P-vs-U split helps. The source LOOM_NO_PARALLEL/LOOM_NO_UNROLL reflects control divergence, which this DSE does not model. Full-trip counts are `A=152`, `LD_rec=26`, `LD_eff=29`, `ST=3`, and `CP=27`, giving the only lower bound, `absolute_cgra_lb=27=max(CP 27, compute 5, load 3, store 1)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
b        t:P4U1 probe:P1U1             4  12   8     27    16     2    27      54      54 latency-bound       15/7/4
b        t:P2U2 probe:P1U1             2  12   4     25    16     2    27      54      54 latency-bound       15/7/4
K        t:P1U4 probe:P1U1             1  12   2     24    16     2    27      54      54 latency-bound       15/7/4
b        t:P2U1 probe:P1U1             2  12   4     15     8     3    27      81      81 latency-bound        7/4/4
b        t:P1U2 probe:P1U1             1  11   2     14     8     3    27      81      81 latency-bound        7/4/4
b        t:P1U1 probe:P1U1             1   6   2      9     4     5    27     135     135 latency-bound        4/4/4

RECOMMENDED: t:P1U4 probe:P1U1  -> exposure=16, pragma_agg=54 (2.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 4 on level 't' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P4U1              24     27     8      54 tie (control/coalescing sit below the binding term)
  P2U2              22     25     4      54 tie (control/coalescing sit below the binding term)
  P1U4              21     24     2      54 tie (control/coalescing sit below the binding term)
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Recommendation and reading

**`t:P1U4 probe:P1U1` is reported as recommended, but note it is
latency-bound** (`p_agg = 54`, exactly `2.00×` the `27` floor), not a
resource-saturation knee. No split in the sweep saturates any resource — the
binding utilization tops out at `util P/L/S = 15/7/4`, every class well under
`100%` — so there is no `E_sat` to land on. The tool falls back to the best
estimate: "recommended" here means the best of a set of equally-latency-bound
options, not a knee.

Every row is `latency-bound` at `cagg = 27`. Increasing the outer `t` exposure
only reduces the **wave count** (`p_agg = waves × 27`): `P1U1` needs 5 waves
(`135`), `P2U1` needs 3 (`81`), and the largest enumerated exposure (4) needs 2
(`54`). The helper enumerates only **power-of-two** factors, so `t`'s exposure tops
out at `4` (`P4U1`/`P2U2`/`P1U4`) and never covers all `5` targets in a single wave
— a wave-rounding artifact of a tiny, non-power-of-two trip. Within *any* wave
`cagg` stays pinned to `CP = 27`, so across the enumerated sweep `p_agg` never drops
below the `2.00×` floor. (A hypothetical non-power-of-two full unroll `t:P1U5` would
cover all 5 targets in one wave and reach `p_agg = 27`, i.e. `1.00×` — but the sweep
does not enumerate it, and it would not change the conclusion that the kernel is
CP-bound.)

The **P-vs-U-at-fixed-product-4** block makes the counterexample explicit: `P4U1`,
`P2U2`, and `P1U4` all **tie** at `p_agg = 54`. Unroll's two levers do fire —
`LD_rec` falls `24 → 21` (row-iterator amortization) and `ST` falls `8 → 2`
(`output_indices[t]` coalescing) as the split shifts from parallel to unroll —
but both terms sit far below the CP-binding term, so the tie is exact: the split
is legal and the levers work, yet none of it reaches the binding aggregate.

**Takeaway.** binary_search is dependency-bound throughout; only shortening the
data-dependent recurrence (fewer probes) would help. It complements
`tridiag_solve` — a non-associative *sequential* recurrence — by adding the
**data-dependent-termination** dimension: the carry is not just serial but its
trip count and branch are input-dependent, and the worst-case probe count sets
the floor.

## Comparing against measured DFG simulator cycles

Measured DFG simulator comparisons should use the shared rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).

Note the deliberate gap from the ASAP `binary_search_eval.md`: that model reports
`CP = 48` because it adds three no-predication compare→body gaps per probe (a
10-cycle body). This DSE does **not** model predication gating, so its per-probe
carry chain is ~6 cycles and `CP = 27` — a more optimistic figure. Both models
agree on the conclusion: binary_search is dependency-bound and fabric width is
irrelevant; they differ only on how conservatively the per-probe body is costed.
For this kernel the DSE floor credits `output_indices` coalescing and iterator
amortization on the outer `t` level, so treat `absolute_cgra_lb = 27` as this
optional DSE floor rather than a scalar CGRA aggregate.

## Note on full outer unroll

For this fixed `M = 5` fixture, a hypothetical non-power-of-two outer unroll
`t:P1U5` would be optimal in this DSE model: it exposes all five independent
target searches in one wave, so `waves = 1` and `p_agg = 27`. This is optimal
because it removes the second wave, not because it shortens any individual
binary search. Each target's `while` loop still follows its own serial
left/right recurrence. Any legal split with `P_tot * U >= 5` would tie on cycles
in the model; `t:P1U5` is simply the unroll-heavy version, and the current helper
does not list it because it enumerates only power-of-two factors. The source
still uses `LOOM_NO_PARALLEL` / `LOOM_NO_UNROLL`, so this remains a hypothetical
DSE comparison unless those pragmas change.
