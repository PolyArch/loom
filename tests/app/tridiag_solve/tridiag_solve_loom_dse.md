# tridiag_solve Loom-Pragma DSE (lane-aware + vector coalescing) — the counterexample

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps only the tridiag-specific
setup, helper output, and recommendation.

Kernel: `tests/app/tridiag_solve/tridiag_solve.cpp` (forward elimination sweep)

```cpp
for (uint32_t i = 1; i < N; i++) {
    float m = input_b[i] - input_a[i] * c_prime[i - 1];   // needs c_prime[i-1]
    c_prime[i] = input_c[i] / m;
    d_prime[i] = (input_d[i] - input_a[i] * d_prime[i - 1]) / m;  // needs d_prime[i-1]
}
```

This kernel **does not demonstrate the P-vs-U distinction**, and it is included
to show *why* under the lane-aware model. It is the complement of
[`vecsum`](../vecsum/vecsum_loom_dse.md):

- `vecsum` carries `sum` but the carry is **associative**, so `LOOM_REDUCE(+)`
  legalizes parallel workers (the escape hatch for a loop-carried dependence).
- `tridiag_solve` carries `c_prime`/`d_prime` through a **non-associative
  division chain**. There is no reduction to exploit and no independent
  iterations. `LOOM_PARALLEL` is illegal and `LOOM_REDUCE` does not apply.

The source may *write* `LOOM_PARALLEL()`, but the carry makes it a no-op: the
model never enumerates a parallel factor `> 1` and forces `P_tot = 1`. The spec
names this family explicitly (non-associative recurrences: `tridiag_solve`,
`trsv_lower/upper`, `gauss_seidel_step`, `kmp_table`).

Model: the lane-aware + vector-coalescing Loom-pragma DSE in
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py tridiag_solve --config 6x6 --top 24
```

## Why there is no distinction

The lane-aware model separates `LOOM_PARALLEL` from `LOOM_UNROLL` through two
DSE-specific effects: vector coalescing of contiguous memory groups, and
control-overhead amortization when spatial unroll shares one iterator across
multiple bodies. Arithmetic (`P_pe`) and the critical path `CP` remain a single
global pool for the kernel's intended math. For this forward sweep one effect is
**inapplicable** and the other is **inert**:

- **`LOOM_PARALLEL` is illegal → `P_tot = 1`.** The forward sweep carries
  `c_prime[i-1]`/`d_prime[i-1]` through a division chain, so there is no
  parallelizable level and no `p` to vary. `LOOM_UNROLL(U)` is the only legal
  knob.
- **Control amortization does not apply; coalescing is real but inert.** A serial
  recurrence cannot be spatially flattened, so unrolling it is a no-op here — the
  model charges the iterator **per iteration** (the sequential exception), and
  there is no shared-iterator saving for `U` to earn. Coalescing, by contrast, is
  real: the contiguous `a`/`b`/`c`/`d` streams *do* fuse into 256-bit vector ops
  (`V = 4`), 64 vector loads in all. Those plus the 64 un-amortized per-iteration
  iterator reads give `LD_rec = 128` (`LD_eff = 130`, `ST = 96`) — loads and
  stores that already sit comfortably inside the machine lanes (`load = 11`,
  `store = 8`, both well under `L = S = 12`). But the aggregate is set by the
  **serial critical path** `CP = 194 = absolute_cgra_lb` -- the `~3*(N-1)`
  division-chain depth -- which no coalescing and no pragma choice can shorten.

Because `CP` dominates, every unroll factoring lands on the identical
latency-bound aggregate, so the tool collapses them into one equivalence group.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): tridiag_solve  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[64,sequential]`; The forward sweep carries a NON-associative recurrence (division chain): LOOM_PARALLEL is illegal (p forced to 1) and the serial CP dominates. Input streams coalesce but it does not matter -> the kernel stays critical-path bound with no P-vs-U distinction. Full-trip counts are `A=512`, `LD_rec=128`, `LD_eff=130`, `ST=96`, and `CP=194`, giving the only lower bound, `absolute_cgra_lb=194=max(CP 194, compute 15, load 11, store 8)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        i:P1U1                        1  12  12    130    64     1   194     194     194 latency-bound        8/6/4

RECOMMENDED: i:P1U1  -> exposure=64, pragma_agg=194 (1.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U contrast: no parallelizable level.
```

All `U` factorings are semantically equivalent, so the helper displays only the
canonical `P1U1` representative.

- `absolute_cgra_lb = 194` is **CP-bound** (the serial recurrence), not
  resource-bound. Among the resource classes the binding one is `P`
  (`compute = 15`), but even that sits far below `CP`, so all 12 load and 12
  store lanes go under-utilized: `util P/L/S = 8/6/4`, every class `< 100%`,
  the signature of a **latency-bound** wave (`cagg = p_agg = sched = 194 = CP`).
- The recommendation is `i:P1U1` at exposure 64, `pragma_agg = 194`, exactly
  **1.00× the floor**. Because the aggregate is pinned to `CP`, the selected
  representative is the smallest legal exposure; adding unroll buys no estimated
  gain.

## Takeaway

Under the lane-aware model, three conditions make a kernel unable to show the
P-vs-U distinction; here two of them fire and the third makes the point moot:

1. **A non-associative carried dependence forbids parallel workers**
   (`P_tot = 1`), so there is no `P` to vary against `U`.
2. **The one applicable split effect is off the binding path.** The contiguous
   `a`/`b`/`c`/`d` streams coalesce (unroll's other lever, control amortization,
   does not apply to a serial recurrence), but the resulting memory term already
   sits far below `CP`, so shaving it changes nothing.
3. **The aggregate is set by `CP`**, which lives in the global arithmetic/
   dependence pool. When `CP` binds, exposure is irrelevant by construction.

Contrast with the earlier banking-era eval, which reached the same "no
distinction" verdict for a *different* reason: it modeled a single worker as a
single memory port (`active_L = active_S = 1`) and reported the loads
serializing to a **resource-bound** `322`. The lane-aware model discards that
port assumption — full lanes and vector coalescing bring loads/stores well under
capacity — so the kernel is now correctly **latency-bound at `CP = 194`**. The
conclusion survives the model change, but the mechanism is now the critical
path, with coalescing real yet inert, rather than a lack of banking. A kernel
that is merely **compute-bound** (arithmetic dominates but is still parallel)
can still show smaller non-binding load/store/control terms for unroll-heavy
splits; those improvements matter only if they move the binding aggregate.
