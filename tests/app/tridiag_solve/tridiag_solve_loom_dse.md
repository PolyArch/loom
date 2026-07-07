# tridiag_solve Loom-Pragma DSE (lane-aware + vector coalescing) — the counterexample

> **Objective:** minimize cycles subject to the hard ≤12 load-lane / ≤12
> store-lane per-cycle limit; the recommendation is the **lane-saturation knee** —
> the smallest exposure whose coalesced traffic saturates the binding resource.
> (Here the serial critical path binds before any lane does, so the knee is the
> floor `i:P1U1`.) Not an area or control/body tradeoff.

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

Regenerate: `python3 tests/scripts/loom_dse.py tridiag_solve --config 6x6`

## Why there is no distinction

The lane-aware model separates `LOOM_PARALLEL` from `LOOM_UNROLL` only through
**vector coalescing** on the load/store axis; arithmetic (`P_pe`) and the
critical path `CP` are a single **global pool** that does not care whether
exposure came from `p` or `U`. Two facts kill the distinction here:

- **`LOOM_PARALLEL` is illegal → `P_tot = 1`.** The forward sweep carries
  `c_prime[i-1]`/`d_prime[i-1]` through a division chain, so there is no
  parallelizable level and no `p` to vary. `LOOM_UNROLL(U)` is the only legal
  knob.
- **Coalescing is real but inert.** The input streams `a`, `b`, `c`, `d` are
  contiguous over `i`, so unrolled adjacent loads/stores *do* coalesce into
  256-bit vector ops (`V = 4`). This is why the full-trip lane-slot counts drop
  to `LD = 130`, `ST = 96` (from the uncoalesced scalar counts), and why loads
  and stores fit comfortably inside the machine lanes (`load = 11`,
  `store = 8`, both well under `L = S = 12`). But the aggregate is set by the
  **serial critical path** `CP = 194 = absolute_cgra_lb` — the `≈ 3·(N−1)`
  division-chain depth — which no coalescing and no pragma can shorten. The
  coalescing that would normally bias toward `LOOM_UNROLL` is exercised, then
  swamped by `CP`.

Because `CP` dominates, every unroll factoring lands on the identical
latency-bound aggregate, so the tool collapses them into one equivalence group.

## Results (the entire sweep is one row)

```text
absolute_cgra_lb = 194  (full-trip, fully-coalesced aggregate over full lanes L=12,S=12; the ONLY lower bound)
full-trip counts: A=512 LD=130 ST=96 CP=194 | compute=15 load=11 store=8
binding class (full trip) = P   (P_pe=36, L=12, S=12; V=4 64-bit elems/vec)

flags    split                      Ptot  aL  aS   exp   wav  cagg   p_agg   sched class           util P/L/S
-------------------------------------------------------------------------------------------------------------
K        i:P1U1  (+3 eq)               1  12  12    64     1   194     194     194 latency-bound        8/6/4

RECOMMENDED: i:P1U1  -> exposure=64, pragma_agg=194 (1.00x the floor), latency-bound
P-vs-U contrast: no parallelizable level.
```

All four `U` factorings collapse to the same row (`+3 eq`) — there is nothing to
choose.

- `absolute_cgra_lb = 194` is **CP-bound** (the serial recurrence), not
  resource-bound. Among the resource classes the binding one is `P`
  (`compute = 15`), but even that sits far below `CP`, so all 12 load and 12
  store lanes go under-utilized: `util P/L/S = 8/6/4`, every class `< 100%`,
  the signature of a **latency-bound** wave (`cagg = p_agg = sched = 194 = CP`).
- The recommendation is `i:P1U1` at exposure 64, `pragma_agg = 194`, exactly
  **1.00× the floor**. It is flagged `K` (the saturation knee `E_sat`): once the
  aggregate is pinned to `CP`, the knee is at the smallest exposure, and adding
  unroll buys no estimated gain.

## Takeaway

Under the lane-aware model, three conditions make a kernel unable to show the
P-vs-U distinction; here two of them fire and the third makes the point moot:

1. **A non-associative carried dependence forbids parallel workers**
   (`P_tot = 1`), so there is no `P` to vary against `U`.
2. **Coalescing — the only axis that separates `P` from `U`** — is *available*
   (the contiguous `a`/`b`/`c`/`d` streams coalesce, cutting `LD`/`ST` to
   `130`/`96` lane-slots) but **inert**: the memory terms already sit far below
   the lanes, so shaving them changes nothing.
3. **The aggregate is set by `CP`**, which lives in the global arithmetic/
   dependence pool that this load/store-focused model treats identically for
   `LOOM_PARALLEL` and `LOOM_UNROLL`. When `CP` binds, exposure is irrelevant by
   construction.

Contrast with the earlier banking-era eval, which reached the same "no
distinction" verdict for a *different* reason: it modeled a single worker as a
single memory port (`active_L = active_S = 1`) and reported the loads
serializing to a **resource-bound** `322`. The lane-aware model discards that
port assumption — full lanes and vector coalescing bring loads/stores well under
capacity — so the kernel is now correctly **latency-bound at `CP = 194`**. The
conclusion survives the model change, but the mechanism is now the critical
path, with coalescing real yet inert, rather than a lack of banking. A kernel
that is merely **compute-bound** (arithmetic dominates but is still parallel)
hits only condition (3): `P` and `U` become interchangeable because coalescing
touches only the load/store axis — `batchnorm` sits near that boundary.
