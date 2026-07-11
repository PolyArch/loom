# Edge Update Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
`edge_update`-specific setup, helper output, and recommendation.

Kernel: `tests/app/edge_update/edge_update.cpp` — copy 16 CSR edge weights, then
find and overwrite the first matching edge for `src=2`, `dst=4`.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py edge_update --config 6x6 --max-parallel 8 --max-unroll 8 --top 16
```

## Edge-update-specific setup

- Fixture: `num_edges=16`, row degree 3, first-match search length `K=2`,
  matching `main.cpp`.
- The source contains no Loom parallel/unroll pragma. The CSR scan is
  data-dependent and first-match ordered; directly parallelizing it can change
  behavior when duplicate destinations exist.
- The helper uses the committed concrete DAG from `cgra_schedule.py`. The copy
  and matched overwrite remain in one schedulable region: the overwrite is a
  WAW on a value that is never read between stores, so it is not a RAW phase
  barrier. Full-trip counts are `A=40`, `LD_rec=LD_eff=38`, `ST=37`, `CP=6`.

## Results (`--top 16`)

```text
# Loom pragma DSE (lane-aware + vector coalescing): edge_update  (6x6)

loop nest (outer->inner): kernel[1,sequential]
coalescing: the source has no Loom parallel/unroll pragma. The copy, bounds check, data-dependent CSR search, and matched overwrite are modeled as the concrete serial kernel trace; there is no legal P/U level to sweep.

absolute_cgra_lb = 6  (full-trip, fully-coalesced, invariant-amortized aggregate over full lanes L=12,S=12; the ONLY lower bound)
full-trip counts: A=40 LD_rec=38 LD_eff=38 ST=37 CP=6 | compute=2 load=4 store=4   (load term = ceil(LD_rec/L); invariants amortized)
binding class (full trip) = L   (P_pe=36, L=12, S=12; V=4 64-bit elems/vec)

Only absolute_cgra_lb is a lower bound. pragma_agg / sched_est assume waves do NOT overlap and sit at or above it.
aL = active load lanes = min(recurring loads, L): the recurring loop loads set the lane exposure and the binding load term. LD_eff = recurring + one-time invariant loads (total traffic); invariant loads (loaded once and held) are amortized out of the binding term.
Algorithmic arith/CP is a global pool (P and U tie there). P and U separate on TWO axes, both favoring LOOM_UNROLL: (1) control amortization -- unroll shares one iterator across U bodies, so control ops scale as trip/U (parallel keeps an iterator per worker); (2) vector coalescing of contiguous accesses (bounded by V, gone once U>=V). Sequential carries keep per-iter control on CP.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        kernel:P1U1                   1  12  12     38     1     1     6       6       6 latency-bound     33/67/67

RECOMMENDED: kernel:P1U1  -> exposure=1, pragma_agg=6 (1.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U contrast: no parallelizable level.
```

## Recommendation

Use **`kernel:P1U1`**, the only legal representative. The 6-cycle dependency
chain is longer than the aggregate compute/load/store terms, so
`p_agg = sched = absolute_cgra_lb = 6`. Any future copy-loop pragma exploration
should be modeled explicitly rather than inferred from this no-pragma source.
