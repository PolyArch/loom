# Edge Update Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
`edge_update`-specific setup, helper output, and recommendation.

Kernel: `tests/app/edge_update/edge_update.cpp` — copy 16 CSR edge weights, then
find and overwrite the first matching edge for `src=2`, `dst=4`.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py edge_update --config 6x6 --top 16
```

## Edge-update-specific setup

- Fixture: `num_edges=16`, row degree 3, first-match search length `K=2`,
  matching `main.cpp`.
- The source contains no Loom parallel/unroll pragma. The CSR scan is
  data-dependent and first-match ordered; directly parallelizing it can change
  behavior when duplicate destinations exist.
- The helper uses the committed concrete DAG from `cgra_schedule.py`. The copy
  loop is included as 16 fully expanded scalar copy iterations, whose independent
  memory operations may use the available lanes and overlap the search. This is
  not an assumed `LOOM_PARALLEL` / `LOOM_UNROLL` split, so the helper does not
  sweep copy-loop pragmas or credit unroll-specific coalescing and control
  amortization. The copy and matched overwrite remain in one schedulable region:
  the overwrite is a WAW on a value that is never read between stores, so it is
  not a RAW phase barrier. Full-trip counts are `A=40`, `LD_rec=LD_eff=38`,
  `ST=37`, `CP=6`.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): edge_update  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `kernel[1,sequential]`; the source has no Loom parallel/unroll pragma. The copy, bounds check, data-dependent CSR search, and matched overwrite are modeled as the concrete serial kernel trace; there is no legal P/U level to sweep. Full-trip counts are `A=40`, `LD_rec=38`, `LD_eff=38`, `ST=37`, and `CP=6`, giving the only lower bound, `absolute_cgra_lb=6=max(CP 6, compute 2, load 4, store 4)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        kernel:P1U1                   1  12  12     38     1     1     6       6       6 latency-bound     33/67/67

RECOMMENDED: kernel:P1U1  -> exposure=1, pragma_agg=6 (1.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U contrast: no parallelizable level.
```

## Recommendation

Use **`kernel:P1U1`** only as the canonical whole-kernel label for this no-sweep
model. It is not a pragma recommendation for the actual search loop, and it does
not assign `P1U1` to the copy loop. The first-match search remains sequential by
dependence, while the fully expanded copy work is included in the operation
counts and schedule without an explicit `P/U` choice. The 6-cycle search/update
dependency chain is longer than the aggregate compute/load/store terms, so
`p_agg = sched = absolute_cgra_lb = 6`. A copy-loop pragma recommendation would
require a separate DSE that exposes the copy loop as a parallelizable level.
