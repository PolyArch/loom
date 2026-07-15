# Interpolate Linear Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
`interpolate_linear`-specific setup, helper output, and recommendation.

Kernel: `tests/app/interpolate_linear/interpolate_linear.cpp` — independently
search and interpolate one output for each query point.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py interpolate_linear --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 16
```

## Interpolate-linear-specific setup

- Fixture: `N_data=32`, `N_query=64`, `input_x[i]=i`, and
  `input_xq[q]=0.5*q`, matching `main.cpp` and `interpolate_linear_eval.md`.
- The 64 `q` lanes are independent. Their private sequential `k` searches
  execute 1,024 probes total: 63 lanes break on a hit, while `q=63`
  (`xq=31.5`) executes 31 failed probes and the final failing bound check.
- The helper uses the same deterministic wave-averaged concrete-trace convention
  as `clz`: it sums each candidate's actual contiguous query waves, then rounds
  one representative chunk upward. Full exposure is exact.
- Only `input_xq[q]` and `output_yq[q]` coalesce across adjacent q-unrolled
  lanes. The conditionally executed search loads and indirect interpolation
  loads remain recurring scalar traffic. The model assumes no cross-query
  cache/broadcast reuse of read-only `input_x`/`input_y`; unlike `gemv`'s fixed
  affine `x[j]` set, these accesses depend on private search termination or the
  data-dependent selected `i`.
- Full-trip DSE counts are `A=5573`, `LD_rec=3410`, `LD_eff=3412`,
  `ST=1105`, `CP=289`; `absolute_cgra_lb=289`, critical-path-bound.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): interpolate_linear  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `q[64,parallel]`; q lanes are independent, but each lane keeps its private sequential data-dependent k search and selected i state. The helper uses the concrete main.cpp trace (1024 probes: 63 hits and one final-check no-hit lane) and the same deterministic wave-average convention as clz. q-unroll coalesces only contiguous input_xq[q] / output_yq[q] boundary traffic and amortizes q control. Search input_x[k] loads and tail input_x/input_y[i] loads remain recurring scalar accesses: conditional termination and data-dependent indices prevent vector coalescing, and no cross-query cache/broadcast reuse is assumed. Full-trip counts are `A=5573`, `LD_rec=3410`, `LD_eff=3412`, `ST=1105`, and `CP=289`, giving the only lower bound, `absolute_cgra_lb=289=max(CP 289, compute 155, load 285, store 93)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
         q:P32U2                      32  12  12   3459    64     1   289     289     289 resource-bound   54/100/33
         q:P16U4                      16  12  12   3427    64     1   289     289     289 latency-bound     54/99/33
         q:P8U8                        8  12  12   3419    64     1   289     289     289 latency-bound     54/99/32
         q:P4U16                       4  12  12   3415    64     1   289     289     289 latency-bound     54/99/32
         q:P2U32                       2  12  12   3413    64     1   289     289     289 latency-bound     54/99/32
K        q:P1U64                       1  12  12   3412    64     1   289     289     289 latency-bound     54/99/32
b        q:P64U1                      64  12  12   3523    64     1   294     294     294 resource-bound   54/100/35
b        q:P32U1                      32  12  12   1763    32     2   220     440     440 latency-bound     36/67/23
b        q:P16U2                      16  12  12   1731    32     2   220     440     440 latency-bound     36/66/22
b        q:P8U4                        8  12  12   1715    32     2   220     440     440 latency-bound     35/65/21
b        q:P4U8                        4  12  12   1711    32     2   220     440     440 latency-bound     35/65/21
b        q:P2U16                       2  12  12   1709    32     2   220     440     440 latency-bound     35/65/21
b        q:P1U32                       1  12  12   1708    32     2   220     440     440 latency-bound     35/65/21
b        q:P16U1                      16  12  12    883    16     4   186     744     744 latency-bound     22/40/14
b        q:P8U2                        8  12  12    867    16     4   186     744     744 latency-bound     22/39/13
b        q:P4U4                        4  12  12    859    16     4   186     744     744 latency-bound     21/39/13
... (12 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: q:P1U64  -> exposure=64, pragma_agg=289 (1.00x the floor), latency-bound best-estimate fallback
flags: K=recommended (smallest split reaching the best estimate; no resource-bound knee), b=higher wave-serialized estimate.

P-vs-U at fixed product 64 on level 'q' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P64U1            3521   3523  1216     294 1.02x slower (parallel: extra iterators + strided, no coalesce)
  P32U2            3457   3459  1152     289 best
  P16U4            3425   3427  1120     289 best
  P8U8            3417   3419  1112     289 best
  P4U16           3413   3415  1108     289 best
  P2U32           3411   3413  1106     289 best
  P1U64           3410   3412  1105     289 best

4x4 recommendation: q:P1U64.
8x8 recommendation: q:P1U64.
```

## Recommendation

Use **`q:P1U64`** as the lowest-traffic member of the best 289-cycle group. The
best-coalesced path never reaches a resource-saturation knee: even at full
exposure, the load term is 285 cycles and the 289-cycle no-hit recurrence still
sets the floor. Full q-unroll nevertheless reaches that floor with one residual
q iterator, 16 vector boundary loads, and 16 vector boundary stores. `P32U2`
through `P2U32` tie at `p_agg = sched = 289`, but retain more iterator and
boundary lane-slot traffic; `P64U1` loses the coalescing/control benefit and
rises to 294 cycles.
