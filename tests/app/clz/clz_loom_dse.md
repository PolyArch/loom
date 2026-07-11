# CLZ Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the `clz`-specific
setup, helper output, and recommendation.

Kernel: `tests/app/clz/clz.cpp` — count leading zeros independently for each
32-bit input word. The outer `i` loop is `LOOM_PARALLEL()` + `LOOM_UNROLL(8)`;
each lane contains a private data-dependent sequential `while` recurrence.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py clz --config 6x6 --max-parallel 8 --max-unroll 8 --top 16
```

## CLZ-specific setup

- Fixture: `N = 256`, one zero lane, 255 nonzero lanes, `sum(K) = 3211`, and
  `max(K) = 31`, matching `main.cpp`.
- Only outer `i` is parallelizable. The `mask` shift, `count` update, and exit
  test form a per-lane sequential recurrence; the helper preserves the concrete
  branch/trip distribution and sums contiguous waves before forming its
  representative per-wave counts and critical path.
- Contiguous `i` unroll coalesces `input_data` and `output_count` in groups of
  `V = 4`; coalescing does not cross parallel workers. Outer control amortizes
  once per worker per wave, while the data-dependent scan remains scalar.
- Full-trip counts are `A = 13612`, `LD_rec = 6997`, `LD_eff = 6998`,
  `ST = 6997`, `CP = 163`; therefore `absolute_cgra_lb = 584`, load/store-bound.

## Results (`--top 16`)

```text
# Loom pragma DSE (lane-aware + vector coalescing): clz  (6x6)

loop nest (outer->inner): i[256,parallel]
coalescing: outer i is parallel; each lane has a private data-dependent while recurrence whose trip count is the concrete main.cpp leading-zero count. Contiguous i-unroll coalesces boundary input/output traffic and amortizes outer control, while the longest K=31 lane keeps CP at 163 once exposed.

absolute_cgra_lb = 584  (full-trip, fully-coalesced, invariant-amortized aggregate over full lanes L=12,S=12; the ONLY lower bound)
full-trip counts: A=13612 LD_rec=6997 LD_eff=6998 ST=6997 CP=163 | compute=379 load=584 store=584   (load term = ceil(LD_rec/L); invariants amortized)
binding class (full trip) = L   (P_pe=36, L=12, S=12; V=4 64-bit elems/vec)

Only absolute_cgra_lb is a lower bound. pragma_agg / sched_est assume waves do NOT overlap and sit at or above it.
aL = active load lanes = min(recurring loads, L): the recurring loop loads set the lane exposure and the binding load term. LD_eff = recurring + one-time invariant loads (total traffic); invariant loads (loaded once and held) are amortized out of the binding term.
Algorithmic arith/CP is a global pool (P and U tie there). P and U separate on TWO axes, both favoring LOOM_UNROLL: (1) control amortization -- unroll shares one iterator across U bodies, so control ops scale as trip/U (parallel keeps an iterator per worker); (2) vector coalescing of contiguous accesses (bounded by V, gone once U>=V). Sequential carries keep per-iter control on CP.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        i:P8U8                        8  12  12   1758    64     4   147     588     588 resource-bound  65/100/100
b        i:P8U4                        8  12  12    884    32     8    83     664     664 latency-bound     58/89/89
b        i:P4U8                        4  12  12    880    32     8    83     664     664 latency-bound     58/89/89
b        i:P8U2                        8  12  12    451    16    16    78    1248    1248 latency-bound     32/49/49
b        i:P4U4                        4  12  12    443    16    16    78    1248    1248 latency-bound     31/47/47
b        i:P2U8                        2  12  12    441    16    16    78    1248    1248 latency-bound     31/47/47
b        i:P8U1                        8  12  12    234     8    32    75    2400    2400 latency-bound     17/27/27
b        i:P4U2                        4  12  12    226     8    32    75    2400    2400 latency-bound     17/25/25
b        i:P2U4                        2  12  12    222     8    32    75    2400    2400 latency-bound     16/25/25
b        i:P1U8                        1  12  12    221     8    32    75    2400    2400 latency-bound     16/25/25
b        i:P4U1                        4  12  12    118     4    64    74    4736    4736 latency-bound      9/14/14
b        i:P2U2                        2  12  12    114     4    64    74    4736    4736 latency-bound      9/14/14
b        i:P1U4                        1  12  12    112     4    64    74    4736    4736 latency-bound      8/14/14
b        i:P2U1                        2  12  12     60     2   128    72    9216    9216 latency-bound        6/7/7
b        i:P1U2                        1  12  12     58     2   128    72    9216    9216 latency-bound        6/7/7
b        i:P1U1                        1  12  12     31     1   256    71   18176   18176 latency-bound        3/4/4

RECOMMENDED: i:P8U8  -> exposure=64, pragma_agg=588 (1.01x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 32 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P8U4             883    884   883     664 tie (control/coalescing sit below the binding term)
  P4U8             879    880   879     664 tie (control/coalescing sit below the binding term)
```

## Recommendation

Use **`i:P8U8`** for this bounded sweep. Exposure 64 reaches the first
resource-bound row with `p_agg = sched = 588`, only `1.01x` the 584-cycle
aggregate floor. Smaller exposures remain latency-bound because their
wave-serialized per-wave scan depths dominate. At fixed exposure 32,
`P8U4` and `P4U8` tie in cycles; the helper still reports the small unroll-side
traffic advantage, but it sits below the binding term.
