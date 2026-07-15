# CLZ Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the `clz`-specific
setup, helper output, and recommendation.

Kernel: `tests/app/clz/clz.cpp` — count leading zeros independently for each
32-bit input word. The outer `i` loop is `LOOM_PARALLEL()` + `LOOM_UNROLL(8)`;
each lane contains a private data-dependent sequential `while` recurrence.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py clz --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 16
```

## CLZ-specific setup

- Fixture: `N = 256`, one zero lane, 255 nonzero lanes, `sum(K) = 3211`, and
  `max(K) = 31`, matching `main.cpp`.
    - Important: critical path (CP) is dependent on the input. In this evaluation, it is set by max(K) = 31
- Only outer `i` is parallelizable. The `mask` shift, `count` update, and exit
  test form a per-lane sequential recurrence; the helper preserves the concrete
  branch/trip distribution and sums contiguous waves before forming its
  representative per-wave counts and critical path.
- Contiguous `i` unroll coalesces `input_data` and `output_count` in groups of
  `V = 4`; coalescing does not cross parallel workers. Outer control amortizes
  once per worker per wave, while the data-dependent scan remains scalar.
- Full-trip counts are `A = 13612`, `LD_rec = 6997`, `LD_eff = 6998`,
  `ST = 6997`, `CP = 163`; therefore `absolute_cgra_lb = 584`, load/store-bound.

For a candidate with `W` contiguous waves, the helper defines
`representative_wave_CP = ceil(sum(maximum lane depth in each wave) / W)`.
CLZ needs this representative because its input-dependent lane depths make the
waves unequal, while the DSE evaluates one representative chunk and multiplies
its aggregate by `W`; charging the full-trip `CP = 163` to every wave would
overstate latency, whereas averaging individual lanes would ignore that each
wave waits for its slowest lane.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): clz  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[256,parallel]`; outer i is parallel; each lane has a private data-dependent while recurrence whose trip count is the concrete main.cpp leading-zero count. Contiguous i-unroll coalesces boundary input/output traffic and amortizes outer control, while the longest K=31 lane keeps CP at 163 once exposed. Full-trip counts are `A=13612`, `LD_rec=6997`, `LD_eff=6998`, `ST=6997`, and `CP=163`, giving the only lower bound, `absolute_cgra_lb=584=max(CP 163, compute 379, load 584, store 584)`, with load pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
         i:P2U32                       2  12  12   1752    64     4   146     584     584 resource-bound  65/100/100
K        i:P1U64                       1  12  12   1751    64     4   146     584     584 resource-bound  65/100/100
o        i:P4U32                       4  12  12   3503   128     2   292     584     584 resource-bound  65/100/100
o        i:P2U64                       2  12  12   3501   128     2   292     584     584 resource-bound  65/100/100
o        i:P1U128                      1  12  12   3500   128     2   292     584     584 resource-bound  65/100/100
o        i:P8U32                       8  12  12   7005   256     1   584     584     584 resource-bound  65/100/100
o        i:P4U64                       4  12  12   7001   256     1   584     584     584 resource-bound  65/100/100
o        i:P2U128                      2  12  12   6999   256     1   584     584     584 resource-bound  65/100/100
o        i:P1U256                      1  12  12   6998   256     1   584     584     584 resource-bound  65/100/100
o        i:P16U16                     16  12  12   7013   256     1   585     585     585 resource-bound  65/100/100
o        i:P16U8                      16  12  12   3515   128     2   293     586     586 resource-bound  65/100/100
o        i:P8U16                       8  12  12   3507   128     2   293     586     586 resource-bound  65/100/100
o        i:P32U8                      32  12  12   7029   256     1   586     586     586 resource-bound  65/100/100
         i:P8U8                        8  12  12   1758    64     4   147     588     588 resource-bound  65/100/100
         i:P4U16                       4  12  12   1754    64     4   147     588     588 resource-bound  65/100/100
o        i:P64U4                      64  12  12   7061   256     1   589     589     589 resource-bound  65/100/100
... (29 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: i:P1U64  -> exposure=64, pragma_agg=584 (1.00x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 64 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P64U1            1861   1862  1861     624 1.07x slower (parallel: extra iterators + strided, no coalesce)
  P32U2            1797   1798  1797     600 1.03x slower (parallel: extra iterators + strided, no coalesce)
  P16U4            1765   1766  1765     592 1.01x slower (parallel: extra iterators + strided, no coalesce)
  P8U8            1757   1758  1757     588 1.01x slower (parallel: extra iterators + strided, no coalesce)
  P4U16           1753   1754  1753     588 1.01x slower (parallel: extra iterators + strided, no coalesce)
  P2U32           1751   1752  1751     584 best
  P1U64           1750   1751  1750     584 best

4x4 recommendation: i:P1U32.
8x8 recommendation: i:P1U64.
```

## Recommendation

Use **`i:P1U64`**. Exposure 64 is the first resource-bound power-of-two exposure,
and the unroll-heavy split reaches `p_agg = sched = 584`, exactly the aggregate
floor. `P8U8` exposes the same number of words but reports 588 cycles because
eight workers retain more iterator traffic and cannot coalesce across worker
partitions. Smaller exposures remain latency-bound because their wave-serialized
per-wave scan depths dominate.
