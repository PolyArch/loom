# Histogram Bin Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
`hist_bin`-specific setup, helper output, and recommendation.

Kernel: `tests/app/hist_bin/hist_bin.cpp` - zero the bins, classify the input,
and accumulate one associative bucket per resolved bin.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py hist_bin --config 6x6 --top 16
```

## Histogram-specific setup

- Fixture: `N=1024`, `num_bins=10`, `min_val=0`, `max_val=100`, and
  `input[i]=i%100`, matching `main.cpp` and `hist_bin_eval.md`.
- The annotated `zero_i` loop is the tunable parallel phase. Each candidate
  emits its zero-fill waves, including the partial tail; unroll coalesces
  contiguous bin stores and amortizes one iterator per active worker.
- All zero-fill waves precede one fixed count region. Each bucket reads the zero
  identity written by the first phase, so this is a true RAW barrier, and the
  dominant 1024-input count work is never repeated per zero-fill wave.
- Because the builder emits every zero-fill wave and the fixed count region,
  `cagg` is already phase-composed for this kernel and equals `p_agg`; `wav`
  reports only the number of zero-fill waves.
- Every input is valid. Both range comparisons and the `bin >= num_bins`
  comparison execute, but the clamp assignment is never taken because the
  resolved bins are already in `0..9`.
- The concrete fan-ins are `{110, 110, 104, 100, 100, 100, 100, 100, 100,
  100}`. Each bin is an output-centric associative tree rather than a serial
  read-modify-write chain. Contiguous input loads coalesce; data-dependent
  scatter traffic and memory-backed `bin` scalars remain scalar.
- At the floor-only full exposure `zero_i:P1U10` (not a searched power-of-two
  candidate), full-trip counts are `A=6148`,
  `LD_rec=2305`, `LD_eff=2309`, `ST=2052`, and region-summed `CP=17`; the two
  ordered-region aggregates give `absolute_cgra_lb=193`, load-bound.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): hist_bin  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `zero_i[10,parallel]`; the annotated zero_i loop is parallel; its contiguous output stores coalesce within each unrolled worker, and its phase-local waves include the partial tail. All zero-fill waves are ordered before one fixed 1024-input count region because the later updates read those zero identities. All inputs take the valid guard path; the bin clamp compare executes but its assignment arm is untaken. Concrete per-bin fan-ins are 110,110,104,100,100,100,100,100,100,100 and form associative output-centric trees. Contiguous input loads coalesce, while data-dependent output scatter loads/stores and memory-backed bin scalars remain scalar. The dominant scatter phase has no independent tiled P/U level and executes exactly once for every zero_i candidate. Full-trip counts are `A=6148`, `LD_rec=2305`, `LD_eff=2309`, `ST=2052`, and `CP=17`, giving the only lower bound, `absolute_cgra_lb=193` from the sum of 2 ordered-region aggregates (region-summed CP 17, compute ceilings 172, load ceilings 193, and store ceilings 172); `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
         zero_i:P4U2                   4  12  12   2313     8     2   194     194     263 resource-bound   89/100/89
         zero_i:P2U4                   2  12  12   2311     8     2   194     194     263 resource-bound   89/100/89
K        zero_i:P1U8                   1  12  12   2310     8     2   194     194     263 resource-bound   89/100/89
         zero_i:P4U1                   4  12  12   2318     4     3   195     195     266 resource-bound   89/100/89
         zero_i:P2U2                   2  12  12   2313     4     3   195     195     266 resource-bound   89/100/89
         zero_i:P1U4                   1  12  12   2311     4     3   195     195     266 resource-bound   89/100/89
         zero_i:P8U1                   8  12  12   2318     8     2   195     195     263 resource-bound    89/99/89
         zero_i:P2U1                   2  12  12   2318     2     5   197     197     272 resource-bound   89/100/89
         zero_i:P1U2                   1  12  12   2313     2     5   197     197     272 resource-bound   89/100/89
         zero_i:P1U1                   1  12  12   2318     1    10   202     202     287 resource-bound   90/100/90

RECOMMENDED: zero_i:P1U8  -> exposure=8, pragma_agg=194 (1.01x the floor), phase-composed
flags: K=recommended (smallest tunable-phase exposure that reaches the best phase-composed estimate).

P-vs-U at fixed product 8 on level 'zero_i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P8U1            2314   2318  2068     195 1.01x slower (parallel: extra iterators + strided, no coalesce)
  P4U2            2309   2313  2058     194 best
  P2U4            2307   2311  2054     194 best
  P1U8            2306   2310  2053     194 best
```

## Recommendation

Use **`zero_i:P1U8`**. It is the most unroll-heavy representative at the
smallest legal power-of-two exposure reaching the best phase-composed estimate:
one eight-bin wave plus a two-bin tail gives `p_agg=194`, `1.01x` the 193-cycle
floor, with `sched=263`. `P2U4` and `P4U2` tie on cycles but retain more iterator
and memory traffic; `P8U1` loses vector coalescing and rises to `p_agg=195`. The
fixed bucketed count region executes once for every candidate. `sched` is an
estimate, not a lower bound.
