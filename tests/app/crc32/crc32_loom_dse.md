# CRC32 Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the `crc32`-specific
setup, helper output, and recommendation.

Kernel: `tests/app/crc32/crc32.cpp` — update one CRC state across all words,
bytes, and bits.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py crc32 --config 6x6 --max-parallel 8 --max-unroll 8 --top 16
```

## CRC32-specific setup

- Fixture: `N=256` and `K=4065` true polynomial-XOR bit iterations, matching
  `main.cpp` and `crc32_eval.md`.
- The bit result carries into the next bit, byte, and word. This non-associative
  recurrence makes outer `i` sequential despite the source pragma; `P>1` is
  illegal and unroll labels cannot flatten the carry.
- The helper reuses the complete concrete source trace. Full-trip counts are
  `A=51682`, `LD_rec=LD_eff=18945`, `ST=19971`, `CP=50152`; the critical path
  dominates the aggregate resource terms.

## Results (`--top 16`)

```text
# Loom pragma DSE (lane-aware + vector coalescing): crc32  (6x6)

loop nest (outer->inner): i[256,sequential]
coalescing: crc is a non-associative carried state across all bytes and bits. Parallel factors are illegal and unroll cannot flatten the trace, so every displayed U choice is an equivalent alias of the same fully consumed serial DAG.

absolute_cgra_lb = 50152  (full-trip, fully-coalesced, invariant-amortized aggregate over full lanes L=12,S=12; the ONLY lower bound)
full-trip counts: A=51682 LD_rec=18945 LD_eff=18945 ST=19971 CP=50152 | compute=1436 load=1579 store=1665   (load term = ceil(LD_rec/L); invariants amortized)
binding class (full trip) = S   (P_pe=36, L=12, S=12; V=4 64-bit elems/vec)

Only absolute_cgra_lb is a lower bound. pragma_agg / sched_est assume waves do NOT overlap and sit at or above it.
aL = active load lanes = min(recurring loads, L): the recurring loop loads set the lane exposure and the binding load term. LD_eff = recurring + one-time invariant loads (total traffic); invariant loads (loaded once and held) are amortized out of the binding term.
Algorithmic arith/CP is a global pool (P and U tie there). P and U separate on TWO axes, both favoring LOOM_UNROLL: (1) control amortization -- unroll shares one iterator across U bodies, so control ops scale as trip/U (parallel keeps an iterator per worker); (2) vector coalescing of contiguous accesses (bounded by V, gone once U>=V). Sequential carries keep per-iter control on CP.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        i:P1U1  (+3 eq)               1  12  12  18945   256     1 50152   50152   50152 latency-bound        3/3/3

RECOMMENDED: i:P1U1  -> exposure=256, pragma_agg=50152 (1.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U contrast: no parallelizable level.
```

## Recommendation

Use **`i:P1U1`**. The other enumerated unroll labels collapse into the same
serial DAG (`+3 eq`), and `p_agg = sched = absolute_cgra_lb = 50152` because
there are no independent checksum waves to expose. A parallel split would race
or duplicate the single carried CRC state.
