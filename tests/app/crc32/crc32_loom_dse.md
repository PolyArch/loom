# CRC32 Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the `crc32`-specific
setup, helper output, and recommendation.

Kernel: `tests/app/crc32/crc32.cpp` — update one CRC state across all words,
bytes, and bits.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py crc32 --config 6x6 --top 16
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

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): crc32  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[256,sequential]`; crc is a non-associative carried state across all bytes and bits. Parallel factors are illegal and unroll cannot flatten the trace, so equivalent unroll labels use the canonical P1U1 representative for the fully consumed serial DAG. Full-trip counts are `A=51682`, `LD_rec=18945`, `LD_eff=18945`, `ST=19971`, and `CP=50152`, giving the only lower bound, `absolute_cgra_lb=50152=max(CP 50152, compute 1436, load 1579, store 1665)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        i:P1U1                        1  12  12  18945   256     1 50152   50152   50152 latency-bound        3/3/3

RECOMMENDED: i:P1U1  -> exposure=256, pragma_agg=50152 (1.00x the floor), latency-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U contrast: no parallelizable level.
```

## Recommendation

Use **`i:P1U1`**. Every unroll label maps to the same serial DAG, so the helper
uses the canonical representative. `p_agg = sched = absolute_cgra_lb = 50152`
because there are no independent checksum waves to expose. A parallel split
would race or duplicate the single carried CRC state.
