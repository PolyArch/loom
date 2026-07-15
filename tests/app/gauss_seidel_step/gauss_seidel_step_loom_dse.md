# Gauss-Seidel Step Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
`gauss_seidel_step`-specific setup, helper output, and recommendation.

Kernel: `tests/app/gauss_seidel_step/gauss_seidel_step.cpp` - update one
Gauss-Seidel sweep in row order.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py gauss_seidel_step --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 16
```

## Gauss-Seidel-specific setup

- Fixture: `N=32`, matching `main.cpp` and `gauss_seidel_step_eval.md`.
- The lower-triangle read of `output_x[j]` makes row `i` depend on earlier row
  stores. The source `LOOM_PARALLEL()` and `LOOM_UNROLL(8)` annotations cannot
  remove this RAW chain, so `P>1` is illegal and equivalent unroll labels use a
  canonical `P1U1` representative.
- The two `j` sums are fully consumed associative row reductions. All terms
  except the newest `output_x[i-1]` term can reduce while the row waits; the
  final combine preserves the six-cycle row recurrence and `CP=198`.
- The read-only `input_x` vector is loaded once as eight invariant vectors.
  Independent contiguous `input_A` row segments and already-ready
  lower-triangle `output_x[0..i-2]` prefixes coalesce; the newest
  `output_x[i-1]` read and sequential output store remain scalar.
- Full-trip DSE counts are `A=3136`, `LD_rec=527`, `LD_eff=536`, `ST=64`,
  and `CP=198`; `absolute_cgra_lb=198`, critical-path-bound.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): gauss_seidel_step  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[32,sequential]`; the outer i loop is a true in-place Gauss-Seidel recurrence: row i reads output_x values written by earlier rows, so parallel factors are illegal and unroll labels cannot flatten the sweep. The lower and upper j sums are fully consumed associative reductions with no j-loop control. The read-only input_x vector is loaded once and held; independent contiguous input_A row segments and already-ready lower-triangle output_x prefixes coalesce. The newest output_x[i-1] read and sequential output_x stores stay scalar to preserve the row recurrence. Equivalent i-unroll labels use the canonical P1U1 representative. Full-trip counts are `A=3136`, `LD_rec=527`, `LD_eff=536`, `ST=64`, and `CP=198`, giving the only lower bound, `absolute_cgra_lb=198=max(CP 198, compute 88, load 44, store 6)`, with critical-path pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        i:P1U1                        1  12  12    536    32     1   198     198     198 latency-bound      44/22/3

RECOMMENDED: i:P1U1  -> exposure=32, pragma_agg=198 (1.00x the floor), latency-bound best-estimate fallback
flags: K=recommended (smallest split reaching the best estimate; no resource-bound knee), b=higher wave-serialized estimate.

P-vs-U contrast: no parallelizable level.

4x4 recommendation: i:P1U1.
8x8 recommendation: i:P1U1.
```

## Recommendation

Use **`i:P1U1`** as the canonical legal representation. The complete sweep is
already one carried DAG, so `p_agg = sched = absolute_cgra_lb = 198`; exposing
more rows cannot create independent work without violating Gauss-Seidel's
in-place dependence. The source unroll hint does not change that legality.
