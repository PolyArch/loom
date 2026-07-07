# vecsum Loom-Pragma DSE (lane-aware + vector coalescing)

> **Objective:** minimize cycles subject to the hard ≤12 load-lane / ≤12
> store-lane per-cycle limit; the recommendation is the **lane-saturation knee** —
> the smallest exposure whose coalesced traffic saturates the binding resource.
> Not an area or control/body tradeoff.

Kernel: `tests/app/vecsum/vecsum.cpp`

```cpp
LOOM_REDUCE(+)
uint32_t sum = init_value;
LOOM_PARALLEL(4)
for (uint32_t i = 0; i < N; i++) {
    sum += A[i];
}
```

`vecsum` is the **reduction** case: the loop carries `sum`, so plain parallel is
illegal — but the carry is **associative** (`+`), so `LOOM_REDUCE(+)` legalizes
parallel workers as independent partial sums merged by a `log`-depth tree. This
is the escape hatch for a loop-carried dependence (contrast
[`tridiag_solve`](../tridiag_solve/tridiag_solve_loom_dse.md), whose carry is
**not** associative and stays serial).

Regenerate: `python3 tests/scripts/loom_dse.py vecsum --config 6x6 --max-parallel 16`

## Model: lane-aware + vector coalescing (no banking)

This estimate has **no memory banking and no per-worker memory ports**. The only
per-cycle caps are the machine load/store **lanes** (`6x6`: `P_pe = 36`,
`L = S = 12`). Arithmetic (`P_pe`) and the critical path `CP` come from a
**global pool**: `LOOM_PARALLEL` and `LOOM_UNROLL` draw the same compute budget,
and the model does not separate them on the arithmetic axis. Control-overhead
amortization is not modeled, so the per-iteration iterator work (load / add /
store / compare) is charged per exposed iteration regardless of the split.

The one physical asymmetry the model keeps is **vector coalescing** on the
load/store axis: iterations unrolled inside one worker touch **adjacent**
elements, so a contiguous group of `V = 4` 64-bit elements coalesces into one
256-bit vector memory op (one lane-slot, free unpack/pack). `LOOM_PARALLEL`
workers **stride** across partitions and do not coalesce, so the model is biased
**toward** `LOOM_UNROLL` for contiguous groups. That credit is bounded by `V` and
**vanishes once `U >= V`, or when the contiguous dimension is a fully-consumed
reduction.**

The only lower bound is `absolute_cgra_lb`: the full-trip, fully-coalesced
aggregate over the full lanes `L`/`S`. The phrase "lower bound" is **not** applied
to any pragma candidate estimate — `pragma_agg` and `sched` assume waves do not
overlap and therefore sit at or above `absolute_cgra_lb`.

## Why vecsum is P/U-symmetric

`vecsum` is a reduction over the **contiguous** array `A`, and it is exactly the
case the spec calls out as showing **no `LOOM_PARALLEL`-vs-`LOOM_UNROLL`
distinction**:

- **The reduction is fully consumed in one wave.** The whole array collapses into
  one scalar, so the exposure equals the trip count (`exp = 256`) regardless of
  the `p`/`U` split — there is no partial-wave structure for a split to reshape.
- **`A` is contiguous, so it coalesces the same way for any split.** Whether
  iterations are exposed by parallel workers or by unroll, `A`'s 256 element
  loads coalesce to `~trip/V = 64` vector loads. The coalescing credit is a
  property of the contiguous run, not of which pragma exposed it, and a
  fully-consumed reduction reaches the maximal contiguous run either way — the
  `LOOM_UNROLL` bias has nothing left to earn.
- **The per-element iterator overhead is scalar and identical either way.** The
  iterator load / add / store / compare are not contiguous element accesses, so
  they never coalesce and are charged per exposed iteration under both pragmas.

Every remaining term is either compute / `CP` served by the global arithmetic
pool or scalar iterator work — neither distinguishes parallel from unroll. This
is the spec's **compute- / latency- / equal-exposure symmetry**, so the tool
collapses every candidate into a single equivalence group.

(Contrast `axpy` / `batchnorm`, whose contiguous output is written **per
iteration** rather than fully reduced: there, unrolled contiguous stores coalesce
while parallel strided stores do not, so those kernels **do** show a coalescing
distinction.)

## Setup

- `6x6` (`P_pe = 36`, `L = 12`, `S = 12`), `V = 4` 64-bit elements per vector op,
  `N = 256`, reduction fully consumed in one wave.
- Full-trip counts (fully coalesced): `A = 768`, `LD = 322`, `ST = 257`,
  `CP = 11` → `compute = 22`, `load = 27`, `store = 22`.
- `A`'s 256 element loads coalesce to `~trip/V = 64` vector loads; the remaining
  `LD` lane-slots are scalar iterator reads that never coalesce.
- **Binding class = `L`** (load): `absolute_cgra_lb = 27 = ceil(322 / 12)`, the
  fully-coalesced load-lane floor. `CP = 11` is dominated by the log-depth
  reduction merge tree.

## Results (`--max-parallel 16`)

```text
flags    split                      Ptot  aL  aS   exp   wav  cagg   p_agg   sched class           util P/L/S
-------------------------------------------------------------------------------------------------------------
K        i:P1U1  (+19 eq)              1  12  12   256     1    27      27      30 resource-bound   81/100/81

P-vs-U at fixed product 64 on level 'i' (other levels at P1U1):
  split           LD    ST   p_agg note
  P16U4            322   257      27 tie (fully coalesced or reduction-bound)
  P8U8            322   257      27 tie (fully coalesced or reduction-bound)
```

`(+19 eq)` = 20 candidates collapse into one equivalence group; they share the
same `LD`/`ST`/`p_agg`. The `P16U4` and `P8U8` rows make the symmetry explicit:
at fixed product `64`, parallel-heavy and unroll-heavy factorings post identical
`LD = 322`, `ST = 257`, `p_agg = 27` — the note reads *tie (fully coalesced or
reduction-bound)*.

Column / flag glossary:

- `split` — per-level `PaUb` factoring; `Ptot` — total parallel workers.
- `aL` / `aS` — active load / store lanes (`min(P_tot·w, L/S)`).
- `exp` — iterations exposed per wave; `wav` — waves (`ceil(trip/exp)`).
- `cagg` — `chunk_aggregate`; `p_agg` — `pragma_exposure_aggregate` (wave-summed);
  `sched` — finite-resource `schedule_estimate`. Neither `p_agg` nor `sched` is a
  lower bound.
- `class` — `latency-bound` (aggregate set by `CP`) vs `resource-bound`
  (aggregate set by a lane/compute term).
- `util P/L/S` — per-class utilization; the binding class reads `100%` exactly
  when the wave is resource-bound.
- Flags: `K` = recommended (saturation knee `E_sat`); `b` = bandwidth-starved
  (latency-bound, resources idle); `o` = oversubscribed (past the knee, no
  estimate gain).

## Recommendation

All candidates land in one equivalence group, so the choice is **indifferent**:
`LOOM_PARALLEL`, `LOOM_UNROLL`, and any factoring of their product produce the
same estimate. The tool reports the smallest representative, `P1U1`, as the knee:

- `exposure = 256`, `pragma_agg = 27` cycles (**1.00×** `absolute_cgra_lb`),
  **resource-bound** on the load lanes (`util L = 100%`, compute/store at `81%`).
- The fully-consumed reduction exposes all 256 iterations in one wave even at
  `P1U1`, and the coalesced load stream already fills all `12` load lanes, so the
  candidate sits **at** the saturation knee `E_sat`. There is no bandwidth-starved
  regime below it to climb out of and no oversubscribed regime above it to avoid.

**Contrast with the old banking model.** The previous banking-aware eval scaled
reduction workers to `P = 16` to fill `12` load *banks* (`active_L =
min(P_tot, L)`) and reported parallel as up to `2×` faster than unroll. That
gap was an artifact of per-worker banking. Under the lane-aware + vector-
coalescing model there are no banks: `A`'s loads coalesce to `~trip/V` vector
loads and are capped only by the `12` machine load lanes for **any** split, so
parallel no longer beats unroll. The current source `P = 4` is neither starved nor
sub-optimal — it is simply one point in the equivalence group.
