# gemv Loom-Pragma DSE (lane-aware + vector coalescing)

> **Objective:** minimize cycles subject to the hard ≤12 load-lane / ≤12
> store-lane per-cycle limit; the recommendation is the **lane-saturation knee** —
> the smallest exposure whose coalesced traffic saturates the binding resource.
> Not an area or control/body tradeoff.

Kernel: `tests/app/gemv/gemv.cpp` — `output_y[i] = alpha·Σ_j A[i,j]·x[j] + beta·input_y[i]`

```cpp
LOOM_MEMORY_BANK(4, block) LOOM_STREAM const uint32_t* A, ...
for (uint32_t i = 0; i < M; i++) {        // rows: parallel
    uint32_t sum = 0;
    for (uint32_t j = 0; j < N; j++)      // cols: reduction
        sum += A[i * N + j] * x[j];
    output_y[i] = alpha * sum + beta * input_y[i];
}
```

`gemv` is the flagship **nested** case: an outer **parallel** row loop (`i`) and an
inner **reduction** column loop (`j`). The `LOOM_MEMORY_BANK(4, block)` annotation on
`A` is left in the source, but the provisional lane-aware DSE **ignores** it — there
are no banks, no per-worker ports, and no bank caps in this model. The recommendation
below is built entirely on machine lanes and vector coalescing, not on that pragma.

Regenerate: `python3 tests/scripts/loom_dse.py gemv --config 6x6 --max-parallel 8 --top 14`

## Coalescing (the key structure)

Under the lane-aware + vector-coalescing model the only per-cycle caps are the machine
load/store lanes (`L = S = 12` at `6x6`) and the **global** arithmetic pool
(`P_pe = 36`, shared by `LOOM_PARALLEL` and `LOOM_UNROLL` alike). The one axis on which
the two pragmas differ is **vector coalescing**: `V = 4` adjacent 64-bit elements
touched by consecutive unrolled iterations of one worker collapse into a single vector
lane-slot (free unpack/pack), whereas parallel workers stride across partitions and
cannot coalesce. This is a bias *toward* `LOOM_UNROLL` for contiguous groups, bounded
by `V` and gone once `U ≥ V`.

Two facts make `gemv` almost fully symmetric in the two pragmas:

- **The dot-product loads (`A[i][j]`, `x[j]`) are contiguous over `j`, and `j` is a
  fully-consumed reduction.** Whatever the split on `i`, each row's 64 column elements
  are consumed in full, so they coalesce identically — the `A`/`x` loads are
  **P/U-symmetric**. The reduction dim `j` is **inert** to the p/u choice: parallel or
  unroll on `j` only reshapes the product/merge tree and changes no lane demand.
- **`y[i]` / `output_y[i]` are contiguous over `i`.** So `LOOM_UNROLL(i)` coalesces the
  output stores while `LOOM_PARALLEL(i)` strides across rows and does not. This is the
  only genuine asymmetry — but `gemv` is load-limited (dominated by `A` loads and the
  per-column `j`-induction), so the store-side edge is **second order** and does not
  move the binding term.

Net: `gemv` shows little `LOOM_PARALLEL`-vs-`LOOM_UNROLL` distinction — it is largely
symmetric, and the fixed-product contrast comes out a tie.

## Setup

- `6x6`: `P_pe = 36`, `L = S = 12`, `V = 4` (64-bit elements per vector op). `M = N = 64`.
- Full-trip op counts: `A = 16640`, `LD = 5220`, `ST = 4176`, `CP = 11` →
  `compute = 463`, `load = 435`, `store = 348`.
- `absolute_cgra_lb = 463` — the full-trip, fully-coalesced aggregate over the full
  lanes, and the **only** lower bound. It is pinned by the arithmetic term
  (`ceil(16640/36) = 463`); the coalesced `A`-load term (`435`) sits nearly balanced
  with it. At finite exposure the load lane fills first (`util_L` reaches 100% at the
  knee while compute trails at 94%), which is why the tool marks `L` as the binding
  class for exposure selection. No pragma estimate is a lower bound — every
  `pragma_agg`/`sched` sits at or above `463`.

## Results (top of the sweep)

```text
flags    split                      Ptot  aL  aS   exp   wav  cagg   p_agg   sched class           util P/L/S
-------------------------------------------------------------------------------------------------------------
o        i:P8U8 j:P1U1  (+15 eq)       8  12  12  4096     1   463     463     580 resource-bound   100/94/75
o        i:P4U1 j:P1U1  (+15 eq)       4  12  12   256    16    29     464     624 resource-bound  100/100/76
o        i:P2U2 j:P1U1  (+15 eq)       2  12  12   256    16    29     464     624 resource-bound  100/100/76
o        i:P1U4 j:P1U1  (+15 eq)       1  12  12   256    16    29     464     624 resource-bound  100/100/76
o        i:P8U1 j:P1U1  (+15 eq)       8  12  12   512     8    58     464     600 resource-bound   100/98/76
o        i:P4U2 j:P1U1  (+15 eq)       4  12  12   512     8    58     464     600 resource-bound   100/97/76
o        i:P1U8 j:P1U1  (+31 eq)       1  12  12   512     8    58     464     600 resource-bound   100/97/76
o        i:P8U2 j:P1U1  (+15 eq)       8  12  12  1024     4   116     464     592 resource-bound   100/96/76
o        i:P2U8 j:P1U1  (+31 eq)       2  12  12  1024     4   116     464     588 resource-bound   100/95/75
o        i:P4U8 j:P1U1  (+31 eq)       4  12  12  2048     2   232     464     584 resource-bound   100/94/75
         i:P2U1 j:P1U1  (+15 eq)       2  12  12   128    32    16     512     640 resource-bound   94/100/69
K        i:P1U2 j:P1U1  (+15 eq)       1  12  12   128    32    16     512     640 resource-bound   94/100/69
b        i:P1U1 j:P1U1  (+15 eq)       1  12  12    64    64    11     704     832 latency-bound     73/82/55
```

Every listed row carries `+15` (or `+31`) `equivalent` candidates — those are the
inner-`j` pragma variations, which are **inert** and collapse to the same numbers.
This is the "no distinction on that level" case: `j` is a fully-consumed reduction, so
parallelizing or unrolling it only reshapes the merge tree and changes no lane demand.

## The P-vs-U distinction (row level)

At a fixed product on the row level `i` (inner `j` held at `P1U1`), the two pragmas are
a tie:

```text
P-vs-U at fixed product 32 on level 'i' (other levels at P1U1):
  split           LD    ST   p_agg note
  P8U4           2620  2088     464 tie (fully coalesced or reduction-bound)
  P4U8           2620  2088     464 tie (fully coalesced or reduction-bound)
```

Parallel-heavy (`P8U4`) and unroll-heavy (`P4U8`) splits produce **identical** `LD`,
`ST`, and `p_agg`. The dominant `A`/`x` loads coalesce the same way regardless of the
split (contiguous, fully-consumed reduction over `j`), and because the kernel is
load-limited the store-side unroll advantage never surfaces in the binding term. There
is no throughput reason to prefer one over the other on the load path.

## Recommendation

**`LOOM_UNROLL(2)` on rows (`i`), inner `j` left at `1`.** That is the recommended
candidate `i:P1U2 j:P1U1`: exposure `128` over `32` waves, `pragma_agg = 512`
(`1.11×` the `463` floor), resource-bound. It is the saturation knee `E_sat` — the
smallest exposure at which the load lane becomes resource-bound (`cagg = 16 > CP = 11`).
Just below it, `i:P1U1` (exposure `64`) is **latency-bound**: `cagg = CP = 11` and the
resource classes idle (flagged `b`, bandwidth-starved). Above it, every larger candidate
is flagged `o` (oversubscribed): extra workers/unroll only shrink the wave-serialization
gap (`p_agg` creeps from `512` toward `463`), not the steady-state floor, while transient
backlog, area, and mapping pressure grow.

> `i:P2U1` ties `i:P1U2` **exactly** (same exposure `128`, `p_agg = 512`, identical
> `util 94/100/69`). The tool breaks the tie toward `LOOM_UNROLL` because `output_y[i]`
> is contiguous over `i`, so unrolled stores coalesce while parallel stores stride — a
> second-order preference that does not change the binding term here. On the dominant
> load path the two are interchangeable.

## Contrast with the old banking model

The previous eval recommended `LOOM_PARALLEL(4)` on rows and claimed "parallel beats
unroll up to 4×." That conclusion came entirely from a banking assumption: `A` was
modeled under a `LOOM_MEMORY_BANK(4, block)` cap that let only row-parallelism add `A`
load bandwidth (capped at 4 rows) while unroll appeared to serialize. The
current lane-aware model has **no banks, no ports, and no bank caps** — it ignores
`LOOM_MEMORY_BANK`. The `A`/`x` loads coalesce identically for any split (contiguous,
fully-consumed reduction over `j`), so parallel and unroll are symmetric on the dominant
load path. The old row-parallel advantage was an artifact of the bank cap and is gone;
under this model `gemv` is a tie, tie-broken only by the second-order store-coalescing
preference toward unroll.
