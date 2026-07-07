# batchnorm Loom-Pragma DSE (lane-aware + vector coalescing)

> **Objective:** minimize cycles subject to the hard ≤12 load-lane / ≤12
> store-lane per-cycle limit; the recommendation is the **lane-saturation knee** —
> the smallest exposure whose coalesced traffic saturates the binding resource.
> Not an area or control/body tradeoff.

Kernel: `tests/app/batchnorm/batchnorm.cpp` —
`output[c,h,w] = gamma[c]·(input[c,h,w] − mean[c])·inv_std[c] + beta[c]`

A three-level nest, **all parallel** (`c`, `h`, `w`) — no carried dependence.
`input`/`output` are laid out channel-major and are **contiguous over the
innermost dim `w`** (`idx = c·(H·W) + h·W + w`).

```cpp
LOOM_PARALLEL()
LOOM_UNROLL()
for (uint32_t c = 0; c < C; c++) {
    float inv_std = 1.0f / sqrtf(variance[c] + epsilon);
    for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = 0; w < W; w++) {
            uint32_t idx = c * (H * W) + h * W + w;
            float normalized = (input[idx] - mean[c]) * inv_std;
            output[idx] = gamma[c] * normalized + beta[c];
        }
    }
}
```

Regenerate: `python3 tests/scripts/loom_dse.py batchnorm --config 6x6 --max-parallel 8 --top 18`

## Model: lane-aware + vector coalescing

This estimate exposes **no per-worker memory partitions**: the only per-cycle
caps are the machine lanes. At `6x6`, `P_pe = 36` arithmetic lanes and
`L = S = 12` load/store lanes. Arithmetic (`P_pe`) and the critical path `CP` are
a **single global pool** — `LOOM_PARALLEL` and `LOOM_UNROLL` draw from the same
compute budget and do not separate it. Control-overhead amortization is **not**
modeled: iterator load/add/store/compare is charged per *exposed* iteration
regardless of how the exposure is split into workers vs. unroll.

The one physical asymmetry the model keeps is **vector coalescing on the
load/store axis**. Unrolled iterations inside a single worker touch **adjacent**
elements, so a contiguous run of `V = 4` (256-bit / four 64-bit) elements
coalesces into one vector memory op that occupies **one** lane-slot (unpack/pack
is free). `LOOM_PARALLEL` workers **stride** across partitions and do **not**
coalesce. The model is therefore biased **toward `LOOM_UNROLL`** for contiguous
groups; the advantage is bounded by `V` and vanishes once `U ≥ V`.

For batchnorm this asymmetry lives entirely on the innermost dim `w`:
`LOOM_UNROLL(w)` fuses a worker's adjacent `w`-accesses into `ceil(U_w/V)` vector
ops, while `LOOM_PARALLEL(w)` strides one element per worker. The `c` and `h`
dims are **strided** for `input`/`output` (their steps are `H·W` and `W`), so
they never coalesce and `LOOM_PARALLEL` vs `LOOM_UNROLL` is **symmetric** on
`c`/`h`. `mean`/`variance`/`gamma`/`beta` are per-channel invariants, loaded once
per exposed channel.

`absolute_cgra_lb` is the full-trip, fully-coalesced aggregate over the **full**
lanes `L`/`S` — the **only** lower bound. Every candidate `pragma_agg`/`sched`
sits at or above it; "lower bound" is never applied to a pragma estimate.

## Setup

- `6x6`; `C=4, H=W=8` (256 pixels). Full-trip counts (with `w`-coalescing
  applied): `A=1620 LD=376 ST=356 CP=8` → `compute=45 load=32 store=30`.
  `absolute_cgra_lb = 45`, set by **compute** (`ceil(1620/36)`).
- **Binding class = `P` (arithmetic).** With coalescing, `w`-contiguous input
  loads collapse from a scalar 48 lane-cycles to `load=32`, and stores to
  `store=30`, both now *below* `compute=45`. batchnorm carries ~4 arithmetic ops
  per pixel against ~1 coalesced input load, so it sits **near the compute/load
  boundary** — a heavier per-pixel load would flip the binder back to memory, and
  `w`-unrolling (coalescing) is exactly what keeps the load lanes from re-binding.

## Results (`--top 18`)

Verbatim tool output. Groups collapse pragma factorings with identical estimates
(`+N eq`). Flags: `K` = recommended knee, `b` = bandwidth-starved
(latency-bound), `o` = oversubscribed (past the knee), blank = feasible below the
top region.

```text
flags    split                      Ptot  aL  aS   exp   wav  cagg   p_agg   sched class           util P/L/S
-------------------------------------------------------------------------------------------------------------
o        c:P1U4 h:P1U8 w:P4U2  (+11 eq)    4  12  12   256     1    45      45      60 resource-bound   100/82/78
o        c:P1U4 h:P1U8 w:P1U8  (+23 eq)    1  12  12   256     1    45      45      54 resource-bound   100/71/67
o        c:P1U2 h:P1U8 w:P4U2  (+7 eq)    4  12  12   128     2    23      46      64 resource-bound   100/83/78
o        c:P1U4 h:P1U4 w:P4U2  (+8 eq)    4  12  12   128     2    23      46      64 resource-bound   100/87/78
o        c:P1U2 h:P1U8 w:P1U8  (+15 eq)    1  12  12   128     2    23      46      56 resource-bound   100/70/65
o        c:P1U4 h:P1U4 w:P1U8  (+17 eq)    1  12  12   128     2    23      46      56 resource-bound   100/74/65
         c:P1U1 h:P1U8 w:P4U2  (+3 eq)    4  12  12    64     4    12      48      68 resource-bound   100/83/75
         c:P1U2 h:P1U4 w:P4U2  (+5 eq)    4  12  12    64     4    12      48      68 resource-bound   100/83/75
         c:P1U4 h:P1U2 w:P4U2  (+5 eq)    4  12  12    64     4    12      48      72 resource-bound   100/92/75
         c:P1U2 h:P1U8 w:P2U2  (+7 eq)    2  12  12    64     4    12      48      72 resource-bound   100/92/83
K        c:P1U1 h:P1U8 w:P1U8  (+7 eq)    1  12  12    64     4    12      48      60 resource-bound   100/75/67
         c:P1U2 h:P1U4 w:P1U8  (+11 eq)    1  12  12    64     4    12      48      60 resource-bound   100/75/67
         c:P1U2 h:P1U8 w:P1U4  (+7 eq)    1  12  12    64     4    12      48      64 resource-bound   100/83/75
         c:P1U4 h:P1U2 w:P1U8  (+11 eq)    1  12  12    64     4    12      48      64 resource-bound   100/83/67
o        c:P1U2 h:P1U8 w:P8U1  (+7 eq)    8  12  12   128     2    24      48      80 resource-bound   96/100/96
o        c:P1U4 h:P1U8 w:P2U2  (+11 eq)    2  12  12   128     2    24      48      66 resource-bound   100/88/79
o        c:P1U4 h:P1U8 w:P1U4  (+11 eq)    1  12  12   128     2    24      48      60 resource-bound   100/75/71
o        c:P1U4 h:P1U8 w:P8U1  (+11 eq)    8  12  12   256     1    48      48      77 resource-bound   94/100/96
... (90 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: c:P1U1 h:P1U8 w:P1U8  -> exposure=64, pragma_agg=48 (1.07x the floor), resource-bound
```

## The `LOOM_PARALLEL`-vs-`LOOM_UNROLL` distinction

Because the binder is the **global** arithmetic pool, `P` and `U` are identical on
the compute/`CP` axis. They separate **only** where `LOOM_UNROLL` coalesces a
contiguous group that `LOOM_PARALLEL` would stride — and for batchnorm that is
only the innermost dim `w`.

**`c`/`h` are symmetric (strided, no coalescing).** The tool's fixed-product
sweep on the channel level confirms it — `P4U1`, `P2U2`, and `P1U4` all land on
the same load/store terms and the same estimate:

```text
P-vs-U at fixed product 4 on level 'c' (other levels at P1U1):
  split           LD    ST   p_agg note
  P4U1             36    16     512 tie (fully coalesced or reduction-bound)
  P2U2             36    16     512 tie (fully coalesced or reduction-bound)
  P1U4             36    16     512 tie (fully coalesced or reduction-bound)
```

**`w` favors unroll (coalescing).** Compare two rows in the table above that
share the same `c`/`h` split and exposure (128) and differ **only** on `w`:

| `w` split | coalesces? | `cagg` | `p_agg` | util P/L/S |
|-----------|:----------:|-------:|--------:|-----------|
| `w:P1U8` (unroll) | yes | 23 | **46** | 100/**70**/65 |
| `w:P8U1` (parallel) | no (strides) | 24 | 48 | 96/**100**/96 |

`LOOM_UNROLL(w)` fuses each worker's 8 adjacent `w`-loads into 2 vector ops, so
its load lanes sit at 70% while the strided `LOOM_PARALLEL(w)` version saturates
them at 100% and needs a slightly larger aggregate (24 vs 23). Since the kernel
is compute-bound (`util_P = 100%` in both), the coalescing shows up as **load-lane
headroom**, not a smaller floor — but that headroom is exactly what keeps `w`
from becoming the binding class. This is the intended, mentor-confirmed reversal
from the earlier `P`-favoring model: for a contiguous innermost dim, **unroll
wins on the load/store term** (while `U_w < V = 4`).

## Recommendation

**Recommended knee: `c:P1U1 h:P1U8 w:P1U8` — exposure 64, `pragma_agg = 48`
(`1.07×` the floor of 45), resource-bound.**

The exposure is `E_sat`, the smallest exposure at which the binding class
(compute) becomes resource-bound. Below it (exposure 32) the wave is
**latency-bound**: `cagg = CP = 8` and the arithmetic lanes idle part of every
wave (`b` rows). At exposure 64 the compute term overtakes `CP` (`cagg = 12`),
so the wave is resource-bound and each added wave runs at the compute-limited
rate. This exposure fully unrolls one channel's `H·W = 64` pixels per wave and
walks the 4 channels as 4 waves, reloading the per-channel invariants once per
wave.

The recommended split is **pure unroll** (`P_tot = 1`): unrolling `h` and `w`
inside one worker both saturates the compute pool and coalesces the contiguous
`w`-run, so it needs no striding parallel workers and carries the least transient
backlog. It ties at `pragma_agg = 48` with the sibling exposure-64 factorings
(the blank rows); the tool reports the all-unroll representative as the cleanest.

Rows past the knee (`o`) only shave the wave-serialization penalty — `pragma_agg`
creeps from 48 down toward 45 (the floor is reached at exposure 256, a single
wave) — while unrolling more iterations for **no steady-state throughput gain**
and linearly growing area/backlog. The `1.07×` gap is a wave-serialization
artifact of the non-overlap assumption, not a bandwidth deficit; pipelined
dataflow already sustains the compute-bound rate at the knee. Take exposure 64;
do not chase the last 7% by unrolling all 256 iterations.
