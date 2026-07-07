# conv2d Loom-Pragma DSE (lane-aware + vector coalescing)

> **Objective:** minimize cycles subject to the hard ≤12 load-lane / ≤12
> store-lane per-cycle limit; the recommendation is the **lane-saturation knee** —
> the smallest exposure whose coalesced traffic saturates the binding resource.
> Not an area or control/body tradeoff.

Kernel: `tests/app/conv2d/conv2d.cpp` —
`output[co,oh,ow] = Σ_{ci,kh,kw} input[...]·weight[...]`

Modeled as a two-level nest: an outer **parallel** loop over output pixels
(`out = C_out·OH·OW`) and an inner **reduction** over the `K = C_in·KH·KW` taps.

Regenerate: `python3 tests/scripts/loom_dse.py conv2d --config 6x6 --max-parallel 8 --top 14`

## Model / assumptions (lane-aware, no banking)

This is the lane-aware + vector-coalescing DSE. There is **no banking model**:
no bank counts and no bank caps. The provisional DSE deliberately
**ignores `LOOM_MEMORY_BANK`** (`conv2d.cpp:61` still carries
`LOOM_MEMORY_BANK(4, block)` on `input`, but this estimate does not read it).
The only caps on exposure are the machine lanes: for `6x6`, `P_pe = 36`
arithmetic lanes and `L = S = 12` load/store lanes. Arithmetic (`P_pe`) and the
critical path `CP` are a **global pool** — they do not distinguish exposure that
came from `LOOM_PARALLEL` from exposure that came from `LOOM_UNROLL`, so `P` and
`U` tie on the compute and latency terms. Control-overhead amortization is **not
modeled**: induction is charged per exposed iteration regardless of the split.

The **one** asymmetry between `LOOM_PARALLEL` and `LOOM_UNROLL` is **vector
coalescing**. Unrolled iterations in a single worker touch **adjacent** elements,
so a contiguous run of `V = 4` (64-bit) same-array accesses coalesces into one
256-bit vector memory op — one lane-slot, with free unpack/pack. Parallel workers
instead **stride** across data partitions and do not coalesce. The model is
therefore biased **toward `LOOM_UNROLL`** for contiguous groups, but the effect is
bounded by `V` and vanishes once `U ≥ V` or when the contiguous dimension is a
fully-consumed reduction.

`absolute_cgra_lb` is the full-trip, fully-coalesced aggregate over the **full**
lanes `L`/`S`. It is the **only** lower bound; the wave-summed `pragma_agg` and
`sched_est` both assume waves do **not** overlap and therefore sit at or above it.
The phrase "lower bound" is never applied to any pragma candidate estimate.

### conv2d specifics

- **`input`** is accessed with a strided **halo** pattern over the taps, so it
  does **not** coalesce and **dominates the load term**.
- **`weight`** is contiguous over the taps, so it coalesces — but the tap loop is
  a fully-consumed reduction, so weight coalescing is **split-inert** (it is the
  same regardless of how `out` is split into `P`/`U`).
- **`output`** is contiguous over `out`, so `LOOM_UNROLL(out)` coalesces the
  output stores while `LOOM_PARALLEL(out)` strides. This is a genuine store-side
  edge for unroll — but the kernel is **load-bound on `input`** (stores are far
  from binding), so it is **second order**. Net: conv2d is largely
  **P/U-symmetric** on `out`.
- Halo (input-window) reuse across neighboring output pixels and weight sharing
  are **not** modeled — loads are counted per tap (conservative).

## Setup

- `6x6`; `C_in=3, C_out=4, H=W=8, KH=KW=3, stride=1` → `out = 144`, `K = 27`.
- Full-trip counts: `A=15696 LD=8929 ST=4068 CP=8` → `compute=436 load=745
  store=339`. `absolute_cgra_lb = 745` (`ceil(8929/12)`, the load term over the
  full `L = 12` lanes). The binding class is **L**: conv2d is load-bound on the
  strided `input`, whose halo accesses do not coalesce.
- The full-trip `LD = 8929` is the **fully-coalesced** load lane-slot count — the
  contiguous weight run coalesces over the taps, but the strided input stays
  scalar, so input sets the term.

## Results (verbatim `--top 14`)

```text
flags    split                      Ptot  aL  aS   exp   wav  cagg   p_agg   sched class           util P/L/S
-------------------------------------------------------------------------------------------------------------
o        out:P8U2 tap:P1U1  (+12 eq)    8  12  12   432     9    83     747     774 resource-bound   59/100/46
o        out:P2U8 tap:P1U1  (+25 eq)    2  12  12   432     9    83     747     765 resource-bound   59/100/46
o        out:P4U1 tap:P1U1  (+12 eq)    4  12  12   108    36    21     756     828 resource-bound   62/100/48
o        out:P2U2 tap:P1U1  (+12 eq)    2  12  12   108    36    21     756     828 resource-bound   62/100/48
o        out:P1U4 tap:P1U1  (+12 eq)    1  12  12   108    36    21     756     828 resource-bound   62/100/48
o        out:P8U1 tap:P1U1  (+12 eq)    8  12  12   216    18    42     756     792 resource-bound   60/100/48
o        out:P4U2 tap:P1U1  (+12 eq)    4  12  12   216    18    42     756     792 resource-bound   60/100/45
o        out:P1U8 tap:P1U1  (+25 eq)    1  12  12   216    18    42     756     792 resource-bound   60/100/45
         out:P2U1 tap:P1U1  (+12 eq)    2  12  12    54    72    11     792     936 resource-bound   64/100/45
K        out:P1U2 tap:P1U1  (+12 eq)    1  12  12    54    72    11     792     936 resource-bound   64/100/45
o        out:P4U8 tap:P1U1  (+25 eq)    4  12  12   864     5   166     830     840 resource-bound   58/100/46
o        out:P8U8 tap:P1U1  (+12 eq)    8  12  12  1728     3   331     993    1002 resource-bound   59/100/46
b        out:P1U1 tap:P1U1  (+12 eq)    1  12  12    27   144     8    1152    1440 latency-bound     50/75/38
```

Flags: `K` = recommended (saturation knee `E_sat`); `b` = bandwidth-starved
(latency-bound, resources idle); `o` = oversubscribed (past the knee, no estimate
gain). Each row carries `+12`/`+25` equivalent inner-tap variations (inert — the
taps are a fully-consumed reduction, so tap `P`/`U` only reshapes the product
merge tree). All rows run over the full `L = S = 12` lanes (`aL = aS = 12`) — no
cap holds them back, in contrast to the old banking eval.

## The P-vs-U distinction

Because conv2d is load-bound on the strided `input`, `P` and `U` on the `out`
level are **symmetric** across the whole ranking (equal `p_agg` at equal
exposure): `out:P2U1` and `out:P1U2` both land at `p_agg = 792` (exposure 54), and
`out:P4U1`/`out:P2U2`/`out:P1U4` all land at `756` (exposure 108). The only place
unroll could edge parallel is the **contiguous output store** — but stores are not
the binding class (`store = 339 ` vs `load = 745`), so that coalescing edge is
second order.

Fixed product `P·U = 32` on the `out` level (tool's own comparison, other levels
`P1U1`):

```text
  split           LD    ST   p_agg note
  P8U4           1985   904     830 tie (fully coalesced or reduction-bound)
  P4U8           1985   904     830 tie (fully coalesced or reduction-bound)
```

The unroll-heavy `P4U8` and the parallel-heavy `P8U4` produce **identical** `LD`,
`ST`, and `p_agg` — input never coalesces (halo stride) and weight coalescing is
split-inert (reduction), so the split cannot move the binding load term.

## Recommendation

```text
RECOMMENDED: out:P1U2 tap:P1U1  ->  exposure=54, pragma_agg=792 (1.06x the floor), resource-bound
```

The recommended knee is `out:P1U2 tap:P1U1` — exposure `54` (2 output pixels ×
their full 27-tap reduction), `pragma_agg = 792`, i.e. `1.06×` the
`absolute_cgra_lb` of `745`. This is the smallest exposure at which the binding
load class becomes resource-bound (`E_sat`); at `6x6`, `out:P2U1` is the exact
tie (same `792`), and the tool breaks the tie toward `U` for the (second-order)
output-store coalescing. Larger exposures (`P4U1`, `P8U2`, …, flagged `o`) are
**oversubscribed**: they only creep `p_agg` toward the floor through per-wave
rounding while stacking transient backlog and area. The single-worker
`out:P1U1` (flagged `b`) is **bandwidth-starved** — latency-bound, with the load
lanes idle (`util L = 75%`).

### Contrast with the prior banking eval

The old eval modeled `input`'s `LOOM_MEMORY_BANK(4, block)` as a hard cap
`active_L = min(P_tot, 4, L) ≤ 4`, so conv2d behaved like gemv: parallelism
scaled input bandwidth only to 4 banks, the recommendation was
`LOOM_PARALLEL(4)` to "fill the 4 banks," and the advice to go faster was to
**raise `input`'s banking to 12**. Under the lane-aware model there is no such
cap — every candidate runs over the full `L = 12` lanes. Consequences:

- The lower bound **drops** from the old scalar `985` (`ceil(11809/12)`) to `745`
  (`ceil(8929/12)`), because the DSE now credits **weight coalescing** in the
  full-trip aggregate.
- The recommendation is no longer a bank-fill but the **`E_sat` knee**
  (`out:P1U2`, exposure `54`), and `P` vs `U` on `out` is **symmetric** (not
  "parallel beats unroll up to 4 banks"). The `input` halo simply doesn't
  coalesce, so no split reaches below the load-bound floor — raising banking is
  no longer the lever, because banking is not modeled at all.
