# conv2d Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps only the conv2d-specific
setup, helper output, and recommendation.

Kernel: `tests/app/conv2d/conv2d.cpp` —
`output[co,oh,ow] = Σ_{ci,kh,kw} input[...]·kernel[...]`

Current source pragma:

```cpp
LOOM_PARALLEL(4, contiguous)
for (uint32_t co = 0; co < C_out; co++) {
    LOOM_UNROLL(4)
    for (uint32_t oh = 0; oh < OH; oh++) {
        LOOM_TRIPCOUNT_FULL(16, 16, 1, 64)
        for (uint32_t ow = 0; ow < OW; ow++) {
            float sum = 0.0f;
            for (uint32_t ci = 0; ci < C_in; ci++) {
                for (uint32_t kh = 0; kh < KH; kh++) {
                    for (uint32_t kw = 0; kw < KW; kw++) {
                        ...
                        sum += input_val * kernel_val;
                    }
                }
            }
            output[co * (OH * OW) + oh * OW + ow] = sum;
        }
    }
}
```

The helper models this as a two-level nest: a dependency-parallel flattened
output-pixel level `out = C_out·OH·OW`, and a fully consumed tap reduction
`tap = C_in·KH·KW`.

Model: the lane-aware + vector-coalescing Loom-pragma DSE in
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py conv2d --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 20
```

## Why P and U differ

`out` is dependency-parallel: each output pixel writes a distinct element.
`tap` is a reduction, so it is tree-reduced and fully consumed inside each output
lane; tap-level P/U choices are performance-inert in this DSE.

The source still annotates `input` with `LOOM_MEMORY_BANK(4, block)`, but the
current DSE deliberately ignores explicit banking and bank conflicts. The only
memory caps are the machine lanes (`L = S = 12` for `6x6`).

The binding traffic is the strided halo `input` window over the taps. Those
loads do not coalesce, so conv2d remains load-bound. `kernel` is contiguous over
the tap reduction, but that coalescing is split-inert because the reduction is
fully consumed. `output` stores are contiguous over `out`, so unroll can coalesce
stores; unroll also amortizes the `out` iterator because fewer workers carry
fewer iterator load/add/store/compare sets. The unroll advantage is real but
modest because the binding load term is still dominated by non-coalesced input
loads.

## Setup

- Resource config: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`); vector width
  `V = 4` (one 256-bit vector op per four 64-bit elements)
- Dimensions: `C_in = 3`, `C_out = 4`, `H = W = 8`, `KH = KW = 3`, `stride = 1`,
  so `out = 144` and `tap = 27`.
- Full-trip counts (full unroll, fully coalesced where legal):
  `A = 7634`, `LD_rec = 4897`, `LD_eff = 4898`, `ST = 37`, `CP = 8`, giving
  `compute = ceil(7634/36) = 213`, `load = ceil(4897/12) = 409`,
  `store = ceil(37/12) = 4`.
- `absolute_cgra_lb = 409` — the full-trip, **fully-coalesced,
  invariant-amortized** aggregate (`max(8, 213, 409, 4) = 409`), and the
  **only** lower bound. The binding class is **load (`L`)**.

## Results

`out` is the 144-lane dependency-parallel output level and `tap` is the fully
consumed 27-tap reduction. The full-trip floor is `absolute_cgra_lb = 409`
(`A = 7634`, `LD_rec = 4897`, `LD_eff = 4898`, `ST = 37`, `CP = 8`), and
**only** that value is a lower bound; `p_agg` and `sched` are wave-serialized
estimates. `aL` is based on recurring loads only, while `LD_eff` also includes
one-time invariant loads. The binding class is load (`L`) because non-coalesced
input-halo loads dominate; `LOOM_UNROLL(out)` still helps by coalescing output
stores and amortizing the `out` iterator.

```text
# Loom pragma DSE (lane-aware + vector coalescing): conv2d  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `out[144,parallel], tap[27,reduction]`; output pixels (out = C_out*OH*OW) are parallel; the K = C_in*KH*KW taps are a fully-consumed reduction (tree-reduced -> no tap iterator). input is strided over taps (halo) so it does NOT coalesce and dominates loads; weight is contiguous but reduction-inert; output is contiguous over out. LOOM_UNROLL(out) beats LOOM_PARALLEL(out) two ways: it coalesces the output stores and amortizes the out iterator (charged once per worker). Load-bound on the strided input, so the edge is modest. Halo reuse / weight sharing not modeled. Full-trip counts are `A=7634`, `LD_rec=4897`, `LD_eff=4898`, `ST=37`, and `CP=8`, giving the only lower bound, `absolute_cgra_lb=409=max(CP 8, compute 213, load 409, store 4)`, with load pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
o        out:P4U2 tap:P1U1             4  12   8    277   216    18    23     414     504 resource-bound    52/100/4
o        out:P2U4 tap:P1U1             2  12   4    275   216    18    23     414     504 resource-bound    52/100/4
o        out:P1U8 tap:P1U1             1  12   3    274   216    18    23     414     504 resource-bound    52/100/4
o        out:P8U2 tap:P1U1             8  12  12    553   432     9    46     414     459 resource-bound    52/100/4
o        out:P4U4 tap:P1U1             4  12   8    549   432     9    46     414     459 resource-bound    52/100/2
o        out:P2U8 tap:P1U1             2  12   6    547   432     9    46     414     459 resource-bound    52/100/2
o        out:P1U16 tap:P1U1            1  12   5    546   432     9    46     414     459 resource-bound    52/100/2
o        out:P16U1 tap:P1U1           16  12  12    561   432     9    47     423     459 resource-bound    53/100/6
         out:P4U1 tap:P1U1             4  12   8    141   108    36    12     432     612 resource-bound    58/100/8
         out:P2U2 tap:P1U1             2  12   4    139   108    36    12     432     612 resource-bound    50/100/8
K        out:P1U4 tap:P1U1             1  12   2    138   108    36    12     432     612 resource-bound    50/100/8
o        out:P8U1 tap:P1U1             8  12  12    281   216    18    24     432     504 resource-bound    54/100/8
o        out:P4U8 tap:P1U1             4  12  12   1093   864     5    91     455     480 resource-bound    53/100/1
o        out:P2U16 tap:P1U1            2  12  10   1091   864     5    91     455     480 resource-bound    53/100/1
o        out:P1U32 tap:P1U1            1  12   9   1090   864     5    91     455     480 resource-bound    53/100/1
o        out:P16U2 tap:P1U1           16  12  12   1105   864     5    92     460     480 resource-bound    52/100/3
o        out:P8U4 tap:P1U1             8  12  12   1097   864     5    92     460     480 resource-bound    52/100/2
o        out:P32U1 tap:P1U1           32  12  12   1121   864     5    94     470     490 resource-bound    52/100/6
o        out:P8U8 tap:P1U1             8  12  12   2185  1728     3   182     546     561 resource-bound    52/100/1
o        out:P4U16 tap:P1U1            4  12  12   2181  1728     3   182     546     561 resource-bound    52/100/1
... (16 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: out:P1U4 tap:P1U1  -> exposure=108, pragma_agg=432 (1.06x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 4 on level 'out' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P4U1             140    141     8     432 tie (control/coalescing sit below the binding term)
  P2U2             138    139     4     432 tie (control/coalescing sit below the binding term)
  P1U4             137    138     2     432 tie (control/coalescing sit below the binding term)

4x4 recommendation: out:P1U2 tap:P1U1.
8x8 recommendation: out:P1U4 tap:P1U1.
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Reading the fixed-product block

The fixed-product block isolates the `out` split at exposure product 8 while the
tap reduction stays fully consumed. `P8U1` pays eight worker iterators and cannot
coalesce the contiguous output stores, so it lands at `p_agg = 432`. The
unroll-heavy rows lower both `LD_rec` and `ST`; the best rows tie at
`p_agg = 414` because the remaining binding term is the non-coalesced input
loads. This is why conv2d now favors unroll at fixed product, but only modestly.

## Recommendation

**`out:P1U4 tap:P1U1` is the recommended knee (`K`)**: exposure `108`,
`p_agg = 432` (`1.06×` the `409`-cycle floor), resource-bound. This is the
smallest exposure where the best-coalesced candidate becomes resource-bound; it
exposes four output pixels per wave, with each output lane consuming the full
27-tap reduction.

Rows above the knee (`o`) mostly reduce wave-serialization rounding toward the
floor while increasing transient backlog and mapping pressure. The stale
banking-style explanation does not apply here: this DSE ignores the source
`LOOM_MEMORY_BANK(4, block)` cap, and the recommendation is not "fill four input
banks." The useful split-side effect is output-level unroll plus iterator
amortization, while the sustained floor remains load-bound on the strided input
halo.

## Comparing against measured DFG simulator cycles

Use the shared comparison rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For conv2d, the DSE floor is below the scalar Metric-1 aggregate in the main eval
because this model credits vector coalescing and control amortization and ignores
explicit banking. It is a Loom-pragma estimate, not a replacement for the main
ASAP/CGRA eval.
