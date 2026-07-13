# batchnorm Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file keeps only the batchnorm-specific
setup, helper output, and recommendation.

Kernel: `tests/app/batchnorm/batchnorm.cpp` —
`output[c,h,w] = gamma[c]·(input[c,h,w] − mean[c])·inv_std[c] + beta[c]`

Current source pragma:

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

Model: the lane-aware + vector-coalescing Loom-pragma DSE in
[`DSE_rules.md`](../DSE_rules.md) and the "Optional Loom-Pragma Design-Space
Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

Regenerate:

```bash
python3 tests/scripts/loom_dse.py batchnorm --config 6x6 --top 24
```

## Why P and U differ

All three source loops are dependency-parallel: each `(c,h,w)` lane writes a
distinct `output[idx]`, and the per-channel values are read-only for that lane.
The memory layout is channel-major, so only the innermost `w` dimension is
contiguous for `input` and `output`.

`LOOM_UNROLL(w)` exposes adjacent `w` elements inside one worker, allowing vector
loads/stores. `LOOM_PARALLEL(w)` strides across workers and does not coalesce
with its neighbors. The `c` and `h` dimensions are strided for input/output, so
they do not get a coalescing credit; however, unroll on any level still amortizes
loop control because one iterator is charged per worker per wave, not per
element. `mean[c]`, `variance[c]`, `gamma[c]`, and `beta[c]` are loaded once per
exposed channel and reused across its spatial pixels.

The full-trip floor is compute-bound, so these split choices mostly show up as
load/store headroom and schedule estimate changes, not a lower
`absolute_cgra_lb`.

## Setup

- Resource config: `6x6` (`P_pe = 36`, `L = 12`, `S = 12`); vector width
  `V = 4` (one 256-bit vector op per four 64-bit elements)
- Trip counts: `C = 4`, `H = 8`, `W = 8` (`N = 256` pixels)
- Full-trip counts (full unroll, fully coalesced over `w`):
  `A = 1038`, `LD_rec = 81`, `LD_eff = 85`, `ST = 65`, `CP = 8`, giving
  `compute = ceil(1038/36) = 29`, `load = ceil(81/12) = 7`,
  `store = ceil(65/12) = 6`.
- `absolute_cgra_lb = 29` — the full-trip, **fully-coalesced,
  invariant-amortized** aggregate (`max(8, 29, 7, 6) = 29`), and the **only**
  lower bound. The binding class is **compute (`P`)**.

## Results

`c`, `h`, and `w` are all dependency-parallel, with only the innermost `w`
contiguous for input/output coalescing. The full-trip floor is
`absolute_cgra_lb = 29` (`A = 1038`, `LD_rec = 81`, `LD_eff = 85`, `ST = 65`,
`CP = 8`), and **only** that value is a lower bound; `p_agg` and `sched` are
wave-serialized estimates. The binding class is compute (`P`), so unroll mostly
shows up as load/store headroom: `LOOM_UNROLL(w)` coalesces contiguous memory,
and unroll on any level amortizes iterator control.

```text
# Loom pragma DSE (lane-aware + vector coalescing): batchnorm  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `c[4,parallel], h[8,parallel], w[8,parallel]`; input/output are contiguous over the innermost w. LOOM_UNROLL(w) coalesces a worker's adjacent w-accesses (ceil(U_w/V) vector ops) while LOOM_PARALLEL(w) strides -> unroll-on-w beats parallel-on-w on the load/store term (while U_w < V). c/h are strided for input and do not coalesce, but LOOM_UNROLL on ANY level still amortizes the iterator (charged once per worker over the c*h*w worker set), so unroll cuts control ops even where coalescing cannot. Compute-bound, so those load/store savings show as lane headroom, not a lower floor. mean/variance/gamma/beta are per-channel invariants (once per exposed channel). Full-trip counts are `A=1038`, `LD_rec=81`, `LD_eff=85`, `ST=65`, and `CP=8`, giving the only lower bound, `absolute_cgra_lb=29=max(CP 8, compute 29, load 7, store 6)`, with compute pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
o        c:P1U4 h:P1U8 w:P4U2          4  12  12    152   256     1    29      29      36 resource-bound   100/45/38
o        c:P1U4 h:P2U4 w:P2U4  (+4 eq)    4  12  12     88   256     1    29      29      32 resource-bound   100/24/21
o        c:P1U4 h:P1U8 w:P2U4  (+2 eq)    2  12  12     86   256     1    29      29      32 resource-bound   100/24/21
o        c:P1U4 h:P1U8 w:P1U8          1  12  12     85   256     1    29      29      32 resource-bound   100/24/21
o        c:P1U2 h:P1U8 w:P8U1          8  12  12    148   128     2    15      30      58 resource-bound   100/80/80
o        c:P1U2 h:P2U4 w:P4U2  (+1 eq)    8  12  12     84   128     2    15      30      40 resource-bound   100/47/40
o        c:P1U2 h:P4U2 w:P2U4  (+3 eq)    8  12  12     52   128     2    15      30      34 resource-bound   100/27/27
o        c:P1U4 h:P1U4 w:P8U1  (+2 eq)    8  12  12    156   128     2    15      30      58 resource-bound   100/87/80
o        c:P1U4 h:P2U2 w:P4U2  (+4 eq)    8  12  12     92   128     2    15      30      40 resource-bound   100/53/40
o        c:P1U4 h:P4U1 w:P2U4  (+7 eq)    8  12  12     60   128     2    15      30      34 resource-bound   100/33/27
o        c:P1U2 h:P1U8 w:P4U2          4  12  12     80   128     2    15      30      40 resource-bound   100/47/40
o        c:P1U2 h:P2U4 w:P2U4  (+3 eq)    4  12  12     48   128     2    15      30      34 resource-bound   100/27/20
o        c:P1U4 h:P1U4 w:P4U2  (+2 eq)    4  12  12     88   128     2    15      30      40 resource-bound   100/47/40
o        c:P1U4 h:P1U8 w:P4U1          4  12  12    152   128     2    15      30      56 resource-bound   100/87/73
o        c:P1U4 h:P2U2 w:P2U4  (+7 eq)    4  12  12     56   128     2    15      30      34 resource-bound   100/33/20
o        c:P1U2 h:P1U8 w:P2U4  (+2 eq)    2  12  12     46   128     2    15      30      34 resource-bound   100/27/20
o        c:P1U4 h:P1U4 w:P2U4  (+4 eq)    2  12  12     54   128     2    15      30      34 resource-bound   100/33/20
o        c:P1U4 h:P1U8 w:P2U2          2  12  12     86   128     2    15      30      40 resource-bound   100/47/40
o        c:P1U2 h:P1U8 w:P1U8          1  12  12     45   128     2    15      30      34 resource-bound   100/27/20
o        c:P1U4 h:P1U4 w:P1U8  (+1 eq)    1  12  12     53   128     2    15      30      34 resource-bound   100/33/20
o        c:P1U4 h:P2U4 w:P8U1  (+1 eq)   16  12  12    292   256     1    30      30      54 resource-bound   100/80/77
o        c:P1U4 h:P4U2 w:P4U2  (+2 eq)   16  12  12    164   256     1    30      30      36 resource-bound   100/47/40
o        c:P1U4 h:P8U1 w:P2U4  (+4 eq)   16  12  12    100   256     1    30      30      32 resource-bound   100/27/23
o        c:P1U4 h:P1U8 w:P8U1          8  12  12    284   256     1    30      30      53 resource-bound   100/80/73
K        c:P1U1 h:P1U8 w:P1U8          1  12  12     25    64     4     8      32      44 resource-bound   100/25/25
... (194 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: c:P1U1 h:P1U8 w:P1U8  -> exposure=64, pragma_agg=32 (1.10x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 4 on level 'c' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P4U1              24     28     8     512 tie (control/coalescing sit below the binding term)
  P2U2              22     26     6     512 tie (control/coalescing sit below the binding term)
  P1U4              21     25     5     512 tie (control/coalescing sit below the binding term)
```

For flag and column meanings, see
[`DSE_rules.md#table-columns-and-flags`](../DSE_rules.md#table-columns-and-flags).

## Reading the fixed-product block

The helper's fixed-product block isolates the outer `c` level with `h` and `w`
left at `P1U1`. The aggregate ties because such tiny exposure is dominated by
the per-wave critical path, but the underlying traffic still moves in the
expected direction: `LD_rec` drops from `24` to `21` as the split moves from
`P4U1` to `P1U4`, because unroll carries fewer worker iterators.

The stronger visible split is on the innermost `w` level. At full exposure, the
unroll-heavy row `c:P1U4 h:P1U8 w:P1U8` has `LD_eff = 85` and `sched = 32`, while
the more parallel `w:P4U2` row carries `LD_eff = 152` and `sched = 36` at the
same `p_agg = 29`. The floor stays compute-bound, but `w` unroll keeps memory
pressure comfortably below the arithmetic term.

## Recommendation

**`c:P1U1 h:P1U8 w:P1U8` is the recommended knee (`K`)**: exposure `64`,
`p_agg = 32` (`1.10×` the `29`-cycle floor), resource-bound. It exposes one
channel's full `H×W` tile per wave and runs four waves over `C = 4`.

The recommended split is unroll-heavy with `P_tot = 1`: it reaches the compute
saturation knee while coalescing the contiguous `w` accesses and carrying the
fewest iterators. Larger exposures (`o`) only reduce the wave-serialization gap
toward the floor while adding transient backlog and mapping pressure. The old
"parallel fills memory lanes" framing is stale for batchnorm; under the current
model, unroll is the useful axis for contiguous `w` traffic and for control
amortization.

## Comparing against measured DFG simulator cycles

Use the shared comparison rules in
[`DSE_rules.md#comparing-measured-dfg-cycles`](../DSE_rules.md#comparing-measured-dfg-cycles).
For batchnorm, the DSE floor is below the scalar Metric-1 aggregate in the main
eval because this model credits `w`-vector coalescing, channel/spatial control
amortization, and one-time invariant loads. It is a Loom-pragma estimate, not a
replacement for the main ASAP/CGRA eval.
