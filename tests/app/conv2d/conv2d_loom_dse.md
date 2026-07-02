# conv2d Loom-Pragma DSE (banking-aware)

Kernel: `tests/app/conv2d/conv2d.cpp` —
`output[co,oh,ow] = Σ_{ci,kh,kw} input[...]·weight[...]`

Modeled as a two-level nest: an outer **parallel** loop over output pixels
(`out = C_out·OH·OW`) and an inner **reduction** over the `K = C_in·KH·KW` taps.

Regenerate: `python3 tests/scripts/loom_dse.py conv2d --config 6x6 --top 6`

## Banking / assumptions

Output pixels are independent → the outer loop is parallel and partitions the
input/output over the pixel workers (`B = P_tot`, `active_L = min(P_tot, L)`).
The taps are a reduction fully consumed per pixel; tap parallel/unroll only
reshapes the (product-only) merge tree. conv2d is strongly **load-bound**
(2 loads per tap: `input` and `weight`).

> Assumption: input-window (halo) reuse across neighboring output pixels and
> weight sharing across pixels are **not** modeled — loads are counted
> conservatively per tap. This inflates absolute load counts but does not change
> the P-vs-U conclusion (both splits see the same op counts).

## Setup

- `6x6`; `C_in=3, C_out=4, H=W=8, KH=KW=3, stride=1` → `out = 144`, `K = 27`.
- Full-trip counts: `A=15696 LD=11809 ST=4176 CP=8` → `compute=436 load=985
  store=348`. `absolute_cgra_lb = 985` (`ceil(11809/12)`), reached at
  `active_L = 12` (`P_tot >= 12`).

## Results (top of the sweep)

```text
flags  split               Ptot  aL  aS   exp  wav  cagg  p_agg  sched  class           util P/L/S
------ ------------------- ----- --- --- ---- ---- ----- ------ ------ --------------- ------------
o      out:P8U2 tap:P1U1     8   8   8  432    9   165   1485   1512  resource-bound  30/100/35
K      out:P8U1 tap:P1U1     8   8   8  216   18    83   1494   1530  resource-bound  30/100/35
o      out:P8U4 tap:P1U1     8   8   8  864    5   329   1645   1660  resource-bound  29/100/35
b      out:P4U2 tap:P1U1     4   4   4  216   18   165   2970   3024  resource-bound  15/100/35
b      out:P1U1 tap:P1U1     1   1   1   27  144    83  11952  12240  resource-bound   5/100/35
```

Each row has `+12 equivalent` inner-tap variations (inert, like gemv's inner
column loop).

## The P-vs-U distinction

Fixed product `P·U = 8` on the output-pixel level:

| out split | active_L | p_agg | reading |
|-----------|---------:|------:|---------|
| `P=8,U=1` | 8 | 1494 | best (8 ports) |
| `P=4,U=2` | 4 | 2970 | 2.0× slower |
| `P=2,U=4` | 2 | 5922 | 4.0× slower |
| `P=1,U=8` | 1 | 11826 | 7.9× slower |

A clean bandwidth-scaling curve: doubling parallel halves the estimate, doubling
unroll leaves it (serialized on one port).

## Recommendation

**`LOOM_PARALLEL(8)` over output pixels, `LOOM_UNROLL(1)`, taps at `1`.** This is
the knee at the maximum bandwidth reachable with `P <= 8` (`active_L = 8`,
`1.52×` the floor). Rows past it (`o`) only amortize per-wave rounding. To reach
the `985` floor, expose `P >= 12` output pixels (12 banks); unroll and tap
parallelism cannot help.
