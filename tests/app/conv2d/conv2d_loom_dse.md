# conv2d Loom-Pragma DSE (banking-aware)

Kernel: `tests/app/conv2d/conv2d.cpp` —
`output[co,oh,ow] = Σ_{ci,kh,kw} input[...]·weight[...]`

Modeled as a two-level nest: an outer **parallel** loop over output pixels
(`out = C_out·OH·OW`) and an inner **reduction** over the `K = C_in·KH·KW` taps.

Regenerate: `python3 tests/scripts/loom_dse.py conv2d --config 6x6 --top 6`

## Banking / assumptions

Output pixels are independent → the outer loop is parallel. **`input` carries
`LOOM_MEMORY_BANK(4, block)`** (`conv2d.cpp:61`), so its bank count is **capped
at 4**: `active_L = min(P_tot, 4, L) <= 4`. This makes conv2d behave like gemv —
parallelism scales input bandwidth only up to the 4-bank cap. The taps are a
reduction fully consumed per pixel; tap parallel/unroll only reshapes the
(product-only) merge tree. conv2d is strongly **load-bound** (2 loads per tap).

> Assumptions: (1) `weight` is uncapped but, in this single-binding-array model,
> rides the same modeled load width as the capped `input` (conservative — a
> per-array port model could let weight use more lanes). (2) input-window (halo)
> reuse across neighboring output pixels and weight sharing are **not** modeled —
> loads are counted per tap. Neither changes the P-vs-U conclusion (both splits
> see the same op counts).

## Setup

- `6x6`; `C_in=3, C_out=4, H=W=8, KH=KW=3, stride=1` → `out = 144`, `K = 27`.
- Full-trip counts: `A=15696 LD=11809 ST=4176 CP=8` → `compute=436 load=985
  store=348`. `absolute_cgra_lb = 985` (`ceil(11809/12)`, full lanes) — but
  `input`'s 4-bank cap means no pragma reaches it: the real ceiling is
  `active_L = 4`, exactly as in gemv.

## Results (selected rows)

These rows are hand-picked to show the bandwidth-scaling curve down to a single
worker; they are **not** the verbatim `--top 6` listing. The helper's top 6 ranks
`out:P4U2 tap:P1U1` (`p_agg = 2970`) and `out:P8U4 tap:P1U1` (`p_agg = 3285`)
above the bandwidth-starved `out:P2U1` / `out:P1U1` shown here — the latter are
kept because they make the `P`-halving progression legible. Regenerate the full
ranking with the command above.

```text
flags  split               Ptot  aL  aS   exp  wav  cagg  p_agg  sched  class           util P/L/S
------ ------------------- ----- --- --- ---- ---- ----- ------ ------ --------------- ------------
o      out:P8U2 tap:P1U1     8   4   8  432    9   329   2961   2970  resource-bound  15/100/18
o      out:P4U4 tap:P1U1     4   4   4  432    9   329   2961   2988  resource-bound  15/100/35
o      out:P8U1 tap:P1U1     8   4   8  216   18   165   2970   2988  resource-bound  15/100/18
K      out:P4U1 tap:P1U1     4   4   4  108   36    83   2988   3060  resource-bound  16/100/35
b      out:P2U1 tap:P1U1     2   2   2   54   72    83   5976   6120  resource-bound   8/100/35
b      out:P1U1 tap:P1U1     1   1   1   27  144    83  11952  12240  resource-bound   5/100/35
```

Each row has `+12 equivalent` inner-tap variations (inert, like gemv's inner
column loop). Note `out:P8` rows (`active_L` still `4`, flagged `o`) waste
workers on the 4-bank cap — exactly the gemv lesson.

## The P-vs-U distinction

Fixed product `P·U = 8` on the output-pixel level:

| out split | active_L | p_agg | reading |
|-----------|---------:|------:|---------|
| `P=8,U=1` | 4 | 2970 | best (bank-capped at 4) |
| `P=4,U=2` | 4 | 2970 | **equal** — 4 workers already fill 4 banks |
| `P=2,U=4` | 2 | 5922 | 2.0× slower — unroll serializes |
| `P=1,U=8` | 1 | 11826 | 4.0× slower — unroll serializes |

Parallel beats unroll up to the 4-bank cap, then saturates (`P=8` ties `P=4`),
mirroring gemv.

## Recommendation

**`LOOM_PARALLEL(4)` over output pixels, `LOOM_UNROLL(1)`, taps at `1`.** `P=4`
exactly fills `input`'s 4 banks (`active_L = 4`, `3.03×` the floor); more
output-pixel parallelism (`P=8`, flagged `o`) wastes workers on 4 banks, and tap
parallelism is inert. To go faster you must **raise `input`'s banking**
(`LOOM_MEMORY_BANK(12)` to use all 12 lanes → the `985` floor); unroll and tap
parallelism cannot help.

> The current source (`conv2d.cpp:81`) already uses `LOOM_PARALLEL(4, contiguous)`
> on the output-channel loop — i.e. `P=4` on the parallel dim, which is exactly
> the bank-cap-aligned choice this model recommends. Its `LOOM_UNROLL(4)` on the
> `oh` loop adds no load bandwidth (it only enlarges a worker's body); banking
> `input` beyond 4 is the only lever that would help.
