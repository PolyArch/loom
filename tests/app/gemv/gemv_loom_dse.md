# gemv Loom-Pragma DSE (banking-aware)

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

`gemv` is the flagship **nested** case: an outer **parallel** row loop and an
inner **reduction** column loop. It shows that *where* you place parallelism
matters, not just how much.

Regenerate: `python3 tests/scripts/loom_dse.py gemv --config 6x6 --top 8`

## Banking (the key structure)

`A` carries `LOOM_MEMORY_BANK(4, block)` — **block-partitioned over rows (`i`)**.
So:

- `A`'s effective banks `B_L = min(4, p_i)`: row-parallelism scales `A` load
  bandwidth, but **only up to 4** (the bank cap).
- Column-parallelism (`p_j`) adds **no** `A` ports, because `A` is not banked
  over columns. Inner `j` parallel/unroll only reshapes the (product-only)
  reduction tree — it does **not** change throughput.
- `x[j]` is a broadcast vector (loaded once per chunk, reused across the chunk's
  rows). `output_y` is partitioned over rows (`B_S = p_i`).

`gemv` is load-bound on `A` (`M·N` loads dominate).

## Setup

- `6x6`, `M = N = 64`. Full-trip counts: `A=16640 LD=8388 ST=4224 CP=11` →
  `compute=463 load=699 store=352`.
- `absolute_cgra_lb = 699` (`ceil(8388/12)`, full lanes). But `A`'s 4-bank cap
  means no pragma reaches it: the real ceiling is `active_L = min(p_i, 4) = 4`.

## Results (top of the sweep)

```text
flags  split           Ptot  aL  aS   exp  wav  cagg  p_agg  sched  class           util P/L/S
------ --------------- ----- --- --- ---- ---- ----- ------ ------ --------------- ------------
o      i:P8U1 j:P1U1     8   4   8  512    8   277   2216   2232  resource-bound  21/100/24
K      i:P4U1 j:P1U1     4   4   4  256   16   147   2352   2384  resource-bound  20/100/45
b      i:P2U1 j:P1U1     2   2   2  128   32   164   5248   5312  resource-bound   9/100/40
b      i:P1U1 j:P1U1     1   1   1   64   64   198  12672  12736  resource-bound   4/100/33
```

Every row has `+15 equivalent` candidates — those are all the inner-`j`
pragma variations, which are **inert** (they collapse to the same numbers). That
is the "does not demonstrate the distinction on that level" case: inner-column
parallelism/unroll changes nothing because `A` is banked over rows, not columns.

## The P-vs-U distinction (row level)

At fixed product `P·U = 8` on rows:

| row split | P_tot | active_L | p_agg | reading |
|-----------|------:|---------:|------:|---------|
| `P=8,U=1` | 8 | 4 | 2216 | best (bank-capped at 4) |
| `P=4,U=2` | 4 | 4 | 2216 | **equal** — 4 workers already saturate 4 banks |
| `P=2,U=4` | 2 | 2 | 4432 | 2.0× slower — unroll serializes |
| `P=1,U=8` | 1 | 1 | 8864 | 4.0× slower — unroll serializes |

Two lessons in one table: parallel beats unroll (up to 4×), **and** the benefit
of parallel saturates at the 4-bank cap (`P=8,U=1` ties `P=4,U=2`).

## Recommendation

**`LOOM_PARALLEL(4)` on rows, `LOOM_UNROLL(1)`, inner loop left at `1`.**
`p_i = 4` exactly fills `A`'s 4 banks (`active_L = 4`); more row-parallelism
(`P=8`, flagged `o`) wastes workers on 4 banks with no throughput gain, and inner
`j` parallelism is inert. To go faster you must **increase `A`'s banking**
(`LOOM_MEMORY_BANK(8)` or 2-D banking over columns), not add unroll or inner
parallelism.

> The `o` rows show a slightly lower `p_agg` than the `K` row (`2216` vs `2352`).
> That is **not** more bandwidth (both are `active_L=4`); it is the broadcast-`x`
> reload amortizing over larger chunks (fewer waves = fewer `x` re-reads). It
> costs extra workers/unroll for a wave-serialization artifact that vanishes
> under pipelined execution, so it is flagged oversubscribed.
