# AXPY Loom-Pragma DSE (banking-aware)

Kernel: `tests/app/axpy/axpy.cpp` — loop `compute_loop`

Current source pragma:

```cpp
compute_loop:
LOOM_PARALLEL(4, contiguous)
LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
for (uint32_t i = 0; i < N; i++) {
    output_y[i] = alpha * input_x[i] + input_y[i];
}
```

This file selects `LOOM_PARALLEL(P)` / `LOOM_UNROLL(U)` under the **banking-aware
load/store model** of the "Optional Loom-Pragma Design-Space Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md). It
is a design-space *estimate*, not a lower bound, RTL, or bank-conflict model.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py axpy --config 6x6
```

## Why P and U now differ

`compute_loop` is dependency-parallel (each iteration writes a distinct
`output_y[i]`; `alpha` is read-only). Under the earlier product-only model, `P`
and `U` were indistinguishable — only `P·U` mattered. This model separates them
on the **load/store axis** through *banking*:

- **`LOOM_PARALLEL(P)`** partitions `input_x`/`input_y`/`output_y` into `P`
  contiguous banks, one per worker, so `P` workers load/store on **independent
  ports** — bandwidth scales with `P`.
- **`LOOM_UNROLL(U)`** enlarges one worker's body; its `U` iterations' accesses
  share that one worker's **single port** and serialize — a per-body penalty.

The concurrent load width a chunk can use is `active_L = min(P_tot, banks, L)`,
which rises with `P` but **not** with `U`. Arithmetic stays a global pool, so the
distinction lives entirely in loads/stores (which is where `axpy` is bound). The
op counts of a chunk still depend only on `P·U`; only `active_L`/`active_S`
change with the split.

## Setup

- Resource config: `6x6` (`P = 36`, `L = 12`, `S = 12`)
- Trip count: `256`; distribution `contiguous`
- Per-iteration demand: `L = 3` (`input_x`, `input_y`, `i`), `S = 2`
  (`output_y`, `i`), `P = 4` (mul, add, `i++`, compare); `CP = 4`
- Binding memory class: **loads** (`ld_iter = 3`, the largest term)
- `absolute_cgra_lb = 65` — full-trip aggregate over full lanes
  (`ceil(770/12) = 65`), the **only** lower bound. Reached only when `active_L`
  hits `12`, i.e. `P_tot >= 12`.

## Results

```text
flags  split    Ptot  aL  aS   exp  wav  cagg  p_agg  sched  class           util P/L/S
------ -------- ----- --- --- ---- ---- ----- ------ ------ --------------- ------------
o      i:P8U8      8   8   8   64    4    25    100    108  resource-bound  32/100/64
o      i:P8U4      8   8   8   32    8    13    104    120  resource-bound  31/100/62
o      i:P8U2      8   8   8   16   16     7    112    144  resource-bound  29/100/57
K      i:P8U1      8   8   8    8   32     4    128    192  resource-bound  25/100/50
b      i:P4U8      4   4   4   32    8    25    200    216  resource-bound  16/100/64
b      i:P4U4      4   4   4   16   16    13    208    240  resource-bound  15/100/62
b      i:P4U2      4   4   4    8   32     7    224    288  resource-bound  14/100/57
b      i:P4U1*     4   4   4    4   64     4    256    384  resource-bound  25/100/50
b      i:P2U8      2   2   2   16   16    25    400    432  resource-bound   8/100/64
b      i:P1U1      1   1   1    1  256     5   1280   1536  resource-bound  20/100/40
```

`K` = recommended (saturation knee at max bandwidth), `b` = bandwidth-starved
(memory ports idle, raise `P`), `o` = oversubscribed (extra exposure past the
knee, no throughput gain), `*` = current source pragma. `p_agg` =
`pragma_exposure_aggregate` (wave-summed; above the floor), `sched` =
`schedule_estimate`.

## The P-vs-U distinction, made concrete

At a **fixed product** `P·U = 8`, the split alone changes the estimate by up to
~6×:

| split | P_tot | active_L | p_agg | reading |
|-------|------:|---------:|------:|---------|
| `P=8,U=1` | 8 | 8 | 128 | best — all 8 ports stream |
| `P=4,U=2` | 4 | 4 | 224 | 1.8× slower — 4 ports |
| `P=2,U=4` | 2 | 2 | 416 | 3.2× slower |
| `P=1,U=8` | 1 | 1 | 832 | 6.5× slower — one port, fully serialized |

Same total work, same op counts, same `P·U`; the only difference is that eight
parallel workers stream on eight banks while one fully-unrolled worker piles all
8× the traffic onto one port. This is exactly "unroll increases the penalty per
loop body; parallel enables multiple streams."

## Recommendation

**`LOOM_PARALLEL(8)` with `LOOM_UNROLL(1)`** (the `K` row): the smallest exposure
that saturates the load ports at the maximum bandwidth reachable with
`P <= 8`. Reasoning:

- The current source pragma **`P=4, U=1`** is *bandwidth-starved*: it uses only
  4 of the (up to 8) worker banks and 4 of the 12 load lanes, leaving load
  bandwidth on the floor (`p_agg = 256`, `4.0×` the lower bound). Its load lane
  sits at only `4/12` of the fabric.
- **`P=8, U=1`** doubles the banks to 8 (`active_L = 8`), cutting `p_agg` to
  `128` (`2.0×`). This is the knee: the load class is fully used every cycle
  within a wave.
- Rows past the knee (`P=8, U=2/4/8`, flagged `o`) shave `p_agg` further only
  through per-wave invariant-reload amortization and ceiling rounding — the
  steady-state load rate is unchanged (`active_L` is still 8) — while adding
  unroll area. They are oversubscription, not speed.
- **To beat `128` you must add banks, not unroll.** `active_L` is capped at
  `P_tot` here; raising `LOOM_PARALLEL` to `12` (or banking the arrays to 12)
  would reach `active_L = 12` and the `65`-cycle floor. Unrolling never will.

## Comparing against measured DFG simulator cycles

The estimate brackets the true cost as
`absolute_cgra_lb (65) <= pragma_exposure_aggregate <= schedule_estimate`. When
measured DFG execution cycles are available for a candidate, read
`sim / absolute_cgra_lb` as total distance from the resource floor, and
`sim / schedule_estimate` as overhead beyond the finite-resource schedule (DFG
lowering, mapping, handshake backpressure, memory latency). Real dataflow
pipelines the waves, so it can fall **below** `p_agg`/`sched` toward the floor.

## Notes

`util%(P/L/S)` is steady-state per-class utilization (`term / aggregate`). The
binding class (loads) reads `100%` at and beyond the knee. Unlike the earlier
model, `util_L` and the load term now depend on `P` (via `active_L`), not just
`P·U` — that is the banking asymmetry. This model deliberately captures only the
memory-port asymmetry (favoring `P`) and not control-overhead amortization
(which would favor `U`); see the spec.
