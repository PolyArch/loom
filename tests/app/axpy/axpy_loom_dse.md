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

Regenerate (sweep parallel up to 16 so the 12 load lanes can be saturated):

```bash
python3 tests/scripts/loom_dse.py axpy --config 6x6 --max-parallel 16
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

## Results (`--max-parallel 16`)

```text
flags  split     Ptot  aL  aS   exp  wav  cagg  p_agg  sched  class           util P/L/S
------ --------- ----- --- --- ---- ---- ----- ------ ------ --------------- ------------
o      i:P16U8     16  12  12  128    2    33     66     70  resource-bound  45/100/67
o      i:P16U4     16  12  12   64    4    17     68     76  resource-bound  47/100/65
o      i:P16U2     16  12  12   32    8     9     72     88  resource-bound  44/100/67
K      i:P16U1     16  12  12   16   16     5     80    112  resource-bound  40/100/60
b      i:P8U1      8    8   8    8   32     4    128    192  resource-bound  25/100/50
b      i:P4U1*     4    4   4    4   64     4    256    384  resource-bound  25/100/50
b      i:P2U1      2    2   2    2  128     4    512    768  resource-bound  25/100/50
b      i:P1U1      1    1   1    1  256     5   1280   1536  resource-bound  20/100/40
```

### Column glossary

- `flags`: quick status markers for the candidate. `K` is the recommended knee,
  `b` is bandwidth-starved, `o` is oversubscribed, and `*` marks the current
  source pragma.
- `split`: the loop pragma split being tested. For example, `i:P16U1` means
  `LOOM_PARALLEL(16)` and `LOOM_UNROLL(1)` on loop `i`.
- `Ptot`: total parallel workers requested by the split.
- `aL` / `aS`: active load/store lanes after banking and hardware caps. These
  are the lanes the candidate can actually use; for example, `P16` exposes 16
  workers, but `aL` clamps to 12 on the `6x6` config because only 12 load lanes
  exist.
- `exp`: exposed iterations per wave, roughly `P * U` for this single loop.
- `wav`: number of waves needed to cover the 256 loop iterations.
- `cagg`: aggregate estimate for one exposed wave.
- `p_agg`: wave-summed `pragma_exposure_aggregate`; lower is better for comparing
  candidates, but this is still an estimate above the `absolute_cgra_lb` floor,
  not a lower bound.
- `sched`: finite-resource `schedule_estimate`; also an estimate, not a lower
  bound.
- `class`: whether the wave is limited by critical-path latency or by a resource
  class. Every shown AXPY row is load-resource-bound.
- `util P/L/S`: per-class utilization within the active resources for compute,
  loads, and stores. `L = 100` means the candidate saturates its active load
  lanes; it does not necessarily mean all 12 physical load lanes are reachable
  unless `aL = 12`.

`K` = recommended (fills the 12 load lanes, `active_L = 12`), `b` =
bandwidth-starved (memory ports idle, raise `P`), `o` = oversubscribed (extra
exposure past the knee, no throughput gain), `*` = current source pragma.
`active_L` reaches the full `L = 12` only at `P_tot >= 12`; the powers-of-two
grid's first such value is `16` (rows abbreviated). `p_agg` =
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

The binding load class saturates the fabric only at `active_L = L = 12`, i.e.
`P_tot >= 12`. Walking the table:

- The current source pragma **`P=4, U=1`** is *bandwidth-starved*: it uses only
  4 of the 12 load lanes, leaving load bandwidth on the floor (`p_agg = 256`,
  `4.0×` the lower bound).
- **`P=8, U=1`** doubles the banks to 8 (`active_L = 8`, `p_agg = 128`, `2.0×`) —
  better, but still four lanes idle. This is the ceiling if the fabric caps
  workers at 8.
- **`P=16, U=1` is the recommended knee** (`K`): it fills all 12 lanes
  (`active_L = 12`), reaching `p_agg = 80` (`1.23×` the `65`-cycle floor).
  `P_tot = 12` would hit `active_L = 12` exactly; the powers-of-two grid can only
  land on `16` (with 4 workers beyond the 12 lanes), so an arbitrary Loom factor
  of `12` is the ideal and `16` the grid's approximation.
- Rows with more unroll at fixed `P` (flagged `o`) shave `p_agg` only through
  per-wave invariant-reload amortization and ceiling rounding — the steady-state
  load rate is unchanged — while adding area. **Unroll never raises `active_L`;
  only more parallel/banks do.**

## Comparing against measured DFG simulator cycles

The bracket `absolute_cgra_lb (65) <= pragma_exposure_aggregate <=
schedule_estimate` relates **model quantities only** — it is *not* a bound on
measured DFG cycles. Only `absolute_cgra_lb` is a lower bound (on the resource
model); `p_agg` and `schedule_estimate` embed the no-overlap wave assumption and
sit above it, and neither is an upper bound on real hardware. When measured DFG
execution cycles are available for a candidate, compare them: `sim /
absolute_cgra_lb` is the total distance from the resource floor, `sim / p_agg`
the distance from the wave-serialized aggregate, and `sim / schedule_estimate`
the overhead beyond the finite-resource schedule (DFG lowering, mapping,
handshake backpressure, memory latency). Real pipelined dataflow can fall
**below** `p_agg`/`sched` toward the floor, and real overheads can push measured
cycles **above** `sched` — so measured cycles may land on either side.

## Notes

`util%(P/L/S)` is steady-state per-class utilization (`term / aggregate`). The
binding class (loads) reads `100%` at and beyond the knee. Unlike the earlier
model, `util_L` and the load term now depend on `P` (via `active_L`), not just
`P·U` — that is the banking asymmetry. This model deliberately captures only the
memory-port asymmetry (favoring `P`) and not control-overhead amortization
(which would favor `U`); see the spec.
