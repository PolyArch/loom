# batchnorm Loom-Pragma DSE (banking-aware)

Kernel: `tests/app/batchnorm/batchnorm.cpp` —
`output[c,h,w] = gamma[c]·(input[c,h,w] − mean[c])·inv_std[c] + beta[c]`

A three-level nest, **all parallel** (`c`, `h`, `w`) — no carried dependence.

Regenerate: `python3 tests/scripts/loom_dse.py batchnorm --config 6x6 --top 8`

## Banking

`input`/`output` are partitioned over all three parallel dims, so
`B = P_tot = p_c·p_h·p_w`. With three dims, `P_tot` easily exceeds the 12 load
lanes, so `active_L` can reach the full `L = 12` (unlike the single-level kernels,
which are capped at `P <= 8`). `mean`/`variance`/`gamma`/`beta` are per-channel
invariants (loaded once per exposed channel).

## Setup

- `6x6`; `C=4, H=W=8` (256 pixels). Full-trip counts: `A=1620 LD=568 ST=548
  CP=8` → `compute=45 load=48 store=46`. `absolute_cgra_lb = 48` (`ceil(568/12)`).
- **Note the narrow margin:** `compute=45`, `store=46`, `load=48`. batchnorm has
  ~4 arithmetic ops per pixel against ~1 input load, so it sits close to the
  compute/load boundary — see below.

## Results (top of the sweep; `+N equiv` = P/U factorings and inner splits)

```text
flags  split                 Ptot  aL  aS   exp  wav  cagg  p_agg  sched  class           util P/L/S
------ --------------------- ----- --- --- ---- ---- ----- ------ ------ --------------- ------------
o      c:P1U2 h:P2U4 w:P8U1    16  12  12  128    2    24     48     80  resource-bound  96/100/96
o      c:P1U1 h:P2U4 w:P8U1    16  12  12   64    4    13     52     84  resource-bound  92/100/92
K      c:P1U2 h:P4U1 w:P4U1    16  12  12   32    8     8     64    104  resource-bound  88/100/88
```

## The P-vs-U distinction

Fixed product `P·U = 4` on the channel level (others at `P1U1`):

| c split | active_L | p_agg | reading |
|---------|---------:|------:|---------|
| `P=4,U=1` | 4 | 576 | best |
| `P=2,U=2` | 2 | 1152 | 2.0× slower |
| `P=1,U=4` | 1 | 2304 | 4.0× slower |

The distinction holds, but note the **headroom is thin**: at the recommended
knee the compute lane is already at `88%` utilization. If the per-pixel
arithmetic were a little heavier, batchnorm would become **compute-bound**, and
since arithmetic is a global pool, P and U would then *stop* differing (see the
"when a kernel shows no distinction" note in the DSE analysis).

## Recommendation

**Distribute `P·U ≈ 16` across the parallel dims to reach `active_L = 12`** (all
load lanes) with the smallest exposure — e.g. `h=4, w=4` parallel (`P_tot = 16`),
minimal unroll. This hits `active_L = 12` and `p_agg = 64` (`1.33×` the floor,
the tightest of the six kernels because three parallel dims can fully fill the
lanes). Rows past the knee (`o`) squeeze the last few percent only by amortizing
the per-channel invariant reloads over larger tiles — a wave-serialization
artifact, not more bandwidth.
