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

## Results (selected rows; `+N equiv` = P/U factorings and inner splits)

These are three illustrative rows — the widest-exposure `o` row, a mid `o` row,
and the recommended `K` knee — hand-picked to bracket the sweep, **not** the
verbatim `--top 8` listing (which shows several more `o` rows between the first
and the `K` row). Regenerate the full ranking with the command above.

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

**Distribute parallelism across the dims until `active_L = 12`** (all load
lanes), with minimal unroll. The spec-level saturation target is
`P_tot = 12` — the smallest worker count that fills the 12 load lanes. The helper
enumerates **powers of two**, so its grid can only land on `P_tot = 16` (e.g.
`h=4, w=4`), which also gives `active_L = min(16, 12) = 12` but overshoots the
true target of 12; an arbitrary Loom factoring (e.g. `c=3, h=2, w=2` → 12, or
`c=4, h=3`) would hit `P_tot = 12` exactly.

Keep two things separate here: the **port-width target** and the **wave-summed
estimate**. `active_L = 12` is only the port-width condition — it says the load
lanes are full; it does **not** by itself pin `p_agg`, which also depends on the
tiling's exposure (and hence its wave count). The `p_agg = 64` figure (`1.33×`
the floor — the tightest of the six kernels, because three parallel dims can fill
the lanes) belongs specifically to the recommended `c:P1U2 h:P4U1 w:P4U1` row:
`P_tot = 16` at exposure `32`, i.e. 8 waves. Other configurations that also reach
`active_L = 12` land higher, because their smaller tiles run more waves — the
powers-of-two `h=4, w=4` (exposure 16) gives `p_agg = 128`, and the arbitrary
`P_tot = 12` factorings above (exposure 12) give `p_agg = 192–256`. So the
guidance is: hit `active_L = 12` for full ports, then take the smallest exposure
that still saturates. Rows past that knee (`o`) squeeze the last few percent only
by amortizing the per-channel invariant reloads over larger tiles — a
wave-serialization artifact, not more bandwidth.
