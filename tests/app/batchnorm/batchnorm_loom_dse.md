# batchnorm Loom-Pragma DSE (extended analytical profile)

Shared definitions, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This note records only the Batchnorm setup,
reproducible helper output, and recommendation.

Kernel: `tests/app/batchnorm/batchnorm.cpp`. The helper fixture uses
`C=4`, `H=W=8` and the named `shared-spad-4k-v4` analytical target.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py batchnorm --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 24
```

## Batchnorm-specific setup

The legal orders are source `(c,h,w)` and interchange `(c,w,h)`. Only `w`
increments the channel-major address by one, so source order can coalesce its
innermost accesses at fixed `V=4`; making `h` innermost leaves stride `W=8` and
therefore scalar traffic. Input and output remain direct. The per-channel
`mean`, `variance`, `gamma`, and `beta` parameters are loaded once; `inv_std` is
computed once per exposed channel, and all five values are reused across its
spatial points. The selected `h` unroll derives the structural `h->w` jam without
adding arithmetic or control.

## Results

```text
# Loom pragma DSE (analytic_prefilter): batchnorm  (6x6)

Evidence: `analytic_prefilter`; target `shared-spad-4k-v4`; preload mode `serial`; one 4096-byte scratchpad shared across this kernel; 4 cyclic single-ported banks; 1-cycle access; fixed V=4.
Search: complete legal power-of-two factors through each trip count.
Candidates: 1200 legal, 1200 deduplicated groups; `absolute_cgra_lb=29` is the profile-global floor.

flags    candidate                                                                                                plan_lb   p_agg   sched tiles  cap_B   bank lb/s class           util P/L/S
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
o        c:P1U4 h:P1U8 w:P4U2 order=c>h>w tile=untiled                                                                 29      29      36     1      0         0/0 resource-bound   100/38/38
o        c:P1U4 h:P2U4 w:P2U4 order=c>h>w tile=untiled                                                                 29      29      32     1      0         0/0 resource-bound   100/21/21
o        c:P1U4 h:P4U2 w:P1U8 order=c>h>w tile=untiled                                                                 29      29      32     1      0         0/0 resource-bound   100/21/21
o        c:P2U2 h:P1U8 w:P2U4 order=c>h>w tile=untiled                                                                 29      29      32     1      0         0/0 resource-bound   100/21/21
o        c:P2U2 h:P2U4 w:P1U8 order=c>h>w tile=untiled                                                                 29      29      32     1      0         0/0 resource-bound   100/21/21
o        c:P4U1 h:P1U8 w:P1U8 order=c>h>w tile=untiled                                                                 29      29      32     1      0         0/0 resource-bound   100/21/21
o        c:P1U4 h:P1U8 w:P4U2 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P1U4 h:P2U4 w:P2U4 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P1U4 h:P4U2 w:P1U8 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P2U2 h:P1U8 w:P2U4 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P2U2 h:P2U4 w:P1U8 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P4U1 h:P1U8 w:P1U8 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P1U4 h:P1U8 w:P2U4 order=c>h>w tile=untiled                                                                 29      29      32     1      0         0/0 resource-bound   100/21/21
o        c:P1U4 h:P2U4 w:P1U8 order=c>h>w tile=untiled                                                                 29      29      32     1      0         0/0 resource-bound   100/21/21
o        c:P2U2 h:P1U8 w:P1U8 order=c>h>w tile=untiled                                                                 29      29      32     1      0         0/0 resource-bound   100/21/21
o        c:P1U4 h:P1U8 w:P2U4 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P1U4 h:P2U4 w:P1U8 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P2U2 h:P1U8 w:P1U8 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P1U4 h:P1U8 w:P1U8 order=c>h>w tile=untiled                                                                 29      29      32     1      0         0/0 resource-bound   100/21/21
o        c:P1U4 h:P1U8 w:P1U8 order=c>w>h tile=untiled                                                                 29      29      53     1      0         0/0 resource-bound   100/76/76
o        c:P1U2 h:P1U8 w:P8U1 order=c>h>w tile=untiled                                                                 30      30      58     1      0         0/0 resource-bound   100/80/80
o        c:P1U2 h:P2U4 w:P4U2 order=c>h>w tile=untiled                                                                 30      30      40     1      0         0/0 resource-bound   100/40/40
o        c:P1U2 h:P4U2 w:P2U4 order=c>h>w tile=untiled                                                                 30      30      34     1      0         0/0 resource-bound   100/27/27
o        c:P1U2 h:P8U1 w:P1U8 order=c>h>w tile=untiled                                                                 30      30      34     1      0         0/0 resource-bound   100/27/27
K        c:P1U1 h:P1U8 w:P1U8 order=c>h>w tile=untiled                                                                 29      32      44     1      0         0/0 resource-bound   100/25/25
... (1175 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: c:P1U1 h:P1U8 w:P1U8 order=c>h>w tile=untiled  -> plan_lb=29, p_agg=32, sched=44, resource-bound
flags: K=recommended, b=below knee (latency-bound or recurring-traffic immature), o=oversubscribed.
Order: `c>h>w`.
Jam: h->w.
Memory: input=direct; output=direct.
Tile: untiled; tails=none; num_tiles=1.
Capacity: 0/4096 B; held_region=0 B; refill_frame=0 B; frame_bases_B=none.
Banks: bank_lb=0 cycles; bank_sched=0 cycles; gap=0 cycles.
Traffic: preload=0 scalar elements, 0 external-L ops, 0 scratchpad-S ops; spad_reads=0 scalar requests after jam fan-out; avoided_direct=0 scalar external loads.

Ideal-DMA sensitivity (same config): c:P1U1 h:P1U8 w:P1U8 order=c>h>w tile=untiled -> absolute_cgra_lb=29, plan_lb=29, p_agg=32, sched=44.
Assumption: inactive ping-pong fill does not contend with current-tile scratchpad reads.

4x4 recommendation: c:P1U1 h:P1U8 w:P1U8 order=c>h>w tile=untiled.
8x8 recommendation: c:P1U2 h:P1U8 w:P1U8 order=c>h>w tile=untiled.
```

## Recommendation

The canonical serial knee is `c:P1U1 h:P1U8 w:P1U8` in source order. It exposes
one channel's full spatial plane per wave, keeps `w` contiguous, and reaches
`p_agg=32` with `sched=44` against the 29-cycle profile floor. The interchanged
order has the same arithmetic floor but materially higher load/store pressure
because innermost `h` is strided. There is no resident allocation or preload, so
the ideal-DMA sensitivity is intentionally identical to serial.
