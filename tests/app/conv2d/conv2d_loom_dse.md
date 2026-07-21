# conv2d Loom-Pragma DSE (extended analytical profile)

Shared definitions, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This note records only the Conv2d setup,
reproducible helper output, and recommendation.

Kernel: `tests/app/conv2d/conv2d.cpp` computes
`output[co,oh,ow] = sum_tap(input[...] * kernel[...])`.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py conv2d --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 24
```

## Conv2d-specific setup

The helper fixture uses `C_in=3`, `C_out=4`, `H=W=8`, `KH=KW=3`, stride one,
so the independent output levels are `co=4`, `oh=ow=6` and the fully consumed
reduction is `tap=27`. All six `co/oh/ow` permutations are legal with `tap`
pinned innermost. Tile sizes are searched independently on `co`, `oh`, and `ow`.

For every concrete tile the helper constructs exact unique input-halo, weight,
and output address sets. Input and weight are derived `resident_shared`; output
is streaming/direct. The whole reuse set is 192 input plus 108 weight elements,
or 1,200 bytes, so it is legal in serial mode and in the ideal-DMA profile's
2,400-byte double buffer. Direct input remains 27 scalar tap loads per reader;
the compact resident input uses nine contiguous scratchpad groups. Declared jam
edges share input across unrolled `co` copies and weights across unrolled spatial
copies, while each output retains its own reduction.

## Results

```text
# Loom pragma DSE (analytic_prefilter): conv2d  (6x6)

Evidence: `analytic_prefilter`; target `shared-spad-4k-v4`; preload mode `serial`; one 4096-byte scratchpad shared across this kernel; 4 cyclic single-ported banks; 1-cycle access; fixed V=4.
Search: complete legal power-of-two factors through each trip count.
Candidates: 15360 legal, 15360 deduplicated groups; `absolute_cgra_lb=453` is the profile-global floor.

flags    candidate                                                                                                plan_lb   p_agg   sched tiles  cap_B   bank lb/s class           util P/L/S
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
K        co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>oh>ow>tap tile=co:4,oh:6,ow:6                                      453     453     511     1   1200     453/511 resource-bound     57/10/2
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>ow>oh>tap tile=co:4,oh:6,ow:6                                      453     453     511     1   1200     453/511 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=oh>co>ow>tap tile=co:4,oh:6,ow:6                                      453     453     511     1   1200     453/511 resource-bound     57/10/2
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=oh>ow>co>tap tile=co:4,oh:6,ow:6                                      453     453     511     1   1200     453/511 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=ow>co>oh>tap tile=co:4,oh:6,ow:6                                      453     453     511     1   1200     453/511 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=ow>oh>co>tap tile=co:4,oh:6,ow:6                                      453     453     511     1   1200     453/511 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>oh>ow>tap tile=co:4,oh:4,ow:6                                      492     492     550     2   1008     492/550 resource-bound     57/10/2
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>oh>ow>tap tile=co:4,oh:6,ow:4                                      492     492     550     2   1008     492/550 resource-bound     57/10/2
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>ow>oh>tap tile=co:4,oh:4,ow:6                                      492     492     550     2   1008     492/550 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>ow>oh>tap tile=co:4,oh:6,ow:4                                      492     492     550     2   1008     492/550 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=oh>co>ow>tap tile=co:4,oh:4,ow:6                                      492     492     550     2   1008     492/550 resource-bound     57/10/2
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=oh>co>ow>tap tile=co:4,oh:6,ow:4                                      492     492     550     2   1008     492/550 resource-bound     57/10/2
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=oh>ow>co>tap tile=co:4,oh:4,ow:6                                      492     492     550     2   1008     492/550 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=oh>ow>co>tap tile=co:4,oh:6,ow:4                                      492     492     550     2   1008     492/550 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=ow>co>oh>tap tile=co:4,oh:4,ow:6                                      492     492     550     2   1008     492/550 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=ow>co>oh>tap tile=co:4,oh:6,ow:4                                      492     492     550     2   1008     492/550 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=ow>oh>co>tap tile=co:4,oh:4,ow:6                                      492     492     550     2   1008     492/550 resource-bound     57/10/4
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=ow>oh>co>tap tile=co:4,oh:6,ow:4                                      492     492     550     2   1008     492/550 resource-bound     57/10/4
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=co>oh>ow>tap tile=co:4,oh:6,ow:6                                      507     507     567     1   1200     507/567 resource-bound     50/10/1
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=co>ow>oh>tap tile=co:4,oh:6,ow:6                                      507     507     567     1   1200     507/567 resource-bound     50/10/3
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=oh>co>ow>tap tile=co:4,oh:6,ow:6                                      507     507     567     1   1200     507/567 resource-bound     50/10/1
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=oh>ow>co>tap tile=co:4,oh:6,ow:6                                      507     507     567     1   1200     507/567 resource-bound     50/10/3
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=ow>co>oh>tap tile=co:4,oh:6,ow:6                                      507     507     567     1   1200     507/567 resource-bound     50/10/3
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=ow>oh>co>tap tile=co:4,oh:6,ow:6                                      507     507     567     1   1200     507/567 resource-bound     50/10/3
... (15336 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>oh>ow>tap tile=co:4,oh:6,ow:6  -> plan_lb=453, p_agg=453, sched=511, resource-bound
flags: K=recommended, b=below knee (latency-bound or recurring-traffic immature), o=oversubscribed.
Order: `co>oh>ow>tap`.
Jam: co->oh, co->ow, co->tap[input], oh->ow, oh->tap[weight], ow->tap[weight].
Memory: input=resident_shared(base_elem=0,replicas=1,lifetime=refill/tile,capacity_x=1); weight=resident_shared(base_elem=192,replicas=1,lifetime=refill/tile,capacity_x=1); output=direct.
Tile: co:4,oh:6,ow:6; tails=none; num_tiles=1.
Capacity: 1200/4096 B; held_region=0 B; refill_frame=1200 B; frame_bases_B=0.
Banks: bank_lb=453 cycles; bank_sched=511 cycles; gap=58 cycles.
Traffic: preload=300 scalar elements, 75 external-L ops, 75 scratchpad-S ops; spad_reads=1404 scalar requests after jam fan-out; avoided_direct=1104 scalar external loads.
Direct-memory audit reference (excluded from resident-profile legality, ranking, and floor): plan_lb=213, p_agg=215, sched=253.

Ideal-DMA sensitivity (same config): co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=oh>co>ow>tap tile=co:4,oh:4,ow:4 -> absolute_cgra_lb=432, plan_lb=432, p_agg=432, sched=490.
Assumption: inactive ping-pong fill does not contend with current-tile scratchpad reads.

4x4 recommendation: co:P1U4 oh:P1U2 ow:P1U2 tap:P1U1 order=oh>co>ow>tap tile=co:4,oh:6,ow:4.
8x8 recommendation: co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>oh>ow>tap tile=co:4,oh:6,ow:6.
```

## Recommendation

Canonical serial selects the whole reuse set with source order and
`co/oh/ow = P1U4/P1U4/P1U4`. The one-tile plan preloads 300 unique elements,
avoids 1,104 scalar external loads, reaches its 453-cycle profile floor, and has
`sched=511`; the 58-cycle bank gap makes the named single-ported scratchpad the
dominant modeled constraint. The direct audit reference is faster in cycles but
does not receive the resident profile's external-traffic reduction and is not a
placement alternative in this search.

The ideal-DMA sensitivity chooses `order=oh>co>ow>tap` with
`tile=co:4,oh:4,ow:4`, reaching `p_agg=432` and `sched=490`. That result depends
on double-buffer capacity, preload/compute overlap, and the printed no-contention
assumption; it does not replace the canonical serial recommendation.
