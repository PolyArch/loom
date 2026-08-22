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

The `main.cpp` smoke-test fixture uses `C_in=3`, `C_out=4`, `H=W=8`,
`KH=KW=3`, and unit height/width strides, so the independent output levels are
`co=4`, `oh=ow=6` and the fully consumed reduction is `tap=27`. All six
`co/oh/ow` permutations are legal with `tap` pinned innermost. There is no
tile-size search. The whole-kernel address sets are 192 input, 108 weight, and
144 output elements. Input and weight form one 1,200-byte proposed
`resident_shared` set; output stays direct. If capacity is below 1,200 bytes,
both reuse-bearing buffers become `direct-fallback`. The explicit complete jam
choices range from `none` through input-only or weight-only sharing to
`share-all`; each nonempty plan requires the corresponding outer `U>1`. Jam may
share input across unrolled `co` copies and weights across unrolled spatial
copies, while every output retains its own reduction.

## Results

```text
# Loom pragma DSE (analytic_prefilter): conv2d  (6x6)

Evidence: `analytic_prefilter`; target `shared-spad-4k-r2w2-v4`; one 4096-byte scratchpad shared across this kernel; R=2, W=2; 1-cycle non-pipelined access; fixed V=4.
Search: complete legal power-of-two factors through each trip count.
Candidates: 4374 legal, 4374 deduplicated groups; `absolute_cgra_lb=256` is the profile-global floor.

flags    candidate                                                                                                        plan_lb   p_agg   sched  cap_B   spad lb/s class           util P/L/S
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
K        co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>oh>ow>tap jam=share-all                                                    256     265     268   1200     256/256 resource-bound     95/17/3
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>ow>oh>tap jam=share-all                                                    256     265     275   1200     256/256 resource-bound     95/17/6
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=oh>co>ow>tap jam=share-all                                                    256     265     268   1200     256/256 resource-bound     95/17/3
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=oh>ow>co>tap jam=share-all                                                    256     265     275   1200     256/256 resource-bound     95/17/6
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=ow>co>oh>tap jam=share-all                                                    256     265     275   1200     256/256 resource-bound     95/17/6
         co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=ow>oh>co>tap jam=share-all                                                    256     265     275   1200     256/256 resource-bound     95/17/6
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=co>oh>ow>tap jam=share-all                                                    284     284     284   1200     284/284 resource-bound     88/18/2
b        co:P1U4 oh:P1U4 ow:P1U2 tap:P1U1 order=co>oh>ow>tap jam=share-all                                                    284     284     287   1200     284/284 resource-bound     88/18/4
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=co>ow>oh>tap jam=share-all                                                    284     284     290   1200     284/284 resource-bound     88/18/6
b        co:P1U4 oh:P1U4 ow:P1U2 tap:P1U1 order=co>ow>oh>tap jam=share-all                                                    284     284     290   1200     284/284 resource-bound     88/18/6
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=oh>co>ow>tap jam=share-all                                                    284     284     284   1200     284/284 resource-bound     88/18/2
b        co:P1U4 oh:P1U4 ow:P1U2 tap:P1U1 order=oh>co>ow>tap jam=share-all                                                    284     284     287   1200     284/284 resource-bound     88/18/4
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=oh>ow>co>tap jam=share-all                                                    284     284     290   1200     284/284 resource-bound     88/18/6
b        co:P1U4 oh:P1U4 ow:P1U2 tap:P1U1 order=oh>ow>co>tap jam=share-all                                                    284     284     290   1200     284/284 resource-bound     88/18/6
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=ow>co>oh>tap jam=share-all                                                    284     284     290   1200     284/284 resource-bound     88/18/6
b        co:P1U4 oh:P1U4 ow:P1U2 tap:P1U1 order=ow>co>oh>tap jam=share-all                                                    284     284     290   1200     284/284 resource-bound     88/18/6
b        co:P1U4 oh:P1U2 ow:P1U4 tap:P1U1 order=ow>oh>co>tap jam=share-all                                                    284     284     290   1200     284/284 resource-bound     88/18/6
b        co:P1U4 oh:P1U4 ow:P1U2 tap:P1U1 order=ow>oh>co>tap jam=share-all                                                    284     284     290   1200     284/284 resource-bound     88/18/6
         co:P1U4 oh:P1U4 ow:P2U2 tap:P1U1 order=co>oh>ow>tap jam=share-all                                                    284     284     285   1200     284/284 resource-bound     87/17/3
         co:P1U4 oh:P2U2 ow:P1U4 tap:P1U1 order=co>oh>ow>tap jam=share-all                                                    284     284     284   1200     284/284 resource-bound     87/17/2
         co:P1U4 oh:P1U4 ow:P2U2 tap:P1U1 order=co>ow>oh>tap jam=share-all                                                    284     284     287   1200     284/284 resource-bound     87/17/6
         co:P1U4 oh:P2U2 ow:P1U4 tap:P1U1 order=co>ow>oh>tap jam=share-all                                                    284     284     287   1200     284/284 resource-bound     87/17/6
         co:P1U4 oh:P1U4 ow:P2U2 tap:P1U1 order=oh>co>ow>tap jam=share-all                                                    284     284     285   1200     284/284 resource-bound     87/17/3
         co:P1U4 oh:P2U2 ow:P1U4 tap:P1U1 order=oh>co>ow>tap jam=share-all                                                    284     284     284   1200     284/284 resource-bound     87/17/2
... (4350 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>oh>ow>tap jam=share-all  -> plan_lb=256, p_agg=265, sched=268, resource-bound
flags: K=recommended family knee, b=below that row's family knee (latency-bound or recurring-traffic immature), o=oversubscribed relative to that row's family knee.
Order: `co>oh>ow>tap`.
Jam: share-all: co->tap[input], oh->tap[weight], ow->tap[weight].
Memory: input=resident_shared(base_elem=0,replicas=1,bytes=768); weight=resident_shared(base_elem=192,replicas=1,bytes=432); output=direct.
Capacity: 1200/4096 B; proposed=1200 B; fallback=no.
Scratchpad ports: lb=256 cycles; sched=256 cycles; gap=0 cycles.
Traffic: preload=300 scalar elements, 75 external-L ops, 75 scratchpad-W ops; spad_reads=1404 scalar requests after jam fan-out; avoided_direct=1104 scalar external loads.

4x4 recommendation: co:P1U4 oh:P1U1 ow:P1U2 tap:P1U1 order=co>oh>ow>tap jam=share-input-weight-ow.
8x8 recommendation: co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>oh>ow>tap jam=share-all.
```

## Recommendation

The selected family knee is
`co:P1U4 oh:P1U4 ow:P1U4 tap:P1U1 order=co>oh>ow>tap jam=share-all`.
The resident plan preloads 300 unique elements in 75 external-load and
scratchpad-write operations, reduces recurring resident traffic to 1,404 scalar
requests after jam fan-out, and avoids 1,104 scalar external loads. The fallback
two-load/two-store-port target gives `absolute_cgra_lb=plan_lb=256`,
`p_agg=265`, and `sched=268`; the scratchpad-port floor and schedule both equal
256 cycles for this selected plan.
