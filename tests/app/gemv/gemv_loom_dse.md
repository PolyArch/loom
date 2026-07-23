# gemv Loom-Pragma DSE (extended analytical profile)

Shared definitions, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This note records only the GEMV setup,
reproducible helper output, and recommendation.

Kernel: `tests/app/gemv/gemv.cpp` computes
`output_y[i] = alpha * sum_j(A[i,j] * x[j]) + beta * input_y[i]`.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py gemv --config 6x6 --brief-config 4x4 --brief-config 8x8 --top 24
```

## GEMV-specific setup

The fixture uses `M=N=64` and source order `(i,j)`. Every row-unrolled split
is evaluated with explicit `jam=none`; when `U_i>1`, it may also select the
complete `jam=i-j-share-x` plan. That plan advances the unrolled rows through
`j` together and shares `x[j]`, while every row keeps its own reduction tree;
ordinary unrolling receives no such credit. The 64-element `x` working set is a
256-byte `resident_shared` allocation at the default capacity and becomes
`direct-fallback` below 256 bytes. `A`, `input_y`, and `output_y` remain
direct.

## Results

```text
# Loom pragma DSE (analytic_prefilter): gemv  (6x6)

Evidence: `analytic_prefilter`; target `shared-spad-4k-r1w1-v4`; one 4096-byte scratchpad shared across this kernel; R=1, W=1; 1-cycle non-pipelined access; fixed V=4.
Search: complete legal power-of-two factors through each trip count.
Candidates: 49 legal, 49 deduplicated groups; `absolute_cgra_lb=248` is the profile-global floor.

flags    candidate                                                                                                        plan_lb   p_agg   sched  cap_B   spad lb/s class           util P/L/S
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
K        i:P1U8 j:P1U1 order=i>j jam=i-j-share-x                                                                              248     248     296    256     144/144 resource-bound    100/45/3
o        i:P2U8 j:P1U1 order=i>j jam=i-j-share-x                                                                              248     248     268    256     144/144 resource-bound    100/43/2
o        i:P1U16 j:P1U1 order=i>j jam=i-j-share-x                                                                             248     248     268    256       80/80 resource-bound    100/41/2
o        i:P4U8 j:P1U1 order=i>j jam=i-j-share-x                                                                              248     248     254    256     144/144 resource-bound    100/42/1
o        i:P2U16 j:P1U1 order=i>j jam=i-j-share-x                                                                             248     248     254    256       80/80 resource-bound    100/41/1
o        i:P1U32 j:P1U1 order=i>j jam=i-j-share-x                                                                             248     248     254    256       48/48 resource-bound    100/39/1
o        i:P8U8 j:P1U1 order=i>j jam=i-j-share-x                                                                              248     248     251    256     144/144 resource-bound    100/42/1
o        i:P4U16 j:P1U1 order=i>j jam=i-j-share-x                                                                             248     248     251    256       80/80 resource-bound    100/40/1
o        i:P2U32 j:P1U1 order=i>j jam=i-j-share-x                                                                             248     248     251    256       48/48 resource-bound    100/39/1
o        i:P1U64 j:P1U1 order=i>j jam=i-j-share-x                                                                             248     248     251    256       32/32 resource-bound    100/38/1
b        i:P1U4 j:P1U1 order=i>j jam=i-j-share-x                                                                              272     272     368    256     272/272 resource-bound     94/44/6
         i:P2U4 j:P1U1 order=i>j jam=i-j-share-x                                                                              272     272     296    256     272/272 resource-bound     91/44/3
o        i:P4U4 j:P1U1 order=i>j jam=i-j-share-x                                                                              272     272     272    256     272/272 resource-bound     91/44/2
o        i:P8U4 j:P1U1 order=i>j jam=i-j-share-x                                                                              272     272     272    256     272/272 resource-bound     91/43/2
o        i:P16U4 j:P1U1 order=i>j jam=i-j-share-x                                                                             272     272     272    256     272/272 resource-bound     91/43/1
b        i:P1U2 j:P1U1 order=i>j jam=i-j-share-x                                                                              528     528     528    256     528/528 resource-bound     50/31/6
b        i:P2U2 j:P1U1 order=i>j jam=i-j-share-x                                                                              528     528     528    256     528/528 resource-bound     47/28/3
         i:P4U2 j:P1U1 order=i>j jam=i-j-share-x                                                                              528     528     528    256     528/528 resource-bound     47/27/2
o        i:P8U2 j:P1U1 order=i>j jam=i-j-share-x                                                                              528     528     528    256     528/528 resource-bound     46/27/2
o        i:P16U2 j:P1U1 order=i>j jam=i-j-share-x                                                                             528     528     528    256     528/528 resource-bound     46/26/1
o        i:P32U2 j:P1U1 order=i>j jam=i-j-share-x                                                                             528     528     528    256     528/528 resource-bound     46/26/1
         i:P1U1 j:P1U1 order=i>j jam=none                                                                                    1040    1040    1040    256   1040/1040 resource-bound     25/19/6
o        i:P2U1 j:P1U1 order=i>j jam=none                                                                                    1040    1040    1040    256   1040/1040 resource-bound     25/19/3
o        i:P1U2 j:P1U1 order=i>j jam=none                                                                                    1040    1040    1040    256   1040/1040 resource-bound     25/19/3
... (25 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: i:P1U8 j:P1U1 order=i>j jam=i-j-share-x  -> plan_lb=248, p_agg=248, sched=296, resource-bound
flags: K=recommended family knee, b=below that row's family knee (latency-bound or recurring-traffic immature), o=oversubscribed relative to that row's family knee.
Order: `i>j`.
Jam: i-j-share-x: i->j[x].
Memory: x=resident_shared(base_elem=0,replicas=1,bytes=256); A=direct; input_y=direct; output_y=direct.
Capacity: 256/4096 B; proposed=256 B; fallback=no.
Scratchpad ports: lb=144 cycles; sched=144 cycles; gap=0 cycles.
Traffic: preload=64 scalar elements, 16 external-L ops, 16 scratchpad-W ops; spad_reads=512 scalar requests after jam fan-out; avoided_direct=448 scalar external loads.

4x4 recommendation: i:P1U2 j:P1U1 order=i>j jam=i-j-share-x.
8x8 recommendation: i:P1U8 j:P1U1 order=i>j jam=i-j-share-x.
```

## Recommendation

The selected family knee is
`i:P1U8 j:P1U1 order=i>j jam=i-j-share-x`. It pays one 64-element preload,
reduces the modeled `x` demand to 512 scalar requests after jam fan-out, and
eliminates 448 scalar external loads. On the default one-load/one-store-port
target it reaches `plan_lb=p_agg=248`, `sched=296`, and 144 cycles of
scratchpad-port pressure. The explicit `jam=none` family remains in the search;
it is not treated as a future state of the selected jammed family.
