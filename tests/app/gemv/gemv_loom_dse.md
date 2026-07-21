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

The fixture uses `M=N=64` and source order `(i,j)`. Unrolling rows derives the
declared `i->j[x]` jam: rows inside one unrolled worker advance through `j`
together and share `x[j]`, while every row keeps its own reduction tree. `x` is
loaded once into a 256-byte `resident_shared` held allocation; `A`, `input_y`,
and `output_y` remain direct. The source `LOOM_MEMORY_BANK` annotation still does
not define external banking; the four-bank diagnostics below apply only to the
named analytical scratchpad profile.

## Results

```text
# Loom pragma DSE (analytic_prefilter): gemv  (6x6)

Evidence: `analytic_prefilter`; target `shared-spad-4k-v4`; preload mode `serial`; one 4096-byte scratchpad shared across this kernel; 4 cyclic single-ported banks; 1-cycle access; fixed V=4.
Search: complete legal power-of-two factors through each trip count.
Candidates: 28 legal, 28 deduplicated groups; `absolute_cgra_lb=248` is the profile-global floor.

flags    candidate                                                                                                plan_lb   p_agg   sched tiles  cap_B   bank lb/s class           util P/L/S
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
K        i:P1U8 j:P1U1 order=i>j tile=untiled                                                                         248     248     296     1    256     144/144 resource-bound    100/45/3
o        i:P2U8 j:P1U1 order=i>j tile=untiled                                                                         248     248     268     1    256     144/144 resource-bound    100/43/2
o        i:P1U16 j:P1U1 order=i>j tile=untiled                                                                        248     248     268     1    256       80/80 resource-bound    100/41/2
o        i:P4U8 j:P1U1 order=i>j tile=untiled                                                                         248     248     254     1    256     144/144 resource-bound    100/42/1
o        i:P2U16 j:P1U1 order=i>j tile=untiled                                                                        248     248     254     1    256       80/80 resource-bound    100/41/1
o        i:P1U32 j:P1U1 order=i>j tile=untiled                                                                        248     248     254     1    256       48/48 resource-bound    100/39/1
o        i:P8U8 j:P1U1 order=i>j tile=untiled                                                                         248     248     251     1    256     144/144 resource-bound    100/42/1
o        i:P4U16 j:P1U1 order=i>j tile=untiled                                                                        248     248     251     1    256       80/80 resource-bound    100/40/1
o        i:P2U32 j:P1U1 order=i>j tile=untiled                                                                        248     248     251     1    256       48/48 resource-bound    100/39/1
o        i:P1U64 j:P1U1 order=i>j tile=untiled                                                                        248     248     251     1    256       32/32 resource-bound    100/38/1
b        i:P1U4 j:P1U1 order=i>j tile=untiled                                                                         272     272     368     1    256     272/272 resource-bound     94/44/6
         i:P2U4 j:P1U1 order=i>j tile=untiled                                                                         272     272     296     1    256     272/272 resource-bound     91/44/3
o        i:P4U4 j:P1U1 order=i>j tile=untiled                                                                         272     272     272     1    256     272/272 resource-bound     91/44/2
o        i:P8U4 j:P1U1 order=i>j tile=untiled                                                                         272     272     272     1    256     272/272 resource-bound     91/43/2
o        i:P16U4 j:P1U1 order=i>j tile=untiled                                                                        272     272     272     1    256     272/272 resource-bound     91/43/1
b        i:P1U2 j:P1U1 order=i>j tile=untiled                                                                         528     528     528     1    256     528/528 resource-bound     50/31/6
b        i:P2U2 j:P1U1 order=i>j tile=untiled                                                                         528     528     528     1    256     528/528 resource-bound     47/28/3
         i:P4U2 j:P1U1 order=i>j tile=untiled                                                                         528     528     528     1    256     528/528 resource-bound     47/27/2
o        i:P8U2 j:P1U1 order=i>j tile=untiled                                                                         528     528     528     1    256     528/528 resource-bound     46/27/2
o        i:P16U2 j:P1U1 order=i>j tile=untiled                                                                        528     528     528     1    256     528/528 resource-bound     46/26/1
o        i:P32U2 j:P1U1 order=i>j tile=untiled                                                                        528     528     528     1    256     528/528 resource-bound     46/26/1
b        i:P1U1 j:P1U1 order=i>j tile=untiled                                                                        1040    1040    1040     1    256   1040/1040 resource-bound     25/19/6
b        i:P2U1 j:P1U1 order=i>j tile=untiled                                                                        1040    1040    1040     1    256   1040/1040 resource-bound     25/19/3
b        i:P4U1 j:P1U1 order=i>j tile=untiled                                                                        1040    1040    1040     1    256   1040/1040 resource-bound     23/19/2
... (4 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: i:P1U8 j:P1U1 order=i>j tile=untiled  -> plan_lb=248, p_agg=248, sched=296, resource-bound
flags: K=recommended, b=below knee (latency-bound or recurring-traffic immature), o=oversubscribed.
Order: `i>j`.
Jam: i->j[x].
Memory: x=resident_shared(base_elem=0,replicas=1,lifetime=held/kernel,capacity_x=1); A=direct; input_y=direct; output_y=direct.
Tile: untiled; tails=none; num_tiles=1.
Capacity: 256/4096 B; held_region=256 B; refill_frame=0 B; frame_bases_B=none.
Banks: bank_lb=144 cycles; bank_sched=144 cycles; gap=0 cycles.
Traffic: preload=64 scalar elements, 16 external-L ops, 16 scratchpad-S ops; spad_reads=512 scalar requests after jam fan-out; avoided_direct=448 scalar external loads.
Direct-memory audit reference (excluded from resident-profile legality, ranking, and floor): plan_lb=232, p_agg=232, sched=280.

Ideal-DMA sensitivity (same config): i:P1U8 j:P1U1 order=i>j tile=untiled -> absolute_cgra_lb=248, plan_lb=248, p_agg=248, sched=296.
Assumption: inactive ping-pong fill does not contend with current-tile scratchpad reads.

4x4 recommendation: i:P1U2 j:P1U1 order=i>j tile=untiled.
8x8 recommendation: i:P1U8 j:P1U1 order=i>j tile=untiled.
```

## Recommendation

The canonical serial knee is `i:P1U8 j:P1U1`. It is the first row whose
dominant recurring demand is mature: smaller unroll factors are already
bank-bound, but they still repeat more `x` reads and are therefore below the
knee. The selected resident plan pays one 64-element preload, eliminates 448
scalar external `x` loads, and reaches its 248-cycle profile floor with a
296-cycle deterministic schedule estimate. The direct audit reference remains
outside the resident-profile ranking; its lower cycle estimate shows that this
study's value is reduced external traffic, not an assumed scratchpad speedup.
