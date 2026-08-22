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

The `main.cpp` smoke-test fixture uses `M=32`, `N=48` and source order `(i,j)`.
Every row-unrolled split is evaluated with explicit `jam=none`; when `U_i>1`,
it may also select the complete `jam=i-j-share-x` plan. That plan advances the
unrolled rows through `j` together and shares `x[j]`, while every row keeps its
own reduction tree; ordinary unrolling receives no such credit. The 48-element
`x` working set is a 192-byte `resident_shared` allocation at the default
capacity and becomes `direct-fallback` below 192 bytes. `A`, `input_y`, and
`output_y` remain direct.

## Results

```text
# Loom pragma DSE (analytic_prefilter): gemv  (6x6)

Evidence: `analytic_prefilter`; target `shared-spad-4k-r2w2-v4`; one 4096-byte scratchpad shared across this kernel; R=2, W=2; 1-cycle non-pipelined access; fixed V=4.
Search: complete legal power-of-two factors through each trip count.
Candidates: 36 legal, 36 deduplicated groups; `absolute_cgra_lb=94` is the profile-global floor.

flags    candidate                                                                                                        plan_lb   p_agg   sched  cap_B   spad lb/s class           util P/L/S
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
K        i:P1U4 j:P1U1 order=i>j jam=i-j-share-x                                                                               94      94     150    192       54/54 resource-bound    100/55/9
o        i:P2U4 j:P1U1 order=i>j jam=i-j-share-x                                                                               94      94     118    192       54/54 resource-bound    100/50/5
o        i:P1U8 j:P1U1 order=i>j jam=i-j-share-x                                                                               94      94     118    192       30/30 resource-bound    100/45/5
o        i:P4U4 j:P1U1 order=i>j jam=i-j-share-x                                                                               94      94     102    192       54/54 resource-bound    100/48/2
o        i:P2U8 j:P1U1 order=i>j jam=i-j-share-x                                                                               94      94     102    192       30/30 resource-bound    100/43/2
o        i:P1U16 j:P1U1 order=i>j jam=i-j-share-x                                                                              94      94     102    192       18/18 resource-bound    100/41/2
o        i:P8U4 j:P1U1 order=i>j jam=i-j-share-x                                                                               94      94      96    192       54/54 resource-bound    100/48/2
o        i:P4U8 j:P1U1 order=i>j jam=i-j-share-x                                                                               94      94      96    192       30/30 resource-bound    100/42/1
o        i:P2U16 j:P1U1 order=i>j jam=i-j-share-x                                                                              94      94      96    192       18/18 resource-bound    100/40/1
o        i:P1U32 j:P1U1 order=i>j jam=i-j-share-x                                                                              94      94      96    192       12/12 resource-bound    100/39/1
         i:P2U2 j:P1U1 order=i>j jam=i-j-share-x                                                                              102     102     150    192     102/102 resource-bound     92/58/8
o        i:P4U2 j:P1U1 order=i>j jam=i-j-share-x                                                                              102     102     118    192     102/102 resource-bound     92/54/4
o        i:P8U2 j:P1U1 order=i>j jam=i-j-share-x                                                                              102     102     102    192     102/102 resource-bound     92/54/4
o        i:P16U2 j:P1U1 order=i>j jam=i-j-share-x                                                                             102     102     102    192     102/102 resource-bound     92/53/3
b        i:P1U2 j:P1U1 order=i>j jam=i-j-share-x                                                                              102     182     230    192     102/102 latency-bound      55/36/9
         i:P2U1 j:P1U1 order=i>j jam=none                                                                                     198     198     230    192     198/198 resource-bound     50/42/8
         i:P1U2 j:P1U1 order=i>j jam=none                                                                                     198     198     230    192     198/198 resource-bound     50/42/8
o        i:P4U1 j:P1U1 order=i>j jam=none                                                                                     198     198     198    192     198/198 resource-bound     50/38/4
o        i:P2U2 j:P1U1 order=i>j jam=none                                                                                     198     198     198    192     198/198 resource-bound     46/38/4
o        i:P1U4 j:P1U1 order=i>j jam=none                                                                                     198     198     198    192     198/198 resource-bound     46/38/4
o        i:P8U1 j:P1U1 order=i>j jam=none                                                                                     198     198     198    192     198/198 resource-bound     48/38/4
o        i:P4U2 j:P1U1 order=i>j jam=none                                                                                     198     198     198    192     198/198 resource-bound     46/35/2
o        i:P2U4 j:P1U1 order=i>j jam=none                                                                                     198     198     198    192     198/198 resource-bound     46/35/2
o        i:P1U8 j:P1U1 order=i>j jam=none                                                                                     198     198     198    192     198/198 resource-bound     46/35/2
... (12 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: i:P1U4 j:P1U1 order=i>j jam=i-j-share-x  -> plan_lb=94, p_agg=94, sched=150, resource-bound
flags: K=recommended family knee, b=below that row's family knee (latency-bound or recurring-traffic immature), o=oversubscribed relative to that row's family knee.
Order: `i>j`.
Jam: i-j-share-x: i->j[x].
Memory: x=resident_shared(base_elem=0,replicas=1,bytes=192); A=direct; input_y=direct; output_y=direct.
Capacity: 192/4096 B; proposed=192 B; fallback=no.
Scratchpad ports: lb=54 cycles; sched=54 cycles; gap=0 cycles.
Traffic: preload=48 scalar elements, 12 external-L ops, 12 scratchpad-W ops; spad_reads=384 scalar requests after jam fan-out; avoided_direct=336 scalar external loads.

4x4 recommendation: i:P1U2 j:P1U1 order=i>j jam=i-j-share-x.
8x8 recommendation: i:P1U8 j:P1U1 order=i>j jam=i-j-share-x.
```

## Recommendation

The selected family knee is
`i:P1U4 j:P1U1 order=i>j jam=i-j-share-x`. It pays one 48-element preload,
reduces the modeled `x` demand to 384 scalar requests after jam fan-out, and
eliminates 336 scalar external loads. On the fallback two-load/two-store-port
target it reaches `plan_lb=p_agg=94`, `sched=150`, and 54 cycles of
scratchpad-port pressure. The explicit `jam=none` family remains in the search;
it is not treated as a future state of the selected jammed family.
