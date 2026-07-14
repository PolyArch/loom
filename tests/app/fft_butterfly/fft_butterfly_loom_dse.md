# FFT Butterfly Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
`fft_butterfly`-specific setup, helper output, and recommendation.

Kernel: `tests/app/fft_butterfly/fft_butterfly.cpp` - copy a 16-point complex
input, then execute four in-place radix-2 butterfly stages.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py fft_butterfly --config 6x6 --top 16
```

## FFT-butterfly-specific setup

- Fixture: `N=16` and four stages, matching `main.cpp` and
  `fft_butterfly_eval.md`.
- The sweep applies to the annotated copy loop. Its real/imaginary input and
  output streams are contiguous over `copy_i`, so unroll coalesces all four
  streams and amortizes the copy iterator.
- The helper follows the DSE specification's `V=4` convention for all element
  types, including this kernel's `float` arrays.
- Copy waves complete before stage 1. Stages `s=1..4` then execute once in true
  ordered regions because each stage reads in-place values written by the
  previous stage; copy exposure never duplicates stage work.
- Because the builder emits every copy wave and all four fixed stages, `cagg`
  is already phase-composed for this kernel and equals `p_agg`; `wav` reports
  only the number of copy waves.
- Within a stage, `k` blocks are independent, but `j` is sequential. It carries
  both its iterator and the generated twiddle recurrence `w <- w*wm`, preserving
  the validated stage critical paths `8`, `11`, `17`, and `33`.
- Full-trip DSE counts are `A=701`, `LD_rec=252`, `LD_eff=253`, `ST=302`, and
  `CP=71`; the ordered-region `absolute_cgra_lb` is 71 cycles.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): fft_butterfly  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `copy_i[16,parallel]`; the annotated copy loop is parallel and its two input/output streams are contiguous, so copy_i-unroll coalesces them and amortizes copy control. Copy waves are ordered before four fixed once-only FFT stage regions. Those stages are barrier-ordered by in-place array RAW hazards; within each stage k blocks are independent, but j remains sequential because both its iterator and the generated twiddle w<-w*wm are carried recurrences. Thus a copy candidate's waves do not repeat the stage work. Full-trip counts are `A=701`, `LD_rec=252`, `LD_eff=253`, `ST=302`, and `CP=71`, giving the only lower bound, `absolute_cgra_lb=71` from the sum of 5 ordered-region aggregates (region-summed CP 71, compute ceilings 22, load ceilings 23, and store ceilings 28); `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
         copy_i:P4U4                   4  12  12    256    16     1    71      71      75 resource-bound    31/32/41
         copy_i:P2U8                   2  12  12    254    16     1    71      71      75 resource-bound    31/32/39
K        copy_i:P1U16                  1  12  12    253    16     1    71      71      75 resource-bound    31/32/39
         copy_i:P8U2                   8  12  12    268    16     1    72      72      76 resource-bound    31/33/42
         copy_i:P4U2                   4  12  12    268     8     2    73      73      78 resource-bound    32/33/41
         copy_i:P2U4                   2  12  12    256     8     2    73      73      78 resource-bound    32/33/40
         copy_i:P1U8                   1  12  12    254     8     2    73      73      78 resource-bound    32/33/40
         copy_i:P8U1                   8  12  12    292     8     2    74      74      80 resource-bound    31/35/43
         copy_i:P16U1                 16  12  12    292    16     1    74      74      78 resource-bound    30/35/43
         copy_i:P4U1                   4  12  12    292     4     4    77      77      84 resource-bound    32/34/42
         copy_i:P2U2                   2  12  12    268     4     4    77      77      84 resource-bound    32/34/40
         copy_i:P1U4                   1  12  12    256     4     4    77      77      84 resource-bound    32/34/40
         copy_i:P2U1                   2  12  12    292     2     8    85      85      96 resource-bound    34/35/41
         copy_i:P1U2                   1  12  12    268     2     8    85      85      96 resource-bound    34/35/41
         copy_i:P1U1                   1  12  12    292     1    16   101     101     120 resource-bound    37/38/43

RECOMMENDED: copy_i:P1U16  -> exposure=16, pragma_agg=71 (1.00x the floor), phase-composed
flags: K=recommended (smallest tunable-phase exposure that reaches the best phase-composed estimate).

P-vs-U at fixed product 16 on level 'copy_i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P16U1             291    292   341      74 1.04x slower (parallel: extra iterators + strided, no coalesce)
  P8U2             267    268   317      72 1.01x slower (parallel: extra iterators + strided, no coalesce)
  P4U4             255    256   305      71 best
  P2U8             253    254   303      71 best
  P1U16            252    253   302      71 best
```

## Recommendation

Use **`copy_i:P1U16`**. A single fully unrolled copy wave reaches
`p_agg=absolute_cgra_lb=71`, compared with 73 cycles for the source-like
`P1U8` split. `P4U4` and `P2U8` tie the same aggregate estimate, but `P1U16`
uses the fewest workers and retains the lowest memory/control counts. The
finite-resource `sched=75` remains an estimate above the floor; the four-cycle
gap comes from time-local pressure inside the fixed ordered regions, not from
additional stage parallelism.
