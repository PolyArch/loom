# Gather Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the `gather`-specific
setup, helper output, and recommendation.

Kernel: `tests/app/gather/gather.cpp` - gather valid indexed values with
`dst[i] = src[indices[i]]`.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py gather --config 6x6 --top 16
```

## Gather-specific setup

- Fixture: `N=1024`, `src_size=256`, and all generated indices valid, matching
  `main.cpp` and `gather_eval.md`.
- Iterations are independent. Duplicate indices may alias only on read-only
  `src`, while every iteration writes a distinct `dst[i]`.
- `indices[i]` and `dst[i]` are contiguous over `i` and may coalesce within one
  unrolled worker. `src[indices[i]]` remains an indirect scalar access because
  the loaded indices are not an affine vector address sequence.
- The helper follows the DSE specification's `V=4` convention for all element
  types, including this kernel's `uint32_t` arrays.
- Full-trip DSE counts are `A=1026`, `LD_rec=1281`, `LD_eff=1283`, `ST=257`,
  and `CP=4`; `absolute_cgra_lb=107`, load-bound.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): gather  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `i[1024,parallel]`; i is parallel and every concrete fixture lane takes the valid arm. indices[i] and dst[i] are contiguous, so i-unroll coalesces those streams and amortizes the iterator. src[indices[i]] remains an indirect scalar load: the loaded indices are not an affine address sequence the vector interface may coalesce, although the read-only loads are independent and may occupy separate load lanes. The DSE uses the spec-wide V=4 convention despite the uint32_t element type. Full-trip counts are `A=1026`, `LD_rec=1281`, `LD_eff=1283`, `ST=257`, and `CP=4`, giving the only lower bound, `absolute_cgra_lb=107=max(CP 4, compute 29, load 107, store 22)`, with load pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
o        i:P4U256                      4  12  12   1286  1024     1   107     107     109 resource-bound   27/100/21
o        i:P2U512                      2  12  12   1284  1024     1   107     107     108 resource-bound   27/100/21
o        i:P1U1024                     1  12  12   1283  1024     1   107     107     108 resource-bound   27/100/21
o        i:P4U64                       4  12  12    326   256     4    27     108     116 resource-bound   30/100/22
o        i:P2U128                      2  12  12    324   256     4    27     108     112 resource-bound   30/100/22
o        i:P1U256                      1  12  12    323   256     4    27     108     112 resource-bound   30/100/22
o        i:P8U64                       8  12  12    650   512     2    54     108     112 resource-bound   28/100/22
o        i:P4U128                      4  12  12    646   512     2    54     108     110 resource-bound   28/100/20
o        i:P2U256                      2  12  12    644   512     2    54     108     110 resource-bound   28/100/20
o        i:P1U512                      1  12  12    643   512     2    54     108     110 resource-bound   28/100/20
o        i:P16U64                     16  12  12   1298  1024     1   108     108     110 resource-bound   28/100/21
o        i:P8U128                      8  12  12   1290  1024     1   108     108     109 resource-bound   27/100/20
o        i:P16U32                     16  12  12    658   512     2    55     110     112 resource-bound   29/100/22
o        i:P32U32                     32  12  12   1314  1024     1   110     110     111 resource-bound   28/100/22
o        i:P4U16                       4  12  12     86    64    16     7     112     144 resource-bound   29/100/29
o        i:P2U32                       2  12  12     84    64    16     7     112     144 resource-bound   29/100/29
K        i:P1U32                       1  12   9     43    32    32     4     128     192 resource-bound   25/100/25
... (49 more groups omitted; use --top 0 for the full sweep)

RECOMMENDED: i:P1U32  -> exposure=32, pragma_agg=128 (1.20x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 32 on level 'i' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P32U1              96     98    64     256 2.00x slower (parallel: extra iterators + strided, no coalesce)
  P16U2              64     66    32     192 1.50x slower (parallel: extra iterators + strided, no coalesce)
  P8U4              48     50    16     128 best
  P4U8              44     46    12     128 best
  P2U16             42     44    10     128 best
  P1U32             41     43     9     128 best
```

## Recommendation

Use **`i:P1U32`**. It is the lowest-worker representative among the
best-coalesced exposure-32 candidates, where the load term first reaches the
4-cycle dependency depth. Its `p_agg=128` is `1.20x` the 107-cycle aggregate
floor. More exposure only reduces wave-serialization rounding toward that floor;
it does not improve the modeled steady-state rate once the indirect source-load
stream has saturated the load lanes.
