# Col2im Loom-Pragma DSE (lane-aware + vector coalescing)

Shared DSE rules, table columns, and simulator-comparison caveats live in
[`../DSE_rules.md`](../DSE_rules.md). This file only records the
`col2im`-specific setup, helper output, and recommendation.

Kernel: `tests/app/col2im/col2im.cpp` — scatter column entries into overlapping
image pixels. The source marks channel `c` parallel and `kh` unrolled.

Regenerate:

```bash
python3 tests/scripts/loom_dse.py col2im --config 6x6 --top 16
```

## Col2im-specific setup

- Fixture: `C=3`, `H=W=8`, `KH=KW=3`, `OH=OW=6`, matching `main.cpp`.
- Channels are independent. Within one channel, overlapping `kh/kw`
  contributions target the same output pixels and are consumed as associative
  output-centric reduction buckets; they are not modeled as independent
  scatter stores.
- `kh` is fully consumed, so its displayed P/U labels are equivalent. Channel
  slices are separated by `H*W`, so accesses do not coalesce across `c`.
- The DSE removes the eval's 1,365 source-level induction steps and restores one
  residual `c` iterator per active worker. Thus `c` unroll amortizes control,
  while `c` parallel retains one iterator per worker.
- Full-trip DSE counts are `A=12756`, `LD_rec=1945`, `LD_eff=1952`,
  `ST=1165`, `CP=13`; `absolute_cgra_lb=355`, compute-bound.

## Results

```text
# Loom pragma DSE (lane-aware + vector coalescing): col2im  (6x6)

Search: complete legal power-of-two factors through each trip count.
Loop nest: `c[3,parallel], kh[3,reduction]`; channels are independent. For each exposed channel, overlapping kh/kw contributions are consumed as output-centric associative reduction buckets, so kh is fully consumed and its P/U labels are equivalent. Channel slices are separated by H*W and do not coalesce across c. The eval's per-iteration induction work is removed, then one residual c iterator is charged per active worker, so c-unroll amortizes control while c-parallel retains one iterator per worker. Full-trip counts are `A=12756`, `LD_rec=1945`, `LD_eff=1952`, `ST=1165`, and `CP=13`, giving the only lower bound, `absolute_cgra_lb=355=max(CP 13, compute 355, load 163, store 98)`, with compute pressure binding; `p_agg` and `sched` are wave-serialized estimates.

flags    split                      Ptot  aL  aS LD_eff   exp   wav  cagg   p_agg   sched class           util P/L/S
--------------------------------------------------------------------------------------------------------------------
K        c:P1U1 kh:P1U1                1  12  12    656     3     3   119     357     357 resource-bound   100/46/28
o        c:P2U1 kh:P1U1                2  12  12   1305     6     2   237     474     474 resource-bound   100/46/27
o        c:P1U2 kh:P1U1                1  12  12   1304     6     2   237     474     474 resource-bound   100/46/27

RECOMMENDED: c:P1U1 kh:P1U1  -> exposure=3, pragma_agg=357 (1.01x the floor), resource-bound
flags: K=recommended (saturation knee E_sat), b=bandwidth-starved (latency-bound: resources idle), o=oversubscribed (past the knee, no estimate gain).

P-vs-U at fixed product 2 on level 'c' (other levels at P1U1):
  split        LD_rec LD_eff    ST   p_agg note
  P2U1            1298   1305   778     474 tie (control/coalescing sit below the binding term)
  P1U2            1297   1304   777     474 tie (control/coalescing sit below the binding term)
```

## Recommendation

Use **`c:P1U1 kh:P1U1`** as the smallest representative saturation knee.
It gives `p_agg = sched = 357`, or `1.01x` the 355-cycle aggregate floor.
Exposing two channels per wave raises the wave-serialized estimate to 474
without improving the already compute-bound full-trip result. At fixed channel
exposure two, `P1U2` saves one residual iterator versus `P2U1`, although the
compute ceiling leaves both at 474 cycles. The `(+2 eq)` labels are alternate
`kh` factorizations of the same fully consumed reduction.
