# vecsum Loom-Pragma DSE (banking-aware)

Kernel: `tests/app/vecsum/vecsum.cpp`

```cpp
LOOM_REDUCE(+)
uint32_t sum = init_value;
LOOM_PARALLEL(4)
for (uint32_t i = 0; i < N; i++) {
    sum += A[i];
}
```

`vecsum` is the **reduction** case: the loop carries `sum`, so plain parallel is
illegal — but the carry is **associative** (`+`), so `LOOM_REDUCE(+)` legalizes
parallel workers as independent partial sums merged by a `log`-depth tree. This
is the escape hatch for a loop-carried dependence (contrast
[`tridiag_solve`](../tridiag_solve/tridiag_solve_loom_dse.md), whose carry is
**not** associative and stays serial).

Regenerate: `python3 tests/scripts/loom_dse.py vecsum --config 6x6`

## Banking

`A` is partitioned across the `LOOM_REDUCE` workers (`B_L = P_tot`), so
parallelism scales load bandwidth: `active_L = min(P_tot, L)`. The only output
is the scalar `sum` (one store); the store class is otherwise per-worker iterator
write-backs (also `B_S = P_tot`). `vecsum` is **load-bound** (`N` loads, one
result), so loads drive the choice.

## Setup

- `6x6` (`P=36, L=12, S=12`), `N = 256`, reduction fully consumed in one chunk.
- Full-trip counts: `A=768 LD=514 ST=257 CP=11` → `compute=22 load=43 store=22`.
- `absolute_cgra_lb = 43` (`ceil(514/12)`), reached only at `active_L = 12`
  (`P_tot >= 12`). The `CP = 11` is the reduction-tree depth `ceil(log2 256)`.

## Results

```text
flags  split       Ptot  aL  aS   exp  wav  cagg  p_agg  sched  class           util P/L/S
------ ----------- ----- --- --- ---- ---- ----- ------ ------ --------------- ------------
K      i:P8U1  x4     8   8   8  256    1    65     65     67  resource-bound  34/100/51
b      i:P4U1* x4     4   4   4  256    1   129    129    131  resource-bound  17/100/50
b      i:P2U1  x4     2   2   2  256    1   257    257    259  resource-bound   9/100/50
b      i:P1U1  x4     1   1   1  256    1   514    514    515  resource-bound   4/100/50
```

`x4` = four `P·U` factorings collapse to one row (op counts are product-only;
only `active_L` differs). `*` = current source pragma (`P=4`).

## The P-vs-U distinction

At fixed product `P·U = 32`: `P=8,U=4` → `active_L=8`, `p_agg=65`; `P=4,U=8` →
`active_L=4`, `p_agg=129` (**2.0× slower**). Parallel reduction workers each
stream `A` from their own bank; unroll piles the loads onto one worker's port.
The merge tree contributes the same `log`-depth either way (product-only), so the
distinction is entirely on the load ports.

## Recommendation

**`LOOM_PARALLEL(8)` (reduction), `LOOM_UNROLL(1)`** — the knee at maximum
reachable load bandwidth. The current `P=4` is *bandwidth-starved*: `active_L=4`,
`p_agg=129` (`3.0×` the floor) with the load lane at only `4/12` of the fabric.
Doubling to `P=8` halves it to `65` (`1.5×`). Reaching the `43` floor needs
`active_L=12`, i.e. `P>=12` reduction workers (banking `A` to 12) — unroll cannot
get there.
