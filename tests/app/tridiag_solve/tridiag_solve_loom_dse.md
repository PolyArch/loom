# tridiag_solve Loom-Pragma DSE (banking-aware) — the counterexample

Kernel: `tests/app/tridiag_solve/tridiag_solve.cpp` (forward elimination sweep)

```cpp
for (uint32_t i = 1; i < N; i++) {
    float m = input_b[i] - input_a[i] * c_prime[i - 1];   // needs c_prime[i-1]
    c_prime[i] = input_c[i] / m;
    d_prime[i] = (input_d[i] - input_a[i] * d_prime[i - 1]) / m;  // needs d_prime[i-1]
}
```

This kernel **does not demonstrate the P-vs-U distinction**, and it is included
to show *why*. It is the complement of [`vecsum`](../vecsum/vecsum_loom_dse.md):

- `vecsum` carries `sum` but the carry is **associative**, so `LOOM_REDUCE(+)`
  legalizes parallel workers (the escape hatch for a loop-carried dependence).
- `tridiag_solve` carries `c_prime`/`d_prime` through a **non-associative
  division chain**. There is no reduction to exploit and no independent
  iterations. `LOOM_PARALLEL` is illegal and `LOOM_REDUCE` does not apply.

The source even *writes* `LOOM_PARALLEL()`, but the carry makes it a no-op: the
model forces `P_tot = 1`. The spec names this family explicitly (non-associative
recurrences: `tridiag_solve`, `trsv_lower/upper`, `gauss_seidel_step`,
`kmp_table`).

Regenerate: `python3 tests/scripts/loom_dse.py tridiag_solve --config 6x6`

## Why there is no distinction

With `P_tot` forced to `1`:

- Only **one** worker exists → **one** load/store port → `active_L = active_S =
  1`. Parallelism cannot add ports (there are no independent streams to bank).
- `LOOM_UNROLL(U)` is the only legal knob, and it adds no bank/port — its `U`
  accesses share the single worker's port. Since this model deliberately does not
  credit control-overhead amortization, **unroll changes nothing** either.
- The floor is the **serial critical path**: `CP = 194 = absolute_cgra_lb`
  (the `≈ 3·(N−1)` division-chain depth), which no pragma can shorten.

## Results (the entire sweep is one row)

```text
flags  split       Ptot  aL  aS   exp  wav  cagg  p_agg  sched  class           util P/L/S
------ ----------- ----- --- --- ---- ---- ----- ------ ------ --------------- ------------
K      i:P1U1 x4      1   1   1   64    1   322    322    325  resource-bound   5/100/60
```

All four `U` factorings collapse to the same row — there is nothing to choose.
`P-vs-U contrast: no parallelizable level.`

- Full-trip counts: `A=512 LD=322 ST=192 CP=194` → `compute=15 load=27 store=16`.
- `absolute_cgra_lb = 194` is **CP-bound** (the serial recurrence), not
  resource-bound. Even with all 12 lanes the kernel cannot beat `194`.
- At the only achievable point (`active_L = 1`) loads serialize to `322` — worse
  than the `194` floor and unfixable, because you can neither parallelize the
  loads nor the recurrence.

## Takeaway

Three conditions make a kernel unable to show the P-vs-U distinction, all present
here in the extreme: **(1)** a non-associative carried dependence forbids
parallel workers (`P_tot = 1`), so there is no `P` to vary; **(2)** with one
worker there is one memory port, so banking has nothing to scale; **(3)** the
aggregate is set by the critical path, which is on the arithmetic/dependence axis
that this load/store-focused model treats as a global pool. A kernel that is
merely **compute-bound** (arithmetic dominates, but still parallel) hits only
condition (3): `P` and `U` become interchangeable because only loads/stores are
banking-aware — `batchnorm` sits near that boundary.
