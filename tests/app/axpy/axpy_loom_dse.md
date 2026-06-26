# AXPY Loom-Pragma DSE

Kernel: `tests/app/axpy/axpy.cpp`

Loop: `compute_loop`

Current source pragma:

```cpp
compute_loop:
LOOM_PARALLEL(4, contiguous)
LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
for (uint32_t i = 0; i < N; i++) {
    output_y[i] = alpha * input_x[i] + input_y[i];
}
```

This file evaluates finite Loom pragma choices separately from `axpy_eval.md`.
The normal eval reports the fully-unrolled ASAP and CGRA full-DAG models; this
file asks which `LOOM_PARALLEL(P)` / `LOOM_UNROLL(U)` pair to choose, and how far
a measured DFG simulator run sits from the model. It implements the
"Optional Loom-Pragma Design-Space Estimate" section of
[`docs/spec-kernel-performance.md`](../../../docs/spec-kernel-performance.md).

## Setup

`compute_loop` is dependency-parallel: each iteration reads `input_x[i]`,
`input_y[i]`, and `alpha`, then writes a distinct `output_y[i]`. There is no
carried scalar state, in-place memory dependence, or reduction.

- Resource config: `6x6` (`P = 36`, `L = 12`, `S = 12`)
- Trip count: `256` from the typical `LOOM_TRIPCOUNT_FULL` value
- Schedule: `contiguous`
- Candidate factors: powers of two up to `8` for both parallelism and unroll

For this one parallel loop, with `E = exposed_iters = min(trip_count, P * U)`,
`full_waves = trip_count // E`, and `tail = trip_count % E`:

```text
chunk(E)                  = E-iteration exposed chunk DAG
pragma_exposure_aggregate = full_waves * aggregate(chunk(E)) + (aggregate(chunk(tail)) if tail else 0)
schedule_estimate         = full_waves * scheduled(chunk(E)) + (scheduled(chunk(tail)) if tail else 0)
```

One iteration costs `P = 4` (mul, add, `i++`, compare), `L = 3` (`input_x`,
`input_y`, `i`), `S = 2` (`output_y`, `i` writeback); `alpha`/`N` are hoisted
loads charged once per wave, and `CP = 4`.

## The bracket (only one of these is a lower bound)

```text
absolute_cgra_lb  <=  pragma_exposure_aggregate  <=  schedule_estimate
```

- `absolute_cgra_lb = 65` is the full-trip aggregate CGRA lower bound — the same
  Metric-1 lower bound applied to the whole loop, independent of `P`/`U`. It is
  load-bound: `ceil((3*256 + 2) / 12) = ceil(770/12) = 65`. **This is the only
  lower bound here.**
- `pragma_exposure_aggregate` and `schedule_estimate` both assume waves run
  **sequentially** (no overlap). Real Loom dataflow pipelines waves, so both sit
  **above** the hardware floor and **must not** be called lower bounds. Their
  values therefore fall monotonically as exposure grows — which is exactly why
  minimizing them is not how we pick `P`/`U` (see below).

This is a design-space estimate, not a bound, RTL timing model, mapper model, or
memory-bank-conflict model.

## Results

Command:

```bash
python3 tests/scripts/loom_dse.py axpy --config 6x6 --trip-count 256 --max-parallel 8 --max-unroll 8
```

Rows are grouped when multiple `P,U` pairs expose the same chunk. `wave_pen` is
`pragma_exposure_aggregate / absolute_cgra_lb` (the wave-serialization penalty of
that exposure, **not** a hardware cost — it vanishes as `E -> trip_count`).

| candidates | exposed | waves | pragma_agg | sched_est | wave_pen | class | backlog P/L/S |
|------------|--------:|------:|-----------:|----------:|---------:|-------|---------------|
| `P=8,U=8` *(oversub)* | 64 | 4 | 68 | 76 | 1.05 | resource-bound | 28/**182**/0 |
| `P=4,U=8`, `P=8,U=4` *(oversub)* | 32 | 8 | 72 | 88 | 1.11 | resource-bound | 0/86/0 |
| **`P=2,U=8`, `P=4,U=4`, `P=8,U=2` ◄ knee** | 16 | 16 | 80 | 112 | 1.23 | resource-bound | 0/38/0 |
| `P=1,U=8`, `P=2,U=4`, `P=4,U=2`, `P=8,U=1` | 8 | 32 | 128 | 160 | 1.97 | latency-bound | 0/14/2 |
| `P=1,U=4`, `P=2,U=2`, `P=4,U=1`* | 4 | 64 | 256 | 256 | 3.94 | latency-bound | 0/2/0 |
| `P=1,U=2`, `P=2,U=1` | 2 | 128 | 512 | 512 | 7.88 | latency-bound | 0/0/0 |
| `P=1,U=1` | 1 | 256 | 1024 | 1024 | 15.75 | latency-bound | 0/0/0 |

`*` marks the pragma currently written in `axpy.cpp`. `◄ knee` marks the
recommended exposure.

## How the recommendation is made

We do **not** pick `P`/`U` by minimizing `schedule_estimate` (that always selects
the maximum `P * U`, because the wave-summed estimate falls monotonically with
exposure — an artifact of the no-overlap assumption). We also do **not** require
zero scheduler backlog (that always selects the *smallest* exposure and the
*worst* throughput — the only zero-backlog rows here are `exposed ∈ {1, 2}` at
512–1024 cycles).

Instead we find the **saturation knee** `E_sat`: the smallest exposure at which
the binding resource class is fully used every cycle within a wave rather than
idling under the critical-path latency. The binding class is loads
(`3 loads/iter`, `L = 12`), and `CP = 4`, so

```text
E_sat = smallest E with ceil(E * 3 / 12) >= 4   ->   E_sat = 13
```

Below `E_sat` the load lanes idle part of each wave (`latency-bound`); at and
above `E_sat` each wave is `resource-bound`. The smallest enumerated exposure
`>= E_sat` is **16**, so the recommended pragmas are `P=2,U=8`, `P=4,U=4`, or
`P=8,U=2`.

- The current source pragma `P=4,U=1` (`exposed = 4`) is **below** the knee:
  latency-bound, `64` waves, `pragma_exposure_aggregate = 256` (3.94× the floor).
- The knee `exposed = 16` reaches `pragma_exposure_aggregate = 80` (1.23× the
  floor) with peak load backlog `38`.
- The largest candidate `P=8,U=8` (`exposed = 64`) only improves the aggregate to
  `68` (1.05×) but pays peak load backlog `182`. Past the knee, extra exposure is
  **oversubscription**: the steady-state throughput floor is unchanged, so the
  shrinking aggregate is just per-wave ceiling rounding and invariant-reload
  amortization, bought with linearly growing transient backlog and hardware area.

Backlog is reported as a diagnostic, not a constraint: it is a transient artifact
of releasing a fully-unrolled chunk's loads at cycle 1, not a steady-state
hardware property.

## Comparing against measured DFG simulator cycles

DFG simulator **execution** cycles are imported measured data (from a separate
sheet); this model does not run the simulator. Provide them per candidate via a
CSV, or a single value for the current source pragma:

```bash
python3 tests/scripts/loom_dse.py axpy --config 6x6 --trip-count 256 \
    --max-parallel 8 --max-unroll 8 --sim-metrics-csv temp/axpy_sim_metrics.csv

python3 tests/scripts/loom_dse.py axpy --sim-exec-cycles 1234
```

CSV schema (header required; `candidate_id`/`notes` are for traceability only):

```text
kernel,candidate_id,parallel,unroll,schedule,trip_count,sim_exec_cycles,notes
axpy,axpy-P4-U1,4,1,contiguous,256,1234,current source pragma
```

When measured data is present, the report adds `sim_exec`, `sim/abs`,
`sim/pragma`, and `sim/sched` columns (`n/a` for candidates with no measured
value). Read them as:

- `sim / absolute_cgra_lb` — total distance from the true resource floor;
- `sim / pragma_exposure_aggregate` — distance from the chosen exposure's
  wave-serialized aggregate;
- `sim / schedule_estimate` — overhead **not** explained by the finite-resource
  schedule model (DFG lowering, mapping, handshake backpressure, memory latency,
  routing).

## Extension Notes

For nested loops, keep pragma placement explicit, for example outer `P=4` with
inner `U=2` versus outer `U=2` with inner `P=4`. Equal exposure can still differ
by carried dependencies, reduction trees, memory grouping, and tail behavior.

For reductions, record whether `LOOM_REDUCE` is present and model the per-worker
partials plus the final merge tree; the binding class and `E_sat` reasoning
carries over, with the merge tree adding to the per-wave critical path.
