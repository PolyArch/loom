# Project Overview
Loom is a full stack framework for Domain-specific Accelerator, from C++ source code to Hardware Backend

# Documentation
- Check `docs` to understand `loom` design specification, there are important concepts like: `dataflow`, `fabric`, `adg`, `cosim`, each one of them has its own specification documents.

# Spec-First Development (MANDATORY)
All code changes MUST use the specification documents (`docs/spec-*.md`) as the authoritative implementation baseline. Spec is code.
- Before implementing or modifying any feature, read the relevant `docs/spec-*.md` first.
- If the current implementation deviates from spec, the code MUST be aligned to spec.
- If after careful analysis you believe the codebase diverges significantly from spec, or that the spec contains fundamental contradictions, you MUST stop and notify the user. Do NOT proceed with an implementation that deviates from spec under any circumstances.
- When in doubt, the spec wins. Never assume the existing code is correct over what the spec says.
- When uncertain or unclear about any design decision, read `docs/spec-*.md` first before proceeding.

# Project Rules
- Header files (like .h/.cuh etc) in `include`; Implementation files (like .c/.cpp/.cu etc) in `lib`.
- When splitting files, split them into multiple `.cpp`/`.h` files (not `.inc` includes).
- Projects under `externals` are external projects that are used in this project via source compilation, avoid modifying them.
- Files in `temp` folder are allowed to bypass English-only and file size rules.

# Useful Hints
- If EDA related tools (like `vcs`, `verilator`, `verdi`, `dc_shell`, `fc_shell`, etc.) cannot be found, please try `module avail` and use `module load ...` to load environments.
- For example, if you want to run verilator, prepend `module load verilator && ...`; for vcs/verdi, prepend `module load synopsys/vcs synopsys/verdi && ...` to command you want to run. Prepend with `module purge && ...` can clean the loaded tools.
- For tests that need verilator and vcs/verdi, you can do `module purge && module load synopsys/vcs synopsys/verdi verilator && ...` to load all of them.
- Use `module load synopsys/vcs synopsys/verdi verilator && make check` can quickly check all tests with correct environments.

# ExecPlans (for Codex only)
When writing complex features or significant refactors, use an ExecPlan (as described in .agent/PLANS.md) from design to implementation.
The ExecPlan can be stored in `temp` folder as a living execution plan, the filename should be timestamp-ed like `ExecPlan-<YYYYMMDD-hhmmss>.md`, timestamp should use `date +"%Y%m%d-%H%M%S"`. 

# SystemVerilog Style Rules
- Every `begin`-`end` block must have a named label (`: label_name`).
- Loop variables must be declared at the top of the enclosing procedural block (`always`, `initial`, `function`), not inline in `for`. Use `iter_var0`, `iter_var1`, ... as loop variable names (numbered by nesting depth).

# End To End Test Pipeline
Use the following sequence as the end to end test pipeline:
- `ninja -C build clean-loom`
- `ninja -C build loom`
- `ninja -C build check-loom`

# Performance Modeling

## Goal
Estimate the lower bound on latency and upper bound on throughput for each
kernel under `tests/app/`, under an ASAP dataflow model with unlimited
hardware: 1-cycle-per-op, unbounded fan-out, infinite functional units,
infinite memory bandwidth, and full unrolling of every dimension with no
carried dependence. The model answers "what is the shortest possible schedule
for this DAG?", not "what would a real machine deliver?"

## Conventions

1. **Op counts include every dynamic operation in the kernel.** Counts
   measure total dynamic work and are independent of scheduling — unrolling
   does not change them. All of the following are counted:
   - **Algorithmic arithmetic** — the kernel's intended math.
   - **Loop-carried state updates** — accumulator merges, scan steps,
      recurrence updates. A reduction over `N` values still costs `N − 1`
      adds even when tree-scheduled.
   - **Memory I/O** — every load and store (see Convention 6 for uniform
      1-cycle costing). Includes algorithmic array loads/stores, kernel-input
      boundary loads, kernel-output boundary stores, and per-access load/
      store for any named scalar (including loop-carried accumulators and
      induction variables).
   - **Induction variables** — each loop iteration charges a load (read
      iterator), an add (increment), a store (write iterator), and a compare
      (bound check). The bound itself is hoisted under the loop-invariant rule below.
   - **Address generation** — charged at the array indexing operator
      `[]` only. Each `[]` access costs one address-add per dimension per
      iteration in incremental-stride form. For non-affine indexing
      (gather/scatter, `a[idx[i]]`), charge the loads and arithmetic the
      source dictates for the inner reference, then 1 address-add for the
      outer `[]`. Adds that produce a named source-level scalar are
      regular `adds`, even when that scalar is used exclusively as an
      array index downstream (e.g. `idx = c·HW + h·W + w; a[idx];` — the
      two adds for `idx` are regular `adds`, and `a[idx]` charges 1
      address-add per access). Address-arithmetic adds are tracked as a
      separate `address_adds` category and are NOT lumped into the regular
      `adds` total. (Induction-variable increments, `i ← i+1`, remain
      regular `adds`.)
   - **Loop-invariant hoisting** — any value whose inputs are loop-
     invariant is computed once and broadcast to all consumers via free
     fan-out. Charged 1× to op counts (compute, load, and store all
     count once, not per iter). Applies recursively: a value is loop-
     invariant if every input is a kernel input, a constant, or itself
     loop-invariant. Anything depending on the iterator or carried
     state is per-iter and charged normally.
   - **Dead computations** — ops whose results are not consumed by any output. They still count as work.
   - **Control flow (no predication)** — branches are not predicated.
      The condition's compare executes first, and **no instructions
      inside an if/else body may execute before that compare retires**.
      Only the ops on the taken branch are counted for that iteration.
      Both the compare and the selected branch's ops lie on the critical
      path: a body op fires no earlier than the cycle after its gating
      compare completes, and nested `if`s serialize cumulatively (each
      nested compare adds another compare→body gap to the chain). For
      early-exit / data-dependent-termination loops, count only the
      iters that actually execute; the termination compare sits on the
      critical path.

      This rule applies uniformly to every source `if`, `if/else`,
      ternary `?:`, and conditional store — **including** patterns the
      Loom compiler internally lowers to `handshake.mux`,
      `arith.select`, or gated control tokens (see `convertIf` and
      `convertStore` in `lib/loom/Conversion/SCFToHandshakeConvert.cpp`).
      The IR lowering happens in real hardware; the performance model is
      intentionally conservative and does not take credit for it.
      Concrete consequences:
      - `if (c) x = a; else x = b;` and `x = c ? a : b;` — only the
      taken arm is counted; no mux bitop is charged; the arm cannot
      begin before `c` resolves.
      - `if (c) array[i] = v;` — the value compute, address-gen, and
      store all wait for `c` to resolve; no AND-enable bitop is
      charged.
      - Nested `if (a) { if (b) { ... } }` — the inner body's ops wait
      for both `a` AND `b` to resolve in sequence (two compare→body
      gaps on the critical path).

2. **`total_cycles` is the critical-path depth of the kernel's dataflow DAG.**
   Longest dependence chain from input to output, 1 cycle per op. Any two ops
   with no path between them schedule in the same cycle. Ops counted under
   Convention 1 that lie off the critical path do not extend `total_cycles`.

3. **Loops are classified by carried dependence:**
   - **Parallel dim** — no value produced in iter `i` is consumed in iter
      `i+1` (via register, accumulator, or in-place memory). Fully unrolled;
      contributes the per-iter critical path *once* (not multiplied by trip).
   - **Sequential dim** — has a carried scalar/register dep. Contributes
      `trip_count × II` to the critical path, where `II` is the latency of
      the carried recurrence.
   - **Reduction dim** — carried dep is an associative op (sum, product,
      min, max, xor, and, or). Tree-reduced: contributes `ceil(log2(trip))`
      to the critical path. Op count stays at `trip − 1` ops.
   Non-associative recurrences (modular state, division chains, KMP-table-
   style) stay sequential — `trip × II`.

  4. **Address arithmetic and loop control are counted as ops (per Convention 1)
   and frequently lie on the critical path.** Address arithmetic typically
   sits on the per-iteration critical path because the load it feeds is on
   that chain (counter → address → load → compute → store). Induction
   carry — the iterator's `i ← i+1` update — is the sequential recurrence
   for any sequential dim, so it directly governs `II` for that dim.
   Treat address gen and induction carry as first-class contributors to
   `total_cycles`, not as free overhead. Example: in FFT, butterfly index
   computation forms a significant fraction of the per-stage critical path.

5. **Dead computations are counted as ops** but by definition cannot lie on
   the critical path to any output — they extend op counts but never
   `total_cycles`.

6. **No register/memory distinction in load/store cost.** Every load and
   store costs 1 cycle, regardless of whether the target is a local scalar,
   an induction variable, a loop-carried accumulator, or an array element.
   No "register-resident" exemption — each named read is a 1-cycle load and
   each named write is a 1-cycle store, same as array access. Anonymous
   dataflow values (unnamed intermediates flowing directly op-to-op without
   a source-level name) remain free. Under full unrolling of a parallel
   dim, each unrolled instance has its own private storage; no aliasing
   assumed.

   Schedule-level corollary: when a reduction dim is tree-scheduled
   (Convention 3), the source-level accumulator collapses into the tree's
   dataflow edges and contributes no loads/stores of its own — only the
   `N` inputs (`N` loads) and the final result (1 store) are charged.

You can cite the convention detiails, but don't cite the convention numbers directly in the eval.md files. 

## Per-kernel statistics
- `total_cycles` — critical-path depth (symbolic in size params).
- `critical_path` — symbolic decomposition, e.g.
   `1 (load) + 1 (mul) + ceil(log2(K)) (reduce) + 1 (store)`.
- Per loop dim: `{name, trip_count, kind: parallel | sequential | reduction, II}`.
   `II` is only meaningful for sequential dims.
- Op counts (loads, stores, adds, address_adds, multiplies, divides,
   compares, bitops, transcendentals: sqrt/exp/cos/sin/log) aggregated
   across all sources from Convention 1. `address_adds` is tracked
   separately from `adds` (address arithmetic is not lumped into regular
   adds; induction increments stay in `adds`). Optionally split into
   `algorithmic` vs. `overhead` (address + induction + dead + scalar L/S)
   for interpretability.

## When conventions break (revisit case-by-case)
- **Non-associative recurrences (II > 1)**: `tridiag_solve`,
   `trsv_lower/upper`, `gauss_seidel_step`, `kmp_table`. The reduction case
   in Convention 3 does not apply; the recurrence is fundamentally serial.
- **Data-dependent termination**: binary search, popcount-while,
   `string_compare`, `wildcard_match`. Trip count and termination predicate
   are input-dependent; the termination compare sits on the critical path
   (Convention 4 exception applies).
- **In-place updates that alias across iterations**: stencils, scans, sorts.
   The carried dep is via memory, not register; check aliasing before
   classifying a dim as parallel.
- **Floating-point reductions**: technically non-associative, but we tree-
   reduce them anyway under this model (lowest-latency bound). Flag the
   kernel if bit-equivalence to a serial reference matters.

## Difficulty classification
Kernels are classified L1–L5 by performance-analysis difficulty; see
`/home/ankaijin/loom/kernel_perf_difficulty.csv`.
- L1 Static-Affine        — closed-form polynomial in size params
- L2 Branched-Bounded     — affine with bounded conditional bodies
- L3 Aggregate-Dependent  — needs input aggregates (NNZ, Σcounts, Σdeg)
- L4 Value-Distribution   — needs value distribution (bit lengths, prefix lengths)
- L5 Structure-Dependent  — needs input ordering/topology (quicksort, BFS)

Under the ASAP model, many L1 kernels collapse from O(N) cycles to O(1) or
O(log N); difficulty class still reflects the information needed to predict
op counts and trip counts, which is unchanged.

## Artifacts
- `kernel_perf_difficulty.csv` — classification + rough formulas for 127 kernels.
- `tests/app/<kernel>/<kernel>_eval.md` — per-kernel eval (cycles, ops, DDG).
   Note: existing eval files were written under the prior serial model with
   reduced op-count scope; they need re-evaluation against the ASAP +
   full-op-count + uniform-L/S conventions above.