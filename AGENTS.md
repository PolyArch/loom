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
   - **Address generation** — a bare-variable or bare-named-scalar subscript,
      e.g. `a[i]` or `a[idx]`, charges **zero** address_adds and contributes no
      address cycle of its own: once the index value is available, the element is
      reached directly.

      When the `[]` expression itself contains arithmetic, evaluate that subscript
      expression as a normal expression DAG before the access can fire. Each op in
      the expression costs 1 cycle and participates in the critical path according
      to its data dependencies. Adds/subs inside the subscript are counted as
      `address_adds`, not regular `adds`; multiplies/divides/shifts/bitops inside
      the subscript are counted in their normal op categories. Loop-invariant
      hoisting applies recursively inside index expressions.

      Examples:
      - `a[i+1]`: one `address_add`; the load waits for `i+1`.
      - `a[i+j+k]`: two `address_adds`, tree-scheduled as a 2-level add DAG.
      - `a[2*i]`: one multiply; no `address_add`.
      - `a[ci*(H*W) + h*W + w]`: evaluate the index expression functionally:
         cycle 1 computes `H*W` and `h*W`; cycle 2 computes `ci*(H*W)` and
         `h*W + w`; cycle 3 computes the full index. The access fires after that.
      
      For non-affine indexing such as `a[idx[i]]`, charge the loads and arithmetic
      required to compute `idx[i]`; the outer access `a[loaded_idx]` then indexes on
      a loaded scalar value and charges no additional address_add.
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

4. **Inline address arithmetic and loop control are counted as ops (per
   Convention 1) and lie on the critical path when present.** When a subscript 
   contains inline arithmetic, the full subscript expression DAG sits on the 
   per-iteration critical path when the load/store it feeds is on that
   chain. A bare subscript carries no address arithmetic and
   adds no such cycle — the access fires as soon as the index value and the
   array are available. Induction carry — the iterator's `i ← i+1` update —
   is the sequential recurrence for any sequential dim, so it directly
   governs `II` for that dim. Treat inline address gen and induction carry
   as first-class contributors to `total_cycles`, not as free overhead.
   Example: in FFT, butterfly index computation (`a[k+j+m/2]`) forms a
   significant fraction of the per-stage critical path.

5. **Dead computations are counted as ops** but by definition cannot lie on
   the critical path to any output — they extend op counts but never
   `total_cycles`.

6. **Load/store cost for memory-backed values.** Every load and store of
   a memory-backed value costs 1 cycle. A source-level scalar is
   memory-backed if **any** of:
   - it has more than one assignment site in source,
   - it carries state across loop iterations (iter-arg / accumulator /
      induction variable), or
   - it aliases or destinations into an array / output buffer.

   Memory-backed scalars charge 1 cycle per named read and 1 cycle per
   named write — no "register-resident" exemption.

   **One load per iteration (fan-out within an iteration).** A memory-
   backed scalar read several times within a single iteration with **no
   intervening write** is loaded **once**; that one load fans out to every
   use in the iteration at no extra cost. Reads collapse to a single load
   only across the span between writes — a read that follows a write to the
   same scalar within the same iteration is a fresh load. Examples:
   - `crc & 1` and `crc >> 1` in the same bit iter (no write between) —
      **one** load of `crc`, fanned to both.
   - `(value & mask)` in the loop test and `mask >>= 1` in the body —
      **one** load of `mask` per iter, fanned to test and body.
   - `inplace[i]` read for a compare-swap, then re-read after `inplace[i]++`
      has written it — **two** loads, because a write separates the reads.

   A scalar assigned exactly once at its declaration and not loop-carried
   is **anonymous dataflow**: free fan-out from the defining op to all
   consumers, no L/S cost. Loom runs `mem2reg` at `-O1`, so such scalars
   have no IR-level memory cell — the source name is metadata, not storage.

   Anonymous arithmetic intermediates (unnamed op-to-op results) remain
   free. Under full unrolling of a parallel dim, each unrolled instance
   has its own private storage; no aliasing assumed.

   **Schedule-level corollary** (unchanged): when a reduction dim is
   tree-scheduled (Convention 3), the source-level accumulator collapses
   into the tree's dataflow edges and contributes no loads/stores of its
   own — only the `N` inputs (`N` loads) and the final result (1 store)
   are charged.

   **Examples:**
   - `uint32_t value = input_data[i]; if (value == 0) ...` — `value`
      assigned once, not carried → anonymous. The load of `input_data[i]`
      flows directly to the cmp.
   - `uint32_t count = 0; while (...) { count++; }` — `count` reassigned
      per iter → memory-backed. Each iter charges load + add + store.
   - `int x; if (c) x = a; else x = b;` — two write sites → memory-backed
      (conservative; we don't do phi-insertion analysis).

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

## ASAP Model Notes
Each *_eval.md file will have a header title "ASAP Model Notes". The text under this header is where I brainstorm how the performs. Do not edit the text directly under this header. Please point out any mistakes you see in that section so I can manually go over and fix them. You are free to edit the text in the rest of the file as you see fit. 