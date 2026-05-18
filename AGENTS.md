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
Estimate cycle counts and op counts (loads, stores, adds, multiplies, etc.) for
each kernel under `tests/app/`, under a 1-cycle-per-op dataflow execution model
with sufficient hardware resources.

## Conventions
1. Loop-carried values (accumulators, induction vars) are register-resident.
   No L/S charged on access; one load at entry, one store at exit if observed.
2. Address arithmetic is ignored. Adds/muls used solely for index/pointer math
   (e.g., `i + lag` for `&x[i+lag]`) are not counted.
3. Loop control is ignored. Iterator increments (`i++`) and bound compare/branch
   are assumed absorbed by branch prediction; no cycle or op cost charged.
4. Outer-loop work is a separate additive term. Total = inner_cycles + outer_cycles
   (prologue inits, epilogue stores). Not folded into per-iter inner cost.
5. Pipeline fill is dropped. Per-iteration feed-forward compute is treated as
   one combinational stage; only the loop-carried back-edge costs an II.
   Inner cycles = trip_count × II.

## Per-kernel statistics
- `inner_cycles`, `outer_cycles`, `total_cycles`
- `II` (initiation interval — longest loop-carried recurrence latency)
- Op counts: loads, stores, adds, multiplies, divides, compares, bitops,
  transcendentals (sqrt/exp/cos/sin/log)

## When conventions break (revisit case-by-case)
- Short inner loops with deep DAGs: `mat3x3_mult`, `quat_mult`, `cross_product`,
  `crc32`, `gf_mul`. Pipeline fill becomes non-negligible.
- Multi-op loop-carried recurrences (II > 1): `tridiag_solve`, `trsv_lower/upper`,
  `gauss_seidel_step`, `kmp_table`. Must model recurrence latency explicitly.
- Data-dependent branches: binary search variants, popcount-while, early-break
  loops (`string_compare`, `wildcard_match`). Convention 3 leaks here.

## Difficulty classification
Kernels are classified L1–L5 by performance-analysis difficulty; see
`/home/ankaijin/loom/kernel_perf_difficulty.csv`.
- L1 Static-Affine        — closed-form polynomial in size params
- L2 Branched-Bounded     — affine with bounded conditional bodies
- L3 Aggregate-Dependent  — needs input aggregates (NNZ, Σcounts, Σdeg)
- L4 Value-Distribution   — needs value distribution (bit lengths, prefix lengths)
- L5 Structure-Dependent  — needs input ordering/topology (quicksort, BFS)

## Artifacts
- `kernel_perf_difficulty.csv` — classification + rough formulas for 127 kernels.
- `tests/app/<kernel>/<kernel>_eval.md` — per-kernel eval (cycles, ops, DDG).
