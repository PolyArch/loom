# Project Overview
Loom is a full stack framework for Domain-specific Accelerator, from C++ source code to Hardware Backend

# Documentation
- Check `docs` to understand `loom` design specification, there are important concepts like: `dataflow`, `fabric`, `adg`, `cosim`, each one of them has its own specification documents.

# Core Engineering Principles
These principles apply unless a repository defines stricter rules.

## 1. Occam’s Razor
Do not introduce concepts, abstractions, states, mechanisms, or exceptions unless they are necessary.
This does not mean minimizing complexity at all costs. Complete systems may be complex, but their complexity should emerge from a small set of **distilled, essential structures**, composed through abstraction, encapsulation, and nesting.
Distilled is not the same as simple.
A superficially simple primitive may merely push complexity onto every caller. Prefer a small number of expressive, coherent primitives over many narrow primitives connected by ad hoc conventions.
Optimize for the minimum **conceptual surface area**, not the minimum line count.
Prefer:
* composition over special cases;
* general invariants over repeated local patches;
* a small coherent core over many overlapping mechanisms;
* removing a concept over adding machinery to compensate for it.
Before adding a new entity, determine whether it expresses an essential distinction or merely preserves accidental complexity.

## 2. Test-Driven Development
Treat tests as executable specifications of observable behavior.
For new behavior or bug fixes, define the contract with a failing test before entrenching the implementation. Implement the smallest coherent change, then refactor without changing the tested semantics.
Test stable contracts, invariants, boundaries, and meaningful failure modes. Do not encode incidental implementation details as requirements.
Avoid:
* tests written only to increase coverage;
* excessive mocking;
* snapshots without semantic assertions;
* duplicated assertions across layers;
* large test infrastructures for unstable behavior.
A regression is not fully fixed until its reintroduction can be detected.
Tests should make incorrect changes difficult without making valid refactoring expensive.

## 3. Single Source of Truth
Every fact, rule, schema, configuration value, or state transition must have one semantic owner.
Other representations must be derived, generated, referenced, cached, or validated against that source. They must not become independent authorities.
When duplication is necessary, make the relationship explicit:
* identify the canonical source;
* derive secondary forms mechanically where possible;
* define synchronization and invalidation rules;
* prevent silent divergence.
SSOT does not require centralizing all code. It requires that each truth be defined exactly once.

When two representations disagree, remove the competing authority rather than adding reconciliation logic.

## 4. Eliminate Slop
Do not over-engineer unstable functionality. Early implementation should establish the correct behavior and expose the essential model, not anticipate every hypothetical requirement.
After the behavior is proven, perform a deliberate simplification pass.
Remove:
* speculative abstractions;
* one-use wrappers and interfaces;
* redundant validation;
* temporary branches and flags;
* stale compatibility paths;
* unnecessary configuration;
* duplicated logic;
* defensive handling for states that should be structurally impossible;
* tests that preserve implementation shape rather than semantics.
Ask:
1. Is this design appropriate for the project’s current stage?
2. Can the same behavior be expressed with fewer concepts, states, or execution paths?
3. Did implementation reveal that any abstraction is unnecessary?
4. Can a stronger invariant replace several defensive patches?
Prefer deleting code over explaining unnecessary code. Prefer repairing the model over accumulating exceptions around it.
## Governing Principle
> Build complete systems from distilled foundations. Define behavior explicitly, represent each truth once, and continuously remove accidental complexity.

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

Before analyzing or updating ASAP performance results, read
`tests/app/ASAP_rules.md`. It is the authoritative specification for dynamic
operation counts, critical-path scheduling, loop classification, required
per-kernel statistics, exceptional cases, and difficulty classification.

Evaluation files may cite the rule details, but should not refer to rules by
number.

## ASAP Model Notes
Each *_eval.md file will have a header title "ASAP Model Notes". The text under this header is where I brainstorm how the performs. Do not edit the text directly under this header. Please point out any mistakes you see in that section so I can manually go over and fix them. You are free to edit the text in the rest of the file as you see fit. 

## CGRA-Constrained Model
When adding or updating CGRA-constrained eval sections, read
`docs/spec-kernel-performance.md` first. Preserve the aggregate resource lower
bound and, when modeling time-local resource pressure, also report the
deterministic finite-resource schedule estimate. Report the resource
configuration, aggregate cycles, scheduled cycles, gap cycles/ratio, and local
P/L/S pressure summary. Do not call the finite-resource list-schedule estimate
a lower bound; it is an estimate for the spec-defined scheduling policy.
Continue to leave text directly under `ASAP Model Notes` untouched.