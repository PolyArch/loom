# Project Overview
Loom is a full stack framework for Domain-specific Accelerator, from C++ source code to Hardware Backend

# Documentation
- Check `docs` to understand `loom` design specification, there are important concepts like: `dataflow`, `fabric`, `adg`, `cosim`, each one of them has its own specification documents.

# Project Rules
- Header files (like .h/.cuh etc) in `include`; Implementation files (like .c/.cpp/.cu etc) in `lib`.
- When splitting files, split them into multiple `.cpp`/`.h` files (not `.inc` includes).
- Projects under `externals` are external projects that are used in this project via source compilation, avoid modifying them.
- Files in `temp` folder are allowed to bypass English-only and file size rules.

# Useful Hints
- If EDA related tools (like `vcs`, `verilator`, `verdi`, `dc_shell`, `fc_shell`, etc.) cannot be found, please try `module avail` and use `module load ...` to load environments.
- For example, if you want to run verilator, prepend `module load verilator && ...`; for vcs/verdi, prepend `module load synopsys/vcs synopsys/verdi && ...` to command you want to run. Prepend with `module purge && ...` can clean the loaded tools.
- For tests that need verilator and vcs/verdi, you can do `module purge && module load synopsys/vcs synopsys/verdi verilator && ...` to load all of them.

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
