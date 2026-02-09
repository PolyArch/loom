# Project Overview
Loom is a full stack framework for Domain-specific Accelerator, from C++ source code to Hardware Backend

# Documentation
- Check `docs` to understand `loom` design specification, there are important concepts like: `dataflow`, `fabric`, `adg`, `cosim`, each one of them has its own specification documents.

# Referencing Code Locations
- **NEVER** use exact line-number ranges (e.g. `path/to/file:123-456`) to reference or quote code locations in plans, documentation, comments, or code. Line numbers shift as development proceeds, making such references fragile and misleading.
- **Always** identify locations by semantic anchors: function/class/method names, section/chapter headings, or other stable identifiers.
- If a line-number range is absolutely unavoidable, use a deliberately broad/approximate range and clearly mark it as approximate. Even so, this practice **should be minimized**.

# General Rules
- Avoid terms to describe development progress (`FIXED`, `Step`, `Week`, `Section`, `Phase`, `AC-x`, `round-i`, `step-i` etc) in code comments or commit message or PR body.
- Avoid AI tools name (like Codex, Claude, Grok, Gemini, ...) in code comments or git commit message (including authorship) or PR body.
- `TODO`, `FIXME` are allowed in code comments.
- Ideal file size is less than 1300 lines. If a file is more than 1800 lines, please split it into multiple modular and functional-equivalent files. When spliting files, DO NOT use simple spliting as multiple `.inc` files and just including them. Please split them into multiple `.cpp`/`.h` files.
- Files in this project should contain English-only char, NO CJK and NO Emoji. Files in `temp` folder are allowed to bypass this rule.
- Header files (like .h/.cuh etc) in `include`; Implementation files (like .c/.cpp/.cu etc) in `lib`.
- Chat response to user should be in the same language as user's
- Projects under `externals` are external projects that are used in this project via source compilation, avoid modifying them.

# Useful Hints
- If EDA related tools (like `vcs`, `verilator`, `verdi`, `dc_shell`, `fc_shell`, etc.) cannot be found, please try `module avail` and use `module load ...` to load environments. 

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
- Note: prefix command with `CCACHE_DISABLE=1 CCACHE_TEMPDIR=/tmp` if there is ccache-related permission issue.
