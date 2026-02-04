# Project Overview
Loom is a full stack framework for Domain-specific Accelerator, from C++ source code to Hardware Backend

# Documentation
- Check `docs` to understand `loom` design specification.
  - `docs/spec-pragma.md` describes the loom pragma system, used by programmer to provide hint to loom compiler.
  - `docs/spec-cli.md` describes the `loom` compiler command line interface, anything about the loom cli should be documented here.

# General Rules
- Avoid terms to describe development progress (`FIXED`, `Step`, `Week`, `Section`, `Phase`, `AC-x`, etc) in code comments or commit message or PR body.
- Avoid AI tools name (like Codex, Claude, Grok, Gemini, ...) in code comments or git commit message (including authorship) or PR body.
- `TODO`, `FIXME` are allowed in code comments.
- Ideal file size is less than 1300 lines. If a file is more than 1800 lines, please split it into multiple modular and functional-equivalent files. When spliting files, DO NOT use simple spliting as multiple `.inc` files and just including them. Please split them into multiple `.cpp`/`.h` files.
- Files in this project should contain English-only char, NO CJK and NO Emoji.
- Chat response to user should be in the same language as user's
- Projects under `externals` are external projects that are used in this project via source compilation, avoid modifying them.

# ExecPlans
When writing complex features or significant refactors, use an ExecPlan (as described in .agent/PLANS.md) from design to implementation.
The ExecPlan can be stored in `temp` folder as a living execution plan, the filename should be timestamp-ed like `ExecPlan-<YYYYMMDD-hhmmss>.md`, timestamp should use `date +"%Y%m%d-%H%M%S"`. 

# End To End Test Pipeline
Use the following sequence as the end to end test pipeline:
- `ninja -C build clean-loom`
- `ninja -C build loom`
- `ninja -C build check-loom`
- Note: prefix command with `CCACHE_DISABLE=1 CCACHE_TEMPDIR=/tmp` if there is ccache-related permission issue.
