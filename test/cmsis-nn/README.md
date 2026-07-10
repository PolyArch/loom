# CMSIS-NN DFG Integration

`test/frontend/cmsis_nn_dfg.test` is the only frontend CMSIS-NN
integration entry point. It runs `run_cmsis_nn_dfg.sh` over the real
sources selected in `cmsis_nn_integration_targets.txt`.

For every target, the runner:

- compiles the source with `loom-cc` and requires nonempty LLVM IR;
- raises the IR with `loom-raise` and requires nonempty MLIR;
- lowers it with `loom-lower` and requires nonempty dataflow MLIR;
- parses the result with `loom-raise-opt`;
- checks that the public source symbol remains;
- requires a launched dataflow graph with no residual SCF operations.

The target table format is:

```text
source|triple|cpu|source_symbol|extra_cflags
```

`extra_cflags` is optional and contains whitespace-separated compiler flags.
Sources are relative to `externals/cmsis-nn/Source`.

Run the integration directly from the repository root:

```bash
bash test/cmsis-nn/run_cmsis_nn_dfg.sh
```

`TARGETS_OVERRIDE` can select a focused target table and `OUT_OVERRIDE`
can isolate generated artifacts. Tool paths can be overridden with the
corresponding `LOOM_*` environment variables.

This table records current executable-DFG coverage, not the canonical
CMSIS-NN suite inventory. The canonical inventory remains every tracked
`Source/**/*.c` file at the pinned submodule commit.

Files under `externals/` remain unmodified. Per-source compiler changes belong
in the target table.
