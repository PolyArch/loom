# CMSIS-DSP Representative DFG Integration

`test/frontend/cmsis_dsp_dfg.test` is the only frontend CMSIS-DSP
integration entry point. It runs `run_cmsis_dsp_dfg.sh` over the real
sources selected in `cmsis_dsp_integration_targets.txt`, representing
fast math, interpolation, and matrix code paths.

For every target, the runner:

- compiles the source with `loom-cc` and requires nonempty LLVM IR;
- raises the IR with `loom-raise` and requires nonempty MLIR;
- lowers it with `loom-lower` and requires nonempty dataflow MLIR;
- parses the result with `loom-raise-opt`;
- checks that the public source symbol remains and that a matching
  `dataflow.thread` or `dataflow.graph.func` definition exists.

The target table format is:

```text
source|triple|cpu|source_symbol|extra_cflags
```

`extra_cflags` is optional and contains whitespace-separated compiler
flags. Sources are relative to `externals/cmsis-dsp/Source`.

Run the integration directly from the repository root:

```bash
bash test/cmsis-dsp/run_cmsis_dsp_dfg.sh
```

`TARGETS_OVERRIDE` can select a focused target table and `OUT_OVERRIDE`
can isolate generated artifacts. Tool paths can be overridden with the
corresponding `LOOM_*` environment variables.

This table is representative integration coverage, not the canonical
CMSIS-DSP suite inventory. The canonical inventory remains every tracked
`Source/**/*.c` file at the pinned submodule commit.

Files under `externals/` remain unmodified. Per-source compiler changes
belong in the target table.
