# CMSIS-NN Raise Smoke Coverage

`test/frontend/cmsis_nn_raise_smoke.test` runs
`run_cmsis_nn_raise_smoke.sh` over the real sources selected in
`cmsis_nn_raise_smoke_targets.txt`.

For every target, the runner:

- compiles the source with `loom-cc` and requires nonempty LLVM IR;
- raises the IR with `loom-raise` and requires nonempty MLIR;
- parses the result with `loom-raise-opt`;
- checks that the public source symbol remains.

This is source-to-SCF coverage. It does not claim that the source contains an
explicit SpatialCore ownership boundary or produces a canonical graph.

The target table format is:

```text
source|triple|cpu|source_symbol|extra_cflags
```

`extra_cflags` is optional and contains whitespace-separated compiler flags.
Sources are relative to `externals/cmsis-nn/Source`.

Run the smoke path directly from the repository root:

```bash
bash test/cmsis-nn/run_cmsis_nn_raise_smoke.sh
```

`SMOKE_TARGETS_OVERRIDE` can select another explicit smoke table and
`OUT_OVERRIDE` can isolate generated artifacts. Tool paths can be overridden
with the corresponding `LOOM_*` environment variables.

The canonical CMSIS-NN inventory is every tracked `Source/**/*.c` file at the
pinned submodule commit. The command
`python3 test/corpus_inventory.py list --suite cmsis-nn` derives and emits that
complete set. The smoke table is validated as a strict inventory subset before
the compiler runs; it records only current source-to-SCF smoke coverage and
cannot add or remove suite members.

Files under `externals/` remain unmodified. Sparse compiler metadata needed by
the smoke path belongs in the smoke table and does not imply support for any
unattempted inventory source.
