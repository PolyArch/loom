# LoomBench application corpus

Small, self-contained C/C++ programs used as drop-in targets for the Loom
compiler frontend. `manifest.json` is the LoomBench membership source of truth.
The manifest-driven native runner covers every listed case, compiles both
source variants, executes every resulting binary, and compares stdout
byte-for-byte with the checked-in expected output.

The manifest-driven IR runner provides source-to-IR integration coverage. The
default `raise` tier contains `vecadd`, `matmul`, `spmm`, `gather`, and
`edge_update`. No manifest case currently belongs to the `dfg` tier. A default
source-to-DFG tier requires a legal schedule provenance or DSE selection path;
unsupported explicit requests fail closed instead of publishing an incomplete
graph. The runner compiles only the source whose stem is `main_func`; the
native runner remains responsible for both `main_func` and `main_inline`
variants.

These programs intentionally have no dependency on the top-level Loom build
or libraries. Stock `gcc`/`g++` and compatible drop-in drivers can compile the
same manifest entries.

## Layout

```
test/app/
  README.md
  manifest.json              -- ordered case metadata and tier selection
  native_runner.py           -- reusable native-runner API and command-line entry
  ir_runner.py               -- source-to-SCF and source-to-DFG integration runner
  vecadd/
    main_func.c              -- kernel implemented as a separate function
    main_inline.c            -- equivalent kernel inlined into main
    expected.txt             -- exact expected stdout bytes
```

The two source variants exercise the shapes that downstream compiler passes
need to handle: a separate function called from `main` and an equivalent
inline expression nest inside `main`. Both variants must produce byte-identical
stdout, including trailing newlines.

## Native execution

From the repository root, run the full manifest:

```sh
python3 test/app/native_runner.py --all
```

Run one or more selected cases:

```sh
python3 test/app/native_runner.py --case vecadd --case gemm
```

Use `--jobs` to set the case worker count. `LOOM_NATIVE_RUNNER_JOBS`,
`LOOM_TEST_JOBS`, and `JOBS` provide environment defaults in that order.

```sh
python3 test/app/native_runner.py --all --jobs 8
```

Build products are written below `build/test-runs/native-runner` by default,
with one deterministic directory per case. `--build-root` selects another
root without creating output in case source directories.

## IR integration

Run the five default source-to-SCF cases:

```sh
python3 test/app/ir_runner.py --stage raise
```

Use one or more explicit `--case` options to run named manifest entries:

```sh
python3 test/app/ir_runner.py --stage raise \
  --case vecadd \
  --case matmul
```

Explicit cases do not need to belong to the selected stage's default tier.
This permits focused consumers to request IR without broadening the default
integration set. An explicit `--stage dfg --case matmul` request is the
fail-closed provenance anchor and returns a nonzero status until a legal
schedule selection path is available.

The IR runner executes cases sequentially and applies each case's
`compiler_flags`. It writes these artifacts below
`build/test-runs/app-ir-runner/<case>` by default:

```text
main_func.ll
main_func.scf.mlir
main_func.dfg.mlir  # dfg stage only
```

`--manifest` selects another manifest and `--build-root` selects another output
root. `LOOM_CC`, `LOOM_CXX`, `LOOM_RAISE`, `LOOM_LOWER`, and
`LOOM_RAISE_OPT` override the Loom tools. Defaults are resolved below
`build/bin`.

## Compiler override

The default C command begins with
`gcc -std=c11 -O2 -Wall -Wextra -Werror`. The default C++ command begins with
`g++ -std=c++17 -O2 -Wall -Wextra -Werror`. Per-case `compiler_flags` from the
manifest follow these defaults so a case can override them. Per-case
`link_flags` are placed at the end of each compiler command.

Use explicit driver paths with `--cc` and `--cxx`:

```sh
python3 test/app/native_runner.py --all \
  --cc build/bin/loom-cc \
  --cxx build/bin/loom-c++
```

Relative driver paths are resolved against the directory from which the
runner was invoked. When the command-line options are omitted, `CC` and `CXX`
provide overrides before the `gcc` and `g++` defaults.

## Manifest controls

`--manifest` selects an alternate manifest. Each case entry supplies its
language, ordered source list, expected executable names, expected stdout file,
compiler flags, and link flags. The source and executable lists must have equal
length. Every case carries the `run` tier. The `raise` tier selects the default
IR integration cases; the `dfg` tier is empty until source lowering can supply
legal schedule provenance. Explicit IR requests are not restricted by those
tiers. Successful `loom-lower` publication is the DFG semantic validity gate;
the runner retains only artifact and parseability checks after that native
validation.

The repository-wide `test/corpus_inventory.py` command validates this manifest
and combines its cases with the two CMSIS suites. Running `list` without case
selectors emits the complete inventory; `--case loombench:NAME` selects an
explicit LoomBench case. Manifest tiers remain integration subsets and do not
change membership.

## Determinism notes

* Inputs are hard-coded, with no random, time, or environment lookups.
* Floats are printed with `%.6f` and integers with stable format strings.
* Expected output is read as bytes and compared without trimming whitespace.
* Diagnostics are emitted in manifest order even when cases run concurrently.
