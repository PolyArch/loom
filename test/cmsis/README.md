# CMSIS-Core LLVM IR drop-in smoke pipeline

`run_cmsis_ir.sh` drives `loom-cc` against a representative subset of
ARM CMSIS-Core sources living under
`externals/cmsis-core/CMSIS/Core/Test/src/`. Each source is compiled
with `--target=<arm-triple> -mcpu=<cortex-m*> -emit-llvm -S` and the
resulting `.ll` is inspected for two cheap invariants:

1. The IR carries the expected normalized ARM target triple
   (e.g. `thumbv6m-unknown-none-eabi`), confirming clang switched to the
   ARM frontend even though loom-cc's bundled backend is x86_64-only.
2. The IR contains a `define` / `declare` for at least one of the
   expected function symbols listed for that source.

This is a pure smoke check: no behavioral assertions on the IR, no
codegen step (the bundled LLVM is built with `LLVM_TARGETS_TO_BUILD=host`
and has no ARM backend, but the clang frontend can still emit IR for
any triple it knows).

## Layout

| Path                       | Role                                                        |
|----------------------------|-------------------------------------------------------------|
| `cmsis_targets.txt`        | One row per source: triple, cpu, expected triple/symbols.   |
| `run_cmsis_ir.sh`          | Loops over rows, invokes loom-cc, asserts the invariants.   |
| `out/<basename>.ll`        | Per-source IR artifact (regenerated each run).              |
| `out/<basename>.log`       | loom-cc stdout/stderr captured for failures.                |

`out/` is gitignored together with everything under `out/`; do not
commit the `.ll` artifacts.

## Running locally

From the repo root:

```bash
bash test/cmsis/run_cmsis_ir.sh
```

The script honors `LOOM_CC` if you want to point at an alternate
binary; otherwise it resolves
`build/tools/loom-cc/loom-cc` relative to the repo root. If the
binary is missing the script exits 2 with a hint to build it.

## Coverage

| profile / cpu               | sources                          |
|-----------------------------|----------------------------------|
| `thumbv6m` / Cortex-M0      | `apsr.c`, `clz.c`, `nop.c`, `dmb.c` |
| `thumbv7m` / Cortex-M3      | `dsb.c`, `isb.c`, `basepri.c`     |
| `thumbv7em` / Cortex-M4     | `ldrex.c`, `strex.c`, `msp.c`     |
| `thumbv8m.main` / Cortex-M33 | `ldaex.c`, `stl.c`               |

The set was chosen to span the four main Cortex-M profiles and to
exercise distinct intrinsic families (special-register read/write,
barrier intrinsics, CLZ/NOP, BASEPRI, load-exclusive / store-exclusive,
and the v8m load-acquire-exclusive / store-release pair).

## Extending the source list

1. Pick a `.c` file under
   `externals/cmsis-core/CMSIS/Core/Test/src/`.
2. Read its `// REQUIRES:` sentinel and look up the matching triple/cpu
   in the table below (also documented inside `cmsis_targets.txt`).
3. Append a new pipe-separated row to `cmsis_targets.txt` with:
   `src_relpath | triple | mcpu | normalized_triple | expected_symbols | extra_cflags`
   The normalized triple is what clang emits in the IR (run the script
   once and grep the failing log if you are not sure).

Sentinel-to-flag map (mirrors upstream's lit config):

| `// REQUIRES:`   | `--target=`                  | `-mcpu=`     |
|------------------|------------------------------|--------------|
| `thumbv6m`       | `thumbv6m-none-eabi`         | `cortex-m0`  |
| `thumbv7m`       | `thumbv7m-none-eabi`         | `cortex-m3`  |
| `thumbv7em`      | `thumbv7em-none-eabi`        | `cortex-m4`  |
| `thumbv8m_base`  | `thumbv8m.base-none-eabi`    | `cortex-m23` |
| `thumbv8m_main`  | `thumbv8m.main-none-eabi`    | `cortex-m33` |
| `armv81mml`      | `thumbv8.1m.main-none-eabi`  | `cortex-m55` |

If a source needs an extra preprocessor define beyond `-mcpu=`, add it
to the `extra_cflags` column rather than editing the source itself.

## Constraint: do not modify externals

Sources under `externals/cmsis-core/` are vendored verbatim from ARM.
The pipeline must remain a no-touch drop-in: any per-source tweak
belongs in `cmsis_targets.txt`. If a source genuinely cannot compile
even with correct flags, drop it from the list and document the reason
in the commit message.
