# CMSIS-DSP LLVM IR drop-in smoke pipeline

`run_cmsis_dsp_ir.sh` drives `loom-cc` against a representative subset
of ARM CMSIS-DSP sources living under
`externals/cmsis-dsp/Source/<subdir>/`. Each source is compiled with
`--target=<arm-triple> -mcpu=<cortex-m*> -emit-llvm -S` and the
resulting `.ll` is inspected for two cheap invariants:

1. The IR carries the expected normalized ARM target triple
   (e.g. `thumbv7em-unknown-none-eabi`), confirming clang switched to
   the ARM frontend even though loom-cc's bundled backend is x86_64-only.
2. The IR contains a `define` for every expected function symbol
   listed for that source. ALL listed symbols must be present (the
   pipeline does not pass on any-of); `declare` does not satisfy the
   check because we want the wrapper bodies to actually lower.

This is a pure smoke check: no behavioral assertions on the IR, no
codegen step (the bundled LLVM is built with `LLVM_TARGETS_TO_BUILD=host`
and has no ARM backend, but the clang frontend can still emit IR for
any triple it knows).

## Layout

| Path                         | Role                                                        |
|------------------------------|-------------------------------------------------------------|
| `cmsis_dsp_targets.txt`      | One row per source: triple, cpu, expected triple/symbols.   |
| `run_cmsis_dsp_ir.sh`        | Loops over rows, invokes loom-cc, asserts the invariants.   |
| `out/<basename>.ll`          | Per-source IR artifact (regenerated each run).              |
| `out/<basename>.log`         | loom-cc stdout/stderr captured for failures.                |

`out/` is gitignored together with everything under `out/`; do not
commit the `.ll` artifacts.

## Running locally

From the repo root:

```bash
bash test/cmsis-dsp/run_cmsis_dsp_ir.sh
```

The script honors `LOOM_CC` if you want to point at an alternate
binary; otherwise it resolves `build/bin/loom-cc` relative to the
repo root. If the binary is missing the script exits 2 with a hint
to build it.

## Include path requirements

CMSIS-DSP headers depend on CMSIS-Core (`cmsis_compiler.h` and the
platform CPU defines come from there), so the runner threads three
include paths into every invocation, in this order:

1. `externals/cmsis-core/CMSIS/Core/Include` -- CMSIS-Core public.
2. `externals/cmsis-dsp/Include` -- CMSIS-DSP public.
3. `externals/cmsis-dsp/PrivateInclude` -- internal headers (e.g.
   `arm_compiler_specific.h`) that several `.c` files include
   directly.

If you add a new row to `cmsis_dsp_targets.txt` and the source needs
yet another include root, wire it into the runner explicitly rather
than reaching into the source tree -- the externals are vendored
verbatim and must stay untouched.

## Libc-header strategy: `-isystem /usr/include`

CMSIS-DSP transitively pulls in `<string.h>`, `<math.h>`, and
`<stdint.h>`. We are cross-compiling for thumb but only emitting IR
(no link, no ARM codegen here, since the bundled LLVM is host-only),
so it is enough to point clang at the host's glibc headers via
`-isystem /usr/include` and pretend we are hosted
(`-D__STDC_HOSTED__=1`). The emitted IR's data layout still reflects
the ARM target chosen by `--target=`; the host headers only affect
preprocessing.

One wrinkle: glibc's `gnu/stubs.h` dispatches to a per-arch stub file
(`gnu/stubs-32.h`, `gnu/stubs-64.h`, `gnu/stubs-x32.h`) by checking
`__x86_64__`, `__LP64__`, and `__ILP32__`. The selected ARM triple
defines `__ILP32__=1` and not `__x86_64__`, so vanilla
`-isystem /usr/include` lands on the missing 32-bit stubs file. The
runner pins the dispatch to the LP64 stub explicitly:

```
-D__x86_64__=1 -D__LP64__=1 -U__ILP32__
```

These defines only steer preprocessing through `gnu/stubs.h`. They
do not change the target triple, the data layout, or the lowered
intrinsics in the emitted IR -- those remain the ARM ones the
`--target=`/`-mcpu=` pair selected. The combination is documented
in the runner's file header comment.

We accept this for parse-level smoke because the alternative (a
freestanding sysroot or a vendored ARM newlib) would be much heavier
and is not justified for a drop-in test that does not link.

## Coverage

Sources are picked from the per-feature subdirectories under
`externals/cmsis-dsp/Source/`:

| subdir                | sources                                                      |
|-----------------------|--------------------------------------------------------------|
| `BasicMathFunctions`  | `arm_add_q15.c`, `arm_mult_f32.c`, `arm_offset_f32.c`, `arm_abs_f32.c` |
| `FastMathFunctions`   | `arm_sin_f32.c`, `arm_sqrt_q15.c`                            |
| `FilteringFunctions`  | `arm_fir_f32.c` (hard-float), `arm_biquad_cascade_df1_f32.c` (hard-float) |
| `MatrixFunctions`     | `arm_mat_mult_f32.c`, `arm_mat_add_f32.c`                    |
| `StatisticsFunctions` | `arm_mean_f32.c` (v8m.main), `arm_max_f32.c`, `arm_var_f32.c` (hard-float) |
| `SupportFunctions`    | `arm_copy_f32.c`, `arm_fill_f32.c`                           |
| `TransformFunctions`  | `arm_cfft_f32.c`                                             |

The default triple for f32 sources is `thumbv7em-none-eabi
-mcpu=cortex-m4`; sources whose names imply hard-float, or that are
worth a hard-float-ABI smoke, use `thumbv7em-none-eabihf
-mcpu=cortex-m4 -mfloat-abi=hard`. One row uses
`thumbv8m.main-none-eabi -mcpu=cortex-m33` so the v8m-main profile
is exercised end-to-end.

## Extending the source list

1. Pick a `.c` file under `externals/cmsis-dsp/Source/<subdir>/`.
   Prefer f32 variants when in doubt; q15/q31 saturation paths add
   noise but do not strengthen the smoke.
2. Append a pipe-separated row to `cmsis_dsp_targets.txt`:
   `src_relpath | triple | mcpu | normalized_triple | expected_symbols | extra_cflags`
3. The normalized triple is what clang emits in the IR (run the
   script once and grep the failing log if you are not sure).
4. If the source's primary definition is gated behind a feature you
   do not enable (`ARM_MATH_NEON`, `ARM_MATH_MVEF`, ...), pick a
   different source. Do not enable feature flags globally.
5. If a row needs an extra preprocessor define beyond `-mcpu=`, put
   it in the `extra_cflags` column rather than editing the source.

## Constraint: do not modify externals

Everything under `externals/cmsis-dsp/` and `externals/cmsis-core/`
is vendored verbatim from ARM. The pipeline must remain a no-touch
drop-in: any per-source tweak belongs in `cmsis_dsp_targets.txt` or
in the runner's flag composition. If a source genuinely cannot
compile even with correct flags, drop it from the list and document
the reason in the commit message.
