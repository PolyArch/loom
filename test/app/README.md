# test/app -- numeric-kernel smoke tests

Small, self-contained C/C++ programs used as drop-in targets for the Loom
compiler frontend. Each kernel is a complete CMake project that builds with
stock `gcc`/`g++` and produces deterministic stdout that a shell script
compares against an `expected.txt`.

These programs intentionally have **no dependency on the top-level Loom
CMakeLists or libraries**. They are independent so that a future
`loom-cc` / `loom-c++` drop-in driver can build them by overriding
`CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` only.

## Layout

```
test/app/
  README.md
  run_all.sh                 -- runs every kernel's run_check.sh
  vecadd/                    -- C, float element-wise add (N=64)
  gemm/                      -- C++, float matrix multiply (M=N=K=8)
  dotproduct/                -- C, float dot product (N=64)
  conv1d/                    -- C++, float 1-D convolution (N=64, K=5)
  reduction/                 -- C, int sum reduction (N=128)
```

Each kernel directory contains:

```
<kernel>/
  CMakeLists.txt             -- self-contained, defines two executables
  main_func.{c,cpp}          -- kernel implemented as a separate function
  main_inline.{c,cpp}        -- equivalent kernel inlined into main
  expected.txt               -- expected stdout (single line)
  run_check.sh               -- build, run, and diff against expected.txt
  .gitignore                 -- ignores the per-kernel build/ tree
```

The two source variants exercise the two shapes that downstream compiler
passes need to handle: a separate function called from `main` (CallSite with
explicit semantics) and an equivalent inline expression nest inside `main`.
Both variants must produce byte-identical stdout.

## Running

From the repo root:

```sh
bash test/app/run_all.sh
```

This loops over every kernel and prints a `PASS`/`FAIL` summary. The script
exits non-zero if any kernel fails.

To run a single kernel:

```sh
bash test/app/vecadd/run_check.sh
```

## Compiler override (gcc baseline vs future loom drop-in)

`run_check.sh` and `run_all.sh` honor the standard `CC` and `CXX`
environment variables and forward them to CMake via
`-DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX`. The defaults are `gcc`
and `g++`.

```sh
# default gcc/g++ baseline
bash test/app/run_all.sh

# future loom drop-in
CC=loom-cc CXX=loom-c++ bash test/app/run_all.sh
```

## Determinism notes

* Inputs are hard-coded (no random, no time, no environment lookups).
* Floats are printed with `%.6f` and ints with `%d` to keep stdout
  byte-identical across glibc versions.
* `expected.txt` is checked in and validated by `run_check.sh` on every
  invocation.
