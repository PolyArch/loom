// AC-PERF-3: synth_n5000 / greedy default-threads must run in median < 5 s.
// Gated behind the `long-perf` lit feature, which lit.cfg.py enables only
// when the environment variable LOOM_PERF=long is set.

// REQUIRES: long-perf

// RUN: %python %S/perf_runner.py --algo greedy --n 5000 --seed 42 --threads 0 --runs 7 --max-median-ms 5000 | FileCheck %s

// CHECK: PERF: ALGO=greedy N=5000 threads=0 median_ms={{[0-9.]+}}
// CHECK-NEXT: PERF: PASS
