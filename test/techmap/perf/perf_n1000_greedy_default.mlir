// AC-PERF-2: synth_n1000 / greedy default-threads must run in median < 500 ms.

// RUN: %python %S/perf_runner.py --algo greedy --n 1000 --seed 42 --threads 0 --runs 7 --max-median-ms 500 | FileCheck %s

// CHECK: PERF: ALGO=greedy N=1000 threads=0 median_ms={{[0-9.]+}}
// CHECK-NEXT: PERF: PASS
