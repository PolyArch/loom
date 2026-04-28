// Perf budget: synth_n100 / greedy single-thread must run in median < 50 ms.
// The runner pins to core 0 with taskset, takes 7 timed runs, and prints
// a stable PERF: line FileCheck can lock onto.

// RUN: %python %S/perf_runner.py --algo greedy --n 100 --seed 42 --threads 1 --runs 7 --max-median-ms 50 | FileCheck %s

// CHECK: PERF: ALGO=greedy N=100 threads=1 median_ms={{[0-9.]+}}
// CHECK-NEXT: PERF: PASS
