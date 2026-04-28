// Perf budget: synth_n5000 / greedy default-threads must run in median < 5 s.
// This protects the implicit cap "5000-node synthetic graph partitions in
// under 5 seconds with default threads" that several downstream consumers
// rely on. The wall-clock cost is ~6 s of test time, judged worth carrying
// in the default suite.

// RUN: %python %S/perf_runner.py --algo greedy --n 5000 --seed 42 --threads 0 --runs 7 --max-median-ms 5000 | FileCheck %s

// CHECK: PERF: ALGO=greedy N=5000 threads=0 median_ms={{[0-9.]+}}
// CHECK-NEXT: PERF: PASS
