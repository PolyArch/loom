// RUN: loom-parallel-test --workers 8 --for 1000 | FileCheck %s

// parallelFor over [0..1000) summing into a std::atomic<int64_t> must
// produce 0+1+...+999 = 499500 regardless of worker count or completion
// order.

// CHECK: sum=499500
