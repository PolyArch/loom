// RUN: loom-parallel-test --workers 1 --map 5 | FileCheck %s

// workers=1 must bypass the pool entirely and produce identical output to
// the multi-threaded codepath.

// CHECK:      result[0]=0
// CHECK-NEXT: result[1]=1
// CHECK-NEXT: result[2]=4
// CHECK-NEXT: result[3]=9
// CHECK-NEXT: result[4]=16
