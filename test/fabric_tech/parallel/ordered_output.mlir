// RUN: loom-parallel-test --workers 4 --map 10 | FileCheck %s

// parallelMap must preserve input index order regardless of completion
// order. Closure: i -> i*i over [0..10).

// CHECK:      result[0]=0
// CHECK-NEXT: result[1]=1
// CHECK-NEXT: result[2]=4
// CHECK-NEXT: result[3]=9
// CHECK-NEXT: result[4]=16
// CHECK-NEXT: result[5]=25
// CHECK-NEXT: result[6]=36
// CHECK-NEXT: result[7]=49
// CHECK-NEXT: result[8]=64
// CHECK-NEXT: result[9]=81
