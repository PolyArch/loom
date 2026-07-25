// RUN: loom-parallel-test --workers 4 --serial 5 | FileCheck %s

// runSerialInOrder must visit indices strictly in ascending order even when
// the WorkerPool is configured with multiple workers. This is the
// serial-fallback boundary used by code that mutates MLIR.

// CHECK:      serial[0]
// CHECK-NEXT: serial[1]
// CHECK-NEXT: serial[2]
// CHECK-NEXT: serial[3]
// CHECK-NEXT: serial[4]
