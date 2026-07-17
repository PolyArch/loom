// RUN: not loom-raise-opt --loom-lower-for-to-graph %s 2>&1 | FileCheck %s

// CHECK: error: {{.*}}cannot extract loop-carried memory capability '!llvm.ptr'; project the recurrence to an explicit index domain
// CHECK-NOT: dataflow.graph.func
dataflow.thread private @pointer_walk(%src: !llvm.ptr, %n: index)
    ctrl (%start: none) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %n step %c1
      iter_args(%current = %src) -> (!llvm.ptr) {
    %next = llvm.getelementptr %current[1]
        : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  }
  %sink = llvm.ptrtoint %result : !llvm.ptr to i64
  dataflow.thread.yield
}
