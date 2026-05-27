// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// Counted scf.while loop with a do-while shape (the increment lives in
// the `before` region and the comparison is on the bumped iv) lifts to
// scf.for. This is the shape produced by --lift-cf-to-scf followed by
// --canonicalize on raised LLVM IR.

// CHECK-LABEL: func.func @counted_reduce_sum
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
func.func @counted_reduce_sum(%buf: memref<?xf32>, %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %r:2 = scf.while (%iv = %c0, %acc = %f0) : (index, f32) -> (index, f32) {
      %v = memref.load %buf[%iv] : memref<?xf32>
      %sum = arith.addf %acc, %v : f32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi ne, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum : index, f32
    } do {
    ^bb0(%iv: index, %acc: f32):
      scf.yield %iv, %acc : index, f32
    }
    return %r#1 : f32
}
