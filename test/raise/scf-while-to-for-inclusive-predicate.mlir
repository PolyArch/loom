// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// The do-while uplift only matches the EXACT shape upstream
// `--lift-cf-to-scf` emits for an LLVM counted loop with the increment
// in the latch: an `arith.cmpi ne %iv_next, %ub`. Inclusive predicates
// (sle / ule / sge / uge) and reversed predicates (sgt / ugt / slt /
// ult on iv_next) MUST NOT trigger the rewrite -- doing so without
// adjusting lb / ub or the iteration direction would change the trip
// count.

// CHECK-LABEL: func.func @sle_predicate_kept
// CHECK: scf.while
// CHECK-NOT: scf.for {{.*}} iter_args
func.func @sle_predicate_kept(%buf: memref<?xf32>,
                              %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %r:2 = scf.while (%iv = %c0, %acc = %f0) : (index, f32) -> (index, f32) {
      %v = memref.load %buf[%iv] : memref<?xf32>
      %sum = arith.addf %acc, %v : f32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi sle, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum : index, f32
    } do {
    ^bb0(%iv: index, %acc: f32):
      scf.yield %iv, %acc : index, f32
    }
    return %r#1 : f32
}

// CHECK-LABEL: func.func @sgt_predicate_kept
// CHECK: scf.while
// CHECK-NOT: scf.for {{.*}} iter_args
func.func @sgt_predicate_kept(%buf: memref<?xf32>,
                              %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %r:2 = scf.while (%iv = %c0, %acc = %f0) : (index, f32) -> (index, f32) {
      %v = memref.load %buf[%iv] : memref<?xf32>
      %sum = arith.addf %acc, %v : f32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi sgt, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum : index, f32
    } do {
    ^bb0(%iv: index, %acc: f32):
      scf.yield %iv, %acc : index, f32
    }
    return %r#1 : f32
}
