// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// Counted do-while loops that continue while the bumped induction value
// is strictly below the loop-invariant upper bound have the same half-open
// trip space as scf.for [%lb, %ub) when the positive step reaches %ub.

// CHECK-LABEL: func.func @ult_predicate_lifts
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK-NOT: scf.while
func.func @ult_predicate_lifts(%buf: memref<?xi32>, %n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %z = arith.constant 0 : i32
    %r:2 = scf.while (%iv = %c0, %acc = %z) : (index, i32) -> (index, i32) {
      %v = memref.load %buf[%iv] : memref<?xi32>
      %sum = arith.addi %acc, %v : i32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi ult, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum : index, i32
    } do {
    ^bb0(%iv: index, %acc: i32):
      scf.yield %iv, %acc : index, i32
    }
    return %r#1 : i32
}
