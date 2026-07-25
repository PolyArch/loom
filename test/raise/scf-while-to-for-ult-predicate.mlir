// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// A post-tested (do-while) scf.while that continues while the bumped
// induction value is strictly below the loop-invariant upper bound is NOT
// mechanically equivalent to scf.for. The do-while body runs at least once
// even when %lb >= %ub, and the failed scf.condition forwards the bumped
// induction value, which overshoots %ub unless the positive step lands on
// %ub exactly. Proving trip-count and exit-value equivalence needs a
// loop-semantics analysis the mechanical raising pass does not own, so the
// loop must stay as scf.while.

// CHECK-LABEL: func.func @ult_do_while_kept
// CHECK: scf.while
// CHECK-NOT: scf.for {{.*}} iter_args
func.func @ult_do_while_kept(%buf: memref<?xi32>, %n: index) -> i32 {
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

// Observable-result drift anchor. The loop runs with %iv = 0, 3, 6; the
// failed scf.condition forwards %iv_n = 9 (the first bumped value not below
// %ub = 8). The bundled upstream reconstruction computes the last in-range
// value, %lb + (ceil((%ub - %lb) / %step) - 1) * %step = 6 -- not the upper
// bound 8 -- and neither equals the exact forwarded 9. Exact preservation
// therefore requires the scf.while to remain.

// CHECK-LABEL: func.func @ult_overshoot_observable_result_kept
// CHECK: scf.while
// CHECK: scf.condition(%{{.*}}) %{{.*}} : index
// CHECK: return %{{.*}} : index
// CHECK-NOT: scf.for
func.func @ult_overshoot_observable_result_kept() -> index {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %z = arith.constant 0 : i32
    %r:2 = scf.while (%iv = %c0, %acc = %z) : (index, i32) -> (index, i32) {
      %iv_n = arith.addi %iv, %c3 : index
      %cond = arith.cmpi ult, %iv_n, %c8 : index
      scf.condition(%cond) %iv_n, %acc : index, i32
    } do {
    ^bb0(%iv: index, %acc: i32):
      scf.yield %iv, %acc : index, i32
    }
    return %r#0 : index
}
