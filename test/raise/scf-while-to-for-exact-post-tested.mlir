// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// A finite latch-tested loop whose positive constant step lands exactly on a
// greater constant upper bound has the same body domain and exit state as
// scf.for. The induction result is exactly the upper bound, while the other
// state lane becomes the scf.for result.

// CHECK-LABEL: func.func @exact_latch_counted
// CHECK: %[[STATE:.*]] = scf.for %[[IV:.*]] = %{{.*}} to %[[UPPER:.*]] step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32)
// CHECK: %[[NEXT:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK-NOT: arith.cmpi
// CHECK: scf.yield %[[NEXT]] : i32
// CHECK: return %[[UPPER]], %[[STATE]] : i64, i32
// CHECK-NOT: scf.while
func.func @exact_latch_counted() -> (i64, i32) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %z = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %result:2 = scf.while (%iv = %c0, %acc = %z) : (i64, i32) -> (i64, i32) {
    %next_acc = arith.addi %acc, %one : i32
    %next_iv = arith.addi %iv, %c1 overflow<nsw, nuw> : i64
    %more = arith.cmpi ne, %next_iv, %c4 : i64
    scf.condition(%more) %next_iv, %next_acc : i64, i32
  } do {
  ^bb0(%iv: i64, %acc: i32):
    scf.yield %iv, %acc : i64, i32
  }
  return %result#0, %result#1 : i64, i32
}

// A non-landing positive step would wrap before equality and is not a finite
// scf.for domain. It remains sequential scf.while.

// CHECK-LABEL: func.func @nonlanding_kept
// CHECK: scf.while
// CHECK: arith.cmpi ne
// CHECK-NOT: scf.for
func.func @nonlanding_kept() {
  %c0 = arith.constant 0 : i64
  %c3 = arith.constant 3 : i64
  %c4 = arith.constant 4 : i64
  %result = scf.while (%iv = %c0) : (i64) -> i64 {
    %next_iv = arith.addi %iv, %c3 : i64
    %more = arith.cmpi ne, %next_iv, %c4 : i64
    scf.condition(%more) %next_iv : i64
  } do {
  ^bb0(%iv: i64):
    scf.yield %iv : i64
  }
  return
}

// A dominating positive guard closes the dynamic zero-based, unit-step
// domain. The widened bound is positive exactly when its source is positive.

// CHECK-LABEL: func.func @guarded_dynamic_uplifts
// CHECK: %[[POSITIVE:.*]] = arith.cmpi sgt, %arg0, %{{.*}} : i32
// CHECK: %[[BOTH:.*]] = arith.andi %[[POSITIVE]], %{{.*}} : i1
// CHECK: scf.if %[[BOTH]]
// CHECK: %[[UPPER:.*]] = arith.extui %arg0 nneg : i32 to i64
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[UPPER]] step %{{.*}}
// CHECK-NOT: scf.while
func.func @guarded_dynamic_uplifts(%n: i32, %other: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %positive = arith.cmpi sgt, %n, %c0_i32 : i32
  %other_positive = arith.cmpi sgt, %other, %c0_i32 : i32
  %both = arith.andi %positive, %other_positive : i1
  scf.if %both {
    %upper = arith.extui %n nneg : i32 to i64
    %result = scf.while (%iv = %c0_i64) : (i64) -> i64 {
      %next_iv = arith.addi %iv, %c1_i64 overflow<nsw, nuw> : i64
      %more = arith.cmpi ne, %next_iv, %upper : i64
      scf.condition(%more) %next_iv : i64
    } do {
    ^bb0(%iv: i64):
      scf.yield %iv : i64
    }
  }
  return
}

// Without a dominating positivity proof, zero makes the post-tested source
// execute through wrap while scf.for would execute no iterations.

// CHECK-LABEL: func.func @unguarded_dynamic_kept
// CHECK: scf.while
// CHECK-NOT: scf.for
func.func @unguarded_dynamic_kept(%upper: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %result = scf.while (%iv = %c0) : (i64) -> i64 {
    %next_iv = arith.addi %iv, %c1 overflow<nsw, nuw> : i64
    %more = arith.cmpi ne, %next_iv, %upper : i64
    scf.condition(%more) %next_iv : i64
  } do {
  ^bb0(%iv: i64):
    scf.yield %iv : i64
  }
  return
}
