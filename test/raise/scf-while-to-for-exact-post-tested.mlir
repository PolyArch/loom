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
