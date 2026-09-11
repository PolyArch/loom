// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// A zero-based unit-step latch-tested loop with a dynamic bound has an exact
// counted domain when its unit update cannot wrap: a wrapping update would be
// poison at the latch. A signed contract yields a signed scf.for, an unsigned
// contract alone yields an unsigned one, and no contract keeps scf.while.

// CHECK-LABEL: func.func @signed_no_wrap
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[UPPER:.*]] step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i64)
// CHECK-NOT: unsigned
// CHECK: scf.yield
// CHECK: return %[[UPPER]], %{{.*}} : i64, i64
func.func @signed_no_wrap(%upper: i64) -> (i64, i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %result:2 = scf.while (%iv = %c0, %acc = %c0) : (i64, i64) -> (i64, i64) {
    %next_acc = arith.addi %acc, %iv : i64
    %next_iv = arith.addi %iv, %c1 overflow<nsw> : i64
    %more = arith.cmpi ne, %next_iv, %upper : i64
    scf.condition(%more) %next_iv, %next_acc : i64, i64
  } do {
  ^bb0(%iv: i64, %acc: i64):
    scf.yield %iv, %acc : i64, i64
  }
  return %result#0, %result#1 : i64, i64
}

// CHECK-LABEL: func.func @unsigned_no_wrap
// CHECK: scf.for unsigned %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK-NOT: scf.while
func.func @unsigned_no_wrap(%upper: i64) -> (i64, i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %result:2 = scf.while (%iv = %c0, %acc = %c0) : (i64, i64) -> (i64, i64) {
    %next_acc = arith.addi %acc, %iv : i64
    %next_iv = arith.addi %iv, %c1 overflow<nuw> : i64
    %more = arith.cmpi ne, %next_iv, %upper : i64
    scf.condition(%more) %next_iv, %next_acc : i64, i64
  } do {
  ^bb0(%iv: i64, %acc: i64):
    scf.yield %iv, %acc : i64, i64
  }
  return %result#0, %result#1 : i64, i64
}

// CHECK-LABEL: func.func @wrapping_update
// CHECK: scf.while
// CHECK-NOT: scf.for
func.func @wrapping_update(%upper: i64) -> (i64, i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %result:2 = scf.while (%iv = %c0, %acc = %c0) : (i64, i64) -> (i64, i64) {
    %next_acc = arith.addi %acc, %iv : i64
    %next_iv = arith.addi %iv, %c1 : i64
    %more = arith.cmpi ne, %next_iv, %upper : i64
    scf.condition(%more) %next_iv, %next_acc : i64, i64
  } do {
  ^bb0(%iv: i64, %acc: i64):
    scf.yield %iv, %acc : i64, i64
  }
  return %result#0, %result#1 : i64, i64
}
