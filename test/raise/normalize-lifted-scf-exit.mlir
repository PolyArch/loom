// RUN: loom-raise-opt --loom-normalize-lifted-scf-exit %s | FileCheck %s
// RUN: loom-raise-opt --loom-normalize-lifted-scf-exit --loom-scf-while-to-for %s | FileCheck %s --check-prefix=UPLIFT

// CHECK-LABEL: func.func @normalize_all_unused
// CHECK: %[[NEXT:.*]] = arith.addi
// CHECK-NEXT: %[[CONTINUE:.*]] = arith.cmpi ne, %[[NEXT]], %arg0
// CHECK-NEXT: scf.condition(%[[CONTINUE]]) %[[NEXT]]
// CHECK-NOT: scf.if
// CHECK-NOT: arith.trunci
func.func @normalize_all_unused(%bound: i64) {
  %iv0 = arith.constant 0 : i64
  %step = arith.constant 1 : i64
  %flag0 = arith.constant 0 : i32
  %flag1 = arith.constant 1 : i32
  %iv_poison = ub.poison : i64
  %result = scf.while (%iv = %iv0) : (i64) -> i64 {
    %next = arith.addi %iv, %step : i64
    %exit = arith.cmpi eq, %next, %bound : i64
    %lifted:3 = scf.if %exit -> (i64, i32, i32) {
      scf.yield %iv_poison, %flag1, %flag0 : i64, i32, i32
    } else {
      scf.yield %next, %flag0, %flag1 : i64, i32, i32
    }
    %selector = arith.trunci %lifted#2 : i32 to i1
    scf.condition(%selector) %lifted#0 : i64
  } do {
  ^bb0(%iv: i64):
    scf.yield %iv : i64
  }
  return
}

// A live while result carries poison on the exit edge. The normalizer must
// leave the entire scaffold intact instead of manufacturing a non-poison
// result or exposing it to counted-loop uplift.
// CHECK-LABEL: func.func @preserve_live_exit_value
// CHECK: %[[LIVE_POISON:.*]] = ub.poison : i64
// CHECK: %[[LOOP:.*]] = scf.while
// CHECK: %[[LIFTED:.*]]:3 = scf.if
// CHECK: scf.yield %[[LIVE_POISON]],
// CHECK: %[[SELECTOR:.*]] = arith.trunci %[[LIFTED]]#2
// CHECK-NEXT: scf.condition(%[[SELECTOR]]) %[[LIFTED]]#0
// CHECK: return %[[LOOP]]
// UPLIFT-LABEL: func.func @preserve_live_exit_value
// UPLIFT: %[[LIVE_POISON:.*]] = ub.poison : i64
// UPLIFT: %[[LOOP:.*]] = scf.while
// UPLIFT: %[[LIFTED:.*]]:3 = scf.if
// UPLIFT: scf.yield %[[LIVE_POISON]],
// UPLIFT: %[[SELECTOR:.*]] = arith.trunci %[[LIFTED]]#2
// UPLIFT-NEXT: scf.condition(%[[SELECTOR]]) %[[LIFTED]]#0
// UPLIFT: return %[[LOOP]]
func.func @preserve_live_exit_value(%bound: i64) -> i64 {
  %iv0 = arith.constant 0 : i64
  %step = arith.constant 1 : i64
  %flag0 = arith.constant 0 : i32
  %flag1 = arith.constant 1 : i32
  %iv_poison = ub.poison : i64
  %result = scf.while (%iv = %iv0) : (i64) -> i64 {
    %next = arith.addi %iv, %step : i64
    %exit = arith.cmpi eq, %next, %bound : i64
    %lifted:3 = scf.if %exit -> (i64, i32, i32) {
      scf.yield %iv_poison, %flag1, %flag0 : i64, i32, i32
    } else {
      scf.yield %next, %flag0, %flag1 : i64, i32, i32
    }
    %selector = arith.trunci %lifted#2 : i32 to i1
    scf.condition(%selector) %lifted#0 : i64
  } do {
  ^bb0(%iv: i64):
    scf.yield %iv : i64
  }
  return %result : i64
}

// The pinned lift layout is [loop values..., discriminator, shouldRepeat].
// Reordering those results is a near-match, not a lift-owned scaffold.
// CHECK-LABEL: func.func @preserve_reordered_suffix
// CHECK: %[[REORDERED_POISON:.*]] = ub.poison : i64
// CHECK: scf.while
// CHECK: %[[REORDERED:.*]]:3 = scf.if
// CHECK: scf.yield %{{.*}}, %[[REORDERED_POISON]],
// CHECK: %[[REORDERED_SELECTOR:.*]] = arith.trunci %[[REORDERED]]#0
// CHECK-NEXT: scf.condition(%[[REORDERED_SELECTOR]]) %[[REORDERED]]#1
// UPLIFT-LABEL: func.func @preserve_reordered_suffix
// UPLIFT: %[[REORDERED_POISON:.*]] = ub.poison : i64
// UPLIFT: scf.while
// UPLIFT: %[[REORDERED:.*]]:3 = scf.if
// UPLIFT: scf.yield %{{.*}}, %[[REORDERED_POISON]],
// UPLIFT: %[[REORDERED_SELECTOR:.*]] = arith.trunci %[[REORDERED]]#0
// UPLIFT-NEXT: scf.condition(%[[REORDERED_SELECTOR]]) %[[REORDERED]]#1
func.func @preserve_reordered_suffix(%bound: i64, %output: !llvm.ptr) {
  %iv0 = arith.constant 0 : i64
  %step = arith.constant 1 : i64
  %flag0 = arith.constant 0 : i32
  %flag1 = arith.constant 1 : i32
  %stored = arith.constant 0 : i32
  %iv_poison = ub.poison : i64
  %result = scf.while (%iv = %iv0) : (i64) -> i64 {
    llvm.store %stored, %output : i32, !llvm.ptr
    %next = arith.addi %iv, %step : i64
    %exit = arith.cmpi eq, %next, %bound : i64
    %reordered:3 = scf.if %exit -> (i32, i64, i32) {
      scf.yield %flag0, %iv_poison, %flag1 : i32, i64, i32
    } else {
      scf.yield %flag1, %next, %flag0 : i32, i64, i32
    }
    %selector = arith.trunci %reordered#0 : i32 to i1
    scf.condition(%selector) %reordered#1 : i64
  } do {
  ^bb0(%iv: i64):
    scf.yield %iv : i64
  }
  return
}
