// RUN: loom-raise-opt --loom-normalize-lifted-scf-exit %s | FileCheck %s
// RUN: loom-raise-opt --loom-normalize-lifted-scf-exit --loom-scf-while-to-for %s | FileCheck %s --check-prefix=UPLIFT
// RUN: loom-raise-opt --mlir-print-debuginfo --loom-normalize-lifted-scf-exit %s | FileCheck %s --check-prefix=LOCATION

// CHECK-LABEL: func.func @normalize_all_unused
// CHECK: %[[NEXT:.*]] = arith.addi
// CHECK-NEXT: %[[CONTINUE:.*]] = arith.cmpi ne, %[[NEXT]], %arg0
// CHECK-NEXT: scf.condition(%[[CONTINUE]]) %[[NEXT]]
// CHECK-NOT: scf.if
// CHECK-NOT: arith.trunci
// LOCATION-LABEL: func.func @normalize_all_unused
// LOCATION: %[[LOC_NEXT:.*]] = arith.addi
// LOCATION-NEXT: %[[LOC_CONTINUE:.*]] = arith.cmpi ne, %[[LOC_NEXT]], %arg0
// LOCATION-SAME: loc(#[[COMPARISON_LOC:loc[0-9]+]])
// LOCATION-NEXT: scf.condition(%[[LOC_CONTINUE]]) %[[LOC_NEXT]]
// LOCATION-SAME: loc(#[[CONDITION_LOC:loc[0-9]+]])
// LOCATION-DAG: #[[COMPARISON_LOC]] = loc("source-comparison")
// LOCATION-DAG: #[[CONDITION_LOC]] = loc("lifted-condition")
func.func @normalize_all_unused(%bound: i64) {
  %iv0 = arith.constant 0 : i64
  %step = arith.constant 1 : i64
  %flag0 = arith.constant 0 : i32
  %flag1 = arith.constant 1 : i32
  %iv_poison = ub.poison : i64
  %result = scf.while (%iv = %iv0) : (i64) -> i64 {
    %next = arith.addi %iv, %step : i64
    %exit = arith.cmpi eq, %next, %bound : i64 loc("source-comparison")
    %lifted:3 = scf.if %exit -> (i64, i32, i32) {
      scf.yield %iv_poison, %flag1, %flag0 : i64, i32, i32
    } else {
      scf.yield %next, %flag0, %flag1 : i64, i32, i32
    }
    %selector = arith.trunci %lifted#2 : i32 to i1
    scf.condition(%selector) %lifted#0 : i64 loc("lifted-condition")
  } do {
  ^bb0(%iv: i64):
    scf.yield %iv : i64
  }
  return
}

// The lift can put the continuation on the then edge. That polarity uses the
// original comparison directly instead of synthesizing its inverse.
// CHECK-LABEL: func.func @normalize_repeat_then
// CHECK: %[[THEN_NEXT:.*]] = arith.addi
// CHECK-NEXT: %[[THEN_CONTINUE:.*]] = arith.cmpi ne, %[[THEN_NEXT]], %arg0
// CHECK-NEXT: scf.condition(%[[THEN_CONTINUE]]) %[[THEN_NEXT]]
// CHECK-NOT: scf.if
// CHECK-NOT: arith.trunci
func.func @normalize_repeat_then(%bound: i64) {
  %iv0 = arith.constant 0 : i64
  %step = arith.constant 1 : i64
  %flag0 = arith.constant 0 : i32
  %flag1 = arith.constant 1 : i32
  %iv_poison = ub.poison : i64
  %result = scf.while (%iv = %iv0) : (i64) -> i64 {
    %next = arith.addi %iv, %step : i64
    %continue = arith.cmpi ne, %next, %bound : i64
    %lifted:3 = scf.if %continue -> (i64, i32, i32) {
      scf.yield %next, %flag0, %flag1 : i64, i32, i32
    } else {
      scf.yield %iv_poison, %flag1, %flag0 : i64, i32, i32
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

// The loop-value results must feed scf.condition in positional order. This
// near-match preserves the exact result suffix and swaps only two equal-typed
// condition operands.
// CHECK-LABEL: func.func @preserve_swapped_condition_values
// CHECK: scf.while
// CHECK: %[[SWAPPED_CONDITION:.*]]:4 = scf.if
// CHECK: %[[SWAPPED_SELECTOR:.*]] = arith.trunci %[[SWAPPED_CONDITION]]#3
// CHECK-NEXT: scf.condition(%[[SWAPPED_SELECTOR]]) %[[SWAPPED_CONDITION]]#1, %[[SWAPPED_CONDITION]]#0
// UPLIFT-LABEL: func.func @preserve_swapped_condition_values
// UPLIFT: scf.while
// UPLIFT: %[[SWAPPED_CONDITION:.*]]:4 = scf.if
// UPLIFT: %[[SWAPPED_SELECTOR:.*]] = arith.trunci %[[SWAPPED_CONDITION]]#3
// UPLIFT-NEXT: scf.condition(%[[SWAPPED_SELECTOR]]) %[[SWAPPED_CONDITION]]#1, %[[SWAPPED_CONDITION]]#0
func.func @preserve_swapped_condition_values(%bound: i64,
                                              %output: !llvm.ptr) {
  %lhs0 = arith.constant 0 : i64
  %rhs0 = arith.constant 1 : i64
  %step = arith.constant 1 : i64
  %flag0 = arith.constant 0 : i32
  %flag1 = arith.constant 1 : i32
  %lhs_poison = ub.poison : i64
  %rhs_poison = ub.poison : i64
  %result:2 = scf.while (%lhs = %lhs0, %rhs = %rhs0)
      : (i64, i64) -> (i64, i64) {
    llvm.store %flag0, %output : i32, !llvm.ptr
    %lhs_next = arith.addi %lhs, %step : i64
    %rhs_next = arith.addi %rhs, %step : i64
    %exit = arith.cmpi eq, %lhs_next, %bound : i64
    %lifted:4 = scf.if %exit -> (i64, i64, i32, i32) {
      scf.yield %lhs_poison, %rhs_poison, %flag1, %flag0
          : i64, i64, i32, i32
    } else {
      scf.yield %lhs_next, %rhs_next, %flag0, %flag1
          : i64, i64, i32, i32
    }
    %selector = arith.trunci %lifted#3 : i32 to i1
    scf.condition(%selector) %lifted#1, %lifted#0 : i64, i64
  } do {
  ^bb0(%lhs: i64, %rhs: i64):
    scf.yield %lhs, %rhs : i64, i64
  }
  return
}

// The lift scaffold's after region is the identity continuation emitted by
// createStructuredDoWhileLoopOp(): one block whose only op is an scf.yield
// forwarding every after-block argument. This case anchors the single-op
// boundary: an llvm.store makes the region nontrivial even though its yield is
// otherwise an identity.
// CHECK-LABEL: func.func @preserve_nontrivial_after_region
// CHECK: scf.while
// CHECK: %[[LIFTED:.*]]:3 = scf.if
// CHECK: %[[SELECTOR:.*]] = arith.trunci %[[LIFTED]]#2
// CHECK-NEXT: scf.condition(%[[SELECTOR]]) %[[LIFTED]]#0
// CHECK: llvm.store
// CHECK: scf.yield %{{.*}} : i64
// UPLIFT-LABEL: func.func @preserve_nontrivial_after_region
// UPLIFT: scf.while
// UPLIFT: %[[LIFTED:.*]]:3 = scf.if
// UPLIFT: %[[SELECTOR:.*]] = arith.trunci %[[LIFTED]]#2
// UPLIFT-NEXT: scf.condition(%[[SELECTOR]]) %[[LIFTED]]#0
// UPLIFT: llvm.store
// UPLIFT: scf.yield %{{.*}} : i64
func.func @preserve_nontrivial_after_region(%bound: i64, %output: !llvm.ptr) {
  %iv0 = arith.constant 0 : i64
  %step = arith.constant 1 : i64
  %flag0 = arith.constant 0 : i32
  %flag1 = arith.constant 1 : i32
  %stored = arith.constant 0 : i32
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
    llvm.store %stored, %output : i32, !llvm.ptr
    scf.yield %iv : i64
  }
  return
}

// A sole after-region scf.yield must also preserve argument order. Swapping
// equal-typed block arguments is not the lift-owned identity continuation.
// CHECK-LABEL: func.func @preserve_swapped_after_values
// CHECK: scf.while
// CHECK: %[[SWAPPED_AFTER:.*]]:4 = scf.if
// CHECK: %[[SWAPPED_AFTER_SELECTOR:.*]] = arith.trunci %[[SWAPPED_AFTER]]#3
// CHECK-NEXT: scf.condition(%[[SWAPPED_AFTER_SELECTOR]]) %[[SWAPPED_AFTER]]#0, %[[SWAPPED_AFTER]]#1
// CHECK: ^bb0(%[[AFTER_LHS:.*]]: i64, %[[AFTER_RHS:.*]]: i64):
// CHECK-NEXT: scf.yield %[[AFTER_RHS]], %[[AFTER_LHS]] : i64, i64
// UPLIFT-LABEL: func.func @preserve_swapped_after_values
// UPLIFT: scf.while
// UPLIFT: %[[SWAPPED_AFTER:.*]]:4 = scf.if
// UPLIFT: %[[SWAPPED_AFTER_SELECTOR:.*]] = arith.trunci %[[SWAPPED_AFTER]]#3
// UPLIFT-NEXT: scf.condition(%[[SWAPPED_AFTER_SELECTOR]]) %[[SWAPPED_AFTER]]#0, %[[SWAPPED_AFTER]]#1
// UPLIFT: ^bb0(%[[AFTER_LHS:.*]]: i64, %[[AFTER_RHS:.*]]: i64):
// UPLIFT-NEXT: scf.yield %[[AFTER_RHS]], %[[AFTER_LHS]] : i64, i64
func.func @preserve_swapped_after_values(%bound: i64, %output: !llvm.ptr) {
  %lhs0 = arith.constant 0 : i64
  %rhs0 = arith.constant 1 : i64
  %step = arith.constant 1 : i64
  %flag0 = arith.constant 0 : i32
  %flag1 = arith.constant 1 : i32
  %lhs_poison = ub.poison : i64
  %rhs_poison = ub.poison : i64
  %result:2 = scf.while (%lhs = %lhs0, %rhs = %rhs0)
      : (i64, i64) -> (i64, i64) {
    llvm.store %flag0, %output : i32, !llvm.ptr
    %lhs_next = arith.addi %lhs, %step : i64
    %rhs_next = arith.addi %rhs, %step : i64
    %exit = arith.cmpi eq, %lhs_next, %bound : i64
    %lifted:4 = scf.if %exit -> (i64, i64, i32, i32) {
      scf.yield %lhs_poison, %rhs_poison, %flag1, %flag0
          : i64, i64, i32, i32
    } else {
      scf.yield %lhs_next, %rhs_next, %flag0, %flag1
          : i64, i64, i32, i32
    }
    %selector = arith.trunci %lifted#3 : i32 to i1
    scf.condition(%selector) %lifted#0, %lifted#1 : i64, i64
  } do {
  ^bb0(%lhs: i64, %rhs: i64):
    scf.yield %rhs, %lhs : i64, i64
  }
  return
}
