// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// The upstream utility accepts any loop-invariant induction step, but an
// scf.for is only exact for a positive step: for %lb < %ub and step zero
// the source while does not terminate while the generated scf.for violates
// its semantic contract, and a runtime step of unproven sign may be zero or
// negative, making the two trip counts disagree. The pass therefore uplifts
// only a statically proven unit step. Both rejection cases
// below have dead results, so only the step gate -- not the
// no-external-users gate -- can be keeping them as scf.while. The
// unit-step sibling in the same callable still uplifts: candidacy is
// decided per loop.

// CHECK-LABEL: func.func @step_gate
// CHECK: scf.while
// CHECK: arith.cmpi slt
// CHECK: scf.while
// CHECK: arith.cmpi slt
// CHECK: scf.while
// CHECK: arith.cmpi slt
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NOT: scf.while
func.func @step_gate(%n: index, %step: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    // Step zero: the source while does not terminate for %n > 0, and an
    // scf.for with step zero violates the scf.for semantic contract.
    %zero:2 = scf.while (%iv = %c0, %acc = %c0) : (index, index) -> (index, index) {
      %cond = arith.cmpi slt, %iv, %n : index
      scf.condition(%cond) %iv, %acc : index, index
    } do {
    ^bb0(%iv: index, %acc: index):
      %iv_n = arith.addi %iv, %c0 : index
      %acc_n = arith.addi %acc, %c3 : index
      scf.yield %iv_n, %acc_n : index, index
    }
    // Step of unproven sign: %step is a runtime value that may be zero or
    // negative, so the source while and any generated scf.for need not have
    // the same trip count.
    %unknown:2 = scf.while (%iv = %c0, %acc = %c0) : (index, index) -> (index, index) {
      %cond = arith.cmpi slt, %iv, %n : index
      scf.condition(%cond) %iv, %acc : index, index
    } do {
    ^bb0(%iv: index, %acc: index):
      %iv_n = arith.addi %iv, %step : index
      %acc_n = arith.addi %acc, %c3 : index
      scf.yield %iv_n, %acc_n : index, index
    }
    // A positive non-unit step can still wrap before reaching a dynamic upper
    // bound. This concrete i8 loop wraps from 126 to -128 and does not
    // terminate, so it cannot become an scf.for.
    %c126 = arith.constant 126 : i8
    %c127 = arith.constant 127 : i8
    %c2_i8 = arith.constant 2 : i8
    %wrapping = scf.while (%iv = %c126) : (i8) -> i8 {
      %cond = arith.cmpi slt, %iv, %c127 : i8
      scf.condition(%cond) %iv : i8
    } do {
    ^bb0(%iv: i8):
      %iv_n = arith.addi %iv, %c2_i8 : i8
      scf.yield %iv_n : i8
    }
    // A unit step cannot overflow while the signed less-than condition is
    // true, so this sibling has the exact scf.for trip count and uplifts. Its
    // carried arguments are forwarded
    // to scf.condition in reordered form, which the upstream utility also
    // accepts; the step proof must still find the induction add.
    %positive:2 = scf.while (%iv = %c0, %acc = %c0) : (index, index) -> (index, index) {
      %cond = arith.cmpi slt, %iv, %n : index
      scf.condition(%cond) %acc, %iv : index, index
    } do {
    ^bb0(%acc: index, %iv: index):
      %iv_n = arith.addi %iv, %c1 : index
      %acc_n = arith.addi %acc, %c3 : index
      scf.yield %iv_n, %acc_n : index, index
    }
    return
}
