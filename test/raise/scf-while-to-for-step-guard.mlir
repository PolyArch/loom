// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// The upstream utility accepts any loop-invariant induction step, but an
// scf.for is only exact for a positive step: for %lb < %ub and step zero
// the source while does not terminate while the generated scf.for violates
// its semantic contract, and a runtime step of unproven sign may be zero or
// negative, making the two trip counts disagree. The pass therefore uplifts
// only a statically proven positive constant step. Both rejection cases
// below have dead results, so only the step gate -- not the
// no-external-users gate -- can be keeping them as scf.while. The
// positive-step sibling in the same callable still uplifts: candidacy is
// decided per loop.

// CHECK-LABEL: func.func @step_gate
// CHECK: scf.while
// CHECK: arith.cmpi slt
// CHECK: scf.while
// CHECK: arith.cmpi slt
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NOT: scf.while
func.func @step_gate(%n: index, %step: index) {
    %c0 = arith.constant 0 : index
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
    // Positive constant step: the trip count is the exact scf.for trip
    // count, so this sibling uplifts. Its carried arguments are forwarded
    // to scf.condition in reordered form, which the upstream utility also
    // accepts; the step proof must still find the induction add.
    %positive:2 = scf.while (%iv = %c0, %acc = %c0) : (index, index) -> (index, index) {
      %cond = arith.cmpi slt, %iv, %n : index
      scf.condition(%cond) %acc, %iv : index, index
    } do {
    ^bb0(%acc: index, %iv: index):
      %iv_n = arith.addi %iv, %c3 : index
      %acc_n = arith.addi %acc, %c3 : index
      scf.yield %iv_n, %acc_n : index, index
    }
    return
}
