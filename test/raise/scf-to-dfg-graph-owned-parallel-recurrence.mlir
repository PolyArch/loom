// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s

// A recurrence nested in graph-owned parallel syntax cannot be extracted until
// the structured program selects a concrete P[] representation. Keeping the
// recurrence in place prevents a nested graph.done from escaping to the
// enclosing thread.yield frontier.

// CHECK: error: 'scf.for' op cannot extract a recurrence nested in scf.forall/scf.parallel without a selected graph-owned P[] representation
// CHECK-LABEL: dataflow.thread private @parallel_recurrence
// CHECK: scf.forall
// CHECK: scf.for {{.*}} iter_args
// CHECK-NOT: dataflow.graph.launch @g_parallel_recurrence
// CHECK: dataflow.thread.yield

dataflow.thread private @parallel_recurrence(%n: index) ctrl (%ctrl: none) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.forall (%lane) in (%n) {
    %sum = scf.for %i = %zero to %n step %one iter_args(%state = %lane) -> (index) {
      %next = arith.addi %state, %i : index
      scf.yield %next : index
    }
  }
  dataflow.thread.yield
}
