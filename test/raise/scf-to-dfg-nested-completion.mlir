// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s

// CHECK-LABEL: dataflow.thread private @nested_reduction
// CHECK: %[[BRANCH_DONE:.*]] = scf.if %{{.*}} -> (none)
// CHECK: %{{.*}}, %[[GRAPH_DONE:.*]] = dataflow.graph.launch
// CHECK: scf.yield %[[GRAPH_DONE]] : none
// CHECK: else
// CHECK: scf.yield %{{.*}} : none
// CHECK: dataflow.thread.yield %[[BRANCH_DONE]] : none
// CHECK-LABEL: dataflow.graph private @g_nested_reduction_0
// CHECK-NOT: scf.

dataflow.thread private @nested_reduction(
    %enabled: i1, %limit: index) ctrl (%start: none) {
  scf.if %enabled {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %initial = arith.constant 0 : i32
    %sum = scf.for %index = %zero to %limit step %one
        iter_args(%state = %initial) -> (i32) {
      %next = arith.addi %state, %state : i32
      scf.yield %next : i32
    }
  } else {
  }
  dataflow.thread.yield
}
