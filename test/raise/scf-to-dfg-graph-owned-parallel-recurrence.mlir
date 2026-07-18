// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s

// A provenance-marked fixed-width forall is already graph-owned. Each static
// lane recursively lowers its inner recurrence, and the graph joins the lane
// exits before retirement.

// CHECK-LABEL: dataflow.thread private @parallel_recurrence
// CHECK: dataflow.graph.launch @g_parallel_recurrence_0
// CHECK: dataflow.thread.yield
// CHECK-LABEL: dataflow.thread private @selected_nested_recurrence
// CHECK: dataflow.graph.launch @g_selected_nested_recurrence_0
// CHECK: dataflow.thread.yield
// CHECK-LABEL: dataflow.graph private @g_parallel_recurrence_0
// CHECK: dataflow.stream
// CHECK: dataflow.carry
// CHECK: dataflow.store
// CHECK: dataflow.stream
// CHECK: dataflow.carry
// CHECK: dataflow.store
// CHECK: dataflow.sync
// CHECK-NOT: scf.
// CHECK: dataflow.graph.return

dataflow.thread private @parallel_recurrence(
    %n: index, %memory: memref<?xindex>) ctrl (%ctrl: none) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.forall (%lane) in (2) {
    %sum = scf.for %i = %zero to %n step %one iter_args(%state = %lane) -> (index) {
      %next = arith.addi %state, %i : index
      scf.yield %next : index
    }
    memref.store %sum, %memory[%lane] : memref<?xindex>
  } {loom.parallel_group = 3 : i64}
  dataflow.thread.yield
}

// A selected parallel group remains one graph candidate through enclosing
// selection and recursively owns a nested recurrence in each lane.
// CHECK-LABEL: dataflow.graph private @g_selected_nested_recurrence_0
// CHECK: dataflow.demux
// CHECK: dataflow.stream
// CHECK: dataflow.carry
// CHECK: dataflow.store
// CHECK: dataflow.stream
// CHECK: dataflow.carry
// CHECK: dataflow.store
// CHECK: dataflow.sync
// CHECK: dataflow.mux
// CHECK-NOT: scf.
// CHECK: dataflow.graph.return

dataflow.thread private @selected_nested_recurrence(
    %condition: i1, %n: index, %memory: memref<?xindex>) ctrl (%ctrl: none) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  scf.if %condition {
    scf.parallel (%lane) = (%zero) to (%two) step (%one) {
      %sum = scf.for %i = %zero to %n step %one
          iter_args(%state = %lane) -> (index) {
        %next = arith.addi %state, %i : index
        scf.yield %next : index
      }
      memref.store %sum, %memory[%lane] : memref<?xindex>
      scf.reduce
    } {loom.parallel_schedule, loom.parallel_group = 4 : i64}
  }
  dataflow.thread.yield
}
