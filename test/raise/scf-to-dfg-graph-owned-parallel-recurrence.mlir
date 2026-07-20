// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s

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
  "loom.spatial_region"(%n, %memory)
      <{operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%limit: index, %target: memref<?xindex>):
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.forall (%lane) in (2) {
        %sum = scf.for %i = %zero to %limit step %one
            iter_args(%state = %lane) -> (index) {
          %next = arith.addi %state, %i : index
          scf.yield %next : index
        }
        memref.store %sum, %target[%lane] : memref<?xindex>
      } {loom.parallel_group = 3 : i64}
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "g_parallel_recurrence_0", source_maps = []} :
      (index, memref<?xindex>) -> ()
  dataflow.thread.yield
}

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
  "loom.spatial_region"(%condition, %n, %memory)
      <{operandSegmentSizes = array<i32: 2, 0, 1, 0>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%selected: i1, %limit: index, %target: memref<?xindex>):
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      scf.if %selected {
        scf.parallel (%lane) = (%zero) to (%two) step (%one) {
          %sum = scf.for %i = %zero to %limit step %one
              iter_args(%state = %lane) -> (index) {
            %next = arith.addi %state, %i : index
            scf.yield %next : index
          }
          memref.store %sum, %target[%lane] : memref<?xindex>
          scf.reduce
        } {loom.parallel_schedule, loom.parallel_group = 4 : i64}
      }
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "g_selected_nested_recurrence_0", source_maps = []} :
      (i1, index, memref<?xindex>) -> ()
  dataflow.thread.yield
}
