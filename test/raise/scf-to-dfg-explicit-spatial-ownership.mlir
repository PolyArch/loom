// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s --implicit-check-not=@g_host_container_0 --implicit-check-not=@g_instruction_only_0

// CHECK-LABEL: func.func @host_container
// CHECK: scf.for
// CHECK-NOT: dataflow.graph
// CHECK: return

// CHECK-LABEL: dataflow.thread private @instruction_only
// CHECK-NOT: dataflow.graph.launch
// CHECK: memref.store
// CHECK: dataflow.thread.yield

// CHECK-LABEL: dataflow.thread private @selected_spatial
// CHECK: dataflow.graph.launch @selected_graph
// CHECK: dataflow.thread.yield

// CHECK-LABEL: dataflow.graph private @selected_graph
// CHECK: dataflow.store
// CHECK: dataflow.graph.return
// CHECK-NOT: loom.spatial_region

func.func @host_container(%target: memref<4xi32>, %value: i32) {
  %zero = arith.constant 0 : index
  %four = arith.constant 4 : index
  %one = arith.constant 1 : index
  scf.for %index = %zero to %four step %one {
    memref.store %value, %target[%index] : memref<4xi32>
  }
  return
}

dataflow.thread private @instruction_only(
    %target: memref<1xi32>, %value: i32) ctrl (%ctrl: none) {
  %zero = arith.constant 0 : index
  memref.store %value, %target[%zero] : memref<1xi32>
  dataflow.thread.yield
}

dataflow.thread private @selected_spatial(
    %target: memref<1xi32>, %value: i32) ctrl (%ctrl: none) {
  "loom.spatial_region"(%value, %target)
      <{operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%payload: i32, %memory: memref<1xi32>):
      %zero = arith.constant 0 : index
      memref.store %payload, %memory[%zero] : memref<1xi32>
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "selected_graph", source_maps = []} :
      (i32, memref<1xi32>) -> ()
  dataflow.thread.yield
}
