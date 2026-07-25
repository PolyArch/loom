// RUN: loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading %s | FileCheck %s --implicit-check-not=loom.spatial_region

// Freeze carries its typed poison policy through the selected graph boundary.
// CHECK-LABEL: dataflow.thread private @selected_freeze
// CHECK: dataflow.graph.launch @selected_freeze_graph
// CHECK-LABEL: dataflow.graph private @selected_freeze_graph
// CHECK: llvm.freeze

dataflow.thread private @selected_freeze(%input: i32) ctrl (%start: none) {
  %result = "loom.spatial_region"(%input)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        resultSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%value: i32):
      %stable = llvm.freeze %value : i32
      "loom.spatial_yield"(%stable)
          <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
  }) {graph_name = "selected_freeze_graph", source_maps = []} :
      (i32) -> i32
  dataflow.thread.yield
}
