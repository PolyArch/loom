// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch

// Freeze is preserved in S0, but it cannot enter a selected Spatial region
// until the canonical operation schema owns its exceptional-value transition.
// Candidate failure leaves the selected region and its imported operation
// intact and publishes no partial graph.
// CHECK: error: loom-lower-graph-memory: operation 'llvm.freeze' is not a registered canonical Dataflow actor or a supported graph-lowering operation
// CHECK-LABEL: dataflow.thread private @selected_freeze
// CHECK: loom.spatial_region
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
