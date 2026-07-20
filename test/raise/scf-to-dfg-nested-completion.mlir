// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s

// CHECK-LABEL: dataflow.thread private @nested_spatial
// CHECK: %[[BRANCH_DONE:.*]] = scf.if %{{.*}} -> (none)
// CHECK: %[[GRAPH_DONE:.*]] = dataflow.graph.launch @nested_graph
// CHECK: scf.yield %[[GRAPH_DONE]] : none
// CHECK: else
// CHECK: scf.yield %{{.*}} : none
// CHECK: dataflow.thread.yield %[[BRANCH_DONE]] : none
// CHECK-LABEL: dataflow.graph private @nested_graph
// CHECK: dataflow.store
// CHECK-NOT: scf.
// CHECK: dataflow.graph.return

dataflow.thread private @nested_spatial(
    %enabled: i1, %target: memref<1xi32>, %value: i32)
    ctrl (%start: none) {
  scf.if %enabled {
    "loom.spatial_region"(%value, %target)
        <{operandSegmentSizes = array<i32: 1, 0, 1, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%payload: i32, %memory: memref<1xi32>):
        %zero = arith.constant 0 : index
        memref.store %payload, %memory[%zero] : memref<1xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "nested_graph", source_maps = []} :
        (i32, memref<1xi32>) -> ()
  } else {
  }
  dataflow.thread.yield
}
