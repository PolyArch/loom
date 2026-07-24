// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/channel.mlir | FileCheck %s --check-prefix=SUCCESS
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/mixed-channel.mlir 2>&1 | FileCheck %s --check-prefix=FAILURE --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch

// SUCCESS-LABEL: dataflow.thread private @channel_sender
// SUCCESS: dataflow.graph.launch @channel_graph
// SUCCESS-LABEL: dataflow.graph private @channel_graph
// SUCCESS-SAME: -> i32
// SUCCESS: dataflow.sync
// SUCCESS-NOT: dataflow.channel.send
// SUCCESS-NOT: loom.spatial_region
// SUCCESS: dataflow.graph.return values() streams(%{{.*}} : i32)

// FAILURE: error: loom-lower-graph-memory: one stream binding cannot contain parallel endpoint sites without a deterministic merge
// FAILURE-LABEL: dataflow.thread private @channel_sender
// FAILURE: loom.spatial_region
// FAILURE: dataflow.channel.send
// FAILURE-LABEL: dataflow.thread private @parallel_channel_sender
// FAILURE: loom.spatial_region
// FAILURE: dataflow.channel.send

//--- channel.mlir
dataflow.thread private @channel_sender(
    %channel: !dataflow.channel<i32>, %message: i32) ctrl (%start: none) {
  "loom.spatial_region"(%message, %channel)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 1>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%payload: i32, %output: !dataflow.channel<i32>):
      dataflow.channel.send %output, %payload : !dataflow.channel<i32>
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "channel_graph", source_maps = []} :
      (i32, !dataflow.channel<i32>) -> ()
  dataflow.thread.yield
}

//--- mixed-channel.mlir
dataflow.thread private @channel_sender(
    %channel: !dataflow.channel<i32>, %message: i32) ctrl (%start: none) {
  "loom.spatial_region"(%message, %channel)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 1>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%payload: i32, %output: !dataflow.channel<i32>):
      dataflow.channel.send %output, %payload : !dataflow.channel<i32>
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "channel_graph", source_maps = []} :
      (i32, !dataflow.channel<i32>) -> ()
  dataflow.thread.yield
}

dataflow.thread private @parallel_channel_sender(
    %channel: !dataflow.channel<i32>, %message: i32) ctrl (%start: none) {
  "loom.spatial_region"(%message, %channel)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 1>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%payload: i32, %output: !dataflow.channel<i32>):
      scf.forall (%lane) in (2) {
        dataflow.channel.send %output, %payload : !dataflow.channel<i32>
      }
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "parallel_channel_graph", source_maps = []} :
      (i32, !dataflow.channel<i32>) -> ()
  dataflow.thread.yield
}
