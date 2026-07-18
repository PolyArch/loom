// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/success.mlir | FileCheck %s --check-prefix=SUCCESS
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/failure.mlir 2>&1 | FileCheck %s --check-prefix=FAILURE --implicit-check-not=loom.spatial_region --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/channel.mlir | FileCheck %s --check-prefix=CHANNEL

// SUCCESS-LABEL: dataflow.thread private @reduction
// SUCCESS: dataflow.graph.launch @g_reduction_0
// SUCCESS-LABEL: dataflow.graph private @g_reduction_0
// SUCCESS-NOT: scf.
// SUCCESS-NOT: cf.
// SUCCESS-NOT: loom.spatial_region
// SUCCESS: dataflow.stream
// SUCCESS: dataflow.load
// SUCCESS: dataflow.graph.return

// FAILURE: error: loom-lower-graph-memory: raw scf.forall requires a selected schedule and provenance before graph-region lowering
// FAILURE-LABEL: dataflow.thread private @valid_candidate
// FAILURE: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// FAILURE: dataflow.thread.yield
// FAILURE-LABEL: dataflow.thread private @parallel_candidate
// FAILURE: scf.forall
// FAILURE: memref.load
// FAILURE: memref.store
// FAILURE: dataflow.thread.yield

// CHANNEL-LABEL: dataflow.thread private @channel_sender
// CHANNEL: dataflow.graph.launch @g_channel_sender_0
// CHANNEL-LABEL: dataflow.graph private @g_channel_sender_0
// CHANNEL-SAME: -> i32
// CHANNEL: dataflow.sync
// CHANNEL-NOT: dataflow.channel.send
// CHANNEL-NOT: loom.spatial_region
// CHANNEL: dataflow.graph.return values() streams(%{{.*}} : i32)

//--- success.mlir
dataflow.thread private @reduction(%buffer: memref<?xi32>, %count: index)
    ctrl (%start: none) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %initial = arith.constant 0 : i32
  %sum = scf.for %index = %zero to %count step %one
      iter_args(%state = %initial) -> (i32) {
    %value = memref.load %buffer[%index] : memref<?xi32>
    %next = arith.addi %state, %value : i32
    scf.yield %next : i32
  }
  dataflow.thread.yield
}

//--- failure.mlir
dataflow.thread private @valid_candidate(%value: i32) ctrl (%start: none) {
  %sum = arith.addi %value, %value : i32
  dataflow.thread.yield
}

dataflow.thread private @parallel_candidate(%source: memref<?xi32>,
                                            %target: memref<?xi32>)
    ctrl (%start: none) iv (%base: index) {
  scf.forall (%lane) in (4) {
    %index = arith.addi %base, %lane : index
    %value = memref.load %source[%index] : memref<?xi32>
    memref.store %value, %target[%index] : memref<?xi32>
  }
  dataflow.thread.yield
}

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
  }) {graph_name = "g_channel_sender_0", source_maps = []} :
      (i32, !dataflow.channel<i32>) -> ()
  dataflow.thread.yield
}
