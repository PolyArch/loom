// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/success.mlir | FileCheck %s --check-prefix=SUCCESS
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/failure.mlir 2>&1 | FileCheck %s --check-prefix=FAILURE
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/channel.mlir 2>&1 | FileCheck %s --check-prefix=CHANNEL

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
// FAILURE-COUNT-2: "loom.spatial_region"
// FAILURE-NOT: dataflow.graph private
// FAILURE-NOT: dataflow.graph.launch

// CHANNEL: stream endpoint conversion is not implemented; spatial candidate cannot be published
// CHANNEL: "loom.spatial_region"
// CHANNEL-NOT: dataflow.graph private
// CHANNEL-NOT: dataflow.graph.launch

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
  dataflow.channel.send %channel, %message : !dataflow.channel<i32>
  dataflow.thread.yield
}
