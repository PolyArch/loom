// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s

// CHECK: #[[SOURCE_MAP:.*]] = affine_map<() -> ()>
// CHECK-LABEL: dataflow.thread private @stream_producer
// CHECK: dataflow.graph.launch @producer_graph
// CHECK-SAME: stream_inputs()
// CHECK-SAME: stream_outputs(%arg0)
// CHECK-LABEL: dataflow.thread private @stream_consumer
// CHECK: dataflow.graph.launch @consumer_graph
// CHECK-SAME: stream_inputs(%arg0 source_map #[[SOURCE_MAP]])
// CHECK-SAME: stream_outputs()
// CHECK-LABEL: dataflow.thread private @loop_stream_producer
// CHECK: dataflow.graph.launch @loop_producer_graph
// CHECK-SAME: stream_outputs(%arg2)
// CHECK-LABEL: dataflow.thread private @loop_stream_consumer
// CHECK: dataflow.graph.launch @loop_consumer_graph
// CHECK-SAME: stream_inputs(%arg2 source_map #[[SOURCE_MAP]])
// CHECK-LABEL: dataflow.graph private @producer_graph(
// CHECK-SAME: %{{.*}}: none, %{{.*}}: i32) -> i32
// CHECK-SAME: input_segments = array<i32: 1, 0, 0>
// CHECK-SAME: result_segments = array<i32: 0, 1, 0>
// CHECK: dataflow.sync
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return values() streams(%{{.*}} : i32)
// CHECK-LABEL: dataflow.graph private @consumer_graph(
// CHECK-SAME: %{{.*}}: none, %{{.*}}: i32, %{{.*}}: memref<1xi32>) -> ()
// CHECK-SAME: input_segments = array<i32: 0, 1, 1>
// CHECK-SAME: result_segments = array<i32: 0, 0, 0>
// CHECK: dataflow.sync
// CHECK: dataflow.store
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return
// CHECK-LABEL: dataflow.graph private @loop_producer_graph(
// CHECK-SAME: -> i32
// CHECK-SAME: input_segments = array<i32: 1, 0, 1>
// CHECK-SAME: result_segments = array<i32: 0, 1, 0>
// CHECK: dataflow.stream
// CHECK: dataflow.carry
// CHECK: dataflow.load
// CHECK: dataflow.sync
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return values() streams(%{{.*}} : i32)
// CHECK-LABEL: dataflow.graph private @loop_consumer_graph(
// CHECK-SAME: input_segments = array<i32: 1, 1, 1>
// CHECK-SAME: result_segments = array<i32: 0, 0, 0>
// CHECK: dataflow.stream
// CHECK: dataflow.carry
// CHECK: dataflow.sync
// CHECK: dataflow.store
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return
// CHECK-NOT: loom.spatial_region

module {
  dataflow.thread private @stream_producer(
      %output: !dataflow.channel<i32>, %message: i32) ctrl (%ctrl: none) {
    "loom.spatial_region"(%message, %output)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%payload: i32, %channel: !dataflow.channel<i32>):
        dataflow.channel.send %channel, %payload
            : !dataflow.channel<i32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "producer_graph", source_maps = []} :
        (i32, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @stream_consumer(
      %input: !dataflow.channel<i32>, %memory: memref<1xi32>)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%input, %memory)
        <{operandSegmentSizes = array<i32: 0, 1, 1, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%channel: !dataflow.channel<i32>, %target: memref<1xi32>):
        %message = dataflow.channel.receive %channel
            : !dataflow.channel<i32>
        %zero = arith.constant 0 : index
        memref.store %message, %target[%zero] : memref<1xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {
      graph_name = "consumer_graph",
      source_maps = [affine_map<() -> ()>]
    } : (!dataflow.channel<i32>, memref<1xi32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @loop_stream_producer(
      %count: index, %source: memref<?xi32>,
      %output: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    "loom.spatial_region"(%count, %source, %output)
        <{operandSegmentSizes = array<i32: 1, 0, 1, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%limit: index, %memory: memref<?xi32>,
           %channel: !dataflow.channel<i32>):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        scf.for %index = %zero to %limit step %one {
          %message = memref.load %memory[%index] : memref<?xi32>
          dataflow.channel.send %channel, %message
              : !dataflow.channel<i32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "loop_producer_graph", source_maps = []} :
        (index, memref<?xi32>, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @loop_stream_consumer(
      %count: index, %target: memref<?xi32>,
      %input: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    "loom.spatial_region"(%count, %input, %target)
        <{operandSegmentSizes = array<i32: 1, 1, 1, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%limit: index, %channel: !dataflow.channel<i32>,
           %memory: memref<?xi32>):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        scf.for %index = %zero to %limit step %one {
          %message = dataflow.channel.receive %channel
              : !dataflow.channel<i32>
          memref.store %message, %memory[%index] : memref<?xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {
      graph_name = "loop_consumer_graph",
      source_maps = [affine_map<() -> ()>]
    } : (index, !dataflow.channel<i32>, memref<?xi32>) -> ()
    dataflow.thread.yield
  }

  func.func @bind_stream(
      %channel: !dataflow.channel<i32>, %message: i32,
      %memory: memref<1xi32>) {
    %producer = dataflow.thread.launch @stream_producer(%channel, %message)
        : (!dataflow.channel<i32>, i32) -> !dataflow.thread_token
    %consumer = dataflow.thread.launch @stream_consumer(%channel, %memory)
        : (!dataflow.channel<i32>, memref<1xi32>)
          -> !dataflow.thread_token
    return
  }
}
