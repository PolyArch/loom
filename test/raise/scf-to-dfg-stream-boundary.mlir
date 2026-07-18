// RUN: loom-raise-opt --loom-lower-for-to-graph %s > %t.lowered.mlir
// RUN: FileCheck %s < %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph branch_relay_graph --arg 0=true --arg 1=2 --arg 2=3 --arg 2=7 --arg 2=9 --arg 2=11 --arg 2=13 --arg 2=17 --output %t.branch-true.json
// RUN: FileCheck %s --check-prefix=BRANCH-TRUE < %t.branch-true.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph branch_relay_graph --arg 0=false --arg 1=2 --arg 2=3 --arg 2=5 --arg 2=6 --arg 2=17 --output %t.branch-false.json
// RUN: FileCheck %s --check-prefix=BRANCH-FALSE < %t.branch-false.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph branch_relay_graph --arg 0=true --arg 1=0 --arg 2=21 --arg 2=23 --output %t.branch-zero.json
// RUN: FileCheck %s --check-prefix=BRANCH-ZERO < %t.branch-zero.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph dependent_consumer_graph --arg 0=7 --arg 0=9 --memref 1=0,0 --output %t.dependent-true.json
// RUN: FileCheck %s --check-prefix=DEPENDENT-TRUE < %t.dependent-true.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph dependent_consumer_graph --arg 0=0 --memref 1=5,6 --output %t.dependent-false.json
// RUN: FileCheck %s --check-prefix=DEPENDENT-FALSE < %t.dependent-false.json

// BRANCH-TRUE: "final_stream_outputs": [
// BRANCH-TRUE-NEXT: [
// BRANCH-TRUE-NEXT: "i32:3",
// BRANCH-TRUE-NEXT: "i32:7",
// BRANCH-TRUE-NEXT: "i32:9",
// BRANCH-TRUE-NEXT: "i32:11",
// BRANCH-TRUE-NEXT: "i32:13",
// BRANCH-TRUE-NEXT: "i32:17"
// BRANCH-TRUE: "status": "pass"
// BRANCH-FALSE: "final_stream_outputs": [
// BRANCH-FALSE-NEXT: [
// BRANCH-FALSE-NEXT: "i32:3",
// BRANCH-FALSE-NEXT: "i32:5",
// BRANCH-FALSE-NEXT: "i32:6",
// BRANCH-FALSE-NEXT: "i32:17"
// BRANCH-FALSE: "status": "pass"
// BRANCH-ZERO: "final_stream_outputs": [
// BRANCH-ZERO-NEXT: [
// BRANCH-ZERO-NEXT: "i32:21",
// BRANCH-ZERO-NEXT: "i32:23"
// BRANCH-ZERO: "status": "pass"
// DEPENDENT-TRUE: "final_memory_state": {
// DEPENDENT-TRUE-NEXT: "arg1": [
// DEPENDENT-TRUE-NEXT: "i32:7",
// DEPENDENT-TRUE-NEXT: "i32:9"
// DEPENDENT-TRUE: "status": "pass"
// DEPENDENT-FALSE: "final_memory_state": {
// DEPENDENT-FALSE-NEXT: "arg1": [
// DEPENDENT-FALSE-NEXT: "i32:0",
// DEPENDENT-FALSE-NEXT: "i32:6"
// DEPENDENT-FALSE: "status": "pass"

// CHECK: #[[SOURCE_MAP:.*]] = affine_map<() -> ()>
// CHECK-LABEL: dataflow.thread private @stream_producer
// CHECK: dataflow.graph.launch @producer_graph
// CHECK-SAME: stream_inputs()
// CHECK-SAME: memories(%arg1)
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
// CHECK-LABEL: dataflow.thread private @sequential_stream_producer
// CHECK: dataflow.graph.launch @sequential_producer_graph
// CHECK-SAME: stream_outputs(%arg0)
// CHECK-LABEL: dataflow.thread private @sequential_stream_consumer
// CHECK: dataflow.graph.launch @sequential_consumer_graph
// CHECK-SAME: stream_inputs(%arg0 source_map #[[SOURCE_MAP]])
// CHECK-LABEL: dataflow.thread private @branch_stream_relay
// CHECK: dataflow.graph.launch @branch_relay_graph
// CHECK-SAME: stream_inputs(%arg0 source_map #[[SOURCE_MAP]])
// CHECK-SAME: stream_outputs(%arg1)
// CHECK-LABEL: dataflow.thread private @optional_stream_producer
// CHECK: dataflow.graph.launch @optional_producer_graph
// CHECK-SAME: stream_outputs(%arg0)
// CHECK-LABEL: dataflow.thread private @dependent_stream_consumer
// CHECK: dataflow.graph.launch @dependent_consumer_graph
// CHECK-SAME: stream_inputs(%arg0 source_map #[[SOURCE_MAP]])
// CHECK-LABEL: dataflow.graph private @producer_graph(
// CHECK-SAME: %{{.*}}: none, %[[PRODUCER_PAYLOAD:[[:alnum:]_]+]]: i32, %{{.*}}: memref<1xi32>) -> i32
// CHECK-SAME: input_segments = array<i32: 1, 0, 1>
// CHECK-SAME: result_segments = array<i32: 0, 1, 0>
// CHECK: %[[PRODUCER_STORE_DONE:[[:alnum:]_]+]] = dataflow.store
// CHECK: dataflow.sync %[[PRODUCER_STORE_DONE]], %[[PRODUCER_PAYLOAD]] : (none, i32) -> (none, i32)
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
// CHECK-LABEL: dataflow.graph private @sequential_producer_graph(
// CHECK-SAME: -> i32
// CHECK: dataflow.stream
// CHECK: %[[SEQUENCE_ORDINAL:[[:alnum:]_]+]] = arith.index_cast
// CHECK: dataflow.mux %[[SEQUENCE_ORDINAL]], %{{.*}}, %{{.*}}, %{{.*}} : (index, i32, i32, i32) -> i32
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return values() streams(%{{.*}} : i32)
// CHECK-LABEL: dataflow.graph private @sequential_consumer_graph(
// CHECK-SAME: %{{.*}}: none, %[[SEQUENCE_INPUT:[[:alnum:]_]+]]: i32, %{{.*}}: memref<2xi32>)
// CHECK: dataflow.stream
// CHECK: %[[SEQUENCE_LANE:[[:alnum:]_]+]] = arith.trunci
// CHECK: dataflow.demux %[[SEQUENCE_LANE]], %[[SEQUENCE_INPUT]] : (i1, i32) -> (i32, i32)
// CHECK: dataflow.store
// CHECK: dataflow.store
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return
// CHECK-LABEL: dataflow.graph private @branch_relay_graph(
// CHECK-SAME: %{{.*}}: none, %[[BRANCH_SELECT:[[:alnum:]_]+]]: i1, %{{.*}}: index, %[[BRANCH_INPUT:[[:alnum:]_]+]]: i32) -> i32
// CHECK: dataflow.stream
// CHECK: dataflow.demux %{{.*}}, %[[BRANCH_INPUT]] : (index, i32) -> (i32, i32, i32, i32, i32)
// CHECK: dataflow.stream
// CHECK: dataflow.invariant %{{.*}}, %[[BRANCH_SELECT]] : i1
// CHECK: dataflow.gate
// CHECK: dataflow.mux %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, i32, i32, i32, i32, i32) -> i32
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return values() streams(%{{.*}} : i32)
// CHECK-LABEL: dataflow.graph private @optional_producer_graph(
// CHECK: %{{.*}}, %[[OPTIONAL_PHASE:[[:alnum:]_]+]] = dataflow.stream
// CHECK: %[[OPTIONAL_EVENTS:[[:alnum:]_]+]]:2 = dataflow.demux %[[OPTIONAL_PHASE]], %{{.*}} : (i1, none) -> (none, none)
// CHECK: %[[OPTIONAL_ACTIVE:[[:alnum:]_]+]] = dataflow.mux %{{.*}}, %{{.*}}, %{{.*}} : (i1, i1, i1) -> i1
// CHECK: %[[OPTIONAL_ACTIVE_EVENTS:[[:alnum:]_]+]]:2 = dataflow.demux %[[OPTIONAL_ACTIVE]], %[[OPTIONAL_EVENTS]]#1 : (i1, none) -> (none, none)
// CHECK: dataflow.sync %[[OPTIONAL_ACTIVE_EVENTS]]#1, %{{.*}} : (none, i32) -> (none, i32)
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return values() streams(%{{.*}} : i32)
// CHECK-LABEL: dataflow.graph private @dependent_consumer_graph(
// CHECK-SAME: %{{.*}}: none, %[[DEPENDENT_INPUT:[[:alnum:]_]+]]: i32, %{{.*}}: memref<2xi32>)
// CHECK: %[[DEPENDENT_CONDITION:[[:alnum:]_]+]] = arith.cmpi ne, %[[FIRST_SYNC:[[:alnum:]_]+]]#1, %{{.*}} : i32
// CHECK: %[[DEPENDENT_IV:[[:alnum:]_]+]], %{{.*}} = dataflow.stream
// CHECK: %[[DEPENDENT_STATIC:[[:alnum:]_]+]] = arith.trunci %[[DEPENDENT_IV]] : i32 to i1
// CHECK: dataflow.mux %[[DEPENDENT_CONDITION]], %{{.*}}, %{{.*}} : (i1, i1, i1) -> i1
// CHECK: %[[DEPENDENT_ACTIVE:[[:alnum:]_]+]] = dataflow.mux %[[DEPENDENT_STATIC]], %{{.*}}, %{{.*}} : (i1, i1, i1) -> i1
// CHECK: %[[DEPENDENT_ACTIVE_ORDINALS:[[:alnum:]_]+]]:2 = dataflow.demux %[[DEPENDENT_ACTIVE]], %[[DEPENDENT_IV]] : (i1, i32) -> (i32, i32)
// CHECK: %[[DEPENDENT_ROUTE:[[:alnum:]_]+]] = arith.trunci %[[DEPENDENT_ACTIVE_ORDINALS]]#1 : i32 to i1
// CHECK: %[[DEPENDENT_LANES:[[:alnum:]_]+]]:2 = dataflow.demux %[[DEPENDENT_ROUTE]], %[[DEPENDENT_INPUT]] : (i1, i32) -> (i32, i32)
// CHECK: %[[FIRST_SYNC]]:2 = dataflow.sync %{{.*}}, %[[DEPENDENT_LANES]]#0 : (none, i32) -> (none, i32)
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return
// CHECK-NOT: loom.spatial_region

module {
  dataflow.thread private @stream_producer(
      %output: !dataflow.channel<i32>, %memory: memref<1xi32>, %message: i32)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%message, %memory, %output)
        <{operandSegmentSizes = array<i32: 1, 0, 1, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%payload: i32, %target: memref<1xi32>,
           %channel: !dataflow.channel<i32>):
        %zero = arith.constant 0 : index
        %stored = arith.constant 42 : i32
        memref.store %stored, %target[%zero] : memref<1xi32>
        dataflow.channel.send %channel, %payload
            : !dataflow.channel<i32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "producer_graph", source_maps = []} :
        (i32, memref<1xi32>, !dataflow.channel<i32>) -> ()
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
          %first = dataflow.channel.receive %channel
              : !dataflow.channel<i32>
          %second = dataflow.channel.receive %channel
              : !dataflow.channel<i32>
          memref.store %first, %memory[%index] : memref<?xi32>
          memref.store %second, %memory[%index] : memref<?xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {
      graph_name = "loop_consumer_graph",
      source_maps = [affine_map<() -> ()>]
    } : (index, !dataflow.channel<i32>, memref<?xi32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @sequential_stream_producer(
      %output: !dataflow.channel<i32>, %first: i32, %second: i32,
      %third: i32) ctrl (%ctrl: none) {
    "loom.spatial_region"(%first, %second, %third, %output)
        <{operandSegmentSizes = array<i32: 3, 0, 0, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%first_message: i32, %second_message: i32,
           %third_message: i32, %channel: !dataflow.channel<i32>):
        dataflow.channel.send %channel, %first_message
            : !dataflow.channel<i32>
        dataflow.channel.send %channel, %second_message
            : !dataflow.channel<i32>
        dataflow.channel.send %channel, %third_message
            : !dataflow.channel<i32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "sequential_producer_graph", source_maps = []} :
        (i32, i32, i32, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @sequential_stream_consumer(
      %input: !dataflow.channel<i32>, %memory: memref<2xi32>)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%input, %memory)
        <{operandSegmentSizes = array<i32: 0, 1, 1, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%channel: !dataflow.channel<i32>, %target: memref<2xi32>):
        %first = dataflow.channel.receive %channel
            : !dataflow.channel<i32>
        %second = dataflow.channel.receive %channel
            : !dataflow.channel<i32>
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        memref.store %first, %target[%zero] : memref<2xi32>
        memref.store %second, %target[%one] : memref<2xi32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {
      graph_name = "sequential_consumer_graph",
      source_maps = [affine_map<() -> ()>]
    } : (!dataflow.channel<i32>, memref<2xi32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @branch_stream_relay(
      %input: !dataflow.channel<i32>, %output: !dataflow.channel<i32>,
      %condition: i1, %count: index) ctrl (%ctrl: none) {
    "loom.spatial_region"(%condition, %count, %input, %output)
        <{operandSegmentSizes = array<i32: 2, 1, 0, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%select: i1, %limit: index,
           %source: !dataflow.channel<i32>,
           %sink: !dataflow.channel<i32>):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %prefix = dataflow.channel.receive %source
            : !dataflow.channel<i32>
        dataflow.channel.send %sink, %prefix : !dataflow.channel<i32>
        scf.for %iteration = %zero to %limit step %one {
          scf.if %select {
            %on_true_first = dataflow.channel.receive %source
                : !dataflow.channel<i32>
            dataflow.channel.send %sink, %on_true_first
                : !dataflow.channel<i32>
            %on_true_second = dataflow.channel.receive %source
                : !dataflow.channel<i32>
            dataflow.channel.send %sink, %on_true_second
                : !dataflow.channel<i32>
          } else {
            %on_false = dataflow.channel.receive %source
                : !dataflow.channel<i32>
            dataflow.channel.send %sink, %on_false : !dataflow.channel<i32>
          }
        }
        %suffix = dataflow.channel.receive %source
            : !dataflow.channel<i32>
        dataflow.channel.send %sink, %suffix : !dataflow.channel<i32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {
      graph_name = "branch_relay_graph",
      source_maps = [affine_map<() -> ()>]
    } : (i1, index, !dataflow.channel<i32>, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @optional_stream_producer(
      %output: !dataflow.channel<i32>, %condition: i1, %message: i32)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%condition, %message, %output)
        <{operandSegmentSizes = array<i32: 2, 0, 0, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%select: i1, %payload: i32,
           %sink: !dataflow.channel<i32>):
        scf.if %select {
          dataflow.channel.send %sink, %payload : !dataflow.channel<i32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "optional_producer_graph", source_maps = []} :
        (i1, i32, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @dependent_stream_consumer(
      %input: !dataflow.channel<i32>, %memory: memref<2xi32>)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%input, %memory)
        <{operandSegmentSizes = array<i32: 0, 1, 1, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%channel: !dataflow.channel<i32>, %target: memref<2xi32>):
        %first = dataflow.channel.receive %channel
            : !dataflow.channel<i32>
        %zero_value = arith.constant 0 : i32
        %condition = arith.cmpi ne, %first, %zero_value : i32
        %zero = arith.constant 0 : index
        memref.store %first, %target[%zero] : memref<2xi32>
        scf.if %condition {
          %second = dataflow.channel.receive %channel
              : !dataflow.channel<i32>
          %one = arith.constant 1 : index
          memref.store %second, %target[%one] : memref<2xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {
      graph_name = "dependent_consumer_graph",
      source_maps = [affine_map<() -> ()>]
    } : (!dataflow.channel<i32>, memref<2xi32>) -> ()
    dataflow.thread.yield
  }

  func.func @bind_stream(
      %channel: !dataflow.channel<i32>, %message: i32,
      %memory: memref<1xi32>) {
    %producer = dataflow.thread.launch
        @stream_producer(%channel, %memory, %message)
        : (!dataflow.channel<i32>, memref<1xi32>, i32)
          -> !dataflow.thread_token
    %consumer = dataflow.thread.launch @stream_consumer(%channel, %memory)
        : (!dataflow.channel<i32>, memref<1xi32>)
          -> !dataflow.thread_token
    return
  }
}
