// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/repeated-coordinate.mlir | FileCheck %s --check-prefix=REPEATED
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/rate-conversion.mlir | FileCheck %s --check-prefix=RATE
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/coordinate-identity.mlir | FileCheck %s --check-prefix=IDENTITY
// RUN: loom-raise-opt --loom-lower-scf-to-dfg %t.dir/multicast.mlir | FileCheck %s --check-prefix=MULTICAST

//--- repeated-coordinate.mlir
module {
  dataflow.thread private @feedback domain(#dataflow.thread_domain<dense>)(
      %input_channel: !dataflow.channel<i32>,
      %output_channel: !dataflow.channel<i32>, %message: i32, %enabled: i1,
      %choose_received: i1, %choose_mux: i1) ctrl (%ctrl: none) {
    dataflow.channel.send %output_channel, %message
        : !dataflow.channel<i32>
    %next = scf.if %enabled -> (i32) {
      %received = "loom.spatial_region"(%input_channel)
          <{operandSegmentSizes = array<i32: 0, 1, 0, 0>,
            resultSegmentSizes = array<i32: 1, 0>}> ({
        ^bb0(%input: !dataflow.channel<i32>):
          %payload = dataflow.channel.receive %input
              : !dataflow.channel<i32>
          "loom.spatial_yield"(%payload)
              <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
      }) {
        graph_name = "feedback_graph",
        source_maps = [affine_map<() -> ()>]
      } : (!dataflow.channel<i32>) -> i32
      %selective = dataflow.mux %choose_mux, %message, %received
          : (i1, i32, i32) -> i32
      %selected = scf.if %choose_received -> (i32) {
        scf.yield %received : i32
      } else {
        scf.yield %selective : i32
      }
      dataflow.channel.send %output_channel, %selected
          : !dataflow.channel<i32>
      %doubled = arith.addi %message, %message : i32
      scf.yield %doubled : i32
    } else {
      scf.yield %message : i32
    }
    dataflow.channel.send %output_channel, %next
        : !dataflow.channel<i32>
    dataflow.thread.yield
  }

  func.func @repeat_feedback(
      %channel: !dataflow.channel<i32>, %message: i32, %enabled: i1,
      %choose_received: i1, %choose_mux: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      %token = dataflow.thread.launch
          @feedback(%channel, %channel, %message, %enabled, %choose_received,
                    %choose_mux)
          : (!dataflow.channel<i32>, !dataflow.channel<i32>, i32, i1, i1, i1)
            -> !dataflow.thread_token
    }
    return
  }
}

// Producer occurrences contribute two or three events while consumer
// occurrences contribute zero or one. Flat ordinals cross launch boundaries,
// so neither later send can be paired with the same apparent launch
// occurrence. One mixed SSA path does not prove readiness for the other path.
// REPEATED-LABEL: dataflow.thread private @feedback domain(#dataflow.thread_domain<dense>)
// REPEATED: dataflow.channel.send
// REPEATED: %[[BRANCH:.*]]:2 = scf.if
// REPEATED: %[[RECEIVED:.*]], %[[DONE:.*]] = dataflow.graph.launch @feedback_graph
// REPEATED: %[[SELECTIVE:.*]] = dataflow.mux {{.*}}, {{.*}}, %[[RECEIVED]]
// REPEATED: %[[SELECTED:.*]] = scf.if
// REPEATED: scf.yield %[[RECEIVED]] : i32
// REPEATED: scf.yield %[[SELECTIVE]] : i32
// REPEATED: dataflow.graph.wait %[[DONE]] : none
// REPEATED-NEXT: dataflow.channel.send {{.*}}, %[[SELECTED]]
// REPEATED: %[[NEXT:.*]] = arith.addi
// REPEATED: scf.yield %[[NEXT]], %[[DONE]] : i32, none
// REPEATED-NOT: dataflow.graph.wait
// REPEATED: dataflow.channel.send {{.*}}, %[[BRANCH]]#0
// REPEATED-NOT: dataflow.graph.wait
// REPEATED-LABEL: func.func @repeat_feedback
// REPEATED: scf.for
// REPEATED: dataflow.thread.launch @feedback

//--- rate-conversion.mlir
module {
  dataflow.thread private @burst domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>, %message: i32)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%message, %channel)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%payload: i32, %output: !dataflow.channel<i32>):
        dataflow.channel.send %output, %payload : !dataflow.channel<i32>
        dataflow.channel.send %output, %payload : !dataflow.channel<i32>
        dataflow.channel.send %output, %payload : !dataflow.channel<i32>
        dataflow.channel.send %output, %payload : !dataflow.channel<i32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "burst_graph", source_maps = []} :
        (i32, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @single domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %received = dataflow.channel.receive %channel
        : !dataflow.channel<i32>
    dataflow.thread.yield
  }

  func.func @rate_conversion(
      %channel: !dataflow.channel<i32>, %message: i32) {
    %producer = dataflow.thread.launch @burst(%channel, %message)
        : (!dataflow.channel<i32>, i32) -> !dataflow.thread_token
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iteration = %c0 to %c4 step %c1 {
      %consumer = dataflow.thread.launch @single(%channel)
          : (!dataflow.channel<i32>) -> !dataflow.thread_token
    }
    return
  }
}

// One producer occurrence contributes four events while four ordered consumer
// occurrences contribute one each; publication must not impose segments.
// RATE-LABEL: dataflow.thread private @burst domain(#dataflow.thread_domain<dense>)
// RATE: dataflow.graph.launch @burst_graph
// RATE-NOT: dataflow.graph.wait
// RATE-LABEL: dataflow.thread private @single domain(#dataflow.thread_domain<dense>)
// RATE: dataflow.channel.receive
// RATE-LABEL: func.func @rate_conversion
// RATE: dataflow.thread.launch @burst
// RATE: scf.for
// RATE: dataflow.thread.launch @single
// RATE-LABEL: dataflow.graph private @burst_graph
// RATE: %[[RATE_CONTROL0:[^: ]+]]:2 = dataflow.sync
// RATE-NEXT: %[[RATE_EVENT0:[^: ]+]]:2 = dataflow.sync
// RATE-NEXT: %[[RATE_CONTROL1:[^: ]+]]:2 = dataflow.sync
// RATE-NEXT: %[[RATE_EVENT1:[^: ]+]]:2 = dataflow.sync
// RATE-NEXT: %[[RATE_CONTROL2:[^: ]+]]:2 = dataflow.sync
// RATE-NEXT: %[[RATE_EVENT2:[^: ]+]]:2 = dataflow.sync
// RATE-NEXT: %[[RATE_CONTROL3:[^: ]+]]:2 = dataflow.sync
// RATE-NEXT: %[[RATE_EVENT3:[^: ]+]]:2 = dataflow.sync
// RATE: %[[RATE_LEFT:[^ ]+]] = dataflow.mux %[[RATE_LEFT_SELECTOR:[^,]+]], %[[RATE_EVENT0]]#1, %[[RATE_EVENT1]]#1 : (i1, i32, i32) -> i32
// RATE: %[[RATE_RIGHT:[^ ]+]] = dataflow.mux %[[RATE_RIGHT_SELECTOR:[^,]+]], %[[RATE_EVENT2]]#1, %[[RATE_EVENT3]]#1 : (i1, i32, i32) -> i32
// RATE: %[[RATE_PAYLOAD:[^ ]+]] = dataflow.mux %[[RATE_ROOT_SELECTOR:[^,]+]], %[[RATE_LEFT]], %[[RATE_RIGHT]] : (i1, i32, i32) -> i32
// RATE: %[[RATE_COMMIT:[^ ]+]] = dataflow.mux %[[RATE_ROOT_SELECTOR]], {{.*}} : (i1, none, none) -> none
// RATE: %[[RATE_DRAIN:[^: ]+]]:2 = dataflow.sync %[[RATE_COMMIT]], %[[RATE_PAYLOAD]] : (none, i32) -> (none, i32)
// RATE: dataflow.graph.return values() streams(%[[RATE_DRAIN]]#1 : i32)

//--- coordinate-identity.mlir
module {
  dataflow.thread private @identity_feedback domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>,
      %graph_output: !dataflow.channel<i32>, %message: i32, %use_graph: i1)
      ctrl (%ctrl: none) iv (%iv: index) {
    dataflow.channel.send %channel, %message : !dataflow.channel<i32>
    "loom.spatial_region"(%channel)
        <{operandSegmentSizes = array<i32: 0, 1, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: !dataflow.channel<i32>):
        %received = dataflow.channel.receive %input
            : !dataflow.channel<i32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {
      graph_name = "identity_feedback_graph",
      source_maps = [affine_map<(d0) -> (d0)>]
    } : (!dataflow.channel<i32>) -> ()
    scf.if %use_graph {
      "loom.spatial_region"(%message, %graph_output)
          <{operandSegmentSizes = array<i32: 1, 0, 0, 1>,
            resultSegmentSizes = array<i32: 0, 0>}> ({
        ^bb0(%payload: i32, %output: !dataflow.channel<i32>):
          dataflow.channel.send %output, %payload
              : !dataflow.channel<i32>
          "loom.spatial_yield"()
              <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
      }) {graph_name = "identity_producer_graph", source_maps = []} :
          (i32, !dataflow.channel<i32>) -> ()
    } else {
      dataflow.channel.send %channel, %message : !dataflow.channel<i32>
    }
    dataflow.thread.yield
  }

  dataflow.thread private @identity_sink domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) iv (%iv: index) {
    "loom.spatial_region"(%channel)
        <{operandSegmentSizes = array<i32: 0, 1, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: !dataflow.channel<i32>):
        %received = dataflow.channel.receive %input
            : !dataflow.channel<i32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {
      graph_name = "identity_sink_graph",
      source_maps = [affine_map<(d0) -> (d0)>]
    } : (!dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }

  func.func @coordinate_identity(
      %channel: !dataflow.channel<i32>,
      %graph_output: !dataflow.channel<i32>, %message: i32,
      %use_graph: i1, %extent: index) {
    %token = dataflow.thread.launch
        @identity_feedback(%channel, %graph_output, %message, %use_graph)
        grid(%extent)
        : (!dataflow.channel<i32>, !dataflow.channel<i32>, i32, i1)
          -> !dataflow.thread_token
    %sink = dataflow.thread.launch @identity_sink(%graph_output) grid(%extent)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}

// Identity source_map and equal coordinates do not prove equal event
// positions or reverse publication causality.
// IDENTITY: #map = affine_map<(d0) -> (d0)>
// IDENTITY-LABEL: dataflow.thread private @identity_feedback domain(#dataflow.thread_domain<dense>)
// IDENTITY: dataflow.channel.send
// IDENTITY: %[[CONSUMER_DONE:.*]] = dataflow.graph.launch @identity_feedback_graph
// IDENTITY: %[[BRANCH_DONE:.*]] = scf.if
// IDENTITY: %[[PRODUCER_DONE:.*]] = dataflow.graph.launch @identity_producer_graph deps(%[[CONSUMER_DONE]])
// IDENTITY-NOT: dataflow.graph.wait
// IDENTITY: scf.yield %[[PRODUCER_DONE]] : none
// IDENTITY: else
// IDENTITY: dataflow.graph.wait %[[CONSUMER_DONE]] : none
// IDENTITY-NEXT: dataflow.channel.send
// IDENTITY: scf.yield
// IDENTITY-NOT: dataflow.graph.wait
// IDENTITY-LABEL: dataflow.graph private @identity_feedback_graph
// IDENTITY-NOT: dataflow.graph.wait
// IDENTITY-LABEL: dataflow.graph private @identity_producer_graph
// IDENTITY-NOT: dataflow.graph.wait

//--- multicast.mlir
module {
  dataflow.thread private @multicast_source domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>, %first: i32, %second: i32)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%first, %second, %channel)
        <{operandSegmentSizes = array<i32: 2, 0, 0, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%first_payload: i32, %second_payload: i32,
           %output: !dataflow.channel<i32>):
        dataflow.channel.send %output, %first_payload
            : !dataflow.channel<i32>
        dataflow.channel.send %output, %second_payload
            : !dataflow.channel<i32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "multicast_graph", source_maps = []} :
        (i32, i32, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @left domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %first = dataflow.channel.receive %channel : !dataflow.channel<i32>
    %second = dataflow.channel.receive %channel : !dataflow.channel<i32>
    dataflow.thread.yield
  }

  dataflow.thread private @right domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %first = dataflow.channel.receive %channel : !dataflow.channel<i32>
    %second = dataflow.channel.receive %channel : !dataflow.channel<i32>
    dataflow.thread.yield
  }

  func.func @multicast(
      %channel: !dataflow.channel<i32>, %first: i32, %second: i32) {
    %producer = dataflow.thread.launch
        @multicast_source(%channel, %first, %second)
        : (!dataflow.channel<i32>, i32, i32) -> !dataflow.thread_token
    %left = dataflow.thread.launch @left(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %right = dataflow.thread.launch @right(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}

// Each consumer binding observes the producer's same flat nth-message
// sequence independently.
// MULTICAST-LABEL: dataflow.thread private @multicast_source domain(#dataflow.thread_domain<dense>)
// MULTICAST: dataflow.graph.launch @multicast_graph
// MULTICAST-NOT: dataflow.graph.wait
// MULTICAST-LABEL: dataflow.thread private @left domain(#dataflow.thread_domain<dense>)
// MULTICAST: dataflow.channel.receive
// MULTICAST: dataflow.channel.receive
// MULTICAST-LABEL: dataflow.thread private @right domain(#dataflow.thread_domain<dense>)
// MULTICAST: dataflow.channel.receive
// MULTICAST: dataflow.channel.receive
// MULTICAST-LABEL: func.func @multicast
// MULTICAST-SAME: (%[[CHANNEL:[^: ]+]]: !dataflow.channel<i32>
// MULTICAST: dataflow.thread.launch @multicast_source(%[[CHANNEL]],
// MULTICAST: dataflow.thread.launch @left(%[[CHANNEL]])
// MULTICAST: dataflow.thread.launch @right(%[[CHANNEL]])
// MULTICAST-LABEL: dataflow.graph private @multicast_graph
// MULTICAST: %[[MULTICAST_EVENT0:[^: ]+]]:2 = dataflow.sync %{{.*}}#0, %{{.*}}#1 : (none, i32) -> (none, i32)
// MULTICAST: %[[MULTICAST_EVENT1:[^: ]+]]:2 = dataflow.sync %{{.*}}#1, %{{.*}}#1 : (none, i32) -> (none, i32)
// MULTICAST: %[[MULTICAST_PAYLOAD:[^ ]+]] = dataflow.mux {{[^,]+}}, %[[MULTICAST_EVENT0]]#1, %[[MULTICAST_EVENT1]]#1 : (i1, i32, i32) -> i32
// MULTICAST: %[[MULTICAST_COMMIT:[^ ]+]] = dataflow.mux {{[^,]+}}, %[[MULTICAST_EVENT0]]#0, %[[MULTICAST_EVENT1]]#0 : (i1, none, none) -> none
// MULTICAST: %[[MULTICAST_DRAIN:[^: ]+]]:2 = dataflow.sync %[[MULTICAST_COMMIT]], %[[MULTICAST_PAYLOAD]] : (none, i32) -> (none, i32)
// MULTICAST: dataflow.graph.return values() streams(%[[MULTICAST_DRAIN]]#1 : i32)
