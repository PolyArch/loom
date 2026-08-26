// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/channel.mlir | FileCheck %s --check-prefix=SUCCESS
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/mixed-channel.mlir 2>&1 | FileCheck %s --check-prefix=FAILURE --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/choice-local-repeat.mlir 2>&1 | FileCheck %s --check-prefix=CHOICE-LOCAL --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/nested-choice.mlir 2>&1 | FileCheck %s --check-prefix=NESTED-CHOICE --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/repeat-domain.mlir 2>&1 | FileCheck %s --check-prefix=REPEAT-DOMAIN --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/narrow-switch.mlir 2>&1 | FileCheck %s --check-prefix=NARROW-SWITCH --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch

// SUCCESS-LABEL: dataflow.thread private @channel_sender domain(#dataflow.thread_domain<dense>)
// SUCCESS: dataflow.graph.launch @channel_graph
// SUCCESS-LABEL: dataflow.graph private @channel_graph
// SUCCESS-SAME: -> i32
// SUCCESS: dataflow.sync
// SUCCESS-NOT: dataflow.channel.send
// SUCCESS-NOT: loom.spatial_region
// SUCCESS: dataflow.graph.return values() streams(%{{.*}} : i32)

// FAILURE: error: loom-lower-graph-memory: one stream binding cannot contain parallel endpoint sites without a deterministic merge
// FAILURE-LABEL: dataflow.thread private @channel_sender domain(#dataflow.thread_domain<dense>)
// FAILURE: loom.spatial_region
// FAILURE: dataflow.channel.send

// CHOICE-LOCAL: error: loom-lower-graph-memory: conditional stream repetition requires a conditional event-count projection
// CHOICE-LOCAL-LABEL: dataflow.thread private @choice_local_sender domain(#dataflow.thread_domain<dense>)
// CHOICE-LOCAL: loom.spatial_region
// CHOICE-LOCAL: scf.index_switch
// CHOICE-LOCAL: dataflow.channel.send

// NESTED-CHOICE: error: loom-lower-graph-memory: nested stream choices require a hierarchical conditional-event projection
// NESTED-CHOICE-LABEL: dataflow.thread private @nested_choice_sender domain(#dataflow.thread_domain<dense>)
// NESTED-CHOICE: loom.spatial_region
// NESTED-CHOICE: scf.if
// NESTED-CHOICE: scf.index_switch
// NESTED-CHOICE: dataflow.channel.send

// REPEAT-DOMAIN: error: loom-lower-graph-memory: cross-scope stream repetition domain is not available at schedule activation
// REPEAT-DOMAIN-LABEL: dataflow.thread private @repeat_domain_receiver domain(#dataflow.thread_domain<dense>)
// REPEAT-DOMAIN: loom.spatial_region
// REPEAT-DOMAIN: dataflow.channel.receive
// REPEAT-DOMAIN: scf.for
// NARROW-SWITCH: error: loom-lower-graph-memory: scf.index_switch lane count exceeds the configured index width
// NARROW-SWITCH-LABEL: dataflow.thread private @narrow_switch domain(#dataflow.thread_domain<dense>)
// NARROW-SWITCH: loom.spatial_region
// NARROW-SWITCH: scf.index_switch
// FAILURE-LABEL: dataflow.thread private @parallel_channel_sender domain(#dataflow.thread_domain<dense>)
// FAILURE: loom.spatial_region
// FAILURE: dataflow.channel.send

//--- channel.mlir
dataflow.thread private @channel_sender domain(#dataflow.thread_domain<dense>)(
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
dataflow.thread private @channel_sender domain(#dataflow.thread_domain<dense>)(
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

dataflow.thread private @parallel_channel_sender domain(#dataflow.thread_domain<dense>)(
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

//--- choice-local-repeat.mlir
dataflow.thread private @choice_local_sender domain(#dataflow.thread_domain<dense>)(
    %channel: !dataflow.channel<i32>, %selector: index, %message: i32)
    ctrl (%start: none) {
  "loom.spatial_region"(%selector, %message, %channel)
      <{operandSegmentSizes = array<i32: 2, 0, 0, 1>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%select: index, %payload: i32,
         %output: !dataflow.channel<i32>):
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %bound = arith.constant 4 : index
      scf.index_switch %select
      case 1 {
        scf.for %iv = %zero to %bound step %one {
          dataflow.channel.send %output, %payload
              : !dataflow.channel<i32>
        }
        scf.yield
      }
      default {
        dataflow.channel.send %output, %payload : !dataflow.channel<i32>
        scf.yield
      }
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "choice_local_graph", source_maps = []} :
      (index, i32, !dataflow.channel<i32>) -> ()
  dataflow.thread.yield
}

//--- nested-choice.mlir
dataflow.thread private @nested_choice_sender domain(#dataflow.thread_domain<dense>)(
    %channel: !dataflow.channel<i32>, %outer: i1, %inner: index,
    %message: i32) ctrl (%start: none) {
  "loom.spatial_region"(%outer, %inner, %message, %channel)
      <{operandSegmentSizes = array<i32: 3, 0, 0, 1>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%outer_select: i1, %inner_select: index, %payload: i32,
         %output: !dataflow.channel<i32>):
      scf.if %outer_select {
        scf.index_switch %inner_select
        case 1 {
          dataflow.channel.send %output, %payload
              : !dataflow.channel<i32>
          scf.yield
        }
        default {
          dataflow.channel.send %output, %payload
              : !dataflow.channel<i32>
          scf.yield
        }
      } else {
        dataflow.channel.send %output, %payload : !dataflow.channel<i32>
      }
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "nested_choice_graph", source_maps = []} :
      (i1, index, i32, !dataflow.channel<i32>) -> ()
  dataflow.thread.yield
}

//--- repeat-domain.mlir
dataflow.thread private @repeat_domain_receiver domain(#dataflow.thread_domain<dense>)(
    %channel: !dataflow.channel<i32>) ctrl (%start: none) {
  "loom.spatial_region"(%channel)
      <{operandSegmentSizes = array<i32: 0, 1, 0, 0>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%input: !dataflow.channel<i32>):
      %count = dataflow.channel.receive %input : !dataflow.channel<i32>
      %limit = arith.index_cast %count : i32 to index
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %iv = %zero to %limit step %one {
        %message = dataflow.channel.receive %input : !dataflow.channel<i32>
      }
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {
    graph_name = "repeat_domain_graph",
    source_maps = [affine_map<() -> ()>]
  } : (!dataflow.channel<i32>) -> ()
  dataflow.thread.yield
}

//--- narrow-switch.mlir
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 1>>
} {
  dataflow.thread private @narrow_switch domain(#dataflow.thread_domain<dense>)(
      %output: !dataflow.channel<i32>, %selector: index, %message: i32)
      ctrl (%start: none) {
    "loom.spatial_region"(%selector, %message, %output)
        <{operandSegmentSizes = array<i32: 2, 0, 0, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%select: index, %payload: i32,
           %channel: !dataflow.channel<i32>):
        scf.index_switch %select
        case 0 {
          dataflow.channel.send %channel, %payload : !dataflow.channel<i32>
          scf.yield
        }
        case 1 {
          dataflow.channel.send %channel, %payload : !dataflow.channel<i32>
          scf.yield
        }
        default {
          dataflow.channel.send %channel, %payload : !dataflow.channel<i32>
          scf.yield
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "narrow_switch_graph", source_maps = []} :
        (index, i32, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }
}
