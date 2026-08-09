// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/registered-spatial.mlir | FileCheck %s --check-prefix=SPATIAL-REGISTERED
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/registered-intrinsic-spatial.mlir | FileCheck %s --check-prefix=SPATIAL-INTRINSIC
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/unregistered-spatial.mlir 2>&1 | FileCheck %s --check-prefix=SPATIAL-UNREGISTERED --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/unregistered-graph.mlir 2>&1 | FileCheck %s --check-prefix=GRAPH-REJECT
// RUN: loom-dfg-sim %t.dir/registered-final.mlir --graph registered_final --output %t.registered.json
// RUN: FileCheck %s --check-prefix=VALIDATOR-REGISTERED < %t.registered.json
// RUN: loom-dfg-sim %t.dir/canonical-backend-gap.mlir --graph canonical_backend_gap --arg 0=1 --arg 1=2 --output %t.canonical-backend-gap.json
// RUN: FileCheck %s --check-prefix=CANONICAL-BACKEND-GAP < %t.canonical-backend-gap.json
// RUN: not loom-dfg-sim %t.dir/unregistered-graph.mlir --graph unregistered_graph_actor --arg 0=1 --output %t.unregistered.json 2>&1 | FileCheck %s --check-prefix=VALIDATOR-REJECT
// RUN: loom-raise-opt --loom-lower-graph-memory %t.dir/vector-to-integer-bitcast.mlir -o %t.vector-bitcast.mlir
// RUN: FileCheck %s --check-prefix=GRAPH-BITCAST < %t.vector-bitcast.mlir
// RUN: loom-dfg-sim %t.vector-bitcast.mlir --graph vector_to_integer_bitcast --arg 0=513 --output %t.vector-bitcast.json
// RUN: FileCheck %s --check-prefix=VALIDATOR-BITCAST < %t.vector-bitcast.json

// SPATIAL-REGISTERED-LABEL: dataflow.thread private @registered_spatial domain(#dataflow.thread_domain<dense>)
// SPATIAL-REGISTERED: %{{.*}}, %[[DONE:.*]] = dataflow.graph.launch @registered_actor_graph
// SPATIAL-REGISTERED: dataflow.thread.yield %[[DONE]] : none
// SPATIAL-REGISTERED-LABEL: dataflow.graph private @registered_actor_graph
// SPATIAL-REGISTERED: arith.fptosi
// SPATIAL-REGISTERED: dataflow.graph.return
// SPATIAL-REGISTERED-NOT: loom.spatial_region

// SPATIAL-INTRINSIC-LABEL: dataflow.thread private @registered_intrinsic_spatial domain(#dataflow.thread_domain<dense>)
// SPATIAL-INTRINSIC: %{{.*}}, %[[DONE:.*]] = dataflow.graph.launch @registered_intrinsic_actor_graph
// SPATIAL-INTRINSIC: dataflow.thread.yield %[[DONE]] : none
// SPATIAL-INTRINSIC-LABEL: dataflow.graph private @registered_intrinsic_actor_graph
// SPATIAL-INTRINSIC: llvm.call_intrinsic "llvm.fptosi.sat.i16.f32"
// SPATIAL-INTRINSIC: dataflow.graph.return
// SPATIAL-INTRINSIC-NOT: loom.spatial_region

// SPATIAL-UNREGISTERED: error: loom-lower-graph-memory: operation 'llvm.mlir.undef' is not a registered canonical Dataflow actor or a supported graph-lowering operation
// SPATIAL-UNREGISTERED-LABEL: dataflow.thread private @unregistered_spatial domain(#dataflow.thread_domain<dense>)
// SPATIAL-UNREGISTERED: loom.spatial_region
// SPATIAL-UNREGISTERED: llvm.mlir.undef

//--- registered-spatial.mlir
dataflow.thread private @registered_spatial domain(#dataflow.thread_domain<dense>)(%input: f32) ctrl (%start: none) {
  %result = "loom.spatial_region"(%input)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        resultSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%value: f32):
      %converted = arith.fptosi %value : f32 to i32
      "loom.spatial_yield"(%converted)
          <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
  }) {graph_name = "registered_actor_graph", source_maps = []} :
      (f32) -> i32
  dataflow.thread.yield
}

//--- unregistered-spatial.mlir
dataflow.thread private @unregistered_spatial domain(#dataflow.thread_domain<dense>)(%input: i32) ctrl (%start: none) {
  %result = "loom.spatial_region"(%input)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        resultSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%value: i32):
      %undefined = llvm.mlir.undef : i32
      %sum = llvm.add %value, %undefined : i32
      "loom.spatial_yield"(%sum)
          <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
  }) {graph_name = "unregistered_actor_graph", source_maps = []} :
      (i32) -> i32
  dataflow.thread.yield
}

//--- registered-intrinsic-spatial.mlir
dataflow.thread private @registered_intrinsic_spatial domain(#dataflow.thread_domain<dense>)(%input: f32) ctrl (%start: none) {
  %result = "loom.spatial_region"(%input)
      <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        resultSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%value: f32):
      %converted = llvm.call_intrinsic "llvm.fptosi.sat.i16.f32"(%value)
          : (f32) -> i16
      "loom.spatial_yield"(%converted)
          <{operandSegmentSizes = array<i32: 1, 0>}> : (i16) -> ()
  }) {graph_name = "registered_intrinsic_actor_graph", source_maps = []} :
      (f32) -> i16
  dataflow.thread.yield
}

//--- registered-final.mlir
module {
  dataflow.graph private @registered_final(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 1.0 : f32} : f32
    %converted = arith.fptosi %value : f32 to i32
    %swapped = llvm.intr.bswap(%converted) : (i32) -> i32
    %published:2 = dataflow.sync %start, %swapped
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

// VALIDATOR-REGISTERED: "graph": "registered_final"
// VALIDATOR-REGISTERED: "status": "pass"

//--- canonical-backend-gap.mlir
module {
  dataflow.graph private @canonical_backend_gap(
      %start: none, %lhs: i32, %rhs: i32) -> i32 {
    %frozen = llvm.freeze %lhs : i32
    %published:2 = dataflow.sync %start, %frozen
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

// CANONICAL-BACKEND-GAP: "unsupported op: llvm.freeze"
// CANONICAL-BACKEND-GAP: "graph": "canonical_backend_gap"
// CANONICAL-BACKEND-GAP: "status": "unsupported"

//--- unregistered-graph.mlir
module {
  dataflow.graph private @unregistered_graph_actor(
      %start: none, %input: i32) -> i32 {
    %undefined = llvm.mlir.undef : i32
    %sum = llvm.add %input, %undefined : i32
    dataflow.graph.return %start, %sum : none, i32
  }
}

// GRAPH-REJECT: error: loom-lower-graph-memory: operation 'llvm.mlir.undef' is not a registered canonical Dataflow actor or a supported graph-lowering operation
// VALIDATOR-REJECT: finalized graph contains unregistered actor 'llvm.mlir.undef'

//--- vector-to-integer-bitcast.mlir
module {
  dataflow.graph private @vector_to_integer_bitcast(
      %start: none, %input: vector<2xi8>) -> i16 {
    %packed = llvm.bitcast %input : vector<2xi8> to i16
    %published:2 = dataflow.sync %start, %packed
        : (none, i16) -> (none, i16)
    dataflow.graph.return %published#0, %published#1 : none, i16
  }
}

// GRAPH-BITCAST: dataflow.pack %arg1 : vector<2xi8> -> i16
// GRAPH-BITCAST-NOT: llvm.bitcast
// VALIDATOR-BITCAST: "i16:513"
// VALIDATOR-BITCAST: "dataflow.pack": 1
// VALIDATOR-BITCAST: "status": "pass"
