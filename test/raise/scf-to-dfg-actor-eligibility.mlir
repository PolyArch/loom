// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/unregistered-graph.mlir 2>&1 | FileCheck %s --check-prefix=GRAPH-REJECT
// RUN: loom-dfg-sim %t.dir/registered-final.mlir --graph registered_final --output %t.registered.json
// RUN: FileCheck %s --check-prefix=VALIDATOR-REGISTERED < %t.registered.json
// RUN: loom-dfg-sim %t.dir/canonical-backend-gap.mlir --graph canonical_backend_gap --arg 0=1 --arg 1=2 --output %t.canonical-backend-gap.json
// RUN: FileCheck %s --check-prefix=CANONICAL-BACKEND-GAP < %t.canonical-backend-gap.json
// RUN: not loom-dfg-sim %t.dir/unregistered-graph.mlir --graph unregistered_graph_actor --arg 0=1 --output %t.unregistered.json 2>&1 | FileCheck %s --check-prefix=VALIDATOR-REJECT
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/vector-to-integer-bitcast.mlir 2>&1 | FileCheck %s --check-prefix=GRAPH-BITCAST-REJECT
// RUN: not loom-dfg-sim %t.dir/vector-to-integer-bitcast.mlir --graph vector_to_integer_bitcast --arg 0=513 --output %t.vector-bitcast.json 2>&1 | FileCheck %s --check-prefix=VALIDATOR-BITCAST-REJECT

//--- registered-final.mlir
module {
  dataflow.graph private @registered_final(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 1.0 : f32} : f32
    %converted = llvm.fptosi %value : f32 to i32
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
    %sum = llvm.add %lhs, %rhs : i32
    %either = llvm.or %lhs, %rhs : i32
    %published:2 = dataflow.sync %start, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

// CANONICAL-BACKEND-GAP: "unsupported op: llvm.add"
// CANONICAL-BACKEND-GAP-NEXT: "unsupported op: llvm.or"
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

// GRAPH-BITCAST-REJECT: error: loom-lower-graph-memory: operation 'llvm.bitcast' is not a registered canonical Dataflow actor or a supported graph-lowering operation
// VALIDATOR-BITCAST-REJECT: finalized graph contains unregistered actor 'llvm.bitcast'
