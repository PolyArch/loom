// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/registered.mlir | FileCheck %s --check-prefix=REGISTERED
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/unregistered-graph.mlir 2>&1 | FileCheck %s --check-prefix=GRAPH-REJECT
// RUN: not loom-raise-opt --loom-lower-for-to-graph %t.dir/unregistered-thread.mlir 2>&1 | FileCheck %s --check-prefix=THREAD-REJECT
// RUN: loom-dfg-sim %t.dir/registered-final.mlir --graph registered_final --output %t.registered.json
// RUN: FileCheck %s --check-prefix=VALIDATOR-REGISTERED < %t.registered.json
// RUN: not loom-dfg-sim %t.dir/unregistered-graph.mlir --graph unregistered_graph_actor --arg 0=1 --output %t.unregistered.json 2>&1 | FileCheck %s --check-prefix=VALIDATOR-REJECT

//--- registered.mlir
module {
  dataflow.thread private @registered_llvm_compute(
      %dst: !llvm.ptr, %seed: f32) ctrl (%ctrl: none) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.for %i = %c0 to %c1 step %c1
        iter_args(%value = %seed) -> (f32) : i32 {
      scf.yield %value : f32
    }
    %converted = llvm.fptosi %result : f32 to i32
    %swapped = llvm.intr.bswap(%converted) : (i32) -> i32
    llvm.store %swapped, %dst : i32, !llvm.ptr
    dataflow.thread.yield
  }
}

// REGISTERED-LABEL: dataflow.graph private @g_registered_llvm_compute_0
// REGISTERED: llvm.fptosi
// REGISTERED: llvm.intr.bswap
// REGISTERED: dataflow.graph.return

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

//--- unregistered-thread.mlir
module {
  dataflow.thread private @unregistered_thread_actor(%input: i32)
      ctrl (%ctrl: none) {
    %undefined = llvm.mlir.undef : i32
    %sum = llvm.add %input, %undefined : i32
    dataflow.thread.yield
  }
}

// THREAD-REJECT: error: loom-lower-for-to-graph: operation 'llvm.mlir.undef' is not a registered canonical Dataflow actor or a supported graph-lowering operation
