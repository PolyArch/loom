// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/registered.mlir | FileCheck %s --check-prefix=REGISTERED
// RUN: not loom-raise-opt --loom-lower-graph-memory %t.dir/unregistered-graph.mlir 2>&1 | FileCheck %s --check-prefix=GRAPH-REJECT
// RUN: not loom-raise-opt --loom-lower-for-to-graph %t.dir/unregistered-thread.mlir 2>&1 | FileCheck %s --check-prefix=THREAD-REJECT
// RUN: loom-dfg-sim %t.dir/registered-final.mlir --graph registered_final --output %t.registered.json
// RUN: FileCheck %s --check-prefix=VALIDATOR-REGISTERED < %t.registered.json
// RUN: loom-dfg-sim %t.dir/canonical-backend-gap.mlir --graph canonical_backend_gap --arg 0=1 --arg 1=2 --output %t.canonical-backend-gap.json
// RUN: FileCheck %s --check-prefix=CANONICAL-BACKEND-GAP < %t.canonical-backend-gap.json
// RUN: not loom-dfg-sim %t.dir/unregistered-graph.mlir --graph unregistered_graph_actor --arg 0=1 --output %t.unregistered.json 2>&1 | FileCheck %s --check-prefix=VALIDATOR-REJECT
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/masked-epilogues.mlir | FileCheck %s --check-prefix=MASKED-EPILOGUE

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

//--- canonical-backend-gap.mlir
module {
  dataflow.graph private @canonical_backend_gap(
      %start: none, %lhs: i32, %rhs: i32) -> i32 {
    %sum = llvm.add %lhs, %rhs : i32
    %published:2 = dataflow.sync %start, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

// CANONICAL-BACKEND-GAP-DAG: "graph": "canonical_backend_gap"
// CANONICAL-BACKEND-GAP-DAG: "status": "unsupported"
// CANONICAL-BACKEND-GAP-DAG: "unsupported op: llvm.add"

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

//--- masked-epilogues.mlir
module {
  dataflow.thread private @pack_epilogue(
      %limit: index, %mask: i2, %seed0: i8, %seed1: i8,
      %out: memref<?xi16>)
      ctrl (%ctrl: none) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lane0, %lane1 = scf.for %i = %c0 to %limit step %c1
        iter_args(%value0 = %seed0, %value1 = %seed1) -> (i8, i8) {
      scf.yield %value0, %value1 : i8, i8
    }
    %packed = dataflow.pack %lane0, %lane1 mask %mask {vec_size = 2 : i64}
        : (i8, i8, i2) -> i16
    memref.store %packed, %out[%c0] : memref<?xi16>
    dataflow.thread.yield
  }

  dataflow.thread private @unpack_epilogue(
      %limit: index, %mask: i2, %seed: i16, %out: memref<?xi8>)
      ctrl (%ctrl: none) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %packed = scf.for %i = %c0 to %limit step %c1
        iter_args(%value = %seed) -> (i16) {
      scf.yield %value : i16
    }
    %lane0, %lane1 =
      dataflow.unpack %packed, %mask {vec_size = 2 : i64}
        : (i16, i2) -> (i8, i8)
    memref.store %lane0, %out[%c0] : memref<?xi8>
    dataflow.thread.yield
  }
}

// MASKED-EPILOGUE-LABEL: dataflow.thread private @pack_epilogue
// MASKED-EPILOGUE: dataflow.pack
// MASKED-EPILOGUE-LABEL: dataflow.thread private @unpack_epilogue
// MASKED-EPILOGUE: dataflow.unpack
// MASKED-EPILOGUE-LABEL: dataflow.graph private @g_pack_epilogue_0
// MASKED-EPILOGUE-NOT: dataflow.pack
// MASKED-EPILOGUE: dataflow.graph.return
// MASKED-EPILOGUE-LABEL: dataflow.graph private @g_unpack_epilogue_0
// MASKED-EPILOGUE-NOT: dataflow.unpack
// MASKED-EPILOGUE: dataflow.graph.return
