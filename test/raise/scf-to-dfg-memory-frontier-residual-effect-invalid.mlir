// RUN: not loom-raise-opt --split-input-file --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s

// A residual LLVM write has no SSA completion event that can participate in
// graph.return.complete. Graph-memory must reject it instead of treating graph
// start or an unrelated value as retirement authority.

// CHECK: error: loom-lower-graph-memory: residual memory operation 'llvm.store' has no explicit completion event
// CHECK-LABEL: dataflow.graph.func private @residual_store
// CHECK: llvm.load
// CHECK-NOT: dataflow.load
dataflow.graph.func private @residual_store(
    %start: none, %value: i32, %source: !llvm.ptr, %destination: !llvm.ptr)
    -> (none, i32)
    attributes {input_segments = array<i32: 1, 0, 2>,
                result_segments = array<i32: 1, 0, 0>} {
  %loaded = llvm.load %source : !llvm.ptr -> i32
  llvm.store volatile %value, %destination : i32, !llvm.ptr
  dataflow.graph.return %start, %loaded : none, i32
}

// -----

// Volatile and atomic reads are observable effects even though they produce a
// value. They are not ordinary residual reads that a returned value may cover.

// CHECK: error: loom-lower-graph-memory: residual memory operation 'llvm.load' has no explicit completion event
dataflow.graph.func private @residual_volatile_load(
    %start: none, %base: !llvm.ptr) -> (none, i32)
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %value = llvm.load volatile %base : !llvm.ptr -> i32
  dataflow.graph.return %start, %value : none, i32
}
