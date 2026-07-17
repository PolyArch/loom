// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s

// CHECK: error: loom-lower-graph-memory: effectful or unmodeled graph operation 'llvm.call' is unsupported
// CHECK-LABEL: dataflow.graph.func private @would_be_rewritten
// CHECK: memref.load
// CHECK-NOT: dataflow.load
// CHECK-LABEL: dataflow.graph.func private @nested_effect
// CHECK: scf.if
// CHECK: llvm.call @side_effect

llvm.func @side_effect()

dataflow.graph.func private @would_be_rewritten(
    %start: none, %index: index, %a: memref<?xi32>) -> none
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %value = memref.load %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

dataflow.graph.func private @nested_effect(
    %start: none, %condition: i1) -> none {
  scf.if %condition {
    llvm.call @side_effect() : () -> ()
  }
  dataflow.graph.return %start : none
}
