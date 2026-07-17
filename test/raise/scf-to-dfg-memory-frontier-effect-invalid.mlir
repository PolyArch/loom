// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s

// CHECK: error: loom-lower-graph-memory: operation 'llvm.call' is not a registered canonical Dataflow actor or a supported graph-lowering operation
// CHECK-LABEL: dataflow.graph private @would_be_rewritten
// CHECK: memref.load
// CHECK-NOT: dataflow.load
// CHECK-LABEL: dataflow.graph private @nested_effect
// CHECK: scf.if
// CHECK: llvm.call @side_effect

llvm.func @side_effect()

dataflow.graph private @would_be_rewritten(
    %start: none, %index: index, %a: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %value = memref.load %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

dataflow.graph private @nested_effect(
    %start: none, %condition: i1) -> () {
  scf.if %condition {
    llvm.call @side_effect() : () -> ()
  }
  dataflow.graph.return %start : none
}
