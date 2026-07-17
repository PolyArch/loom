// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s

// CHECK: error: loom-lower-graph-memory: raw scf.parallel requires a selected schedule and provenance before graph-region lowering
// CHECK-LABEL: dataflow.graph.func private @would_be_rewritten
// CHECK: memref.load
// CHECK-NOT: dataflow.load
// CHECK-LABEL: dataflow.graph.func private @raw_parallel
// CHECK: scf.parallel
dataflow.graph.func private @would_be_rewritten(
    %start: none, %index: index, %a: memref<?xi32>) -> none
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %value = memref.load %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

dataflow.graph.func private @raw_parallel(
    %start: none, %lb: index, %ub: index, %step: index,
    %a: memref<?xi32>) -> none
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  scf.parallel (%i) = (%lb) to (%ub) step (%step) {
    %value = memref.load %a[%i] : memref<?xi32>
    scf.reduce
  }
  dataflow.graph.return %start : none
}
