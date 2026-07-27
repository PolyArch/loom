// RUN: loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading %s | FileCheck %s

// Graph publication stages ordinary callable declarations while lowering the
// selected graph. A multi-block callable therefore anchors safe body removal:
// its cross-block SSA uses must not survive destruction in the staging clone,
// and the original callable body must remain intact in the published module.

// CHECK-LABEL: llvm.func @helper
// CHECK: %[[VALUE:.*]] = llvm.load
// CHECK: llvm.br ^[[EXIT:.*]]
// CHECK: ^[[EXIT]]:
// CHECK: llvm.add %[[VALUE]], %[[VALUE]]
// CHECK-LABEL: dataflow.graph private @selected_graph
// CHECK: dataflow.store

llvm.func @helper(%pointer: !llvm.ptr) -> i32 {
  %value = llvm.load %pointer : !llvm.ptr -> i32
  llvm.br ^exit
^exit:
  %sum = llvm.add %value, %value : i32
  llvm.return %sum : i32
}

dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
    %target: memref<1xi32>, %value: i32) ctrl (%ctrl: none) {
  "loom.spatial_region"(%value, %target)
      <{operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        resultSegmentSizes = array<i32: 0, 0>}> ({
    ^bb0(%payload: i32, %memory: memref<1xi32>):
      %zero = arith.constant 0 : index
      memref.store %payload, %memory[%zero] : memref<1xi32>
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
  }) {graph_name = "selected_graph", source_maps = []} :
      (i32, memref<1xi32>) -> ()
  dataflow.thread.yield
}
