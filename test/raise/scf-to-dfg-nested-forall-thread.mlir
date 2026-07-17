// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s

// Nested graph-owned parallel syntax remains a structured candidate until a
// concrete schedule and provenance are selected. It must not be published as
// a canonical graph.

module {
  func.func @launch(%dst: !llvm.ptr) {
    %c4 = arith.constant 4 : index
    %completion = dataflow.thread.launch @t_nested_forall(%dst) grid(%c4) : (!llvm.ptr) -> !dataflow.thread_token
    return
  }

  dataflow.thread private @t_nested_forall(%arg0: !llvm.ptr) ctrl (%ctrl: none) iv (%iv: index) {
    %mem = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to memref<?xi32>
    %outer_ub = arith.constant 2 : index
    %inner_ub = arith.constant 2 : index
    scf.forall (%i) in (%outer_ub) {
      scf.forall (%j) in (%inner_ub) {
        %row = arith.muli %i, %inner_ub : index
        %idx = arith.addi %row, %j : index
        %value = arith.index_cast %idx : index to i32
        memref.store %value, %mem[%idx] : memref<?xi32>
      }
    }
    dataflow.thread.yield
  }

  dataflow.thread private @t_if_nested_forall(%arg0: !llvm.ptr) ctrl (%ctrl: none) iv (%iv: index) {
    %mem = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to memref<?xi32>
    %false = arith.constant false
    %outer_ub = arith.constant 2 : index
    %inner_ub = arith.constant 2 : index
    scf.if %false {
    } else {
      scf.forall (%i) in (%outer_ub) {
        scf.forall (%j) in (%inner_ub) {
          %row = arith.muli %i, %inner_ub : index
          %idx = arith.addi %row, %j : index
          %value = arith.index_cast %idx : index to i32
          memref.store %value, %mem[%idx] : memref<?xi32>
        }
      }
    }
    dataflow.thread.yield
  }
}

// CHECK: error: loom-lower-graph-memory: raw scf.forall requires a selected schedule and provenance before graph-region lowering
// CHECK: "loom.spatial_region"
// CHECK-NOT: dataflow.graph private
// CHECK-NOT: dataflow.graph.launch
