// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s

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
        %done = dataflow.store %mem[%idx] %value %ctrl : memref<?xi32>
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
          %done = dataflow.store %mem[%idx] %value %ctrl : memref<?xi32>
        }
      }
    }
    dataflow.thread.yield
  }
}

// CHECK-LABEL: dataflow.thread private @t_nested_forall
// CHECK: dataflow.graph.launch @g_t_nested_forall_0
// CHECK-LABEL: dataflow.thread private @t_if_nested_forall
// CHECK: dataflow.graph.launch @g_t_if_nested_forall_0
// CHECK-LABEL: dataflow.graph.func private @g_t_nested_forall_0
// CHECK: scf.forall
// CHECK: scf.forall
// CHECK: dataflow.store
// CHECK-LABEL: dataflow.graph.func private @g_t_if_nested_forall_0
// CHECK: scf.if
// CHECK: scf.forall
// CHECK: scf.forall
// CHECK: dataflow.store
