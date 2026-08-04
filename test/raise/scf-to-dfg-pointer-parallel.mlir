// RUN: loom-raise-opt --loom-lower-graph-memory %s | FileCheck %s

// CHECK-LABEL: dataflow.graph private @pointer_parallel(
// CHECK: llvm.getelementptr inbounds
// CHECK: dataflow.store
// CHECK-NOT: llvm.store
// CHECK-NOT: scf.
// CHECK: dataflow.graph.return
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph private @pointer_parallel(
      %ctrl: none, %pointer: !llvm.ptr) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %value = arith.constant 7 : i32
    scf.forall (%lane) = (%c0) to (%c8) step (%c1) {
      %wide = arith.index_cast %lane : index to i64
      %address = llvm.getelementptr inbounds %pointer[%wide]
          : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %value, %address : i32, !llvm.ptr
    }
    dataflow.graph.return %ctrl : none
  }
}
