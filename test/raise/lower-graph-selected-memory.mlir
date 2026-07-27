// RUN: loom-raise-opt --canonicalize --loom-lower-graph-memory %s \
// RUN:   | FileCheck %s

// A branch-selected pointer does not become a dynamic memory capability.
// The canonicalizer projects this source shape to an arith.select. Memory
// lowering must restore selected execution around the load so exactly one
// branch performs a memory effect and each load retains a graph-memory owner.

// CHECK-LABEL: dataflow.graph private @branch_selected_load
// CHECK-COUNT-2: dataflow.load
// CHECK: dataflow.mux
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @branch_selected_load(
      %ctrl: none, %choose_a: i1, %ordinal: i64,
      %a: !llvm.ptr, %b: !llvm.ptr) -> f32
      attributes {input_segments = array<i32: 2, 0, 2>,
                  result_segments = array<i32: 1, 0, 0>} {
    %a_ptr = llvm.getelementptr inbounds %a[%ordinal]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %b_ptr = llvm.getelementptr inbounds %b[%ordinal]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %selected = scf.if %choose_a -> (!llvm.ptr) {
      scf.yield %a_ptr : !llvm.ptr
    } else {
      scf.yield %b_ptr : !llvm.ptr
    }
    %value = llvm.load %selected : !llvm.ptr -> f32
    dataflow.graph.return %ctrl, %value : none, f32
  }
}
