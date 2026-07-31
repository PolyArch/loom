// RUN: not loom-dfg-sim %s --graph global_table_load --arg 0=1 --output %t.json 2>&1 | FileCheck %s

// CHECK: finalized graph contains unregistered actor 'llvm.mlir.addressof'

module {
  llvm.mlir.global external constant @lookup_table() : !llvm.array<3 x i32>

  dataflow.graph private @global_table_load(%ctrl: none, %idx: i32)
      -> (i32)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %base = llvm.mlir.addressof @lookup_table : !llvm.ptr
    %elem = llvm.getelementptr inbounds|nuw %base[%idx] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4 x i8>
    %data = llvm.load %elem {alignment = 4 : i64} : !llvm.ptr -> i32
    %published:2 = dataflow.sync %ctrl, %data
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
