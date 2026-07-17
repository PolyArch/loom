// RUN: not loom-pnr-map --dfg-mlir %s --graph global_table_load --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload global_table_load --output %t.mapping.csv --artifact %t.mapping.json 2>&1 | FileCheck %s --check-prefix=GLOBAL

// GLOBAL: finalized graph contains residual pointer operation 'llvm.mlir.addressof'

module {
  llvm.mlir.global external constant @lookup_table() : !llvm.array<3 x i32>

  dataflow.graph private @global_table_load(%ctrl: none, %idx: i32)
      -> (i32) {
    %base = llvm.mlir.addressof @lookup_table : !llvm.ptr
    %elem = llvm.getelementptr inbounds|nuw %base[%idx] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4 x i8>
    %data = llvm.load %elem {alignment = 4 : i64} : !llvm.ptr -> i32
    dataflow.graph.return %ctrl, %data : none, i32
  }
}
