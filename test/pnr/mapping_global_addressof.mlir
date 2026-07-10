// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-signal-window --output %t.dir/shared-signal-window.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph global_table_load --hardware-mlir %t.dir/shared-signal-window.mlir --hardware shared_signal_window_adg --workload global_table_load --output %t.dir/global_table_load.mapping.csv --artifact %t.dir/global_table_load.mapping.json
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/global_table_load.mapping.json

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "llvm.load"
// JSON-NOT: "operation": "llvm.mlir.addressof"
// JSON-NOT: "unsupported PnR graph operation: llvm.mlir.addressof"

module {
  llvm.mlir.global external constant @lookup_table() : !llvm.array<3 x i32>

  dataflow.graph.func private @global_table_load(%ctrl: none, %idx: i32)
      -> (none, i32) {
    %base = llvm.mlir.addressof @lookup_table : !llvm.ptr
    %elem = llvm.getelementptr inbounds|nuw %base[%idx] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4 x i8>
    %data = llvm.load %elem {alignment = 4 : i64} : !llvm.ptr -> i32
    dataflow.graph.return %ctrl, %data : none, i32
  }
}
