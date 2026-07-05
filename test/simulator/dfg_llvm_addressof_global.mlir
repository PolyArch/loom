// RUN: loom-dfg-sim %s --graph global_table_load --arg 0=none --arg 1=1 --global-memref lookup_table=11,22,33 --output %t.pass.json
// RUN: FileCheck %s --check-prefix=PASS < %t.pass.json
// RUN: loom-dfg-sim %s --graph global_table_load --arg 0=none --arg 1=1 --output %t.blocked.json
// RUN: FileCheck %s --check-prefix=BLOCKED < %t.blocked.json

// PASS-DAG: "status": "pass"
// PASS-DAG: "final_outputs": [
// PASS-DAG: "none",
// PASS-DAG: "i32:22"
// PASS-DAG: "llvm.mlir.addressof": 1
// PASS-DAG: "llvm.getelementptr": 1
// PASS-DAG: "llvm.load": 1

// BLOCKED-DAG: "status": "blocked"
// BLOCKED-DAG: "pointer memory fixture is missing"

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
