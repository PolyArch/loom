// RUN: loom-dfg-sim %s --graph llvm_load_ptr --arg 0=none --memref 1=100,102,105 --output %t.ptr.json
// RUN: FileCheck %s --check-prefix=PTR < %t.ptr.json
// RUN: loom-dfg-sim %s --graph llvm_load_gep --arg 0=none --memref 1=100,102,105 --output %t.gep.json
// RUN: FileCheck %s --check-prefix=GEP < %t.gep.json

// PTR-DAG: "workload": "llvm_load_ptr"
// PTR-DAG: "graph": "llvm_load_ptr"
// PTR-DAG: "status": "pass"
// PTR-DAG: "llvm.load": 1
// PTR-DAG: "i32:100"

// GEP-DAG: "workload": "llvm_load_gep"
// GEP-DAG: "graph": "llvm_load_gep"
// GEP-DAG: "status": "pass"
// GEP-DAG: "llvm.getelementptr": 1
// GEP-DAG: "llvm.load": 1
// GEP-DAG: "i32:102"

module {
  dataflow.graph.func private @llvm_load_ptr(%ctrl: none, %ptr: !llvm.ptr)
      -> (none, i32) {
    %data = llvm.load %ptr {alignment = 4 : i64} : !llvm.ptr -> i32
    dataflow.graph.return %ctrl, %data : none, i32
  }

  dataflow.graph.func private @llvm_load_gep(%ctrl: none, %ptr: !llvm.ptr)
      -> (none, i32) {
    %next = llvm.getelementptr inbounds|nuw %ptr[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data = llvm.load %next {alignment = 4 : i64} : !llvm.ptr -> i32
    dataflow.graph.return %ctrl, %data : none, i32
  }
}
