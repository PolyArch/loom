// RUN: loom-dfg-sim %s --graph llvm_load_ptr --memref 0=100,102,105 --output %t.ptr.json
// RUN: FileCheck %s --check-prefix=PTR < %t.ptr.json
// RUN: loom-dfg-sim %s --graph llvm_load_gep --memref 0=100,102,105 --output %t.gep.json
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
      -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %data = llvm.load %ptr {alignment = 4 : i64} : !llvm.ptr -> i32
    %published:2 = dataflow.sync %ctrl, %data
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph.func private @llvm_load_gep(%ctrl: none, %ptr: !llvm.ptr)
      -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %next = llvm.getelementptr inbounds|nuw %ptr[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data = llvm.load %next {alignment = 4 : i64} : !llvm.ptr -> i32
    %published:2 = dataflow.sync %ctrl, %data
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
