// RUN: loom-dfg-sim %s --graph llvm_store_fneg --arg 0=none --memref 1=-3.500000e+00 --memref 2=0.000000e+00,0.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "llvm_store_fneg"
// CHECK-DAG: "graph": "llvm_store_fneg"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "llvm.load": 1
// CHECK-DAG: "llvm.fneg": 1
// CHECK-DAG: "llvm.store": 1
// CHECK-DAG: "arg2": [
// CHECK-DAG: "f32:0"
// CHECK-DAG: "f32:3.500000"

module {
  dataflow.graph.func private @llvm_store_fneg(%ctrl: none, %src: !llvm.ptr,
                                               %dst: !llvm.ptr) -> none {
    %dst_next = llvm.getelementptr inbounds|nuw %dst[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %loaded = llvm.load %src {alignment = 4 : i64} : !llvm.ptr -> f32
    %negated = llvm.fneg %loaded : f32
    llvm.store %negated, %dst_next {alignment = 4 : i64} : f32, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }
}
