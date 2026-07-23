// The i64 pointer offsets need a 64-bit canonical index, so lowering and
// simulation share that configured width rather than the default.
// RUN: env LOOM_INDEX_WIDTH=64 loom-raise-opt --loom-lower-graph-memory %s \
// RUN:   -o %t.lowered.mlir
// RUN: env LOOM_INDEX_WIDTH=64 loom-dfg-sim %t.lowered.mlir --graph llvm_store_fneg --memref 0=-3.500000e+00 --memref 1=0.000000e+00,0.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "llvm_store_fneg"
// CHECK-DAG: "graph": "llvm_store_fneg"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dataflow.load": 1
// CHECK-DAG: "llvm.fneg": 1
// CHECK-DAG: "dataflow.store": 1
// CHECK-DAG: "arg1": [
// CHECK-DAG: "f32:0"
// CHECK-DAG: "f32:3.500000"

module {
  dataflow.graph private @llvm_store_fneg(%ctrl: none, %src: !llvm.ptr,
                                               %dst: !llvm.ptr) -> ()
      attributes {input_segments = array<i32: 0, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %offset = dataflow.constant %ctrl {const_value = 4 : i64} : i64
    %dst_next = llvm.getelementptr inbounds|nuw %dst[%offset]
        : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %loaded = llvm.load %src {alignment = 4 : i64} : !llvm.ptr -> f32
    %negated = llvm.fneg %loaded : f32
    llvm.store %negated, %dst_next {alignment = 4 : i64} : f32, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }
}
