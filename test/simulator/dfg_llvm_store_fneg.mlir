// RUN: loom-dfg-sim %s --graph llvm_store_fneg --memref 0=-3.500000e+00 --memref 1=0.000000e+00,0.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "llvm_store_fneg"
// CHECK-DAG: "graph": "llvm_store_fneg"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dataflow.load": 1
// CHECK-DAG: "arith.negf": 1
// CHECK-DAG: "dataflow.store": 1
// CHECK-DAG: "arg1": [
// CHECK-DAG: "f32:0"
// CHECK-DAG: "f32:3.500000"

module {
  dataflow.graph private @llvm_store_fneg(%ctrl: none,
      %src: memref<?xf32>, %dst: memref<?xf32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %src_index = dataflow.constant %ctrl {const_value = 0 : index} : index
    %dst_index = dataflow.constant %ctrl {const_value = 1 : index} : index
    %loaded, %read = dataflow.load %src[%src_index] %ctrl : memref<?xf32>
    %negated = arith.negf %loaded : f32
    %stored = dataflow.store %dst[%dst_index] %negated %read : memref<?xf32>
    dataflow.graph.return %stored : none
  }
}
