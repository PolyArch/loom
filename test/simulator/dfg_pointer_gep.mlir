// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph offset_carry_ptr --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=1.250000e+00 --memref 4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00 --memref 5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "offset_carry_ptr"
// CHECK-DAG: "graph": "offset_carry_ptr"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dataflow.load": 4
// CHECK-DAG: "dataflow.store": 4
// CHECK-DAG: "arg5": [
// CHECK-DAG: "f32:2.250000",
// CHECK-DAG: "f32:3.250000",
// CHECK-DAG: "f32:-2.250000",
// CHECK-DAG: "f32:5.500000"

module {
  dataflow.graph private @offset_carry_ptr(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %bias: f32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> ()
      attributes {input_segments = array<i32: 4, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    scf.for %i = %lb to %ub step %step : i32 {
      %src_at = llvm.getelementptr %src[%i]
          : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %dst_at = llvm.getelementptr %dst[%i]
          : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %data = llvm.load %src_at : !llvm.ptr -> f32
      %sum = arith.addf %data, %bias : f32
      llvm.store %sum, %dst_at : f32, !llvm.ptr
    }
    dataflow.graph.return %ctrl : none
  }
}
