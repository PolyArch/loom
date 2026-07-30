// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph offset_carry_graph \
// RUN:   --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=1.250000e+00 \
// RUN:   --memref 4=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00 \
// RUN:   --memref 5=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00 \
// RUN:   --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "offset_carry_graph"
// CHECK-DAG: "graph": "offset_carry_graph"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dataflow.load": 4
// CHECK-DAG: "dataflow.store": 4
// CHECK-DAG: "arg5": [
// CHECK-DAG: "f32:2.250000",
// CHECK-DAG: "f32:3.250000",
// CHECK-DAG: "f32:-2.250000",
// CHECK-DAG: "f32:5.500000"

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.thread private @offset_carry
      domain(#dataflow.thread_domain<dense>)(
          %src: !llvm.ptr, %dst: !llvm.ptr,
          %lb: i32, %ub: i32, %step: i32, %bias: f32)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%lb, %ub, %step, %bias, %src, %dst)
        <{operandSegmentSizes = array<i32: 4, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%lower: i32, %upper: i32, %stride: i32, %addend: f32,
           %source: !llvm.ptr, %destination: !llvm.ptr):
        scf.for %i = %lower to %upper step %stride : i32 {
          %src_at = llvm.getelementptr %source[%i]
              : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %dst_at = llvm.getelementptr %destination[%i]
              : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %data = llvm.load %src_at : !llvm.ptr -> f32
          %sum = arith.addf %data, %addend : f32
          llvm.store %sum, %dst_at : f32, !llvm.ptr
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "offset_carry_graph", source_maps = []} :
        (i32, i32, i32, f32, !llvm.ptr, !llvm.ptr) -> ()
    dataflow.thread.yield
  }
}
