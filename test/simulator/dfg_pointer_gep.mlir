// RUN: loom-dfg-sim %s --graph offset_carry_ptr --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=1.250000e+00 --memref 5=1.000000e+00,2.000000e+00,-3.500000e+00,4.250000e+00 --memref 6=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "offset_carry_ptr"
// CHECK-DAG: "graph": "offset_carry_ptr"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "arg6": [
// CHECK-DAG: "f32:2.250000",
// CHECK-DAG: "f32:3.250000",
// CHECK-DAG: "f32:-2.250000",
// CHECK-DAG: "f32:5.500000"

module {
  dataflow.graph.func private @offset_carry_ptr(%ctrl: none, %lb: i32,
                                                %ub: i32, %step: i32,
                                                %bias: f32, %src: !llvm.ptr,
                                                %dst: !llvm.ptr)
      -> (none, !llvm.ptr, !llvm.ptr) {
    %zero = dataflow.constant %ctrl {const_value = 0 : index} : index
    %index, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i32
    %bias_i = dataflow.invariant %rwc, %bias : f32
    %src_cur = dataflow.carry %rwc, %src, %src_next : !llvm.ptr
    %dst_cur = dataflow.carry %rwc, %dst, %dst_next : !llvm.ptr
    %src_mem = builtin.unrealized_conversion_cast %src_cur : !llvm.ptr to memref<?xf32>
    %dst_mem = builtin.unrealized_conversion_cast %dst_cur : !llvm.ptr to memref<?xf32>
    %src_next = llvm.getelementptr inbounds|nuw %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data, %load_done = dataflow.load %src_mem[%zero] %ctrl : memref<?xf32>
    %sum = arith.addf %bias_i, %data : f32
    %dst_next = llvm.getelementptr inbounds|nuw %dst_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %store_done = dataflow.store %dst_mem[%zero] %sum %ctrl : memref<?xf32>
    %done:2 = dataflow.sync %load_done, %store_done : (none, none) -> (none, none)
    dataflow.graph.return %done#0, %src_cur, %dst_cur : none, !llvm.ptr, !llvm.ptr
  }
}
