// RUN: loom-dfg-sim %s --graph cmsis_nn_vec_mat_mult_s8 --arg 0=none --memref 1=1,-2,3 --memref 2=4,-1,2,-3,5,1 --memref 3=0,0 --memref 4=10,-4 --memref 5=0,0 --arg 6=1 --arg 7=3 --arg 8=1073741824 --arg 9=1 --arg 10=3 --arg 11=2 --arg 12=-128 --arg 13=127 --arg 14=1 --arg 15=-2 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "workload": "cmsis_nn_vec_mat_mult_s8"
// CHECK-DAG: "graph": "cmsis_nn_vec_mat_mult_s8"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "arith.muli": 8
// CHECK-DAG: "llvm.load": 14
// CHECK-DAG: "llvm.store": 2
// CHECK-DAG: "final_outputs": [
// CHECK-DAG: "i32:0"
// CHECK-DAG: "arg5": [
// CHECK-DAG: "i8:20"
// CHECK-DAG: "i8:-18"

module {
  llvm.func @arm_nn_vec_mat_mult_t_s8(
      !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32

  dataflow.graph.func private @cmsis_nn_vec_mat_mult_s8(
      %ctrl: none,
      %lhs: !llvm.ptr,
      %rhs: !llvm.ptr,
      %kernel_sum: !llvm.ptr,
      %bias: !llvm.ptr,
      %dst: !llvm.ptr,
      %lhs_offset: i32,
      %dst_offset: i32,
      %dst_multiplier: i32,
      %dst_shift: i32,
      %rhs_cols: i32,
      %rhs_rows: i32,
      %activation_min: i32,
      %activation_max: i32,
      %address_offset: i32,
      %rhs_offset: i32) -> (none, i32) {
    %status = llvm.call @arm_nn_vec_mat_mult_t_s8(
        %lhs, %rhs, %kernel_sum, %bias, %dst,
        %lhs_offset, %dst_offset, %dst_multiplier, %dst_shift,
        %rhs_cols, %rhs_rows, %activation_min, %activation_max,
        %address_offset, %rhs_offset)
        : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
           i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    dataflow.graph.return %ctrl, %status : none, i32
  }
}
