// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

module {
  llvm.func @arm_nn_vec_mat_mult_t_s8(
      !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32

  dataflow.graph.func private @g_cmsis_vec_mat_mult(
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

// CHECK-LABEL: dataflow.graph.func private @g_cmsis_vec_mat_mult
// CHECK-NOT: llvm.call
// CHECK: dataflow.load
// CHECK: arith.muli
// CHECK: arith.select
// CHECK: dataflow.store
// CHECK: dataflow.graph.return
