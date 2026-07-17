// RUN: loom-raise-opt --loom-lower-known-library-calls --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph cmsis_nn_vec_mat_mult_s8 --arg 0=1 --arg 1=3 --arg 2=1073741824 --arg 3=1 --arg 4=3 --arg 5=2 --arg 6=-128 --arg 7=127 --arg 8=1 --arg 9=-2 --memref 10=1,-2,3 --memref 11=4,-1,2,-3,5,1 --memref 12=0,0 --memref 13=10,-4 --memref 14=0,0 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK: "arg14": [
// CHECK-NEXT: "i8:20",
// CHECK-NEXT: "i8:-18"
// CHECK-NEXT: ]
// CHECK: "final_outputs": [
// CHECK-NEXT: "none",
// CHECK-NEXT: "i32:0"
// CHECK-NEXT: ]
// CHECK: "status": "pass"

module {
  llvm.func @arm_nn_vec_mat_mult_t_s8(
      !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32

  dataflow.graph private @cmsis_nn_vec_mat_mult_s8(
      %ctrl: none,
      %lhs_offset: i32,
      %dst_offset: i32,
      %dst_multiplier: i32,
      %dst_shift: i32,
      %rhs_cols: i32,
      %rhs_rows: i32,
      %activation_min: i32,
      %activation_max: i32,
      %address_offset: i32,
      %rhs_offset: i32,
      %lhs: !llvm.ptr,
      %rhs: !llvm.ptr,
      %kernel_sum: !llvm.ptr,
      %bias: !llvm.ptr,
      %dst: !llvm.ptr) -> (i32)
      attributes {input_segments = array<i32: 10, 0, 5>,
                  result_segments = array<i32: 1, 0, 0>} {
    %status = llvm.call @arm_nn_vec_mat_mult_t_s8(
        %lhs, %rhs, %kernel_sum, %bias, %dst,
        %lhs_offset, %dst_offset, %dst_multiplier, %dst_shift,
        %rhs_cols, %rhs_rows, %activation_min, %activation_max,
        %address_offset, %rhs_offset)
        : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
           i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32
    %retired:2 = dataflow.sync %ctrl, %status : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }
}
