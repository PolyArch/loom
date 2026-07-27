// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch

// A library spelling and matching arity do not prove call semantics. The call
// remains under the imported LLVM declaration and makes only the candidate
// that selected it for SpatialCore non-finalizable.
// CHECK: error: loom-lower-graph-memory: operation 'llvm.call' is not a registered canonical Dataflow actor or a supported graph-lowering operation
// CHECK-LABEL: llvm.func @arm_nn_vec_mat_mult_t_s8
// CHECK-LABEL: dataflow.thread private @selected_library_call domain(#dataflow.thread_domain<dense>)
// CHECK: loom.spatial_region
// CHECK: llvm.call @arm_nn_vec_mat_mult_t_s8

llvm.func @arm_nn_vec_mat_mult_t_s8(i32, i32, i32, i32, i32, i32, i32,
                                    i32, i32, i32, i32, i32, i32, i32,
                                    i32) -> i32

dataflow.thread private @selected_library_call domain(#dataflow.thread_domain<dense>)(
    %arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32,
    %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32,
    %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32)
    ctrl (%start: none) {
  %status = "loom.spatial_region"(
      %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8,
      %arg9, %arg10, %arg11, %arg12, %arg13, %arg14)
      <{operandSegmentSizes = array<i32: 15, 0, 0, 0>,
        resultSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%value0: i32, %value1: i32, %value2: i32, %value3: i32,
         %value4: i32, %value5: i32, %value6: i32, %value7: i32,
         %value8: i32, %value9: i32, %value10: i32, %value11: i32,
         %value12: i32, %value13: i32, %value14: i32):
      %result = llvm.call @arm_nn_vec_mat_mult_t_s8(
          %value0, %value1, %value2, %value3, %value4, %value5, %value6,
          %value7, %value8, %value9, %value10, %value11, %value12,
          %value13, %value14)
          : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
             i32, i32, i32, i32) -> i32
      "loom.spatial_yield"(%result)
          <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
  }) {graph_name = "selected_library_call_graph", source_maps = []} :
      (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
       i32, i32, i32) -> i32
  dataflow.thread.yield
}
