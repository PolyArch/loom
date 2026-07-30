// RUN: not loom-raise-opt --loom-lower-scf-to-dfg %s 2>&1 | FileCheck %s

// A residual loop-carried pointer proves that SCF memory normalization did not
// establish the required invariant capability plus integer offset. Graph
// publication rejects it instead of creating a pointer-valued dataflow.carry.

// CHECK: cannot lower loop-carried memory capability '!llvm.ptr' through dataflow.carry
// CHECK-NOT: dataflow.graph private @pointer_carry_graph

module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.thread private @pointer_carry
      domain(#dataflow.thread_domain<dense>)(
          %base: !llvm.ptr, %lower: index, %upper: index, %step: index)
      ctrl (%ctrl: none) {
    %value = "loom.spatial_region"(%lower, %upper, %step, %base)
        <{operandSegmentSizes = array<i32: 3, 0, 1, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%lb: index, %ub: index, %stride: index, %memory: !llvm.ptr):
        %final = scf.for %i = %lb to %ub step %stride
            iter_args(%current = %memory) -> (!llvm.ptr) {
          %next = llvm.getelementptr %current[1]
              : (!llvm.ptr) -> !llvm.ptr, i8
          scf.yield %next : !llvm.ptr
        }
        %loaded = llvm.load %final : !llvm.ptr -> i8
        "loom.spatial_yield"(%loaded)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (i8) -> ()
    }) {graph_name = "pointer_carry_graph", source_maps = []} :
        (index, index, index, !llvm.ptr) -> i8
    dataflow.thread.yield
  }
}
