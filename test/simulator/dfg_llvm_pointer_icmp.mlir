// RUN: not loom-dfg-sim %s --graph pointer_icmp_residual --memref 0=1 --memref 1=2 --output %t.json 2>&1 | FileCheck %s

// CHECK: finalized graph contains residual pointer operation 'llvm.icmp'

module {
  dataflow.graph private @pointer_icmp_residual(
      %ctrl: none, %lhs: !llvm.ptr, %rhs: !llvm.ptr) -> ()
      attributes {input_segments = array<i32: 0, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %comparison = llvm.icmp "eq" %lhs, %rhs : !llvm.ptr
    dataflow.graph.return %ctrl : none
  }
}
