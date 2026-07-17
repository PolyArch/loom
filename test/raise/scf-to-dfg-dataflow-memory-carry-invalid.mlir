// RUN: not loom-raise-opt --loom-lower-graph-memory %s 2>&1 | FileCheck %s

// An already-formed dataflow recurrence is subject to the same capability
// plane rule as structured control.
// CHECK: error: cannot lower memory capability '!llvm.ptr' through dataflow.carry
dataflow.graph.func private @pointer_dataflow_carry(
    %start: none, %phase: i1, %pointer: !llvm.ptr) -> none
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %carried = dataflow.carry %phase, %pointer, %pointer : !llvm.ptr
  dataflow.graph.return %start : none
}
