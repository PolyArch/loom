// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_PE_SRC_MISMATCH

// Source position mismatch: in(1) at operand position 0, in(0) at operand
// position 1. Per spec, in(j) at position i requires j == i.
fabric.temporal_pe @tpe_bad(%in0: !dataflow.tagged<i32, i4>, %in1: !dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  {instruction_mem = ["inst[0]: when(tag=3) out(0, tag=1) = add(0) in(1), in(0)"]}
  -> (!dataflow.tagged<i32, i4>) {
  fabric.pe @fu_add(%a: i32, %b: i32) -> (i32) {
    %r = arith.addi %a, %b : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>, %b: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.instance @tpe_bad(%a, %b) : (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
