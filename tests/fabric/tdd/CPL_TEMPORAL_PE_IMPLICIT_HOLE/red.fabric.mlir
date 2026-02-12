// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_PE_IMPLICIT_HOLE

// Explicit invalid at slot 1, but implicit hole between slot 2 and slot 4.
fabric.temporal_pe @tpe_bad(%in0: !dataflow.tagged<i32, i4>, %in1: !dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 5, reg_fifo_depth = 0]
  {instruction_mem = [
    "inst[0]: when(tag=3) out(0, tag=1) = add(0) in(0), in(1)",
    "inst[1]: invalid",
    "inst[2]: when(tag=4) out(0, tag=2) = add(0) in(0), in(1)",
    "inst[4]: when(tag=5) out(0, tag=3) = add(0) in(0), in(1)"
  ]}
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
