// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_TEMPORAL_PE_REG_FIFO_DEPTH

// num_register = 0 but reg_fifo_depth = 2 (must be 0 when no registers).
fabric.temporal_pe @tpe_bad(%in: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 2] {
  fabric.pe @fu0(%a: i32) -> (i32) {
    %r = arith.addi %a, %a : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.instance @tpe_bad(%a) : (!dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
