// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_TEMPORAL_PE_DEST_COUNT

// 2 outputs but instruction has only 1 destination.
fabric.temporal_pe @tpe_bad(%in0: !dataflow.tagged<i32, i4>, %in1: !dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  {instruction_mem = [
    "inst[0]: when(tag=3) out(0, tag=1) = add(0) in(0), in(1)"
  ]}
  -> (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) {
  fabric.pe @fu_add(%a: i32, %b: i32) -> (i32, i32) {
    %r = arith.addi %a, %b : i32
    fabric.yield %r, %r : i32, i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>, %b: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) {
  %o0, %o1 = fabric.instance @tpe_bad(%a, %b) : (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>)
  fabric.yield %o0, %o1 : !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
}
