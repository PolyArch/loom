// RUN: loom --adg %s

// Valid: instruction_mem slot indices are strictly ascending (0, 2).
fabric.temporal_pe @tpe_ok(%in0: !dataflow.tagged<i32, i4>, %in1: !dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 4, reg_fifo_depth = 0]
  {instruction_mem = [
    "inst[0]: when(tag=3) out(0, tag=1) = add(0) in(0), in(1)",
    "inst[2]: when(tag=4) out(0, tag=2) = add(0) in(0), in(1)"
  ]}
  -> (!dataflow.tagged<i32, i4>) {
  fabric.pe @fu_add(%a: i32, %b: i32) -> (i32) {
    %r = arith.addi %a, %b : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>, %b: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.instance @tpe_ok(%a, %b) : (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
