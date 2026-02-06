// RUN: loom --adg %s

// Valid: all ports use the same tagged type !dataflow.tagged<i32, i4>.
fabric.temporal_pe @tpe_ok(%in0: !dataflow.tagged<i32, i4>, %in1: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0] {
  fabric.pe @fu0(%a: i32, %b: i32) -> (i32) {
    %r = arith.addi %a, %b : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>, %b: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.instance @tpe_ok(%a, %b) : (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
