// RUN: loom --adg %s

// A valid temporal_pe with num_instruction = 4 (>= 1).
fabric.temporal_pe @tpe0(%in: !dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 4, reg_fifo_depth = 0]
  -> (!dataflow.tagged<i32, i4>) {
  fabric.pe @fu0(%a: i32) -> (i32) {
    %r = arith.addi %a, %a : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.instance @tpe0(%a) : (!dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
