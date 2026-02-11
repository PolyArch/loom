// RUN: loom --adg %s

// Valid: temporal_pe wraps an external arith PE via fabric.instance.
// The external PE uses arith (non-dataflow) operations, so this is allowed.
fabric.pe @arith_pe(%a: i32) -> (i32) {
  %r = arith.addi %a, %a : i32
  fabric.yield %r : i32
}

fabric.temporal_pe @tpe_ok(%in: !dataflow.tagged<i32, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  -> (!dataflow.tagged<i32, i4>) {
  fabric.pe @fu0(%a: i32) -> (i32) {
    %r = fabric.instance @arith_pe(%a) : (i32) -> (i32)
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>) {
  %out = fabric.instance @tpe_ok(%a) : (!dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>)
  fabric.yield %out : !dataflow.tagged<i32, i4>
}
