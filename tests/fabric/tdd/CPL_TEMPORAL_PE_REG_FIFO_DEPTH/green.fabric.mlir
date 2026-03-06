// RUN: loom --adg %s

// Valid: num_register = 0, reg_fifo_depth = 0 (must be 0 when no registers).
fabric.temporal_pe @tpe_noreg(%in: !dataflow.tagged<!dataflow.bits<32>, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  fabric.pe @fu0(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
    ^bb0(%a: i32):
    %r = arith.addi %a, %a : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

// Valid: num_register = 2, reg_fifo_depth = 4 (>= 1 when registers present).
fabric.temporal_pe @tpe_reg(%in: !dataflow.tagged<!dataflow.bits<32>, i4>)
  [num_register = 2, num_instruction = 2, reg_fifo_depth = 4]
  -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  fabric.pe @fu1(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
    ^bb0(%a: i32):
    %r = arith.addi %a, %a : i32
    fabric.yield %r : i32
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %out = fabric.instance @tpe_noreg(%a) : (!dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>)
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i4>
}
